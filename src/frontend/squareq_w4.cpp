#include "dif/frontend/squareq_w4.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <unordered_map>

namespace dif::frontend {
namespace {

constexpr std::uint64_t kHadamardBlock = 256U;
constexpr std::uint64_t kGroup = 64U;
constexpr const char *kFormat = "squareq_w4_v1";

std::string read_text(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    fail("cannot read " + path.string());
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

std::uint64_t number_field(const json::Value &object, const char *key,
                           const std::string &context) {
  const auto *value = object.find(key);
  if (!value)
    fail("squareq-plan.json lacks " + std::string(key) + " for " + context);
  const auto number = value->number();
  if (number < 0.0 || number != static_cast<double>(static_cast<std::uint64_t>(number)))
    fail("squareq-plan.json field " + std::string(key) + " is not a count for " + context);
  return static_cast<std::uint64_t>(number);
}

// Normalized Sylvester Hadamard H_256 in BF16: every entry is +-1/16, exact.
runtime::Tensor hadamard_256_bf16() {
  std::vector<std::uint8_t> bytes(kHadamardBlock * kHadamardBlock * 2U);
  for (std::uint64_t i = 0U; i < kHadamardBlock; ++i)
    for (std::uint64_t j = 0U; j < kHadamardBlock; ++j) {
      const bool negative = (std::popcount(i & j) & 1U) != 0U;
      const std::uint16_t bits = negative ? 0xBD80U : 0x3D80U; // -+0.0625
      std::memcpy(bytes.data() + (i * kHadamardBlock + j) * 2U, &bits, 2U);
    }
  return runtime::Tensor{ir::DType::BF16, {kHadamardBlock, kHadamardBlock},
                         std::move(bytes)};
}

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

std::uint16_t float_to_bf16(float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

// lora_down [K,R] BF16 -> (H_bd lora_down) with the normalized block
// Hadamard-256 along K (fast Walsh-Hadamard in double, exact 1/16 scale).
runtime::Tensor rotate_rows_h256(const runtime::Tensor &lora_down,
                                 std::uint64_t in_features, std::uint64_t rank) {
  runtime::Tensor rotated{ir::DType::BF16, {in_features, rank}, {}};
  rotated.bytes.resize(in_features * rank * 2U);
  const auto *source = lora_down.data();
  std::vector<double> column(kHadamardBlock);
  for (std::uint64_t r = 0U; r < rank; ++r) {
    for (std::uint64_t block = 0U; block < in_features / kHadamardBlock; ++block) {
      for (std::uint64_t i = 0U; i < kHadamardBlock; ++i) {
        std::uint16_t bits = 0U;
        std::memcpy(&bits, source + ((block * kHadamardBlock + i) * rank + r) * 2U, 2U);
        column[i] = bf16_to_float(bits);
      }
      for (std::uint64_t stride = 1U; stride < kHadamardBlock; stride *= 2U)
        for (std::uint64_t i = 0U; i < kHadamardBlock; i += 2U * stride)
          for (std::uint64_t j = i; j < i + stride; ++j) {
            const double a = column[j], b = column[j + stride];
            column[j] = a + b;
            column[j + stride] = a - b;
          }
      for (std::uint64_t i = 0U; i < kHadamardBlock; ++i) {
        const auto bits = float_to_bf16(static_cast<float>(column[i] / 16.0));
        std::memcpy(rotated.bytes.data() + ((block * kHadamardBlock + i) * rank + r) * 2U,
                    &bits, 2U);
      }
    }
  }
  return rotated;
}

} // namespace

SquareQW4RewriteResult rewrite_linear_weights_squareq_w4(
    ir::Program &program, runtime::TensorMap &bindings,
    std::span<const std::uint32_t> checkpoint_tensors,
    std::span<const std::string> checkpoint_names,
    const std::filesystem::path &slab_directory, SquareQW4Mode mode) {
  if (checkpoint_tensors.size() != checkpoint_names.size())
    fail("SquareQ W4 rewrite: checkpoint tensor/name lists differ in length");
  const auto plan = json::parse(read_text(slab_directory / "squareq-plan.json"));
  if (!plan.is_object())
    fail("squareq-plan.json is not an object");
  const auto *format = plan.find("format");
  if (!format || format->string() != kFormat)
    fail("SquareQ slab format is not " + std::string(kFormat));
  if (number_field(plan, "hblock", "plan") != kHadamardBlock ||
      number_field(plan, "group", "plan") != kGroup)
    fail("SquareQ slab uses an unsupported Hadamard block or scale group");
  const auto *layers = plan.find("layers");
  if (!layers || !layers->is_object())
    fail("squareq-plan.json lacks a layers object");
  const auto index =
      weights::read_safetensors_index(slab_directory / "model.safetensors.index.json");

  SquareQW4RewriteResult result;
  result.format = kFormat;
  result.mode = mode;
  if (const auto *totals = plan.find("totals"))
    if (const auto *cos = totals->find("cos_w_min"))
      result.plan_cos_w_min = cos->number();

  std::map<std::filesystem::path, weights::SafeTensorFile> shards;
  auto shard_for = [&](const std::string &tensor_name)
      -> const weights::SafeTensorFile & {
    const auto found = index.weight_map.find(tensor_name);
    if (found == index.weight_map.end())
      fail("SquareQ slab index lacks " + tensor_name);
    auto shard = shards.find(found->second);
    if (shard == shards.end())
      shard = shards.emplace(found->second,
                             weights::read_safetensors(found->second)).first;
    return shard->second;
  };
  auto mapped = [&](const std::string &tensor_name) {
    return weights::map_safetensor(shard_for(tensor_name), tensor_name);
  };

  auto next_tensor = std::uint32_t{0U};
  auto next_operation = std::uint32_t{0U};
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id);

  std::uint32_t hadamard_id = 0U;
  std::uint32_t seen_layers = 0U;
  std::unordered_map<std::uint32_t, bool> visited;
  for (std::size_t index_in_list = 0U; index_in_list < checkpoint_tensors.size();
       ++index_in_list) {
    const auto &name = checkpoint_names[index_in_list];
    const auto weight_id = checkpoint_tensors[index_in_list];
    const auto *layer = layers->find(name);
    if (!layer)
      continue;
    if (visited[weight_id])
      continue;
    visited[weight_id] = true;
    ++seen_layers;
    const auto *weight_desc_pointer = program.tensor(weight_id);
    if (!weight_desc_pointer || weight_desc_pointer->dtype != ir::DType::BF16 ||
        weight_desc_pointer->dims.size() != 2U)
      fail("SquareQ W4 key " + name + " is not a BF16 rank-2 program weight");
    const ir::TensorDesc weight_copy = *weight_desc_pointer;
    const auto *weight_desc = &weight_copy;
    const auto out_features = weight_desc->dims[0];
    const auto in_features = weight_desc->dims[1];
    if (number_field(*layer, "out", name) != out_features ||
        number_field(*layer, "in", name) != in_features)
      fail("SquareQ W4 plan geometry disagrees with the program weight " + name);
    if (in_features % kHadamardBlock != 0U || in_features % kGroup != 0U)
      fail("SquareQ W4 key " + name + " has an input width not divisible by 256");
    const auto rank = number_field(*layer, "rank", name);
    if (rank == 0U)
      fail("SquareQ W4 key " + name + " has rank zero");
    if (result.rank == 0U)
      result.rank = static_cast<std::uint32_t>(rank);
    else if (result.rank != rank)
      fail("SquareQ W4 slab mixes low-rank ranks");
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const ir::Operation &operation) {
          return operation.opcode == ir::Opcode::Linear &&
                 (operation.inputs.size() == 2U || operation.inputs.size() == 3U) &&
                 operation.inputs[1] == weight_id;
        });
    if (linear == program.operations.end())
      fail("SquareQ W4 key " + name + " is not consumed by a Linear");
    const auto consumers = std::count_if(
        program.operations.begin(), program.operations.end(),
        [&](const ir::Operation &operation) {
          return std::find(operation.inputs.begin(), operation.inputs.end(),
                           weight_id) != operation.inputs.end();
        });
    if (consumers != 1)
      fail("SquareQ W4 key " + name + " has more than one consumer");
    const auto base = name.size() > 7U && name.ends_with(".weight")
                          ? name.substr(0U, name.size() - 7U)
                          : name;

    auto qweight = mapped(base + ".qweight");
    if (qweight.dims != std::vector<std::uint64_t>{out_features, in_features / 2U} ||
        qweight.byte_size() != out_features * in_features / 2U)
      fail("SquareQ W4 qweight shape disagrees for " + name);
    qweight.dtype = ir::DType::I8; // same bytes; the slab labels them U8
    auto wscales = mapped(base + ".wscales");
    const auto groups = in_features / kGroup;
    if (wscales.dtype != ir::DType::BF16 ||
        wscales.dims != std::vector<std::uint64_t>{groups, out_features})
      fail("SquareQ W4 wscales shape disagrees for " + name);
    // DequantizeInt4 wants scales [out, in/group]; the slab stores [in/group, out].
    runtime::Tensor scales{ir::DType::BF16, {out_features, groups}, {}};
    scales.bytes.resize(out_features * groups * 2U);
    {
      const auto *source = wscales.data();
      for (std::uint64_t g = 0U; g < groups; ++g)
        for (std::uint64_t o = 0U; o < out_features; ++o)
          std::memcpy(scales.bytes.data() + (o * groups + g) * 2U,
                      source + (g * out_features + o) * 2U, 2U);
    }
    auto lora_down = mapped(base + ".lora_down");
    auto lora_up = mapped(base + ".lora_up");
    if (lora_down.dtype != ir::DType::BF16 ||
        lora_down.dims != std::vector<std::uint64_t>{in_features, rank} ||
        lora_up.dtype != ir::DType::BF16 ||
        lora_up.dims != std::vector<std::uint64_t>{out_features, rank})
      fail("SquareQ W4 low-rank shapes disagree for " + name);

    if (mode == SquareQW4Mode::DequantBf16 && hadamard_id == 0U) {
      hadamard_id = ++next_tensor;
      program.tensors.push_back({hadamard_id, ir::DType::BF16, weight_desc->roles,
                                 {kHadamardBlock, kHadamardBlock}});
      bindings.emplace(hadamard_id, hadamard_256_bf16());
    }
    if (mode == SquareQW4Mode::Int8Compute) {
      if (linear->inputs.size() != 2U)
        fail("SquareQ W4 INT8 compute does not take a biased Linear yet: " + name);
      // lora_down rotated along K by the block Hadamard (H symmetric), so the
      // low-rank branch lives in the same rotated space as the residual.
      lora_down = rotate_rows_h256(lora_down, in_features, rank);
    }
    const auto roles = weight_desc->roles;
    const auto q_id = ++next_tensor;
    const auto s_id = ++next_tensor;
    const auto ld_id = ++next_tensor;
    const auto lu_id = ++next_tensor;
    const auto d_id = ++next_tensor;
    const auto d2_id = ++next_tensor;
    const auto r2_id = ++next_tensor;
    const auto r_id = ++next_tensor;
    const auto l_id = ++next_tensor;
    const auto w_id = ++next_tensor;
    const std::vector<std::uint64_t> full{out_features, in_features};
    const std::vector<std::uint64_t> blocks{out_features * in_features / kHadamardBlock,
                                            kHadamardBlock};
    program.tensors.push_back({q_id, ir::DType::I8, roles, {out_features, in_features / 2U}});
    program.tensors.push_back({s_id, ir::DType::BF16, roles, {out_features, groups}});
    program.tensors.push_back({ld_id, ir::DType::BF16, roles, {in_features, rank}});
    program.tensors.push_back({lu_id, ir::DType::BF16, roles, {out_features, rank}});
    program.tensors.push_back({d_id, ir::DType::BF16, ir::TensorRole::Internal, full});
    program.tensors.push_back({l_id, ir::DType::BF16, ir::TensorRole::Internal, full});
    program.tensors.push_back({w_id, ir::DType::BF16, ir::TensorRole::Internal, full});
    if (mode == SquareQW4Mode::DequantBf16) {
      // Only the dequant route materializes the unrotated weight through the
      // exact Hadamard Linear; the INT8 route folds the rotation into the
      // activation quantizer and never declares these.
      program.tensors.push_back({d2_id, ir::DType::BF16, ir::TensorRole::Internal, blocks});
      program.tensors.push_back({r2_id, ir::DType::BF16, ir::TensorRole::Internal, blocks});
      program.tensors.push_back({r_id, ir::DType::BF16, ir::TensorRole::Internal, full});
    }
    result.quantized_bytes += qweight.byte_size() + scales.byte_size() +
                              lora_down.byte_size() + lora_up.byte_size();
    result.bf16_bytes_replaced += weight_desc->byte_count();
    bindings.emplace(q_id, std::move(qweight));
    bindings.emplace(s_id, std::move(scales));
    bindings.emplace(ld_id, std::move(lora_down));
    bindings.emplace(lu_id, std::move(lora_up));
    bindings.erase(weight_id);
    // The BF16 weight has no consumer left: drop its declaration too, so no
    // executor asks for a binding that the slab intentionally replaces.
    program.tensors.erase(
        std::remove_if(program.tensors.begin(), program.tensors.end(),
                       [&](const ir::TensorDesc &tensor) {
                         return tensor.id == weight_id;
                       }),
        program.tensors.end());

    auto rewritten = *linear;
    const auto linear_index =
        static_cast<std::size_t>(std::distance(program.operations.begin(), linear));
    std::vector<ir::Operation> inserted;
    if (mode == SquareQW4Mode::DequantBf16) {
      rewritten.inputs[1] = w_id;
      inserted = {
          {++next_operation, ir::Opcode::DequantizeInt4, {q_id, s_id}, {d_id},
           {ir::Attribute::u64(ir::AttrKey::GroupSize, kGroup)}},
          {++next_operation, ir::Opcode::Reshape, {d_id}, {d2_id}, {}},
          {++next_operation, ir::Opcode::Linear, {d2_id, hadamard_id}, {r2_id}, {}},
          {++next_operation, ir::Opcode::Reshape, {r2_id}, {r_id}, {}},
          {++next_operation, ir::Opcode::Linear, {lu_id, ld_id}, {l_id}, {}},
          {++next_operation, ir::Opcode::Add, {r_id, l_id}, {w_id}, {}},
      };
    } else {
      // Rotated effective weight W_r = dequant(residual) + lora_up lora_down_rot^T,
      // row-quantized on device; activations rotated by H256 and row-quantized;
      // CUTLASS scaled INT8 GEMM.
      const auto *x = program.tensor(linear->inputs[0]);
      if (!x || x->dtype != ir::DType::BF16)
        fail("SquareQ W4 INT8 compute needs a BF16 Linear input: " + name);
      const auto wq_id = ++next_tensor;
      const auto wscale_id = ++next_tensor;
      const auto xq_id = ++next_tensor;
      const auto xscale_id = ++next_tensor;
      const auto x_rows = x->element_count() / x->dims.back();
      program.tensors.push_back({wq_id, ir::DType::I8, ir::TensorRole::Internal, full});
      program.tensors.push_back({wscale_id, ir::DType::F32, ir::TensorRole::Internal, {out_features}});
      program.tensors.push_back({xq_id, ir::DType::I8, ir::TensorRole::Internal, x->dims});
      program.tensors.push_back({xscale_id, ir::DType::F32, ir::TensorRole::Internal, {x_rows}});
      rewritten.opcode = ir::Opcode::LinearInt8Scaled;
      rewritten.inputs = {xq_id, wq_id, xscale_id, wscale_id};
      rewritten.attributes.clear();
      inserted = {
          {++next_operation, ir::Opcode::DequantizeInt4, {q_id, s_id}, {d_id},
           {ir::Attribute::u64(ir::AttrKey::GroupSize, kGroup)}},
          {++next_operation, ir::Opcode::Linear, {lu_id, ld_id}, {l_id}, {}},
          {++next_operation, ir::Opcode::Add, {d_id, l_id}, {w_id}, {}},
          {++next_operation, ir::Opcode::QuantizeInt8Rows, {w_id}, {wq_id, wscale_id},
           {ir::Attribute::u64(ir::AttrKey::BlockSize, kHadamardBlock),
            ir::Attribute::u64(ir::AttrKey::Implementation,
                               static_cast<std::uint64_t>(ir::Int8RowQuantization::Direct))}},
          {++next_operation, ir::Opcode::QuantizeInt8Rows, {linear->inputs[0]},
           {xq_id, xscale_id},
           {ir::Attribute::u64(ir::AttrKey::BlockSize, kHadamardBlock),
            ir::Attribute::u64(ir::AttrKey::Implementation,
                               static_cast<std::uint64_t>(
                                   ir::Int8RowQuantization::H256F32SylvesterConvRot))}},
      };
    }
    if (mode == SquareQW4Mode::Int8Compute)
      for (std::size_t index = 0U; index + 1U < inserted.size(); ++index)
        result.weight_chain_operations.push_back(inserted[index].id);
    program.operations.at(linear_index) = std::move(rewritten);
    program.operations.insert(
        program.operations.begin() + static_cast<std::ptrdiff_t>(linear_index),
        inserted.begin(), inserted.end());
    ++result.linear_count;
    result.names.push_back(name);
  }
  if (seen_layers != layers->object().size())
    fail("SquareQ W4 plan lists " + std::to_string(layers->object().size()) +
         " layers but only " + std::to_string(seen_layers) +
         " matched checkpoint keys bound in this program");
  if (result.linear_count == 0U)
    fail("SquareQ W4 slab matched no Linear weight in this program");
  ir::verify(program);
  return result;
}

} // namespace dif::frontend
