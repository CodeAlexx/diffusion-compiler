#include "dif/frontend/h3_lora.hpp"
#include "dif/opt/lora.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace dif::frontend {
namespace {
struct Projection {
  std::uint32_t operation{};
  std::uint64_t row{}, rows{}, input{};
};
using Sites = std::map<std::string, std::vector<Projection>>;

Sites sites(const ir::Program &program, const weights::WeightBundle &bundle) {
  Sites result;
  for (const auto &binding : bundle.bindings) {
    auto name = binding.tensor_name;
    if (!name.ends_with(".weight")) continue;
    name.resize(name.size() - 7U);
    if (!name.starts_with("blocks.") &&
        !name.starts_with("token_refiner.blocks.")) continue;
    if (!(name.ends_with(".attn.qkv_proj") ||
          name.ends_with(".attn.out_proj") || name.ends_with(".mlp.fc1") ||
          name.ends_with(".mlp.fc2"))) continue;
    std::vector<std::pair<std::uint32_t, std::uint64_t>> weights;
    if (name.ends_with(".attn.qkv_proj")) {
      const auto layout = std::find_if(program.operations.begin(),
          program.operations.end(), [&](const auto &op) {
            return op.opcode == ir::Opcode::H3DeinterleaveQkvWeight &&
                   op.inputs == std::vector<std::uint32_t>{binding.tensor_id};
          });
      if (layout == program.operations.end() || layout->outputs.size() != 3U)
        fail("H3 LoRA cannot locate source-shaped QKV layout: " + name);
      std::uint64_t offset = 0U;
      for (const auto id : layout->outputs) {
        weights.emplace_back(id, offset);
        offset += program.tensor(id)->dims.at(0);
      }
    } else weights.emplace_back(binding.tensor_id, 0U);
    for (const auto &[id, offset] : weights) {
      std::size_t count = 0U;
      for (const auto &op : program.operations) {
        if (op.opcode != ir::Opcode::Linear || op.inputs.size() != 2U ||
            op.inputs[1] != id) continue;
        const auto *weight = program.tensor(id);
        result[name].push_back({op.id, offset, weight->dims.at(0),
                                weight->dims.at(1)});
        ++count;
      }
      if (count != 1U) fail("H3 LoRA requires one projection consumer: " + name);
    }
  }
  if (result.empty()) fail("H3 LoRA found no named base projections");
  return result;
}

runtime::Tensor rows(const runtime::Tensor &value, std::uint64_t start,
                      std::uint64_t count) {
  if (value.dims.size() != 2U || start > value.dims[0] ||
      count > value.dims[0] - start) fail("H3 LoRA row slice out of bounds");
  const auto stride = value.byte_size() / value.dims[0];
  if (value.is_mapped())
    return runtime::map_tensor_slice(value.mapping, value.dtype,
        {count, value.dims[1]}, value.mapping_offset + start * stride,
        count * stride);
  return {value.dtype, {count, value.dims[1]},
      std::vector<std::uint8_t>(value.data() + start * stride,
                               value.data() + (start + count) * stride)};
}

bool nonzero_finite(const runtime::Tensor &tensor) {
  const auto f32 = runtime::convert_float_tensor(tensor, ir::DType::F32);
  bool nonzero = false;
  for (const auto value : f32.f32()) {
    if (!std::isfinite(value)) fail("H3 LoRA has nonfinite tensor values");
    nonzero |= value != 0.0F;
  }
  return nonzero;
}
} // namespace

H3LoraResult add_h3_lora(const ir::Program &program,
                         const weights::WeightBundle &base_bundle,
                         const std::vector<H3LoraRequest> &requests) {
  H3LoraResult result{program, {}, 0U, 0U};
  if (requests.empty()) return result;
  const auto named = sites(program, base_bundle);
  for (const auto &request : requests) {
    if (!std::isfinite(request.multiplier) || request.multiplier == 0.0F)
      fail("H3 LoRA multiplier must be finite and nonzero");
    const auto file = weights::read_safetensors(request.path);
    std::set<std::string> consumed;
    std::size_t active = 0U;
    for (const auto &[name, projections] : named) {
      std::vector<std::string> prefixes{"diffusion_model." + name, name};
      if (name.starts_with("blocks.")) {
        auto legacy = "lora_unet_" + name;
        std::replace(legacy.begin(), legacy.end(), '.', '_');
        prefixes.push_back(std::move(legacy));
      }
      std::string selected, a_key, b_key;
      for (const auto &prefix : prefixes) {
        const auto legacy = prefix.starts_with("lora_unet_");
        const auto a = prefix + (legacy ? ".lora_down.weight" : ".lora_A.weight");
        const auto b = prefix + (legacy ? ".lora_up.weight" : ".lora_B.weight");
        if (bool(file.find(a)) != bool(file.find(b)))
          fail("H3 LoRA incomplete A/B pair: " + prefix);
        if (!file.find(a)) continue;
        if (!selected.empty()) fail("H3 LoRA ambiguous duplicate layout: " + name);
        selected = prefix; a_key = a; b_key = b;
      }
      if (selected.empty()) continue;
      auto down = runtime::convert_float_tensor(weights::map_safetensor(file, a_key),
                                                ir::DType::BF16);
      auto up = runtime::convert_float_tensor(weights::map_safetensor(file, b_key),
                                              ir::DType::BF16);
      const auto rank = down.dims.size() == 2U ? down.dims[0] : 0U;
      std::uint64_t total_rows = 0U;
      for (const auto &projection : projections) total_rows += projection.rows;
      if (rank == 0U || down.dims != std::vector<std::uint64_t>{rank, projections[0].input} ||
          up.dims != std::vector<std::uint64_t>{total_rows, rank})
        fail("H3 LoRA shape mismatch: " + name);
      const bool nonzero = nonzero_finite(down) & nonzero_finite(up);
      float alpha = static_cast<float>(rank);
      const auto alpha_key = selected + ".alpha";
      if (file.find(alpha_key)) {
        const auto value = runtime::convert_float_tensor(
            weights::map_safetensor(file, alpha_key), ir::DType::F32);
        if (value.element_count() != 1U) fail("H3 LoRA alpha must be scalar");
        alpha = value.f32()[0]; consumed.insert(alpha_key);
      }
      if (!std::isfinite(alpha) || alpha <= 0.0F)
        fail("H3 LoRA alpha must be finite and positive");
      volatile float scaled_alpha = std::abs(request.multiplier) * alpha;
      const float source_scale = scaled_alpha / static_cast<float>(rank);
      if (!std::isfinite(source_scale) || source_scale == 0.0F)
        fail("H3 LoRA effective scale is nonfinite or zero");
      consumed.insert(a_key); consumed.insert(b_key);
      if (!nonzero) continue; // Zero-initialized sites are legal, an all-zero file is not.
      for (const auto &projection : projections) {
        auto adapted = opt::insert_lora(result.program,
            {rank, double(source_scale) * double(rank),
             ir::DType::BF16, {projection.operation}});
        const auto &site = adapted.sites.at(0);
        // Mojo's mul_scalar takes an F32 scalar even when delta is BF16.
        // Generic training LoRA intentionally uses a compute-dtype Fill;
        // preserve its contract and express H3's different boundary using
        // the existing F32-factor LinearBlend(delta, zero, scale) primitive.
        const auto combine = std::find_if(adapted.program.operations.begin(),
            adapted.program.operations.end(), [&](const auto &op) {
              return op.opcode == ir::Opcode::Add &&
                     op.outputs == std::vector<std::uint32_t>{site.output};
            });
        const auto scaled_id = combine->inputs.at(1);
        auto multiply = std::find_if(adapted.program.operations.begin(),
            adapted.program.operations.end(), [&](const auto &op) {
              return op.outputs == std::vector<std::uint32_t>{scaled_id};
            });
        const auto delta_id = multiply->inputs.at(0);
        const auto zero_id = multiply->inputs.at(1);
        for (auto &op : adapted.program.operations)
          if (op.outputs == std::vector<std::uint32_t>{zero_id})
            op.attributes = {ir::Attribute::f64(ir::AttrKey::Value, 0.0)};
        std::uint32_t factor_id = 1U;
        for (const auto &tensor : adapted.program.tensors)
          factor_id = std::max(factor_id, tensor.id + 1U);
        adapted.program.tensors.push_back({factor_id, ir::DType::F32,
            ir::TensorRole::Constant, {1U}});
        multiply->opcode = ir::Opcode::LinearBlend;
        multiply->inputs = {delta_id, zero_id, factor_id};
        runtime::Tensor factor{ir::DType::F32, {1U}, std::vector<std::uint8_t>(4U)};
        std::memcpy(factor.bytes.data(), &source_scale, sizeof(source_scale));
        result.constants.emplace(factor_id, std::move(factor));
        auto b = rows(up, projection.row, projection.rows);
        if (request.multiplier < 0.0F) {
          auto f = runtime::convert_float_tensor(b, ir::DType::F32);
          for (auto &v : f.f32()) v = -v;
          b = runtime::convert_float_tensor(f, ir::DType::BF16);
        }
        result.constants.emplace(site.down, down);
        result.constants.emplace(site.up, std::move(b));
        result.program = std::move(adapted.program);
        // Immutable adapter weights upload once, unlike dynamic training parameters.
        for (auto &tensor : result.program.tensors)
          if (tensor.id == site.down || tensor.id == site.up)
            tensor.roles = ir::TensorRole::Constant;
        ++result.projections;
      }
      ++active; ++result.adapters;
    }
    for (const auto &[name, entry] : file.tensors) {
      (void)entry;
      if ((name.find("lora_") != std::string::npos || name.ends_with(".alpha")) &&
          !consumed.contains(name))
        fail("H3 LoRA contains unsupported or unconsumed adapter tensor: " + name);
    }
    if (active == 0U) fail("H3 LoRA contains no nonzero applicable adapters: " + request.path.string());
  }
  ir::verify(result.program);
  return result;
}

std::vector<runtime::RunOptions::ConvRotLinearBinding>
h3_lora_convrot_bindings(const ir::Program &base_program,
                         const weights::WeightBundle &base_bundle,
                         const std::filesystem::path &cache_path,
                         std::uint32_t resident_layers,
                         std::uint32_t attention_layers,
                         std::uint32_t mlp_layers) {
  const auto named = sites(base_program, base_bundle);
  const auto cache = weights::read_safetensors(cache_path);
  const auto metadata = weights::map_safetensor(cache, "__meta__.h3_convrot");
  if (metadata.dtype != ir::DType::I32 || metadata.byte_size() != 14U * 4U)
    fail("H3 LoRA requires native single-scale H256 ConvRot v1 cache");
  std::array<std::uint32_t, 14> identity{};
  std::memcpy(identity.data(), metadata.data(), metadata.byte_size());
  // Same format contract as validate_h3_convrot_metadata; index digest alone
  // is not a cross-checkpoint identity proof. The runner owns the sealed path.
  if (identity[0] != 0x44494643U || identity[1] != 1U || identity[2] != 256U ||
      identity[3] != 1U || identity[5] != 4U)
    fail("H3 LoRA ConvRot format identity mismatch");
  std::vector<runtime::RunOptions::ConvRotLinearBinding> result;
  for (const auto &[name, projections] : named) {
    if (!name.starts_with("blocks.")) continue;
    const auto end = name.find('.', 7U);
    const auto layer = static_cast<std::uint32_t>(std::stoul(name.substr(7U, end - 7U)));
    const auto slot = name.ends_with("attn.qkv_proj") ? 0U :
                      name.ends_with("attn.out_proj") ? 1U :
                      name.ends_with("mlp.fc1") ? 2U : 3U;
    if (layer >= (slot < 2U ? attention_layers : mlp_layers)) continue;
    if (layer >= identity[4]) fail("H3 LoRA ConvRot cache has insufficient layers");
    const auto prefix = "block." + std::to_string(layer);
    auto w = weights::map_safetensor(cache, prefix + ".convrot_weight." + std::to_string(slot));
    auto s = weights::map_safetensor(cache, prefix + ".convrot_scale." + std::to_string(slot));
    if (w.dtype != ir::DType::I8 || s.dtype != ir::DType::F32 ||
        w.dims.size() != 2U || s.dims != std::vector<std::uint64_t>{w.dims[0]})
      fail("H3 LoRA ConvRot physical shape mismatch");
    s.dims.push_back(1U);
    for (const auto &projection : projections) {
      auto scales = rows(s, projection.row, projection.rows);
      scales.dims = {projection.rows};
      auto weight = rows(w, projection.row, projection.rows);
      if (weight.dims != std::vector<std::uint64_t>{projection.rows, projection.input})
        fail("H3 LoRA ConvRot projection dimensions disagree with base");
      result.push_back({projection.operation, std::move(weight), std::move(scales),
                        cache_path, layer < resident_layers});
    }
  }
  if (result.empty()) fail("H3 LoRA ConvRot selected no base projections");
  return result;
}
} // namespace dif::frontend
