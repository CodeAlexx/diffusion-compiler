#include "dif/compiler/compiler.hpp"
#include "dif/compiler/kernel_template.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <array>
#include <iomanip>
#include <sstream>
#include <unordered_map>
#include <vector>

namespace dif::compiler {
namespace {

std::string function_name(const ir::Operation &op) {
  return "dif_op_" + std::to_string(op.id);
}

// Generated kernels are functions of opcode, tensor geometry/dtypes, and
// attributes -- never tensor or operation identity. Repeated model blocks and
// unrolled schedules therefore share one compiled entrypoint. The executor
// still keeps an operation-id -> entrypoint map, so semantic provenance and
// launch arguments remain per operation while NVRTC sees only unique code.
std::string kernel_identity(const ir::Program &program,
                            const ir::Operation &operation) {
  std::ostringstream key;
  key << static_cast<std::uint32_t>(operation.opcode) << '|';
  const auto tensor = [&](std::uint32_t id) {
    const auto *description = program.tensor(id);
    if (!description)
      fail("CUDA kernel identity references a missing tensor");
    key << static_cast<std::uint32_t>(description->dtype) << ':';
    for (const auto extent : description->dims)
      key << extent << ',';
    key << ';';
  };
  key << 'i' << operation.inputs.size() << ':';
  for (const auto id : operation.inputs)
    tensor(id);
  key << 'o' << operation.outputs.size() << ':';
  for (const auto id : operation.outputs)
    tensor(id);
  key << 'a' << operation.attributes.size() << ':';
  for (const auto &attribute : operation.attributes)
    key << static_cast<std::uint32_t>(attribute.key) << ','
        << static_cast<std::uint32_t>(attribute.kind) << ','
        << attribute.bits << ';';
  return std::move(key).str();
}

struct LowbitLinearFusion {
  const ir::Operation *dequant{};
  const ir::Operation *linear{};
  std::uint64_t qkv_component{3U};
  std::uint64_t qkv_head_dim{};
};

std::unordered_map<std::uint32_t, LowbitLinearFusion>
find_lowbit_linear_fusions(const ir::Program &program,
                           std::unordered_set<std::uint32_t> &skipped) {
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>>
      consumers;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::unordered_map<std::uint32_t, LowbitLinearFusion> fusions;
  // H3 source-shaped QKV is dequantized, deinterleaved, then consumed by three
  // independent Linear operations. Fuse the complete chain only when none of
  // its intermediate values are observable or shared.
  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::H3DeinterleaveQkvWeight ||
        operation.inputs.size() != 1U || operation.outputs.size() != 3U)
      continue;
    const auto dequant_found = producer.find(operation.inputs[0]);
    if (dequant_found == producer.end() ||
        dequant_found->second->opcode != ir::Opcode::DequantizeInt5 ||
        consumers[operation.inputs[0]].size() != 1U ||
        program.tensor(operation.inputs[0])->has_role(ir::TensorRole::Output))
      continue;
    bool eligible = true;
    std::array<const ir::Operation *, 3> linears{};
    for (std::size_t component = 0; component < operation.outputs.size();
         ++component) {
      const auto weight = operation.outputs[component];
      const auto &uses = consumers[weight];
      if (uses.size() != 1U || uses.front()->opcode != ir::Opcode::Linear ||
          uses.front()->inputs.size() < 2U || uses.front()->inputs[1] != weight ||
          uses.front()->u64(ir::AttrKey::Implementation, 1U) != 3U ||
          program.tensor(weight)->has_role(ir::TensorRole::Output)) {
        eligible = false;
        break;
      }
      linears[component] = uses.front();
    }
    if (!eligible)
      continue;
    const auto head_dim = operation.u64(ir::AttrKey::HeadDim, 0U);
    for (std::size_t component = 0; component < linears.size(); ++component)
      fusions.emplace(linears[component]->id,
                      LowbitLinearFusion{dequant_found->second,
                                         linears[component], component,
                                         head_dim});
    skipped.insert(dequant_found->second->id);
    skipped.insert(operation.id);
  }

  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::Linear ||
        operation.inputs.size() < 2U || fusions.contains(operation.id))
      continue;
    if (operation.u64(ir::AttrKey::Implementation, 1U) != 3U)
      continue;
    const auto weight = operation.inputs[1];
    const auto dequant_found = producer.find(weight);
    if (dequant_found == producer.end() ||
        dequant_found->second->opcode != ir::Opcode::DequantizeInt5 ||
        consumers[weight].size() != 1U ||
        program.tensor(weight)->has_role(ir::TensorRole::Output))
      continue;
    fusions.emplace(operation.id,
                    LowbitLinearFusion{dequant_found->second, &operation});
    skipped.insert(dequant_found->second->id);
  }
  return fusions;
}

void emit_lowbit_linear(std::ostringstream &out, const ir::Program &program,
                        const LowbitLinearFusion &fusion) {
  const auto &linear = *fusion.linear;
  const auto &dequant = *fusion.dequant;
  const auto *input = program.tensor(linear.inputs[0]);
  const auto *weight = program.tensor(linear.inputs[1]);
  const auto *output = program.tensor(linear.outputs[0]);
  const auto *source_weight = program.tensor(dequant.outputs[0]);
  const auto k = weight->dims[1];
  const auto n = weight->dims[0];
  const auto m = input->element_count() / k;
  const auto group = dequant.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = k / group;
  const auto row_bytes = source_weight->dims[1] * 5U / 8U;
  const bool column_scaled = dequant.inputs.size() == 3U;
  const bool biased = linear.inputs.size() == 3U;
  if (m == 0U || m > 32U)
    fail("direct packed INT5 Linear currently admits flattened M in [1,32]");
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  std::string parameters =
      "const dif_scalar* x, const unsigned char* packed, const dif_scalar* scales, ";
  if (column_scaled)
    parameters += "const dif_scalar* column_scales, ";
  if (biased)
    parameters += "const dif_scalar* bias, ";
  parameters += "dif_scalar* y";
  const std::string source_row =
      fusion.qkv_component < 3U
          ? "(row / " + text(fusion.qkv_head_dim) + "ULL) * " +
                text(fusion.qkv_head_dim * 3U) + "ULL + " +
                text(fusion.qkv_component * fusion.qkv_head_dim) + "ULL + (row % " +
                text(fusion.qkv_head_dim) + "ULL)"
          : std::string("row");
  std::string accumulators, fma, shuffle, partials;
  for (std::uint64_t batch = 0; batch < m; ++batch) {
    const auto acc = "acc" + text(batch);
    accumulators += "  float " + acc + " = 0.0f;\n";
    fma += "    " + acc + " = fmaf(dif_load(x, " + text(batch * k) + "ULL + col), w, " + acc + ");\n";
    shuffle += "    " + acc + " += __shfl_down_sync(0xffffffffU, " + acc + ", offset);\n";
    partials += "    partials[warp * " + text(m) + "ULL + " + text(batch) + "ULL] = " + acc + ";\n";
  }
  out << render_kernel_template(
      "lowbit_linear",
      {{"function", function_name(linear)},
       {"parameters", parameters},
       {"n", text(n)},
       {"source_row", source_row},
       {"accumulators", accumulators},
       {"k", text(k)},
       {"row_bytes", text(row_bytes)},
       {"groups", text(groups)},
       {"group", text(group)},
       {"column_scale", column_scaled ? "w *= dif_load(column_scales, col);" : ""},
       {"fma", fma},
       {"shuffle", shuffle},
       {"partials", partials},
       {"m", text(m)},
       {"bias", biased ? "total += dif_load(bias, row);" : ""}});
  (void)output;
}


// BigVGAN audio decode emitters.
// Contract: generic launch — one thread per output element, geometry baked
// as literals, F32 accumulation, dif_load/dif_store storage boundary. The
// loop order mirrors the CPU reference exactly (forward: in-channel outer,
// kernel-tap inner; transposed: gather with the padded input index
// ascending — the same per-output add order as the CPU scatter), so
// CPU-vs-CUDA differences stay at FP-contraction level.
void emit_conv1d(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *weight = program.tensor(op.inputs[1]);
  const auto *output = program.tensor(op.outputs[0]);
  const bool biased = op.inputs.size() == 3U;
  const auto stride = op.u64(ir::AttrKey::Stride, 1U);
  const auto dilation = op.u64(ir::AttrKey::Dilation, 1U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto pad_left = op.u64(ir::AttrKey::PadLeft, 0U);
  const auto pad_right = op.u64(ir::AttrKey::PadRight, 0U);
  const auto replicate = op.u64(ir::AttrKey::PadMode, 0U) == 1U;
  const auto transposed = op.boolean(ir::AttrKey::Transposed, false);
  const auto trim_left = op.u64(ir::AttrKey::TrimLeft, 0U);
  const auto in_channels = input->dims[1];
  const auto length = input->dims[2];
  const auto kernel = weight->dims[2];
  const auto out_channels = output->dims[1];
  const auto out_length = output->dims[2];
  const auto count = output->element_count();
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  const auto padded = length + pad_left + pad_right;
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  // Padded-coordinate sampler: position maps to input index pos-PadLeft,
  // replicate-clamped or zero outside; emitted as a literal expression.
  const auto clamped_sample = [&](const std::string &position) {
    const auto shifted = "(" + position + " - " + text(pad_left) + "LL)";
    if (!replicate)
      return "((" + shifted + " >= 0LL && " + shifted + " < " + text(length) +
             "LL) ? dif_load(xrow, (unsigned long long)" + shifted + ") : 0.0f)";
    const auto last = text(length - 1U) + "LL";
    return "dif_load(xrow, (unsigned long long)(" + shifted + " < 0LL ? 0LL : (" +
           shifted + " > " + last + " ? " + last + " : " + shifted + ")))";
  };
  const std::string accumulate =
      transposed
          ? render_kernel_template(
                "conv1d_transposed",
                {{"out_per_group", text(out_per_group)},
                 {"trim_left", text(trim_left)},
                 {"in_per_group", text(in_per_group)},
                 {"in_channels", text(in_channels)},
                 {"length", text(length)},
                 {"kernel", text(kernel)},
                 {"kernel_minus_one", text(kernel - 1U)},
                 {"stride", text(stride)},
                 {"padded_minus_one", text(padded - 1U)},
                 {"sample", clamped_sample("pi")}})
          : render_kernel_template(
                "conv1d_forward",
                {{"in_per_group", text(in_per_group)},
                 {"in_channels", text(in_channels)},
                 {"length", text(length)},
                 {"kernel", text(kernel)},
                 {"stride", text(stride)},
                 {"dilation", text(dilation)},
                 {"sample", clamped_sample("p")}});
  out << render_kernel_template(
      "conv1d", {{"function", function_name(op)},
                 {"bias_parameter", biased ? "const dif_scalar* bias, " : ""},
                 {"count", text(count)},
                 {"out_length", text(out_length)},
                 {"out_channels", text(out_channels)},
                 {"out_stride", text(out_channels * out_length)},
                 {"out_per_group", text(out_per_group)},
                 {"accumulate", accumulate},
                 {"bias", biased ? "acc += dif_load(bias, oc);" : ""}});
}

void emit_channel_rms_norm(std::ostringstream &out,
                           const ir::Program &program,
                           const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto axis = op.u64(ir::AttrKey::Axis, 1U);
  const auto channels = input->dims[axis];
  std::uint64_t inner = 1U;
  for (std::size_t index = static_cast<std::size_t>(axis + 1U);
       index < input->dims.size(); ++index)
    inner *= input->dims[index];
  const auto vectors = input->element_count() / channels;
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto epsilon = static_cast<float>(
      op.f64(ir::AttrKey::Epsilon, 1.0e-12));
  const auto scale = std::sqrt(static_cast<float>(channels));
  const auto literal = [](float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(9) << value;
    return text.str();
  };
  std::string reduction;
  for (auto stride = block / 2U; stride != 0U; stride /= 2U)
    reduction += "  if (c < " + std::to_string(stride) +
                 "ULL) reduction[c] += reduction[c + " +
                 std::to_string(stride) + "ULL];\n  __syncthreads();\n";
  out << render_kernel_template(
      "channel_rms_norm", {{"function", function_name(op)},
                           {"vectors", std::to_string(vectors)},
                           {"inner", std::to_string(inner)},
                           {"channels", std::to_string(channels)},
                           {"reduction", reduction},
                           {"epsilon", literal(epsilon)},
                           {"scale", literal(scale)}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_group_norm(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto channels = input->dims[1];
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto channels_per_group = channels / groups;
  std::uint64_t inner = 1U;
  for (std::size_t axis = 2U; axis < input->dims.size(); ++axis)
    inner *= input->dims[axis];
  const auto elements = channels_per_group * inner;
  const auto block = ir::group_norm_block_size(op, *input);
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  std::string reduction;
  for (auto stride = block / 2U; stride != 0U; stride /= 2U) {
    const auto text = std::to_string(stride);
    reduction += "  if (lane < " + text + "ULL) {\n    reduction[lane] += reduction[lane + " +
                 text + "ULL];\n    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + " +
                 text + "ULL];\n  }\n  __syncthreads();\n";
  }
  out << render_kernel_template(
      "group_norm", {{"function", function_name(op)},
                     {"vectors", std::to_string(input->dims[0] * groups)},
                     {"groups", std::to_string(groups)},
                     {"channels", std::to_string(channels)},
                     {"channels_per_group", std::to_string(channels_per_group)},
                     {"inner", std::to_string(inner)},
                     {"elements", std::to_string(elements)},
                     {"reduction", reduction},
                     {"epsilon", epsilon_literal.str()}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_upsample_nearest_2d(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto channels = input->dims[1];
  const auto input_h = input->dims[2];
  const auto input_w = input->dims[3];
  const auto output_h = output->dims[2];
  const auto output_w = output->dims[3];
  const auto scale_h = op.u64(ir::AttrKey::ScaleH, 1U);
  const auto scale_w = op.u64(ir::AttrKey::ScaleW, 1U);
  out << render_kernel_template(
      "upsample_nearest_2d",
      {{"function", function_name(op)},
       {"count", std::to_string(output->element_count())},
       {"output_w", std::to_string(output_w)},
       {"output_h", std::to_string(output_h)},
       {"output_hw", std::to_string(output_w * output_h)},
       {"channels", std::to_string(channels)},
       {"output_chw", std::to_string(channels * output_h * output_w)},
       {"input_h", std::to_string(input_h)},
       {"scale_h", std::to_string(scale_h)},
       {"input_w", std::to_string(input_w)},
       {"scale_w", std::to_string(scale_w)}});
}

void emit_pad_constant(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  std::ostringstream value_literal;
  value_literal << std::scientific << std::setprecision(9) << value;
  const auto &o = output->dims;
  const auto &in = input->dims;
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  if (input->dims.size() == 4U) {
    out << render_kernel_template(
        "pad_constant_4d",
        {{"function", function_name(op)},
         {"count", text(output->element_count())},
         {"out_w", text(o[3])}, {"out_h", text(o[2])},
         {"out_hw", text(o[2] * o[3])}, {"out_c", text(o[1])},
         {"out_chw", text(o[1] * o[2] * o[3])},
         {"top", text(top)}, {"bottom_edge", text(top + in[2])},
         {"west", text(west)}, {"east_edge", text(west + in[3])},
         {"value", value_literal.str()},
         {"in_c", text(in[1])}, {"in_h", text(in[2])}, {"in_w", text(in[3])}});
  } else {
    const auto front = op.u64(ir::AttrKey::PadFront, 0U);
    out << render_kernel_template(
        "pad_constant_5d",
        {{"function", function_name(op)},
         {"count", text(output->element_count())},
         {"out_w", text(o[4])}, {"out_h", text(o[3])},
         {"out_hw", text(o[3] * o[4])}, {"out_t", text(o[2])},
         {"out_thw", text(o[2] * o[3] * o[4])}, {"out_c", text(o[1])},
         {"out_cthw", text(o[1] * o[2] * o[3] * o[4])},
         {"front", text(front)}, {"back_edge", text(front + in[2])},
         {"top", text(top)}, {"bottom_edge", text(top + in[3])},
         {"west", text(west)}, {"east_edge", text(west + in[4])},
         {"value", value_literal.str()},
         {"in_c", text(in[1])}, {"in_t", text(in[2])}, {"in_h", text(in[3])},
         {"in_w", text(in[4])}});
  }
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_pad_reflect(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto front = op.u64(ir::AttrKey::PadFront, 0U);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  const auto &o = output->dims;
  const auto &in = input->dims;
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  if (input->dims.size() == 4U) {
    out << render_kernel_template(
        "pad_reflect_4d",
        {{"function", function_name(op)},
         {"count", text(output->element_count())},
         {"out_w", text(o[3])}, {"out_h", text(o[2])},
         {"out_hw", text(o[2] * o[3])}, {"out_c", text(o[1])},
         {"out_chw", text(o[1] * o[2] * o[3])},
         {"top", text(top)}, {"west", text(west)},
         {"in_c", text(in[1])}, {"in_h", text(in[2])}, {"in_w", text(in[3])}});
  } else {
    out << render_kernel_template(
        "pad_reflect_5d",
        {{"function", function_name(op)},
         {"count", text(output->element_count())},
         {"out_w", text(o[4])}, {"out_h", text(o[3])},
         {"out_hw", text(o[3] * o[4])}, {"out_t", text(o[2])},
         {"out_thw", text(o[2] * o[3] * o[4])}, {"out_c", text(o[1])},
         {"out_cthw", text(o[1] * o[2] * o[3] * o[4])},
         {"front", text(front)}, {"top", text(top)}, {"west", text(west)},
         {"in_c", text(in[1])}, {"in_t", text(in[2])}, {"in_h", text(in[3])},
         {"in_w", text(in[4])}});
  }
}

void emit_snake_beta(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto channels = input->dims[1];
  const auto length = input->dims[2];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-9));
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  out << render_kernel_template(
      "snake_beta", {{"function", function_name(op)},
                     {"count", std::to_string(count)},
                     {"length", std::to_string(length)},
                     {"channels", std::to_string(channels)},
                     {"epsilon", epsilon_literal.str()}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_header(std::ostringstream &out) {
  out << R"CUDA(
typedef float dif_f32;
typedef unsigned short dif_bf16;
typedef unsigned short dif_f16;
extern "C" __device__ float dif_load_f32(const dif_f32* value, unsigned long long index) {
  return value[index];
}
extern "C" __device__ void dif_store_f32(dif_f32* value, unsigned long long index, float input) {
  value[index] = input;
}
extern "C" __device__ float dif_round_f32(float input) { return input; }
extern "C" __device__ float dif_load_bf16(const dif_bf16* value, unsigned long long index) {
  return __uint_as_float((unsigned int)value[index] << 16U);
}
extern "C" __device__ void dif_store_bf16(dif_bf16* value, unsigned long long index, float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  value[index] = (dif_bf16)((bits + rounding) >> 16U);
}
extern "C" __device__ float dif_round_bf16(float input) {
  unsigned int bits = __float_as_uint(input);
  unsigned int rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return __uint_as_float(((bits + rounding) >> 16U) << 16U);
}
extern "C" __device__ float dif_load_f16(const dif_f16* value, unsigned long long index) {
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(value[index]));
  return result;
}
extern "C" __device__ void dif_store_f16(dif_f16* value, unsigned long long index, float input) {
  dif_f16 result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(input));
  value[index] = result;
}
extern "C" __device__ float dif_round_f16(float input) {
  dif_f16 rounded;
  float result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(rounded) : "f"(input));
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(rounded));
  return result;
}
extern "C" __device__ float dif_silu(float x) {
  return x / (1.0f + expf(-x));
}
)CUDA";
}

ir::DType operation_float_dtype(const ir::Program &program,
                                const ir::Operation &op) {
  for (const auto &ids : {&op.outputs, &op.inputs}) {
    for (const auto id : *ids) {
      const auto dtype = program.tensor(id)->dtype;
      if (dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
          dtype == ir::DType::F16)
        return dtype;
    }
  }
  fail("CUDA operation has no supported float tensor");
}

void begin_float_operation(std::ostringstream &out, ir::DType dtype) {
  if (dtype == ir::DType::F32) {
    out << "#define dif_scalar dif_f32\n#define dif_load dif_load_f32\n"
           "#define dif_store dif_store_f32\n#define dif_round dif_round_f32\n";
    return;
  }
  if (dtype == ir::DType::BF16) {
    out << "#define dif_scalar dif_bf16\n#define dif_load dif_load_bf16\n"
           "#define dif_store dif_store_bf16\n#define dif_round dif_round_bf16\n";
    return;
  }
  if (dtype == ir::DType::F16) {
    out << "#define dif_scalar dif_f16\n#define dif_load dif_load_f16\n"
           "#define dif_store dif_store_f16\n#define dif_round dif_round_f16\n";
    return;
  }
  fail("CUDA source emitter admits f32, bf16, or f16 operations");
}

void end_float_operation(std::ostringstream &out) {
  out << "#undef dif_scalar\n#undef dif_load\n#undef dif_store\n"
         "#undef dif_round\n";
}

void emit_cast(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto type = [](ir::DType dtype) -> const char * {
    if (dtype == ir::DType::F32)
      return "dif_f32";
    if (dtype == ir::DType::BF16)
      return "dif_bf16";
    if (dtype == ir::DType::F16)
      return "dif_f16";
    fail("CUDA Cast admits f32, bf16, and f16 storage");
  };
  const auto load = input->dtype == ir::DType::F32
                        ? "dif_load_f32"
                        : input->dtype == ir::DType::BF16 ? "dif_load_bf16"
                                                          : "dif_load_f16";
  const auto store = output->dtype == ir::DType::F32
                         ? "dif_store_f32"
                         : output->dtype == ir::DType::BF16 ? "dif_store_bf16"
                                                            : "dif_store_f16";
  out << render_kernel_template(
      "cast", {{"function", function_name(op)},
               {"input_type", type(input->dtype)},
               {"output_type", type(output->dtype)},
               {"count", std::to_string(count)},
               {"load", load},
               {"store", store}});
}

void emit_quantize_int8_rows(std::ostringstream &out,
                             const ir::Program &program,
                             const ir::Operation &op) {
  const auto *quantized = program.tensor(op.outputs[0]);
  const auto columns = quantized->dims.back();
  const auto rows = quantized->element_count() / columns;
  const auto dynamic_clip =
      program.tensor(op.inputs.back())->dtype == ir::DType::F32;
  const auto data_input_count =
      op.inputs.size() - static_cast<std::size_t>(dynamic_clip);
  const auto input_parameters = [&] {
    std::string result;
    for (std::size_t index = 0U; index < data_input_count; ++index)
      result += "const dif_bf16* x" + std::to_string(index) + ",";
    if (dynamic_clip)
      result += "const dif_f32* clip_ratio,";
    return result;
  }();
  const auto load_input = [&] {
    std::string result;
    std::uint64_t prefix = 0U;
    for (std::size_t index = 0U; index < data_input_count; ++index) {
      const auto width = program.tensor(op.inputs[index])->dims.back();
      const auto load = "dif_load_bf16(x" + std::to_string(index) +
                        ",row*" + std::to_string(width) + "ULL+column-" +
                        std::to_string(prefix) + "ULL)";
      if (index + 1U == data_input_count) {
        result += load;
      } else {
        result += "(column<" + std::to_string(prefix + width) + "ULL?" +
                  load + ":";
      }
      prefix += width;
    }
    result.append(data_input_count - 1U, ')');
    return result;
  }();
  const auto residual2 = op.outputs.size() == 4U;
  const auto implementation = static_cast<ir::Int8RowQuantization>(
      op.u64(ir::AttrKey::Implementation,
             static_cast<std::uint64_t>(
                 ir::Int8RowQuantization::Direct)));
  const auto clip_ratio = op.f64(ir::AttrKey::Scale, 1.0);
  std::ostringstream clip_ratio_literal;
  clip_ratio_literal << std::scientific << std::setprecision(9)
                     << static_cast<float>(clip_ratio);
  const auto clip_ratio_source =
      dynamic_clip ? std::string{"dif_load_f32(clip_ratio,0ULL)"}
                   : clip_ratio_literal.str() + "f";
  const auto sylvester =
      implementation == ir::Int8RowQuantization::H256F32SylvesterConvRot;
  const std::string parameters =
      input_parameters + "signed char* q, float* scales" +
      (residual2 ? ", signed char* q2, float* scales2" : "");
  if (ir::is_convrot_int8_row_quantization(implementation)) {
    const auto f32_convrot =
        implementation == ir::Int8RowQuantization::H256F32ConvRot ||
        implementation == ir::Int8RowQuantization::H256F32SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096F32SignedConvRot ||
        sylvester;
    const auto signed_rotation =
        implementation == ir::Int8RowQuantization::H256SignedConvRot ||
        implementation == ir::Int8RowQuantization::H256F32SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096SignedConvRot ||
        implementation == ir::Int8RowQuantization::H4096F32SignedConvRot;
    const auto rotation_group =
        (implementation == ir::Int8RowQuantization::H4096SignedConvRot ||
         implementation == ir::Int8RowQuantization::H4096F32SignedConvRot)
            ? 4096U
            : 256U;
    auto rotation_stages = 0U;
    for (auto width = rotation_group; width > 1U; width /= 4U)
      ++rotation_stages;
    // Per-column sign flip keyed by a splitmix-style hash of the column.
    const std::string signed_rotation_source =
        signed_rotation
            ? "unsigned h = (unsigned)column + " +
                  std::to_string(0x9e3779b9U) +
                  "U;\n    h = (h ^ (h >> 16)) * 0x7feb352dU;\n"
                  "    h = (h ^ (h >> 15)) * 0x846ca68bU;\n    h ^= h >> 16;\n"
                  "    value = (h & 1U) ? -value : value;"
            : std::string{};
    const std::string butterfly =
        sylvester ? "values[i] = 0.5f * (x0 + x1 + x2 + x3);\n"
                    "      values[i + stride] = 0.5f * (x0 - x1 + x2 - x3);\n"
                    "      values[i + 2U * stride] = 0.5f * (x0 + x1 - x2 - x3);\n"
                    "      values[i + 3U * stride] = 0.5f * (x0 - x1 - x2 + x3);"
                  : "values[i] = 0.5f * (x0 + x1 + x2 - x3);\n"
                    "      values[i + stride] = 0.5f * (x0 + x1 - x2 + x3);\n"
                    "      values[i + 2U * stride] = 0.5f * (x0 - x1 + x2 + x3);\n"
                    "      values[i + 3U * stride] = 0.5f * (-x0 + x1 + x2 + x3);";
    const std::string encode =
        f32_convrot
            ? "float value = values[column];\n    float divided = value / scale;\n"
              "    int encoded = (int)nearbyintf(divided);\n"
              "    encoded = encoded > 127 ? 127 : (encoded < -127 ? -127 : encoded);"
            : "float value = dif_round_bf16(values[column]);\n"
              "    float divided = dif_round_bf16(value / scale_bf16);\n"
              "    int encoded = (int)nearbyintf(divided);\n"
              "    encoded = encoded > 127 ? 127 : (encoded < -128 ? -128 : encoded);";
    const auto bf16_scales = signed_rotation && !f32_convrot;
    const std::string residual =
        residual2
            ? render_kernel_template(
                  "quantize_int8_rows_convrot_residual2",
                  {{"columns", std::to_string(columns)},
                   {"value", f32_convrot ? "values[column]"
                                         : "dif_round_bf16(values[column])"},
                   {"scale2_store", bf16_scales ? "scale2_bf16" : "scale2"},
                   {"residual",
                    f32_convrot
                        ? "fmaf(-(float)q[base + column], stored_scale, value)"
                        : "dif_round_bf16(fmaf(-(float)q[base + column], "
                          "stored_scale, value))"},
                   {"divided", f32_convrot
                                   ? "residual / scale2"
                                   : "dif_round_bf16(residual / scale2_bf16)"},
                   {"clamp_low", f32_convrot ? "-127" : "-128"}})
            : std::string{};
    out << render_kernel_template(
        "quantize_int8_rows_convrot",
        {{"function", function_name(op)},
         {"parameters", parameters},
         {"rows", std::to_string(rows)},
         {"columns", std::to_string(columns)},
         {"load_input", load_input},
         {"signed_rotation", signed_rotation_source},
         {"rotation_stages", std::to_string(rotation_stages)},
         {"rotation_lanes", std::to_string(rotation_group / 4U)},
         {"rotation_group", std::to_string(rotation_group)},
         {"butterfly", butterfly},
         {"clip_ratio", clip_ratio_source},
         {"scale_store", bf16_scales ? "scale_bf16" : "scale"},
         {"encode", encode},
         {"residual2", residual}});
    return;
  }
  const std::string residual =
      residual2 ? render_kernel_template(
                      "quantize_int8_rows_direct_residual2",
                      {{"columns", std::to_string(columns)},
                       {"load_input", load_input}})
                : std::string{};
  out << render_kernel_template(
      "quantize_int8_rows_direct",
      {{"function", function_name(op)},
       {"parameters", parameters},
       {"rows", std::to_string(rows)},
       {"columns", std::to_string(columns)},
       {"load_input", load_input},
       {"clip_ratio", clip_ratio_source},
       {"residual2", residual}});
}

void emit_dequantize_int8_blocks(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scales = program.tensor(op.inputs[1]);
  const auto count = input->element_count();
  const auto columns = input->dims[1];
  const auto scale_columns = scales->dims[1];
  const auto block = op.u64(ir::AttrKey::BlockSize, 0U);
  out << render_kernel_template(
      "dequantize_int8_blocks", {{"function", function_name(op)},
                                 {"count", std::to_string(count)},
                                 {"columns", std::to_string(columns)},
                                 {"scale_columns", std::to_string(scale_columns)},
                                 {"block", std::to_string(block)}});
}

void emit_quantize_fp8_rows(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  out << render_kernel_template(
      "quantize_fp8_rows", {{"function", function_name(op)},
                            {"rows", std::to_string(rows)},
                            {"columns", std::to_string(columns)}});
}

void emit_linear_fp8_output_scale(std::ostringstream &out,
                                  const ir::Program &program,
                                  const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *weight = program.tensor(op.inputs[1]);
  const auto rows = input->element_count() / input->dims.back();
  const auto columns = weight->dims.front();
  const auto count = rows * columns;
  out << render_kernel_template(
      "linear_fp8_output_scale", {{"function", function_name(op)},
                                  {"count", std::to_string(count)},
                                  {"columns", std::to_string(columns)}});
}

void emit_quantize_fp8_blocks32(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scales = program.tensor(op.outputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto blocks = (columns + 31U) / 32U;
  const auto scale_inner_dimension = scales->dims.at(1);
  out << render_kernel_template(
      "quantize_fp8_blocks32", {{"function", function_name(op)},
                                {"rows", std::to_string(rows)},
                                {"blocks", std::to_string(blocks)},
                                {"columns", std::to_string(columns)},
                                {"scale_inner", std::to_string(scale_inner_dimension)}});
}

void emit_elementwise(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op, const char *expression) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "elementwise", {{"function", function_name(op)},
                      {"count", std::to_string(count)},
                      {"expression", expression}});
}

void emit_affine_last_dim(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[1])->dims[0];
  const bool biased = op.inputs.size() == 3U;
  out << render_kernel_template(
      "affine_last_dim",
      {{"function", function_name(op)},
       {"parameters",
        biased ? "const dif_scalar* x, const dif_scalar* scale, "
                 "const dif_scalar* bias, dif_scalar* y"
               : "const dif_scalar* x, const dif_scalar* scale, dif_scalar* y"},
       {"count", std::to_string(count)},
       {"width", std::to_string(width)},
       {"bias", biased ? "value = dif_round(value + dif_load(bias, col));"
                       : ""}});
}

void emit_clamp(std::ostringstream &out, const ir::Program &program,
                const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto lower = static_cast<float>(op.f64(ir::AttrKey::Lower, 0.0));
  const auto upper = static_cast<float>(op.f64(ir::AttrKey::Upper, 1.0));
  std::ostringstream lower_literal, upper_literal;
  lower_literal << std::scientific << std::setprecision(9) << lower;
  upper_literal << std::scientific << std::setprecision(9) << upper;
  out << render_kernel_template(
      "clamp", {{"function", function_name(op)},
                {"count", std::to_string(count)},
                {"lower", lower_literal.str()},
                {"upper", upper_literal.str()}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_silu(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "silu", {{"function", function_name(op)},
               {"count", std::to_string(count)}});
}

void emit_gelu(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto approximation = static_cast<ir::GeluApproximation>(
      op.u64(ir::AttrKey::Approximation, 0U));
  const std::string body =
      approximation == ir::GeluApproximation::ExactErf
          ? "dif_store(y, i, 5.0e-1f * v * (1.0f + erff(v * 7.071067812e-1f)));"
      : approximation == ir::GeluApproximation::QuickSigmoid
          ? "dif_store(y, i, v / (1.0f + expf(-1.702f * v)));"
          : "float c = v * v * v;\n"
            "    float z = 7.978845608e-1f * (v + 4.471500218e-2f * c);\n"
            "    dif_store(y, i, 5.0e-1f * v * (1.0f + tanhf(z)));";
  out << render_kernel_template(
      "gelu", {{"function", function_name(op)},
               {"count", std::to_string(count)},
               {"approximation", body}});
}

// The training ops below may legally mix storage dtypes across their
// arguments (e.g. BF16 prediction into an F32 loss, BF16 parameters with F32
// moments), so they bypass the per-operation dif_load/dif_store macro system
// and name the typed load/store helpers per argument.  All arithmetic stays
// in F32 registers; rounding happens only at the typed store.
const char *typed_scalar(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_f32";
  if (dtype == ir::DType::BF16)
    return "dif_bf16";
  if (dtype == ir::DType::F16)
    return "dif_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

const char *typed_load(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_load_f32";
  if (dtype == ir::DType::BF16)
    return "dif_load_bf16";
  if (dtype == ir::DType::F16)
    return "dif_load_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

const char *typed_store(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_store_f32";
  if (dtype == ir::DType::BF16)
    return "dif_store_bf16";
  if (dtype == ir::DType::F16)
    return "dif_store_f16";
  fail("CUDA training emitter admits f32, bf16, or f16 storage");
}

void emit_sigmoid(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "sigmoid", {{"function", function_name(op)},
                  {"count", std::to_string(count)}});
}

void emit_reshape(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "reshape", {{"function", function_name(op)},
                  {"count", std::to_string(count)}});
}

void emit_broadcast_to(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto rank_pad = output->dims.size() - input->dims.size();
  std::vector<std::uint64_t> strides(input->dims.size(), 1U);
  for (std::size_t axis = input->dims.size(); axis-- > 1U;)
    strides[axis - 1U] = strides[axis] * input->dims[axis];
  std::string axes;
  for (std::size_t axis = output->dims.size(); axis-- > 0U;) {
    axes += "    at = coordinate % " + std::to_string(output->dims[axis]) +
            "ULL; coordinate /= " + std::to_string(output->dims[axis]) +
            "ULL;\n";
    if (axis >= rank_pad && input->dims[axis - rank_pad] != 1U)
      axes += "    source += at * " +
              std::to_string(strides[axis - rank_pad]) + "ULL;\n";
  }
  out << render_kernel_template(
      "broadcast_to", {{"function", function_name(op)},
                       {"count", std::to_string(output->element_count())},
                       {"axes", axes}});
}

void emit_slice(std::ostringstream &out, const ir::Program &program,
                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto selected =
      static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  std::vector<std::uint64_t> strides(input->dims.size(), 1U);
  for (std::size_t axis = input->dims.size(); axis-- > 1U;)
    strides[axis - 1U] = strides[axis] * input->dims[axis];
  std::string axes;
  for (std::size_t axis = output->dims.size(); axis-- > 0U;) {
    axes += "    at = coordinate % " + std::to_string(output->dims[axis]) +
            "ULL; coordinate /= " + std::to_string(output->dims[axis]) +
            "ULL;\n";
    if (axis == selected)
      axes += "    at += " + std::to_string(start) + "ULL;\n";
    axes += "    source += at * " + std::to_string(strides[axis]) + "ULL;\n";
  }
  out << render_kernel_template(
      "slice", {{"function", function_name(op)},
                {"count", std::to_string(output->element_count())},
                {"axes", axes}});
}

void emit_rotary_frequency(std::ostringstream &out,
                           const ir::Program &program,
                           const ir::Operation &op) {
  const auto *positions = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto sequence = positions->dims[1];
  const auto axes = positions->dims[2];
  const auto pairs = output->dims[2];
  std::ostringstream theta_literal, ntk_literal;
  theta_literal << std::setprecision(17) << op.f64(ir::AttrKey::Theta, 10000.0);
  ntk_literal << std::setprecision(17) << op.f64(ir::AttrKey::Ntk, 1.0);
  out << render_kernel_template(
      "rotary_frequency",
      {{"function", function_name(op)},
       {"count", std::to_string(output->element_count())},
       {"pairs", std::to_string(pairs)},
       {"sequence", std::to_string(sequence)},
       {"axes", std::to_string(axes)},
       {"theta", theta_literal.str()},
       {"ntk", ntk_literal.str()}});
  // The historical emitter left the stream at precision 17, defaultfloat.
  out << std::setprecision(17) << std::defaultfloat;
}

void emit_rotary_apply(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *cosine = program.tensor(op.inputs[1]);
  const auto sequence = input->dims[1];
  const auto heads = input->dims[2];
  const auto dim = input->dims[3];
  const auto pairs = cosine->dims[2];
  const bool half_split =
      op.u64(ir::AttrKey::RotaryLayout, 0U) ==
      static_cast<std::uint64_t>(ir::RotaryLayout::HalfSplit);
  // Interleaved: element d belongs to pair d/2, partner (d^1); HalfSplit:
  // pair d % pairs, partner d +- pairs. `second` selects the imaginary half.
  const auto pairs_text = std::to_string(pairs);
  out << render_kernel_template(
      "rotary_apply",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(input->dtype)},
       {"count", std::to_string(input->element_count())},
       {"dim", std::to_string(dim)},
       {"heads", std::to_string(heads)},
       {"sequence", std::to_string(sequence)},
       {"rotated", std::to_string(2U * pairs)},
       {"pair", half_split ? "d % " + pairs_text + "ULL" : "d / 2ULL"},
       {"pairs", pairs_text},
       {"load", typed_load(input->dtype)},
       {"even_index", half_split ? "base + pair" : "base + 2ULL * pair"},
       {"odd_index", half_split ? "base + pair + " + pairs_text + "ULL"
                                : "base + 2ULL * pair + 1ULL"},
       {"is_second", half_split ? "(d >= " + pairs_text + "ULL)" : "(d & 1ULL)"},
       {"store", typed_store(input->dtype)}});
}

void emit_boolean_mask_to_bias(std::ostringstream &out,
                               const ir::Program &program,
                               const ir::Operation &op) {
  const auto *mask = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto sequence = output->dims[2];
  const bool vector_mask = mask->dims.size() == 2U;
  const bool mask_queries = op.boolean(ir::AttrKey::MaskQueries, true);
  const auto seq = std::to_string(sequence);
  std::string valid;
  if (vector_mask) {
    if (mask_queries)
      valid += "mask[batch * " + seq + "ULL + query] && ";
    valid += "mask[batch * " + seq + "ULL + key]";
  } else {
    valid = "mask[(batch * " + seq + "ULL + query) * " + seq + "ULL + key]";
  }
  out << render_kernel_template(
      "boolean_mask_to_bias",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(output->dtype)},
       {"count", std::to_string(output->element_count())},
       {"sequence", seq},
       {"valid", valid},
       {"store", typed_store(output->dtype)}});
}

void emit_mse_loss(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto *prediction = program.tensor(op.inputs[0]);
  const auto *loss = program.tensor(op.outputs[0]);
  const auto count = prediction->element_count();
  out << render_kernel_template(
      "mse_loss", {{"function", function_name(op)},
                   {"scalar", typed_scalar(prediction->dtype)},
                   {"loss_scalar", typed_scalar(loss->dtype)},
                   {"count", std::to_string(count)},
                   {"load", typed_load(prediction->dtype)},
                   {"store", typed_store(loss->dtype)}});
}

void emit_mse_loss_backward(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *prediction = program.tensor(op.inputs[0]);
  const auto *grad_loss = program.tensor(op.inputs[2]);
  const auto *grad_prediction = program.tensor(op.outputs[0]);
  const auto count = grad_prediction->element_count();
  out << render_kernel_template(
      "mse_loss_backward",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(prediction->dtype)},
       {"grad_loss_scalar", typed_scalar(grad_loss->dtype)},
       {"grad_scalar", typed_scalar(grad_prediction->dtype)},
       {"count", std::to_string(count)},
       {"grad_loss_load", typed_load(grad_loss->dtype)},
       {"grad_store", typed_store(grad_prediction->dtype)},
       {"load", typed_load(prediction->dtype)}});
}

void emit_linear_backward_input(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *weight = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto inner = weight->dims[1];
  const auto outputs = weight->dims[0];
  out << render_kernel_template(
      "linear_backward_input", {{"function", function_name(op)},
                                {"count", std::to_string(count)},
                                {"inner", std::to_string(inner)},
                                {"outputs", std::to_string(outputs)}});
}

void emit_linear_backward_weight(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto *grad_weight = program.tensor(op.outputs[0]);
  const auto count = grad_weight->element_count();
  // Geometry from the [N,K] weight gradient itself: rank-agnostic over both
  // admitted operand forms (same-rank broadcast and flatten).
  const auto outputs = grad_weight->dims[0];
  const auto inner = grad_weight->dims[1];
  const auto rows = input->element_count() / inner;
  out << render_kernel_template(
      "linear_backward_weight", {{"function", function_name(op)},
                                 {"count", std::to_string(count)},
                                 {"inner", std::to_string(inner)},
                                 {"rows", std::to_string(rows)},
                                 {"outputs", std::to_string(outputs)}});
}

void emit_bias_backward(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto width = program.tensor(op.outputs[0])->dims[0];
  const auto rows = grad_output->element_count() / width;
  out << render_kernel_template(
      "bias_backward", {{"function", function_name(op)},
                        {"width", std::to_string(width)},
                        {"rows", std::to_string(rows)}});
}

void emit_silu_backward(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "silu_backward", {{"function", function_name(op)},
                        {"count", std::to_string(count)}});
}

// Shared by the group-normalization backward emitters: the unrolled
// two-array shared-memory tree the forward emitter also uses.
std::string group_reduction_fragment(std::uint64_t block) {
  std::string reduction;
  for (auto stride = block / 2U; stride != 0U; stride /= 2U) {
    const auto text = std::to_string(stride);
    reduction += "  if (lane < " + text + "ULL) {\n    reduction[lane] += reduction[lane + " +
                 text + "ULL];\n    reduction[blockDim.x + lane] += reduction[blockDim.x + lane + " +
                 text + "ULL];\n  }\n  __syncthreads();\n";
  }
  return reduction;
}

void emit_layer_norm_modulate_backward(std::ostringstream &out,
                                       const ir::Program &program,
                                       const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto *scale = program.tensor(op.inputs[4]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto modulation_rows = scale->element_count() / columns;
  std::ostringstream epsilon;
  epsilon << std::setprecision(9)
          << static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  out << render_kernel_template(
      "layer_norm_modulate_backward",
      {{"function", function_name(op)},
       {"count", std::to_string(input->element_count())},
       {"columns", std::to_string(columns)},
       {"rows", std::to_string(rows)},
       {"rows_per_modulation", std::to_string(rows / modulation_rows)},
       {"modulation_count", std::to_string(scale->element_count())},
       {"epsilon", epsilon.str()}});
  out << std::setprecision(9);
}

void emit_gather_rows_backward(std::ostringstream &out,
                               const ir::Program &program,
                               const ir::Operation &op) {
  const auto *grad_input = program.tensor(op.outputs[0]);
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto row_width = grad_input->element_count() / grad_input->dims[0];
  out << render_kernel_template(
      "gather_rows_backward",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(grad_input->dtype)},
       {"count", std::to_string(grad_input->element_count())},
       {"row_width", std::to_string(row_width)},
       {"gathered_rows", std::to_string(grad_output->dims[0])},
       {"load", typed_load(grad_input->dtype)},
       {"store", typed_store(grad_input->dtype)}});
}

void emit_sigmoid_backward(std::ostringstream &out, const ir::Program &program,
                           const ir::Operation &op) {
  out << render_kernel_template(
      "sigmoid_backward",
      {{"function", function_name(op)},
       {"count", std::to_string(
                     program.tensor(op.outputs[0])->element_count())}});
}

void emit_clamp_backward(std::ostringstream &out, const ir::Program &program,
                         const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto count = input->element_count();
  // Infinite bounds are the defaults, and an infinity literal in generated
  // CUDA is a portability hazard; the largest finite float compares the same
  // way for every value the kernel can see.
  const auto bound = [](double value) {
    std::ostringstream text;
    text << std::setprecision(9);
    if (value <= -std::numeric_limits<float>::max())
      text << "-3.402823466e+38";
    else if (value >= std::numeric_limits<float>::max())
      text << "3.402823466e+38";
    else
      text << static_cast<float>(value);
    return text.str();
  };
  out << render_kernel_template(
      "clamp_backward",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(input->dtype)},
       {"count", std::to_string(count)},
       {"load", typed_load(input->dtype)},
       {"store", typed_store(input->dtype)},
       {"lower", bound(op.f64(ir::AttrKey::Lower,
                              -std::numeric_limits<double>::infinity()))},
       {"upper", bound(op.f64(ir::AttrKey::Upper,
                              std::numeric_limits<double>::infinity()))}});
  out << std::setprecision(9);
}

void emit_gelu_backward(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto approximation = static_cast<ir::GeluApproximation>(
      op.u64(ir::AttrKey::Approximation, 0U));
  const std::string derivative =
      approximation == ir::GeluApproximation::ExactErf
          ? "float c = 5.0e-1f * (1.0f + erff(v * 7.071067812e-1f));\n"
            "    float p = 3.989422804e-1f * expf(-5.0e-1f * v * v);\n"
            "    d = c + v * p;"
      : approximation == ir::GeluApproximation::QuickSigmoid
          ? "float s = 1.0f / (1.0f + expf(-1.702f * v));\n"
            "    d = s + v * 1.702f * s * (1.0f - s);"
          : "float c = v * v * v;\n"
            "    float z = 7.978845608e-1f * (v + 4.471500218e-2f * c);\n"
            "    float t = tanhf(z);\n"
            "    float q = 7.978845608e-1f * (1.0f + 1.341450065e-1f * v * v);\n"
            "    d = 5.0e-1f * (1.0f + t) + 5.0e-1f * v * (1.0f - t * t) * q;";
  out << render_kernel_template(
      "gelu_backward", {{"function", function_name(op)},
                        {"count", std::to_string(count)},
                        {"derivative", derivative}});
}

void emit_upsample_nearest_2d_backward(std::ostringstream &out,
                                       const ir::Program &program,
                                       const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto *grad_input = program.tensor(op.outputs[0]);
  const auto text = [](std::uint64_t value) { return std::to_string(value); };
  out << render_kernel_template(
      "upsample_nearest_2d_backward",
      {{"function", function_name(op)},
       {"count", text(grad_input->element_count())},
       {"width", text(grad_input->dims[3])},
       {"height", text(grad_input->dims[2])},
       {"out_width", text(grad_output->dims[3])},
       {"out_height", text(grad_output->dims[2])},
       {"scale_h", text(op.u64(ir::AttrKey::ScaleH, 1U))},
       {"scale_w", text(op.u64(ir::AttrKey::ScaleW, 1U))}});
}

void emit_slice_backward(std::ostringstream &out, const ir::Program &program,
                         const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto *grad_input = program.tensor(op.outputs[0]);
  const auto axis = static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  std::uint64_t inner = 1U;
  for (std::size_t index = axis + 1U; index < grad_input->dims.size(); ++index)
    inner *= grad_input->dims[index];
  const auto text = [](std::uint64_t value) { return std::to_string(value); };
  out << render_kernel_template(
      "slice_backward",
      {{"function", function_name(op)},
       {"count", text(grad_input->element_count())},
       {"inner", text(inner)},
       {"extent", text(grad_input->dims[axis])},
       {"window", text(grad_output->dims[axis])},
       {"start", text(op.u64(ir::AttrKey::Start, 0U))}});
}

void emit_broadcast_to_backward(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto *grad_input = program.tensor(op.outputs[0]);
  const auto rank = grad_output->dims.size();
  const auto pad = rank - grad_input->dims.size();
  std::vector<std::uint64_t> stride(rank, 1U);
  for (std::size_t axis = rank - 1U; axis-- > 0U;)
    stride[axis] = stride[axis + 1U] * grad_output->dims[axis + 1U];
  // Source coordinates give the base offset; axes the broadcast expanded are
  // walked by the generated inner loop instead.
  std::string decompose;
  for (std::size_t axis = grad_input->dims.size(); axis-- > 0U;) {
    const auto extent = grad_input->dims[axis];
    decompose += "    {\n      unsigned long long c = source % " +
                 std::to_string(extent) + "ULL;\n      source /= " +
                 std::to_string(extent) + "ULL;\n";
    if (extent != 1U)
      decompose += "      base += c * " + std::to_string(stride[pad + axis]) +
                   "ULL;\n";
    else
      decompose += "      (void)c;\n";
    decompose += "    }\n";
  }
  std::string expand;
  std::uint64_t repeats = 1U;
  for (std::size_t axis = 0U; axis < rank; ++axis) {
    const auto source =
        axis < pad ? 1U : grad_input->dims[axis - pad];
    if (source != 1U || grad_output->dims[axis] == 1U)
      continue;
    repeats *= grad_output->dims[axis];
    expand += "      {\n        unsigned long long c = remainder % " +
              std::to_string(grad_output->dims[axis]) +
              "ULL;\n        remainder /= " +
              std::to_string(grad_output->dims[axis]) +
              "ULL;\n        offset += c * " + std::to_string(stride[axis]) +
              "ULL;\n      }\n";
  }
  if (expand.empty())
    expand = "      (void)remainder;\n";
  out << render_kernel_template(
      "broadcast_to_backward",
      {{"function", function_name(op)},
       {"count", std::to_string(grad_input->element_count())},
       {"decompose", decompose},
       {"repeats", std::to_string(repeats)},
       {"expand", expand}});
}

void emit_group_norm_backward(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto channels = input->dims[1];
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto channels_per_group = channels / groups;
  std::uint64_t inner = 1U;
  for (std::size_t axis = 2U; axis < input->dims.size(); ++axis)
    inner *= input->dims[axis];
  const auto block = ir::group_norm_block_size(op, *input);
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9)
                  << static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const auto text = [](std::uint64_t value) { return std::to_string(value); };
  out << render_kernel_template(
      "group_norm_backward",
      {{"function", function_name(op)},
       {"vectors", text(input->dims[0] * groups)},
       {"groups", text(groups)},
       {"channels", text(channels)},
       {"channels_per_group", text(channels_per_group)},
       {"inner", text(inner)},
       {"elements", text(channels_per_group * inner)},
       {"epsilon", epsilon_literal.str()},
       {"reduction", group_reduction_fragment(block)}});
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_group_norm_backward_affine(std::ostringstream &out,
                                     const ir::Program &program,
                                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto channels = input->dims[1];
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto channels_per_group = channels / groups;
  std::uint64_t inner = 1U;
  for (std::size_t axis = 2U; axis < input->dims.size(); ++axis)
    inner *= input->dims[axis];
  const auto block = ir::group_norm_block_size(op, *input);
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9)
                  << static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const auto text = [](std::uint64_t value) { return std::to_string(value); };
  out << render_kernel_template(
      "group_norm_backward_affine",
      {{"function", function_name(op)},
       {"batch", text(input->dims[0])},
       {"channels", text(channels)},
       {"channels_per_group", text(channels_per_group)},
       {"inner", text(inner)},
       {"elements", text(channels_per_group * inner)},
       {"epsilon", epsilon_literal.str()},
       {"reduction", group_reduction_fragment(block)}});
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_adamw_update(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto *parameter = program.tensor(op.inputs[0]);
  const auto *gradient = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto learning_rate =
      static_cast<float>(op.f64(ir::AttrKey::LearningRate, 1.0e-3));
  const auto beta1 = static_cast<float>(op.f64(ir::AttrKey::Beta1, 0.9));
  const auto beta2 = static_cast<float>(op.f64(ir::AttrKey::Beta2, 0.999));
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-8));
  const auto weight_decay =
      static_cast<float>(op.f64(ir::AttrKey::WeightDecay, 0.0));
  const auto clip_scale =
      static_cast<float>(op.f64(ir::AttrKey::ClipScale, 1.0));
  // Parameter and gradient storage are typed independently (verifier admits
  // F32/BF16 for each); moments are F32 always.  Every intermediate is an
  // F32 register; the decoupled decay multiplies the parameter BEFORE the
  // moment-based update is subtracted, and weight decay is never folded
  // into the gradient ahead of the moment updates (flame's measured LoRA-A
  // "unlearning" runaway).
  const auto literal = [](float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(9) << value;
    return text.str();
  };
  out << render_kernel_template(
      "adamw_update",
      {{"function", function_name(op)},
       {"parameter_scalar", typed_scalar(parameter->dtype)},
       {"gradient_scalar", typed_scalar(gradient->dtype)},
       {"count", std::to_string(count)},
       {"beta1", literal(beta1)},
       {"beta2", literal(beta2)},
       {"gradient_load", typed_load(gradient->dtype)},
       {"clip_scale", literal(clip_scale)},
       {"parameter_load", typed_load(parameter->dtype)},
       {"learning_rate", literal(learning_rate)},
       {"weight_decay", literal(weight_decay)},
       {"epsilon", literal(epsilon)},
       {"parameter_store", typed_store(parameter->dtype)}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_fill(std::ostringstream &out, const ir::Program &program,
               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  std::ostringstream value_literal;
  value_literal << std::scientific << std::setprecision(9) << value;
  out << render_kernel_template(
      "fill", {{"function", function_name(op)},
               {"count", std::to_string(count)},
               {"value", value_literal.str()}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_gather_rows(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto input_rows = input->dims[0];
  const auto row_width = input->element_count() / input_rows;
  out << render_kernel_template(
      "gather_rows", {{"function", function_name(op)},
                      {"count", std::to_string(count)},
                      {"row_width", std::to_string(row_width)},
                      {"input_rows", std::to_string(input_rows)}});
}

void emit_indexed_update_rows(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *base = program.tensor(op.inputs[0]);
  const auto *updates = program.tensor(op.inputs[1]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto rows = base->dims[0];
  const auto update_rows = updates->dims[0];
  const auto row_width = base->element_count() / rows;
  out << render_kernel_template(
      "indexed_update_rows", {{"function", function_name(op)},
                              {"count", std::to_string(count)},
                              {"row_width", std::to_string(row_width)},
                              {"update_rows", std::to_string(update_rows)}});
}

void emit_select_row_chunks(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *values = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto rows = values->dims[0];
  const auto width = output->dims[1];
  const auto source_width = values->dims[1];
  std::string outputs, chunks;
  for (std::size_t chunk = 0; chunk < op.outputs.size(); ++chunk) {
    const auto o = "o" + std::to_string(chunk);
    outputs += ", dif_scalar* " + o;
    chunks += "    if (source >= 0 && source < " + std::to_string(rows) +
              ") dif_store(" + o + ", i, dif_load(values, (unsigned long long)source * " +
              std::to_string(source_width) + "ULL + " +
              std::to_string(chunk * width) + "ULL + col));\n    else dif_store(" +
              o + ", i, __int_as_float(0x7fffffff));\n";
  }
  out << render_kernel_template(
      "select_row_chunks", {{"function", function_name(op)},
                            {"outputs", outputs},
                            {"count", std::to_string(count)},
                            {"width", std::to_string(width)},
                            {"chunks", chunks}});
}

void emit_sinusoidal_timestep(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto width = output->dims[1];
  const auto half = width / 2U;
  const auto flip = op.boolean(ir::AttrKey::FlipSinToCos, false);
  const auto shift = static_cast<float>(
      op.f64(ir::AttrKey::DownscaleFreqShift, 1.0));
  const auto scale = static_cast<float>(op.f64(ir::AttrKey::Scale, 1.0));
  const auto max_period =
      static_cast<float>(op.f64(ir::AttrKey::MaxPeriod, 10000.0));
  const auto log_period = static_cast<float>(std::log(max_period));
  const auto denominator = static_cast<float>(half) - shift;
  const auto literal = [](float value) {
    std::ostringstream text;
    text << std::scientific << std::setprecision(9) << value;
    return text.str();
  };
  const auto half_text = std::to_string(half);
  out << render_kernel_template(
      "sinusoidal_timestep",
      {{"function", function_name(op)},
       {"count", std::to_string(count)},
       {"width", std::to_string(width)},
       {"paired", std::to_string(2U * half)},
       {"half", half_text},
       {"log_period", literal(log_period)},
       {"denominator", literal(denominator)},
       {"scale", literal(scale)},
       {"select", flip ? "(column < " + half_text + "ULL ? c : s)"
                       : "(column < " + half_text + "ULL ? s : c)"}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_rotary_position(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto *positions = program.tensor(op.inputs[0]);
  const auto *inv_freq = program.tensor(op.inputs[1]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto count = output->element_count();
  const auto axes = positions->dims[1];
  const auto frequencies = inv_freq->dims[0];
  const auto unrepeated_width = axes * frequencies;
  const auto width = 2U * unrepeated_width;
  const auto output_type = output->dtype == ir::DType::F32
                               ? "dif_f32"
                               : output->dtype == ir::DType::BF16 ? "dif_bf16"
                                                                  : "dif_f16";
  const auto store = output->dtype == ir::DType::F32
                         ? "dif_store_f32"
                         : output->dtype == ir::DType::BF16 ? "dif_store_bf16"
                                                            : "dif_store_f16";
  out << render_kernel_template(
      "rotary_position",
      {{"function", function_name(op)},
       {"output_type", output_type},
       {"count", std::to_string(count)},
       {"width", std::to_string(width)},
       {"unrepeated_width", std::to_string(unrepeated_width)},
       {"frequencies", std::to_string(frequencies)},
       {"axes", std::to_string(axes)},
       {"store", store}});
}

void emit_linear_blend(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "linear_blend", {{"function", function_name(op)},
                       {"count", std::to_string(count)}});
}

void emit_flow_euler_step(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto step = op.u64(ir::AttrKey::StepIndex, 0U);
  out << render_kernel_template(
      "flow_euler_step", {{"function", function_name(op)},
                          {"count", std::to_string(count)},
                          {"step", std::to_string(step)},
                          {"next_step", std::to_string(step + 1U)}});
}

void emit_euler_velocity_step(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  out << render_kernel_template(
      "euler_velocity_step", {{"function", function_name(op)},
                              {"count", std::to_string(count)}});
}

void emit_permute(std::ostringstream &out, const ir::Program &program,
                  const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *output = program.tensor(op.outputs[0]);
  const auto rank = input->dims.size();
  std::vector<std::uint64_t> input_strides(rank, 1U);
  for (std::size_t axis = rank - 1U; axis > 0U; --axis)
    input_strides[axis - 1U] = input_strides[axis] * input->dims[axis];
  std::string axes;
  for (std::size_t reverse = rank; reverse-- > 0U;) {
    const auto key = static_cast<ir::AttrKey>(
        static_cast<std::uint32_t>(ir::AttrKey::Permutation0) + reverse);
    const auto input_axis = op.u64(key, 0U);
    axes += "    coordinate = remaining % " +
            std::to_string(output->dims[reverse]) + "ULL; remaining /= " +
            std::to_string(output->dims[reverse]) + "ULL; source += coordinate * " +
            std::to_string(input_strides[input_axis]) + "ULL;\n";
  }
  out << render_kernel_template(
      "permute", {{"function", function_name(op)},
                  {"count", std::to_string(output->element_count())},
                  {"axes", axes}});
}

void emit_concat(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto *output = program.tensor(op.outputs[0]);
  const auto axis = static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  std::uint64_t inner = 1U;
  for (std::size_t dimension = axis + 1U; dimension < output->dims.size();
       ++dimension)
    inner *= output->dims[dimension];
  std::string parameters;
  for (std::size_t input = 0U; input < op.inputs.size(); ++input)
    parameters += "const dif_scalar* input" + std::to_string(input) + ", ";
  parameters += "dif_scalar* output";
  std::string selection;
  std::uint64_t offset = 0U;
  for (std::size_t input = 0U; input < op.inputs.size(); ++input) {
    const auto input_axis = program.tensor(op.inputs[input])->dims[axis];
    selection += std::string(input == 0U ? "    if" : "    else if") +
                 " (axis_coordinate < " + std::to_string(offset + input_axis) +
                 "ULL) {\n      source = (outer * " + std::to_string(input_axis) +
                 "ULL + (axis_coordinate - " + std::to_string(offset) + "ULL)) * " +
                 std::to_string(inner) + "ULL + inner_index;\n      dif_store(output, i, dif_load(input" +
                 std::to_string(input) + ", source));\n    }\n";
    offset += input_axis;
  }
  out << render_kernel_template(
      "concat", {{"function", function_name(op)},
                 {"parameters", parameters},
                 {"count", std::to_string(output->element_count())},
                 {"inner", std::to_string(inner)},
                 {"axis_extent", std::to_string(output->dims[axis])},
                 {"selection", selection}});
}

void emit_patchify_3d(std::ostringstream &out, const ir::Program &program,
                      const ir::Operation &op, bool inverse) {
  const auto *volume =
      program.tensor(inverse ? op.outputs[0] : op.inputs[0]);
  const auto *rows = program.tensor(inverse ? op.inputs[0] : op.outputs[0]);
  const auto patch_t = op.u64(ir::AttrKey::PatchT, 0U);
  const auto patch_h = op.u64(ir::AttrKey::PatchH, 0U);
  const auto patch_w = op.u64(ir::AttrKey::PatchW, 0U);
  const auto channels = volume->dims[1];
  const auto frames = volume->dims[2];
  const auto height = volume->dims[3];
  const auto width = volume->dims[4];
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  out << render_kernel_template(
      "patchify_3d",
      {{"function", function_name(op)},
       {"count", text(rows->element_count())},
       {"row_width", text(rows->dims[1])},
       {"output_width", text(width / patch_w)},
       {"output_height", text(height / patch_h)},
       {"output_frames", text(frames / patch_t)},
       {"patch_w", text(patch_w)}, {"patch_h", text(patch_h)},
       {"patch_t", text(patch_t)},
       {"channels", text(channels)}, {"frames", text(frames)},
       {"height", text(height)}, {"width", text(width)},
       {"transfer",
        inverse ? "dif_store(output, volume_index, dif_load(input, i));"
                : "dif_store(output, i, dif_load(input, volume_index));"}});
}

void emit_rms_norm(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto weight_offset = op.f64(ir::AttrKey::WeightOffset, 0.0);
  std::ostringstream weight_offset_literal;
  weight_offset_literal << std::scientific << std::setprecision(9)
                        << static_cast<float>(weight_offset);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto reduction_tile = op.u64(ir::AttrKey::ReductionTileSize, 0U);
  // Float literals print exactly as the historical std::setprecision(17)
  // stream did, so the generated text (and the PTX cache key) is unchanged.
  std::ostringstream epsilon_literal;
  epsilon_literal << std::setprecision(17) << static_cast<float>(epsilon);
  if (op.u64(ir::AttrKey::Implementation, 1U) == 2U) {
    out << render_kernel_template(
        "rms_norm_welford128", {{"function", function_name(op)},
                                {"rows", std::to_string(rows)},
                                {"epsilon", epsilon_literal.str()}});
    return;
  }
  const auto triton_blocked_reduction =
      block == 512U && columns == block * 12U && reduction_tile == 8192U;
  const auto triton_chunked_reduction =
      block == 512U && columns == block * 12U && reduction_tile == 2048U;
  const auto triton_per_row_reduction = block == 128U && columns == 128U;
  std::string reduction;
  if (triton_blocked_reduction) {
    reduction = render_kernel_template(
        "rms_norm_reduce_blocked", {{"columns", std::to_string(columns)}});
  } else if (triton_chunked_reduction) {
    reduction = render_kernel_template(
        "rms_norm_reduce_chunked", {{"columns", std::to_string(columns)}});
  } else if (triton_per_row_reduction) {
    reduction = render_kernel_template("rms_norm_reduce_per_row", {});
  } else if (columns % 4U == 0U && block >= 128U && block < 512U) {
    reduction = render_kernel_template(
        "rms_norm_reduce_packed4", {{"packs", std::to_string(columns / 4U)},
                                    {"columns", std::to_string(columns)}});
  } else {
    reduction = render_kernel_template(
        "rms_norm_reduce_generic", {{"columns", std::to_string(columns)}});
  }
  const auto triton_inverse = triton_blocked_reduction ||
                              triton_chunked_reduction ||
                              triton_per_row_reduction;
  const std::string inverse =
      triton_inverse
          ? "  float mean, mean_eps, inv;\n"
            "  asm volatile(\"div.full.f32 %0,%1,%2;\" : \"=f\"(mean) : \"f\"(reduction[0]), \"f\"(" +
                std::to_string(columns) + ".0f));\n  mean_eps = mean + " +
                epsilon_literal.str() +
                "f;\n  asm volatile(\"rsqrt.approx.ftz.f32 %0,%1;\" : \"=f\"(inv) : \"f\"(mean_eps));"
          : "  float inv = rsqrtf(reduction[0] / " + std::to_string(columns) +
                ".0f + " + epsilon_literal.str() + "f);";
  out << render_kernel_template(
      "rms_norm", {{"function", function_name(op)},
                   {"rows", std::to_string(rows)},
                   {"columns", std::to_string(columns)},
                   {"reduction", reduction},
                   {"inverse", inverse},
                   {"weight_offset", weight_offset_literal.str()}});
}

void emit_layer_norm(std::ostringstream &out, const ir::Program &program,
                     const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  std::ostringstream epsilon_literal;
  epsilon_literal << std::setprecision(17) << static_cast<float>(epsilon);
  if (input->dtype == ir::DType::BF16 && columns % 4U == 0U &&
      block == 128U) {
    out << render_kernel_template(
        "layer_norm_welford", {{"function", function_name(op)},
                               {"suffix", std::to_string(op.id)},
                               {"rows", std::to_string(rows)},
                               {"packs", std::to_string(columns / 4U)},
                               {"columns", std::to_string(columns)},
                               {"epsilon", epsilon_literal.str()}});
  } else {
    out << render_kernel_template(
        "layer_norm", {{"function", function_name(op)},
                       {"rows", std::to_string(rows)},
                       {"columns", std::to_string(columns)},
                       {"epsilon", epsilon_literal.str()}});
  }
  // The historical emitter left the stream at precision 17.
  out << std::setprecision(17);
}

void emit_layer_norm_modulate(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[0]);
  const auto *scale = program.tensor(op.inputs[3]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto modulation_rows = scale->element_count() / columns;
  const auto rows_per_modulation = rows / modulation_rows;
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  std::ostringstream epsilon_literal;
  epsilon_literal << std::setprecision(17) << static_cast<float>(epsilon);
  // The historical emitter left the stream at precision 17.
  out << std::setprecision(17);
  if (input->dtype == ir::DType::BF16 && columns % 4U == 0U &&
      block == 128U) {
    out << render_kernel_template(
        "layer_norm_modulate_welford128",
        {{"function", function_name(op)},
         {"suffix", std::to_string(op.id)},
         {"rows", text(rows)},
         {"columns", text(columns)},
         {"packs", text(columns / 4U)},
         {"epsilon", epsilon_literal.str()},
         {"rows_per_modulation", text(rows_per_modulation)}});
    return;
  }
  out << render_kernel_template(
      "layer_norm_modulate",
      {{"function", function_name(op)},
       {"rows", text(rows)},
       {"columns", text(columns)},
       {"epsilon", epsilon_literal.str()},
       {"rows_per_modulation", text(rows_per_modulation)}});
}

void emit_rms_norm_modulate(std::ostringstream &out, const ir::Program &program,
                            const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto rows = shape[0];
  const auto cols = shape[1];
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto layout = static_cast<ir::ModulationLayout>(op.u64(
      ir::AttrKey::ModulationLayout,
      static_cast<std::uint64_t>(ir::ModulationLayout::ExplicitScaleShift)));
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  std::ostringstream epsilon_literal;
  epsilon_literal << std::setprecision(17) << static_cast<float>(epsilon);
  if (layout == ir::ModulationLayout::SharedVectorDelta) {
    const auto vectors = program.tensor(op.inputs[2])->dims[0];
    const auto rows_per_vector = rows / vectors;
    const auto weight_offset = op.f64(ir::AttrKey::WeightOffset, 0.0);
    std::ostringstream weight_offset_literal;
    weight_offset_literal << std::scientific << std::setprecision(9)
                          << static_cast<float>(weight_offset);
    const auto reduction_tile =
        op.u64(ir::AttrKey::ReductionTileSize, 0U);
    if (program.tensor(op.inputs[0])->dtype != ir::DType::BF16 ||
        op.u64(ir::AttrKey::BlockSize, 256U) != 512U || cols != 6144U ||
        (reduction_tile != 2048U && reduction_tile != 8192U))
      fail("CUDA shared-vector rms_norm_modulate currently requires the "
           "source-faithful BF16 6144-wide reduction");
    if (reduction_tile == 2048U) {
      out << render_kernel_template(
          "rms_norm_modulate_shared_chunked",
          {{"function", function_name(op)},
           {"rows", text(rows)},
           {"epsilon", epsilon_literal.str()},
           {"rows_per_vector", text(rows_per_vector)},
           {"weight_offset", weight_offset_literal.str()}});
    } else {
      out << render_kernel_template(
          "rms_norm_modulate_shared_blocked",
          {{"function", function_name(op)},
           {"rows", text(rows)},
           {"cols", text(cols)},
           {"epsilon", epsilon_literal.str()},
           {"rows_per_vector", text(rows_per_vector)},
           {"weight_offset", weight_offset_literal.str()}});
    }
    // The historical emitter left the stream at precision 17.
    out << std::setprecision(17);
    return;
  }
  const bool weighted = op.inputs.size() == 4;
  const auto block = op.u64(ir::AttrKey::BlockSize, 256U);
  const auto *scale = program.tensor(op.inputs[weighted ? 2U : 1U]);
  if (weighted && program.tensor(op.inputs[0])->dtype == ir::DType::BF16 &&
      block == 256U && scale->element_count() % cols == 0U &&
      rows % (scale->element_count() / cols) == 0U) {
    const auto rows_per_vector = rows / (scale->element_count() / cols);
    // Port of Serenity's accepted fused BF16 RMSNorm + AdaLN modulation.
    // Preserve the BF16 norm output boundary, then perform modulation in F32
    // with only the final BF16 store.
    out << render_kernel_template(
        "rms_norm_modulate_fused_bf16",
        {{"function", function_name(op)},
         {"rows", text(rows)},
         {"cols", text(cols)},
         {"epsilon", epsilon_literal.str()},
         {"rows_per_vector", text(rows_per_vector)}});
    out << std::setprecision(17);
    return;
  }
  const std::string reduction =
      (cols % 4U == 0U && block >= 128U)
          ? render_kernel_template("rms_norm_modulate_reduce_packed4",
                                   {{"packs", text(cols / 4U)}, {"cols", text(cols)}})
          : render_kernel_template("rms_norm_modulate_reduce_generic",
                                   {{"cols", text(cols)}});
  out << render_kernel_template(
      "rms_norm_modulate",
      {{"function", function_name(op)},
       {"parameters",
        weighted ? "const dif_scalar* x, const dif_scalar* weight, const dif_scalar* scale, "
                   "const dif_scalar* shift, dif_scalar* y"
                 : "const dif_scalar* x, const dif_scalar* scale, const dif_scalar* shift, "
                   "dif_scalar* y"},
       {"rows", text(rows)},
       {"reduction", reduction},
       {"cols", text(cols)},
       {"epsilon", epsilon_literal.str()},
       {"weight_factor", weighted ? " * dif_load(weight, col)" : ""}});
  out << std::setprecision(17);
}

void emit_swiglu(std::ostringstream &out, const ir::Program &program,
                 const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.outputs[0])->dims.back();
  const auto input_width = program.tensor(op.inputs[0])->dims.back();
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  out << render_kernel_template(
      "swiglu",
      {{"function", function_name(op)},
       {"count", std::to_string(count)},
       {"width", std::to_string(width)},
       {"input_width", std::to_string(input_width)},
       {"value_offset", std::to_string(start + (gate_first ? width : 0U))},
       {"gate_offset", std::to_string(start + (gate_first ? 0U : width))}});
}

void emit_bias_add(std::ostringstream &out, const ir::Program &program,
                   const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[1])->dims[0];
  out << render_kernel_template(
      "bias_add", {{"function", function_name(op)},
                   {"count", std::to_string(count)},
                   {"width", std::to_string(width)}});
}

void emit_linear_addmm_prefill(std::ostringstream &out,
                               const ir::Program &program,
                               const ir::Operation &op) {
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto width = program.tensor(op.inputs[2])->element_count();
  out << render_kernel_template(
      "linear_addmm_prefill", {{"function", function_name(op)},
                               {"count", std::to_string(count)},
                               {"width", std::to_string(width)}});
}

void emit_h3_adaln_select(std::ostringstream &out, const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto hidden = shape[1];
  out << render_kernel_template(
      "h3_adaln_select", {{"function", function_name(op)},
                          {"count", std::to_string(count)},
                          {"hidden", std::to_string(hidden)}});
}

void emit_h3_deinterleave_qkv(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto heads = shape[1];
  const auto dim = shape[2];
  const auto packed = 3U * heads * dim;
  out << render_kernel_template(
      "h3_deinterleave_qkv", {{"function", function_name(op)},
                              {"count", std::to_string(count)},
                              {"head_width", std::to_string(heads * dim)},
                              {"dim", std::to_string(dim)},
                              {"packed", std::to_string(packed)},
                              {"triple_dim", std::to_string(3U * dim)},
                              {"double_dim", std::to_string(2U * dim)}});
}

void emit_h3_deinterleave_qkv_weight(std::ostringstream &out,
                                     const ir::Program &program,
                                     const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto dim = op.u64(ir::AttrKey::HeadDim, 0U);
  const auto hidden = shape[1];
  out << render_kernel_template(
      "h3_deinterleave_qkv_weight",
      {{"function", function_name(op)},
       {"count", std::to_string(count)},
       {"hidden", std::to_string(hidden)},
       {"dim", std::to_string(dim)},
       {"dim_hidden", std::to_string(dim * hidden)},
       {"double_dim_hidden", std::to_string(2U * dim * hidden)}});
}

void emit_dequantize_int4(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto rows = shape[0];
  const auto columns = shape[1];
  const auto groups = columns / op.u64(ir::AttrKey::GroupSize, 64U);
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto count = rows * columns;
  const bool outliers = op.inputs.size() == 4U;
  const std::string parameters =
      outliers ? "const unsigned char* packed, const dif_scalar* scales, "
                 "const unsigned char* outlier_indices, "
                 "const dif_scalar* outlier_residuals, dif_scalar* y"
               : "const unsigned char* packed, const dif_scalar* scales, "
                 "dif_scalar* y";
  const std::string outlier =
      outliers ? "unsigned long long gi = row * " + std::to_string(groups) +
                     "ULL + col / " + std::to_string(group) + "ULL;\n"
                     "    if (outlier_indices[gi] == col % " +
                     std::to_string(group) +
                     "ULL)\n      value += dif_load(outlier_residuals, gi);"
               : "";
  out << render_kernel_template(
      "dequantize_int4",
      {{"function", function_name(op)},
       {"parameters", parameters},
       {"count", std::to_string(count)},
       {"columns", std::to_string(columns)},
       {"packed_columns", std::to_string(columns / 2U)},
       {"groups", std::to_string(groups)},
       {"group", std::to_string(group)},
       {"outlier", outlier}});
}

void emit_dequantize_int5(std::ostringstream &out,
                          const ir::Program &program,
                          const ir::Operation &op) {
  const auto &shape = program.tensor(op.outputs[0])->dims;
  const auto rows = shape[0];
  const auto columns = shape[1];
  const auto row_bytes = columns * 5U / 8U;
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = columns / group;
  const auto count = rows * columns;
  const bool column_scaled = op.inputs.size() == 3U;
  const std::string parameters =
      column_scaled ? "const unsigned char* packed, const dif_scalar* scales, "
                      "const dif_scalar* column_scales, dif_scalar* y"
                    : "const unsigned char* packed, const dif_scalar* scales, "
                      "dif_scalar* y";
  out << render_kernel_template(
      "dequantize_int5",
      {{"function", function_name(op)},
       {"parameters", parameters},
       {"count", std::to_string(count)},
       {"columns", std::to_string(columns)},
       {"row_bytes", std::to_string(row_bytes)},
       {"groups", std::to_string(groups)},
       {"group", std::to_string(group)},
       {"column_scale",
        column_scaled ? "value *= dif_load(column_scales, col);" : ""}});
}

void emit_residual_gate(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto *output = program.tensor(op.outputs[0]);
  const auto *gate = program.tensor(op.inputs[2]);
  const auto count = output->element_count();
  const auto width = output->dims.back();
  const auto rows = count / width;
  const auto gate_rows = gate->element_count() / width;
  const auto rows_per_gate = rows / gate_rows;
  out << render_kernel_template(
      "residual_gate", {{"function", function_name(op)},
                        {"count", std::to_string(count)},
                        {"width", std::to_string(width)},
                        {"rows_per_gate", std::to_string(rows_per_gate)}});
}

void emit_qk_norm_rope(std::ostringstream &out, const ir::Program &program,
                       const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  const auto heads = shape[shape.size() - 2U];
  const auto dim = shape.back();
  const auto sequence =
      program.tensor(op.inputs[0])->element_count() / (heads * dim);
  const auto input_sequence =
      shape.size() == 4U ? shape[1] : shape[0];
  const auto &table_shape = program.tensor(op.inputs[2])->dims;
  const auto table_sequence =
      table_shape.size() == 3U ? table_shape[1] : table_shape[0];
  const auto table_start = op.u64(ir::AttrKey::Start, 0U);
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = program.tensor(op.inputs[2])->dims.back();
  const auto epsilon = op.f64(ir::AttrKey::Epsilon, 1.0e-5);
  const auto implementation = op.u64(ir::AttrKey::Implementation, 1U);
  const auto rotary_layout = static_cast<ir::RotaryLayout>(op.u64(
      ir::AttrKey::RotaryLayout,
      static_cast<std::uint64_t>(ir::RotaryLayout::HalfSplit)));
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  std::ostringstream epsilon_literal;
  epsilon_literal << std::setprecision(17) << static_cast<float>(epsilon);
  if (implementation == 2U && rotary_layout == ir::RotaryLayout::Interleaved &&
      program.tensor(op.inputs[0])->dtype == ir::DType::BF16 && dim == 128U &&
      program.tensor(op.inputs[2])->dtype == ir::DType::F32 &&
      table_width * 2U == rotary) {
    out << render_kernel_template(
        "qk_norm_rope_interleaved128",
        {{"function", function_name(op)},
         {"rows", text(sequence * heads)},
         {"epsilon", epsilon_literal.str()},
         {"heads", text(heads)},
         {"input_sequence", text(input_sequence)},
         {"table_sequence", text(table_sequence)},
         {"table_start", text(table_start)},
         {"table_width", text(table_width)}});
    out << std::setprecision(17);
    return;
  }
  if (program.tensor(op.inputs[0])->dtype == ir::DType::BF16 && dim == 128U &&
      table_width == rotary) {
    // Port of Serenity's accepted MiniMax-H3 fused Q/K RMSNorm + partial-RoPE
    // primitive. One lane owns one head value, preserving the 128-lane F32
    // reduction and the required BF16 normalization boundary before RoPE.
    out << render_kernel_template(
        "qk_norm_rope_fused128",
        {{"function", function_name(op)},
         {"rows", text(sequence * heads)},
         {"epsilon", epsilon_literal.str()},
         {"heads", text(heads)},
         {"input_sequence", text(input_sequence)},
         {"table_sequence", text(table_sequence)},
         {"table_start", text(table_start)},
         {"half", text(half)},
         {"table_width", text(table_width)},
         {"rotary", text(rotary)}});
    out << std::setprecision(17);
    return;
  }
  const std::string reduction =
      (dim <= 128U && dim % 4U == 0U)
          ? render_kernel_template("qk_norm_rope_reduce_warp4", {{"dim", text(dim)}})
          : render_kernel_template("qk_norm_rope_reduce_generic", {{"dim", text(dim)}});
  // The tables may be F32 under BF16 q/k, and may be longer than the rows
  // being rotated with this operation reading from an offset into them. A
  // table that lines up one-for-one keeps the bare "s * width" this kernel
  // always emitted.
  const std::string table_load = typed_load(program.tensor(op.inputs[2])->dtype);
  const std::string table_base =
      (table_sequence == input_sequence && table_start == 0U)
          ? "s * " + text(table_width) + "ULL"
          : "(s / " + text(input_sequence) + "ULL * " + text(table_sequence) +
                "ULL + " + text(table_start) + "ULL + s % " +
                text(input_sequence) + "ULL) * " + text(table_width) + "ULL";
  // Interleaved rotation pairs adjacent lanes; the half split pairs a lane
  // with the one half a head away. Only the generic path needs generating --
  // the two specialized kernels above are each written for one layout.
  const std::string rotation =
      rotary_layout == ir::RotaryLayout::Interleaved
          ? "if (d < " + text(rotary) +
                "ULL) {\n"
                "      unsigned long long p = d / 2ULL, partner = (d % 2ULL == 0ULL) ? d + 1ULL : d - 1ULL;\n"
                "      float other = dif_round(dif_load(x, base + partner) * inv * dif_load(weight, partner));\n"
                "      float c = " + table_load + "(cosv, tb + p);\n"
                "      float sn = " + table_load + "(sinv, tb + p);\n"
                "      result = (d % 2ULL == 0ULL) ? (value * c - other * sn) : (other * sn + value * c);\n"
                "    }"
          : "if (d < " + text(half) +
                "ULL) {\n"
                "      float other = dif_round(dif_load(x, base + d + " + text(half) +
                "ULL) * inv * dif_load(weight, d + " + text(half) + "ULL));\n"
                "      float left = dif_round(value * " + table_load + "(cosv, tb + d));\n"
                "      float right = dif_round(other * " + table_load + "(sinv, tb + d));\n"
                "      result = dif_round(left - right);\n"
                "    } else if (d < " + text(rotary) +
                "ULL) {\n"
                "      unsigned long long r = d - " + text(half) + "ULL;\n"
                "      float other = dif_round(dif_load(x, base + r) * inv * dif_load(weight, r));\n"
                "      unsigned long long ti = " +
                (table_width == rotary ? std::string("d") : std::string("r")) +
                ";\n"
                "      float left = dif_round(value * " + table_load + "(cosv, tb + ti));\n"
                "      float right = dif_round(other * " + table_load + "(sinv, tb + ti));\n"
                "      result = dif_round(left + right);\n"
                "    }";
  out << render_kernel_template(
      "qk_norm_rope",
      {{"function", function_name(op)},
       {"rows", text(sequence * heads)},
       {"heads", text(heads)},
       {"dim", text(dim)},
       {"reduction", reduction},
       {"epsilon", epsilon_literal.str()},
       {"table_base", table_base},
       {"table_scalar", typed_scalar(program.tensor(op.inputs[2])->dtype)},
       {"rotation", rotation}});
  out << std::setprecision(17);
}

void emit_attention(std::ostringstream &out, const ir::Program &program,
                    const ir::Operation &op) {
  const auto &shape = program.tensor(op.inputs[0])->dims;
  // Batched attention carries [B,S,H,D]; the historical form is [S,H,D].
  const bool batched = shape.size() == 4U;
  const auto batch = batched ? shape[0] : 1U;
  const auto sequence = shape[batched ? 1U : 0U];
  const auto heads = shape[batched ? 2U : 1U];
  const auto dim = shape[batched ? 3U : 2U];
  // Cross-attention keys carry their own row count; a square program keeps
  // its historical generated source (kend is the query row count).
  const auto kv_sequence =
      program.tensor(op.inputs[1])->dims[batched ? 1U : 0U];
  // GQA (KvHeads attr): query head h reads kv head h/(H/KvHeads).  When
  // KvHeads == H the emitted source is BYTE-IDENTICAL to the pre-GQA kernel
  // (kv_head_expr collapses to "h"), so recorded programs keep their
  // generated-source identity.
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  const std::string kv_head_expr =
      group == 1U ? std::string("h")
                  : "h / " + std::to_string(group) + "ULL";
  const auto scale = op.f64(ir::AttrKey::AttentionScale,
                            1.0 / std::sqrt(static_cast<double>(dim)));
  const auto causal = op.boolean(ir::AttrKey::Causal, false);
  std::ostringstream scale_literal;
  scale_literal << std::setprecision(17) << static_cast<float>(scale);
  // The block's query row is (b*S + s), so the key row it reads is
  // (b*Skv + ks); an unbatched program keeps the bare "ks" it always had.
  const std::string key_row =
      batched ? "(qs / " + std::to_string(sequence) + "ULL * " +
                    std::to_string(kv_sequence) + "ULL + ks)"
              : std::string("ks");
  out << render_kernel_template(
      "attention_exact",
      {{"function", function_name(op)},
       {"items", std::to_string(batch * sequence * heads)},
       {"heads", std::to_string(heads)},
       {"kend",
        causal ? (batched ? "qs % " + std::to_string(sequence) + "ULL + 1ULL"
                          : std::string("qs + 1ULL"))
               : std::to_string(kv_sequence) + "ULL"},
       {"dim", std::to_string(dim)},
       {"key_base", "(" + key_row + " * " + std::to_string(kv_heads) +
                        "ULL + " + kv_head_expr + ") * " +
                        std::to_string(dim) + "ULL"},
       {"scale", scale_literal.str()}});
  // The historical emitter left the stream at precision 17.
  out << std::setprecision(17);
}

void emit_rms_norm_backward(std::ostringstream &out,
                            const ir::Program &program,
                            const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const bool weight_grad = op.outputs.size() == 2U;
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  const std::string weight_gradient =
      weight_grad ? render_kernel_template(
                        "rms_norm_backward_weight",
                        {{"columns", std::to_string(columns)},
                         {"rows", std::to_string(rows)},
                         {"epsilon", epsilon_literal.str()}})
                  : std::string{};
  out << render_kernel_template(
      "rms_norm_backward",
      {{"function", function_name(op)},
       {"weight_parameter", weight_grad ? ", dif_scalar* grad_weight" : ""},
       {"count", std::to_string(count)},
       {"columns", std::to_string(columns)},
       {"epsilon", epsilon_literal.str()},
       {"weight_gradient", weight_gradient}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_rms_norm_modulate_backward(std::ostringstream &out,
                                     const ir::Program &program,
                                     const ir::Operation &op) {
  const bool weighted = op.inputs.size() == 4U;
  const auto *x = program.tensor(op.inputs[1]);
  const auto rows = x->dims[0];
  const auto columns = x->dims[1];
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  const std::string weight_gradient =
      weighted ? render_kernel_template(
                     "rms_norm_modulate_backward_weight",
                     {{"columns", std::to_string(columns)},
                      {"rows", std::to_string(rows)},
                      {"epsilon", epsilon_literal.str()}})
               : std::string{};
  out << render_kernel_template(
      "rms_norm_modulate_backward",
      {{"function", function_name(op)},
       {"parameters",
        weighted ? "const dif_scalar* grad_output, const dif_scalar* x, "
                   "const dif_scalar* weight, const dif_scalar* scale, "
                   "dif_scalar* grad_input, dif_scalar* grad_scale, "
                   "dif_scalar* grad_shift, dif_scalar* grad_weight"
                 : "const dif_scalar* grad_output, const dif_scalar* x, "
                   "const dif_scalar* scale, dif_scalar* grad_input, "
                   "dif_scalar* grad_scale, dif_scalar* grad_shift"},
       {"count", std::to_string(count)},
       {"columns", std::to_string(columns)},
       {"epsilon", epsilon_literal.str()},
       {"dot_weight", weighted ? " * dif_load(weight, k)" : ""},
       {"weight_value", weighted ? "dif_load(weight, col);" : "1.0f;"},
       {"weight_gradient", weight_gradient}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_swiglu_backward(std::ostringstream &out, const ir::Program &program,
                          const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto width = grad_output->dims.back();
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto input_width = program.tensor(op.inputs[1])->dims.back();
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  out << render_kernel_template(
      "swiglu_backward",
      {{"function", function_name(op)},
       {"count", std::to_string(count)},
       {"input_width", std::to_string(input_width)},
       {"start", std::to_string(start)},
       {"window_end", std::to_string(start + width * 2U)},
       {"width", std::to_string(width)},
       {"value_offset", std::to_string(gate_first ? width : 0U)},
       {"gate_offset", std::to_string(gate_first ? 0U : width)},
       {"value_slot_test", gate_first ? ">=" : "<"}});
}

void emit_qk_norm_rope_backward(std::ostringstream &out,
                                const ir::Program &program,
                                const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  // The kernel indexes flat and derives its table row as row/heads, which is
  // already right for [B,S,H,D]; only the counts have to notice the batch.
  const auto dim = input->dims.back();
  const auto heads = input->dims[input->dims.size() - 2U];
  const auto count = input->element_count();
  const auto sequence = count / dim / heads;
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto *table = program.tensor(op.inputs[3]);
  const auto table_width = table->dims.back();
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const bool weight_grad = op.outputs.size() == 2U;
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  // The rotation tables are commonly F32 while q/k are BF16, and they may be
  // longer than the rows being rotated, with this operation reading from an
  // offset into them (an image stream whose tables follow the text stream).
  // Both facts come from the forward operation rather than from a default.
  const std::string table_load = typed_load(table->dtype);
  const auto &input_shape = input->dims;
  const auto input_sequence =
      input_shape.size() == 4U ? input_shape[1] : input_shape[0];
  const auto table_sequence =
      table->dims.size() == 3U ? table->dims[1] : table->dims[0];
  const auto table_start = op.u64(ir::AttrKey::Start, 0U);
  // A table that lines up one-for-one with the rows collapses to the bare
  // "token * width" this kernel always emitted.
  const std::string table_base =
      (table_sequence == input_sequence && table_start == 0U)
          ? "token * " + text(table_width) + "ULL"
          : "(token / " + text(input_sequence) + "ULL * " +
                text(table_sequence) + "ULL + " + text(table_start) +
                "ULL + token % " + text(input_sequence) + "ULL) * " +
                text(table_width) + "ULL";
  const auto rotary_layout = static_cast<ir::RotaryLayout>(op.u64(
      ir::AttrKey::RotaryLayout,
      static_cast<std::uint64_t>(ir::RotaryLayout::HalfSplit)));
  // rot(k): the rotation-transpose of the upstream gradient at offset k of
  // one head row (rb = row base, tb = table base), F32 registers.
  const auto rotated = [&](const std::string &row_base,
                           const std::string &table_base,
                           const std::string &k) {
    const auto half_text = text(half);
    if (rotary_layout == ir::RotaryLayout::Interleaved) {
      // Adjacent pairs rotate together, so the transpose pairs the same way:
      // an even lane takes its own cosine and its partner's sine, an odd lane
      // takes its partner's negated sine and its own cosine.  Both read the
      // pair's table entry, which integer division gives either way.
      const auto pair = table_base + " + " + k + " / 2ULL";
      return "(" + k + " < " + text(rotary) + "ULL ? ((" + k +
             " % 2ULL) == 0ULL ? dif_load(grad_output, " + row_base + " + " +
             k + ") * " + table_load + "(cosv, " + pair +
             ") + dif_load(grad_output, " + row_base + " + " + k +
             " + 1ULL) * " + table_load + "(sinv, " + pair +
             ") : dif_load(grad_output, " + row_base + " + " + k +
             ") * " + table_load + "(cosv, " + pair +
             ") - dif_load(grad_output, " + row_base + " + " + k +
             " - 1ULL) * " + table_load + "(sinv, " + pair +
             ")) : dif_load(grad_output, " + row_base + " + " + k + "))";
    }
    return "(" + k + " < " + half_text + "ULL ? dif_load(grad_output, " + row_base +
           " + " + k + ") * " + table_load + "(cosv, " + table_base + " + " + k +
           ") + dif_load(grad_output, " + row_base + " + " + k + " + " + half_text +
           "ULL) * " + table_load + "(sinv, " + table_base + " + " +
           (table_width == rotary ? k + " + " + half_text + "ULL" : k) +
           ") : (" + k + " < " + text(rotary) + "ULL ? -dif_load(grad_output, " +
           row_base + " + " + k + " - " + half_text + "ULL) * " + table_load +
           "(sinv, " + table_base + " + " + k + " - " + half_text +
           "ULL) + dif_load(grad_output, " +
           row_base + " + " + k + ") * " + table_load + "(cosv, " + table_base + " + " +
           (table_width == rotary ? k : k + " - " + half_text + "ULL") +
           ") : dif_load(grad_output, " + row_base + " + " + k + ")))";
  };
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  const std::string weight_gradient =
      weight_grad ? render_kernel_template(
                        "qk_norm_rope_backward_weight",
                        {{"dim", text(dim)},
                         {"rows", text(sequence * heads)},
                         {"heads", text(heads)},
                         {"table_base", table_base},
                         {"epsilon", epsilon_literal.str()},
                         {"rotated_i", rotated("rrb", "rtb", "i")}})
                  : std::string{};
  out << render_kernel_template(
      "qk_norm_rope_backward",
      {{"function", function_name(op)},
       {"weight_parameter", weight_grad ? ", dif_scalar* grad_weight" : ""},
       {"table_scalar", typed_scalar(table->dtype)},
       {"count", text(count)},
       {"dim", text(dim)},
       {"heads", text(heads)},
       {"table_base", table_base},
       {"epsilon", epsilon_literal.str()},
       {"rotated_k", rotated("rb", "tb", "k")},
       {"rotated_d", rotated("rb", "tb", "d")},
       {"weight_gradient", weight_gradient}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_layer_norm_backward(std::ostringstream &out,
                              const ir::Program &program,
                              const ir::Operation &op) {
  const auto *input = program.tensor(op.inputs[1]);
  const auto columns = input->dims.back();
  const auto rows = input->element_count() / columns;
  const auto count = rows * columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::ostringstream epsilon_literal;
  epsilon_literal << std::scientific << std::setprecision(9) << epsilon;
  out << render_kernel_template(
      "layer_norm_backward", {{"function", function_name(op)},
                              {"count", std::to_string(count)},
                              {"columns", std::to_string(columns)},
                              {"epsilon", epsilon_literal.str()},
                              {"rows", std::to_string(rows)}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_residual_gate_backward(std::ostringstream &out,
                                 const ir::Program &program,
                                 const ir::Operation &op) {
  const auto *grad_output = program.tensor(op.inputs[0]);
  const auto *gate = program.tensor(op.inputs[2]);
  const auto count = program.tensor(op.outputs[0])->element_count();
  const auto columns = grad_output->dims.back();
  const auto rows = grad_output->element_count() / columns;
  const auto gate_rows = gate->element_count() / columns;
  const auto rows_per_gate = rows / gate_rows;
  // An ungated broadcast is the historical case: the gate index is the
  // element index and the reduction is a single term, so the emitted text is
  // exactly what it always was.
  const std::string gate_index =
      rows_per_gate == 1U
          ? std::string("i")
          : "i / " + std::to_string(rows_per_gate * columns) + "ULL * " +
                std::to_string(columns) + "ULL + i % " +
                std::to_string(columns) + "ULL";
  const std::string reduction =
      rows_per_gate == 1U
          ? "accumulator = dif_load(grad_output, i) * dif_load(branch, i);"
          : "unsigned long long row = i / " + std::to_string(columns) +
                "ULL * " + std::to_string(rows_per_gate) + "ULL, column = i % " +
                std::to_string(columns) + "ULL;\n"
                "    for (unsigned long long r = row; r < row + " +
                std::to_string(rows_per_gate) + "ULL; ++r)\n"
                "      accumulator = fmaf(dif_load(grad_output, r * " +
                std::to_string(columns) +
                "ULL + column), dif_load(branch, r * " +
                std::to_string(columns) + "ULL + column), accumulator);";
  out << render_kernel_template(
      "residual_gate_backward",
      {{"function", function_name(op)},
       {"count", std::to_string(count)},
       {"gate_count", std::to_string(gate->element_count())},
       {"gate_index", gate_index},
       {"gate_reduction", reduction}});
}

void emit_attention_lse(std::ostringstream &out, const ir::Program &program,
                        const ir::Operation &op) {
  const auto *q = program.tensor(op.inputs[0]);
  // Batched attention carries [B,S,H,D]; the historical form is [S,H,D].
  const bool batched = q->dims.size() == 4U;
  const auto batch = batched ? q->dims[0] : 1U;
  const auto sequence = q->dims[batched ? 1U : 0U];
  const auto heads = q->dims[batched ? 2U : 1U];
  const auto dim = q->dims[batched ? 3U : 2U];
  const auto count = batch * sequence * heads;
  // Cross attention: the keys carry their own row count.
  const auto *k_tensor = program.tensor(op.inputs[1]);
  const auto kv_sequence = k_tensor->dims[batched ? 1U : 0U];
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  // Collapses to "h" when KvHeads == H, keeping the emitted source
  // byte-identical to the pre-GQA kernel for every existing program.
  const std::string kv_head_expr =
      group == 1U ? std::string("h")
                  : "h / " + std::to_string(group) + "ULL";
  // The key base gains the batch offset only when there is a batch, so an
  // unbatched program emits exactly the text it always did.
  const std::string key_row =
      batched ? "(qs / " + std::to_string(sequence) + "ULL * " +
                    std::to_string(kv_sequence) + "ULL + ks)"
              : std::string("ks");
  const std::string key_base = "(" + key_row + " * " +
                               std::to_string(kv_heads) + "ULL + " +
                               kv_head_expr + ") * " + std::to_string(dim) +
                               "ULL";
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  std::ostringstream scale_literal;
  scale_literal << std::scientific << std::setprecision(9) << scale;
  out << render_kernel_template(
      "attention_lse",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(q->dtype)},
       {"count", std::to_string(count)},
       {"heads", std::to_string(heads)},
       {"dim", std::to_string(dim)},
       {"kend", causal ? (batched ? "qs % " + std::to_string(sequence) +
                                        "ULL + 1ULL"
                                  : std::string("qs + 1ULL"))
                       : std::to_string(kv_sequence) + "ULL"},
       {"key_base", key_base},
       {"load", typed_load(q->dtype)},
       {"scale", scale_literal.str()}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

void emit_attention_backward(std::ostringstream &out,
                             const ir::Program &program,
                             const ir::Operation &op) {
  const auto *q = program.tensor(op.inputs[1]);
  const auto *k_tensor = program.tensor(op.inputs[2]);
  const bool batched = q->dims.size() == 4U;
  const auto batch = batched ? q->dims[0] : 1U;
  const auto sequence = q->dims[batched ? 1U : 0U];
  const auto heads = q->dims[batched ? 2U : 1U];
  const auto dim = q->dims[batched ? 3U : 2U];
  const auto kv_sequence = k_tensor->dims[batched ? 1U : 0U];
  const auto kv_heads = k_tensor->dims[batched ? 2U : 1U];
  const auto group = heads / kv_heads;
  const auto count = batch * sequence * heads * dim;
  const auto kv_count = batch * kv_sequence * kv_heads * dim;
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto text = [](std::uint64_t v) { return std::to_string(v); };
  std::ostringstream scale_literal;
  scale_literal << std::scientific << std::setprecision(9) << scale;
  // Every index is an expression so one template serves the unbatched
  // [S,H,D] form, the batched [B,S,H,D] form, and cross attention where the
  // keys carry their own row count.  The batch offset appears only when
  // there is a batch, so an unbatched program emits the simple form.
  // Query rows are numbered ((b*S + s)*H + h); key rows ((b*Skv + ks)*KV + kh).
  const std::string query_batch =
      "row / " + text(sequence * heads) + "ULL";
  const std::string key_batch =
      "krow / " + text(kv_sequence * kv_heads) + "ULL";
  const std::string s_expr =
      batched ? "row / " + text(heads) + "ULL % " + text(sequence) + "ULL"
              : "row / " + text(heads) + "ULL";
  const std::string ks_expr =
      batched
          ? "krow / " + text(kv_heads) + "ULL % " + text(kv_sequence) + "ULL"
          : "krow / " + text(kv_heads) + "ULL";
  // The dq branch reads keys, so it offsets by the QUERY thread's batch into
  // the key geometry; the dk/dv branch reads queries and does the reverse.
  const std::string key_row =
      batched ? "(" + query_batch + " * " + text(kv_sequence) + "ULL + ks)"
              : std::string("ks");
  const std::string query_row =
      batched ? "(" + key_batch + " * " + text(sequence) + "ULL + qs)"
              : std::string("qs");
  out << render_kernel_template(
      "attention_backward",
      {{"function", function_name(op)},
       {"scalar", typed_scalar(q->dtype)},
       {"count", text(count)},
       {"kv_count", text(kv_count)},
       {"dim", text(dim)},
       {"heads", text(heads)},
       {"group", text(group)},
       {"kend", causal ? "s + 1ULL" : text(kv_sequence) + "ULL"},
       {"load", typed_load(q->dtype)},
       {"s_expr", s_expr},
       {"ks_expr", ks_expr},
       {"kv_heads", text(kv_heads)},
       {"dq_key_base", "(" + key_row + " * " + text(kv_heads) + "ULL + kh) * " +
                           text(dim) + "ULL"},
       {"peer_query_base", "(" + query_row + " * " + text(heads) +
                               "ULL + qh) * " + text(dim) + "ULL"},
       {"peer_lse_index", query_row + " * " + text(heads) + "ULL + qh"},
       {"scale", scale_literal.str()},
       {"store", typed_store(q->dtype)},
       {"qs_start", causal ? "ks" : "0ULL"},
       {"sequence", text(sequence)}});
  // The historical emitter left the stream at precision 9, defaultfloat.
  out << std::setprecision(9) << std::defaultfloat;
}

// ---------------------------------------------------------------------------
// Elementwise region fusion (opt-in candidate property).
//
// A region is a single-consumer tree of pointwise operations {Add, Multiply,
// SiLU, Clamp, Cast, BiasAdd, ResidualGate, SwiGlu} collapsed into ONE kernel
// launched at the region's terminal ("anchor") operation; every interior
// operation is subsumed via skipped_operations and its output never touches
// global memory.  Fusion is a deliberate candidate property, not silent
// global behavior: an operation participates only when it carries
// Implementation=2 (stamped by `difc set-elementwise-fusion`, a fingerprinted
// transform).  Unstamped programs emit byte-identically to the pre-fusion
// compiler.
//
// Numerics contract: BYTE-IDENTICAL to unfused execution.  Each stage is
// computed in F32 registers and rounded to that intermediate's storage dtype
// in-register (dif_round_*) exactly where the unfused kernel rounds at its
// store.  Cross-boundary FMA contraction is prevented where it can occur --
// a mul-topped stage (Multiply, SwiGlu) feeding a consumer's add -- by
// emitting that top-level multiply as __fmul_rn, which is bit-identical to
// the lone rn-rounded FMUL of the unfused kernel and is never contracted
// (an identity mul.rn barrier was algebraically eliminated by ptxas and an
// empty asm barrier only stops front-end contraction; both measured by the
// multiply->add byte gate).  Intra-stage expression trees are structural
// copies of the unfused emitters, so intra-stage contraction stays
// identical on both sides.
//
// Memory-safety contract (the planner places tensors at slot base and
// best-fit-reuses non-dedicated slots by liveness): an external input still
// live at the anchor's position -- or holding a dedicated slot (Input,
// Output, resident Constant) -- can never alias the anchor's output.  An
// input that dies inside the region is admitted only when the program has no
// streamed constants, the region is contiguous in program order, and one of:
// (a) its aligned size is smaller than the output's (the planner cannot hand
// its slot to the output), (b) the anchor output is dedicated (nothing
// writes the dead slot before the anchor), or (c) every region read of it is
// at the thread's own output element index with equal element count and
// dtype size (per-thread read-before-write on identical bytes).  Anything
// else fails safe to per-op emission.
//
// Launch-geometry contract: the executor derives geometry from the anchor's
// opcode, so the anchor must be an element-per-thread pointwise operation.
// This is why the RmsNorm family cannot HEAD a fused region here: its
// one-block-per-row reduction structure is unreachable from a pointwise
// anchor without O(columns) redundant work per thread, so such candidates
// fail safe to per-op emission (recorded design boundary).
// ---------------------------------------------------------------------------

constexpr std::uint64_t kElementwiseFusionOptIn = 2U;

bool elementwise_fusable(ir::Opcode opcode) {
  switch (opcode) {
  case ir::Opcode::Add:
  case ir::Opcode::Multiply:
  case ir::Opcode::SiLU:
  case ir::Opcode::Clamp:
  case ir::Opcode::Cast:
  case ir::Opcode::BiasAdd:
  case ir::Opcode::ResidualGate:
  case ir::Opcode::SwiGlu:
    return true;
  default:
    return false;
  }
}

// Whether the operation reads this input slot at the same element index it
// writes its output (SwiGlu remaps rows/columns; BiasAdd's bias broadcasts).
bool elementwise_identity_slot(ir::Opcode opcode, std::size_t input_slot) {
  if (opcode == ir::Opcode::SwiGlu)
    return false;
  if (opcode == ir::Opcode::BiasAdd)
    return input_slot == 0U;
  if (opcode == ir::Opcode::ResidualGate && input_slot == 2U)
    return false;
  return true;
}

struct ElementwiseRegion {
  const ir::Operation *anchor{};
  std::vector<const ir::Operation *> members; // program order, anchor last
  std::unordered_map<std::uint32_t, const ir::Operation *> produced_by;
};

std::unordered_map<std::uint32_t, ElementwiseRegion>
find_elementwise_regions(const ir::Program &program,
                         std::unordered_set<std::uint32_t> &skipped) {
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>>
      consumers;
  std::unordered_map<std::uint32_t, std::uint64_t> position;
  bool streamed_program = false;
  for (const auto &tensor : program.tensors)
    if (tensor.has_role(ir::TensorRole::Streamed))
      streamed_program = true;
  for (std::uint64_t index = 0; index < program.operations.size(); ++index) {
    const auto &operation =
        program.operations[static_cast<std::size_t>(index)];
    position.emplace(operation.id, index);
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }
  const auto eligible = [&](const ir::Operation &operation) {
    return elementwise_fusable(operation.opcode) &&
           operation.u64(ir::AttrKey::Implementation, 1U) ==
               kElementwiseFusionOptIn &&
           !skipped.contains(operation.id) && operation.outputs.size() == 1U;
  };
  // Merge edge P -> C: P's single output is a role-free internal value with
  // exactly one consumer, and both operations opted in.
  std::unordered_map<std::uint32_t, const ir::Operation *> merge_consumer;
  for (const auto &operation : program.operations) {
    if (!eligible(operation))
      continue;
    const auto output = operation.outputs[0];
    const auto &uses = consumers[output];
    if (program.tensor(output)->roles != 0U || uses.size() != 1U ||
        !eligible(*uses.front()))
      continue;
    merge_consumer.emplace(operation.id, uses.front());
  }
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> groups;
  for (const auto &operation : program.operations) {
    if (!eligible(operation))
      continue;
    const auto *terminal = &operation;
    while (true) {
      const auto next = merge_consumer.find(terminal->id);
      if (next == merge_consumer.end())
        break;
      terminal = next->second;
    }
    groups[terminal->id].push_back(&operation);
  }

  std::unordered_map<std::uint32_t, ElementwiseRegion> regions;
  // Deterministic acceptance order: iterate terminals in program order.
  std::vector<std::uint32_t> terminals;
  terminals.reserve(groups.size());
  for (const auto &[terminal_id, members] : groups)
    if (members.size() >= 2U)
      terminals.push_back(terminal_id);
  std::sort(terminals.begin(), terminals.end(),
            [&](std::uint32_t a, std::uint32_t b) {
              return position.at(a) < position.at(b);
            });
  for (const auto terminal_id : terminals) {
    auto members = groups.at(terminal_id);
    std::sort(members.begin(), members.end(),
              [&](const ir::Operation *a, const ir::Operation *b) {
                return position.at(a->id) < position.at(b->id);
              });
    const auto *anchor = members.back();
    if (anchor->id != terminal_id)
      continue; // defensive: the terminal must be last in program order
    const auto anchor_position = position.at(anchor->id);
    const auto *anchor_output = program.tensor(anchor->outputs[0]);
    std::unordered_set<std::uint32_t> in_region;
    for (const auto *member : members)
      in_region.insert(member->id);
    // Identity-index propagation from the anchor toward producers.
    std::unordered_map<std::uint32_t, bool> identity;
    identity.emplace(anchor->id, true);
    for (auto it = members.rbegin(); it != members.rend(); ++it) {
      const auto *member = *it;
      const auto member_identity = identity.at(member->id);
      for (std::size_t slot = 0; slot < member->inputs.size(); ++slot) {
        const auto found = producer.find(member->inputs[slot]);
        if (found == producer.end() ||
            !in_region.contains(found->second->id))
          continue;
        identity.emplace(found->second->id,
                         member_identity &&
                             elementwise_identity_slot(member->opcode, slot));
      }
    }
    // External inputs and how they are read.
    std::vector<std::uint32_t> external_order;
    std::unordered_map<std::uint32_t, bool> external_identity;
    bool eligible_region = true;
    for (const auto *member : members) {
      for (std::size_t slot = 0; slot < member->inputs.size(); ++slot) {
        const auto input = member->inputs[slot];
        const auto found = producer.find(input);
        if (found != producer.end() &&
            in_region.contains(found->second->id))
          continue;
        if (found != producer.end() && skipped.contains(found->second->id)) {
          eligible_region = false; // produced by another lowering's interior
          break;
        }
        const bool read_identity =
            identity.at(member->id) &&
            elementwise_identity_slot(member->opcode, slot);
        const auto emplaced = external_identity.emplace(input, read_identity);
        if (emplaced.second)
          external_order.push_back(input);
        else
          emplaced.first->second = emplaced.first->second && read_identity;
      }
      if (!eligible_region)
        break;
    }
    // The executor marshals launch arguments into a fixed 16-pointer array
    // (launch_inputs plus the anchor's one output).
    if (!eligible_region || external_order.size() > 15U)
      continue;
    const auto dedicated = [](const ir::TensorDesc &tensor) {
      return tensor.has_role(ir::TensorRole::Input) ||
             tensor.has_role(ir::TensorRole::Output) ||
             (tensor.has_role(ir::TensorRole::Constant) &&
              !tensor.has_role(ir::TensorRole::Streamed));
    };
    const auto align256 = [](std::uint64_t bytes) {
      return (bytes + 255U) & ~static_cast<std::uint64_t>(255U);
    };
    const bool contiguous = anchor_position -
                                position.at(members.front()->id) + 1U ==
                            members.size();
    for (const auto input : external_order) {
      const auto *description = program.tensor(input);
      if (dedicated(*description))
        continue;
      std::uint64_t last_use = 0U;
      for (const auto *use : consumers[input])
        last_use = std::max(last_use, position.at(use->id));
      if (last_use >= anchor_position)
        continue; // live at the anchor: the planner keeps its slot
      if (streamed_program || !contiguous) {
        eligible_region = false;
        break;
      }
      if (align256(description->byte_count()) <
          align256(anchor_output->byte_count()))
        continue; // slot too small for the planner to hand to the output
      if (dedicated(*anchor_output))
        continue; // nothing writes the dead slot before the anchor
      if (external_identity.at(input) &&
          description->element_count() == anchor_output->element_count() &&
          ir::dtype_size(description->dtype) ==
              ir::dtype_size(anchor_output->dtype))
        continue; // per-thread read-before-write on identical bytes
      eligible_region = false;
      break;
    }
    if (!eligible_region)
      continue;
    ElementwiseRegion region;
    region.anchor = anchor;
    region.members = members;
    for (const auto *member : members) {
      if (member == anchor)
        continue;
      region.produced_by.emplace(member->outputs[0], member);
      skipped.insert(member->id);
    }
    regions.emplace(anchor->id, std::move(region));
  }
  return regions;
}

const char *typed_round(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return "dif_round_f32";
  if (dtype == ir::DType::BF16)
    return "dif_round_bf16";
  if (dtype == ir::DType::F16)
    return "dif_round_f16";
  fail("CUDA elementwise fusion admits f32, bf16, or f16 storage");
}

std::string formatted_float(double value) {
  std::ostringstream text;
  text << std::scientific << std::setprecision(9)
       << static_cast<float>(value) << "f";
  return text.str();
}

struct ElementwiseFusionEmitter {
  ElementwiseFusionEmitter(const ir::Program &program_reference,
                           const ElementwiseRegion &region_reference)
      : program(program_reference), region(region_reference) {}

  const ir::Program &program;
  const ElementwiseRegion &region;
  std::ostringstream body;
  std::vector<std::uint32_t> arguments;
  std::unordered_map<std::uint32_t, std::size_t> argument_index;
  std::unordered_map<std::string, std::string> stage_variables;
  std::unordered_map<std::string, std::pair<std::string, std::string>>
      row_variables;
  std::size_t next_variable{};

  std::string argument(std::uint32_t tensor) {
    auto found = argument_index.find(tensor);
    if (found == argument_index.end()) {
      found = argument_index.emplace(tensor, arguments.size()).first;
      arguments.push_back(tensor);
    }
    return "a" + std::to_string(found->second);
  }

  std::string value(std::uint32_t tensor, const std::string &index) {
    const auto member = region.produced_by.find(tensor);
    if (member == region.produced_by.end())
      return std::string(typed_load(program.tensor(tensor)->dtype)) + "(" +
             argument(tensor) + "," + index + ")";
    return stage(*member->second, index);
  }

  // A rounded, barriered stage value: the in-register equivalent of the
  // unfused kernel's store followed by the consumer's load.  The barrier
  // pins the rounded value so FMA contraction cannot cross the boundary.
  std::string stage(const ir::Operation &op, const std::string &index) {
    const auto key = std::to_string(op.id) + "|" + index;
    const auto found = stage_variables.find(key);
    if (found != stage_variables.end())
      return found->second;
    const auto expression = stage_expression(op, index);
    const auto name = "s" + std::to_string(next_variable++);
    body << "float " << name << "="
         << typed_round(program.tensor(op.outputs[0])->dtype) << "("
         << expression << ");asm(\"\" : \"+f\"(" << name << "));";
    stage_variables.emplace(key, name);
    return name;
  }

  std::pair<std::string, std::string> row_column(const ir::Operation &op,
                                                 const std::string &index,
                                                 std::uint64_t width) {
    const auto key = std::to_string(op.id) + "|" + index;
    auto found = row_variables.find(key);
    if (found == row_variables.end()) {
      const auto suffix = std::to_string(next_variable++);
      const auto row = "r" + suffix;
      const auto column = "c" + suffix;
      body << "unsigned long long " << row << "=(" << index << ")/" << width
           << "ULL," << column << "=(" << index << ")%" << width << "ULL;";
      found = row_variables.emplace(key, std::make_pair(row, column)).first;
    }
    return found->second;
  }

  // The unrounded value the unfused kernel would pass to dif_store,
  // structurally mirroring each per-op emitter (same operand order, same
  // interior dif_round placement) so intra-stage codegen matches.  Operands
  // are evaluated into locals in input-slot order so argument registration
  // (and therefore launch_inputs and the generated source) is deterministic.
  //
  // Boundary-contraction rule: FMA contraction merges a multiply into the
  // add that consumes it.  A stage whose TOP-LEVEL operation is a bare
  // multiply (Multiply, SwiGlu) is a lone rn-rounded FMUL in the unfused
  // kernel, so it is emitted through __fmul_rn -- bit-identical to that
  // FMUL and guaranteed never contracted into the consumer's add (ptxas
  // eliminated a mul.rn-by-1.0 barrier and contracted anyway; measured by
  // the multiply->add byte gate).  Add-topped stages cannot fuse outward
  // (there is no add->mul or add->add fusion), and interior expression
  // trees are left verbatim so intra-stage contraction stays identical to
  // the unfused kernels (e.g. ResidualGate's gate*branch may fuse with its
  // own residual add on BOTH sides).
  std::string stage_expression(const ir::Operation &op,
                               const std::string &index) {
    const auto *output = program.tensor(op.outputs[0]);
    switch (op.opcode) {
    case ir::Opcode::Add: {
      const auto left = value(op.inputs[0], index);
      const auto right = value(op.inputs[1], index);
      return left + "+" + right;
    }
    case ir::Opcode::Multiply: {
      const auto left = value(op.inputs[0], index);
      const auto right = value(op.inputs[1], index);
      return "__fmul_rn(" + left + "," + right + ")";
    }
    case ir::Opcode::SiLU:
      return "dif_silu(" + value(op.inputs[0], index) + ")";
    case ir::Opcode::Cast:
      return value(op.inputs[0], index);
    case ir::Opcode::Clamp:
      return "fminf(" + formatted_float(op.f64(ir::AttrKey::Upper, 1.0)) +
             ",fmaxf(" + formatted_float(op.f64(ir::AttrKey::Lower, 0.0)) +
             "," + value(op.inputs[0], index) + "))";
    case ir::Opcode::BiasAdd: {
      const auto width = program.tensor(op.inputs[1])->dims[0];
      const auto left = value(op.inputs[0], index);
      const auto right =
          value(op.inputs[1], "(" + index + ")%" + std::to_string(width) +
                                  "ULL");
      return left + "+" + right;
    }
    case ir::Opcode::ResidualGate: {
      const auto residual = value(op.inputs[0], index);
      const auto branch = value(op.inputs[1], index);
      const auto *gate_description = program.tensor(op.inputs[2]);
      auto gate_index = index;
      if (gate_description->element_count() != output->element_count()) {
        const auto width = output->dims.back();
        const auto rows = output->element_count() / width;
        const auto gate_rows = gate_description->element_count() / width;
        gate_index = "((" + index + ")/" + std::to_string(width) +
                     "ULL/" + std::to_string(rows / gate_rows) + "ULL)*" +
                     std::to_string(width) + "ULL+(" + index + ")%" +
                     std::to_string(width) + "ULL";
      }
      const auto gate = value(op.inputs[2], gate_index);
      return residual + "+" + std::string(typed_round(output->dtype)) + "(" +
             gate + "*" + branch + ")";
    }
    case ir::Opcode::SwiGlu: {
      const auto width = output->dims.back();
      const auto input_width = program.tensor(op.inputs[0])->dims.back();
      const auto start = op.u64(ir::AttrKey::Start, 0U);
      const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
      const auto [row, column] = row_column(op, index, width);
      const auto lane = [&](std::uint64_t offset) {
        return row + "*" + std::to_string(input_width) + "ULL+" +
               std::to_string(start + offset) + "ULL+" + column;
      };
      const auto value_lane =
          value(op.inputs[0], lane(gate_first ? width : 0U));
      const auto gate_lane =
          value(op.inputs[0], lane(gate_first ? 0U : width));
      return "__fmul_rn(" + value_lane + "," +
             std::string(typed_round(output->dtype)) + "(dif_silu(" +
             gate_lane + ")))";
    }
    default:
      fail("elementwise fusion emitter received an unsupported opcode");
    }
  }
};

void emit_fused_elementwise(std::ostringstream &out,
                            const ir::Program &program,
                            const ElementwiseRegion &region,
                            std::vector<std::uint32_t> &launch_arguments) {
  ElementwiseFusionEmitter emitter(program, region);
  const auto *output = program.tensor(region.anchor->outputs[0]);
  const auto terminal = emitter.stage_expression(*region.anchor, "i");
  std::string parameters;
  for (std::size_t argument = 0; argument < emitter.arguments.size();
       ++argument)
    parameters += std::string("const ") +
                  typed_scalar(program.tensor(emitter.arguments[argument])->dtype) +
                  "* a" + std::to_string(argument) + ", ";
  parameters += std::string(typed_scalar(output->dtype)) + "* y";
  out << render_kernel_template(
      "fused_elementwise", {{"function", function_name(*region.anchor)},
                            {"parameters", parameters},
                            {"count", std::to_string(output->element_count())},
                            {"body", emitter.body.str()},
                            {"store", typed_store(output->dtype)},
                            {"terminal", terminal}});
  launch_arguments = emitter.arguments;
}

} // namespace

GeneratedCuda emit_cuda(const ir::Program &program) {
  ir::verify(program);
  GeneratedCuda generated;
  std::ostringstream source;
  for (const auto &tensor : program.tensors) {
    if (tensor.dtype != ir::DType::F32 && tensor.dtype != ir::DType::BF16 &&
        tensor.dtype != ir::DType::F16 &&
        tensor.dtype != ir::DType::I32 && tensor.dtype != ir::DType::I8 &&
        tensor.dtype != ir::DType::FP8E4M3 &&
        tensor.dtype != ir::DType::FP8E8M0 &&
        tensor.dtype != ir::DType::Bool)
      fail("CUDA source emitter admits mixed f32/bf16/f16 plus i32 indices "
           "and packed i8/bool constants");
  }
  emit_header(source);
  std::unordered_map<std::string, std::string> pooled_entrypoints;
  auto fusions = find_lowbit_linear_fusions(program, generated.skipped_operations);
  const auto elementwise_regions =
      find_elementwise_regions(program, generated.skipped_operations);
  for (const auto &op : program.operations) {
    if (generated.skipped_operations.contains(op.id))
      continue;
    if (op.opcode == ir::Opcode::Linear) {
      const auto found = fusions.find(op.id);
      if (found != fusions.end()) {
        generated.entrypoints.emplace(op.id, function_name(op));
        auto inputs = std::vector<std::uint32_t>{op.inputs[0]};
        inputs.insert(inputs.end(), found->second.dequant->inputs.begin(),
                      found->second.dequant->inputs.end());
        if (op.inputs.size() == 3U)
          inputs.push_back(op.inputs[2]);
        generated.launch_inputs.emplace(op.id, std::move(inputs));
        begin_float_operation(source, operation_float_dtype(program, op));
        emit_lowbit_linear(source, program, found->second);
        end_float_operation(source);
      } else if (op.u64(ir::AttrKey::Implementation, 1U) == 3U) {
        fail("direct packed INT5 Linear candidate is not an eligible exclusive "
             "dequantization chain");
      } else if (op.inputs.size() == 3U &&
                 op.u64(ir::AttrKey::LinearBiasMode,
                        static_cast<std::uint64_t>(
                            ir::LinearBiasMode::Epilogue)) ==
                     static_cast<std::uint64_t>(
                         ir::LinearBiasMode::Addmm)) {
        generated.entrypoints.emplace(op.id, function_name(op));
        begin_float_operation(source, operation_float_dtype(program, op));
        emit_linear_addmm_prefill(source, program, op);
        end_float_operation(source);
      }
      continue;
    }
    if (op.opcode == ir::Opcode::Barrier ||
        op.opcode == ir::Opcode::LinearInt8Scaled ||
        op.opcode == ir::Opcode::LinearInt8WeightScaled ||
        op.opcode == ir::Opcode::LinearFp8BlockScaled ||
        op.opcode == ir::Opcode::Conv2d ||
        op.opcode == ir::Opcode::Conv3d ||
        // The convolution gradients are library plans for the same reason
        // the forward is: cuDNN owns the algorithm choice.
        op.opcode == ir::Opcode::Conv2dBackwardInput ||
        op.opcode == ir::Opcode::Conv2dBackwardWeight ||
        op.opcode == ir::Opcode::Conv2dBackwardBias ||
        (op.opcode == ir::Opcode::Attention &&
         op.u64(ir::AttrKey::Implementation, 1U) != 1U))
      continue;
    if (const auto region = elementwise_regions.find(op.id);
        region != elementwise_regions.end()) {
      generated.entrypoints.emplace(op.id, function_name(op));
      std::vector<std::uint32_t> arguments;
      emit_fused_elementwise(source, program, region->second, arguments);
      generated.launch_inputs.emplace(op.id, std::move(arguments));
      continue;
    }
    const auto identity = kernel_identity(program, op);
    if (const auto found = pooled_entrypoints.find(identity);
        found != pooled_entrypoints.end()) {
      generated.entrypoints.emplace(op.id, found->second);
      continue;
    }
    const auto entrypoint = function_name(op);
    pooled_entrypoints.emplace(identity, entrypoint);
    generated.entrypoints.emplace(op.id, entrypoint);
    if (op.opcode == ir::Opcode::Cast) {
      emit_cast(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryPosition) {
      emit_rotary_position(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryFrequency) {
      emit_rotary_frequency(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::RotaryApply) {
      emit_rotary_apply(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::BooleanMaskToBias) {
      emit_boolean_mask_to_bias(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeInt8Rows) {
      emit_quantize_int8_rows(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::DequantizeInt8Blocks) {
      emit_dequantize_int8_blocks(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeFp8Rows) {
      emit_quantize_fp8_rows(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::LinearFp8Scaled) {
      emit_linear_fp8_output_scale(source, program, op);
      continue;
    }
    if (op.opcode == ir::Opcode::QuantizeFp8Blocks32) {
      emit_quantize_fp8_blocks32(source, program, op);
      continue;
    }
    begin_float_operation(source, operation_float_dtype(program, op));
    switch (op.opcode) {
    case ir::Opcode::Add:
      emit_elementwise(source, program, op, "dif_load(a,i)+dif_load(b,i)");
      break;
    case ir::Opcode::Multiply:
      emit_elementwise(source, program, op, "dif_load(a,i)*dif_load(b,i)");
      break;
    case ir::Opcode::AffineLastDim:
      emit_affine_last_dim(source, program, op);
      break;
    case ir::Opcode::SiLU:
      emit_silu(source, program, op);
      break;
    case ir::Opcode::Gelu:
      emit_gelu(source, program, op);
      break;
    case ir::Opcode::Sigmoid:
      emit_sigmoid(source, program, op);
      break;
    case ir::Opcode::Reshape:
      emit_reshape(source, program, op);
      break;
    case ir::Opcode::BroadcastTo:
      emit_broadcast_to(source, program, op);
      break;
    case ir::Opcode::Slice:
      emit_slice(source, program, op);
      break;
    case ir::Opcode::RotaryFrequency:
    case ir::Opcode::RotaryApply:
    case ir::Opcode::BooleanMaskToBias:
      break;
    case ir::Opcode::RmsNorm:
      emit_rms_norm(source, program, op);
      break;
    case ir::Opcode::LayerNorm:
      emit_layer_norm(source, program, op);
      break;
    case ir::Opcode::LayerNormModulate:
      emit_layer_norm_modulate(source, program, op);
      break;
    case ir::Opcode::Clamp:
      emit_clamp(source, program, op);
      break;
    case ir::Opcode::MseLoss:
      emit_mse_loss(source, program, op);
      break;
    case ir::Opcode::MseLossBackward:
      emit_mse_loss_backward(source, program, op);
      break;
    case ir::Opcode::LinearBackwardInput:
      emit_linear_backward_input(source, program, op);
      break;
    case ir::Opcode::LinearBackwardWeight:
      emit_linear_backward_weight(source, program, op);
      break;
    case ir::Opcode::BiasBackward:
      emit_bias_backward(source, program, op);
      break;
    case ir::Opcode::SiLUBackward:
      emit_silu_backward(source, program, op);
      break;
    case ir::Opcode::GeluBackward:
      emit_gelu_backward(source, program, op);
      break;
    case ir::Opcode::UpsampleNearest2dBackward:
      emit_upsample_nearest_2d_backward(source, program, op);
      break;
    case ir::Opcode::SliceBackward:
      emit_slice_backward(source, program, op);
      break;
    case ir::Opcode::BroadcastToBackward:
      emit_broadcast_to_backward(source, program, op);
      break;
    case ir::Opcode::GroupNormBackward:
      emit_group_norm_backward(source, program, op);
      break;
    case ir::Opcode::GroupNormBackwardAffine:
      emit_group_norm_backward_affine(source, program, op);
      break;
    case ir::Opcode::AdamWUpdate:
      emit_adamw_update(source, program, op);
      break;
    case ir::Opcode::Fill:
      emit_fill(source, program, op);
      break;
    case ir::Opcode::GatherRows:
      emit_gather_rows(source, program, op);
      break;
    case ir::Opcode::IndexedUpdateRows:
      emit_indexed_update_rows(source, program, op);
      break;
    case ir::Opcode::Cast:
      break;
    case ir::Opcode::SelectRowChunks:
      emit_select_row_chunks(source, program, op);
      break;
    case ir::Opcode::SinusoidalTimestep:
      emit_sinusoidal_timestep(source, program, op);
      break;
    case ir::Opcode::RotaryPosition:
      break;
    case ir::Opcode::LinearBlend:
      emit_linear_blend(source, program, op);
      break;
    case ir::Opcode::FlowEulerStep:
      emit_flow_euler_step(source, program, op);
      break;
    case ir::Opcode::EulerVelocityStep:
      emit_euler_velocity_step(source, program, op);
      break;
    case ir::Opcode::Permute:
      emit_permute(source, program, op);
      break;
    case ir::Opcode::Concat:
      emit_concat(source, program, op);
      break;
    case ir::Opcode::Patchify3D:
      emit_patchify_3d(source, program, op, false);
      break;
    case ir::Opcode::Unpatchify3D:
      emit_patchify_3d(source, program, op, true);
      break;
    case ir::Opcode::RmsNormModulate:
      emit_rms_norm_modulate(source, program, op);
      break;
    case ir::Opcode::SwiGlu:
      emit_swiglu(source, program, op);
      break;
    case ir::Opcode::ResidualGate:
      emit_residual_gate(source, program, op);
      break;
    case ir::Opcode::Linear:
      break;
    case ir::Opcode::QkNormPartialRope:
      emit_qk_norm_rope(source, program, op);
      break;
    case ir::Opcode::Attention:
      emit_attention(source, program, op);
      break;
    case ir::Opcode::Barrier:
      break;
    case ir::Opcode::BiasAdd:
      emit_bias_add(source, program, op);
      break;
    case ir::Opcode::H3AdaLNSelect:
      emit_h3_adaln_select(source, program, op);
      break;
    case ir::Opcode::H3DeinterleaveQkv:
      emit_h3_deinterleave_qkv(source, program, op);
      break;
    case ir::Opcode::H3DeinterleaveQkvWeight:
      emit_h3_deinterleave_qkv_weight(source, program, op);
      break;
    case ir::Opcode::DequantizeInt4:
      emit_dequantize_int4(source, program, op);
      break;
    case ir::Opcode::DequantizeInt5:
      emit_dequantize_int5(source, program, op);
      break;
    case ir::Opcode::RmsNormBackward:
      emit_rms_norm_backward(source, program, op);
      break;
    case ir::Opcode::RmsNormModulateBackward:
      emit_rms_norm_modulate_backward(source, program, op);
      break;
    case ir::Opcode::SwiGluBackward:
      emit_swiglu_backward(source, program, op);
      break;
    case ir::Opcode::ResidualGateBackward:
      emit_residual_gate_backward(source, program, op);
      break;
    case ir::Opcode::LayerNormBackward:
      emit_layer_norm_backward(source, program, op);
      break;
    case ir::Opcode::QkNormPartialRopeBackward:
      emit_qk_norm_rope_backward(source, program, op);
      break;
    case ir::Opcode::LayerNormModulateBackward:
      emit_layer_norm_modulate_backward(source, program, op);
      break;
    case ir::Opcode::GatherRowsBackward:
      emit_gather_rows_backward(source, program, op);
      break;
    case ir::Opcode::SigmoidBackward:
      emit_sigmoid_backward(source, program, op);
      break;
    case ir::Opcode::ClampBackward:
      emit_clamp_backward(source, program, op);
      break;
    case ir::Opcode::AttentionLse:
      emit_attention_lse(source, program, op);
      break;
    case ir::Opcode::AttentionBackward:
      emit_attention_backward(source, program, op);
      break;
    case ir::Opcode::Conv1d:
      emit_conv1d(source, program, op);
      break;
    case ir::Opcode::Conv2d:
    // The convolution and its three gradients are cuDNN plans, so codegen
    // emits nothing for them.
    case ir::Opcode::Conv2dBackwardInput:
    case ir::Opcode::Conv2dBackwardWeight:
    case ir::Opcode::Conv2dBackwardBias:
      break;
    case ir::Opcode::ChannelRmsNorm:
      emit_channel_rms_norm(source, program, op);
      break;
    case ir::Opcode::GroupNorm:
      emit_group_norm(source, program, op);
      break;
    case ir::Opcode::UpsampleNearest2d:
      emit_upsample_nearest_2d(source, program, op);
      break;
    case ir::Opcode::PadConstant:
      emit_pad_constant(source, program, op);
      break;
    case ir::Opcode::PadReflect:
      emit_pad_reflect(source, program, op);
      break;
    case ir::Opcode::Conv3d:
      break;
    case ir::Opcode::SnakeBeta:
      emit_snake_beta(source, program, op);
      break;
    case ir::Opcode::QuantizeInt8Rows:
    case ir::Opcode::LinearInt8Scaled:
    case ir::Opcode::QuantizeFp8Rows:
    case ir::Opcode::LinearFp8Scaled:
    case ir::Opcode::QuantizeFp8Blocks32:
    case ir::Opcode::LinearFp8BlockScaled:
    case ir::Opcode::DequantizeInt8Blocks:
    case ir::Opcode::LinearInt8WeightScaled:
      break;
    }
    end_float_operation(source);
  }
  generated.source = source.str();
  return generated;
}

ElementwiseFusionCensus
census_elementwise_fusion(const ir::Program &program) {
  ir::verify(program);
  std::unordered_set<std::uint32_t> skipped;
  find_lowbit_linear_fusions(program, skipped);
  const auto regions = find_elementwise_regions(program, skipped);
  ElementwiseFusionCensus census;
  census.regions = regions.size();
  for (const auto &[anchor, region] : regions)
    census.fused_operations += region.members.size();
  census.eliminated_launches = census.fused_operations - census.regions;
  return census;
}

} // namespace dif::compiler
