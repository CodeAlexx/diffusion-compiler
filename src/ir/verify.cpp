#include "dif/ir/verify.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace dif::ir {
namespace {

bool valid_dtype(DType dtype) {
  return dtype == DType::F32 || dtype == DType::BF16 || dtype == DType::F16 ||
         dtype == DType::I8 || dtype == DType::I32;
}

bool supported_float(DType dtype) {
  return dtype == DType::F32 || dtype == DType::BF16 || dtype == DType::F16;
}

bool valid_opcode(Opcode opcode) {
  return opcode >= Opcode::Add && opcode <= Opcode::SnakeBeta;
}

bool valid_attr_key(AttrKey key) {
  return key >= AttrKey::Epsilon && key <= AttrKey::KvHeads;
}

bool valid_attr_kind(AttrKind kind) {
  return kind >= AttrKind::U64 && kind <= AttrKind::Bool;
}

void expect_counts(const Operation &op, std::size_t inputs, std::size_t outputs) {
  if (op.inputs.size() != inputs || op.outputs.size() != outputs) {
    fail("DiffIR op " + std::to_string(op.id) + " (" +
         std::string(opcode_name(op.opcode)) + ") has invalid arity");
  }
}

void same_shape_dtype(const TensorDesc &a, const TensorDesc &b,
                      const Operation &op) {
  if (a.dtype != b.dtype || a.dims != b.dims)
    fail("DiffIR op " + std::to_string(op.id) + " requires equal shape/dtype");
}

// Training kernels accumulate in F32 unconditionally (the flame dtype
// contract: storage may be BF16/F16, opmath is F32).  An explicit
// AccumulatorDType attribute is honored by requiring it to name F32; any
// other requested accumulator is an unsupported semantic and fails closed.
void check_accumulator_f32(const Operation &op) {
  const auto *attribute = op.find(AttrKey::AccumulatorDType);
  if (attribute == nullptr)
    return;
  if (attribute->kind != AttrKind::U64 ||
      attribute->bits != static_cast<std::uint64_t>(DType::F32))
    fail("DiffIR op " + std::to_string(op.id) +
         " requests a non-F32 accumulator; training ops accumulate in F32");
}

const TensorDesc &tensor_or_fail(const Program &program, std::uint32_t id,
                                 const Operation &op) {
  const auto *tensor = program.tensor(id);
  if (!tensor)
    fail("DiffIR op " + std::to_string(op.id) + " references missing tensor " +
         std::to_string(id));
  return *tensor;
}

void verify_attrs(const Operation &op) {
  std::unordered_set<std::uint32_t> keys;
  for (const auto &attr : op.attributes) {
    if (!valid_attr_key(attr.key) || !valid_attr_kind(attr.kind))
      fail("DiffIR op has unknown attribute key or kind");
    if (!keys.insert(static_cast<std::uint32_t>(attr.key)).second)
      fail("DiffIR op has duplicate attribute key");
    if (attr.kind == AttrKind::Bool && attr.bits > 1U)
      fail("DiffIR bool attribute is not zero or one");
    if (attr.kind == AttrKind::F64 && !std::isfinite(std::bit_cast<double>(attr.bits)))
      fail("DiffIR floating attribute is non-finite");
  }
}

void verify_operation(const Program &program, const Operation &op) {
  verify_attrs(op);
  if (op.opcode == Opcode::Barrier) {
    expect_counts(op, 0, 0);
    return;
  }

  if (op.opcode == Opcode::Add || op.opcode == Opcode::Multiply) {
    expect_counts(op, 2, 1);
    const auto &a = tensor_or_fail(program, op.inputs[0], op);
    const auto &b = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(a, b, op);
    same_shape_dtype(a, out, op);
    if (!supported_float(a.dtype))
      fail("elementwise semantics admit f32, bf16, or f16");
    return;
  }

  if (op.opcode == Opcode::AffineLastDim) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("affine_last_dim expects input, scale, optional bias, and one "
           "output");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &scale = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        scale.dtype != input.dtype || scale.dims.size() != 1U ||
        scale.dims[0] != input.dims.back())
      fail("affine_last_dim requires float input/output and scale matching "
           "the final dimension");
    if (op.inputs.size() == 3U) {
      const auto &bias = tensor_or_fail(program, op.inputs[2], op);
      if (bias.dtype != input.dtype || bias.dims != scale.dims)
        fail("affine_last_dim bias must match its scale");
    }
    return;
  }

  if (op.opcode == Opcode::LayerNorm) {
    expect_counts(op, 3, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &bias = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back() || bias.dtype != input.dtype ||
        bias.dims != weight.dims)
      fail("layer_norm requires float input/output and affine vectors "
           "matching the final dimension");
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("layer_norm epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("layer_norm block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::Clamp) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    const auto lower = op.f64(AttrKey::Lower,
                              -std::numeric_limits<double>::infinity());
    const auto upper = op.f64(AttrKey::Upper,
                              std::numeric_limits<double>::infinity());
    if (!supported_float(input.dtype) || !(lower <= upper))
      fail("clamp requires float input/output and lower <= upper");
    return;
  }

  if (op.opcode == Opcode::MseLoss) {
    expect_counts(op, 2, 1);
    const auto &prediction = tensor_or_fail(program, op.inputs[0], op);
    const auto &target = tensor_or_fail(program, op.inputs[1], op);
    const auto &loss = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(prediction, target, op);
    check_accumulator_f32(op);
    if (!supported_float(prediction.dtype) || loss.dtype != DType::F32 ||
        loss.dims != std::vector<std::uint64_t>{1U})
      fail("mse_loss requires equal float inputs and an F32[1] output");
    return;
  }

  if (op.opcode == Opcode::MseLossBackward) {
    expect_counts(op, 3, 1);
    const auto &prediction = tensor_or_fail(program, op.inputs[0], op);
    const auto &target = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_loss = tensor_or_fail(program, op.inputs[2], op);
    const auto &grad_prediction = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(prediction, target, op);
    same_shape_dtype(prediction, grad_prediction, op);
    check_accumulator_f32(op);
    if (!supported_float(prediction.dtype) ||
        grad_loss.dtype != DType::F32 ||
        grad_loss.dims != std::vector<std::uint64_t>{1U})
      fail("mse_loss_backward requires float tensors and F32[1] grad_loss");
    return;
  }

  if (op.opcode == Opcode::LinearBackwardInput) {
    expect_counts(op, 2, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    if (!supported_float(grad_output.dtype) ||
        weight.dtype != grad_output.dtype ||
        grad_input.dtype != grad_output.dtype || grad_output.dims.empty() ||
        grad_input.dims.empty() || weight.dims.size() != 2U)
      fail("linear_backward_input has incompatible float dtypes or ranks");
    // Two admitted geometries, mirroring the forward Linear contract:
    // (a) same-rank leading-broadcast: [...,N] x [N,K] -> [...,K];
    // (b) flatten form: rows = dims[0] and trailing dims flatten, exactly
    //     the shapes forward Linear admits (e.g. grad of a [S,H,D]-declared
    //     projection back to [S,H*D]).  The kernels index the flat
    //     row-major view and are correct for both.
    const bool same_rank_form =
        grad_output.dims.size() == grad_input.dims.size() &&
        std::equal(grad_output.dims.begin(), grad_output.dims.end() - 1,
                   grad_input.dims.begin()) &&
        grad_output.dims.back() == weight.dims[0] &&
        grad_input.dims.back() == weight.dims[1];
    const bool flatten_form =
        grad_output.dims[0] == grad_input.dims[0] &&
        grad_output.element_count() / grad_output.dims[0] == weight.dims[0] &&
        grad_input.element_count() / grad_input.dims[0] == weight.dims[1];
    if (!same_rank_form && !flatten_form)
      fail("linear_backward_input has incompatible float shapes");
    return;
  }

  if (op.opcode == Opcode::LinearBackwardWeight) {
    expect_counts(op, 2, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_weight = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    if (!supported_float(grad_output.dtype) ||
        input.dtype != grad_output.dtype ||
        grad_weight.dtype != grad_output.dtype || grad_output.dims.empty() ||
        input.dims.empty() || grad_weight.dims.size() != 2U)
      fail("linear_backward_weight has incompatible float dtypes or ranks");
    // Same two geometries as linear_backward_input (see the comment there).
    const bool same_rank_form =
        grad_output.dims.size() == input.dims.size() &&
        std::equal(grad_output.dims.begin(), grad_output.dims.end() - 1,
                   input.dims.begin()) &&
        grad_weight.dims[0] == grad_output.dims.back() &&
        grad_weight.dims[1] == input.dims.back();
    const bool flatten_form =
        grad_output.dims[0] == input.dims[0] &&
        grad_weight.dims[0] ==
            grad_output.element_count() / grad_output.dims[0] &&
        grad_weight.dims[1] == input.element_count() / input.dims[0];
    if (!same_rank_form && !flatten_form)
      fail("linear_backward_weight has incompatible float shapes");
    return;
  }

  if (op.opcode == Opcode::BiasBackward) {
    expect_counts(op, 1, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_bias = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    if (!supported_float(grad_output.dtype) || grad_output.dims.empty() ||
        grad_bias.dtype != grad_output.dtype || grad_bias.dims.size() != 1U)
      fail("bias_backward requires a float output gradient and a vector "
           "gradient");
    // The kernel sums the flat row-major view in rows of grad_bias's width.
    // Two named widths are admitted: the final dimension (BiasAdd's
    // broadcast-over-leading-dims contract) and the flattened row width
    // (a Linear bias whose output was declared with split trailing dims,
    // e.g. [S,H,D] with bias [H*D]).
    if (grad_bias.dims[0] != grad_output.dims.back() &&
        grad_bias.dims[0] !=
            grad_output.element_count() / grad_output.dims[0])
      fail("bias_backward gradient width must be the final dimension or "
           "the flattened row width");
    return;
  }

  if (op.opcode == Opcode::SiLUBackward) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype))
      fail("silu_backward admits f32, bf16, or f16 tensors");
    return;
  }

  if (op.opcode == Opcode::AdamWUpdate) {
    expect_counts(op, 5, 3);
    const auto &parameter = tensor_or_fail(program, op.inputs[0], op);
    const auto &gradient = tensor_or_fail(program, op.inputs[1], op);
    const auto &first = tensor_or_fail(program, op.inputs[2], op);
    const auto &second = tensor_or_fail(program, op.inputs[3], op);
    const auto &step = tensor_or_fail(program, op.inputs[4], op);
    const auto &updated = tensor_or_fail(program, op.outputs[0], op);
    const auto &updated_first = tensor_or_fail(program, op.outputs[1], op);
    const auto &updated_second = tensor_or_fail(program, op.outputs[2], op);
    check_accumulator_f32(op);
    // Flame AdamW kernel matrix: parameter and gradient storage may be F32
    // or BF16 independently ({BF16p/BF16g, BF16p/F32g, F32p/F32g,
    // F32p/BF16g}); both moments are F32 ALWAYS, and the updated parameter
    // keeps the parameter's storage dtype.
    const auto adamw_storage = [](DType dtype) {
      return dtype == DType::F32 || dtype == DType::BF16;
    };
    if (!adamw_storage(parameter.dtype) || !adamw_storage(gradient.dtype))
      fail("adamw_update parameter and gradient must be F32 or BF16");
    if (gradient.dims != parameter.dims || first.dims != parameter.dims ||
        second.dims != parameter.dims || updated.dims != parameter.dims ||
        updated_first.dims != parameter.dims ||
        updated_second.dims != parameter.dims)
      fail("adamw_update state must share the parameter shape");
    if (first.dtype != DType::F32 || second.dtype != DType::F32 ||
        updated_first.dtype != DType::F32 ||
        updated_second.dtype != DType::F32)
      fail("adamw_update moments must be F32");
    if (updated.dtype != parameter.dtype)
      fail("adamw_update output must match the parameter dtype");
    const auto learning_rate = op.f64(AttrKey::LearningRate, 1.0e-3);
    const auto beta1 = op.f64(AttrKey::Beta1, 0.9);
    const auto beta2 = op.f64(AttrKey::Beta2, 0.999);
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-8);
    const auto weight_decay = op.f64(AttrKey::WeightDecay, 0.0);
    if (step.dtype != DType::I32 ||
        step.dims != std::vector<std::uint64_t>{1U} ||
        !(learning_rate > 0.0) || beta1 < 0.0 || beta1 >= 1.0 ||
        beta2 < 0.0 || beta2 >= 1.0 || !(epsilon > 0.0) ||
        weight_decay < 0.0)
      fail("adamw_update has an invalid step or hyperparameters");
    return;
  }

  if (op.opcode == Opcode::SiLU) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype))
      fail("silu semantics admit f32, bf16, or f16");
    return;
  }

  if (op.opcode == Opcode::RmsNorm) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back())
      fail("rms_norm requires float input/output and weight matching the "
           "final dimension");
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("rms_norm epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("rms_norm block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::Fill) {
    expect_counts(op, 0, 1);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(out.dtype))
      fail("fill semantics admit f32, bf16, or f16 output");
    return;
  }

  if (op.opcode == Opcode::GatherRows) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &indices = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || input.dims.size() < 2U ||
        indices.dtype != DType::I32 || indices.dims.size() != 1U ||
        out.dtype != input.dtype || out.dims.size() != input.dims.size() ||
        out.dims[0] != indices.dims[0] ||
        !std::equal(input.dims.begin() + 1, input.dims.end(),
                    out.dims.begin() + 1))
      fail("gather_rows requires float [S,...], i32 [M], and float [M,...]");
    return;
  }

  if (op.opcode == Opcode::IndexedUpdateRows) {
    expect_counts(op, 3, 1);
    const auto &base = tensor_or_fail(program, op.inputs[0], op);
    const auto &updates = tensor_or_fail(program, op.inputs[1], op);
    const auto &map = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(base, out, op);
    if (!supported_float(base.dtype) || base.dims.size() < 2U ||
        updates.dtype != base.dtype || updates.dims.size() != base.dims.size() ||
        map.dtype != DType::I32 || map.dims.size() != 1U ||
        map.dims[0] != base.dims[0] ||
        !std::equal(base.dims.begin() + 1, base.dims.end(),
                    updates.dims.begin() + 1))
      fail("indexed_update_rows requires base/output [S,...], updates [M,...], "
           "and destination map i32 [S]");
    return;
  }

  if (op.opcode == Opcode::Cast) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || !supported_float(out.dtype) ||
        input.dtype == out.dtype || input.dims != out.dims)
      fail("cast requires equal-shaped tensors with distinct float dtypes");
    return;
  }

  if (op.opcode == Opcode::SelectRowChunks) {
    if (op.inputs.size() != 2U || op.outputs.empty() ||
        op.outputs.size() > 8U)
      fail("select_row_chunks expects values, indices, and one or more outputs");
    const auto &values = tensor_or_fail(program, op.inputs[0], op);
    const auto &indices = tensor_or_fail(program, op.inputs[1], op);
    const auto &first = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(values.dtype) || values.dims.size() != 2U ||
        indices.dtype != DType::I32 || indices.dims.size() != 1U ||
        first.dtype != values.dtype || first.dims.size() != 2U ||
        first.dims[0] != indices.dims[0] ||
        values.dims[1] != op.outputs.size() * first.dims[1])
      fail("select_row_chunks requires float [T,C*H], i32 [S], and C float "
           "outputs [S,H]");
    for (const auto output : op.outputs)
      same_shape_dtype(first, tensor_or_fail(program, output, op), op);
    return;
  }

  if (op.opcode == Opcode::SinusoidalTimestep) {
    expect_counts(op, 1, 1);
    const auto &timesteps = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (timesteps.dtype != DType::F32 || timesteps.dims.size() != 1U ||
        out.dtype != DType::F32 || out.dims.size() != 2U ||
        out.dims[0] != timesteps.dims[0] || out.dims[1] < 2U)
      fail("sinusoidal_timestep requires f32 [N] and f32 [N,D] with D>=2");
    const auto half = out.dims[1] / 2U;
    const auto shift = op.f64(AttrKey::DownscaleFreqShift, 1.0);
    const auto scale = op.f64(AttrKey::Scale, 1.0);
    const auto max_period = op.f64(AttrKey::MaxPeriod, 10000.0);
    if (!(static_cast<double>(half) > shift) || !(max_period > 0.0) ||
        !std::isfinite(scale))
      fail("sinusoidal_timestep requires half_dim>downscale shift, positive "
           "max period, and finite scale");
    return;
  }

  if (op.opcode == Opcode::RotaryPosition) {
    expect_counts(op, 2, 2);
    const auto &positions = tensor_or_fail(program, op.inputs[0], op);
    const auto &inv_freq = tensor_or_fail(program, op.inputs[1], op);
    const auto &cosine = tensor_or_fail(program, op.outputs[0], op);
    const auto &sine = tensor_or_fail(program, op.outputs[1], op);
    if (positions.dtype != DType::F32 || positions.dims.size() != 2U ||
        inv_freq.dtype != DType::F32 || inv_freq.dims.size() != 1U ||
        !supported_float(cosine.dtype) || cosine.dtype != sine.dtype ||
        cosine.dims != sine.dims || cosine.dims.size() != 2U ||
        cosine.dims[0] != positions.dims[0] ||
        positions.dims[1] >
            std::numeric_limits<std::uint64_t>::max() / inv_freq.dims[0] / 2U ||
        cosine.dims[1] != 2U * positions.dims[1] * inv_freq.dims[0])
      fail("rotary_position requires f32 positions [S,A], f32 inv_freq [F], "
           "and matching float cos/sin [S,2*A*F]");
    return;
  }

  if (op.opcode == Opcode::LinearBlend) {
    expect_counts(op, 3, 1);
    const auto &left = tensor_or_fail(program, op.inputs[0], op);
    const auto &right = tensor_or_fail(program, op.inputs[1], op);
    const auto &factor = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(left, right, op);
    same_shape_dtype(left, out, op);
    if (!supported_float(left.dtype) || factor.dtype != DType::F32 ||
        factor.dims != std::vector<std::uint64_t>{1U})
      fail("linear_blend requires equal float values/output and f32 [1] factor");
    return;
  }

  if (op.opcode == Opcode::FlowEulerStep) {
    expect_counts(op, 4, 1);
    const auto &sample = tensor_or_fail(program, op.inputs[0], op);
    const auto &velocity = tensor_or_fail(program, op.inputs[1], op);
    const auto &timesteps = tensor_or_fail(program, op.inputs[2], op);
    const auto &sigmas = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(sample, velocity, op);
    same_shape_dtype(sample, out, op);
    const auto *step_attribute = op.find(AttrKey::StepIndex);
    if (!supported_float(sample.dtype) || timesteps.dtype != DType::F32 ||
        timesteps.dims.size() != 1U || sigmas.dtype != DType::F32 ||
        sigmas.dims.size() != 1U ||
        timesteps.dims[0] == std::numeric_limits<std::uint64_t>::max() ||
        sigmas.dims[0] != timesteps.dims[0] + 1U || !step_attribute ||
        step_attribute->as_u64() >= timesteps.dims[0])
      fail("flow_euler_step requires equal float sample/velocity/output, "
           "f32 timesteps [K], f32 sigmas [K+1], and StepIndex<K");
    return;
  }

  if (op.opcode == Opcode::Patchify3D ||
      op.opcode == Opcode::Unpatchify3D) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &output = tensor_or_fail(program, op.outputs[0], op);
    const auto *patch_t_attr = op.find(AttrKey::PatchT);
    const auto *patch_h_attr = op.find(AttrKey::PatchH);
    const auto *patch_w_attr = op.find(AttrKey::PatchW);
    if (!patch_t_attr || !patch_h_attr || !patch_w_attr)
      fail("patchify_3d/unpatchify_3d requires PatchT, PatchH, and PatchW");
    const auto patch_t = patch_t_attr->as_u64();
    const auto patch_h = patch_h_attr->as_u64();
    const auto patch_w = patch_w_attr->as_u64();
    const auto &volume = op.opcode == Opcode::Patchify3D ? input : output;
    const auto &rows = op.opcode == Opcode::Patchify3D ? output : input;
    if (!supported_float(input.dtype) || input.dtype != output.dtype ||
        volume.dims.size() != 5U || rows.dims.size() != 2U ||
        patch_t == 0U || patch_h == 0U || patch_w == 0U ||
        volume.dims[2] % patch_t != 0U ||
        volume.dims[3] % patch_h != 0U ||
        volume.dims[4] % patch_w != 0U)
      fail("patchify_3d/unpatchify_3d requires equal float dtype, rank-5 "
           "[B,C,T,H,W], rank-2 rows, and divisible nonzero patches");
    if (patch_t > std::numeric_limits<std::uint64_t>::max() / patch_h ||
        patch_t * patch_h >
            std::numeric_limits<std::uint64_t>::max() / patch_w ||
        patch_t * patch_h * patch_w >
            std::numeric_limits<std::uint64_t>::max() / volume.dims[1])
      fail("patchify_3d/unpatchify_3d patch geometry overflows");
    const auto expected_columns =
        volume.dims[1] * patch_t * patch_h * patch_w;
    const auto expected_rows = volume.element_count() / expected_columns;
    if (rows.dims[0] != expected_rows || rows.dims[1] != expected_columns ||
        rows.element_count() != volume.element_count())
      fail("patchify_3d/unpatchify_3d row geometry does not match volume");
    return;
  }

  if (op.opcode == Opcode::RmsNormModulate) {
    if ((op.inputs.size() != 3 && op.inputs.size() != 4) || op.outputs.size() != 1)
      fail("rms_norm_modulate expects x,[weight],scale,shift and one output");
    const auto &x = tensor_or_fail(program, op.inputs[0], op);
    const auto scale_index = op.inputs.size() == 4 ? 2U : 1U;
    const auto shift_index = op.inputs.size() == 4 ? 3U : 2U;
    const auto &scale = tensor_or_fail(program, op.inputs[scale_index], op);
    const auto &shift = tensor_or_fail(program, op.inputs[shift_index], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(x, scale, op);
    same_shape_dtype(x, shift, op);
    same_shape_dtype(x, out, op);
    if (!supported_float(x.dtype) || x.dims.size() != 2)
      fail("rms_norm_modulate semantics require rank-2 f32, bf16, or f16 tensors");
    if (op.inputs.size() == 4) {
      const auto &weight = tensor_or_fail(program, op.inputs[1], op);
      if (weight.dtype != x.dtype || weight.dims.size() != 1 ||
          weight.dims[0] != x.dims[1])
        fail("rms_norm_modulate weight must match x dtype and [hidden]");
    }
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("rms_norm_modulate epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256);
    if (block < 32 || block > 1024 || (block & (block - 1U)) != 0U)
      fail("rms_norm_modulate block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::SwiGlu) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        input.dims.size() != out.dims.size() || input.dims.empty())
      fail("swiglu semantics require compatible f32, bf16, or f16 tensors");
    auto expected = out.dims;
    expected.back() *= 2U;
    if (input.dims != expected)
      fail("swiglu input final dimension must be twice output final dimension");
    return;
  }

  if (op.opcode == Opcode::ResidualGate) {
    expect_counts(op, 3, 1);
    const auto &residual = tensor_or_fail(program, op.inputs[0], op);
    const auto &branch = tensor_or_fail(program, op.inputs[1], op);
    const auto &gate = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(residual, branch, op);
    same_shape_dtype(residual, gate, op);
    same_shape_dtype(residual, out, op);
    if (!supported_float(residual.dtype))
      fail("residual_gate semantics admit f32, bf16, or f16");
    return;
  }

  if (op.opcode == Opcode::Linear) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("linear expects input, weight, optional bias, and one output");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || weight.dtype != input.dtype ||
        out.dtype != input.dtype || input.dims.empty() ||
        weight.dims.size() != 2 || out.dims.empty())
      fail("linear semantics require uniform f32/bf16/f16 and rank-2 weights");
    const auto rows = input.dims[0];
    const auto inner = input.element_count() / rows;
    const auto out_width = out.element_count() / out.dims[0];
    if (inner != weight.dims[1] || out.dims[0] != rows ||
        out_width != weight.dims[0])
      fail("linear shapes must flatten as [M,K] x [N,K] -> [M,N]");
    if (op.inputs.size() == 3U) {
      const auto &bias = tensor_or_fail(program, op.inputs[2], op);
      if (bias.dtype != input.dtype || bias.dims.size() != 1U ||
          bias.dims[0] != weight.dims[0])
        fail("linear bias must have the graph dtype and shape [N]");
    }
    const auto implementation = op.u64(AttrKey::Implementation, 1U);
    if (implementation != 1U && implementation != 2U && implementation != 3U)
      fail("linear implementation must be 1 (native), 2 (tf32), or 3 "
           "(direct packed INT5)");
    if (input.dtype != DType::F32 && implementation == 2U)
      fail("tf32 Linear implementation requires f32 storage");
    return;
  }

  if (op.opcode == Opcode::QkNormPartialRope) {
    expect_counts(op, 4, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &cos = tensor_or_fail(program, op.inputs[2], op);
    const auto &sin = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.size() != 3 ||
        weight.dtype != input.dtype || weight.dims.size() != 1 ||
        cos.dtype != input.dtype || sin.dtype != input.dtype || cos.dims != sin.dims ||
        cos.dims.size() != 2)
      fail("qk_norm_partial_rope requires input [S,H,D], weight [D], cos/sin [S,R]");
    const auto heads = op.u64(AttrKey::Heads, input.dims[1]);
    const auto head_dim = op.u64(AttrKey::HeadDim, input.dims[2]);
    const auto rotary = op.u64(AttrKey::RotaryDim, cos.dims[1] * 2U);
    if (heads != input.dims[1] || head_dim != input.dims[2] ||
        weight.dims[0] != head_dim || cos.dims[0] != input.dims[0] ||
        rotary == 0 || rotary > head_dim || (rotary % 2U) != 0U ||
        (cos.dims[1] != rotary && cos.dims[1] * 2U != rotary))
      fail("qk_norm_partial_rope shape attributes are inconsistent");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("qk_norm_partial_rope block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::Attention) {
    expect_counts(op, 3, 1);
    const auto &q = tensor_or_fail(program, op.inputs[0], op);
    const auto &k = tensor_or_fail(program, op.inputs[1], op);
    const auto &v = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(k, v, op);
    same_shape_dtype(q, out, op);
    if (!supported_float(q.dtype) || q.dims.size() != 3)
      fail("attention semantics require f32, bf16, or f16 [S,H,D]");
    // Grouped-query attention: k/v carry KvHeads heads (AttrKey 39); query
    // head h reads kv head h/(H/KvHeads).  An absent attribute means
    // KvHeads == H — bit-for-bit the historical contract, so every existing
    // program verifies and fingerprints unchanged.
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[1]);
    if (kv_heads == 0U || q.dims[1] % kv_heads != 0U)
      fail("attention KvHeads must be nonzero and divide the query head "
           "count");
    if (k.dtype != q.dtype || k.dims.size() != 3U ||
        k.dims[0] != q.dims[0] || k.dims[1] != kv_heads ||
        k.dims[2] != q.dims[2])
      fail("attention k/v must be [S,KvHeads,D] with the query dtype");
    const auto implementation = op.u64(AttrKey::Implementation, 1U);
    if (implementation != 1U && implementation != 2U)
      fail("attention implementation must be 1 (generated) or 2 (cuDNN)");
    if (implementation == 2U && q.dtype != DType::BF16 &&
        q.dtype != DType::F16)
      fail("cuDNN attention implementation requires bf16 or f16");
    if (implementation == 1U && q.dims[0] > 4096U)
      fail("naive exact attention is admitted only for S<=4096; use a backend implementation");
    const auto block = op.u64(AttrKey::BlockSize, 64U);
    if (block < 32U || block > 256U || (block & (block - 1U)) != 0U)
      fail("initial fused attention requires a power-of-two block in [32,256]");
    return;
  }

  if (op.opcode == Opcode::BiasAdd) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &bias = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        bias.dtype != input.dtype || bias.dims.size() != 1U ||
        bias.dims[0] != input.dims.back())
      fail("bias_add requires float input/output and matching final-width bias");
    return;
  }

  if (op.opcode == Opcode::H3AdaLNSelect) {
    expect_counts(op, 2, 6);
    const auto &projected = tensor_or_fail(program, op.inputs[0], op);
    const auto &indices = tensor_or_fail(program, op.inputs[1], op);
    if (!supported_float(projected.dtype) || projected.dims.size() != 2U ||
        indices.dtype != DType::I32 || indices.dims.size() != 1U)
      fail("h3_adaln_select requires projected [T,18H] and i32 indices [S]");
    const auto &first = tensor_or_fail(program, op.outputs[0], op);
    if (first.dtype != projected.dtype || first.dims.size() != 2U ||
        first.dims[0] != indices.dims[0] ||
        projected.dims[1] != 18U * first.dims[1])
      fail("h3_adaln_select output shape is inconsistent");
    for (const auto output : op.outputs)
      same_shape_dtype(first, tensor_or_fail(program, output, op), op);
    return;
  }

  if (op.opcode == Opcode::H3DeinterleaveQkv) {
    expect_counts(op, 1, 3);
    const auto &packed = tensor_or_fail(program, op.inputs[0], op);
    const auto &q = tensor_or_fail(program, op.outputs[0], op);
    const auto &k = tensor_or_fail(program, op.outputs[1], op);
    const auto &v = tensor_or_fail(program, op.outputs[2], op);
    same_shape_dtype(q, k, op);
    same_shape_dtype(q, v, op);
    if (!supported_float(packed.dtype) || q.dtype != packed.dtype ||
        packed.dims.size() != 2U || q.dims.size() != 3U ||
        packed.dims[0] != q.dims[0] ||
        packed.dims[1] != 3U * q.dims[1] * q.dims[2] ||
        op.u64(AttrKey::Heads, q.dims[1]) != q.dims[1] ||
        op.u64(AttrKey::HeadDim, q.dims[2]) != q.dims[2])
      fail("h3_deinterleave_qkv shape/attributes are inconsistent");
    return;
  }

  if (op.opcode == Opcode::H3DeinterleaveQkvWeight) {
    expect_counts(op, 1, 3);
    const auto &packed = tensor_or_fail(program, op.inputs[0], op);
    const auto &q = tensor_or_fail(program, op.outputs[0], op);
    const auto &k = tensor_or_fail(program, op.outputs[1], op);
    const auto &v = tensor_or_fail(program, op.outputs[2], op);
    same_shape_dtype(q, k, op);
    same_shape_dtype(q, v, op);
    if (!supported_float(packed.dtype) || q.dtype != packed.dtype ||
        packed.dims.size() != 2U || q.dims.size() != 2U ||
        packed.dims[0] != 3U * q.dims[0] || packed.dims[1] != q.dims[1])
      fail("h3_deinterleave_qkv_weight requires [3N,K] -> three [N,K] tensors");
    const auto heads = op.u64(AttrKey::Heads, 0U);
    const auto head_dim = op.u64(AttrKey::HeadDim, 0U);
    if (heads == 0U || head_dim == 0U || q.dims[0] != heads * head_dim)
      fail("h3_deinterleave_qkv_weight head geometry disagrees with N");
    return;
  }

  if (op.opcode == Opcode::DequantizeInt4) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 4U) ||
        op.outputs.size() != 1U)
      fail("dequantize_int4 expects packed, scales, optional outlier index "
           "and residual, and one output");
    const auto &packed = tensor_or_fail(program, op.inputs[0], op);
    const auto &scales = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto group = op.u64(AttrKey::GroupSize, 64U);
    if (packed.dtype != DType::I8 || packed.dims.size() != 2U ||
        !supported_float(scales.dtype) || scales.dtype != out.dtype ||
        scales.dims.size() != 2U || out.dims.size() != 2U ||
        out.dims[1] % 2U != 0U || packed.dims[0] != out.dims[0] ||
        packed.dims[1] != out.dims[1] / 2U || group < 16U || group > 256U ||
        (group & (group - 1U)) != 0U || out.dims[1] % group != 0U ||
        scales.dims[0] != out.dims[0] ||
        scales.dims[1] != out.dims[1] / group)
      fail("dequantize_int4 requires i8-packed [N,K/2], float scales "
           "[N,K/group], and float output [N,K]");
    if (op.inputs.size() == 4U) {
      const auto &indices = tensor_or_fail(program, op.inputs[2], op);
      const auto &residuals = tensor_or_fail(program, op.inputs[3], op);
      if (indices.dtype != DType::I8 || indices.dims != scales.dims ||
          residuals.dtype != out.dtype || residuals.dims != scales.dims)
        fail("dequantize_int4 outlier correction requires i8 indices and "
             "float residuals shaped like the scales");
    }
    return;
  }

  if (op.opcode == Opcode::DequantizeInt5) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("dequantize_int5 expects packed, group scales, optional column "
           "scales, and one output");
    const auto &packed = tensor_or_fail(program, op.inputs[0], op);
    const auto &scales = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto group = op.u64(AttrKey::GroupSize, 64U);
    if (packed.dtype != DType::I8 || packed.dims.size() != 2U ||
        !supported_float(scales.dtype) || scales.dtype != out.dtype ||
        scales.dims.size() != 2U || out.dims.size() != 2U ||
        out.dims[1] % 8U != 0U || packed.dims[0] != out.dims[0] ||
        packed.dims[1] != out.dims[1] * 5U / 8U || group < 16U ||
        group > 256U || (group & (group - 1U)) != 0U ||
        out.dims[1] % group != 0U || scales.dims[0] != out.dims[0] ||
        scales.dims[1] != out.dims[1] / group)
      fail("dequantize_int5 requires bit-packed [N,5K/8], float scales "
           "[N,K/group], and float output [N,K]");
    if (op.inputs.size() == 3U) {
      const auto &columns = tensor_or_fail(program, op.inputs[2], op);
      if (columns.dtype != out.dtype || columns.dims.size() != 1U ||
          columns.dims[0] != out.dims[1])
        fail("dequantize_int5 column scales must match output dtype and [K]");
    }
    return;
  }

  // DiT backward opcodes (flame backward-equation port).  The gradient of a
  // tensor carries the dtype of its forward tensor (flame BF16_GRAD_DECISION
  // Option A); every cross-element reduction inside these kernels is an F32
  // accumulator, so an explicit AccumulatorDType must name F32 (fail-closed
  // via check_accumulator_f32).

  if (op.opcode == Opcode::RmsNormBackward) {
    if (op.inputs.size() != 3U || op.outputs.empty() || op.outputs.size() > 2U)
      fail("rms_norm_backward expects grad_output, input, weight and one or "
           "two outputs");
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &weight = tensor_or_fail(program, op.inputs[2], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back())
      fail("rms_norm_backward requires float tensors and weight matching the "
           "final dimension");
    if (op.outputs.size() == 2U) {
      const auto &grad_weight = tensor_or_fail(program, op.outputs[1], op);
      if (grad_weight.dtype != weight.dtype || grad_weight.dims != weight.dims)
        fail("rms_norm_backward weight gradient must match the weight");
    }
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("rms_norm_backward epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("rms_norm_backward block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::RmsNormModulateBackward) {
    const bool weighted = op.inputs.size() == 4U;
    if ((op.inputs.size() != 3U && op.inputs.size() != 4U) ||
        op.outputs.size() != op.inputs.size())
      fail("rms_norm_modulate_backward expects grad_output, x, [weight], "
           "scale and dx, dscale, dshift[, dweight]");
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &x = tensor_or_fail(program, op.inputs[1], op);
    const auto &scale = tensor_or_fail(program, op.inputs[weighted ? 3U : 2U], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_scale = tensor_or_fail(program, op.outputs[1], op);
    const auto &grad_shift = tensor_or_fail(program, op.outputs[2], op);
    same_shape_dtype(x, grad_output, op);
    same_shape_dtype(x, scale, op);
    same_shape_dtype(x, grad_input, op);
    same_shape_dtype(x, grad_scale, op);
    same_shape_dtype(x, grad_shift, op);
    check_accumulator_f32(op);
    if (!supported_float(x.dtype) || x.dims.size() != 2U)
      fail("rms_norm_modulate_backward requires rank-2 float tensors");
    if (weighted) {
      const auto &weight = tensor_or_fail(program, op.inputs[2], op);
      const auto &grad_weight = tensor_or_fail(program, op.outputs[3], op);
      if (weight.dtype != x.dtype || weight.dims.size() != 1U ||
          weight.dims[0] != x.dims[1] || grad_weight.dtype != weight.dtype ||
          grad_weight.dims != weight.dims)
        fail("rms_norm_modulate_backward weight and its gradient must match "
             "[hidden]");
    }
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("rms_norm_modulate_backward epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("rms_norm_modulate_backward block size must be a power of two in "
           "[32,1024]");
    return;
  }

  if (op.opcode == Opcode::SwiGluBackward) {
    expect_counts(op, 2, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype) || grad_output.dtype != input.dtype ||
        input.dims.empty() ||
        grad_output.dims.size() != input.dims.size())
      fail("swiglu_backward requires uniform float tensors");
    auto expected = grad_output.dims;
    expected.back() *= 2U;
    if (input.dims != expected)
      fail("swiglu_backward input final dimension must be twice the "
           "grad_output final dimension");
    return;
  }

  if (op.opcode == Opcode::ResidualGateBackward) {
    expect_counts(op, 3, 2);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &branch = tensor_or_fail(program, op.inputs[1], op);
    const auto &gate = tensor_or_fail(program, op.inputs[2], op);
    const auto &grad_branch = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_gate = tensor_or_fail(program, op.outputs[1], op);
    same_shape_dtype(grad_output, branch, op);
    same_shape_dtype(grad_output, gate, op);
    same_shape_dtype(grad_output, grad_branch, op);
    same_shape_dtype(grad_output, grad_gate, op);
    check_accumulator_f32(op);
    if (!supported_float(grad_output.dtype))
      fail("residual_gate_backward admits f32, bf16, or f16");
    return;
  }

  if (op.opcode == Opcode::LayerNormBackward) {
    expect_counts(op, 3, 3);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &weight = tensor_or_fail(program, op.inputs[2], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_weight = tensor_or_fail(program, op.outputs[1], op);
    const auto &grad_bias = tensor_or_fail(program, op.outputs[2], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back() ||
        grad_weight.dtype != weight.dtype || grad_weight.dims != weight.dims ||
        grad_bias.dtype != weight.dtype || grad_bias.dims != weight.dims)
      fail("layer_norm_backward requires float tensors with affine "
           "gradients matching the final dimension");
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("layer_norm_backward epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("layer_norm_backward block size must be a power of two in "
           "[32,1024]");
    return;
  }

  if (op.opcode == Opcode::QkNormPartialRopeBackward) {
    // Backward of the fused per-head RMSNorm + partial halfsplit rotation.
    // The rotation layout (RotaryDim, table width) is explicit in the op
    // semantics, never inferred from shapes (flame's shape-sniff trap: the
    // HiDream-O1 Q/K LoRA-B gradient collapse).
    if (op.inputs.size() != 5U || op.outputs.empty() || op.outputs.size() > 2U)
      fail("qk_norm_partial_rope_backward expects grad_output, input, "
           "weight, cos, sin and one or two outputs");
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &weight = tensor_or_fail(program, op.inputs[2], op);
    const auto &cos = tensor_or_fail(program, op.inputs[3], op);
    const auto &sin = tensor_or_fail(program, op.inputs[4], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype) || input.dims.size() != 3U ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims[2] || cos.dtype != input.dtype ||
        sin.dtype != input.dtype || cos.dims != sin.dims ||
        cos.dims.size() != 2U || cos.dims[0] != input.dims[0])
      fail("qk_norm_partial_rope_backward requires grad/input [S,H,D], "
           "weight [D], cos/sin [S,T]");
    const auto head_dim = input.dims[2];
    const auto rotary = op.u64(AttrKey::RotaryDim, head_dim);
    if (rotary == 0U || rotary > head_dim || (rotary % 2U) != 0U ||
        (cos.dims[1] != rotary && cos.dims[1] * 2U != rotary))
      fail("qk_norm_partial_rope_backward rotary geometry is inconsistent");
    if (op.outputs.size() == 2U) {
      const auto &grad_weight = tensor_or_fail(program, op.outputs[1], op);
      if (grad_weight.dtype != weight.dtype || grad_weight.dims != weight.dims)
        fail("qk_norm_partial_rope_backward weight gradient must match the "
             "weight");
    }
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("qk_norm_partial_rope_backward epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("qk_norm_partial_rope_backward block size must be a power of two "
           "in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::AttentionLse) {
    // Per-(query,head) max-shifted logsumexp of the attention scores.  A
    // cross-row reduction/accumulator, so the output is pinned F32
    // regardless of the Q/K storage dtype (flame dtype contract).
    expect_counts(op, 2, 1);
    const auto &q = tensor_or_fail(program, op.inputs[0], op);
    const auto &k = tensor_or_fail(program, op.inputs[1], op);
    const auto &lse = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    if (!supported_float(q.dtype) || q.dims.size() != 3U)
      fail("attention_lse requires f32, bf16, or f16 [S,H,D] inputs");
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[1]);
    if (kv_heads == 0U || q.dims[1] % kv_heads != 0U)
      fail("attention_lse KvHeads must be nonzero and divide the query head "
           "count");
    if (k.dtype != q.dtype || k.dims.size() != 3U ||
        k.dims[0] != q.dims[0] || k.dims[1] != kv_heads ||
        k.dims[2] != q.dims[2])
      fail("attention_lse k must be [S,KvHeads,D] with the query dtype");
    if (lse.dtype != DType::F32 || lse.dims.size() != 2U ||
        lse.dims[0] != q.dims[0] || lse.dims[1] != q.dims[1])
      fail("attention_lse output must be F32 [S,H]");
    if (q.dims[0] > 4096U)
      fail("decomposed attention backward is admitted only for S<=4096");
    return;
  }

  if (op.opcode == Opcode::AttentionBackward) {
    // Decomposed recompute backward (flame section 2c): P is recomputed in
    // F32 from Q,K and the saved logsumexp; delta = rowsum(dO*O) uses the
    // forward output taken by DIRECT tensor id (flame's saved-O identity
    // trap: a shape-found O destroyed the dO.O^T identity -> grad_norm=inf).
    expect_counts(op, 6, 3);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &q = tensor_or_fail(program, op.inputs[1], op);
    const auto &k = tensor_or_fail(program, op.inputs[2], op);
    const auto &v = tensor_or_fail(program, op.inputs[3], op);
    const auto &forward_output = tensor_or_fail(program, op.inputs[4], op);
    const auto &lse = tensor_or_fail(program, op.inputs[5], op);
    const auto &grad_q = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_k = tensor_or_fail(program, op.outputs[1], op);
    const auto &grad_v = tensor_or_fail(program, op.outputs[2], op);
    same_shape_dtype(q, grad_output, op);
    same_shape_dtype(q, forward_output, op);
    same_shape_dtype(q, grad_q, op);
    same_shape_dtype(k, v, op);
    same_shape_dtype(k, grad_k, op);
    same_shape_dtype(k, grad_v, op);
    check_accumulator_f32(op);
    if (!supported_float(q.dtype) || q.dims.size() != 3U)
      fail("attention_backward requires f32, bf16, or f16 [S,H,D] tensors");
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[1]);
    if (kv_heads == 0U || q.dims[1] % kv_heads != 0U)
      fail("attention_backward KvHeads must be nonzero and divide the query "
           "head count");
    if (k.dtype != q.dtype || k.dims.size() != 3U ||
        k.dims[0] != q.dims[0] || k.dims[1] != kv_heads ||
        k.dims[2] != q.dims[2])
      fail("attention_backward k/v must be [S,KvHeads,D] with the query "
           "dtype");
    if (lse.dtype != DType::F32 || lse.dims.size() != 2U ||
        lse.dims[0] != q.dims[0] || lse.dims[1] != q.dims[1])
      fail("attention_backward saved logsumexp must be F32 [S,H]");
    if (q.dims[0] > 4096U)
      fail("decomposed attention backward is admitted only for S<=4096");
    return;
  }

  if (op.opcode == Opcode::Conv1d) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("conv1d expects input, weight, optional bias, and one output");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || weight.dtype != input.dtype ||
        out.dtype != input.dtype || input.dims.size() != 3U ||
        weight.dims.size() != 3U || out.dims.size() != 3U)
      fail("conv1d requires rank-3 [B,C,L] float input/weight/output of one "
           "dtype");
    const auto stride = op.u64(AttrKey::Stride, 1U);
    const auto dilation = op.u64(AttrKey::Dilation, 1U);
    const auto groups = op.u64(AttrKey::Groups, 1U);
    const auto pad_left = op.u64(AttrKey::PadLeft, 0U);
    const auto pad_right = op.u64(AttrKey::PadRight, 0U);
    const auto pad_mode = op.u64(AttrKey::PadMode, 0U);
    const auto transposed = op.boolean(AttrKey::Transposed, false);
    const auto trim_left = op.u64(AttrKey::TrimLeft, 0U);
    const auto trim_right = op.u64(AttrKey::TrimRight, 0U);
    const auto batch = input.dims[0];
    const auto in_channels = input.dims[1];
    const auto length = input.dims[2];
    const auto kernel = weight.dims[2];
    // Bounds that keep every length expression below inside uint64.
    constexpr auto kShortLimit = std::uint64_t{1} << 16U;
    constexpr auto kLongLimit = std::uint64_t{1} << 40U;
    if (stride == 0U || dilation == 0U || groups == 0U || kernel == 0U ||
        pad_mode > 1U || stride > kShortLimit || dilation > kShortLimit ||
        kernel > kShortLimit || pad_left > kLongLimit ||
        pad_right > kLongLimit || trim_left > kLongLimit ||
        trim_right > kLongLimit || length > kLongLimit)
      fail("conv1d attribute geometry is invalid or out of range");
    if (in_channels % groups != 0U)
      fail("conv1d groups must divide the input channels");
    const auto padded = length + pad_left + pad_right;
    std::uint64_t out_channels = 0U;
    std::uint64_t expected_length = 0U;
    if (transposed) {
      if (dilation != 1U)
        fail("conv1d transposed mode requires dilation 1");
      // ConvTranspose1d checkpoint layout: [C_in, C_out/groups, K] — dim 0
      // is the INPUT channel. The swap against the forward layout is the
      // classic port bug; both directions are checked fail-closed.
      if (weight.dims[0] != in_channels || weight.dims[1] == 0U)
        fail("conv1d transposed weight must be [C_in, C_out/groups, K]");
      out_channels = weight.dims[1] * groups;
      if (padded == 0U)
        fail("conv1d transposed input is empty after padding");
      const auto full = (padded - 1U) * stride + kernel;
      if (trim_left + trim_right >= full)
        fail("conv1d transposed trim removes the whole output");
      expected_length = full - trim_left - trim_right;
    } else {
      if (trim_left != 0U || trim_right != 0U)
        fail("conv1d trim attributes are transposed-only");
      // Forward layout: [C_out, C_in/groups, K].
      if (weight.dims[1] != in_channels / groups || weight.dims[0] == 0U)
        fail("conv1d weight must be [C_out, C_in/groups, K]");
      out_channels = weight.dims[0];
      const auto effective = dilation * (kernel - 1U) + 1U;
      if (padded < effective)
        fail("conv1d kernel does not fit the padded input");
      expected_length = (padded - effective) / stride + 1U;
    }
    if (out_channels % groups != 0U)
      fail("conv1d groups must divide the output channels");
    if (out.dims[0] != batch || out.dims[1] != out_channels ||
        out.dims[2] != expected_length)
      fail("conv1d output geometry does not match its attributes");
    if (op.inputs.size() == 3U) {
      const auto &bias = tensor_or_fail(program, op.inputs[2], op);
      if (bias.dtype != input.dtype || bias.dims.size() != 1U ||
          bias.dims[0] != out_channels)
        fail("conv1d bias must be a [C_out] vector of the input dtype");
    }
    return;
  }

  if (op.opcode == Opcode::SnakeBeta) {
    expect_counts(op, 3, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &alpha = tensor_or_fail(program, op.inputs[1], op);
    const auto &beta = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.size() != 3U ||
        alpha.dtype != input.dtype || alpha.dims.size() != 1U ||
        alpha.dims[0] != input.dims[1] || beta.dtype != alpha.dtype ||
        beta.dims != alpha.dims)
      fail("snake_beta requires rank-3 [B,C,L] input and [C] log-space "
           "alpha/beta vectors");
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-9);
    if (!(epsilon > 0.0))
      fail("snake_beta epsilon must be positive");
    return;
  }

  fail("unsupported DiffIR operation");
}

} // namespace

void verify(const Program &program) {
  if (program.version != kVersion)
    fail("unsupported DiffIR program version");
  if (program.tensors.empty())
    fail("DiffIR program has no tensors");

  std::unordered_set<std::uint32_t> tensor_ids;
  for (const auto &tensor : program.tensors) {
    if (tensor.id == 0 || !tensor_ids.insert(tensor.id).second)
      fail("DiffIR tensor ids must be unique and nonzero");
    if (!valid_dtype(tensor.dtype) || tensor.dims.empty() || tensor.dims.size() > kMaxRank)
      fail("DiffIR tensor has invalid dtype or rank");
    constexpr std::uint32_t valid_roles =
        TensorRole::Input | TensorRole::Output | TensorRole::Constant |
        TensorRole::Streamed | TensorRole::Parameter |
        TensorRole::OptimizerState;
    if ((tensor.roles & ~valid_roles) != 0U)
      fail("DiffIR tensor has unknown role flags");
    if (tensor.has_role(TensorRole::Streamed) &&
        !tensor.has_role(TensorRole::Constant))
      fail("DiffIR streamed tensor must also be constant");
    if (tensor.has_role(TensorRole::Parameter) &&
        (!tensor.has_role(TensorRole::Input) &&
         !tensor.has_role(TensorRole::Output)))
      fail("DiffIR parameter must be an input or output");
    if (tensor.has_role(TensorRole::OptimizerState) &&
        (!tensor.has_role(TensorRole::Input) &&
         !tensor.has_role(TensorRole::Output)))
      fail("DiffIR optimizer state must be an input or output");
    if ((tensor.has_role(TensorRole::Parameter) ||
         tensor.has_role(TensorRole::OptimizerState)) &&
        tensor.has_role(TensorRole::Constant))
      fail("DiffIR mutable training state cannot be constant");
    (void)tensor.byte_count();
  }

  std::unordered_set<std::uint32_t> operation_ids;
  std::unordered_map<std::uint32_t, std::uint32_t> writers;
  std::unordered_set<std::uint32_t> available;
  for (const auto &tensor : program.tensors) {
    if (tensor.has_role(TensorRole::Input) || tensor.has_role(TensorRole::Constant))
      available.insert(tensor.id);
  }

  for (const auto &op : program.operations) {
    if (op.id == 0 || !operation_ids.insert(op.id).second)
      fail("DiffIR operation ids must be unique and nonzero");
    if (!valid_opcode(op.opcode))
      fail("DiffIR operation has unknown opcode");
    for (const auto input : op.inputs) {
      if (!available.contains(input))
        fail("DiffIR operation consumes an unavailable tensor");
    }
    for (const auto output : op.outputs) {
      if (!program.tensor(output))
        fail("DiffIR operation produces an undeclared tensor");
      if (!writers.emplace(output, op.id).second)
        fail("DiffIR tensor has multiple writers");
      if (program.tensor(output)->has_role(TensorRole::Input) ||
          program.tensor(output)->has_role(TensorRole::Constant))
        fail("DiffIR operation cannot overwrite input or constant tensor");
      available.insert(output);
    }
    verify_operation(program, op);
  }

  for (const auto &tensor : program.tensors) {
    if (tensor.has_role(TensorRole::Output) && !available.contains(tensor.id))
      fail("DiffIR output tensor is unavailable");
  }
}

} // namespace dif::ir
