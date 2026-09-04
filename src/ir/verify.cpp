#include "dif/ir/verify.hpp"

#include "dif/support/error.hpp"

#include <array>
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
         dtype == DType::I8 || dtype == DType::I32 || dtype == DType::Bool ||
         dtype == DType::FP8E4M3 || dtype == DType::FP8E8M0;
}

bool supported_float(DType dtype) {
  return dtype == DType::F32 || dtype == DType::BF16 || dtype == DType::F16;
}

bool valid_opcode(Opcode opcode) {
  return opcode_is_registered(static_cast<std::uint32_t>(opcode));
}

bool valid_attr_key(AttrKey key) {
  return key >= AttrKey::Epsilon && key < AttrKey::EndSentinel_;
}

bool valid_attr_kind(AttrKind kind) {
  return kind >= AttrKind::U64 && kind < AttrKind::EndSentinel_;
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
    fail("DiffIR op " + std::to_string(op.id) + " (" +
         std::string(opcode_name(op.opcode)) +
         ") requires equal shape/dtype");
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

  if (op.opcode == Opcode::Gelu) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype))
      fail("gelu semantics admit f32, bf16, or f16");
    const auto *approximation = op.find(AttrKey::Approximation);
    if (approximation == nullptr || approximation->kind != AttrKind::U64 ||
        (approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::Tanh) &&
         approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::ExactErf) &&
         approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::QuickSigmoid)))
      fail("gelu requires an explicit tanh, exact-erf, or quick-sigmoid "
           "approximation");
    return;
  }

  if (op.opcode == Opcode::Sigmoid) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype))
      fail("sigmoid semantics admit f32, bf16, or f16");
    return;
  }

  if (op.opcode == Opcode::Reshape) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        input.element_count() != out.element_count())
      fail("reshape requires equal-count float input/output tensors");
    return;
  }

  if (op.opcode == Opcode::BroadcastTo) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        input.dims.size() > out.dims.size())
      fail("broadcast_to requires same-dtype float tensors and output rank >= input rank");
    const auto pad = out.dims.size() - input.dims.size();
    for (std::size_t axis = 0; axis < input.dims.size(); ++axis) {
      const auto source = input.dims[axis];
      const auto destination = out.dims[pad + axis];
      if (source != 1U && source != destination)
        fail("broadcast_to input dimensions must be one or match output");
    }
    return;
  }

  if (op.opcode == Opcode::Slice) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto *axis_attribute = op.find(AttrKey::Axis);
    const auto *start_attribute = op.find(AttrKey::Start);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        input.dims.size() != out.dims.size() || !axis_attribute ||
        !start_attribute || axis_attribute->kind != AttrKind::U64 ||
        start_attribute->kind != AttrKind::U64)
      fail("slice requires equal-rank float tensors and u64 Axis/Start");
    const auto axis = axis_attribute->as_u64();
    const auto start = start_attribute->as_u64();
    if (axis >= input.dims.size() || start > input.dims[axis] ||
        out.dims[axis] > input.dims[axis] - start)
      fail("slice axis or interval is outside the input");
    for (std::size_t index = 0; index < input.dims.size(); ++index)
      if (index != axis && input.dims[index] != out.dims[index])
        fail("slice may change only its selected axis");
    return;
  }

  if (op.opcode == Opcode::RotaryFrequency) {
    expect_counts(op, 4, 2);
    const auto &positions = tensor_or_fail(program, op.inputs[0], op);
    const auto &pair_axes = tensor_or_fail(program, op.inputs[1], op);
    const auto &pair_indices = tensor_or_fail(program, op.inputs[2], op);
    const auto &axis_dims = tensor_or_fail(program, op.inputs[3], op);
    const auto &cosine = tensor_or_fail(program, op.outputs[0], op);
    const auto &sine = tensor_or_fail(program, op.outputs[1], op);
    if (positions.dtype != DType::F32 || positions.dims.size() != 3U ||
        pair_axes.dtype != DType::I32 || pair_axes.dims.size() != 1U ||
        pair_indices.dtype != DType::I32 || pair_indices.dims != pair_axes.dims ||
        axis_dims.dtype != DType::I32 || axis_dims.dims.size() != 1U ||
        cosine.dtype != DType::F32 || sine.dtype != DType::F32 ||
        cosine.dims != sine.dims || cosine.dims.size() != 3U ||
        cosine.dims[0] != positions.dims[0] ||
        cosine.dims[1] != positions.dims[1] ||
        cosine.dims[2] != pair_axes.dims[0] ||
        positions.dims[2] != axis_dims.dims[0])
      fail("rotary_frequency requires positions [B,L,A], pair maps [P], axis dims [A], and f32 outputs [B,L,P]");
    if (!(op.f64(AttrKey::Theta, 10000.0) > 0.0) ||
        !(op.f64(AttrKey::Ntk, 1.0) > 0.0))
      fail("rotary_frequency theta and ntk must be positive");
    return;
  }

  if (op.opcode == Opcode::RotaryApply) {
    expect_counts(op, 3, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &cosine = tensor_or_fail(program, op.inputs[1], op);
    const auto &sine = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.size() != 4U ||
        cosine.dtype != DType::F32 || sine.dtype != DType::F32 ||
        cosine.dims != sine.dims || cosine.dims.size() != 3U ||
        cosine.dims[0] != input.dims[0] ||
        cosine.dims[1] != input.dims[1] ||
        cosine.dims[2] * 2U > input.dims[3] ||
        (op.u64(AttrKey::RotaryLayout, 0U) !=
             static_cast<std::uint64_t>(RotaryLayout::Interleaved) &&
         op.u64(AttrKey::RotaryLayout, 0U) !=
             static_cast<std::uint64_t>(RotaryLayout::HalfSplit)))
      fail("rotary_apply requires [B,L,H,D], f32 cos/sin [B,L,P], and an explicit interleaved or half-split layout");
    return;
  }

  if (op.opcode == Opcode::BooleanMaskToBias) {
    expect_counts(op, 1, 1);
    const auto &mask = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const bool vector_mask = mask.dims.size() == 2U;
    const bool matrix_mask = mask.dims.size() == 3U;
    if (mask.dtype != DType::Bool || (!vector_mask && !matrix_mask) ||
        !supported_float(out.dtype) || out.dims.size() != 4U ||
        out.dims[0] != mask.dims[0] || out.dims[1] != 1U ||
        out.dims[2] != (vector_mask ? mask.dims[1] : mask.dims[1]) ||
        out.dims[3] != (vector_mask ? mask.dims[1] : mask.dims[2]) ||
        (matrix_mask && mask.dims[1] != mask.dims[2]))
      fail("boolean_mask_to_bias requires bool [B,L] or [B,L,L] and float [B,1,L,L]");
    (void)op.boolean(AttrKey::MaskQueries, true);
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

  if (op.opcode == Opcode::ClampBackward) {
    expect_counts(op, 2, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    const auto lower = op.f64(AttrKey::Lower,
                              -std::numeric_limits<double>::infinity());
    const auto upper = op.f64(AttrKey::Upper,
                              std::numeric_limits<double>::infinity());
    if (!supported_float(input.dtype) || !(lower <= upper))
      fail("clamp_backward requires float tensors and lower <= upper");
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

  if (op.opcode == Opcode::GeluBackward) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype))
      fail("gelu_backward admits f32, bf16, or f16 tensors");
    // The derivative has to be the derivative of the closed form that ran,
    // so the approximation is as explicit here as it is on the forward op.
    const auto *approximation = op.find(AttrKey::Approximation);
    if (approximation == nullptr || approximation->kind != AttrKind::U64 ||
        (approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::Tanh) &&
         approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::ExactErf) &&
         approximation->bits !=
             static_cast<std::uint64_t>(GeluApproximation::QuickSigmoid)))
      fail("gelu_backward requires an explicit tanh, exact-erf, or "
           "quick-sigmoid approximation");
    return;
  }

  if (op.opcode == Opcode::UpsampleNearest2dBackward) {
    expect_counts(op, 1, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    const auto scale_h = op.u64(AttrKey::ScaleH, 1U);
    const auto scale_w = op.u64(AttrKey::ScaleW, 1U);
    check_accumulator_f32(op);
    if (!supported_float(grad_input.dtype) ||
        grad_output.dtype != grad_input.dtype ||
        grad_input.dims.size() != 4U || grad_output.dims.size() != 4U ||
        scale_h == 0U || scale_w == 0U ||
        grad_output.dims[0] != grad_input.dims[0] ||
        grad_output.dims[1] != grad_input.dims[1] ||
        grad_output.dims[2] != grad_input.dims[2] * scale_h ||
        grad_output.dims[3] != grad_input.dims[3] * scale_w)
      fail("upsample_nearest_2d_backward requires NCHW float tensors whose "
           "gradient is the input scaled by ScaleH and ScaleW");
    return;
  }

  if (op.opcode == Opcode::SliceBackward) {
    expect_counts(op, 1, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    const auto *axis_attribute = op.find(AttrKey::Axis);
    const auto *start_attribute = op.find(AttrKey::Start);
    check_accumulator_f32(op);
    if (!supported_float(grad_input.dtype) ||
        grad_output.dtype != grad_input.dtype || axis_attribute == nullptr ||
        start_attribute == nullptr ||
        grad_output.dims.size() != grad_input.dims.size() ||
        grad_input.dims.empty() ||
        axis_attribute->as_u64() >= grad_input.dims.size())
      fail("slice_backward requires same-rank float tensors and an in-range "
           "axis");
    const auto axis = static_cast<std::size_t>(axis_attribute->as_u64());
    const auto start = start_attribute->as_u64();
    for (std::size_t dimension = 0U; dimension < grad_input.dims.size();
         ++dimension)
      if (dimension != axis &&
          grad_output.dims[dimension] != grad_input.dims[dimension])
        fail("slice_backward may differ only on the sliced axis");
    if (grad_output.dims[axis] > grad_input.dims[axis] ||
        start > grad_input.dims[axis] - grad_output.dims[axis])
      fail("slice_backward window is outside the input axis");
    return;
  }

  if (op.opcode == Opcode::BroadcastToBackward) {
    expect_counts(op, 1, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    if (!supported_float(grad_input.dtype) ||
        grad_output.dtype != grad_input.dtype ||
        grad_input.dims.size() > grad_output.dims.size())
      fail("broadcast_to_backward requires same-dtype float tensors and a "
           "source rank no greater than the gradient rank");
    const auto pad = grad_output.dims.size() - grad_input.dims.size();
    for (std::size_t axis = 0U; axis < grad_input.dims.size(); ++axis) {
      const auto source = grad_input.dims[axis];
      const auto destination = grad_output.dims[pad + axis];
      if (source != 1U && source != destination)
        fail("broadcast_to_backward source dimensions must be one or match "
             "the gradient");
    }
    return;
  }

  if (op.opcode == Opcode::GroupNormBackward) {
    if (op.inputs.size() != 3U || op.outputs.size() != 1U)
      fail("group_norm_backward expects x, weight, grad_output and the input "
           "gradient");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[2], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    if (!supported_float(input.dtype) ||
        (input.dims.size() != 4U && input.dims.size() != 5U) ||
        weight.dtype != input.dtype ||
        weight.dims != std::vector<std::uint64_t>{input.dims[1]})
      fail("group_norm_backward requires rank 4 or 5 float x/out and a "
           "[channels] weight");
    const auto groups = op.u64(AttrKey::Groups, 1U);
    if (groups == 0U || input.dims[1] % groups != 0U ||
        !(op.f64(AttrKey::Epsilon, 1.0e-5) > 0.0))
      fail("group_norm_backward has invalid normalization geometry");
    return;
  }

  if (op.opcode == Opcode::GroupNormBackwardAffine) {
    expect_counts(op, 2, 2);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_weight = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_bias = tensor_or_fail(program, op.outputs[1], op);
    check_accumulator_f32(op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(grad_weight, grad_bias, op);
    if (!supported_float(input.dtype) ||
        (input.dims.size() != 4U && input.dims.size() != 5U) ||
        grad_weight.dtype != input.dtype ||
        grad_weight.dims != std::vector<std::uint64_t>{input.dims[1]})
      fail("group_norm_backward_affine requires rank 4 or 5 float x and "
           "[channels] gradients");
    const auto groups = op.u64(AttrKey::Groups, 1U);
    if (groups == 0U || input.dims[1] % groups != 0U ||
        !(op.f64(AttrKey::Epsilon, 1.0e-5) > 0.0))
      fail("group_norm_backward_affine has invalid normalization geometry");
    return;
  }

  if (op.opcode == Opcode::SigmoidBackward) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype))
      fail("sigmoid_backward admits f32, bf16, or f16 tensors");
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
    const auto clip_scale = op.f64(AttrKey::ClipScale, 1.0);
    if (step.dtype != DType::I32 ||
        step.dims != std::vector<std::uint64_t>{1U} ||
        !(learning_rate > 0.0) || beta1 < 0.0 || beta1 >= 1.0 ||
        beta2 < 0.0 || beta2 >= 1.0 || !(epsilon > 0.0) ||
        weight_decay < 0.0 || !(clip_scale > 0.0) || clip_scale > 1.0)
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
    if (!std::isfinite(op.f64(AttrKey::WeightOffset, 0.0)))
      fail("rms_norm weight offset must be finite");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("rms_norm block size must be a power of two in [32,1024]");
    const auto implementation = op.u64(AttrKey::Implementation, 1U);
    if (implementation != 1U && implementation != 2U)
      fail("rms_norm implementation must be 1 (generated) or 2 (vectorized "
           "Welford)");
    if (implementation == 2U &&
        (input.dtype != DType::BF16 || input.dims.back() != 128U ||
         block != 128U))
      fail("vectorized Welford rms_norm requires bf16 width 128 and block "
           "size 128");
    const auto reduction_tile = op.u64(AttrKey::ReductionTileSize, 0U);
    if (reduction_tile != 0U && reduction_tile != 2048U &&
        reduction_tile != 8192U)
      fail("rms_norm reduction tile must be zero, 2048, or 8192");
    if (reduction_tile != 0U &&
        (block != 512U || input.dims.back() != 6144U))
      fail("tiled rms_norm currently requires block 512 and width 6144");
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

  if (op.opcode == Opcode::GatherRowsBackward) {
    expect_counts(op, 2, 1);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &indices = tensor_or_fail(program, op.inputs[1], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    check_accumulator_f32(op);
    // The table's own row count is free -- it is whatever the forward
    // gathered from -- but everything else has to line up with the gather.
    if (!supported_float(grad_input.dtype) || grad_input.dims.size() < 2U ||
        indices.dtype != DType::I32 || indices.dims.size() != 1U ||
        grad_output.dtype != grad_input.dtype ||
        grad_output.dims.size() != grad_input.dims.size() ||
        grad_output.dims[0] != indices.dims[0] ||
        !std::equal(grad_input.dims.begin() + 1, grad_input.dims.end(),
                    grad_output.dims.begin() + 1))
      fail("gather_rows_backward requires float [M,...], i32 [M], and float "
           "[S,...] sharing the row width");
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

  if (op.opcode == Opcode::EulerVelocityStep) {
    expect_counts(op, 4, 1);
    const auto &sample = tensor_or_fail(program, op.inputs[0], op);
    const auto &velocity = tensor_or_fail(program, op.inputs[1], op);
    const auto &current = tensor_or_fail(program, op.inputs[2], op);
    const auto &next = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(sample, velocity, op);
    same_shape_dtype(sample, out, op);
    if (!supported_float(sample.dtype) || current.dtype != DType::F32 ||
        current.dims != std::vector<std::uint64_t>{1U} ||
        next.dtype != DType::F32 || next.dims != current.dims)
      fail("euler_velocity_step requires equal float sample/velocity/output "
           "and f32 [1] current/next timesteps");
    return;
  }

  if (op.opcode == Opcode::Permute) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        input.dims.size() != out.dims.size() || input.dims.empty())
      fail("permute requires equal-rank non-scalar float input/output");
    const auto rank = input.dims.size();
    std::vector<bool> seen(rank, false);
    for (std::size_t axis = 0; axis < rank; ++axis) {
      const auto key = static_cast<AttrKey>(
          static_cast<std::uint32_t>(AttrKey::Permutation0) + axis);
      const auto *attribute = op.find(key);
      if (!attribute || attribute->kind != AttrKind::U64 ||
          attribute->as_u64() >= rank || seen[attribute->as_u64()] ||
          out.dims[axis] != input.dims[attribute->as_u64()])
        fail("permute requires a complete unique axis permutation matching output shape");
      seen[attribute->as_u64()] = true;
    }
    return;
  }

  if (op.opcode == Opcode::Concat) {
    if (op.inputs.size() < 2U || op.outputs.size() != 1U)
      fail("concat requires at least two inputs and one output");
    const auto &first = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto *axis_attribute = op.find(AttrKey::Axis);
    if (!axis_attribute || axis_attribute->kind != AttrKind::U64 ||
        first.dims.empty() || axis_attribute->as_u64() >= first.dims.size() ||
        !supported_float(first.dtype) || out.dtype != first.dtype ||
        out.dims.size() != first.dims.size())
      fail("concat requires an in-range axis and equal-rank float tensors");
    const auto axis = static_cast<std::size_t>(axis_attribute->as_u64());
    std::uint64_t joined = 0U;
    for (const auto input_id : op.inputs) {
      const auto &input = tensor_or_fail(program, input_id, op);
      if (input.dtype != first.dtype || input.dims.size() != first.dims.size())
        fail("concat inputs must have equal dtype and rank");
      for (std::size_t dimension = 0U; dimension < first.dims.size();
           ++dimension)
        if (dimension != axis && input.dims[dimension] != first.dims[dimension])
          fail("concat non-axis dimensions must match");
      if (joined > std::numeric_limits<std::uint64_t>::max() -
                       input.dims[axis])
        fail("concat axis size overflows");
      joined += input.dims[axis];
    }
    for (std::size_t dimension = 0U; dimension < first.dims.size();
         ++dimension)
      if (out.dims[dimension] !=
          (dimension == axis ? joined : first.dims[dimension]))
        fail("concat output shape does not match joined inputs");
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
    const auto layout = static_cast<ModulationLayout>(op.u64(
        AttrKey::ModulationLayout,
        static_cast<std::uint64_t>(ModulationLayout::ExplicitScaleShift)));
    if (layout == ModulationLayout::SharedVectorDelta) {
      if (op.inputs.size() != 4U)
        fail("shared-vector rms_norm_modulate expects x,weight,vector,delta");
      const auto &weight = tensor_or_fail(program, op.inputs[1], op);
      const auto &vector = tensor_or_fail(program, op.inputs[2], op);
      const auto &delta = tensor_or_fail(program, op.inputs[3], op);
      const auto &out = tensor_or_fail(program, op.outputs[0], op);
      same_shape_dtype(x, out, op);
      if (!supported_float(x.dtype) || x.dims.size() != 2U ||
          weight.dtype != x.dtype || weight.dims.size() != 1U ||
          weight.dims[0] != x.dims[1] || vector.dtype != x.dtype ||
          vector.dims.size() != 2U || vector.dims[1] != x.dims[1] ||
          vector.dims[0] == 0U || x.dims[0] % vector.dims[0] != 0U ||
          delta.dtype != x.dtype || delta.dims.size() != 2U ||
          delta.dims[0] != 2U || delta.dims[1] != x.dims[1])
        fail("shared-vector rms_norm_modulate requires x/out [rows,hidden], "
             "weight [hidden], vector [batch,hidden], delta [2,hidden]");
      if (!(op.f64(AttrKey::Epsilon, 1.0e-5) > 0.0) ||
          !std::isfinite(op.f64(AttrKey::WeightOffset, 0.0)))
        fail("shared-vector rms_norm_modulate has invalid norm attributes");
      const auto block = op.u64(AttrKey::BlockSize, 256U);
      const auto reduction_tile = op.u64(AttrKey::ReductionTileSize, 0U);
      if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U ||
          (reduction_tile != 0U && reduction_tile != 2048U &&
           reduction_tile != 8192U) ||
          (reduction_tile != 0U &&
           (block != 512U || x.dims[1] != 6144U)))
        fail("shared-vector rms_norm_modulate has invalid reduction geometry");
      return;
    }
    if (layout != ModulationLayout::ExplicitScaleShift)
      fail("rms_norm_modulate has an unknown modulation layout");
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
    for (std::size_t axis = 0U; axis + 1U < input.dims.size(); ++axis)
      if (input.dims[axis] != out.dims[axis])
        fail("swiglu input and output prefix dimensions must match");
    const auto start = op.u64(AttrKey::Start, 0U);
    if (out.dims.back() >
            (std::numeric_limits<std::uint64_t>::max() - start) / 2U ||
        start + out.dims.back() * 2U > input.dims.back())
      fail("swiglu packed window is outside the input final dimension");
    return;
  }

  if (op.opcode == Opcode::ResidualGate) {
    expect_counts(op, 3, 1);
    const auto &residual = tensor_or_fail(program, op.inputs[0], op);
    const auto &branch = tensor_or_fail(program, op.inputs[1], op);
    const auto &gate = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(residual, branch, op);
    same_shape_dtype(residual, out, op);
    if (!supported_float(residual.dtype) || residual.dims.empty() ||
        gate.dtype != residual.dtype || gate.dims.empty() ||
        gate.dims.back() != residual.dims.back())
      fail("residual_gate requires compatible floating residual/branch/output "
           "and gate rows with the same final dimension");
    const auto rows = residual.element_count() / residual.dims.back();
    const auto gate_rows = gate.element_count() / gate.dims.back();
    if (gate_rows == 0U || rows % gate_rows != 0U)
      fail("residual_gate gate rows must divide residual rows");
    return;
  }

  if (op.opcode == Opcode::LayerNormModulate) {
    expect_counts(op, 5U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &bias = tensor_or_fail(program, op.inputs[2], op);
    const auto &scale = tensor_or_fail(program, op.inputs[3], op);
    const auto &shift = tensor_or_fail(program, op.inputs[4], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || bias.dtype != input.dtype ||
        weight.dims != std::vector<std::uint64_t>{input.dims.back()} ||
        bias.dims != weight.dims || scale.dtype != input.dtype ||
        shift.dtype != input.dtype || scale.dims != shift.dims ||
        scale.dims.empty() || scale.dims.back() != input.dims.back())
      fail("layer_norm_modulate requires floating x/out, [hidden] weight/bias, "
           "and compatible scale/shift rows");
    const auto input_rows = input.element_count() / input.dims.back();
    const auto modulation_rows = scale.element_count() / scale.dims.back();
    if (modulation_rows == 0U || input_rows % modulation_rows != 0U)
      fail("layer_norm_modulate scale/shift rows must divide input rows");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U ||
        !(op.f64(AttrKey::Epsilon, 1.0e-5) > 0.0))
      fail("layer_norm_modulate has invalid normalization geometry");
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
    const auto bias_mode = op.u64(
        AttrKey::LinearBiasMode,
        static_cast<std::uint64_t>(LinearBiasMode::Epilogue));
    if (bias_mode != static_cast<std::uint64_t>(LinearBiasMode::Epilogue) &&
        bias_mode != static_cast<std::uint64_t>(LinearBiasMode::Addmm))
      fail("linear bias mode must be epilogue or addmm");
    if (op.inputs.size() != 3U &&
        bias_mode != static_cast<std::uint64_t>(LinearBiasMode::Epilogue))
      fail("linear addmm bias mode requires a bias input");
    if (op.find(AttrKey::WorkspaceLimitBytes) &&
        op.u64(AttrKey::WorkspaceLimitBytes, 0U) == 0U)
      fail("linear workspace limit must be positive when specified");
    const auto implementation = op.u64(AttrKey::Implementation, 1U);
    if (implementation != 1U && implementation != 2U && implementation != 3U)
      fail("linear implementation must be 1 (native), 2 (tf32), or 3 "
           "(direct packed INT5)");
    if (input.dtype != DType::F32 && implementation == 2U)
      fail("tf32 Linear implementation requires f32 storage");
    return;
  }

  if (op.opcode == Opcode::QuantizeInt8Rows) {
    if (op.inputs.empty() ||
        (op.outputs.size() != 2U && op.outputs.size() != 4U))
      fail("quantize_int8_rows expects BF16 inputs and either two or four "
           "outputs");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &quantized = tensor_or_fail(program, op.outputs[0], op);
    const auto &scales = tensor_or_fail(program, op.outputs[1], op);
    const auto &last_input = tensor_or_fail(program, op.inputs.back(), op);
    const auto dynamic_clip = last_input.dtype == DType::F32;
    const auto data_input_count =
        op.inputs.size() - static_cast<std::size_t>(dynamic_clip);
    if (data_input_count == 0U ||
        (dynamic_clip && last_input.dims != std::vector<std::uint64_t>{1U}))
      fail("quantize_int8_rows runtime clipping input must be one F32 scalar "
           "after at least one BF16 data input");
    std::uint64_t combined_width = 0U;
    const auto rows = input.element_count() / input.dims.back();
    for (std::size_t index = 0U; index < data_input_count; ++index) {
      const auto input_id = op.inputs[index];
      const auto &part = tensor_or_fail(program, input_id, op);
      if (part.dtype != DType::BF16 || part.dims.empty() ||
          part.element_count() / part.dims.back() != rows)
        fail("quantize_int8_rows inputs must be compatible BF16 tensors");
      if (combined_width > std::numeric_limits<std::uint64_t>::max() -
                               part.dims.back())
        fail("quantize_int8_rows combined width overflow");
      combined_width += part.dims.back();
    }
    if (input.dtype != DType::BF16 || input.dims.empty() ||
        quantized.dtype != DType::I8 || quantized.dims.empty() ||
        quantized.dims.back() != combined_width ||
        quantized.element_count() / combined_width != rows ||
        scales.dtype != DType::F32 || scales.dims.size() != 1U ||
        scales.dims[0] != quantized.element_count() / combined_width)
      fail("quantize_int8_rows requires BF16 inputs, combined-shape I8 output, "
           "and one F32 scale per flattened row");
    if (op.outputs.size() == 4U) {
      const auto &residual = tensor_or_fail(program, op.outputs[2], op);
      const auto &residual_scales = tensor_or_fail(program, op.outputs[3], op);
      if (residual.dtype != DType::I8 || residual.dims != quantized.dims ||
          residual_scales.dtype != DType::F32 ||
          residual_scales.dims != scales.dims)
        fail("four-output quantize_int8_rows requires an equal-shape I8 "
             "residual and one F32 residual scale per flattened row");
    }
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block != 256U)
      fail("quantize_int8_rows currently requires BlockSize=256");
    const auto clip_ratio = op.f64(AttrKey::Scale, 1.0);
    if (!(clip_ratio > 0.0) || clip_ratio > 1.0)
      fail("quantize_int8_rows Scale range multiplier must be in (0,1]");
    const auto implementation = static_cast<Int8RowQuantization>(
        op.u64(AttrKey::Implementation,
               static_cast<std::uint64_t>(Int8RowQuantization::Direct)));
    if (implementation != Int8RowQuantization::Direct &&
        implementation != Int8RowQuantization::H256ConvRot &&
        implementation != Int8RowQuantization::H256SignedConvRot &&
        implementation != Int8RowQuantization::H4096SignedConvRot &&
        implementation != Int8RowQuantization::H256F32ConvRot &&
        implementation != Int8RowQuantization::H256F32SignedConvRot &&
        implementation != Int8RowQuantization::H4096F32SignedConvRot &&
        implementation != Int8RowQuantization::H256F32SylvesterConvRot)
      fail("quantize_int8_rows implementation must be direct, H256 ConvRot, "
           "H256 F32 ConvRot, H256 signed ConvRot, H256 F32 signed ConvRot, "
           "H4096 signed ConvRot, H4096 F32 signed ConvRot, or H256 F32 "
           "Sylvester ConvRot");
    if ((implementation == Int8RowQuantization::H256ConvRot ||
         implementation == Int8RowQuantization::H256F32ConvRot ||
         implementation == Int8RowQuantization::H256SignedConvRot ||
         implementation == Int8RowQuantization::H256F32SignedConvRot ||
         implementation == Int8RowQuantization::H256F32SylvesterConvRot) &&
        combined_width % 256U != 0U)
      fail("H256 ConvRot row quantization requires a last dimension divisible "
           "by 256");
    if ((implementation == Int8RowQuantization::H4096SignedConvRot ||
         implementation == Int8RowQuantization::H4096F32SignedConvRot) &&
        combined_width % 4096U != 0U)
      fail("H4096 ConvRot row quantization requires a last dimension "
           "divisible by 4096");
    return;
  }

  if (op.opcode == Opcode::LinearInt8Scaled) {
    expect_counts(op, 4U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &row_scales = tensor_or_fail(program, op.inputs[2], op);
    const auto &column_scales = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (input.dtype != DType::I8 || weight.dtype != DType::I8 ||
        row_scales.dtype != DType::F32 ||
        column_scales.dtype != DType::F32 || out.dtype != DType::BF16 ||
        input.dims.empty() || weight.dims.size() != 2U || out.dims.empty())
      fail("linear_int8_scaled requires I8 input/weight, F32 scales, and "
           "BF16 output");
    const auto rows = input.element_count() / input.dims.back();
    const auto inner = input.dims.back();
    const auto columns = weight.dims[0];
    if (weight.dims[1] != inner || row_scales.dims !=
            std::vector<std::uint64_t>{rows} ||
        column_scales.dims != std::vector<std::uint64_t>{columns} ||
        out.element_count() != rows * columns ||
        out.dims.back() != columns)
      fail("linear_int8_scaled shapes must flatten as [M,K] x [N,K] with "
           "row [M], column [N], and BF16 [M,N]");
    // Weight and column scales are either prepared constants (offline
    // quantization) or internal tensors materialized by earlier operations
    // in the same run (a quantized-storage weight reconstructed and
    // row-quantized on device); an Input/Output role is neither.
    for (const auto *operand : {&weight, &column_scales})
      if (operand->has_role(TensorRole::Input) ||
          operand->has_role(TensorRole::Output) ||
          operand->has_role(TensorRole::Parameter))
        fail("linear_int8_scaled weight and column scales must be constants "
             "or internal tensors");
    return;
  }

  if (op.opcode == Opcode::LinearInt8WeightScaled) {
    expect_counts(op, 3U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &scales = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (input.dtype != DType::BF16 || weight.dtype != DType::I8 ||
        scales.dtype != DType::F32 || out.dtype != DType::BF16 ||
        input.dims.empty() || weight.dims.size() != 2U || out.dims.empty())
      fail("linear_int8_weight_scaled requires BF16 input/output, rank-2 I8 "
           "weight, and F32 scales");
    const auto rows = input.element_count() / input.dims.back();
    const auto inner = input.dims.back();
    const auto columns = weight.dims[0];
    if (weight.dims[1] != inner ||
        scales.dims != std::vector<std::uint64_t>{columns} ||
        out.element_count() != rows * columns || out.dims.back() != columns)
      fail("linear_int8_weight_scaled shapes must flatten as [M,K] x [N,K] "
           "with scale [N] and BF16 [M,N]");
    if (!weight.has_role(TensorRole::Constant) ||
        !scales.has_role(TensorRole::Constant))
      fail("linear_int8_weight_scaled weight and scales must be constants");
    return;
  }

  if (op.opcode == Opcode::QuantizeFp8Rows) {
    expect_counts(op, 1U, 2U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &quantized = tensor_or_fail(program, op.outputs[0], op);
    const auto &scales = tensor_or_fail(program, op.outputs[1], op);
    if (input.dtype != DType::BF16 || input.dims.empty() ||
        quantized.dtype != DType::FP8E4M3 || quantized.dims != input.dims ||
        scales.dtype != DType::F32 ||
        scales.dims != std::vector<std::uint64_t>{
                           input.element_count() / input.dims.back()})
      fail("quantize_fp8_rows requires BF16 input, equal-shape FP8 E4M3 "
           "output, and one F32 scale per flattened row");
    return;
  }

  if (op.opcode == Opcode::LinearFp8Scaled) {
    expect_counts(op, 4U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &row_scales = tensor_or_fail(program, op.inputs[2], op);
    const auto &column_scales = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (input.dtype != DType::FP8E4M3 ||
        weight.dtype != DType::FP8E4M3 ||
        row_scales.dtype != DType::F32 ||
        column_scales.dtype != DType::F32 || out.dtype != DType::BF16 ||
        input.dims.empty() || weight.dims.size() != 2U || out.dims.empty())
      fail("linear_fp8_scaled requires FP8 E4M3 input/weight, F32 scales, "
           "and BF16 output");
    const auto rows = input.element_count() / input.dims.back();
    const auto inner = input.dims.back();
    const auto columns = weight.dims[0];
    if (weight.dims[1] != inner ||
        row_scales.dims != std::vector<std::uint64_t>{rows} ||
        column_scales.dims != std::vector<std::uint64_t>{columns} ||
        out.element_count() != rows * columns || out.dims.back() != columns)
      fail("linear_fp8_scaled shapes must flatten as [M,K] x [N,K] with "
           "row [M], column [N], and BF16 [M,N]");
    if (!weight.has_role(TensorRole::Constant) ||
        !column_scales.has_role(TensorRole::Constant))
      fail("linear_fp8_scaled weight and column scales must be constants");
    return;
  }

  if (op.opcode == Opcode::QuantizeFp8Blocks32) {
    expect_counts(op, 1U, 2U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &quantized = tensor_or_fail(program, op.outputs[0], op);
    const auto &scales = tensor_or_fail(program, op.outputs[1], op);
    if (input.dtype != DType::BF16 || input.dims.empty() ||
        quantized.dtype != DType::FP8E4M3 || quantized.dims != input.dims ||
        scales.dtype != DType::FP8E8M0)
      fail("quantize_fp8_blocks32 requires BF16 input, equal-shape FP8 E4M3 "
           "output, and tiled FP8 E8M0 scales");
    const auto rows = input.element_count() / input.dims.back();
    const auto padded_rows = ((rows + 127U) / 128U) * 128U;
    const auto blocks = (input.dims.back() + 31U) / 32U;
    const auto padded_blocks = ((blocks + 3U) / 4U) * 4U;
    if (scales.dims !=
        std::vector<std::uint64_t>{padded_rows, padded_blocks})
      fail("quantize_fp8_blocks32 scale storage must pad outer rows to 128 "
           "and K/32 blocks to 4");
    return;
  }

  if (op.opcode == Opcode::LinearFp8BlockScaled) {
    expect_counts(op, 4U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &input_scales = tensor_or_fail(program, op.inputs[2], op);
    const auto &weight_scales = tensor_or_fail(program, op.inputs[3], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (input.dtype != DType::FP8E4M3 ||
        weight.dtype != DType::FP8E4M3 ||
        input_scales.dtype != DType::FP8E8M0 ||
        weight_scales.dtype != DType::FP8E8M0 ||
        out.dtype != DType::BF16 || input.dims.empty() ||
        weight.dims.size() != 2U || out.dims.empty())
      fail("linear_fp8_block_scaled requires FP8 E4M3 input/weight, tiled "
           "FP8 E8M0 scales, and BF16 output");
    const auto rows = input.element_count() / input.dims.back();
    const auto inner = input.dims.back();
    const auto columns = weight.dims[0];
    const auto padded_blocks = (((inner + 31U) / 32U + 3U) / 4U) * 4U;
    const auto input_padded_rows = ((rows + 127U) / 128U) * 128U;
    const auto weight_padded_rows = ((columns + 127U) / 128U) * 128U;
    if (weight.dims[1] != inner ||
        input_scales.dims !=
            std::vector<std::uint64_t>{input_padded_rows, padded_blocks} ||
        weight_scales.dims !=
            std::vector<std::uint64_t>{weight_padded_rows, padded_blocks} ||
        out.element_count() != rows * columns || out.dims.back() != columns)
      fail("linear_fp8_block_scaled shapes must flatten as [M,K] x [N,K] "
           "with tiled/padded block scales and BF16 [M,N]");
    if (!weight.has_role(TensorRole::Constant) ||
        !weight_scales.has_role(TensorRole::Constant))
      fail("linear_fp8_block_scaled weight and scales must be constants");
    return;
  }

  if (op.opcode == Opcode::DequantizeInt8Blocks) {
    expect_counts(op, 2U, 1U);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &scales = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto block = op.u64(AttrKey::BlockSize, 0U);
    if (input.dtype != DType::I8 || input.dims.size() != 2U ||
        scales.dtype != DType::F32 || scales.dims.size() != 2U ||
        out.dtype != DType::BF16 || out.dims != input.dims || block == 0U ||
        (block & (block - 1U)) != 0U || block > 256U ||
        scales.dims != std::vector<std::uint64_t>{
                           input.dims[0], (input.dims[1] + block - 1U) / block})
      fail("dequantize_int8_blocks requires rank-2 I8 input, rank-2 F32 "
           "block scales, equal-shape BF16 output, and a power-of-two block "
           "size no larger than 256");
    if (!input.has_role(TensorRole::Constant) ||
        !scales.has_role(TensorRole::Constant))
      fail("dequantize_int8_blocks input and scales must be constants");
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
    const bool compatible_table_rank =
        input.dims.size() == 4U
            ? cos.dims.size() == 3U
            : (cos.dims.size() == 2U ||
               (cos.dims.size() == 3U && cos.dims[0] == 1U));
    if (!supported_float(input.dtype) ||
        (input.dims.size() != 3U && input.dims.size() != 4U) ||
        weight.dtype != input.dtype || weight.dims.size() != 1 ||
        (cos.dtype != input.dtype && cos.dtype != DType::F32) ||
        sin.dtype != cos.dtype || cos.dims != sin.dims ||
        !compatible_table_rank)
      fail("qk_norm_partial_rope requires [S,H,D] or [B,S,H,D], weight "
           "[D], and compatible cos/sin tables");
    const auto head_axis = input.dims.size() - 2U;
    const auto heads = op.u64(AttrKey::Heads, input.dims[head_axis]);
    const auto head_dim = op.u64(AttrKey::HeadDim, input.dims.back());
    const auto table_width = cos.dims.back();
    const auto rotary = op.u64(AttrKey::RotaryDim, table_width * 2U);
    const auto input_batch = input.dims.size() == 4U ? input.dims[0] : 1U;
    const auto input_sequence =
        input.dims.size() == 4U ? input.dims[1] : input.dims[0];
    const auto table_batch = cos.dims.size() == 3U ? cos.dims[0] : 1U;
    const auto table_sequence =
        cos.dims.size() == 3U ? cos.dims[1] : cos.dims[0];
    const auto table_start = op.u64(AttrKey::Start, 0U);
    if (heads != input.dims[head_axis] || head_dim != input.dims.back() ||
        weight.dims[0] != head_dim || table_batch != input_batch ||
        table_start > table_sequence ||
        input_sequence > table_sequence - table_start ||
        rotary == 0 || rotary > head_dim || (rotary % 2U) != 0U ||
        (table_width != rotary && table_width * 2U != rotary))
      fail("qk_norm_partial_rope shape attributes are inconsistent");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("qk_norm_partial_rope block size must be a power of two in [32,1024]");
    return;
  }

  if (op.opcode == Opcode::Attention) {
    if ((op.inputs.size() != 3U && op.inputs.size() != 4U) ||
        op.outputs.size() != 1U)
      fail("attention expects q, k, v, optional additive bias, and one output");
    const auto &q = tensor_or_fail(program, op.inputs[0], op);
    const auto &k = tensor_or_fail(program, op.inputs[1], op);
    const auto &v = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(k, v, op);
    same_shape_dtype(q, out, op);
    if (!supported_float(q.dtype) ||
        (q.dims.size() != 3U && q.dims.size() != 4U))
      fail("attention semantics require float [S,H,D] or [B,S,H,D]");
    // Grouped-query attention: k/v carry KvHeads heads (AttrKey 39); query
    // head h reads kv head h/(H/KvHeads).  An absent attribute means
    // KvHeads == H — bit-for-bit the historical contract, so every existing
    // program verifies and fingerprints unchanged.
    const auto head_axis = q.dims.size() - 2U;
    const auto sequence_axis = q.dims.size() - 3U;
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[head_axis]);
    if (kv_heads == 0U || q.dims[head_axis] % kv_heads != 0U)
      fail("attention KvHeads must be nonzero and divide the query head "
           "count");
    // Cross-attention: K/V may carry their own row count (text keys under
    // image queries). Everything else — batch, head-dim, dtype — matches the
    // query, and a causal mask keeps the historical square contract.
    auto expected_kv = q.dims;
    expected_kv[head_axis] = kv_heads;
    expected_kv[sequence_axis] = k.dims[sequence_axis];
    const auto kv_sequence = k.dims[sequence_axis];
    if (k.dtype != q.dtype || k.dims != expected_kv || kv_sequence == 0U)
      fail("attention k/v must match query batch/head-dim with KvHeads heads "
           "and a nonzero key row count");
    if (kv_sequence != q.dims[sequence_axis] &&
        op.boolean(AttrKey::Causal, false))
      fail("causal attention requires equal query and key row counts");
    if (op.inputs.size() == 4U) {
      const auto &bias = tensor_or_fail(program, op.inputs[3], op);
      const auto batch = q.dims.size() == 4U ? q.dims[0] : 1U;
      const auto sequence = q.dims[sequence_axis];
      if (q.dims.size() != 4U || bias.dtype != q.dtype ||
          bias.dims != std::vector<std::uint64_t>{batch, 1U, sequence,
                                                  kv_sequence})
        fail("attention additive bias must be [B,1,S,Skv] in the query dtype");
    }
    const auto implementation = op.u64(AttrKey::Implementation, 1U);
    if (implementation != 1U && implementation != 2U &&
        implementation != 3U && implementation != 4U)
      fail("attention implementation must be 1 (generated), 2 (cuDNN), or "
           "3 (materialized f32), or 4 (native FlashAttention)");
    if (implementation == 2U && q.dtype != DType::BF16 &&
        q.dtype != DType::F16)
      fail("cuDNN attention implementation requires bf16 or f16");
    if (implementation == 1U && op.inputs.size() != 3U)
      fail("generated attention admits q/k/v without an additive bias; use "
           "cuDNN for masked semantics");
    if (implementation == 1U && q.dims[sequence_axis] > 4096U)
      fail("naive exact attention is admitted only for S<=4096; use a backend implementation");
    if (implementation == 3U &&
        (q.dtype != DType::F32 || q.dims.size() != 3U ||
         q.dims[head_axis] != 1U || kv_heads != 1U ||
         op.inputs.size() != 3U || op.boolean(AttrKey::Causal, false)))
      fail("materialized f32 attention requires noncausal unbatched "
           "f32 [S,1,D] q/k/v without additive bias");
    if (implementation == 4U &&
        (q.dtype != DType::BF16 ||
         (q.dims.size() != 3U && q.dims.size() != 4U) ||
         q.dims.back() != 128U || op.inputs.size() != 3U ||
         op.boolean(AttrKey::Causal, false)))
      fail("native FlashAttention requires noncausal bf16 [S,H,128] or "
           "[B,S,H,128] q/k/v without additive bias");
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
    for (std::size_t axis = 0U; axis + 1U < input.dims.size(); ++axis)
      if (input.dims[axis] != grad_output.dims[axis])
        fail("swiglu_backward input and output prefix dimensions must match");
    const auto start = op.u64(AttrKey::Start, 0U);
    if (grad_output.dims.back() >
            (std::numeric_limits<std::uint64_t>::max() - start) / 2U ||
        start + grad_output.dims.back() * 2U > input.dims.back())
      fail("swiglu_backward packed window is outside the input final dimension");
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
    same_shape_dtype(grad_output, grad_branch, op);
    same_shape_dtype(gate, grad_gate, op);
    check_accumulator_f32(op);
    // The forward admits a gate that governs several rows at once; so must
    // this, or a DiT block cannot be differentiated at all.
    if (!supported_float(grad_output.dtype) || grad_output.dims.empty() ||
        gate.dtype != grad_output.dtype || gate.dims.empty() ||
        gate.dims.back() != grad_output.dims.back())
      fail("residual_gate_backward admits f32, bf16, or f16 with gate rows "
           "sharing the final dimension");
    const auto rows = grad_output.element_count() / grad_output.dims.back();
    const auto gate_rows = gate.element_count() / gate.dims.back();
    if (gate_rows == 0U || rows % gate_rows != 0U)
      fail("residual_gate_backward gate rows must divide the output rows");
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

  if (op.opcode == Opcode::LayerNormModulateBackward) {
    expect_counts(op, 5, 5);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    const auto &input = tensor_or_fail(program, op.inputs[1], op);
    const auto &weight = tensor_or_fail(program, op.inputs[2], op);
    const auto &bias = tensor_or_fail(program, op.inputs[3], op);
    const auto &scale = tensor_or_fail(program, op.inputs[4], op);
    const auto &grad_input = tensor_or_fail(program, op.outputs[0], op);
    const auto &grad_weight = tensor_or_fail(program, op.outputs[1], op);
    const auto &grad_bias = tensor_or_fail(program, op.outputs[2], op);
    const auto &grad_scale = tensor_or_fail(program, op.outputs[3], op);
    const auto &grad_shift = tensor_or_fail(program, op.outputs[4], op);
    same_shape_dtype(input, grad_output, op);
    same_shape_dtype(input, grad_input, op);
    same_shape_dtype(scale, grad_scale, op);
    same_shape_dtype(scale, grad_shift, op);
    check_accumulator_f32(op);
    if (!supported_float(input.dtype) || input.dims.empty() ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back() || bias.dtype != input.dtype ||
        bias.dims != weight.dims || grad_weight.dtype != weight.dtype ||
        grad_weight.dims != weight.dims || grad_bias.dtype != weight.dtype ||
        grad_bias.dims != weight.dims || scale.dtype != input.dtype ||
        scale.dims.empty() || scale.dims.back() != input.dims.back())
      fail("layer_norm_modulate_backward requires float tensors with affine "
           "gradients matching the final dimension and modulation gradients "
           "matching the scale");
    // The same divisibility the forward demands: a modulation row governs a
    // whole number of input rows.
    const auto input_rows = input.element_count() / input.dims.back();
    const auto modulation_rows = scale.element_count() / scale.dims.back();
    if (modulation_rows == 0U || input_rows % modulation_rows != 0U)
      fail("layer_norm_modulate_backward scale rows must divide input rows");
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    if (!(epsilon > 0.0))
      fail("layer_norm_modulate_backward epsilon must be positive");
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (block < 32U || block > 1024U || (block & (block - 1U)) != 0U)
      fail("layer_norm_modulate_backward block size must be a power of two in "
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
    // Whatever the forward admits, this has to admit: the batched
    // [B,S,H,D] form, F32 rotation tables under half-precision q/k, a table
    // longer than the rows being rotated, and a start offset into it.  These
    // are the same checks the forward makes, written the same way, because a
    // gradient that refuses a shape its own forward accepted is a gradient
    // that cannot train the model that produced it.
    const bool compatible_table_rank =
        input.dims.size() == 4U
            ? cos.dims.size() == 3U
            : (cos.dims.size() == 2U ||
               (cos.dims.size() == 3U && cos.dims[0] == 1U));
    if (!supported_float(input.dtype) ||
        (input.dims.size() != 3U && input.dims.size() != 4U) ||
        weight.dtype != input.dtype || weight.dims.size() != 1U ||
        weight.dims[0] != input.dims.back() ||
        (cos.dtype != input.dtype && cos.dtype != DType::F32) ||
        sin.dtype != cos.dtype || cos.dims != sin.dims ||
        !compatible_table_rank)
      fail("qk_norm_partial_rope_backward requires grad/input [S,H,D] or "
           "[B,S,H,D], weight [D], and compatible cos/sin tables");
    const auto head_dim = input.dims.back();
    const auto table_width = cos.dims.back();
    const auto rotary = op.u64(AttrKey::RotaryDim, head_dim);
    const auto input_batch = input.dims.size() == 4U ? input.dims[0] : 1U;
    const auto input_sequence =
        input.dims.size() == 4U ? input.dims[1] : input.dims[0];
    const auto table_batch = cos.dims.size() == 3U ? cos.dims[0] : 1U;
    const auto table_sequence =
        cos.dims.size() == 3U ? cos.dims[1] : cos.dims[0];
    const auto table_start = op.u64(AttrKey::Start, 0U);
    if (table_batch != input_batch || table_start > table_sequence ||
        input_sequence > table_sequence - table_start || rotary == 0U ||
        rotary > head_dim || (rotary % 2U) != 0U ||
        (table_width != rotary && table_width * 2U != rotary))
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
    // Real models carry a batch, so the batched [B,S,H,D] form is accepted
    // alongside the historical [S,H,D] one; every axis is named relative to
    // the end so both ranks share one rule.
    if (!supported_float(q.dtype) ||
        (q.dims.size() != 3U && q.dims.size() != 4U))
      fail("attention_lse requires f32, bf16, or f16 [S,H,D] or [B,S,H,D] "
           "inputs");
    const auto head_axis = q.dims.size() - 2U;
    const auto sequence_axis = q.dims.size() - 3U;
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[head_axis]);
    if (kv_heads == 0U || q.dims[head_axis] % kv_heads != 0U)
      fail("attention_lse KvHeads must be nonzero and divide the query head "
           "count");
    // Cross attention: the keys may carry their own row count, so only the
    // batch, head grouping, and head dim have to match the query.  A causal
    // mask keeps the historical square contract.
    auto expected_kv = q.dims;
    expected_kv[head_axis] = kv_heads;
    expected_kv[sequence_axis] = k.dims.size() == q.dims.size()
                                     ? k.dims[sequence_axis]
                                     : q.dims[sequence_axis];
    if (k.dtype != q.dtype || k.dims != expected_kv ||
        k.dims[sequence_axis] == 0U)
      fail("attention_lse k must be [S,KvHeads,D] with the query dtype");
    auto expected_lse = q.dims;
    expected_lse.pop_back();
    if (lse.dtype != DType::F32 || lse.dims != expected_lse)
      fail("attention_lse output must be F32 [S,H]");
    if (op.boolean(AttrKey::Causal, false) &&
        k.dims[sequence_axis] != q.dims[sequence_axis])
      fail("a causal mask requires the key and query row counts to match");
    if (q.dims[sequence_axis] > 4096U || k.dims[sequence_axis] > 4096U)
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
    if (!supported_float(q.dtype) ||
        (q.dims.size() != 3U && q.dims.size() != 4U))
      fail("attention_backward requires f32, bf16, or f16 [S,H,D] or "
           "[B,S,H,D] tensors");
    const auto head_axis = q.dims.size() - 2U;
    const auto sequence_axis = q.dims.size() - 3U;
    const auto kv_heads = op.u64(AttrKey::KvHeads, q.dims[head_axis]);
    if (kv_heads == 0U || q.dims[head_axis] % kv_heads != 0U)
      fail("attention_backward KvHeads must be nonzero and divide the query "
           "head count");
    // Cross attention: the keys may carry their own row count, so only the
    // batch, head grouping, and head dim have to match the query.  A causal
    // mask keeps the historical square contract.
    auto expected_kv = q.dims;
    expected_kv[head_axis] = kv_heads;
    expected_kv[sequence_axis] = k.dims.size() == q.dims.size()
                                     ? k.dims[sequence_axis]
                                     : q.dims[sequence_axis];
    if (k.dtype != q.dtype || k.dims != expected_kv ||
        k.dims[sequence_axis] == 0U)
      fail("attention_backward k/v must be [S,KvHeads,D] with the query "
           "dtype");
    auto expected_lse = q.dims;
    expected_lse.pop_back();
    if (lse.dtype != DType::F32 || lse.dims != expected_lse)
      fail("attention_backward saved logsumexp must be F32 [S,H]");
    if (op.boolean(AttrKey::Causal, false) &&
        k.dims[sequence_axis] != q.dims[sequence_axis])
      fail("a causal mask requires the key and query row counts to match");
    if (q.dims[sequence_axis] > 4096U || k.dims[sequence_axis] > 4096U)
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

  if (op.opcode == Opcode::Conv2d) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("conv2d expects input, weight, optional bias, and one output");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || weight.dtype != input.dtype ||
        out.dtype != input.dtype || input.dims.size() != 4U ||
        weight.dims.size() != 4U || out.dims.size() != 4U)
      fail("conv2d requires NCHW input/output and OIHW weights of one float dtype");
    const auto stride_h = op.u64(AttrKey::StrideH, 1U);
    const auto stride_w = op.u64(AttrKey::StrideW, 1U);
    const auto dilation_h = op.u64(AttrKey::DilationH, 1U);
    const auto dilation_w = op.u64(AttrKey::DilationW, 1U);
    const auto pad_top = op.u64(AttrKey::PadTop, 0U);
    const auto pad_bottom = op.u64(AttrKey::PadBottom, 0U);
    const auto pad_west = op.u64(AttrKey::PadWest, 0U);
    const auto pad_east = op.u64(AttrKey::PadEast, 0U);
    const auto groups = op.u64(AttrKey::Groups, 1U);
    constexpr auto kLimit = std::uint64_t{1} << 20U;
    if (stride_h == 0U || stride_w == 0U || dilation_h == 0U ||
        dilation_w == 0U || groups == 0U || stride_h > kLimit ||
        stride_w > kLimit || dilation_h > kLimit || dilation_w > kLimit ||
        pad_top > kLimit || pad_bottom > kLimit || pad_west > kLimit ||
        pad_east > kLimit)
      fail("conv2d attributes are invalid or out of range");
    const auto batch = input.dims[0];
    const auto in_channels = input.dims[1];
    const auto height = input.dims[2];
    const auto width = input.dims[3];
    const auto out_channels = weight.dims[0];
    const auto kernel_h = weight.dims[2];
    const auto kernel_w = weight.dims[3];
    if (in_channels % groups != 0U || out_channels % groups != 0U ||
        weight.dims[1] != in_channels / groups || kernel_h == 0U ||
        kernel_w == 0U)
      fail("conv2d weight/groups geometry is invalid");
    const auto effective_h = dilation_h * (kernel_h - 1U) + 1U;
    const auto effective_w = dilation_w * (kernel_w - 1U) + 1U;
    const auto padded_h = height + pad_top + pad_bottom;
    const auto padded_w = width + pad_west + pad_east;
    if (padded_h < effective_h || padded_w < effective_w)
      fail("conv2d kernel does not fit its padded input");
    const auto output_h = (padded_h - effective_h) / stride_h + 1U;
    const auto output_w = (padded_w - effective_w) / stride_w + 1U;
    if (out.dims != std::vector<std::uint64_t>{batch, out_channels, output_h,
                                               output_w})
      fail("conv2d output geometry does not match its attributes");
    if (op.inputs.size() == 3U) {
      const auto &bias = tensor_or_fail(program, op.inputs[2], op);
      if (bias.dtype != input.dtype ||
          bias.dims != std::vector<std::uint64_t>{out_channels})
        fail("conv2d bias must be a [C_out] vector of the input dtype");
    }
    return;
  }

  if (op.opcode == Opcode::Conv3dBackwardInput ||
      op.opcode == Opcode::Conv3dBackwardWeight ||
      op.opcode == Opcode::Conv3dBackwardBias) {
    const bool bias_gradient = op.opcode == Opcode::Conv3dBackwardBias;
    expect_counts(op, bias_gradient ? 1U : 2U, 1U);
    check_accumulator_f32(op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    if (!supported_float(grad_output.dtype) || grad_output.dims.size() != 5U)
      fail("conv3d gradients require an NCDHW float output gradient");
    if (bias_gradient) {
      const auto &grad_bias = tensor_or_fail(program, op.outputs[0], op);
      if (grad_bias.dtype != grad_output.dtype ||
          grad_bias.dims != std::vector<std::uint64_t>{grad_output.dims[1]})
        fail("conv3d_backward_bias produces a [C_out] vector of the gradient "
             "dtype");
      return;
    }
    // As in two dimensions: between the operand and the result the whole
    // forward geometry is present, so it is checked the way the forward is.
    const auto &operand = tensor_or_fail(program, op.inputs[1], op);
    const auto &gradient = tensor_or_fail(program, op.outputs[0], op);
    const auto &input =
        op.opcode == ir::Opcode::Conv3dBackwardInput ? gradient : operand;
    const auto &weight =
        op.opcode == ir::Opcode::Conv3dBackwardInput ? operand : gradient;
    if (operand.dtype != grad_output.dtype ||
        gradient.dtype != grad_output.dtype || input.dims.size() != 5U ||
        weight.dims.size() != 5U)
      fail("conv3d gradients require NCDHW activations and OIDHW weights of "
           "one float dtype");
    const std::array<std::uint64_t, 3> stride{op.u64(AttrKey::StrideT, 1U),
                                              op.u64(AttrKey::StrideH, 1U),
                                              op.u64(AttrKey::StrideW, 1U)};
    const std::array<std::uint64_t, 3> dilation{
        op.u64(AttrKey::DilationT, 1U), op.u64(AttrKey::DilationH, 1U),
        op.u64(AttrKey::DilationW, 1U)};
    const std::array<std::uint64_t, 3> pad_low{op.u64(AttrKey::PadFront, 0U),
                                               op.u64(AttrKey::PadTop, 0U),
                                               op.u64(AttrKey::PadWest, 0U)};
    const std::array<std::uint64_t, 3> pad_high{op.u64(AttrKey::PadBack, 0U),
                                                op.u64(AttrKey::PadBottom, 0U),
                                                op.u64(AttrKey::PadEast, 0U)};
    const auto groups = op.u64(AttrKey::Groups, 1U);
    constexpr auto kLimit = std::uint64_t{1} << 20U;
    bool valid = groups != 0U;
    for (std::size_t axis = 0U; axis < 3U; ++axis)
      valid = valid && stride[axis] != 0U && dilation[axis] != 0U &&
              stride[axis] <= kLimit && dilation[axis] <= kLimit &&
              pad_low[axis] <= kLimit && pad_high[axis] <= kLimit;
    if (!valid)
      fail("conv3d gradient attributes are invalid or out of range");
    const auto in_channels = input.dims[1];
    const auto out_channels = weight.dims[0];
    if (in_channels % groups != 0U || out_channels % groups != 0U ||
        weight.dims[1] != in_channels / groups || weight.dims[2] == 0U ||
        weight.dims[3] == 0U || weight.dims[4] == 0U)
      fail("conv3d gradient weight/groups geometry is invalid");
    std::vector<std::uint64_t> expected{input.dims[0], out_channels, 0U, 0U,
                                        0U};
    for (std::size_t axis = 0U; axis < 3U; ++axis) {
      const auto effective =
          dilation[axis] * (weight.dims[axis + 2U] - 1U) + 1U;
      const auto padded =
          input.dims[axis + 2U] + pad_low[axis] + pad_high[axis];
      if (padded < effective)
        fail("conv3d gradient kernel does not fit its padded input");
      expected[axis + 2U] = (padded - effective) / stride[axis] + 1U;
    }
    if (grad_output.dims != expected)
      fail("conv3d gradient geometry does not match its attributes");
    return;
  }

  if (op.opcode == Opcode::Conv2dBackwardInput ||
      op.opcode == Opcode::Conv2dBackwardWeight ||
      op.opcode == Opcode::Conv2dBackwardBias) {
    const bool bias_gradient = op.opcode == Opcode::Conv2dBackwardBias;
    expect_counts(op, bias_gradient ? 1U : 2U, 1U);
    check_accumulator_f32(op);
    const auto &grad_output = tensor_or_fail(program, op.inputs[0], op);
    if (!supported_float(grad_output.dtype) || grad_output.dims.size() != 4U)
      fail("conv2d gradients require an NCHW float output gradient");
    if (bias_gradient) {
      const auto &grad_bias = tensor_or_fail(program, op.outputs[0], op);
      if (grad_bias.dtype != grad_output.dtype ||
          grad_bias.dims !=
              std::vector<std::uint64_t>{grad_output.dims[1]})
        fail("conv2d_backward_bias produces a [C_out] vector of the gradient "
             "dtype");
      return;
    }
    // The other two see one of (weight, input) and produce the other's
    // gradient, so between the operand and the result the whole forward
    // geometry is present and can be checked the way the forward is.
    const auto &operand = tensor_or_fail(program, op.inputs[1], op);
    const auto &gradient = tensor_or_fail(program, op.outputs[0], op);
    const auto &input =
        op.opcode == ir::Opcode::Conv2dBackwardInput ? gradient : operand;
    const auto &weight =
        op.opcode == ir::Opcode::Conv2dBackwardInput ? operand : gradient;
    if (operand.dtype != grad_output.dtype ||
        gradient.dtype != grad_output.dtype || input.dims.size() != 4U ||
        weight.dims.size() != 4U)
      fail("conv2d gradients require NCHW activations and OIHW weights of "
           "one float dtype");
    const auto stride_h = op.u64(AttrKey::StrideH, 1U);
    const auto stride_w = op.u64(AttrKey::StrideW, 1U);
    const auto dilation_h = op.u64(AttrKey::DilationH, 1U);
    const auto dilation_w = op.u64(AttrKey::DilationW, 1U);
    const auto pad_top = op.u64(AttrKey::PadTop, 0U);
    const auto pad_bottom = op.u64(AttrKey::PadBottom, 0U);
    const auto pad_west = op.u64(AttrKey::PadWest, 0U);
    const auto pad_east = op.u64(AttrKey::PadEast, 0U);
    const auto groups = op.u64(AttrKey::Groups, 1U);
    constexpr auto kLimit = std::uint64_t{1} << 20U;
    if (stride_h == 0U || stride_w == 0U || dilation_h == 0U ||
        dilation_w == 0U || groups == 0U || stride_h > kLimit ||
        stride_w > kLimit || dilation_h > kLimit || dilation_w > kLimit ||
        pad_top > kLimit || pad_bottom > kLimit || pad_west > kLimit ||
        pad_east > kLimit)
      fail("conv2d gradient attributes are invalid or out of range");
    const auto in_channels = input.dims[1];
    const auto out_channels = weight.dims[0];
    if (in_channels % groups != 0U || out_channels % groups != 0U ||
        weight.dims[1] != in_channels / groups || weight.dims[2] == 0U ||
        weight.dims[3] == 0U)
      fail("conv2d gradient weight/groups geometry is invalid");
    const auto effective_h = dilation_h * (weight.dims[2] - 1U) + 1U;
    const auto effective_w = dilation_w * (weight.dims[3] - 1U) + 1U;
    const auto padded_h = input.dims[2] + pad_top + pad_bottom;
    const auto padded_w = input.dims[3] + pad_west + pad_east;
    if (padded_h < effective_h || padded_w < effective_w)
      fail("conv2d gradient kernel does not fit its padded input");
    if (grad_output.dims !=
        std::vector<std::uint64_t>{input.dims[0], out_channels,
                                   (padded_h - effective_h) / stride_h + 1U,
                                   (padded_w - effective_w) / stride_w + 1U})
      fail("conv2d gradient output geometry does not match its attributes");
    return;
  }

  if (op.opcode == Opcode::PadConstant) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        (input.dims.size() != 4U && input.dims.size() != 5U) ||
        out.dims.size() != input.dims.size())
      fail("pad_constant requires matching float NCHW or NCDHW tensors");
    const auto front = op.u64(AttrKey::PadFront, 0U);
    const auto back = op.u64(AttrKey::PadBack, 0U);
    const auto top = op.u64(AttrKey::PadTop, 0U);
    const auto bottom = op.u64(AttrKey::PadBottom, 0U);
    const auto west = op.u64(AttrKey::PadWest, 0U);
    const auto east = op.u64(AttrKey::PadEast, 0U);
    constexpr auto kLimit = std::uint64_t{1} << 20U;
    if (front > kLimit || back > kLimit || top > kLimit ||
        bottom > kLimit || west > kLimit || east > kLimit ||
        !std::isfinite(op.f64(AttrKey::Value, 0.0)))
      fail("pad_constant attributes are invalid or out of range");
    auto expected = input.dims;
    if (expected.size() == 5U)
      expected[2] += front + back;
    else if (front != 0U || back != 0U)
      fail("rank-4 pad_constant cannot use front/back padding");
    const auto height_axis = expected.size() - 2U;
    const auto width_axis = expected.size() - 1U;
    expected[height_axis] += top + bottom;
    expected[width_axis] += west + east;
    if (out.dims != expected)
      fail("pad_constant output geometry does not match its attributes");
    return;
  }

  if (op.opcode == Opcode::Conv3d) {
    if ((op.inputs.size() != 2U && op.inputs.size() != 3U) ||
        op.outputs.size() != 1U)
      fail("conv3d expects input, weight, optional bias, and one output");
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || weight.dtype != input.dtype ||
        out.dtype != input.dtype || input.dims.size() != 5U ||
        weight.dims.size() != 5U || out.dims.size() != 5U)
      fail("conv3d requires NCDHW input/output and OIDHW weights of one float dtype");
    const auto stride_t = op.u64(AttrKey::StrideT, 1U);
    const auto stride_h = op.u64(AttrKey::StrideH, 1U);
    const auto stride_w = op.u64(AttrKey::StrideW, 1U);
    const auto dilation_t = op.u64(AttrKey::DilationT, 1U);
    const auto dilation_h = op.u64(AttrKey::DilationH, 1U);
    const auto dilation_w = op.u64(AttrKey::DilationW, 1U);
    const auto front = op.u64(AttrKey::PadFront, 0U);
    const auto back = op.u64(AttrKey::PadBack, 0U);
    const auto top = op.u64(AttrKey::PadTop, 0U);
    const auto bottom = op.u64(AttrKey::PadBottom, 0U);
    const auto west = op.u64(AttrKey::PadWest, 0U);
    const auto east = op.u64(AttrKey::PadEast, 0U);
    const auto groups = op.u64(AttrKey::Groups, 1U);
    constexpr auto kLimit = std::uint64_t{1} << 20U;
    if (stride_t == 0U || stride_h == 0U || stride_w == 0U ||
        dilation_t == 0U || dilation_h == 0U || dilation_w == 0U ||
        groups == 0U || stride_t > kLimit || stride_h > kLimit ||
        stride_w > kLimit || dilation_t > kLimit ||
        dilation_h > kLimit || dilation_w > kLimit || front > kLimit ||
        back > kLimit || top > kLimit || bottom > kLimit || west > kLimit ||
        east > kLimit)
      fail("conv3d attributes are invalid or out of range");
    const auto in_channels = input.dims[1];
    const auto out_channels = weight.dims[0];
    if (in_channels % groups != 0U || out_channels % groups != 0U ||
        weight.dims[1] != in_channels / groups || weight.dims[2] == 0U ||
        weight.dims[3] == 0U || weight.dims[4] == 0U)
      fail("conv3d weight/groups geometry is invalid");
    const auto effective_t = dilation_t * (weight.dims[2] - 1U) + 1U;
    const auto effective_h = dilation_h * (weight.dims[3] - 1U) + 1U;
    const auto effective_w = dilation_w * (weight.dims[4] - 1U) + 1U;
    const auto padded_t = input.dims[2] + front + back;
    const auto padded_h = input.dims[3] + top + bottom;
    const auto padded_w = input.dims[4] + west + east;
    if (padded_t < effective_t || padded_h < effective_h ||
        padded_w < effective_w)
      fail("conv3d kernel does not fit its padded input");
    const std::vector<std::uint64_t> expected{
        input.dims[0], out_channels,
        (padded_t - effective_t) / stride_t + 1U,
        (padded_h - effective_h) / stride_h + 1U,
        (padded_w - effective_w) / stride_w + 1U};
    if (out.dims != expected)
      fail("conv3d output geometry does not match its attributes");
    if (op.inputs.size() == 3U) {
      const auto &bias = tensor_or_fail(program, op.inputs[2], op);
      if (bias.dtype != input.dtype ||
          bias.dims != std::vector<std::uint64_t>{out_channels})
        fail("conv3d bias must be a [C_out] vector of the input dtype");
    }
    return;
  }

  if (op.opcode == Opcode::ChannelRmsNorm) {
    expect_counts(op, 2, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &gamma = tensor_or_fail(program, op.inputs[1], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    const auto axis = op.u64(AttrKey::Axis, 1U);
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-12);
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (!supported_float(input.dtype) || input.dims.size() < 2U ||
        axis >= input.dims.size() || gamma.dtype != input.dtype ||
        gamma.dims != std::vector<std::uint64_t>{input.dims[axis]} ||
        !(epsilon > 0.0) || block < input.dims[axis] || block > 1024U ||
        (block & (block - 1U)) != 0U)
      fail("channel_rms_norm requires float input/output, [C] gamma, a valid axis, and positive epsilon");
    return;
  }

  if (op.opcode == Opcode::GroupNorm) {
    expect_counts(op, 3, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &weight = tensor_or_fail(program, op.inputs[1], op);
    const auto &bias = tensor_or_fail(program, op.inputs[2], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    same_shape_dtype(input, out, op);
    const auto groups = op.u64(AttrKey::Groups, 1U);
    const auto epsilon = op.f64(AttrKey::Epsilon, 1.0e-5);
    const auto block = op.u64(AttrKey::BlockSize, 256U);
    if (!supported_float(input.dtype) ||
        (input.dims.size() != 4U && input.dims.size() != 5U) ||
        weight.dtype != input.dtype || bias.dtype != input.dtype ||
        weight.dims != std::vector<std::uint64_t>{input.dims[1]} ||
        bias.dims != weight.dims || groups == 0U ||
        input.dims[1] % groups != 0U || !(epsilon > 0.0) ||
        !std::isfinite(epsilon) || block == 0U || block > 1024U ||
        (block & (block - 1U)) != 0U)
      fail("group_norm requires matching NCHW/NCDHW float tensors, [C] affine vectors, valid groups, and positive epsilon");
    return;
  }

  if (op.opcode == Opcode::PadReflect) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    if (!supported_float(input.dtype) || out.dtype != input.dtype ||
        (input.dims.size() != 4U && input.dims.size() != 5U) ||
        out.dims.size() != input.dims.size())
      fail("pad_reflect requires matching float NCHW or NCDHW tensors");
    const auto front = op.u64(AttrKey::PadFront, 0U);
    const auto back = op.u64(AttrKey::PadBack, 0U);
    const auto top = op.u64(AttrKey::PadTop, 0U);
    const auto bottom = op.u64(AttrKey::PadBottom, 0U);
    const auto west = op.u64(AttrKey::PadWest, 0U);
    const auto east = op.u64(AttrKey::PadEast, 0U);
    if ((input.dims.size() == 4U && (front != 0U || back != 0U)) ||
        (input.dims.size() == 5U &&
         (front >= input.dims[2] || back >= input.dims[2])) ||
        top >= input.dims[input.dims.size() - 2U] ||
        bottom >= input.dims[input.dims.size() - 2U] ||
        west >= input.dims.back() || east >= input.dims.back())
      fail("pad_reflect extents must be smaller than their source dimensions");
    auto expected = input.dims;
    if (expected.size() == 5U)
      expected[2] += front + back;
    expected[expected.size() - 2U] += top + bottom;
    expected.back() += west + east;
    if (out.dims != expected)
      fail("pad_reflect output geometry does not match its attributes");
    return;
  }

  if (op.opcode == Opcode::UpsampleNearest2d) {
    expect_counts(op, 1, 1);
    const auto &input = tensor_or_fail(program, op.inputs[0], op);
    const auto &out = tensor_or_fail(program, op.outputs[0], op);
    const auto scale_h = op.u64(AttrKey::ScaleH, 1U);
    const auto scale_w = op.u64(AttrKey::ScaleW, 1U);
    if (!supported_float(input.dtype) || input.dims.size() != 4U ||
        out.dtype != input.dtype || out.dims.size() != 4U || scale_h == 0U ||
        scale_w == 0U || scale_h > 1024U || scale_w > 1024U ||
        out.dims != std::vector<std::uint64_t>{
                        input.dims[0], input.dims[1],
                        input.dims[2] * scale_h, input.dims[3] * scale_w})
      fail("upsample_nearest_2d requires NCHW float tensors and valid integer scales");
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
