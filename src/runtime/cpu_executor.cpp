#include "dif/runtime/executor.hpp"
#include "dif/telemetry/trace_sink.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include "dif/ir/verify.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

namespace dif::runtime {
namespace {

float round_to_storage_dtype(float value, ir::DType dtype);

void validate_bound_inputs(const ir::Program &program, const TensorMap &inputs) {
  for (const auto &desc : program.tensors) {
    if (!desc.has_role(ir::TensorRole::Input) &&
        !desc.has_role(ir::TensorRole::Constant))
      continue;
    const auto it = inputs.find(desc.id);
    if (it == inputs.end())
      fail("missing input tensor " + std::to_string(desc.id));
    it->second.validate();
    if (it->second.dtype != desc.dtype || it->second.dims != desc.dims)
      fail("bound tensor shape/dtype mismatch for id " + std::to_string(desc.id));
  }
}

TensorMap initialize(const ir::Program &program, const TensorMap &inputs) {
  TensorMap tensors = inputs;
  for (const auto &desc : program.tensors) {
    if (!tensors.contains(desc.id))
      tensors.emplace(desc.id, zeros(desc));
  }
  return tensors;
}

void elementwise(const ir::Operation &op, TensorMap &tensors, bool multiply) {
  const auto &a = tensors.at(op.inputs[0]);
  const auto &b = tensors.at(op.inputs[1]);
  auto &out = tensors.at(op.outputs[0]);
  for (std::uint64_t i = 0; i < out.element_count(); ++i) {
    const auto av = load_float(a, i);
    const auto bv = load_float(b, i);
    store_float(out, i, multiply ? av * bv : av + bv);
  }
}

void affine_last_dim(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &scale = tensors.at(op.inputs[1]);
  const auto *bias =
      op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const auto width = input.dims.back();
  for (std::uint64_t index = 0; index < out.element_count(); ++index) {
    const auto column = index % width;
    const auto scaled = round_to_storage_dtype(
        load_float(input, index) * load_float(scale, column), out.dtype);
    store_float(out, index,
                bias ? scaled + load_float(*bias, column) : scaled);
  }
}

void clamp(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto lower = static_cast<float>(op.f64(
      ir::AttrKey::Lower, -std::numeric_limits<double>::infinity()));
  const auto upper = static_cast<float>(op.f64(
      ir::AttrKey::Upper, std::numeric_limits<double>::infinity()));
  for (std::uint64_t index = 0; index < out.element_count(); ++index)
    store_float(out, index,
                std::clamp(load_float(input, index), lower, upper));
}

void silu(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  for (std::uint64_t i = 0; i < out.element_count(); ++i) {
    const auto value = load_float(input, i);
    store_float(out, i, value / (1.0F + std::exp(-value)));
  }
}

void gelu(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto approximation = static_cast<ir::GeluApproximation>(
      op.u64(ir::AttrKey::Approximation, 0U));
  constexpr float kSqrtTwoOverPi = 0.7978845608028654F;
  constexpr float kCubicCoefficient = 0.044715F;
  for (std::uint64_t i = 0; i < out.element_count(); ++i) {
    const auto value = load_float(input, i);
    if (approximation == ir::GeluApproximation::ExactErf) {
      store_float(out, i,
                  0.5F * value *
                      (1.0F + std::erf(value * 0.7071067811865475F)));
    } else {
      const auto cubic = value * value * value;
      const auto inner =
          kSqrtTwoOverPi * (value + kCubicCoefficient * cubic);
      store_float(out, i, 0.5F * value * (1.0F + std::tanh(inner)));
    }
  }
}

void sigmoid(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  for (std::uint64_t index = 0; index < out.element_count(); ++index) {
    const auto value = load_float(input, index);
    store_float(out, index, 1.0F / (1.0F + std::exp(-value)));
  }
}

void reshape(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  std::memcpy(out.mutable_data(), input.data(),
              static_cast<std::size_t>(out.element_count() *
                                       ir::dtype_size(out.dtype)));
}

void broadcast_to(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  std::vector<std::uint64_t> input_strides(input.dims.size(), 1U);
  for (std::size_t axis = input.dims.size(); axis-- > 1U;)
    input_strides[axis - 1U] = input_strides[axis] * input.dims[axis];
  const auto rank_pad = out.dims.size() - input.dims.size();
  for (std::uint64_t output_index = 0; output_index < out.element_count();
       ++output_index) {
    auto coordinate = output_index;
    std::uint64_t input_index = 0U;
    for (std::size_t axis = out.dims.size(); axis-- > 0U;) {
      const auto at_axis = coordinate % out.dims[axis];
      coordinate /= out.dims[axis];
      if (axis >= rank_pad) {
        const auto source_axis = axis - rank_pad;
        if (input.dims[source_axis] != 1U)
          input_index += at_axis * input_strides[source_axis];
      }
    }
    store_float(out, output_index, load_float(input, input_index));
  }
}

void slice_tensor(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto axis =
      static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  const auto start = op.u64(ir::AttrKey::Start, 0U);
  std::vector<std::uint64_t> input_strides(input.dims.size(), 1U);
  for (std::size_t index = input.dims.size(); index-- > 1U;)
    input_strides[index - 1U] = input_strides[index] * input.dims[index];
  for (std::uint64_t output_index = 0; output_index < out.element_count();
       ++output_index) {
    auto coordinate = output_index;
    std::uint64_t input_index = 0U;
    for (std::size_t index = out.dims.size(); index-- > 0U;) {
      auto at_axis = coordinate % out.dims[index];
      coordinate /= out.dims[index];
      if (index == axis)
        at_axis += start;
      input_index += at_axis * input_strides[index];
    }
    store_float(out, output_index, load_float(input, input_index));
  }
}

void rotary_frequency(const ir::Operation &op, TensorMap &tensors) {
  const auto &positions = tensors.at(op.inputs[0]);
  const auto &pair_axes_tensor = tensors.at(op.inputs[1]);
  const auto &pair_indices_tensor = tensors.at(op.inputs[2]);
  const auto &axis_dims_tensor = tensors.at(op.inputs[3]);
  auto &cosine = tensors.at(op.outputs[0]);
  auto &sine = tensors.at(op.outputs[1]);
  const auto *pair_axes =
      reinterpret_cast<const std::int32_t *>(pair_axes_tensor.data());
  const auto *pair_indices =
      reinterpret_cast<const std::int32_t *>(pair_indices_tensor.data());
  const auto *axis_dims =
      reinterpret_cast<const std::int32_t *>(axis_dims_tensor.data());
  const auto batch = positions.dims[0];
  const auto sequence = positions.dims[1];
  const auto axes = positions.dims[2];
  const auto pairs = pair_axes_tensor.dims[0];
  const auto theta = op.f64(ir::AttrKey::Theta, 10000.0);
  const auto ntk = op.f64(ir::AttrKey::Ntk, 1.0);
  for (std::uint64_t pair = 0; pair < pairs; ++pair) {
    if (pair_axes[pair] < 0 ||
        static_cast<std::uint64_t>(pair_axes[pair]) >= axes ||
        pair_indices[pair] < 0 || axis_dims[pair_axes[pair]] <= 0 ||
        (axis_dims[pair_axes[pair]] & 1) != 0 ||
        2 * pair_indices[pair] >= axis_dims[pair_axes[pair]])
      fail("rotary_frequency pair map is outside its declared axis");
  }
  for (std::uint64_t b = 0; b < batch; ++b) {
    for (std::uint64_t token = 0; token < sequence; ++token) {
      for (std::uint64_t pair = 0; pair < pairs; ++pair) {
        const auto axis = static_cast<std::uint64_t>(pair_axes[pair]);
        const auto scale = 2.0 * static_cast<double>(pair_indices[pair]) /
                           static_cast<double>(axis_dims[axis]);
        const auto omega = 1.0 / std::pow(theta * ntk, scale);
        const auto position = static_cast<double>(
            load_float(positions, (b * sequence + token) * axes + axis));
        const auto angle = position * omega;
        const auto index = (b * sequence + token) * pairs + pair;
        store_float(cosine, index, static_cast<float>(std::cos(angle)));
        store_float(sine, index, static_cast<float>(std::sin(angle)));
      }
    }
  }
}

void rotary_apply(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &cosine = tensors.at(op.inputs[1]);
  const auto &sine = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  const auto batch = input.dims[0];
  const auto sequence = input.dims[1];
  const auto heads = input.dims[2];
  const auto dim = input.dims[3];
  const auto pairs = cosine.dims[2];
  for (std::uint64_t b = 0; b < batch; ++b) {
    for (std::uint64_t token = 0; token < sequence; ++token) {
      const auto table_base = (b * sequence + token) * pairs;
      for (std::uint64_t head = 0; head < heads; ++head) {
        const auto base = ((b * sequence + token) * heads + head) * dim;
        for (std::uint64_t pair = 0; pair < pairs; ++pair) {
          const auto even = load_float(input, base + 2U * pair);
          const auto odd = load_float(input, base + 2U * pair + 1U);
          const auto c = load_float(cosine, table_base + pair);
          const auto s = load_float(sine, table_base + pair);
          store_float(out, base + 2U * pair, even * c - odd * s);
          store_float(out, base + 2U * pair + 1U, even * s + odd * c);
        }
        for (std::uint64_t d = 2U * pairs; d < dim; ++d)
          store_float(out, base + d, load_float(input, base + d));
      }
    }
  }
}

void boolean_mask_to_bias(const ir::Operation &op, TensorMap &tensors) {
  const auto &mask = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto *values = reinterpret_cast<const std::uint8_t *>(mask.data());
  const auto batch = out.dims[0];
  const auto sequence = out.dims[2];
  const bool vector_mask = mask.dims.size() == 2U;
  for (std::uint64_t b = 0; b < batch; ++b) {
    for (std::uint64_t query = 0; query < sequence; ++query) {
      for (std::uint64_t key = 0; key < sequence; ++key) {
        const bool valid = vector_mask
                               ? values[b * sequence + query] != 0U &&
                                     values[b * sequence + key] != 0U
                               : values[(b * sequence + query) * sequence +
                                        key] != 0U;
        store_float(out, (b * sequence + query) * sequence + key,
                    valid ? 0.0F
                          : -std::numeric_limits<float>::infinity());
      }
    }
  }
}

void mse_loss(const ir::Operation &op, TensorMap &tensors) {
  const auto &prediction = tensors.at(op.inputs[0]);
  const auto &target = tensors.at(op.inputs[1]);
  auto &loss = tensors.at(op.outputs[0]);
  float sum = 0.0F;
  for (std::uint64_t index = 0U; index < prediction.element_count(); ++index) {
    const auto difference =
        load_float(prediction, index) - load_float(target, index);
    sum = std::fma(difference, difference, sum);
  }
  store_float(loss, 0U,
              sum / static_cast<float>(prediction.element_count()));
}

void mse_loss_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &prediction = tensors.at(op.inputs[0]);
  const auto &target = tensors.at(op.inputs[1]);
  const auto &grad_loss = tensors.at(op.inputs[2]);
  auto &grad_prediction = tensors.at(op.outputs[0]);
  const auto factor = 2.0F * load_float(grad_loss, 0U) /
                      static_cast<float>(prediction.element_count());
  for (std::uint64_t index = 0U; index < prediction.element_count(); ++index)
    store_float(grad_prediction, index,
                (load_float(prediction, index) - load_float(target, index)) *
                    factor);
}

void linear_backward_input(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  auto &grad_input = tensors.at(op.outputs[0]);
  const auto inner = weight.dims[1];
  const auto outputs = weight.dims[0];
  const auto rows = grad_output.element_count() / outputs;
  for (std::uint64_t row = 0U; row < rows; ++row) {
    for (std::uint64_t column = 0U; column < inner; ++column) {
      float value = 0.0F;
      for (std::uint64_t output = 0U; output < outputs; ++output)
        value = std::fma(load_float(grad_output, row * outputs + output),
                         load_float(weight, output * inner + column), value);
      store_float(grad_input, row * inner + column, value);
    }
  }
}

void linear_backward_weight(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &input = tensors.at(op.inputs[1]);
  auto &grad_weight = tensors.at(op.outputs[0]);
  // Geometry from the [N,K] weight gradient itself: rank-agnostic over both
  // admitted operand forms (same-rank broadcast and flatten).
  const auto outputs = grad_weight.dims[0];
  const auto inner = grad_weight.dims[1];
  const auto rows = input.element_count() / inner;
  for (std::uint64_t output = 0U; output < outputs; ++output) {
    for (std::uint64_t column = 0U; column < inner; ++column) {
      float value = 0.0F;
      for (std::uint64_t row = 0U; row < rows; ++row)
        value = std::fma(load_float(grad_output, row * outputs + output),
                         load_float(input, row * inner + column), value);
      store_float(grad_weight, output * inner + column, value);
    }
  }
}

void bias_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  auto &grad_bias = tensors.at(op.outputs[0]);
  const auto width = grad_bias.dims[0];
  const auto rows = grad_output.element_count() / width;
  for (std::uint64_t column = 0U; column < width; ++column) {
    float value = 0.0F;
    for (std::uint64_t row = 0U; row < rows; ++row)
      value += load_float(grad_output, row * width + column);
    store_float(grad_bias, column, value);
  }
}

void silu_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &grad_output = tensors.at(op.inputs[1]);
  auto &grad_input = tensors.at(op.outputs[0]);
  for (std::uint64_t index = 0U; index < input.element_count(); ++index) {
    const auto value = load_float(input, index);
    const auto sigmoid = 1.0F / (1.0F + std::exp(-value));
    const auto derivative = sigmoid * (1.0F + value * (1.0F - sigmoid));
    store_float(grad_input, index,
                load_float(grad_output, index) * derivative);
  }
}

void adamw_update(const ir::Operation &op, TensorMap &tensors) {
  const auto &parameter = tensors.at(op.inputs[0]);
  const auto &gradient = tensors.at(op.inputs[1]);
  const auto &first = tensors.at(op.inputs[2]);
  const auto &second = tensors.at(op.inputs[3]);
  const auto &step_tensor = tensors.at(op.inputs[4]);
  auto &updated = tensors.at(op.outputs[0]);
  auto &updated_first = tensors.at(op.outputs[1]);
  auto &updated_second = tensors.at(op.outputs[2]);
  std::int32_t completed_steps = 0;
  std::memcpy(&completed_steps, step_tensor.data(), sizeof(completed_steps));
  if (completed_steps < 0)
    fail("adamw_update step cannot be negative");
  const auto step = static_cast<float>(completed_steps + 1);
  const auto learning_rate =
      static_cast<float>(op.f64(ir::AttrKey::LearningRate, 1.0e-3));
  const auto beta1 = static_cast<float>(op.f64(ir::AttrKey::Beta1, 0.9));
  const auto beta2 = static_cast<float>(op.f64(ir::AttrKey::Beta2, 0.999));
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-8));
  const auto weight_decay =
      static_cast<float>(op.f64(ir::AttrKey::WeightDecay, 0.0));
  const auto bias1 = 1.0F - std::pow(beta1, step);
  const auto bias2_sqrt = std::sqrt(1.0F - std::pow(beta2, step));
  for (std::uint64_t index = 0U; index < parameter.element_count(); ++index) {
    const auto grad = load_float(gradient, index);
    const auto first_value =
        beta1 * load_float(first, index) + (1.0F - beta1) * grad;
    const auto second_value = beta2 * load_float(second, index) +
                              (1.0F - beta2) * grad * grad;
    const auto decayed =
        load_float(parameter, index) * (1.0F - learning_rate * weight_decay);
    const auto denominator = std::sqrt(second_value) / bias2_sqrt + epsilon;
    const auto parameter_value =
        decayed - (learning_rate / bias1) * first_value / denominator;
    store_float(updated, index, parameter_value);
    store_float(updated_first, index, first_value);
    store_float(updated_second, index, second_value);
  }
}

void cast(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  for (std::uint64_t i = 0; i < out.element_count(); ++i)
    store_float(out, i, load_float(input, i));
}

void select_row_chunks(const ir::Operation &op, TensorMap &tensors) {
  const auto &values = tensors.at(op.inputs[0]);
  const auto &indices_tensor = tensors.at(op.inputs[1]);
  const auto *indices =
      reinterpret_cast<const std::int32_t *>(indices_tensor.data());
  const auto rows = indices_tensor.element_count();
  const auto source_rows = values.dims[0];
  const auto width = tensors.at(op.outputs[0]).dims[1];
  const auto source_width = values.dims[1];
  for (std::uint64_t row = 0; row < rows; ++row) {
    if (indices[row] < 0 ||
        static_cast<std::uint64_t>(indices[row]) >= source_rows)
      fail("select_row_chunks index is out of range");
    const auto source_row = static_cast<std::uint64_t>(indices[row]);
    for (std::size_t chunk = 0; chunk < op.outputs.size(); ++chunk) {
      auto &output = tensors.at(op.outputs[chunk]);
      for (std::uint64_t column = 0; column < width; ++column)
        store_float(output, row * width + column,
                    load_float(values, source_row * source_width +
                                           chunk * width + column));
    }
  }
}

void sinusoidal_timestep(const ir::Operation &op, TensorMap &tensors) {
  const auto &timesteps = tensors.at(op.inputs[0]);
  auto &output = tensors.at(op.outputs[0]);
  const auto rows = timesteps.dims[0];
  const auto width = output.dims[1];
  const auto half = width / 2U;
  const auto flip = op.boolean(ir::AttrKey::FlipSinToCos, false);
  const auto shift = static_cast<float>(
      op.f64(ir::AttrKey::DownscaleFreqShift, 1.0));
  const auto scale = static_cast<float>(op.f64(ir::AttrKey::Scale, 1.0));
  const auto max_period =
      static_cast<float>(op.f64(ir::AttrKey::MaxPeriod, 10000.0));
  const auto log_period = static_cast<float>(std::log(max_period));
  const auto denominator = static_cast<float>(half) - shift;
  for (std::uint64_t row = 0; row < rows; ++row) {
    const auto timestep = load_float(timesteps, row);
    for (std::uint64_t column = 0; column < half; ++column) {
      const auto exponent =
          -log_period * static_cast<float>(column) / denominator;
      const auto frequency = std::exp(exponent);
      // The creator materializes `(t.float() * tfactor)` before broadcasting
      // the frequency multiply. Preserve that F32 boundary and operation order.
      volatile float scaled_timestep = timestep * scale;
      const auto angle = scaled_timestep * frequency;
      const auto sine = std::sin(angle);
      const auto cosine = std::cos(angle);
      store_float(output, row * width + column, flip ? cosine : sine);
      store_float(output, row * width + half + column,
                  flip ? sine : cosine);
    }
    if ((width % 2U) != 0U)
      store_float(output, row * width + width - 1U, 0.0F);
  }
}

void rotary_position(const ir::Operation &op, TensorMap &tensors) {
  const auto &positions = tensors.at(op.inputs[0]);
  const auto &inv_freq = tensors.at(op.inputs[1]);
  auto &cosine = tensors.at(op.outputs[0]);
  auto &sine = tensors.at(op.outputs[1]);
  const auto rows = positions.dims[0];
  const auto axes = positions.dims[1];
  const auto frequencies = inv_freq.dims[0];
  const auto unrepeated_width = axes * frequencies;
  const auto width = 2U * unrepeated_width;
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t column = 0; column < width; ++column) {
      const auto component = column % unrepeated_width;
      const auto axis = component / frequencies;
      const auto frequency = component % frequencies;
      const auto angle = load_float(positions, row * axes + axis) *
                         load_float(inv_freq, frequency);
      store_float(cosine, row * width + column, std::cos(angle));
      store_float(sine, row * width + column, std::sin(angle));
    }
  }
}

void linear_blend(const ir::Operation &op, TensorMap &tensors) {
  const auto &left = tensors.at(op.inputs[0]);
  const auto &right = tensors.at(op.inputs[1]);
  const auto factor = load_float(tensors.at(op.inputs[2]), 0U);
  auto &output = tensors.at(op.outputs[0]);
  for (std::uint64_t index = 0; index < output.element_count(); ++index) {
    const auto left_value = load_float(left, index);
    const auto right_value = load_float(right, index);
    // LinearBlend has explicit F32 operation boundaries.  This matches eager
    // tensor runtimes, where the two multiplies and final add are distinct
    // elementwise operations, and prevents host contraction from changing the
    // source-faithful result by one ULP.
    volatile float complement = 1.0F - factor;
    volatile float weighted_left = factor * left_value;
    volatile float weighted_right = complement * right_value;
    volatile float blended = weighted_left + weighted_right;
    store_float(output, index, blended);
  }
}

void flow_euler_step(const ir::Operation &op, TensorMap &tensors) {
  const auto &sample = tensors.at(op.inputs[0]);
  const auto &velocity = tensors.at(op.inputs[1]);
  const auto &timesteps = tensors.at(op.inputs[2]);
  const auto &sigmas = tensors.at(op.inputs[3]);
  auto &output = tensors.at(op.outputs[0]);
  const auto step = op.u64(ir::AttrKey::StepIndex, 0U);
  const auto timestep = load_float(timesteps, step);
  const auto sigma = load_float(sigmas, step);
  const auto sigma_next = load_float(sigmas, step + 1U);
  const auto sigma_from_timestep = 1.0F - timestep;
  const auto ratio = sigma_next / sigma;
  for (std::uint64_t index = 0; index < output.element_count(); ++index) {
    const auto sample_value = load_float(sample, index);
    const auto velocity_value = load_float(velocity, index);
    // Preserve the same explicit F32 rounding boundaries as the released
    // scheduler's eager tensor operations.
    volatile float velocity_delta = sigma_from_timestep * velocity_value;
    volatile float denoised = sample_value + velocity_delta;
    volatile float complement = 1.0F - ratio;
    volatile float weighted_sample = ratio * sample_value;
    volatile float weighted_denoised = complement * denoised;
    volatile float blended = weighted_sample + weighted_denoised;
    store_float(output, index, blended);
  }
}

float round_to_storage_dtype(float value, ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return value;
  if (dtype == ir::DType::BF16)
    return bf16_to_float(float_to_bf16(value));
  if (dtype == ir::DType::F16)
    return f16_to_float(float_to_f16(value));
  fail("Euler velocity storage dtype is not floating point");
}

void euler_velocity_step(const ir::Operation &op, TensorMap &tensors) {
  const auto &sample = tensors.at(op.inputs[0]);
  const auto &velocity = tensors.at(op.inputs[1]);
  const auto current = load_float(tensors.at(op.inputs[2]), 0U);
  const auto next = load_float(tensors.at(op.inputs[3]), 0U);
  auto &output = tensors.at(op.outputs[0]);
  // The creator performs two eager BF16 operations. Preserve the multiply's
  // storage boundary before the residual add instead of contracting the
  // expression into a single F32 FMA.
  volatile float delta_t = next - current;
  for (std::uint64_t index = 0; index < output.element_count(); ++index) {
    volatile float scaled = delta_t * load_float(velocity, index);
    const auto rounded = round_to_storage_dtype(scaled, sample.dtype);
    volatile float updated = load_float(sample, index) + rounded;
    store_float(output, index, updated);
  }
}

void permute(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &output = tensors.at(op.outputs[0]);
  const auto rank = input.dims.size();
  std::vector<std::uint64_t> input_strides(rank, 1U);
  for (std::size_t axis = rank - 1U; axis > 0U; --axis)
    input_strides[axis - 1U] = input_strides[axis] * input.dims[axis];
  for (std::uint64_t index = 0; index < output.element_count(); ++index) {
    auto remainder = index;
    std::uint64_t source = 0U;
    for (std::size_t reverse = rank; reverse-- > 0U;) {
      const auto coordinate = remainder % output.dims[reverse];
      remainder /= output.dims[reverse];
      const auto key = static_cast<ir::AttrKey>(
          static_cast<std::uint32_t>(ir::AttrKey::Permutation0) + reverse);
      source += coordinate * input_strides[op.u64(key, 0U)];
    }
    store_float(output, index, load_float(input, source));
  }
}

void concat(const ir::Operation &op, TensorMap &tensors) {
  auto &output = tensors.at(op.outputs[0]);
  const auto axis = static_cast<std::size_t>(op.u64(ir::AttrKey::Axis, 0U));
  const auto &first = tensors.at(op.inputs[0]);
  std::uint64_t outer = 1U;
  std::uint64_t inner = 1U;
  for (std::size_t dimension = 0U; dimension < axis; ++dimension)
    outer *= first.dims[dimension];
  for (std::size_t dimension = axis + 1U; dimension < first.dims.size();
       ++dimension)
    inner *= first.dims[dimension];
  const auto element_bytes = ir::dtype_size(first.dtype);
  const auto output_axis = output.dims[axis];
  for (std::uint64_t outer_index = 0U; outer_index < outer; ++outer_index) {
    std::uint64_t output_axis_offset = 0U;
    for (const auto input_id : op.inputs) {
      const auto &input = tensors.at(input_id);
      const auto input_axis = input.dims[axis];
      const auto elements = input_axis * inner;
      const auto source_offset = outer_index * elements;
      const auto destination_offset =
          (outer_index * output_axis + output_axis_offset) * inner;
      std::memcpy(output.mutable_data() + destination_offset * element_bytes,
                  input.data() + source_offset * element_bytes,
                  elements * element_bytes);
      output_axis_offset += input_axis;
    }
  }
}

void patchify_3d(const ir::Operation &op, TensorMap &tensors,
                 bool inverse) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &output = tensors.at(op.outputs[0]);
  const auto &volume = inverse ? output : input;
  const auto &rows = inverse ? input : output;
  const auto patch_t = op.u64(ir::AttrKey::PatchT, 0U);
  const auto patch_h = op.u64(ir::AttrKey::PatchH, 0U);
  const auto patch_w = op.u64(ir::AttrKey::PatchW, 0U);
  const auto channels = volume.dims[1];
  const auto frames = volume.dims[2];
  const auto height = volume.dims[3];
  const auto width = volume.dims[4];
  const auto output_frames = frames / patch_t;
  const auto output_height = height / patch_h;
  const auto output_width = width / patch_w;
  const auto patch_volume = patch_t * patch_h * patch_w;
  for (std::uint64_t row = 0; row < rows.dims[0]; ++row) {
    auto outer = row;
    const auto patch_x = outer % output_width;
    outer /= output_width;
    const auto patch_y = outer % output_height;
    outer /= output_height;
    const auto patch_frame = outer % output_frames;
    const auto batch = outer / output_frames;
    for (std::uint64_t column = 0; column < rows.dims[1]; ++column) {
      auto inner = column;
      const auto offset_x = inner % patch_w;
      inner /= patch_w;
      const auto offset_y = inner % patch_h;
      inner /= patch_h;
      const auto offset_t = inner % patch_t;
      inner /= patch_t;
      const auto channel = inner;
      const auto frame = patch_frame * patch_t + offset_t;
      const auto y = patch_y * patch_h + offset_y;
      const auto x = patch_x * patch_w + offset_x;
      const auto volume_index =
          ((((batch * channels + channel) * frames + frame) * height + y) *
               width +
           x);
      const auto row_index = row * channels * patch_volume + column;
      if (inverse)
        store_float(output, volume_index, load_float(input, row_index));
      else
        store_float(output, row_index, load_float(input, volume_index));
    }
  }
}

void rms_norm(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  auto &out = tensors.at(op.outputs[0]);
  const auto columns = input.dims.back();
  const auto rows = input.element_count() / columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const auto weight_offset =
      static_cast<float>(op.f64(ir::AttrKey::WeightOffset, 0.0));
  for (std::uint64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto value = load_float(input, row * columns + column);
      sum += value * value;
    }
    const auto inverse =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + epsilon);
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto index = row * columns + column;
      store_float(out, index, load_float(input, index) * inverse *
                                  (load_float(weight, column) +
                                   weight_offset));
    }
  }
}

void layer_norm(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto &bias = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  const auto columns = input.dims.back();
  const auto rows = input.element_count() / columns;
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  for (std::uint64_t row = 0; row < rows; ++row) {
    float mean = 0.0F;
    for (std::uint64_t column = 0; column < columns; ++column)
      mean += load_float(input, row * columns + column);
    mean /= static_cast<float>(columns);
    float variance = 0.0F;
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto centered =
          load_float(input, row * columns + column) - mean;
      variance += centered * centered;
    }
    const auto inverse =
        1.0F / std::sqrt(variance / static_cast<float>(columns) + epsilon);
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto index = row * columns + column;
      store_float(out, index,
                  (load_float(input, index) - mean) * inverse *
                          load_float(weight, column) +
                      load_float(bias, column));
    }
  }
}

void fill(const ir::Operation &op, TensorMap &tensors) {
  auto &out = tensors.at(op.outputs[0]);
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  for (std::uint64_t i = 0; i < out.element_count(); ++i)
    store_float(out, i, value);
}

void gather_rows(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &indices_tensor = tensors.at(op.inputs[1]);
  auto &out = tensors.at(op.outputs[0]);
  const auto *indices =
      reinterpret_cast<const std::int32_t *>(indices_tensor.data());
  const auto input_rows = input.dims[0];
  const auto output_rows = out.dims[0];
  const auto row_width = input.element_count() / input_rows;
  for (std::uint64_t row = 0; row < output_rows; ++row) {
    if (indices[row] < 0 ||
        static_cast<std::uint64_t>(indices[row]) >= input_rows)
      fail("gather_rows index is out of range");
    const auto source_row = static_cast<std::uint64_t>(indices[row]);
    for (std::uint64_t column = 0; column < row_width; ++column)
      store_float(out, row * row_width + column,
                  load_float(input, source_row * row_width + column));
  }
}

void indexed_update_rows(const ir::Operation &op, TensorMap &tensors) {
  const auto &base = tensors.at(op.inputs[0]);
  const auto &updates = tensors.at(op.inputs[1]);
  const auto &map_tensor = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  const auto *map = reinterpret_cast<const std::int32_t *>(map_tensor.data());
  const auto rows = base.dims[0];
  const auto update_rows = updates.dims[0];
  const auto row_width = base.element_count() / rows;
  for (std::uint64_t row = 0; row < rows; ++row) {
    if (map[row] < -1 ||
        (map[row] >= 0 &&
         static_cast<std::uint64_t>(map[row]) >= update_rows))
      fail("indexed_update_rows map is out of range");
    const auto &source = map[row] < 0 ? base : updates;
    const auto source_row =
        map[row] < 0 ? row : static_cast<std::uint64_t>(map[row]);
    for (std::uint64_t column = 0; column < row_width; ++column)
      store_float(out, row * row_width + column,
                  load_float(source, source_row * row_width + column));
  }
}

void rms_norm_modulate(const ir::Operation &op, TensorMap &tensors) {
  const auto &x_tensor = tensors.at(op.inputs[0]);
  const auto layout = static_cast<ir::ModulationLayout>(op.u64(
      ir::AttrKey::ModulationLayout,
      static_cast<std::uint64_t>(ir::ModulationLayout::ExplicitScaleShift)));
  if (layout == ir::ModulationLayout::SharedVectorDelta) {
    const auto &weight = tensors.at(op.inputs[1]);
    const auto &vector = tensors.at(op.inputs[2]);
    const auto &delta = tensors.at(op.inputs[3]);
    auto &out = tensors.at(op.outputs[0]);
    const auto rows = x_tensor.dims[0];
    const auto cols = x_tensor.dims[1];
    const auto vectors = vector.dims[0];
    const auto rows_per_vector = rows / vectors;
    const auto epsilon =
        static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
    const auto weight_offset =
        static_cast<float>(op.f64(ir::AttrKey::WeightOffset, 0.0));
    for (std::uint64_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      for (std::uint64_t col = 0; col < cols; ++col) {
        const auto value = load_float(x_tensor, row * cols + col);
        sum += value * value;
      }
      const auto inverse =
          1.0F / std::sqrt(sum / static_cast<float>(cols) + epsilon);
      const auto vector_base = (row / rows_per_vector) * cols;
      for (std::uint64_t col = 0; col < cols; ++col) {
        const auto base = load_float(vector, vector_base + col);
        const auto scale = base + load_float(delta, col);
        const auto shift = base + load_float(delta, cols + col);
        const auto normalized = load_float(x_tensor, row * cols + col) *
                                inverse *
                                (load_float(weight, col) + weight_offset);
        store_float(out, row * cols + col,
                    (1.0F + scale) * normalized + shift);
      }
    }
    return;
  }
  const bool weighted = op.inputs.size() == 4;
  const auto &scale = tensors.at(op.inputs[weighted ? 2 : 1]);
  const auto &shift = tensors.at(op.inputs[weighted ? 3 : 2]);
  const auto *weight = weighted ? &tensors.at(op.inputs[1]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const auto rows = x_tensor.dims[0];
  const auto cols = x_tensor.dims[1];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  for (std::uint64_t row = 0; row < rows; ++row) {
    float sum = 0.0F;
    for (std::uint64_t col = 0; col < cols; ++col) {
      const float value = load_float(x_tensor, row * cols + col);
      sum += value * value;
    }
    const float inv = 1.0F / std::sqrt(sum / static_cast<float>(cols) + epsilon);
    for (std::uint64_t col = 0; col < cols; ++col) {
      const auto index = row * cols + col;
      const float normalized =
          load_float(x_tensor, index) * inv *
          (weight ? load_float(*weight, col) : 1.0F);
      store_float(out, index,
                  normalized * (1.0F + load_float(scale, index)) +
                      load_float(shift, index));
    }
  }
}

void swiglu(const ir::Operation &op, TensorMap &tensors) {
  const auto &input_tensor = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto width = tensors.at(op.outputs[0]).dims.back();
  const auto rows = out.element_count() / width;
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t col = 0; col < width; ++col) {
      const auto base = row * width * 2U;
      const float value = load_float(
          input_tensor, base + (gate_first ? width : 0U) + col);
      const float gate = load_float(
          input_tensor, base + (gate_first ? 0U : width) + col);
      store_float(out, row * width + col,
                  value * (gate / (1.0F + std::exp(-gate))));
    }
  }
}

void bias_add(const ir::Operation &op, TensorMap &tensors) {
  const auto &input_tensor = tensors.at(op.inputs[0]);
  const auto &bias = tensors.at(op.inputs[1]);
  auto &output = tensors.at(op.outputs[0]);
  const auto width = input_tensor.dims.back();
  for (std::uint64_t i = 0; i < input_tensor.element_count(); ++i)
    store_float(output, i,
                load_float(input_tensor, i) + load_float(bias, i % width));
}

void h3_adaln_select(const ir::Operation &op, TensorMap &tensors) {
  const auto &projected = tensors.at(op.inputs[0]);
  const auto &index_tensor = tensors.at(op.inputs[1]);
  const auto *indices = reinterpret_cast<const std::int32_t *>(index_tensor.data());
  const auto &output_tensor = tensors.at(op.outputs[0]);
  const auto sequence = output_tensor.dims[0];
  const auto hidden = output_tensor.dims[1];
  const auto table_rows = tensors.at(op.inputs[0]).dims[0] * 3U;
  for (std::uint64_t chunk = 0; chunk < 6U; ++chunk) {
    auto &output = tensors.at(op.outputs[chunk]);
    for (std::uint64_t row = 0; row < sequence; ++row) {
      if (indices[row] < 0 || static_cast<std::uint64_t>(indices[row]) >= table_rows)
        fail("h3_adaln_select index is out of range");
      const auto source =
          (static_cast<std::uint64_t>(indices[row]) * 6U + chunk) * hidden;
      for (std::uint64_t col = 0; col < hidden; ++col)
        store_float(output, row * hidden + col,
                    load_float(projected, source + col));
    }
  }
}

void h3_deinterleave_qkv(const ir::Operation &op, TensorMap &tensors) {
  const auto &packed = tensors.at(op.inputs[0]);
  const auto &output_tensor = tensors.at(op.outputs[0]);
  const auto sequence = output_tensor.dims[0];
  const auto heads = output_tensor.dims[1];
  const auto dim = output_tensor.dims[2];
  const auto packed_width = 3U * heads * dim;
  for (std::uint64_t component = 0; component < 3U; ++component) {
    auto &output = tensors.at(op.outputs[component]);
    for (std::uint64_t row = 0; row < sequence; ++row) {
      for (std::uint64_t head = 0; head < heads; ++head) {
        for (std::uint64_t d = 0; d < dim; ++d) {
          store_float(
              output, (row * heads + head) * dim + d,
              load_float(
                  packed,
                  row * packed_width + (head * 3U + component) * dim + d));
        }
      }
    }
  }
}

void h3_deinterleave_qkv_weight(const ir::Operation &op, TensorMap &tensors) {
  const auto &packed = tensors.at(op.inputs[0]);
  const auto &output_tensor = tensors.at(op.outputs[0]);
  const auto heads = op.u64(ir::AttrKey::Heads, 0U);
  const auto dim = op.u64(ir::AttrKey::HeadDim, 0U);
  const auto hidden = output_tensor.dims[1];
  for (std::uint64_t component = 0; component < 3U; ++component) {
    auto &output = tensors.at(op.outputs[component]);
    for (std::uint64_t head = 0; head < heads; ++head) {
      for (std::uint64_t d = 0; d < dim; ++d) {
        for (std::uint64_t column = 0; column < hidden; ++column) {
          store_float(
              output, (head * dim + d) * hidden + column,
              load_float(
                  packed,
                  ((head * 3U + component) * dim + d) * hidden + column));
        }
      }
    }
  }
}

void dequantize_int4(const ir::Operation &op, TensorMap &tensors) {
  const auto &packed = tensors.at(op.inputs[0]);
  const auto &scales = tensors.at(op.inputs[1]);
  auto &output = tensors.at(op.outputs[0]);
  const auto rows = output.dims[0];
  const auto columns = output.dims[1];
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = columns / group;
  const auto *outlier_indices =
      op.inputs.size() == 4U ? &tensors.at(op.inputs[2]) : nullptr;
  const auto *outlier_residuals =
      op.inputs.size() == 4U ? &tensors.at(op.inputs[3]) : nullptr;
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto packed_index = row * (columns / 2U) + column / 2U;
      const auto byte = packed.data()[packed_index];
      const auto nibble = static_cast<std::uint8_t>(
          column % 2U == 0U ? byte & 0x0fU : byte >> 4U);
      const auto quantized =
          nibble < 8U ? static_cast<std::int32_t>(nibble)
                      : static_cast<std::int32_t>(nibble) - 16;
      const auto scale = load_float(scales, row * groups + column / group);
      auto value = static_cast<float>(quantized) * scale;
      const auto group_index = row * groups + column / group;
      if (outlier_indices && outlier_residuals &&
          outlier_indices->data()[group_index] == column % group)
        value += load_float(*outlier_residuals, group_index);
      store_float(output, row * columns + column, value);
    }
  }
}

void dequantize_int5(const ir::Operation &op, TensorMap &tensors) {
  const auto &packed = tensors.at(op.inputs[0]);
  const auto &scales = tensors.at(op.inputs[1]);
  auto &output = tensors.at(op.outputs[0]);
  const auto rows = output.dims[0];
  const auto columns = output.dims[1];
  const auto row_bytes = columns * 5U / 8U;
  const auto group = op.u64(ir::AttrKey::GroupSize, 64U);
  const auto groups = columns / group;
  const auto *column_scales =
      op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto bit_offset = column * 5U;
      const auto byte_index = row * row_bytes + bit_offset / 8U;
      const auto shift = static_cast<unsigned>(bit_offset % 8U);
      std::uint16_t word = packed.data()[byte_index];
      if (shift + 5U > 8U)
        word |= static_cast<std::uint16_t>(packed.data()[byte_index + 1U])
                << 8U;
      const auto encoded = static_cast<std::uint8_t>((word >> shift) & 0x1fU);
      const auto quantized =
          encoded < 16U ? static_cast<std::int32_t>(encoded)
                        : static_cast<std::int32_t>(encoded) - 32;
      const auto scale = load_float(scales, row * groups + column / group);
      auto value = static_cast<float>(quantized) * scale;
      if (column_scales)
        value *= load_float(*column_scales, column);
      store_float(output, row * columns + column, value);
    }
  }
}

void rms_norm_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &input = tensors.at(op.inputs[1]);
  const auto &weight = tensors.at(op.inputs[2]);
  auto &grad_input = tensors.at(op.outputs[0]);
  auto *grad_weight =
      op.outputs.size() == 2U ? &tensors.at(op.outputs[1]) : nullptr;
  const auto columns = input.dims.back();
  const auto rows = input.element_count() / columns;
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::vector<float> weight_accumulator(grad_weight ? columns : 0U, 0.0F);
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto base = row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto value = load_float(input, base + column);
      sum += value * value;
    }
    const auto inverse =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + epsilon);
    float dot = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column)
      dot = std::fma(load_float(grad_output, base + column) *
                         load_float(weight, column),
                     load_float(input, base + column), dot);
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto value = load_float(input, base + column);
      const auto gradient = load_float(grad_output, base + column) *
                                load_float(weight, column) * inverse -
                            value * inverse * inverse * inverse * dot /
                                static_cast<float>(columns);
      store_float(grad_input, base + column, gradient);
    }
    if (grad_weight)
      for (std::uint64_t column = 0U; column < columns; ++column)
        weight_accumulator[column] +=
            load_float(grad_output, base + column) *
            load_float(input, base + column) * inverse;
  }
  if (grad_weight)
    for (std::uint64_t column = 0U; column < columns; ++column)
      store_float(*grad_weight, column, weight_accumulator[column]);
}

void rms_norm_modulate_backward(const ir::Operation &op, TensorMap &tensors) {
  const bool weighted = op.inputs.size() == 4U;
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &x_tensor = tensors.at(op.inputs[1]);
  const auto *weight = weighted ? &tensors.at(op.inputs[2]) : nullptr;
  const auto &scale = tensors.at(op.inputs[weighted ? 3U : 2U]);
  auto &grad_input = tensors.at(op.outputs[0]);
  auto &grad_scale = tensors.at(op.outputs[1]);
  auto &grad_shift = tensors.at(op.outputs[2]);
  auto *grad_weight = weighted ? &tensors.at(op.outputs[3]) : nullptr;
  const auto rows = x_tensor.dims[0];
  const auto columns = x_tensor.dims[1];
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::vector<float> weight_accumulator(grad_weight ? columns : 0U, 0.0F);
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto base = row * columns;
    float sum = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto value = load_float(x_tensor, base + column);
      sum += value * value;
    }
    const auto inverse =
        1.0F / std::sqrt(sum / static_cast<float>(columns) + epsilon);
    float dot = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto normed_gradient =
          load_float(grad_output, base + column) *
          (1.0F + load_float(scale, base + column)) *
          (weight ? load_float(*weight, column) : 1.0F);
      dot = std::fma(normed_gradient, load_float(x_tensor, base + column),
                     dot);
    }
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto index = base + column;
      const auto value = load_float(x_tensor, index);
      const auto upstream = load_float(grad_output, index);
      const auto weight_value = weight ? load_float(*weight, column) : 1.0F;
      const auto normed_gradient =
          upstream * (1.0F + load_float(scale, index));
      store_float(grad_input, index,
                  normed_gradient * weight_value * inverse -
                      value * inverse * inverse * inverse * dot /
                          static_cast<float>(columns));
      store_float(grad_scale, index,
                  upstream * value * inverse * weight_value);
      store_float(grad_shift, index, upstream);
      if (grad_weight)
        weight_accumulator[column] += normed_gradient * value * inverse;
    }
  }
  if (grad_weight)
    for (std::uint64_t column = 0U; column < columns; ++column)
      store_float(*grad_weight, column, weight_accumulator[column]);
}

void swiglu_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &input = tensors.at(op.inputs[1]);
  auto &grad_input = tensors.at(op.outputs[0]);
  const auto width = grad_output.dims.back();
  const auto rows = grad_output.element_count() / width;
  const bool gate_first = op.boolean(ir::AttrKey::GateFirst, false);
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto base = row * width * 2U;
    for (std::uint64_t column = 0U; column < width; ++column) {
      const auto value_index = base + (gate_first ? width : 0U) + column;
      const auto gate_index = base + (gate_first ? 0U : width) + column;
      const auto value = load_float(input, value_index);
      const auto gate = load_float(input, gate_index);
      const auto sigmoid = 1.0F / (1.0F + std::exp(-gate));
      const auto upstream = load_float(grad_output, row * width + column);
      store_float(grad_input, value_index, gate * sigmoid * upstream);
      store_float(grad_input, gate_index,
                  sigmoid * (1.0F + gate * (1.0F - sigmoid)) * value *
                      upstream);
    }
  }
}

// Backward of qk_norm_rope: apply the transpose of the recorded rotation to
// the upstream gradient (per pair (d, d+half): ga = c1*g0 + s2*g1,
// gb = -s1*g0 + c2*g1 — the exact adjoint of out0 = c1*a - s1*b,
// out1 = s2*a + c2*b, honoring the duplicated-table convention), then the
// per-head RMSNorm backward.  Layout comes from RotaryDim and the table
// width, never from shape sniffing.
void qk_norm_rope_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &input = tensors.at(op.inputs[1]);
  const auto &weight = tensors.at(op.inputs[2]);
  const auto &cosv = tensors.at(op.inputs[3]);
  const auto &sinv = tensors.at(op.inputs[4]);
  auto &grad_input = tensors.at(op.outputs[0]);
  auto *grad_weight =
      op.outputs.size() == 2U ? &tensors.at(op.outputs[1]) : nullptr;
  const auto sequence = input.dims[0];
  const auto heads = input.dims[1];
  const auto dim = input.dims[2];
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = cosv.dims[1];
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::vector<float> rotated_gradient(dim);
  std::vector<float> weight_accumulator(grad_weight ? dim : 0U, 0.0F);
  for (std::uint64_t sequence_index = 0U; sequence_index < sequence;
       ++sequence_index) {
    const auto table = sequence_index * table_width;
    for (std::uint64_t head = 0U; head < heads; ++head) {
      const auto base = (sequence_index * heads + head) * dim;
      for (std::uint64_t d = 0U; d < half; ++d) {
        const auto second_index = table_width == rotary ? d + half : d;
        const auto cos_first = load_float(cosv, table + d);
        const auto sin_first = load_float(sinv, table + d);
        const auto cos_second = load_float(cosv, table + second_index);
        const auto sin_second = load_float(sinv, table + second_index);
        const auto upstream_first = load_float(grad_output, base + d);
        const auto upstream_second =
            load_float(grad_output, base + d + half);
        rotated_gradient[d] =
            upstream_first * cos_first + upstream_second * sin_second;
        rotated_gradient[d + half] =
            -upstream_first * sin_first + upstream_second * cos_second;
      }
      for (std::uint64_t d = rotary; d < dim; ++d)
        rotated_gradient[d] = load_float(grad_output, base + d);
      float sum = 0.0F;
      for (std::uint64_t d = 0U; d < dim; ++d) {
        const auto value = load_float(input, base + d);
        sum += value * value;
      }
      const auto inverse =
          1.0F / std::sqrt(sum / static_cast<float>(dim) + epsilon);
      float dot = 0.0F;
      for (std::uint64_t d = 0U; d < dim; ++d)
        dot = std::fma(rotated_gradient[d] * load_float(weight, d),
                       load_float(input, base + d), dot);
      for (std::uint64_t d = 0U; d < dim; ++d) {
        const auto value = load_float(input, base + d);
        store_float(grad_input, base + d,
                    rotated_gradient[d] * load_float(weight, d) * inverse -
                        value * inverse * inverse * inverse * dot /
                            static_cast<float>(dim));
        if (grad_weight)
          weight_accumulator[d] += rotated_gradient[d] * value * inverse;
      }
    }
  }
  if (grad_weight)
    for (std::uint64_t d = 0U; d < dim; ++d)
      store_float(*grad_weight, d, weight_accumulator[d]);
}

void layer_norm_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &input = tensors.at(op.inputs[1]);
  const auto &weight = tensors.at(op.inputs[2]);
  auto &grad_input = tensors.at(op.outputs[0]);
  auto &grad_weight = tensors.at(op.outputs[1]);
  auto &grad_bias = tensors.at(op.outputs[2]);
  const auto columns = input.dims.back();
  const auto rows = input.element_count() / columns;
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::vector<float> weight_accumulator(columns, 0.0F);
  std::vector<float> bias_accumulator(columns, 0.0F);
  for (std::uint64_t row = 0U; row < rows; ++row) {
    const auto base = row * columns;
    float mean = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column)
      mean += load_float(input, base + column);
    mean /= static_cast<float>(columns);
    float variance = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto centered = load_float(input, base + column) - mean;
      variance += centered * centered;
    }
    const auto inverse =
        1.0F / std::sqrt(variance / static_cast<float>(columns) + epsilon);
    float gradient_mean = 0.0F;
    float projected_mean = 0.0F;
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto weighted_gradient = load_float(grad_output, base + column) *
                                     load_float(weight, column);
      const auto normalized =
          (load_float(input, base + column) - mean) * inverse;
      gradient_mean += weighted_gradient;
      projected_mean += weighted_gradient * normalized;
    }
    gradient_mean /= static_cast<float>(columns);
    projected_mean /= static_cast<float>(columns);
    for (std::uint64_t column = 0U; column < columns; ++column) {
      const auto index = base + column;
      const auto upstream = load_float(grad_output, index);
      const auto weighted_gradient =
          upstream * load_float(weight, column);
      const auto normalized = (load_float(input, index) - mean) * inverse;
      store_float(grad_input, index,
                  inverse * (weighted_gradient - gradient_mean -
                             normalized * projected_mean));
      weight_accumulator[column] += upstream * normalized;
      bias_accumulator[column] += upstream;
    }
  }
  for (std::uint64_t column = 0U; column < columns; ++column) {
    store_float(grad_weight, column, weight_accumulator[column]);
    store_float(grad_bias, column, bias_accumulator[column]);
  }
}

void residual_gate_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &branch = tensors.at(op.inputs[1]);
  const auto &gate = tensors.at(op.inputs[2]);
  auto &grad_branch = tensors.at(op.outputs[0]);
  auto &grad_gate = tensors.at(op.outputs[1]);
  for (std::uint64_t i = 0U; i < grad_branch.element_count(); ++i) {
    const auto upstream = load_float(grad_output, i);
    store_float(grad_branch, i, upstream * load_float(gate, i));
    store_float(grad_gate, i, upstream * load_float(branch, i));
  }
}

void residual_gate(const ir::Operation &op, TensorMap &tensors) {
  const auto &residual = tensors.at(op.inputs[0]);
  const auto &branch = tensors.at(op.inputs[1]);
  const auto &gate = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  for (std::uint64_t i = 0; i < out.element_count(); ++i)
    store_float(out, i,
                load_float(residual, i) +
                    load_float(gate, i) * load_float(branch, i));
}

void linear(const ir::Operation &op, TensorMap &tensors) {
  const auto &input_tensor = tensors.at(op.inputs[0]);
  const auto &weight_tensor = tensors.at(op.inputs[1]);
  auto &out = tensors.at(op.outputs[0]);
  const auto rows = input_tensor.dims[0];
  const auto inner = input_tensor.element_count() / rows;
  const auto outputs = weight_tensor.dims[0];
  const auto *bias =
      op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  for (std::uint64_t row = 0; row < rows; ++row) {
    for (std::uint64_t column = 0; column < outputs; ++column) {
      float acc = bias ? load_float(*bias, column) : 0.0F;
      for (std::uint64_t k = 0; k < inner; ++k)
        acc = std::fma(load_float(input_tensor, row * inner + k),
                       load_float(weight_tensor, column * inner + k), acc);
      store_float(out, row * outputs + column, acc);
    }
  }
}

void qk_norm_rope(const ir::Operation &op, TensorMap &tensors) {
  const auto &input_tensor = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto &cosv = tensors.at(op.inputs[2]);
  const auto &sinv = tensors.at(op.inputs[3]);
  auto &out = tensors.at(op.outputs[0]);
  const auto sequence = input_tensor.dims[0];
  const auto heads = input_tensor.dims[1];
  const auto dim = input_tensor.dims[2];
  const auto rotary = op.u64(ir::AttrKey::RotaryDim, dim);
  const auto half = rotary / 2U;
  const auto table_width = tensors.at(op.inputs[2]).dims[1];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  std::vector<float> normalized(dim);
  for (std::uint64_t sequence_index = 0; sequence_index < sequence; ++sequence_index) {
    for (std::uint64_t head = 0; head < heads; ++head) {
      const auto base = (sequence_index * heads + head) * dim;
      float sum = 0.0F;
      for (std::uint64_t d = 0; d < dim; ++d)
        sum += load_float(input_tensor, base + d) *
               load_float(input_tensor, base + d);
      const float inv = 1.0F / std::sqrt(sum / static_cast<float>(dim) + epsilon);
      for (std::uint64_t d = 0; d < dim; ++d)
        normalized[d] = load_float(input_tensor, base + d) * inv *
                        load_float(weight, d);
      for (std::uint64_t d = 0; d < half; ++d) {
        const float cosine =
            load_float(cosv, sequence_index * table_width + d);
        const float sine =
            load_float(sinv, sequence_index * table_width + d);
        store_float(out, base + d,
                    normalized[d] * cosine - normalized[d + half] * sine);
        const auto second_index = table_width == rotary ? d + half : d;
        store_float(
            out, base + d + half,
            normalized[d + half] *
                    load_float(cosv,
                               sequence_index * table_width + second_index) +
                normalized[d] *
                    load_float(sinv,
                               sequence_index * table_width + second_index));
      }
      for (std::uint64_t d = rotary; d < dim; ++d)
        store_float(out, base + d, normalized[d]);
    }
  }
}

void attention(const ir::Operation &op, TensorMap &tensors) {
  const auto &q_tensor = tensors.at(op.inputs[0]);
  const auto &k = tensors.at(op.inputs[1]);
  const auto &v = tensors.at(op.inputs[2]);
  const auto *bias =
      op.inputs.size() == 4U ? &tensors.at(op.inputs[3]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const bool batched = q_tensor.dims.size() == 4U;
  const auto batch = batched ? q_tensor.dims[0] : 1U;
  const auto sequence = q_tensor.dims[batched ? 1U : 0U];
  const auto heads = q_tensor.dims[batched ? 2U : 1U];
  const auto dim = q_tensor.dims[batched ? 3U : 2U];
  // GQA: query head h reads kv head h/(H/KvHeads); KvHeads == H (the
  // default) reproduces the historical dense indexing exactly.
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  std::vector<float> probabilities(sequence);
  for (std::uint64_t b = 0; b < batch; ++b) {
    for (std::uint64_t query = 0; query < sequence; ++query) {
      const auto key_end = causal ? query + 1U : sequence;
      for (std::uint64_t head = 0; head < heads; ++head) {
      const auto kv_head = head / group;
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::uint64_t key = 0; key < key_end; ++key) {
        float score = 0.0F;
        for (std::uint64_t d = 0; d < dim; ++d) {
          score = std::fma(
              load_float(q_tensor,
                         ((b * sequence + query) * heads + head) * dim + d),
              load_float(k,
                         ((b * sequence + key) * kv_heads + kv_head) * dim + d),
              score);
        }
        score *= scale;
        if (bias)
          score += load_float(*bias, (b * sequence + query) * sequence + key);
        probabilities[key] = score;
        maximum = std::max(maximum, score);
      }
      if (!std::isfinite(maximum)) {
        for (std::uint64_t d = 0; d < dim; ++d)
          store_float(out,
                      ((b * sequence + query) * heads + head) * dim + d,
                      0.0F);
        continue;
      }
      float denominator = 0.0F;
      for (std::uint64_t key = 0; key < key_end; ++key) {
        probabilities[key] = std::exp(probabilities[key] - maximum);
        denominator += probabilities[key];
      }
      for (std::uint64_t d = 0; d < dim; ++d) {
        float value = 0.0F;
        for (std::uint64_t key = 0; key < key_end; ++key) {
          const auto probability = probabilities[key] / denominator;
          value = std::fma(
              probability,
              load_float(v,
                         ((b * sequence + key) * kv_heads + kv_head) * dim + d),
              value);
        }
        store_float(out,
                    ((b * sequence + query) * heads + head) * dim + d,
                    value);
      }
    }
  }
  }
}

void attention_lse(const ir::Operation &op, TensorMap &tensors) {
  const auto &q_tensor = tensors.at(op.inputs[0]);
  const auto &k = tensors.at(op.inputs[1]);
  auto &lse = tensors.at(op.outputs[0]);
  const auto sequence = q_tensor.dims[0];
  const auto heads = q_tensor.dims[1];
  const auto dim = q_tensor.dims[2];
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  for (std::uint64_t query = 0U; query < sequence; ++query) {
    const auto key_end = causal ? query + 1U : sequence;
    for (std::uint64_t head = 0U; head < heads; ++head) {
      const auto kv_head = head / group;
      float maximum = -std::numeric_limits<float>::infinity();
      std::vector<float> scores(key_end);
      for (std::uint64_t key = 0U; key < key_end; ++key) {
        float score = 0.0F;
        for (std::uint64_t d = 0U; d < dim; ++d)
          score = std::fma(
              load_float(q_tensor, (query * heads + head) * dim + d),
              load_float(k, (key * kv_heads + kv_head) * dim + d), score);
        scores[key] = score * scale;
        maximum = std::max(maximum, scores[key]);
      }
      float denominator = 0.0F;
      for (std::uint64_t key = 0U; key < key_end; ++key)
        denominator += std::exp(scores[key] - maximum);
      store_float(lse, query * heads + head,
                  maximum + std::log(denominator));
    }
  }
}

void attention_backward(const ir::Operation &op, TensorMap &tensors) {
  const auto &grad_output = tensors.at(op.inputs[0]);
  const auto &q_tensor = tensors.at(op.inputs[1]);
  const auto &k = tensors.at(op.inputs[2]);
  const auto &v = tensors.at(op.inputs[3]);
  const auto &forward_output = tensors.at(op.inputs[4]);
  const auto &lse = tensors.at(op.inputs[5]);
  auto &grad_q = tensors.at(op.outputs[0]);
  auto &grad_k = tensors.at(op.outputs[1]);
  auto &grad_v = tensors.at(op.outputs[2]);
  const auto sequence = q_tensor.dims[0];
  const auto heads = q_tensor.dims[1];
  const auto dim = q_tensor.dims[2];
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  const auto kv_heads = op.u64(ir::AttrKey::KvHeads, heads);
  const auto group = heads / kv_heads;
  // dK/dV accumulate across every query AND every query head sharing the kv
  // head, in F32 accumulators (flame dtype contract: cross-element
  // accumulation never round-trips through storage precision); one rounding
  // store per element at the end.  Causal masking leaves the accumulators'
  // zeros for keys beyond every query.
  std::vector<float> key_gradient(grad_k.element_count(), 0.0F);
  std::vector<float> value_gradient(grad_v.element_count(), 0.0F);
  std::vector<float> row_gradient(sequence);
  for (std::uint64_t head = 0U; head < heads; ++head) {
    const auto kv_head = head / group;
    for (std::uint64_t query = 0U; query < sequence; ++query) {
      const auto key_end = causal ? query + 1U : sequence;
      const auto query_base = (query * heads + head) * dim;
      const auto row_lse = load_float(lse, query * heads + head);
      float delta = 0.0F;
      for (std::uint64_t d = 0U; d < dim; ++d)
        delta = std::fma(load_float(grad_output, query_base + d),
                         load_float(forward_output, query_base + d), delta);
      for (std::uint64_t key = 0U; key < key_end; ++key) {
        const auto key_base = (key * kv_heads + kv_head) * dim;
        float score = 0.0F;
        float projected = 0.0F;
        for (std::uint64_t d = 0U; d < dim; ++d) {
          score = std::fma(load_float(q_tensor, query_base + d),
                           load_float(k, key_base + d), score);
          projected = std::fma(load_float(grad_output, query_base + d),
                               load_float(v, key_base + d), projected);
        }
        const auto probability = std::exp(score * scale - row_lse);
        const auto score_gradient =
            probability * (projected - delta) * scale;
        row_gradient[key] = score_gradient;
        for (std::uint64_t d = 0U; d < dim; ++d) {
          key_gradient[key_base + d] =
              std::fma(score_gradient, load_float(q_tensor, query_base + d),
                       key_gradient[key_base + d]);
          value_gradient[key_base + d] =
              std::fma(probability, load_float(grad_output, query_base + d),
                       value_gradient[key_base + d]);
        }
      }
      for (std::uint64_t d = 0U; d < dim; ++d) {
        float accumulator = 0.0F;
        for (std::uint64_t key = 0U; key < key_end; ++key)
          accumulator =
              std::fma(row_gradient[key],
                       load_float(k, (key * kv_heads + kv_head) * dim + d),
                       accumulator);
        store_float(grad_q, query_base + d, accumulator);
      }
    }
  }
  for (std::uint64_t i = 0U; i < grad_k.element_count(); ++i) {
    store_float(grad_k, i, key_gradient[i]);
    store_float(grad_v, i, value_gradient[i]);
  }
}


// BigVGAN-class 1-D convolution: plain / dilated / grouped / depthwise /
// transposed, zero- or replicate-padded, with transposed output trim.
// Index-mapped padding (no materialized pad), F32 accumulator per output
// element — the exact loop ordering of the gated Mojo host oracle
// (serenitymojo models/minimax_h3/audio_decoder.mojo conv1d /
// conv_transpose1d), generalized to attribute-driven geometry.
void conv1d_f32(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto *bias =
      op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const auto stride = op.u64(ir::AttrKey::Stride, 1U);
  const auto dilation = op.u64(ir::AttrKey::Dilation, 1U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto pad_left = op.u64(ir::AttrKey::PadLeft, 0U);
  const auto pad_mode = op.u64(ir::AttrKey::PadMode, 0U);
  const auto transposed = op.boolean(ir::AttrKey::Transposed, false);
  const auto trim_left = op.u64(ir::AttrKey::TrimLeft, 0U);
  const auto batch = input.dims[0];
  const auto in_channels = input.dims[1];
  const auto length = input.dims[2];
  const auto kernel = weight.dims[2];
  const auto out_channels = out.dims[1];
  const auto out_length = out.dims[2];
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  const auto replicate = pad_mode == 1U;
  const auto *input_values = reinterpret_cast<const float *>(input.data());
  const auto *weight_values = reinterpret_cast<const float *>(weight.data());
  const auto *bias_values =
      bias ? reinterpret_cast<const float *>(bias->data()) : nullptr;
  auto *out_values = reinterpret_cast<float *>(out.mutable_data());

  const auto sample = [&](const float *base, std::int64_t position) -> float {
    auto index = position - static_cast<std::int64_t>(pad_left);
    if (index < 0 || index >= static_cast<std::int64_t>(length)) {
      if (!replicate)
        return 0.0F;
      index = std::clamp<std::int64_t>(index, 0,
                                       static_cast<std::int64_t>(length) - 1);
    }
    return base[index];
  };

  for (std::uint64_t b = 0; b < batch; ++b) {
    if (!transposed) {
      for (std::uint64_t oc = 0; oc < out_channels; ++oc) {
        const auto group = oc / out_per_group;
        auto *out_row = out_values + (b * out_channels + oc) * out_length;
        for (std::uint64_t o = 0; o < out_length; ++o) {
          float accumulator = 0.0F;
          for (std::uint64_t ic = 0; ic < in_per_group; ++ic) {
            const auto in_channel = group * in_per_group + ic;
            const auto *in_row =
                input_values + (b * in_channels + in_channel) * length;
            const auto *w_row =
                weight_values + (oc * in_per_group + ic) * kernel;
            const auto start = static_cast<std::int64_t>(o * stride);
            for (std::uint64_t k = 0; k < kernel; ++k)
              accumulator +=
                  sample(in_row,
                         start + static_cast<std::int64_t>(k * dilation)) *
                  w_row[k];
          }
          if (bias_values)
            accumulator += bias_values[oc];
          out_row[o] = accumulator;
        }
      }
    } else {
      const auto padded =
          length + pad_left + op.u64(ir::AttrKey::PadRight, 0U);
      std::vector<float> full_accumulator;
      for (std::uint64_t oc = 0; oc < out_channels; ++oc) {
        const auto group = oc / out_per_group;
        const auto oc_in_group = oc % out_per_group;
        auto *out_row = out_values + (b * out_channels + oc) * out_length;
        full_accumulator.assign(static_cast<std::size_t>(out_length), 0.0F);
        for (std::uint64_t ic = 0; ic < in_per_group; ++ic) {
          const auto in_channel = group * in_per_group + ic;
          const auto *in_row =
              input_values + (b * in_channels + in_channel) * length;
          const auto *w_row =
              weight_values +
              (in_channel * out_per_group + oc_in_group) * kernel;
          for (std::uint64_t i = 0; i < padded; ++i) {
            const auto value =
                sample(in_row, static_cast<std::int64_t>(i));
            const auto scatter_base =
                static_cast<std::int64_t>(i * stride) -
                static_cast<std::int64_t>(trim_left);
            for (std::uint64_t k = 0; k < kernel; ++k) {
              const auto position =
                  scatter_base + static_cast<std::int64_t>(k);
              if (position < 0 ||
                  position >= static_cast<std::int64_t>(out_length))
                continue;
              full_accumulator[static_cast<std::size_t>(position)] +=
                  value * w_row[k];
            }
          }
        }
        const auto bias_value = bias_values ? bias_values[oc] : 0.0F;
        for (std::uint64_t o = 0; o < out_length; ++o)
          out_row[o] = full_accumulator[o] + bias_value;
      }
    }
  }
}

void conv1d(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto *bias = op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  if (input.dtype == ir::DType::F32 && weight.dtype == ir::DType::F32 &&
      out.dtype == ir::DType::F32 &&
      (!bias || bias->dtype == ir::DType::F32)) {
    // Bit-identical fast path: same nesting, same accumulation order, raw
    // pointers instead of per-element load_float dispatch.
    conv1d_f32(op, tensors);
    return;
  }
  const auto stride = op.u64(ir::AttrKey::Stride, 1U);
  const auto dilation = op.u64(ir::AttrKey::Dilation, 1U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto pad_left = op.u64(ir::AttrKey::PadLeft, 0U);
  const auto pad_mode = op.u64(ir::AttrKey::PadMode, 0U);
  const auto transposed = op.boolean(ir::AttrKey::Transposed, false);
  const auto trim_left = op.u64(ir::AttrKey::TrimLeft, 0U);
  const auto batch = input.dims[0];
  const auto in_channels = input.dims[1];
  const auto length = input.dims[2];
  const auto kernel = weight.dims[2];
  const auto out_channels = out.dims[1];
  const auto out_length = out.dims[2];
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  const auto replicate = pad_mode == 1U;

  // Padded-coordinate sample: position p in [0, L+PadL+PadR) maps to input
  // index p-PadL, clamped for replicate pad or zero outside for zero pad.
  const auto sample = [&](std::uint64_t base, std::int64_t position) -> float {
    auto index = position - static_cast<std::int64_t>(pad_left);
    if (index < 0 || index >= static_cast<std::int64_t>(length)) {
      if (!replicate)
        return 0.0F;
      index = std::clamp<std::int64_t>(index, 0,
                                       static_cast<std::int64_t>(length) - 1);
    }
    return load_float(input, base + static_cast<std::uint64_t>(index));
  };

  for (std::uint64_t b = 0; b < batch; ++b) {
    if (!transposed) {
      for (std::uint64_t oc = 0; oc < out_channels; ++oc) {
        const auto group = oc / out_per_group;
        const auto out_base = (b * out_channels + oc) * out_length;
        for (std::uint64_t o = 0; o < out_length; ++o) {
          float accumulator = 0.0F;
          for (std::uint64_t ic = 0; ic < in_per_group; ++ic) {
            const auto in_channel = group * in_per_group + ic;
            const auto in_base = (b * in_channels + in_channel) * length;
            const auto weight_base = (oc * in_per_group + ic) * kernel;
            for (std::uint64_t k = 0; k < kernel; ++k) {
              const auto position = static_cast<std::int64_t>(o * stride) +
                                    static_cast<std::int64_t>(k * dilation);
              accumulator += sample(in_base, position) *
                             load_float(weight, weight_base + k);
            }
          }
          if (bias)
            accumulator += load_float(*bias, oc);
          store_float(out, out_base + o, accumulator);
        }
      }
    } else {
      // Scatter with stride over the padded input, then trim: output
      // position (in full coordinates) = i*stride + k, minus TrimLeft.
      const auto padded =
          length + pad_left + op.u64(ir::AttrKey::PadRight, 0U);
      std::vector<float> full_accumulator;
      for (std::uint64_t oc = 0; oc < out_channels; ++oc) {
        const auto group = oc / out_per_group;
        const auto oc_in_group = oc % out_per_group;
        const auto out_base = (b * out_channels + oc) * out_length;
        full_accumulator.assign(static_cast<std::size_t>(out_length), 0.0F);
        for (std::uint64_t ic = 0; ic < in_per_group; ++ic) {
          const auto in_channel = group * in_per_group + ic;
          const auto in_base = (b * in_channels + in_channel) * length;
          const auto weight_base =
              (in_channel * out_per_group + oc_in_group) * kernel;
          for (std::uint64_t i = 0; i < padded; ++i) {
            const auto value = sample(in_base, static_cast<std::int64_t>(i));
            for (std::uint64_t k = 0; k < kernel; ++k) {
              const auto position = static_cast<std::int64_t>(i * stride + k) -
                                    static_cast<std::int64_t>(trim_left);
              if (position < 0 ||
                  position >= static_cast<std::int64_t>(out_length))
                continue;
              full_accumulator[static_cast<std::size_t>(position)] +=
                  value * load_float(weight, weight_base + k);
            }
          }
        }
        const auto bias_value = bias ? load_float(*bias, oc) : 0.0F;
        for (std::uint64_t o = 0; o < out_length; ++o)
          store_float(out, out_base + o, full_accumulator[o] + bias_value);
      }
    }
  }
}

float round_to_dtype(float value, ir::DType dtype) {
  if (dtype == ir::DType::BF16)
    return bf16_to_float(float_to_bf16(value));
  if (dtype == ir::DType::F16)
    return f16_to_float(float_to_f16(value));
  return value;
}

void conv2d(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto *bias = op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const auto stride_h = op.u64(ir::AttrKey::StrideH, 1U);
  const auto stride_w = op.u64(ir::AttrKey::StrideW, 1U);
  const auto dilation_h = op.u64(ir::AttrKey::DilationH, 1U);
  const auto dilation_w = op.u64(ir::AttrKey::DilationW, 1U);
  const auto pad_top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto pad_west = op.u64(ir::AttrKey::PadWest, 0U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto batch = input.dims[0];
  const auto in_channels = input.dims[1];
  const auto input_h = input.dims[2];
  const auto input_w = input.dims[3];
  const auto out_channels = weight.dims[0];
  const auto kernel_h = weight.dims[2];
  const auto kernel_w = weight.dims[3];
  const auto output_h = out.dims[2];
  const auto output_w = out.dims[3];
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  for (std::uint64_t b = 0U; b < batch; ++b) {
    for (std::uint64_t oc = 0U; oc < out_channels; ++oc) {
      const auto group = oc / out_per_group;
      for (std::uint64_t oh = 0U; oh < output_h; ++oh) {
        for (std::uint64_t ow = 0U; ow < output_w; ++ow) {
          float accumulator = 0.0F;
          for (std::uint64_t ic = 0U; ic < in_per_group; ++ic) {
            const auto source_channel = group * in_per_group + ic;
            for (std::uint64_t kh = 0U; kh < kernel_h; ++kh) {
              const auto ih = static_cast<std::int64_t>(oh * stride_h +
                                                        kh * dilation_h) -
                              static_cast<std::int64_t>(pad_top);
              if (ih < 0 || ih >= static_cast<std::int64_t>(input_h))
                continue;
              for (std::uint64_t kw = 0U; kw < kernel_w; ++kw) {
                const auto iw = static_cast<std::int64_t>(ow * stride_w +
                                                          kw * dilation_w) -
                                static_cast<std::int64_t>(pad_west);
                if (iw < 0 || iw >= static_cast<std::int64_t>(input_w))
                  continue;
                const auto input_index =
                    ((b * in_channels + source_channel) * input_h +
                     static_cast<std::uint64_t>(ih)) *
                        input_w +
                    static_cast<std::uint64_t>(iw);
                const auto weight_index =
                    ((oc * in_per_group + ic) * kernel_h + kh) * kernel_w + kw;
                accumulator += load_float(input, input_index) *
                               load_float(weight, weight_index);
              }
            }
          }
          if (bias)
            accumulator += load_float(*bias, oc);
          const auto output_index =
              ((b * out_channels + oc) * output_h + oh) * output_w + ow;
          store_float(out, output_index, accumulator);
        }
      }
    }
  }
}

void pad_constant(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto value = static_cast<float>(op.f64(ir::AttrKey::Value, 0.0));
  for (std::uint64_t index = 0U; index < out.element_count(); ++index)
    store_float(out, index, value);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  if (input.dims.size() == 4U) {
    const auto channels = input.dims[1];
    const auto input_h = input.dims[2];
    const auto input_w = input.dims[3];
    const auto output_h = out.dims[2];
    const auto output_w = out.dims[3];
    for (std::uint64_t b = 0U; b < input.dims[0]; ++b)
      for (std::uint64_t c = 0U; c < channels; ++c)
        for (std::uint64_t y = 0U; y < input_h; ++y)
          for (std::uint64_t x = 0U; x < input_w; ++x) {
            const auto source = ((b * channels + c) * input_h + y) * input_w + x;
            const auto target =
                ((b * channels + c) * output_h + y + top) * output_w + x + west;
            store_float(out, target, load_float(input, source));
          }
    return;
  }
  const auto channels = input.dims[1];
  const auto input_t = input.dims[2];
  const auto input_h = input.dims[3];
  const auto input_w = input.dims[4];
  const auto output_t = out.dims[2];
  const auto output_h = out.dims[3];
  const auto output_w = out.dims[4];
  const auto front = op.u64(ir::AttrKey::PadFront, 0U);
  for (std::uint64_t b = 0U; b < input.dims[0]; ++b)
    for (std::uint64_t c = 0U; c < channels; ++c)
      for (std::uint64_t t = 0U; t < input_t; ++t)
        for (std::uint64_t y = 0U; y < input_h; ++y)
          for (std::uint64_t x = 0U; x < input_w; ++x) {
            const auto source =
                (((b * channels + c) * input_t + t) * input_h + y) * input_w + x;
            const auto target =
                (((b * channels + c) * output_t + t + front) * output_h + y + top) *
                    output_w +
                x + west;
            store_float(out, target, load_float(input, source));
          }
}

std::uint64_t reflect_coordinate(std::uint64_t position,
                                 std::uint64_t before,
                                 std::uint64_t length) {
  if (position < before)
    return before - position;
  const auto shifted = position - before;
  if (shifted < length)
    return shifted;
  return 2U * length - 2U - shifted;
}

void pad_reflect(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto front = op.u64(ir::AttrKey::PadFront, 0U);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  if (input.dims.size() == 4U) {
    const auto channels = input.dims[1];
    const auto input_h = input.dims[2];
    const auto input_w = input.dims[3];
    const auto output_h = out.dims[2];
    const auto output_w = out.dims[3];
    for (std::uint64_t b = 0U; b < input.dims[0]; ++b)
      for (std::uint64_t c = 0U; c < channels; ++c)
        for (std::uint64_t y = 0U; y < output_h; ++y)
          for (std::uint64_t x = 0U; x < output_w; ++x) {
            const auto source_y = reflect_coordinate(y, top, input_h);
            const auto source_x = reflect_coordinate(x, west, input_w);
            const auto source =
                ((b * channels + c) * input_h + source_y) * input_w +
                source_x;
            const auto target =
                ((b * channels + c) * output_h + y) * output_w + x;
            store_float(out, target, load_float(input, source));
          }
    return;
  }
  const auto channels = input.dims[1];
  const auto input_t = input.dims[2];
  const auto input_h = input.dims[3];
  const auto input_w = input.dims[4];
  const auto output_t = out.dims[2];
  const auto output_h = out.dims[3];
  const auto output_w = out.dims[4];
  for (std::uint64_t b = 0U; b < input.dims[0]; ++b)
    for (std::uint64_t c = 0U; c < channels; ++c)
      for (std::uint64_t t = 0U; t < output_t; ++t)
        for (std::uint64_t y = 0U; y < output_h; ++y)
          for (std::uint64_t x = 0U; x < output_w; ++x) {
            const auto source_t = reflect_coordinate(t, front, input_t);
            const auto source_y = reflect_coordinate(y, top, input_h);
            const auto source_x = reflect_coordinate(x, west, input_w);
            const auto source =
                (((b * channels + c) * input_t + source_t) * input_h +
                 source_y) *
                    input_w +
                source_x;
            const auto target =
                (((b * channels + c) * output_t + t) * output_h + y) *
                    output_w +
                x;
            store_float(out, target, load_float(input, source));
          }
}

void conv3d(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto *bias = op.inputs.size() == 3U ? &tensors.at(op.inputs[2]) : nullptr;
  auto &out = tensors.at(op.outputs[0]);
  const auto stride_t = op.u64(ir::AttrKey::StrideT, 1U);
  const auto stride_h = op.u64(ir::AttrKey::StrideH, 1U);
  const auto stride_w = op.u64(ir::AttrKey::StrideW, 1U);
  const auto dilation_t = op.u64(ir::AttrKey::DilationT, 1U);
  const auto dilation_h = op.u64(ir::AttrKey::DilationH, 1U);
  const auto dilation_w = op.u64(ir::AttrKey::DilationW, 1U);
  const auto front = op.u64(ir::AttrKey::PadFront, 0U);
  const auto top = op.u64(ir::AttrKey::PadTop, 0U);
  const auto west = op.u64(ir::AttrKey::PadWest, 0U);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto in_channels = input.dims[1];
  const auto out_channels = weight.dims[0];
  const auto in_per_group = in_channels / groups;
  const auto out_per_group = out_channels / groups;
  for (std::uint64_t b = 0U; b < input.dims[0]; ++b)
    for (std::uint64_t oc = 0U; oc < out_channels; ++oc) {
      const auto group = oc / out_per_group;
      for (std::uint64_t ot = 0U; ot < out.dims[2]; ++ot)
        for (std::uint64_t oh = 0U; oh < out.dims[3]; ++oh)
          for (std::uint64_t ow = 0U; ow < out.dims[4]; ++ow) {
            float accumulator = 0.0F;
            for (std::uint64_t ic = 0U; ic < in_per_group; ++ic)
              for (std::uint64_t kt = 0U; kt < weight.dims[2]; ++kt) {
                const auto it = static_cast<std::int64_t>(ot * stride_t + kt * dilation_t) -
                                static_cast<std::int64_t>(front);
                if (it < 0 || it >= static_cast<std::int64_t>(input.dims[2]))
                  continue;
                for (std::uint64_t kh = 0U; kh < weight.dims[3]; ++kh) {
                  const auto ih = static_cast<std::int64_t>(oh * stride_h + kh * dilation_h) -
                                  static_cast<std::int64_t>(top);
                  if (ih < 0 || ih >= static_cast<std::int64_t>(input.dims[3]))
                    continue;
                  for (std::uint64_t kw = 0U; kw < weight.dims[4]; ++kw) {
                    const auto iw = static_cast<std::int64_t>(ow * stride_w + kw * dilation_w) -
                                    static_cast<std::int64_t>(west);
                    if (iw < 0 || iw >= static_cast<std::int64_t>(input.dims[4]))
                      continue;
                    const auto source_channel = group * in_per_group + ic;
                    const auto input_index =
                        (((b * in_channels + source_channel) * input.dims[2] +
                          static_cast<std::uint64_t>(it)) *
                             input.dims[3] +
                         static_cast<std::uint64_t>(ih)) *
                            input.dims[4] +
                        static_cast<std::uint64_t>(iw);
                    const auto weight_index =
                        (((oc * in_per_group + ic) * weight.dims[2] + kt) *
                             weight.dims[3] +
                         kh) *
                            weight.dims[4] +
                        kw;
                    accumulator += load_float(input, input_index) *
                                   load_float(weight, weight_index);
                  }
                }
              }
            if (bias)
              accumulator += load_float(*bias, oc);
            const auto output_index =
                (((b * out_channels + oc) * out.dims[2] + ot) * out.dims[3] +
                 oh) *
                    out.dims[4] +
                ow;
            store_float(out, output_index, accumulator);
          }
    }
}

void channel_rms_norm(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &gamma = tensors.at(op.inputs[1]);
  auto &out = tensors.at(op.outputs[0]);
  const auto axis = op.u64(ir::AttrKey::Axis, 1U);
  const auto channels = input.dims[axis];
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-12));
  std::uint64_t inner = 1U;
  for (std::size_t index = static_cast<std::size_t>(axis + 1U);
       index < input.dims.size(); ++index)
    inner *= input.dims[index];
  const auto outer = input.element_count() / (channels * inner);
  const auto scale = std::sqrt(static_cast<float>(channels));
  for (std::uint64_t leading = 0U; leading < outer; ++leading) {
    for (std::uint64_t trailing = 0U; trailing < inner; ++trailing) {
      float squared = 0.0F;
      for (std::uint64_t channel = 0U; channel < channels; ++channel) {
        const auto index = (leading * channels + channel) * inner + trailing;
        const auto value = load_float(input, index);
        squared += value * value;
      }
      const auto denominator = std::max(std::sqrt(squared), epsilon);
      for (std::uint64_t channel = 0U; channel < channels; ++channel) {
        const auto index = (leading * channels + channel) * inner + trailing;
        const auto normalized = round_to_dtype(load_float(input, index) /
                                                   denominator,
                                               input.dtype);
        const auto scaled = round_to_dtype(normalized * scale, input.dtype);
        store_float(out, index, scaled * load_float(gamma, channel));
      }
    }
  }
}

void group_norm(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &weight = tensors.at(op.inputs[1]);
  const auto &bias = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  const auto groups = op.u64(ir::AttrKey::Groups, 1U);
  const auto epsilon = static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-5));
  const auto channels = input.dims[1];
  const auto channels_per_group = channels / groups;
  std::uint64_t inner = 1U;
  for (std::size_t axis = 2U; axis < input.dims.size(); ++axis)
    inner *= input.dims[axis];
  const auto elements_per_group = channels_per_group * inner;
  for (std::uint64_t batch = 0U; batch < input.dims[0]; ++batch) {
    for (std::uint64_t group = 0U; group < groups; ++group) {
      const auto base = (batch * channels + group * channels_per_group) * inner;
      float mean = 0.0F;
      for (std::uint64_t index = 0U; index < elements_per_group; ++index)
        mean += load_float(input, base + index);
      mean /= static_cast<float>(elements_per_group);
      float variance = 0.0F;
      for (std::uint64_t index = 0U; index < elements_per_group; ++index) {
        const auto centered = load_float(input, base + index) - mean;
        variance += centered * centered;
      }
      variance /= static_cast<float>(elements_per_group);
      const auto inverse = 1.0F / std::sqrt(variance + epsilon);
      for (std::uint64_t index = 0U; index < elements_per_group; ++index) {
        const auto channel = group * channels_per_group + index / inner;
        const auto normalized = (load_float(input, base + index) - mean) * inverse;
        store_float(out, base + index,
                    normalized * load_float(weight, channel) +
                        load_float(bias, channel));
      }
    }
  }
}

void upsample_nearest_2d(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  auto &out = tensors.at(op.outputs[0]);
  const auto scale_h = op.u64(ir::AttrKey::ScaleH, 1U);
  const auto scale_w = op.u64(ir::AttrKey::ScaleW, 1U);
  const auto batch = input.dims[0];
  const auto channels = input.dims[1];
  const auto input_h = input.dims[2];
  const auto input_w = input.dims[3];
  const auto output_h = out.dims[2];
  const auto output_w = out.dims[3];
  for (std::uint64_t b = 0U; b < batch; ++b)
    for (std::uint64_t c = 0U; c < channels; ++c)
      for (std::uint64_t oh = 0U; oh < output_h; ++oh)
        for (std::uint64_t ow = 0U; ow < output_w; ++ow) {
          const auto input_index =
              ((b * channels + c) * input_h + oh / scale_h) * input_w +
              ow / scale_w;
          const auto output_index =
              ((b * channels + c) * output_h + oh) * output_w + ow;
          store_float(out, output_index, load_float(input, input_index));
        }
}

// BigVGAN SnakeBeta: y = x + (exp(beta_c) + eps)^-1 * sin(exp(alpha_c) * x)^2,
// alpha/beta stored in LOG space as [C] vectors (linear treatment is a
// near-identity trap; see the audio decode plan).
void snake_beta(const ir::Operation &op, TensorMap &tensors) {
  const auto &input = tensors.at(op.inputs[0]);
  const auto &log_alpha = tensors.at(op.inputs[1]);
  const auto &log_beta = tensors.at(op.inputs[2]);
  auto &out = tensors.at(op.outputs[0]);
  if (input.dtype == ir::DType::F32 &&
      log_alpha.dtype == ir::DType::F32 &&
      log_beta.dtype == ir::DType::F32 && out.dtype == ir::DType::F32) {
    const auto epsilon =
        static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-9));
    const auto batch = input.dims[0];
    const auto channels = input.dims[1];
    const auto length = input.dims[2];
    const auto *input_values = reinterpret_cast<const float *>(input.data());
    const auto *alpha_values =
        reinterpret_cast<const float *>(log_alpha.data());
    const auto *beta_values =
        reinterpret_cast<const float *>(log_beta.data());
    auto *out_values = reinterpret_cast<float *>(out.mutable_data());
    for (std::uint64_t b = 0; b < batch; ++b) {
      for (std::uint64_t c = 0; c < channels; ++c) {
        const auto alpha = std::exp(alpha_values[c]);
        const auto inverse_beta =
            1.0F / (std::exp(beta_values[c]) + epsilon);
        const auto *in_row = input_values + (b * channels + c) * length;
        auto *out_row = out_values + (b * channels + c) * length;
        for (std::uint64_t i = 0; i < length; ++i) {
          const auto x = in_row[i];
          const auto s = std::sin(alpha * x);
          out_row[i] = x + inverse_beta * s * s;
        }
      }
    }
    return;
  }
  const auto epsilon =
      static_cast<float>(op.f64(ir::AttrKey::Epsilon, 1.0e-9));
  const auto batch = input.dims[0];
  const auto channels = input.dims[1];
  const auto length = input.dims[2];
  for (std::uint64_t b = 0; b < batch; ++b) {
    for (std::uint64_t c = 0; c < channels; ++c) {
      const auto alpha = std::exp(load_float(log_alpha, c));
      const auto inverse_beta =
          1.0F / (std::exp(load_float(log_beta, c)) + epsilon);
      const auto base = (b * channels + c) * length;
      for (std::uint64_t i = 0; i < length; ++i) {
        const auto x = load_float(input, base + i);
        const auto s = std::sin(alpha * x);
        store_float(out, base + i, x + inverse_beta * s * s);
      }
    }
  }
}

void execute_operation(const ir::Program &program, const ir::Operation &op,
                       TensorMap &tensors);

void execute_once(const ir::Program &program, TensorMap &tensors) {
  for (const auto &op : program.operations)
    execute_operation(program, op, tensors);
}

void execute_operation(const ir::Program &program, const ir::Operation &op,
                       TensorMap &tensors) {
  (void)program;
  {
    switch (op.opcode) {
    case ir::Opcode::Add:
      elementwise(op, tensors, false);
      break;
    case ir::Opcode::Multiply:
      elementwise(op, tensors, true);
      break;
    case ir::Opcode::AffineLastDim:
      affine_last_dim(op, tensors);
      break;
    case ir::Opcode::SiLU:
      silu(op, tensors);
      break;
    case ir::Opcode::Gelu:
      gelu(op, tensors);
      break;
    case ir::Opcode::Sigmoid:
      sigmoid(op, tensors);
      break;
    case ir::Opcode::Reshape:
      reshape(op, tensors);
      break;
    case ir::Opcode::BroadcastTo:
      broadcast_to(op, tensors);
      break;
    case ir::Opcode::Slice:
      slice_tensor(op, tensors);
      break;
    case ir::Opcode::RotaryFrequency:
      rotary_frequency(op, tensors);
      break;
    case ir::Opcode::RotaryApply:
      rotary_apply(op, tensors);
      break;
    case ir::Opcode::BooleanMaskToBias:
      boolean_mask_to_bias(op, tensors);
      break;
    case ir::Opcode::RmsNorm:
      rms_norm(op, tensors);
      break;
    case ir::Opcode::LayerNorm:
      layer_norm(op, tensors);
      break;
    case ir::Opcode::Clamp:
      clamp(op, tensors);
      break;
    case ir::Opcode::MseLoss:
      mse_loss(op, tensors);
      break;
    case ir::Opcode::MseLossBackward:
      mse_loss_backward(op, tensors);
      break;
    case ir::Opcode::LinearBackwardInput:
      linear_backward_input(op, tensors);
      break;
    case ir::Opcode::LinearBackwardWeight:
      linear_backward_weight(op, tensors);
      break;
    case ir::Opcode::BiasBackward:
      bias_backward(op, tensors);
      break;
    case ir::Opcode::SiLUBackward:
      silu_backward(op, tensors);
      break;
    case ir::Opcode::AdamWUpdate:
      adamw_update(op, tensors);
      break;
    case ir::Opcode::Fill:
      fill(op, tensors);
      break;
    case ir::Opcode::GatherRows:
      gather_rows(op, tensors);
      break;
    case ir::Opcode::IndexedUpdateRows:
      indexed_update_rows(op, tensors);
      break;
    case ir::Opcode::Cast:
      cast(op, tensors);
      break;
    case ir::Opcode::SelectRowChunks:
      select_row_chunks(op, tensors);
      break;
    case ir::Opcode::SinusoidalTimestep:
      sinusoidal_timestep(op, tensors);
      break;
    case ir::Opcode::RotaryPosition:
      rotary_position(op, tensors);
      break;
    case ir::Opcode::LinearBlend:
      linear_blend(op, tensors);
      break;
    case ir::Opcode::FlowEulerStep:
      flow_euler_step(op, tensors);
      break;
    case ir::Opcode::EulerVelocityStep:
      euler_velocity_step(op, tensors);
      break;
    case ir::Opcode::Permute:
      permute(op, tensors);
      break;
    case ir::Opcode::Concat:
      concat(op, tensors);
      break;
    case ir::Opcode::Patchify3D:
      patchify_3d(op, tensors, false);
      break;
    case ir::Opcode::Unpatchify3D:
      patchify_3d(op, tensors, true);
      break;
    case ir::Opcode::RmsNormModulate:
      rms_norm_modulate(op, tensors);
      break;
    case ir::Opcode::SwiGlu:
      swiglu(op, tensors);
      break;
    case ir::Opcode::ResidualGate:
      residual_gate(op, tensors);
      break;
    case ir::Opcode::Linear:
      linear(op, tensors);
      break;
    case ir::Opcode::QkNormPartialRope:
      qk_norm_rope(op, tensors);
      break;
    case ir::Opcode::Attention:
      attention(op, tensors);
      break;
    case ir::Opcode::Barrier:
      break;
    case ir::Opcode::BiasAdd:
      bias_add(op, tensors);
      break;
    case ir::Opcode::H3AdaLNSelect:
      h3_adaln_select(op, tensors);
      break;
    case ir::Opcode::H3DeinterleaveQkv:
      h3_deinterleave_qkv(op, tensors);
      break;
    case ir::Opcode::H3DeinterleaveQkvWeight:
      h3_deinterleave_qkv_weight(op, tensors);
      break;
    case ir::Opcode::DequantizeInt4:
      dequantize_int4(op, tensors);
      break;
    case ir::Opcode::DequantizeInt5:
      dequantize_int5(op, tensors);
      break;
    case ir::Opcode::RmsNormBackward:
      rms_norm_backward(op, tensors);
      break;
    case ir::Opcode::RmsNormModulateBackward:
      rms_norm_modulate_backward(op, tensors);
      break;
    case ir::Opcode::SwiGluBackward:
      swiglu_backward(op, tensors);
      break;
    case ir::Opcode::ResidualGateBackward:
      residual_gate_backward(op, tensors);
      break;
    case ir::Opcode::LayerNormBackward:
      layer_norm_backward(op, tensors);
      break;
    case ir::Opcode::QkNormPartialRopeBackward:
      qk_norm_rope_backward(op, tensors);
      break;
    case ir::Opcode::AttentionLse:
      attention_lse(op, tensors);
      break;
    case ir::Opcode::AttentionBackward:
      attention_backward(op, tensors);
      break;
    case ir::Opcode::Conv1d:
      conv1d(op, tensors);
      break;
    case ir::Opcode::Conv2d:
      conv2d(op, tensors);
      break;
    case ir::Opcode::ChannelRmsNorm:
      channel_rms_norm(op, tensors);
      break;
    case ir::Opcode::UpsampleNearest2d:
      upsample_nearest_2d(op, tensors);
      break;
    case ir::Opcode::PadConstant:
      pad_constant(op, tensors);
      break;
    case ir::Opcode::Conv3d:
      conv3d(op, tensors);
      break;
    case ir::Opcode::GroupNorm:
      group_norm(op, tensors);
      break;
    case ir::Opcode::PadReflect:
      pad_reflect(op, tensors);
      break;
    case ir::Opcode::SnakeBeta:
      snake_beta(op, tensors);
      break;
    }
  }
}

class CpuPreparedExecution final : public PreparedExecution {
public:
  CpuPreparedExecution(ir::Program program, const TensorMap &bindings)
      : program_(std::move(program)) {
    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Constant))
        continue;
      const auto found = bindings.find(desc.id);
      if (found == bindings.end())
        fail("missing constant tensor " + std::to_string(desc.id));
      found->second.validate();
      if (found->second.dtype != desc.dtype || found->second.dims != desc.dims)
        fail("bound constant shape/dtype mismatch for id " +
             std::to_string(desc.id));
      constants_.emplace(desc.id, found->second);
    }
  }

  RunResult run(const TensorMap &inputs, const RunOptions &options) override {
    TensorMap bindings = constants_;
    for (const auto &[id, tensor] : inputs) {
      const auto *desc = program_.tensor(id);
      if (desc && desc->has_role(ir::TensorRole::Input))
        bindings.insert_or_assign(id, tensor);
    }
    validate_bound_inputs(program_, bindings);
    for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
      auto tensors = initialize(program_, bindings);
      execute_once(program_, tensors);
    }

    if (options.iterations == 0)
      fail("run iterations must be nonzero");
    std::vector<double> elapsed;
    elapsed.reserve(options.iterations);
    TensorMap final_tensors;
    const bool tracing = telemetry::trace_events_requested(options);
    const auto trace_origin = std::chrono::steady_clock::now();
    const auto since_origin = [&](std::chrono::steady_clock::time_point at) {
      return std::chrono::duration<double, std::milli>(at - trace_origin)
          .count();
    };
    std::vector<TraceEvent> trace_events;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
      auto tensors = initialize(program_, bindings);
      const auto start = std::chrono::steady_clock::now();
      if (tracing) {
        // The typed reference executor has no submission queue: every
        // operation's host span is its complete execution.
        for (const auto &op : program_.operations) {
          const auto op_start = std::chrono::steady_clock::now();
          execute_operation(program_, op, tensors);
          const auto op_stop = std::chrono::steady_clock::now();
          TraceEvent event;
          event.category = std::string(telemetry::category::operation);
          event.name = "execute";
          event.operation_id = op.id;
          event.opcode = std::string(ir::opcode_name(op.opcode));
          event.host_start_ms = since_origin(op_start);
          event.host_end_ms = since_origin(op_stop);
          event.stream = "host";
          trace_events.push_back(std::move(event));
        }
      } else {
        execute_once(program_, tensors);
      }
      const auto stop = std::chrono::steady_clock::now();
      elapsed.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
      final_tensors = std::move(tensors);
    }

    RunResult result;
    result.backend_name = "cpu";
    result.device_name = "host";
    if (tracing) {
      result.trace_events = std::move(trace_events);
      result.trace_milliseconds =
          since_origin(std::chrono::steady_clock::now());
    }
    result.minimum_milliseconds = *std::min_element(elapsed.begin(), elapsed.end());
    result.maximum_milliseconds = *std::max_element(elapsed.begin(), elapsed.end());
    result.mean_milliseconds =
        std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
    for (const auto &desc : program_.tensors) {
      if (desc.has_role(ir::TensorRole::Output))
        result.outputs.emplace(desc.id, std::move(final_tensors.at(desc.id)));
    }
    telemetry::append_runtime_trace(result, program_, options);
    return result;
  }

  std::string name() const override { return "cpu"; }

private:
  ir::Program program_;
  TensorMap constants_;
};

class CpuExecutor final : public Executor {
public:
  std::unique_ptr<PreparedExecution>
  prepare(const ir::Program &program, const TensorMap &bindings,
          const RunOptions &) override {
    ir::verify(program);
    return std::make_unique<CpuPreparedExecution>(program, bindings);
  }

  std::string name() const override { return "cpu"; }
};

} // namespace

std::unique_ptr<Executor> make_cpu_executor() {
  return std::make_unique<CpuExecutor>();
}

} // namespace dif::runtime
