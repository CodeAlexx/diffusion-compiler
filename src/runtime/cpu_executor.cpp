#include "dif/runtime/executor.hpp"

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
    const auto scaled = load_float(input, index) * load_float(scale, column);
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
  const auto inner = input.dims.back();
  const auto rows = input.element_count() / inner;
  const auto outputs = grad_output.dims.back();
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
      const auto angle = scale * (timestep * frequency);
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
                                  load_float(weight, column));
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
  auto &out = tensors.at(op.outputs[0]);
  const auto sequence = q_tensor.dims[0];
  const auto heads = q_tensor.dims[1];
  const auto dim = q_tensor.dims[2];
  const auto scale = static_cast<float>(op.f64(
      ir::AttrKey::AttentionScale, 1.0 / std::sqrt(static_cast<double>(dim))));
  const bool causal = op.boolean(ir::AttrKey::Causal, false);
  std::vector<float> probabilities(sequence);
  for (std::uint64_t query = 0; query < sequence; ++query) {
    const auto key_end = causal ? query + 1U : sequence;
    for (std::uint64_t head = 0; head < heads; ++head) {
      float maximum = -std::numeric_limits<float>::infinity();
      for (std::uint64_t key = 0; key < key_end; ++key) {
        float score = 0.0F;
        for (std::uint64_t d = 0; d < dim; ++d) {
          score = std::fma(
              load_float(q_tensor, (query * heads + head) * dim + d),
              load_float(k, (key * heads + head) * dim + d), score);
        }
        score *= scale;
        probabilities[key] = score;
        maximum = std::max(maximum, score);
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
              load_float(v, (key * heads + head) * dim + d), value);
        }
        store_float(out, (query * heads + head) * dim + d, value);
      }
    }
  }
}

void execute_once(const ir::Program &program, TensorMap &tensors) {
  for (const auto &op : program.operations) {
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
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
      auto tensors = initialize(program_, bindings);
      const auto start = std::chrono::steady_clock::now();
      execute_once(program_, tensors);
      const auto stop = std::chrono::steady_clock::now();
      elapsed.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
      final_tensors = std::move(tensors);
    }

    RunResult result;
    result.backend_name = "cpu";
    result.device_name = "host";
    result.minimum_milliseconds = *std::min_element(elapsed.begin(), elapsed.end());
    result.maximum_milliseconds = *std::max_element(elapsed.begin(), elapsed.end());
    result.mean_milliseconds =
        std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();
    result.iteration_milliseconds = std::move(elapsed);
    for (const auto &desc : program_.tensors) {
      if (desc.has_role(ir::TensorRole::Output))
        result.outputs.emplace(desc.id, std::move(final_tensors.at(desc.id)));
    }
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
