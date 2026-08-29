#include "dif/compiler/int4.hpp"

#include "dif/ir/verify.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>

namespace dif::compiler {
namespace {

bool graph_float(ir::DType dtype) {
  return dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
         dtype == ir::DType::F16;
}

bool valid_group(std::uint64_t group_size) {
  return group_size >= 16U && group_size <= 256U &&
         (group_size & (group_size - 1U)) == 0U;
}

float load_float_unchecked(ir::DType dtype, const std::uint8_t *data,
                           std::uint64_t index) {
  if (dtype == ir::DType::F32) {
    float value = 0.0F;
    std::memcpy(&value, data + index * sizeof(value), sizeof(value));
    return value;
  }
  std::uint16_t value = 0U;
  std::memcpy(&value, data + index * sizeof(value), sizeof(value));
  return dtype == ir::DType::BF16 ? runtime::bf16_to_float(value)
                                  : runtime::f16_to_float(value);
}

void store_float_unchecked(ir::DType dtype, std::uint8_t *data,
                           std::uint64_t index, float value) {
  if (dtype == ir::DType::F32) {
    std::memcpy(data + index * sizeof(value), &value, sizeof(value));
    return;
  }
  const auto converted = dtype == ir::DType::BF16
                             ? runtime::float_to_bf16(value)
                             : runtime::float_to_f16(value);
  std::memcpy(data + index * sizeof(converted), &converted,
              sizeof(converted));
}

std::uint32_t checked_next(std::uint32_t &next, const char *kind) {
  if (next == 0U || next == std::numeric_limits<std::uint32_t>::max())
    fail(std::string("DiffIR ") + kind + " id space is exhausted");
  return next++;
}

} // namespace

Int4Rewrite rewrite_lowbit_weights(const ir::Program &program,
                                   std::uint32_t bit_width,
                                   std::uint64_t group_size,
                                   Int4Correction correction,
                                   const std::vector<std::uint32_t>
                                       &calibrated_tensor_ids) {
  ir::verify(program);
  if (bit_width != 4U && bit_width != 5U)
    fail("low-bit weight encoding admits 4 or 5 bits");
  if (bit_width != 4U && correction != Int4Correction::None)
    fail("one-outlier correction is currently defined only for INT4");
  if (!valid_group(group_size))
    fail("INT4 group size must be a power of two in [16,256]");

  std::map<std::uint32_t, std::size_t> first_consumers;
  const std::unordered_set<std::uint32_t> calibrated(
      calibrated_tensor_ids.begin(), calibrated_tensor_ids.end());
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    const auto &operation = program.operations[index];
    std::uint32_t candidate = 0U;
    if (operation.opcode == ir::Opcode::Linear &&
        operation.inputs.size() >= 2U)
      candidate = operation.inputs[1];
    else if (operation.opcode == ir::Opcode::H3DeinterleaveQkvWeight &&
             !operation.inputs.empty())
      candidate = operation.inputs[0];
    if (candidate == 0U)
      continue;
    const auto *description = program.tensor(candidate);
    if (!description || !description->has_role(ir::TensorRole::Constant) ||
        !graph_float(description->dtype) || description->dims.size() != 2U)
      continue;
    if (description->dims[1] % group_size != 0U ||
        (description->dims[1] * bit_width) % 8U != 0U)
      fail("low-bit weight tensor " + std::to_string(candidate) +
           " has a K dimension incompatible with the group size");
    first_consumers.try_emplace(candidate, index);
  }

  Int4Rewrite output;
  output.program = program;
  std::uint32_t maximum_tensor = 0U;
  for (const auto &tensor : program.tensors)
    maximum_tensor = std::max(maximum_tensor, tensor.id);
  std::uint32_t maximum_operation = 0U;
  for (const auto &operation : program.operations)
    maximum_operation = std::max(maximum_operation, operation.id);
  if (maximum_tensor == std::numeric_limits<std::uint32_t>::max() ||
      maximum_operation == std::numeric_limits<std::uint32_t>::max())
    fail("DiffIR id space is exhausted");
  auto next_tensor = maximum_tensor + 1U;
  auto next_operation = maximum_operation + 1U;

  std::unordered_map<std::uint32_t, Int4RewriteEntry> entries;
  for (const auto &[source_id, first_consumer] : first_consumers) {
    (void)first_consumer;
    auto *source = static_cast<ir::TensorDesc *>(nullptr);
    for (auto &tensor : output.program.tensors) {
      if (tensor.id == source_id) {
        source = &tensor;
        break;
      }
    }
    if (!source)
      fail("INT4 rewrite lost a source tensor");
    const auto source_dtype = source->dtype;
    const auto source_dims = source->dims;
    source->roles &= ~(static_cast<std::uint32_t>(ir::TensorRole::Constant) |
                       static_cast<std::uint32_t>(ir::TensorRole::Streamed));
    const auto packed_id = checked_next(next_tensor, "tensor");
    const auto scales_id = checked_next(next_tensor, "tensor");
    const auto column_scales_id = calibrated.contains(source_id)
                                      ? checked_next(next_tensor, "tensor")
                                      : 0U;
    const auto outlier_indices_id =
        correction == Int4Correction::OneOutlier
            ? checked_next(next_tensor, "tensor")
            : 0U;
    const auto outlier_residuals_id =
        correction == Int4Correction::OneOutlier
            ? checked_next(next_tensor, "tensor")
            : 0U;
    const auto operation_id = checked_next(next_operation, "operation");
    output.program.tensors.push_back(
        {packed_id, ir::DType::I8, ir::TensorRole::Constant,
         {source_dims[0], source_dims[1] * bit_width / 8U}});
    output.program.tensors.push_back(
        {scales_id, source_dtype, ir::TensorRole::Constant,
         {source_dims[0], source_dims[1] / group_size}});
    if (column_scales_id != 0U)
      output.program.tensors.push_back(
          {column_scales_id, source_dtype, ir::TensorRole::Constant,
           {source_dims[1]}});
    if (correction == Int4Correction::OneOutlier) {
      output.program.tensors.push_back(
          {outlier_indices_id, ir::DType::I8, ir::TensorRole::Constant,
           {source_dims[0], source_dims[1] / group_size}});
      output.program.tensors.push_back(
          {outlier_residuals_id, source_dtype, ir::TensorRole::Constant,
           {source_dims[0], source_dims[1] / group_size}});
    }
    entries.emplace(source_id,
                    Int4RewriteEntry{source_id, packed_id, scales_id,
                                     column_scales_id,
                                     outlier_indices_id, outlier_residuals_id,
                                     operation_id, group_size, bit_width});
  }

  std::vector<ir::Operation> operations;
  operations.reserve(program.operations.size() + entries.size());
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    for (const auto &[source_id, first_consumer] : first_consumers) {
      if (first_consumer != index)
        continue;
      const auto &entry = entries.at(source_id);
      std::vector<std::uint32_t> inputs = {entry.packed_tensor_id,
                                           entry.scales_tensor_id};
      if (entry.column_scales_tensor_id != 0U)
        inputs.push_back(entry.column_scales_tensor_id);
      if (entry.outlier_indices_tensor_id != 0U) {
        inputs.push_back(entry.outlier_indices_tensor_id);
        inputs.push_back(entry.outlier_residuals_tensor_id);
      }
      operations.push_back(
          {entry.dequantize_operation_id,
           bit_width == 4U ? ir::Opcode::DequantizeInt4
                           : ir::Opcode::DequantizeInt5,
           std::move(inputs),
           {entry.source_tensor_id},
           {ir::Attribute::u64(ir::AttrKey::GroupSize, group_size)}});
      output.entries.push_back(entry);
    }
    operations.push_back(program.operations[index]);
  }
  output.program.operations = std::move(operations);
  ir::verify(output.program);
  return output;
}

Int4Rewrite rewrite_int4_weights(const ir::Program &program,
                                 std::uint64_t group_size,
                                 Int4Correction correction) {
  return rewrite_lowbit_weights(program, 4U, group_size, correction);
}

Int4QuantizedTensor quantize_lowbit_weight(const runtime::Tensor &source,
                                           std::uint32_t bit_width,
                                           std::uint64_t group_size,
                                           Int4Correction correction,
                                           const runtime::Tensor *calibration,
                                           float activation_exponent) {
  source.validate();
  if ((bit_width != 4U && bit_width != 5U) ||
      (bit_width != 4U && correction != Int4Correction::None) ||
      !valid_group(group_size) || !graph_float(source.dtype) ||
      source.dims.size() != 2U || source.dims[1] % group_size != 0U ||
      (source.dims[1] * bit_width) % 8U != 0U)
    fail("low-bit quantization requires a rank-2 float [N,K] tensor and a "
         "compatible power-of-two group size in [16,256]");

  const auto rows = source.dims[0];
  const auto columns = source.dims[1];
  const auto groups = columns / group_size;
  const auto row_bytes = columns * bit_width / 8U;
  const auto maximum_quantized =
      static_cast<int>((1U << (bit_width - 1U)) - 1U);
  const auto mask = static_cast<std::uint16_t>((1U << bit_width) - 1U);
  const auto *source_data = source.data();
  Int4QuantizedTensor output;
  std::vector<float> calibration_importance;
  if (calibration) {
    calibration->validate();
    if (!graph_float(calibration->dtype) ||
        calibration->element_count() % columns != 0U ||
        !(activation_exponent >= 0.0F && activation_exponent <= 1.0F))
      fail("activation-aware low-bit quantization requires float calibration "
           "rows with K columns and an exponent in [0,1]");
    const auto calibration_rows = calibration->element_count() / columns;
    std::vector<float> importance(static_cast<std::size_t>(columns));
    float maximum_importance = 0.0F;
    for (std::uint64_t column = 0; column < columns; ++column) {
      double squares = 0.0;
      for (std::uint64_t row = 0; row < calibration_rows; ++row) {
        const auto value = load_float_unchecked(
            calibration->dtype, calibration->data(), row * columns + column);
        if (!std::isfinite(value))
          fail("activation calibration contains non-finite values");
        squares += static_cast<double>(value) * value;
      }
      importance[static_cast<std::size_t>(column)] = static_cast<float>(
          std::sqrt(squares / static_cast<double>(calibration_rows)));
      maximum_importance =
          std::max(maximum_importance,
                   importance[static_cast<std::size_t>(column)]);
    }
    calibration_importance = importance;
    const auto floor = std::max(maximum_importance * 1.0e-4F, 1.0e-12F);
    float minimum_scale = std::numeric_limits<float>::infinity();
    float maximum_scale = 0.0F;
    for (auto &value : importance) {
      value = std::pow(std::max(value, floor), activation_exponent);
      minimum_scale = std::min(minimum_scale, value);
      maximum_scale = std::max(maximum_scale, value);
    }
    const auto normalization = std::sqrt(minimum_scale * maximum_scale);
    output.column_scales = {source.dtype, {columns}, {}};
    output.column_scales.bytes.resize(
        static_cast<std::size_t>(columns * ir::dtype_size(source.dtype)));
    auto *column_scale_data = output.column_scales.mutable_data();
    for (std::uint64_t column = 0; column < columns; ++column) {
      const auto weight_scale = std::clamp(
          importance[static_cast<std::size_t>(column)] / normalization,
          1.0F / 16.0F, 16.0F);
      store_float_unchecked(source.dtype, column_scale_data, column,
                            1.0F / weight_scale);
    }
  }
  output.packed = {ir::DType::I8, {rows, row_bytes}, {}};
  output.packed.bytes.resize(static_cast<std::size_t>(rows * row_bytes));
  output.scales = {source.dtype, {rows, groups}, {}};
  output.scales.bytes.resize(
      static_cast<std::size_t>(rows * groups * ir::dtype_size(source.dtype)));
  if (correction == Int4Correction::OneOutlier) {
    output.outlier_indices = {ir::DType::I8, {rows, groups}, {}};
    output.outlier_indices.bytes.resize(
        static_cast<std::size_t>(rows * groups));
    output.outlier_residuals = {source.dtype, {rows, groups}, {}};
    output.outlier_residuals.bytes.resize(static_cast<std::size_t>(
        rows * groups * ir::dtype_size(source.dtype)));
  }

  auto *packed_data = output.packed.mutable_data();
  auto *scale_data = output.scales.mutable_data();
  const auto *column_scale_data =
      calibration ? output.column_scales.data() : nullptr;
  auto *outlier_index_data = correction == Int4Correction::OneOutlier
                                 ? output.outlier_indices.mutable_data()
                                 : nullptr;
  auto *outlier_residual_data = correction == Int4Correction::OneOutlier
                                    ? output.outlier_residuals.mutable_data()
                                    : nullptr;
  std::vector<double> row_squared_error(static_cast<std::size_t>(rows), 0.0);
  std::vector<double> row_squared_reference(static_cast<std::size_t>(rows),
                                            0.0);
  std::vector<float> row_maximum_error(static_cast<std::size_t>(rows), 0.0F);
  std::vector<std::uint8_t> row_nonfinite(static_cast<std::size_t>(rows), 0U);

  const auto quantize_row = [&](std::uint64_t row) {
    for (std::uint64_t group_index = 0; group_index < groups; ++group_index) {
      const auto begin = row * columns + group_index * group_size;
      float maximum = 0.0F;
      std::uint64_t outlier_offset = 0U;
      for (std::uint64_t offset = 0; offset < group_size; ++offset) {
        const auto column = group_index * group_size + offset;
        const auto multiplier = calibration
                                    ? load_float_unchecked(
                                          source.dtype, column_scale_data,
                                          column)
                                    : 1.0F;
        const auto value = load_float_unchecked(source.dtype, source_data,
                                                begin + offset) /
                           multiplier;
        if (!std::isfinite(value)) {
          row_nonfinite[static_cast<std::size_t>(row)] = 1U;
          return;
        }
        if (std::abs(value) > maximum) {
          maximum = std::abs(value);
          outlier_offset = offset;
        }
      }
      if (correction == Int4Correction::OneOutlier) {
        maximum = 0.0F;
        for (std::uint64_t offset = 0; offset < group_size; ++offset) {
          if (offset != outlier_offset) {
            const auto column = group_index * group_size + offset;
            const auto multiplier =
                calibration
                    ? load_float_unchecked(source.dtype, column_scale_data,
                                           column)
                    : 1.0F;
            maximum = std::max(
                maximum,
                std::abs(load_float_unchecked(source.dtype, source_data,
                                              begin + offset) /
                         multiplier));
          }
        }
        outlier_index_data[static_cast<std::size_t>(row * groups +
                                                     group_index)] =
            static_cast<std::uint8_t>(outlier_offset);
      }
      store_float_unchecked(
          source.dtype, scale_data, row * groups + group_index,
          maximum == 0.0F
              ? 0.0F
              : maximum / static_cast<float>(maximum_quantized));
      if (calibration && maximum != 0.0F) {
        // With integer codes fixed, solve the diagonal activation-weighted
        // least-squares scale exactly. Re-assign codes and solve twice; this
        // keeps packaging linear in weight size while targeting Linear output
        // error rather than raw weight error.
        for (unsigned refinement = 0; refinement < 2U; ++refinement) {
          const auto current_scale = load_float_unchecked(
              source.dtype, scale_data, row * groups + group_index);
          double numerator = 0.0;
          double denominator = 0.0;
          for (std::uint64_t offset = 0; offset < group_size; ++offset) {
            const auto column = group_index * group_size + offset;
            const auto multiplier = load_float_unchecked(
                source.dtype, column_scale_data, column);
            const auto original = load_float_unchecked(
                source.dtype, source_data, begin + offset);
            auto quantized = static_cast<int>(
                std::round((original / multiplier) / current_scale));
            quantized =
                std::clamp(quantized, -maximum_quantized, maximum_quantized);
            const auto basis = static_cast<double>(quantized) * multiplier;
            const auto activation = static_cast<double>(
                calibration_importance[static_cast<std::size_t>(column)]);
            const auto importance_squared = activation * activation;
            numerator += importance_squared * basis * original;
            denominator += importance_squared * basis * basis;
          }
          if (denominator > 0.0) {
            const auto refined = static_cast<float>(numerator / denominator);
            if (std::isfinite(refined) && refined > 0.0F)
              store_float_unchecked(source.dtype, scale_data,
                                    row * groups + group_index, refined);
          }
        }
      }
      const auto scale = load_float_unchecked(
          source.dtype, scale_data, row * groups + group_index);
      for (std::uint64_t offset = 0; offset < group_size; ++offset) {
        const auto index = begin + offset;
        const auto original_value =
            load_float_unchecked(source.dtype, source_data, index);
        const auto column = group_index * group_size + offset;
        const auto multiplier = calibration
                                    ? load_float_unchecked(
                                          source.dtype, column_scale_data,
                                          column)
                                    : 1.0F;
        const auto value = original_value / multiplier;
        auto quantized = 0;
        if (scale != 0.0F)
          quantized = static_cast<int>(std::round(value / scale));
        quantized =
            std::clamp(quantized, -maximum_quantized, maximum_quantized);
        const auto encoded = static_cast<std::uint16_t>(quantized) & mask;
        const auto bit_offset = column * bit_width;
        const auto byte_index = static_cast<std::size_t>(
            row * row_bytes + bit_offset / 8U);
        const auto shift = static_cast<unsigned>(bit_offset % 8U);
        const auto shifted = static_cast<std::uint16_t>(encoded << shift);
        packed_data[byte_index] |=
            static_cast<std::uint8_t>(shifted & 0xffU);
        if (shift + bit_width > 8U)
          packed_data[byte_index + 1U] |=
              static_cast<std::uint8_t>(shifted >> 8U);
        auto reconstructed_scaled = static_cast<float>(quantized) * scale;
        if (correction == Int4Correction::OneOutlier &&
            offset == outlier_offset) {
          store_float_unchecked(source.dtype, outlier_residual_data,
                                row * groups + group_index,
                                value - reconstructed_scaled);
          reconstructed_scaled += load_float_unchecked(
              source.dtype, outlier_residual_data,
              row * groups + group_index);
        }
        const auto reconstructed = reconstructed_scaled * multiplier;
        const auto error =
            static_cast<double>(reconstructed) - original_value;
        row_squared_error[static_cast<std::size_t>(row)] += error * error;
        row_squared_reference[static_cast<std::size_t>(row)] +=
            static_cast<double>(original_value) * original_value;
        row_maximum_error[static_cast<std::size_t>(row)] = std::max(
            row_maximum_error[static_cast<std::size_t>(row)],
            static_cast<float>(std::abs(error)));
      }
    }
  };

  const auto available_workers =
      std::max(1U, std::thread::hardware_concurrency());
  const auto worker_count = static_cast<unsigned>(
      std::min<std::uint64_t>(rows, available_workers));
  std::vector<std::thread> workers;
  workers.reserve(worker_count > 0U ? worker_count - 1U : 0U);
  for (unsigned worker = 1U; worker < worker_count; ++worker) {
    workers.emplace_back([&, worker] {
      for (std::uint64_t row = worker; row < rows; row += worker_count)
        quantize_row(row);
    });
  }
  for (std::uint64_t row = 0U; row < rows; row += worker_count)
    quantize_row(row);
  for (auto &worker : workers)
    worker.join();

  for (std::uint64_t row = 0; row < rows; ++row) {
    if (row_nonfinite[static_cast<std::size_t>(row)] != 0U)
      fail("low-bit quantization rejects non-finite weights");
    output.squared_error +=
        row_squared_error[static_cast<std::size_t>(row)];
    output.squared_reference +=
        row_squared_reference[static_cast<std::size_t>(row)];
    output.maximum_absolute_error = std::max(
        output.maximum_absolute_error,
        row_maximum_error[static_cast<std::size_t>(row)]);
  }
  output.packed.validate();
  output.scales.validate();
  if (calibration)
    output.column_scales.validate();
  if (correction == Int4Correction::OneOutlier) {
    output.outlier_indices.validate();
    output.outlier_residuals.validate();
  }
  return output;
}

Int4QuantizedTensor quantize_int4_weight(const runtime::Tensor &source,
                                         std::uint64_t group_size,
                                         Int4Correction correction) {
  return quantize_lowbit_weight(source, 4U, group_size, correction);
}

} // namespace dif::compiler
