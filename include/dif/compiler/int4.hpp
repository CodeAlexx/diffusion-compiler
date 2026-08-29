#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <vector>

namespace dif::compiler {

enum class Int4Correction { None, OneOutlier };

struct Int4RewriteEntry {
  std::uint32_t source_tensor_id{};
  std::uint32_t packed_tensor_id{};
  std::uint32_t scales_tensor_id{};
  std::uint32_t column_scales_tensor_id{};
  std::uint32_t outlier_indices_tensor_id{};
  std::uint32_t outlier_residuals_tensor_id{};
  std::uint32_t dequantize_operation_id{};
  std::uint64_t group_size{};
  std::uint32_t bit_width{4U};
};

struct Int4Rewrite {
  ir::Program program;
  std::vector<Int4RewriteEntry> entries;
};

struct Int4QuantizedTensor {
  runtime::Tensor packed;
  runtime::Tensor scales;
  runtime::Tensor column_scales;
  runtime::Tensor outlier_indices;
  runtime::Tensor outlier_residuals;
  double squared_error{};
  double squared_reference{};
  float maximum_absolute_error{};
};

// Rewrites rank-2 constants used as Linear weights (and packed H3 QKV
// weights) into resident signed-INT4 payloads plus per-row group scales. The
// original tensor id becomes the dequantized internal value, so consumers do
// not need model-specific changes.
Int4Rewrite rewrite_int4_weights(const ir::Program &program,
                                 std::uint64_t group_size,
                                 Int4Correction correction =
                                     Int4Correction::None);

Int4QuantizedTensor quantize_int4_weight(const runtime::Tensor &source,
                                         std::uint64_t group_size,
                                         Int4Correction correction =
                                             Int4Correction::None);

Int4Rewrite rewrite_lowbit_weights(
    const ir::Program &program, std::uint32_t bit_width,
    std::uint64_t group_size,
    Int4Correction correction = Int4Correction::None,
    const std::vector<std::uint32_t> &calibrated_tensor_ids = {});

Int4QuantizedTensor quantize_lowbit_weight(
    const runtime::Tensor &source, std::uint32_t bit_width,
    std::uint64_t group_size,
    Int4Correction correction = Int4Correction::None,
    // Optional representative Linear inputs. When present, quantization uses
    // activation-RMS channel companding while preserving the original graph
    // interface through a dequantization-side column scale.
    const runtime::Tensor *calibration = nullptr,
    float activation_exponent = 0.5F);

} // namespace dif::compiler
