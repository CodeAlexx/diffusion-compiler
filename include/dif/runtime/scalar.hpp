#pragma once

#include "dif/runtime/tensor.hpp"

#include <cstdint>

namespace dif::runtime {

bool is_float_dtype(ir::DType dtype);
float load_float(const Tensor &tensor, std::uint64_t index);
void store_float(Tensor &tensor, std::uint64_t index, float value);

std::uint16_t float_to_bf16(float value);
float bf16_to_float(std::uint16_t value);
std::uint16_t float_to_f16(float value);
float f16_to_float(std::uint16_t value);

} // namespace dif::runtime
