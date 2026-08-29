#include "dif/runtime/scalar.hpp"

#include "dif/support/error.hpp"

#include <bit>
#include <cmath>
#include <cstring>
#include <limits>

namespace dif::runtime {

bool is_float_dtype(ir::DType dtype) {
  return dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
         dtype == ir::DType::F16;
}

std::uint16_t float_to_bf16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float bf16_to_float(std::uint16_t value) {
  return std::bit_cast<float>(static_cast<std::uint32_t>(value) << 16U);
}

std::uint16_t float_to_f16(float value) {
  const auto bits = std::bit_cast<std::uint32_t>(value);
  const auto sign = static_cast<std::uint16_t>((bits >> 16U) & 0x8000U);
  const auto exponent = static_cast<std::uint32_t>((bits >> 23U) & 0xffU);
  const auto mantissa = bits & 0x7fffffU;

  if (exponent == 0xffU) {
    if (mantissa == 0U)
      return static_cast<std::uint16_t>(sign | 0x7c00U);
    const auto payload = static_cast<std::uint16_t>(mantissa >> 13U);
    return static_cast<std::uint16_t>(sign | 0x7c00U | payload | 1U);
  }

  const auto unbiased = static_cast<std::int32_t>(exponent) - 127;
  if (unbiased > 15)
    return static_cast<std::uint16_t>(sign | 0x7c00U);
  if (unbiased < -24)
    return sign;

  if (unbiased < -14) {
    const auto significand = mantissa | 0x800000U;
    const auto shift = static_cast<unsigned>(-unbiased - 1);
    auto rounded = significand >> shift;
    const auto remainder_mask = (1U << shift) - 1U;
    const auto remainder = significand & remainder_mask;
    const auto halfway = 1U << (shift - 1U);
    if (remainder > halfway || (remainder == halfway && (rounded & 1U) != 0U))
      ++rounded;
    return static_cast<std::uint16_t>(sign | rounded);
  }

  auto half_exponent = static_cast<std::uint32_t>(unbiased + 15);
  auto half_mantissa = mantissa >> 13U;
  const auto remainder = mantissa & 0x1fffU;
  if (remainder > 0x1000U || (remainder == 0x1000U && (half_mantissa & 1U) != 0U)) {
    ++half_mantissa;
    if (half_mantissa == 0x400U) {
      half_mantissa = 0U;
      ++half_exponent;
      if (half_exponent >= 0x1fU)
        return static_cast<std::uint16_t>(sign | 0x7c00U);
    }
  }
  return static_cast<std::uint16_t>(sign | (half_exponent << 10U) |
                                    half_mantissa);
}

float f16_to_float(std::uint16_t value) {
  const auto sign = static_cast<std::uint32_t>(value & 0x8000U) << 16U;
  auto exponent = static_cast<std::uint32_t>((value >> 10U) & 0x1fU);
  auto mantissa = static_cast<std::uint32_t>(value & 0x03ffU);
  std::uint32_t bits = 0U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      bits = sign;
    } else {
      std::int32_t normal_exponent = -14;
      while ((mantissa & 0x0400U) == 0U) {
        mantissa <<= 1U;
        --normal_exponent;
      }
      mantissa &= 0x03ffU;
      bits = sign |
             (static_cast<std::uint32_t>(normal_exponent + 127) << 23U) |
             (mantissa << 13U);
    }
  } else if (exponent == 0x1fU) {
    bits = sign | 0x7f800000U | (mantissa << 13U);
  } else {
    bits = sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
  }
  return std::bit_cast<float>(bits);
}

float load_float(const Tensor &tensor, std::uint64_t index) {
  tensor.validate();
  if (!is_float_dtype(tensor.dtype))
    fail("tensor dtype is not floating point");
  if (index >= tensor.element_count())
    fail("tensor float load index is out of range");
  if (tensor.dtype == ir::DType::F32) {
    float value = 0.0F;
    std::memcpy(&value, tensor.data() + index * sizeof(value), sizeof(value));
    return value;
  }
  std::uint16_t value = 0U;
  std::memcpy(&value, tensor.data() + index * sizeof(value), sizeof(value));
  return tensor.dtype == ir::DType::BF16 ? bf16_to_float(value)
                                         : f16_to_float(value);
}

void store_float(Tensor &tensor, std::uint64_t index, float value) {
  tensor.validate();
  if (!is_float_dtype(tensor.dtype))
    fail("tensor dtype is not floating point");
  if (index >= tensor.element_count())
    fail("tensor float store index is out of range");
  if (tensor.dtype == ir::DType::F32) {
    std::memcpy(tensor.mutable_data() + index * sizeof(value), &value,
                sizeof(value));
    return;
  }
  const auto converted = tensor.dtype == ir::DType::BF16
                             ? float_to_bf16(value)
                             : float_to_f16(value);
  std::memcpy(tensor.mutable_data() + index * sizeof(converted), &converted,
              sizeof(converted));
}

} // namespace dif::runtime
