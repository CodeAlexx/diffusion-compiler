#include "dif/support/torch_cpu_rng.hpp"

#include "dif/support/error.hpp"

#include <cmath>

namespace dif {

#if DIF_TORCH_CPU_RNG_AVX2
bool torch_cpu_rng_avx2_available();
void torch_cpu_normal_fill16_avx2(float *data);
#endif

TorchCpuMt19937::TorchCpuMt19937(std::uint64_t seed) {
  state_[0] = static_cast<std::uint32_t>(seed);
  for (std::uint32_t index = 1U; index < 624U; ++index)
    state_[index] = 1812433253U *
                        (state_[index - 1U] ^ (state_[index - 1U] >> 30U)) +
                    index;
}

std::uint32_t TorchCpuMt19937::random() {
  if (--left_ == 0)
    next_state();
  auto value = state_[next_++];
  value ^= value >> 11U;
  value ^= (value << 7U) & 0x9d2c5680U;
  value ^= (value << 15U) & 0xefc60000U;
  value ^= value >> 18U;
  return value;
}

std::uint32_t TorchCpuMt19937::twist(std::uint32_t left,
                                     std::uint32_t right) {
  const auto mixed = (left & 0x80000000U) | (right & 0x7fffffffU);
  return (mixed >> 1U) ^ ((right & 1U) != 0U ? 0x9908b0dfU : 0U);
}

void TorchCpuMt19937::next_state() {
  left_ = 624;
  next_ = 0U;
  std::uint32_t index = 0U;
  for (; index < 624U - 397U; ++index)
    state_[index] =
        state_[index + 397U] ^ twist(state_[index], state_[index + 1U]);
  for (; index < 623U; ++index)
    state_[index] = state_[index + 397U - 624U] ^
                    twist(state_[index], state_[index + 1U]);
  state_[623U] = state_[396U] ^ twist(state_[623U], state_[0U]);
}

std::vector<float> torch_cpu_normal(TorchCpuMt19937 &generator,
                                    std::size_t count) {
  if (count == 0U)
    return {};
  if (count < 16U)
    fail("Torch CPU normal tensor is smaller than the 16-value fill block");
  std::vector<float> values(count);
  auto uniform = [&]() {
    return static_cast<float>(generator.random() & 0x00ffffffU) /
           16777216.0F;
  };
  for (auto &value : values)
    value = uniform();
  const auto transform = [](float *data) {
#if DIF_TORCH_CPU_RNG_AVX2
    if (torch_cpu_rng_avx2_available()) {
      torch_cpu_normal_fill16_avx2(data);
      return;
    }
#endif
    constexpr float two_pi = 6.2831853071795864769F;
    for (std::size_t index = 0U; index < 8U; ++index) {
      const auto radius = std::sqrt(-2.0F * std::log(1.0F - data[index]));
      const auto theta = two_pi * data[index + 8U];
      data[index] = radius * std::cos(theta);
      data[index + 8U] = radius * std::sin(theta);
    }
  };
  for (std::size_t index = 0U; index + 15U < count; index += 16U)
    transform(values.data() + index);
  if (count % 16U != 0U) {
    const auto offset = count - 16U;
    for (std::size_t index = 0U; index < 16U; ++index)
      values[offset + index] = uniform();
    transform(values.data() + offset);
  }
  return values;
}

std::vector<float> torch_cpu_normal(std::size_t count, std::uint64_t seed) {
  TorchCpuMt19937 generator(seed);
  return torch_cpu_normal(generator, count);
}

bool torch_cpu_normal_uses_avx2() {
#if DIF_TORCH_CPU_RNG_AVX2
  return torch_cpu_rng_avx2_available();
#else
  return false;
#endif
}

} // namespace dif
