#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dif {

// Native reproduction of the CPU generator and normal fill used by
// torch.randn(..., generator=torch.manual_seed(seed), device="cpu") for F32
// contiguous tensors. Keep one instance alive across calls when the source
// runtime consumes several tensors from one generator.
class TorchCpuMt19937 {
public:
  explicit TorchCpuMt19937(std::uint64_t seed);

  std::uint32_t random();

private:
  std::uint32_t state_[624]{};
  std::int32_t left_{1};
  std::uint32_t next_{};

  static std::uint32_t twist(std::uint32_t left, std::uint32_t right);
  void next_state();
};

std::vector<float> torch_cpu_normal(TorchCpuMt19937 &generator,
                                    std::size_t count);
std::vector<float> torch_cpu_normal(std::size_t count, std::uint64_t seed);
bool torch_cpu_normal_uses_avx2();

} // namespace dif
