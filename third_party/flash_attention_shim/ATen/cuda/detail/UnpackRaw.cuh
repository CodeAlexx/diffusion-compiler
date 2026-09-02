#pragma once

#include <ATen/cuda/CUDAGeneratorImpl.h>

#include <cstdint>
#include <tuple>

namespace at::cuda::philox {
__host__ __device__ __forceinline__ std::tuple<std::uint64_t, std::uint64_t>
unpack(at::PhiloxCudaState) {
  return std::make_tuple(0ULL, 0ULL);
}
} // namespace at::cuda::philox
