#pragma once

#include <cstddef>
#include <cstdint>

namespace dif::runtime {

struct CutlassGemmHandle;

struct CutlassGemmResources {
  const char *name{};
  std::uint32_t threadblock_m{};
  std::uint32_t threadblock_n{};
  std::uint32_t threadblock_k{};
  std::uint32_t warp_m{};
  std::uint32_t warp_n{};
  std::uint32_t warp_k{};
  std::uint32_t stages{};
  std::uint32_t threads_per_block{};
  std::uint32_t registers_per_thread{};
  std::uint64_t static_shared_bytes{};
  std::uint64_t dynamic_shared_bytes{};
  std::uint64_t maximum_dynamic_shared_bytes{};
};

// Schedule ids are stable CLI/API identities. All schedules perform the same
// row-major BF16 D = A * B^T operation with FP32 accumulation and BF16 output.
const char *cutlass_gemm_schedule_name(std::uint32_t schedule);

CutlassGemmHandle *create_cutlass_gemm(
    std::uint32_t schedule, std::uint32_t m, std::uint32_t n,
    std::uint32_t k, std::uintptr_t input, std::uintptr_t weight,
    std::uintptr_t output, std::uintptr_t stream, char *error,
    std::size_t error_capacity);

bool launch_cutlass_gemm(CutlassGemmHandle *handle, std::uintptr_t stream,
                         char *error, std::size_t error_capacity);

CutlassGemmResources cutlass_gemm_resources(CutlassGemmHandle *handle);

void destroy_cutlass_gemm(CutlassGemmHandle *handle);

} // namespace dif::runtime
