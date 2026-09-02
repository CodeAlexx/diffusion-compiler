#pragma once

#include <cstddef>
#include <cstdint>

namespace dif::runtime {

struct CutlassGemmHandle;
struct CutlassInt8ScaledGemmHandle;
struct CutlassInt8ScaledF16GemmHandle;

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

// Prepared SM80+ INT8 GEMM with source-faithful dynamic row scale and static
// output-channel scale fused into a BF16 epilogue:
//   BF16 D[m,n] = BF16(I32(A[m,k] * B[n,k]^T) * row[m] * column[n]).
// The handle owns only prepared kernel metadata; all device pointers may be
// rebound per launch so one shape plan is reusable across model blocks.
CutlassInt8ScaledGemmHandle *create_cutlass_int8_scaled_gemm(
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    std::uintptr_t input, std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity);

bool launch_cutlass_int8_scaled_gemm(
    CutlassInt8ScaledGemmHandle *handle, std::uintptr_t input,
    std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity);

void destroy_cutlass_int8_scaled_gemm(CutlassInt8ScaledGemmHandle *handle);

// F16-output companion used by source-F16 semantic Linear operations.  The
// INT8 dot product and F32 scale products are identical to the BF16 route;
// only the observable 16-bit output conversion differs.
CutlassInt8ScaledF16GemmHandle *create_cutlass_int8_scaled_f16_gemm(
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    std::uintptr_t input, std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t bias, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity);

bool launch_cutlass_int8_scaled_f16_gemm(
    CutlassInt8ScaledF16GemmHandle *handle, std::uintptr_t input,
    std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t bias, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity);

void destroy_cutlass_int8_scaled_f16_gemm(
    CutlassInt8ScaledF16GemmHandle *handle);

} // namespace dif::runtime
