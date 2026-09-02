#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace dif::target {

enum class Vendor : std::uint32_t {
  Host = 1,
  Nvidia = 2,
  Unknown = 3,
};

enum class ArchitectureFamily : std::uint32_t {
  Host = 1,
  Kepler = 2,
  Maxwell = 3,
  Pascal = 4,
  Volta = 5,
  Turing = 6,
  Ampere = 7,
  Ada = 8,
  Hopper = 9,
  Blackwell = 10,
  Unknown = 11,
};

std::string_view vendor_name(Vendor vendor);
std::string_view architecture_name(ArchitectureFamily architecture);
ArchitectureFamily classify_nvidia_architecture(std::uint32_t compute_major,
                                                 std::uint32_t compute_minor);

struct PrecisionFeatures {
  bool tensor_cores{};
  bool fp16_tensor_cores{};
  bool bf16_tensor_cores{};
  bool fp8_tensor_cores{};
  bool int8_tensor_cores{};
  bool nvfp4_tensor_cores{};
};

struct ExecutionFeatures {
  bool cuda_graphs{};
  bool async_copy{};
  bool tensor_memory_accelerator{};
};

// Static facts and architecture-defined capabilities. Product name and device
// ordinal are diagnostics; the capability fingerprint intentionally excludes
// both so equivalent devices can share a compatible admitted plan.
struct TargetProfile {
  std::string backend;
  Vendor vendor{Vendor::Unknown};
  ArchitectureFamily architecture{ArchitectureFamily::Unknown};
  std::string product_name;
  std::int32_t device_ordinal{-1};
  std::uint32_t compute_major{};
  std::uint32_t compute_minor{};
  std::uint32_t multiprocessor_count{};
  std::uint32_t warp_size{};
  std::uint64_t total_device_memory_bytes{};
  std::uint64_t l2_cache_bytes{};
  std::uint64_t shared_memory_per_block_bytes{};
  std::uint64_t shared_memory_per_block_optin_bytes{};
  std::uint64_t shared_memory_per_multiprocessor_bytes{};
  PrecisionFeatures precision;
  ExecutionFeatures execution;
  std::uint32_t cuda_driver_version{};
  std::uint32_t cuda_runtime_version{};
  std::uint64_t cublaslt_version{};
  std::uint64_t cudnn_version{};
};

// Dynamic facts and explicit compiler/runtime limits for one execution.
struct RuntimeBudget {
  std::uint64_t free_device_memory_bytes{};
  std::uint64_t reserved_device_memory_bytes{};
  std::uint64_t usable_device_memory_bytes{};
  std::uint64_t total_host_memory_bytes{};
  std::uint64_t available_host_memory_bytes{};
  std::uint64_t host_memory_budget_bytes{};
  std::uint64_t pinned_host_memory_budget_bytes{};
  std::uint64_t workspace_budget_bytes{};
  std::uint64_t staging_budget_bytes{};
};

std::string target_fingerprint(const TargetProfile &profile);
std::string runtime_budget_class(const RuntimeBudget &budget);

} // namespace dif::target
