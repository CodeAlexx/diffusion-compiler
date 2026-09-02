#pragma once

#include "dif/target/profile.hpp"

#include <cstdint>

namespace dif::runtime {

enum class ProbeBackend : std::uint32_t {
  Host = 1,
  Cuda = 2,
};

struct BudgetRequest {
  std::uint64_t reserved_device_memory_bytes{256ULL * 1024ULL * 1024ULL};
  // Zero means use currently available host memory.
  std::uint64_t host_memory_budget_bytes{};
  std::uint64_t pinned_host_memory_budget_bytes{2ULL * 1024ULL * 1024ULL *
                                                1024ULL};
  std::uint64_t workspace_budget_bytes{64ULL * 1024ULL * 1024ULL};
  std::uint64_t staging_budget_bytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
};

struct DeviceProbeResult {
  target::TargetProfile target;
  target::RuntimeBudget budget;
};

target::TargetProfile probe_target(ProbeBackend backend, int device_ordinal = 0);
target::RuntimeBudget probe_runtime_budget(
    const target::TargetProfile &profile,
    const BudgetRequest &request = BudgetRequest{});
DeviceProbeResult probe_device(ProbeBackend backend, int device_ordinal = 0,
                               const BudgetRequest &request = BudgetRequest{});

} // namespace dif::runtime
