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

// Hardware/OS fact: the effective cgroup-v2 memory limit that applies to the
// calling process (the minimum of memory.max and memory.high along the
// hierarchy; 0 when unlimited) and the hierarchy's current charge. File-cache
// pages a process reads are charged to its cgroup, so policies that keep
// cache warm must respect this limit.
struct HostCgroupMemory {
  std::uint64_t limit_bytes{};
  std::uint64_t current_bytes{};
};
HostCgroupMemory probe_host_cgroup_memory();

target::TargetProfile probe_target(ProbeBackend backend, int device_ordinal = 0);
target::RuntimeBudget probe_runtime_budget(
    const target::TargetProfile &profile,
    const BudgetRequest &request = BudgetRequest{});
DeviceProbeResult probe_device(ProbeBackend backend, int device_ordinal = 0,
                               const BudgetRequest &request = BudgetRequest{});

} // namespace dif::runtime
