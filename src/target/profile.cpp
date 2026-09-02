#include "dif/target/profile.hpp"

#include "dif/support/sha256.hpp"

#include <algorithm>
#include <span>
#include <string>

namespace dif::target {
namespace {

void append_bool(std::string &text, bool value) {
  text += value ? "1\n" : "0\n";
}

template <typename T>
void append_number(std::string &text, T value) {
  text += std::to_string(value);
  text.push_back('\n');
}

std::uint64_t gib_bucket(std::uint64_t bytes) {
  constexpr std::uint64_t gib = 1024ULL * 1024ULL * 1024ULL;
  return bytes / gib;
}

std::uint64_t mib_bucket(std::uint64_t bytes) {
  constexpr std::uint64_t mib = 1024ULL * 1024ULL;
  return bytes / mib;
}

} // namespace

std::string_view vendor_name(Vendor vendor) {
  switch (vendor) {
  case Vendor::Host: return "host";
  case Vendor::Nvidia: return "nvidia";
  case Vendor::Unknown: return "unknown";
  }
  return "unknown";
}

std::string_view architecture_name(ArchitectureFamily architecture) {
  switch (architecture) {
  case ArchitectureFamily::Host: return "host";
  case ArchitectureFamily::Kepler: return "kepler";
  case ArchitectureFamily::Maxwell: return "maxwell";
  case ArchitectureFamily::Pascal: return "pascal";
  case ArchitectureFamily::Volta: return "volta";
  case ArchitectureFamily::Turing: return "turing";
  case ArchitectureFamily::Ampere: return "ampere";
  case ArchitectureFamily::Ada: return "ada";
  case ArchitectureFamily::Hopper: return "hopper";
  case ArchitectureFamily::Blackwell: return "blackwell";
  case ArchitectureFamily::Unknown: return "unknown";
  }
  return "unknown";
}

ArchitectureFamily classify_nvidia_architecture(std::uint32_t major,
                                                 std::uint32_t minor) {
  if (major >= 10U)
    return ArchitectureFamily::Blackwell;
  if (major == 9U)
    return ArchitectureFamily::Hopper;
  if (major == 8U && minor == 9U)
    return ArchitectureFamily::Ada;
  if (major == 8U)
    return ArchitectureFamily::Ampere;
  if (major == 7U && minor >= 5U)
    return ArchitectureFamily::Turing;
  if (major == 7U)
    return ArchitectureFamily::Volta;
  if (major == 6U)
    return ArchitectureFamily::Pascal;
  if (major == 5U)
    return ArchitectureFamily::Maxwell;
  if (major == 3U)
    return ArchitectureFamily::Kepler;
  return ArchitectureFamily::Unknown;
}

std::string target_fingerprint(const TargetProfile &profile) {
  std::string identity = "diffusion-compiler-target-v1\n";
  identity += profile.backend + "\n";
  append_number(identity, static_cast<std::uint32_t>(profile.vendor));
  append_number(identity, static_cast<std::uint32_t>(profile.architecture));
  append_number(identity, profile.compute_major);
  append_number(identity, profile.compute_minor);
  append_number(identity, profile.multiprocessor_count);
  append_number(identity, profile.warp_size);
  append_number(identity, profile.total_device_memory_bytes);
  append_number(identity, profile.l2_cache_bytes);
  append_number(identity, profile.shared_memory_per_block_bytes);
  append_number(identity, profile.shared_memory_per_block_optin_bytes);
  append_number(identity, profile.shared_memory_per_multiprocessor_bytes);
  append_bool(identity, profile.precision.tensor_cores);
  append_bool(identity, profile.precision.fp16_tensor_cores);
  append_bool(identity, profile.precision.bf16_tensor_cores);
  append_bool(identity, profile.precision.fp8_tensor_cores);
  append_bool(identity, profile.precision.int8_tensor_cores);
  append_bool(identity, profile.precision.nvfp4_tensor_cores);
  append_bool(identity, profile.execution.cuda_graphs);
  append_bool(identity, profile.execution.async_copy);
  append_bool(identity, profile.execution.tensor_memory_accelerator);
  append_number(identity, profile.cuda_driver_version);
  append_number(identity, profile.cuda_runtime_version);
  append_number(identity, profile.cublaslt_version);
  append_number(identity, profile.cudnn_version);
  const auto bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(identity.data()), identity.size());
  return hex_digest(sha256(bytes));
}

std::string runtime_budget_class(const RuntimeBudget &budget) {
  return "device-usable-" +
         std::to_string(gib_bucket(budget.usable_device_memory_bytes)) +
         "gib_host-" +
         std::to_string(gib_bucket(budget.host_memory_budget_bytes)) +
         "gib_reserve-" +
         std::to_string(mib_bucket(budget.reserved_device_memory_bytes)) +
         "mib_pinned-" +
         std::to_string(mib_bucket(budget.pinned_host_memory_budget_bytes)) +
         "mib_workspace-" +
         std::to_string(mib_bucket(budget.workspace_budget_bytes)) +
         "mib_staging-" +
         std::to_string(mib_bucket(budget.staging_budget_bytes)) + "mib";
}

} // namespace dif::target
