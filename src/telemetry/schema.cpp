#include "dif/telemetry/schema.hpp"

#include "dif/build_info.hpp"
#include "dif/opt/plan.hpp"

#include <string>

namespace dif::telemetry {
namespace {

std::string boolean(bool value) { return value ? "true" : "false"; }

std::string nullable_version(std::uint64_t value) {
  return value == 0U ? "null" : std::to_string(value);
}

} // namespace

std::string serialize_probe(const target::TargetProfile &profile,
                            const target::RuntimeBudget &budget) {
  std::string out = "{\n";
  out += "  \"schema\": {\"name\": " +
         opt::json_quote(kSchemaName) + ", \"version\": " +
         std::to_string(kSchemaVersion) + "},\n";
  out += "  \"kind\": \"device-probe\",\n";
  out += "  \"provenance\": {\"compiler\": " +
         opt::json_quote(build::compiler_name()) + ", \"version\": " +
         opt::json_quote(build::compiler_version()) +
         ", \"revision\": " + opt::json_quote(build::compiler_revision()) +
         "},\n";
  out += "  \"hardware\": {\n";
  out += "    \"backend\": " + opt::json_quote(profile.backend) + ",\n";
  out += "    \"vendor\": " +
         opt::json_quote(target::vendor_name(profile.vendor)) + ",\n";
  out += "    \"product_name\": " + opt::json_quote(profile.product_name) +
         ",\n";
  out += "    \"architecture_family\": " +
         opt::json_quote(target::architecture_name(profile.architecture)) +
         ",\n";
  out += "    \"device_ordinal\": " +
         std::to_string(profile.device_ordinal) + ",\n";
  out += "    \"compute_capability\": {\"major\": " +
         std::to_string(profile.compute_major) + ", \"minor\": " +
         std::to_string(profile.compute_minor) + ", \"sm\": " +
         opt::json_quote(profile.backend == "cuda"
                             ? "sm_" + std::to_string(profile.compute_major) +
                                   std::to_string(profile.compute_minor)
                             : "not-applicable") +
         "},\n";
  out += "    \"multiprocessor_count\": " +
         std::to_string(profile.multiprocessor_count) + ",\n";
  out += "    \"warp_size\": " + std::to_string(profile.warp_size) + ",\n";
  out += "    \"total_vram_bytes\": " +
         std::to_string(profile.total_device_memory_bytes) + ",\n";
  out += "    \"l2_cache_bytes\": " +
         std::to_string(profile.l2_cache_bytes) + ",\n";
  out += "    \"shared_memory\": {\"per_block_bytes\": " +
         std::to_string(profile.shared_memory_per_block_bytes) +
         ", \"per_block_optin_bytes\": " +
         std::to_string(profile.shared_memory_per_block_optin_bytes) +
         ", \"per_multiprocessor_bytes\": " +
         std::to_string(profile.shared_memory_per_multiprocessor_bytes) +
         "},\n";
  out += "    \"precision_features\": {\"tensor_cores\": " +
         boolean(profile.precision.tensor_cores) +
         ", \"fp16_tensor_cores\": " +
         boolean(profile.precision.fp16_tensor_cores) +
         ", \"bf16_tensor_cores\": " +
         boolean(profile.precision.bf16_tensor_cores) +
         ", \"fp8_tensor_cores\": " +
         boolean(profile.precision.fp8_tensor_cores) +
         ", \"int8_tensor_cores\": " +
         boolean(profile.precision.int8_tensor_cores) +
         ", \"nvfp4_tensor_cores\": " +
         boolean(profile.precision.nvfp4_tensor_cores) + "},\n";
  out += "    \"execution_features\": {\"cuda_graphs\": " +
         boolean(profile.execution.cuda_graphs) + ", \"async_copy\": " +
         boolean(profile.execution.async_copy) +
         ", \"tensor_memory_accelerator\": " +
         boolean(profile.execution.tensor_memory_accelerator) + "},\n";
  out += "    \"versions\": {\"cuda_driver\": " +
         nullable_version(profile.cuda_driver_version) +
         ", \"cuda_runtime\": " +
         nullable_version(profile.cuda_runtime_version) +
         ", \"cublaslt\": " + nullable_version(profile.cublaslt_version) +
         ", \"cudnn\": " + nullable_version(profile.cudnn_version) + "},\n";
  out += "    \"target_fingerprint\": " +
         opt::json_quote(target::target_fingerprint(profile)) + "\n";
  out += "  },\n";
  out += "  \"runtime_budget\": {\n";
  out += "    \"free_vram_bytes\": " +
         std::to_string(budget.free_device_memory_bytes) + ",\n";
  out += "    \"reserved_vram_bytes\": " +
         std::to_string(budget.reserved_device_memory_bytes) + ",\n";
  out += "    \"usable_vram_bytes\": " +
         std::to_string(budget.usable_device_memory_bytes) + ",\n";
  out += "    \"total_host_memory_bytes\": " +
         std::to_string(budget.total_host_memory_bytes) + ",\n";
  out += "    \"available_host_memory_bytes\": " +
         std::to_string(budget.available_host_memory_bytes) + ",\n";
  out += "    \"host_memory_budget_bytes\": " +
         std::to_string(budget.host_memory_budget_bytes) + ",\n";
  out += "    \"pinned_host_memory_budget_bytes\": " +
         std::to_string(budget.pinned_host_memory_budget_bytes) + ",\n";
  out += "    \"workspace_budget_bytes\": " +
         std::to_string(budget.workspace_budget_bytes) + ",\n";
  out += "    \"staging_budget_bytes\": " +
         std::to_string(budget.staging_budget_bytes) + ",\n";
  out += "    \"budget_class\": " +
         opt::json_quote(target::runtime_budget_class(budget)) + "\n";
  out += "  }\n";
  out += "}\n";
  return out;
}

} // namespace dif::telemetry
