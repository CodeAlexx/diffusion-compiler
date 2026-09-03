#include "dif/telemetry/schema.hpp"

#include "dif/build_info.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include <chrono>
#include <cstdio>
#include <ctime>
#include <map>
#include <string>

namespace dif::telemetry {
namespace {

Value nullable_version(std::uint64_t value) {
  if (value == 0U)
    return Value(nullptr);
  return Value(value);
}

} // namespace

std::string utc_timestamp_now() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
  gmtime_r(&seconds, &utc);
  char buffer[32];
  std::strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return buffer;
}

Object provenance_section() {
  Object out;
  out.set("compiler", build::compiler_name());
  out.set("version", build::compiler_version());
  out.set("revision", build::compiler_revision());
  out.set("generated_at", utc_timestamp_now());
  return out;
}

Object make_document(std::string_view kind) {
  Object schema;
  schema.set("name", kSchemaName);
  schema.set("version", kSchemaVersion);
  Object out;
  out.set("schema", std::move(schema));
  out.set("kind", kind);
  out.set("provenance", provenance_section());
  return out;
}

Object hardware_section(const target::TargetProfile &profile) {
  Object out;
  out.set("backend", profile.backend);
  out.set("vendor", target::vendor_name(profile.vendor));
  out.set("product_name", profile.product_name);
  out.set("architecture_family",
          target::architecture_name(profile.architecture));
  out.set("device_ordinal", profile.device_ordinal);
  Object compute;
  compute.set("major", profile.compute_major);
  compute.set("minor", profile.compute_minor);
  compute.set("sm", profile.backend == "cuda"
                        ? "sm_" + std::to_string(profile.compute_major) +
                              std::to_string(profile.compute_minor)
                        : std::string("not-applicable"));
  out.set("compute_capability", std::move(compute));
  out.set("multiprocessor_count", profile.multiprocessor_count);
  out.set("warp_size", profile.warp_size);
  out.set("total_vram_bytes", profile.total_device_memory_bytes);
  out.set("l2_cache_bytes", profile.l2_cache_bytes);
  Object shared;
  shared.set("per_block_bytes", profile.shared_memory_per_block_bytes);
  shared.set("per_block_optin_bytes",
             profile.shared_memory_per_block_optin_bytes);
  shared.set("per_multiprocessor_bytes",
             profile.shared_memory_per_multiprocessor_bytes);
  out.set("shared_memory", std::move(shared));
  Object precision;
  precision.set("tensor_cores", profile.precision.tensor_cores);
  precision.set("fp16_tensor_cores", profile.precision.fp16_tensor_cores);
  precision.set("bf16_tensor_cores", profile.precision.bf16_tensor_cores);
  precision.set("fp8_tensor_cores", profile.precision.fp8_tensor_cores);
  precision.set("int8_tensor_cores", profile.precision.int8_tensor_cores);
  precision.set("nvfp4_tensor_cores", profile.precision.nvfp4_tensor_cores);
  out.set("precision_features", std::move(precision));
  Object execution;
  execution.set("cuda_graphs", profile.execution.cuda_graphs);
  execution.set("async_copy", profile.execution.async_copy);
  execution.set("tensor_memory_accelerator",
                profile.execution.tensor_memory_accelerator);
  out.set("execution_features", std::move(execution));
  Object versions;
  versions.set("cuda_driver", nullable_version(profile.cuda_driver_version));
  versions.set("cuda_runtime", nullable_version(profile.cuda_runtime_version));
  versions.set("cublaslt", nullable_version(profile.cublaslt_version));
  versions.set("cudnn", nullable_version(profile.cudnn_version));
  out.set("versions", std::move(versions));
  out.set("target_fingerprint", target::target_fingerprint(profile));
  return out;
}

Object runtime_budget_section(const target::RuntimeBudget &budget) {
  Object out;
  out.set("free_vram_bytes", budget.free_device_memory_bytes);
  out.set("reserved_vram_bytes", budget.reserved_device_memory_bytes);
  out.set("usable_vram_bytes", budget.usable_device_memory_bytes);
  out.set("total_host_memory_bytes", budget.total_host_memory_bytes);
  out.set("available_host_memory_bytes", budget.available_host_memory_bytes);
  out.set("host_memory_budget_bytes", budget.host_memory_budget_bytes);
  out.set("pinned_host_memory_budget_bytes",
          budget.pinned_host_memory_budget_bytes);
  out.set("workspace_budget_bytes", budget.workspace_budget_bytes);
  out.set("staging_budget_bytes", budget.staging_budget_bytes);
  out.set("budget_class", target::runtime_budget_class(budget));
  return out;
}

Object launch_telemetry_section(const runtime::LaunchTelemetry &telemetry) {
  Object out;
  out.set("kernel_launches", telemetry.kernel_launches);
  out.set("cublaslt_matmuls", telemetry.cublaslt_matmuls);
  out.set("cublas_gemms", telemetry.cublas_gemms);
  out.set("cudnn_attention_dispatches", telemetry.cudnn_attention_dispatches);
  out.set("cudnn_convolution_dispatches",
          telemetry.cudnn_convolution_dispatches);
  out.set("cutlass_launches", telemetry.cutlass_launches);
  out.set("ck_attention_dispatches", telemetry.ck_attention_dispatches);
  out.set("h2d_copies", telemetry.h2d_copies);
  out.set("h2d_bytes", telemetry.h2d_bytes);
  out.set("d2h_copies", telemetry.d2h_copies);
  out.set("d2h_bytes", telemetry.d2h_bytes);
  out.set("d2d_copies", telemetry.d2d_copies);
  out.set("d2d_bytes", telemetry.d2d_bytes);
  out.set("event_records", telemetry.event_records);
  out.set("stream_wait_events", telemetry.stream_wait_events);
  out.set("host_event_synchronizes", telemetry.host_event_synchronizes);
  out.set("host_stream_synchronizes", telemetry.host_stream_synchronizes);
  out.set("device_mem_allocs", telemetry.device_mem_allocs);
  out.set("pinned_mem_allocs", telemetry.pinned_mem_allocs);
  return out;
}

Object pipeline_profile_section(const runtime::PipelineProfile &profile) {
  Object out;
  out.set("enabled", profile.enabled);
  out.set("measured_iterations", profile.measured_iterations);
  out.set("resident_weight_bytes", profile.resident_weight_bytes);
  out.set("resident_host_prefault_ms",
          profile.resident_host_prefault_milliseconds);
  out.set("resident_minor_page_faults", profile.resident_minor_page_faults);
  out.set("resident_major_page_faults", profile.resident_major_page_faults);
  out.set("resident_h2d_ms", profile.resident_h2d_milliseconds);
  out.set("resident_upload_ms", profile.resident_upload_milliseconds);
  out.set("streamed_weight_bytes", profile.streamed_weight_bytes);
  out.set("streamed_host_stage_ms", profile.streamed_host_stage_milliseconds);
  out.set("streamed_host_wait_ms", profile.streamed_host_wait_milliseconds);
  out.set("streamed_h2d_ms", profile.streamed_h2d_milliseconds);
  out.set("operation_kernel_ms", profile.operation_kernel_milliseconds);
  out.set("attention_kernel_ms", profile.attention_kernel_milliseconds);
  out.set("non_kernel_device_timeline_ms",
          profile.non_kernel_device_timeline_milliseconds);
  return out;
}

Array operation_timings_section(
    const std::vector<runtime::OperationTiming> &timings) {
  Array out;
  out.reserve(timings.size());
  for (const auto &timing : timings) {
    Object entry;
    entry.set("operation_id", timing.operation_id);
    entry.set("opcode", ir::opcode_name(timing.opcode));
    entry.set("mean_ms", timing.mean_milliseconds);
    entry.set("min_ms", timing.minimum_milliseconds);
    entry.set("max_ms", timing.maximum_milliseconds);
    if (!timing.plan.empty())
      entry.set("plan", timing.plan);
    out.push_back(std::move(entry));
  }
  return out;
}

Array trace_events_section(const std::vector<runtime::TraceEvent> &events) {
  Array out;
  out.reserve(events.size());
  for (const auto &event : events) {
    Object entry;
    entry.set("category", event.category);
    entry.set("name", event.name);
    entry.set("operation_id", event.operation_id);
    entry.set("opcode", event.opcode);
    entry.set("host_start_ms", event.host_start_ms);
    entry.set("host_end_ms", event.host_end_ms);
    entry.set("bytes", event.bytes);
    entry.set("stream", event.stream);
    out.push_back(std::move(entry));
  }
  return out;
}

Object trace_attribution_section(
    const std::vector<runtime::TraceEvent> &events) {
  struct Bucket {
    std::uint64_t count{};
    std::uint64_t bytes{};
    double host_ms{};
  };
  // Stable category order: the vocabulary order, then anything unexpected.
  const std::string_view ordered[] = {
      category::preparation,  category::operation,       category::region,
      category::gemm,         category::attention,       category::convolution,
      category::generated_kernel, category::h2d,         category::d2h,
      category::d2d,          category::staging,         category::wait,
      category::synchronization, category::layout,       category::allocation,
      category::filesystem,   category::stage};
  std::map<std::string, Bucket> buckets;
  for (const auto &event : events) {
    auto &bucket = buckets[event.category];
    ++bucket.count;
    bucket.bytes += event.bytes;
    bucket.host_ms += event.host_end_ms - event.host_start_ms;
  }
  Object out;
  const auto emit = [&](const std::string &name, const Bucket &bucket) {
    Object entry;
    entry.set("count", bucket.count);
    entry.set("bytes", bucket.bytes);
    entry.set("host_ms", bucket.host_ms);
    out.set(name, std::move(entry));
  };
  for (const auto name : ordered) {
    const auto found = buckets.find(std::string(name));
    if (found != buckets.end())
      emit(found->first, found->second);
  }
  for (const auto &[name, bucket] : buckets)
    if (!out.find(name))
      emit(name, bucket);
  return out;
}

std::string serialize_probe(const target::TargetProfile &profile,
                            const target::RuntimeBudget &budget) {
  auto document = make_document(kind::device_probe);
  document.set("hardware", hardware_section(profile));
  document.set("runtime_budget", runtime_budget_section(budget));
  return serialize(Value(std::move(document)));
}

Object runtime_trace_document(const runtime::RunResult &result,
                              std::string_view program_fingerprint,
                              std::uint64_t operation_count,
                              const runtime::RunOptions &options) {
  auto document = make_document(kind::runtime_trace);
  document.set("hardware", hardware_section(result.target_profile));
  document.set("runtime_budget", runtime_budget_section(result.runtime_budget));
  Object program;
  program.set("fingerprint", program_fingerprint);
  program.set("operations", operation_count);
  document.set("program", std::move(program));
  Object execution;
  execution.set("backend", result.backend_name);
  execution.set("device", result.device_name);
  execution.set("warmups", options.warmups);
  execution.set("iterations", options.iterations);
  execution.set("preparation_ms", result.preparation_milliseconds);
  execution.set("preparation_reported", result.preparation_reported);
  execution.set("mean_ms", result.mean_milliseconds);
  execution.set("min_ms", result.minimum_milliseconds);
  execution.set("max_ms", result.maximum_milliseconds);
  execution.set("resident_bytes", result.resident_bytes);
  execution.set("free_bytes_before", result.free_bytes_before);
  execution.set("free_bytes_after", result.free_bytes_after);
  execution.set("generated_source_hash", result.generated_source_hash);
  document.set("execution", std::move(execution));
  document.set("preparation_launch_telemetry",
               launch_telemetry_section(result.preparation_telemetry));
  document.set("run_launch_telemetry",
               launch_telemetry_section(result.run_telemetry));
  document.set("pipeline_profile",
               pipeline_profile_section(result.pipeline_profile));
  document.set("operation_timings",
               operation_timings_section(result.operation_timings));
  Object trace;
  trace.set("preparation_wall_ms", result.preparation_trace_milliseconds);
  trace.set("run_wall_ms", result.trace_milliseconds);
  trace.set("preparation_attribution",
            trace_attribution_section(result.preparation_trace_events));
  trace.set("run_attribution", trace_attribution_section(result.trace_events));
  trace.set("preparation_events",
            trace_events_section(result.preparation_trace_events));
  trace.set("run_events", trace_events_section(result.trace_events));
  document.set("trace", std::move(trace));
  return document;
}

} // namespace dif::telemetry
