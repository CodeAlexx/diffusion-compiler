#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dif::runtime {

using TensorMap = std::unordered_map<std::uint32_t, Tensor>;

struct RunOptions {
  std::uint32_t warmups{2};
  std::uint32_t iterations{5};
  std::uint64_t minimum_free_bytes{256ULL * 1024ULL * 1024ULL};
  std::filesystem::path cache_directory;
  bool trace_operations{false};
  bool overlap_streaming{true};
  bool profile_pipeline{false};
};

struct OperationTiming {
  std::uint32_t operation_id{};
  ir::Opcode opcode{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
};

// Profiling values describe the timed iterations of one prepared CUDA run.
// Host staging is the memcpy from mapped/owned constants into pinned memory;
// for mapped checkpoints it includes any page-fault and storage wait incurred
// by that access. H2D time is the sum of CUDA copy-engine event durations and
// may overlap kernels. Non-kernel device timeline is the measured run event
// minus the sum of per-operation compute-stream events; it includes transfer
// dependencies and host submission gaps on the critical path.
struct PipelineProfile {
  bool enabled{};
  std::uint32_t measured_iterations{};
  std::uint64_t resident_weight_bytes{};
  double resident_upload_milliseconds{};
  std::uint64_t streamed_weight_bytes{};
  double streamed_host_stage_milliseconds{};
  double streamed_host_wait_milliseconds{};
  double streamed_h2d_milliseconds{};
  double operation_kernel_milliseconds{};
  double attention_kernel_milliseconds{};
  double non_kernel_device_timeline_milliseconds{};
};

struct RunResult {
  TensorMap outputs;
  double preparation_milliseconds{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
  std::uint64_t free_bytes_before{};
  std::uint64_t free_bytes_after{};
  std::string device_name;
  std::string backend_name;
  std::string generated_source_hash;
  std::uint64_t resident_bytes{};
  std::vector<OperationTiming> operation_timings;
  PipelineProfile pipeline_profile;
};

class PreparedExecution {
public:
  virtual ~PreparedExecution() = default;
  virtual RunResult run(const TensorMap &inputs, const RunOptions &options) = 0;
  virtual std::string name() const = 0;
  virtual double preparation_milliseconds() const { return 0.0; }
  virtual std::uint64_t resident_bytes() const { return 0U; }
};

class Executor {
public:
  virtual ~Executor() = default;
  virtual std::unique_ptr<PreparedExecution>
  prepare(const ir::Program &program, const TensorMap &bindings,
          const RunOptions &options) = 0;
  RunResult run(const ir::Program &program, const TensorMap &inputs,
                const RunOptions &options);
  virtual std::string name() const = 0;
};

std::unique_ptr<Executor> make_cpu_executor();
std::unique_ptr<Executor> make_cuda_executor(int device = 0);
bool cuda_available();

} // namespace dif::runtime
