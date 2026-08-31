#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"

#include <array>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace dif::runtime {

using TensorMap = std::unordered_map<std::uint32_t, Tensor>;

struct CutlassLinearChoice {
  std::uint32_t operation_id{};
  std::uint32_t schedule{};
};

struct LinearAlgorithmChoice {
  std::uint32_t operation_id{};
  std::uint32_t heuristic_rank{};
};

struct RunOptions {
  std::uint32_t warmups{2};
  std::uint32_t iterations{5};
  std::uint64_t minimum_free_bytes{256ULL * 1024ULL * 1024ULL};
  std::filesystem::path cache_directory;
  // Backend-only prepared execution choices. DiffIR semantics remain
  // unchanged; these ids identify exact Linear operations to benchmark or
  // exact Linear->SwiGlu chains to lower as a fused CUDA primitive.
  std::vector<std::uint32_t> tune_linear_operations;
  std::vector<std::uint32_t> fuse_linear_swiglu_operations;
  std::vector<CutlassLinearChoice> cutlass_linear_operations;
  std::vector<LinearAlgorithmChoice> linear_algorithm_choices;
  // Accepted MiniMax-H3 W8A8 precision route. The cache is the Serenity
  // resident row-scale SafeTensors store; the backend recognizes and replaces
  // only an exclusive Linear->SwiGlu->Linear->ResidualGate MLP chain.
  std::filesystem::path h3_w8a8_cache;
  std::uint32_t h3_w8a8_layer{};
  // Keep a strict W8A8 block prefix resident and refill one reusable block
  // store for the tail, matching Serenity's production low-memory policy.
  std::uint32_t h3_w8a8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  // Accepted MiniMax-H3 groupwise INT8 weight-only route. The Serenity cache
  // stores transformed I8 projection weights with compact F16 group scales;
  // the backend dequantizes each projection into shared BF16 scratch below
  // the unchanged DiffIR semantic boundary.
  std::filesystem::path h3_groupwise_cache;
  std::uint32_t h3_groupwise_layer{};
  // Serenity's accepted prepared AdaLN path. The cache contains the BF16
  // modulation result for each block; the companion DiffTensor is the exact
  // activated BF16 timestep input used to build it and is checked byte-for-
  // byte on every prepared run.
  std::filesystem::path h3_modulation_cache;
  std::filesystem::path h3_modulation_input;
  std::filesystem::path h3_modulation_source_index;
  std::uint32_t h3_modulation_layer{};
  std::uint32_t h3_modulation_steps{};
  std::uint32_t h3_modulation_slice{};
  // Diagnostic subgraphs can contain only a prefix of H3 blocks while using
  // the source schedule cache. A nonzero value validates cache provenance
  // against the complete checkpoint block count, not the sliced graph count.
  std::uint32_t h3_modulation_total_layers{};
  // Accepted architecture-tagged Comfy Kitchen H3 attention route. This is an
  // explicit approximate backend under the semantic Attention operation; an
  // empty path retains the exact DiffIR-selected implementation.
  std::filesystem::path h3_ck_attention_dso;
  std::uint32_t linear_tuning_warmups{3};
  std::uint32_t linear_tuning_iterations{10};
  std::uint32_t linear_tuning_sessions{3};
  bool expand_linear_algorithms{false};
  bool trace_operations{false};
  bool overlap_streaming{true};
  bool profile_pipeline{false};
  // W1-R streamed-transport policy. Default preserves the historical
  // behavior exactly: mapped streamed constants madvise(MADV_DONTNEED)
  // their source pages after every staging copy, which forces a refault
  // on every warmup and iteration. When false, pages are kept across the
  // passes of one run and dropped once at run end instead. Outputs are
  // byte-identical either way; only host paging behavior changes. Any
  // non-default choice must enter difopt candidate identity.
  bool streamed_release_mapped_pages_per_copy{true};
};

struct OperationTiming {
  std::uint32_t operation_id{};
  ir::Opcode opcode{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
};

struct LinearAlgorithmTiming {
  std::uint32_t heuristic_index{};
  std::int32_t algorithm_id{-1};
  std::uint32_t tile_id{};
  std::uint32_t stages_id{};
  std::int32_t split_k{};
  std::uint32_t reduction_scheme{};
  std::uint32_t cta_swizzle{};
  std::uint32_t custom_option{};
  double waves_count{};
  std::uint64_t workspace_bytes{};
  double mean_milliseconds{};
  double minimum_session_milliseconds{};
  double maximum_session_milliseconds{};
};

struct LinearTuningResult {
  std::uint32_t operation_id{};
  std::uint32_t selected_heuristic_index{};
  std::int32_t selected_algorithm_id{-1};
  double default_mean_milliseconds{};
  double selected_mean_milliseconds{};
  double observed_noise_milliseconds{};
  double tuning_milliseconds{};
  bool changed_from_default{};
  std::string decision;
  std::vector<LinearAlgorithmTiming> candidates;
};

struct PrimitiveFusionResult {
  std::uint32_t linear_operation_id{};
  std::uint32_t swiglu_operation_id{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::string implementation;
};

struct GemmPrimitiveResult {
  std::uint32_t operation_id{};
  std::uint32_t schedule{};
  std::string implementation;
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

struct H3W8A8MlpResult {
  std::uint32_t fc1_operation_id{};
  std::uint32_t swiglu_operation_id{};
  std::uint32_t fc2_operation_id{};
  std::uint32_t residual_operation_id{};
  std::uint32_t layer{};
  std::uint32_t chunk_rows{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::string classification;
  std::string implementation;
  std::string cache_path;
  bool resident{};
};

struct H3W8A8AttentionResult {
  std::uint32_t qkv_layout_operation_id{};
  std::array<std::uint32_t, 3> qkv_linear_operation_ids{};
  std::uint32_t output_linear_operation_id{};
  std::uint32_t residual_operation_id{};
  std::uint32_t layer{};
  std::uint32_t chunk_rows{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::string classification;
  std::string implementation;
  std::string cache_path;
  bool resident{};
};

struct H3CKAttentionResult {
  std::uint32_t operation_id{};
  std::uint32_t target_sm{};
  std::uint64_t scratch_bytes{};
  std::string classification;
  std::string implementation;
  std::string dso_path;
};

struct H3GroupwiseInt8Result {
  std::uint32_t qkv_layout_operation_id{};
  std::uint32_t output_linear_operation_id{};
  std::uint32_t fc1_operation_id{};
  std::uint32_t fc2_operation_id{};
  std::uint32_t layer{};
  std::array<std::uint32_t, 4> group_sizes{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t scratch_bytes{};
  std::string classification;
  std::string implementation;
  std::string cache_path;
};

struct H3ModulationCacheResult {
  std::uint32_t linear_operation_id{};
  std::uint32_t select_operation_id{};
  std::uint32_t layer{};
  std::uint64_t cache_bytes{};
  std::uint64_t replaced_weight_bytes{};
  std::string classification;
  std::string implementation;
  std::string cache_path;
  std::string input_path;
};

// Dispatch and transfer counters for one execution phase of the CUDA
// backend. Counters are incremented at the centralized submission sites
// (kernel launches, library GEMM/attention dispatches, every cuMemcpy*, and
// the host-blocking synchronization calls), so they census exactly what the
// runtime submitted -- including profiling-only event records when
// profile_pipeline is enabled. They are always collected; the cost is a
// host-side increment per call.
struct LaunchTelemetry {
  std::uint64_t kernel_launches{};
  std::uint64_t cublaslt_matmuls{};
  std::uint64_t cublas_gemms{};
  std::uint64_t cudnn_attention_dispatches{};
  std::uint64_t cutlass_launches{};
  std::uint64_t ck_attention_dispatches{};
  std::uint64_t h2d_copies{};
  std::uint64_t h2d_bytes{};
  std::uint64_t d2h_copies{};
  std::uint64_t d2h_bytes{};
  std::uint64_t event_records{};
  std::uint64_t stream_wait_events{};
  std::uint64_t host_event_synchronizes{};
  std::uint64_t host_stream_synchronizes{};
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
  // Profile-only resident preparation is split into an explicit mapped-page
  // prefault followed by the upload. The latter is wall time after prefaulting
  // and therefore includes driver pageable-memory staging plus H2D, but not
  // checkpoint page faults.
  double resident_host_prefault_milliseconds{};
  std::uint64_t resident_minor_page_faults{};
  std::uint64_t resident_major_page_faults{};
  double resident_h2d_milliseconds{};
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
  std::vector<LinearTuningResult> linear_tuning_results;
  std::vector<LinearAlgorithmChoice> selected_linear_algorithms;
  std::vector<PrimitiveFusionResult> primitive_fusions;
  std::vector<GemmPrimitiveResult> gemm_primitives;
  std::vector<H3W8A8MlpResult> h3_w8a8_mlps;
  std::vector<H3W8A8AttentionResult> h3_w8a8_attentions;
  std::vector<H3CKAttentionResult> h3_ck_attentions;
  std::vector<H3GroupwiseInt8Result> h3_groupwise_int8;
  std::vector<H3ModulationCacheResult> h3_modulation_caches;
  PipelineProfile pipeline_profile;
  // CUDA backend only; other executors leave these zeroed. The preparation
  // phase counters describe the one-time prepare() of the executing plan;
  // the run counters describe exactly this run() call (warmups, timed
  // iterations, input upload, and output readback included).
  LaunchTelemetry preparation_telemetry;
  LaunchTelemetry run_telemetry;
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
