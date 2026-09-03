#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/target/profile.hpp"

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
  // Explicit compiler-selected groups of independent unbiased Linear
  // operations that share one activation. The CUDA backend may dispatch each
  // group across separate compute streams with separately accounted
  // workspaces; operation math and DiffIR values are unchanged.
  std::vector<std::vector<std::uint32_t>> parallel_linear_groups;
  std::vector<std::uint32_t> fuse_linear_swiglu_operations;
  // Absorb an unbiased Linear's exclusive, immediately-adjacent BiasAdd into
  // the cuBLASLt bias epilogue: one library launch, no materialized
  // intermediate. Ids name the Linear operations (explicit = candidate
  // identity; malformed patterns fail closed). The absorbed launch follows
  // the biased-Linear plan form, whose layouts differ from the unbiased
  // plan, so byte-identity against the separate BiasAdd kernel is
  // program-dependent and must be gated (difopt acceptance discipline)
  // before a candidate is accepted.
  std::vector<std::uint32_t> absorb_linear_bias_operations;
  std::vector<CutlassLinearChoice> cutlass_linear_operations;
  std::vector<LinearAlgorithmChoice> linear_algorithm_choices;
  // Require a cuBLASLt algorithm with no cross-CTA split-K reduction for
  // ordinary Linear operations. The runtime fails closed when heuristics do
  // not expose one; this is compiler execution policy, not a hidden fallback.
  bool deterministic_linear_algorithms{false};
  // Explicit compiler-selected streamed constants to promote into persistent
  // device storage for the lifetime of this prepared execution. The runtime
  // validates and executes this list but never chooses it heuristically.
  std::vector<std::uint32_t> resident_streamed_constants;
  // Generic compiler-selected ConvRot INT8 precision policy for ordinary
  // BF16/F16 DiffIR Linear operations. The cache is sealed to the exact DiffIR
  // fingerprint and stores one rotated I8 weight plus F32 output-channel
  // scales per selected semantic weight tensor.  The runtime supplies the
  // matching dynamic per-row activation transform and executes the same
  // Linear result shape through the shared NVIDIA backend; no model-specific
  // opcode or executor is introduced.
  std::filesystem::path convrot_int8_checkpoint;
  // Explicit prefix of eligible semantic Linear operations to lower through
  // the generic ConvRot cache. Zero means every eligible Linear. This is
  // compiler precision policy: later operations remain on their ordinary
  // exact source-dtype lowering and continue to consume the original bundle
  // weights.
  std::uint32_t convrot_int8_linear_count{};
  // Keep every selected generic ConvRot weight and scale in prepared device
  // storage. The original semantic weights are excluded from the memory plan,
  // so this is the repeated-execution route for decoder-style programs; the
  // default two-slot route remains available for larger streamed models.
  bool convrot_int8_resident{false};
  // Higher-fidelity generic ConvRot lowering: keep the rotated checkpoint in
  // INT8 storage, but rotate activations and dequantize the current weight in
  // BF16 before the ordinary cuBLASLt Linear. This removes activation
  // quantization while retaining the half-size checkpoint lifecycle.
  bool convrot_int8_weight_only_quality{false};
  // Explicit compiler-selected semantic Reshape operations whose internal
  // outputs alias their immutable inputs in the prepared execution plan.
  std::vector<std::uint32_t> alias_reshape_operations;
  // Compiler-proven pure operations whose external inputs remain stable over
  // repeated executions of one prepared plan. The CUDA runtime executes the
  // region once, preserves every crossing output in dedicated device storage,
  // and reuses it only while the declared input bytes remain unchanged. This
  // is generic cross-evaluation execution policy, not model-specific runtime
  // semantics; invalid regions fail closed during preparation.
  std::vector<std::uint32_t> repeated_invariant_operations;
  // Diagnostic-only tensor boundaries to copy immediately after their
  // semantic producer during a single measured execution. This is explicit
  // parity-harness policy and is never enabled by a production plan because
  // the readback intentionally perturbs timing.
  std::vector<std::uint32_t> capture_intermediate_tensors;
  // Exact cuDNN attention engine heuristic selected by the compiler plan:
  // 0=A (default), 1=B, 2=FALLBACK. This changes only plan discovery; the
  // semantic Attention operation, dtype, accumulation, and outputs remain
  // source-faithful and every non-default candidate must pass parity/timing.
  std::uint32_t cudnn_attention_heuristic{};
  // Accepted MiniMax-H3 W8A8 precision route. The cache is the Serenity
  // resident row-scale SafeTensors store; the backend recognizes and replaces
  // only an exclusive Linear->SwiGlu->Linear->ResidualGate MLP chain.
  std::filesystem::path h3_w8a8_cache;
  std::uint32_t h3_w8a8_layer{};
  // Keep a strict W8A8 block prefix resident and refill one reusable block
  // store for the tail, matching Serenity's production low-memory policy.
  std::uint32_t h3_w8a8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  // Project-owned ConvRot INT8 cache derived from the official H3 checkpoint.
  // Projection weights are rotated offline in normalized 256-wide Hadamard
  // groups; matching activations are rotated online before dynamic per-row
  // INT8 quantization. This is an explicit approximate precision policy over
  // the unchanged H3 DiffIR Linear semantics, not a model executor or new IR
  // op. No ComfyUI checkpoint weights participate in this route.
  std::filesystem::path h3_convrot_int8_checkpoint;
  std::uint32_t h3_convrot_int8_layer{};
  // Explicit compiler precision policy: only this many consecutive H3 blocks
  // use ConvRot INT8. Remaining blocks retain ordinary BF16 DiffIR lowering.
  // This is independent from residency, which controls where admitted
  // ConvRot weights live without changing their numerical path.
  std::uint32_t h3_convrot_int8_attention_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_convrot_int8_mlp_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t h3_convrot_int8_resident_layers{
      std::numeric_limits<std::uint32_t>::max()};
  // Explicit compiler-selected row tile for direct H3 INT8 MLP projections.
  // Larger tiles trade reusable scratch capacity for fewer, fuller GEMMs. The
  // choice is prepared once, reported in the result, and must be replayed by
  // an admitted execution plan; it does not alter DiffIR semantics.
  std::uint32_t h3_int8_mlp_chunk_rows{1024U};
  // Use shape-prepared cuBLASLt IMMA plans for direct H3 INT8 projections
  // instead of the legacy cublasGemmEx dispatcher. Off by default until the
  // complete-model numerical, memory, and timing candidate is admitted.
  bool h3_int8_cublaslt{false};
  std::uint32_t h3_int8_cublaslt_heuristic_rank{};
  bool h3_int8_cublaslt_tune{false};
  // Candidate ConvRot path: fuse FC1's dynamic row/static channel scaling
  // into an INT8 tensor-core GEMM epilogue and preserve the observable BF16
  // projection boundary without materializing an I32 output tensor. Off by
  // default until whole-step parity and timing admit it.
  bool h3_int8_cutlass_scaled_fc1{false};
  // Candidate ConvRot path: extend the same source-observable scaled BF16
  // CUTLASS epilogue to QKV, attention output, FC1, and FC2. This removes all
  // reusable I32 projection scratch from the prepared 50-block executor while
  // retaining the exact cuDNN attention route. Off by default until complete
  // trajectory and decoded-quality gates admit it.
  bool h3_int8_cutlass_scaled_all{false};
  // Higher-fidelity ConvRot activation quantization policy. Zero preserves
  // the historical one-scale-per-row contract. A positive multiple of the
  // 256-wide rotation group computes independent dynamic activation scales
  // for each K chunk, evaluates the same INT8 dot product chunk-by-chunk, and
  // accumulates the scaled partials in F32 before the observable BF16
  // projection boundary. The official ConvRot checkpoint and its static
  // per-output-channel weight scales are unchanged.
  std::uint32_t h3_int8_convrot_scale_chunk{};
  // Diagnostic separation of static grouped-weight scaling from dynamic
  // activation scaling. When true, grouped ConvRot weights retain their
  // configured K chunks while activations use the original global row scale.
  bool h3_int8_convrot_global_activation_scale{false};
  // Precision-correction rows for the H3 ConvRot projection chains. The
  // selected sequence rows are recomputed from the same resident rotated I8
  // weights with BF16 activations/dequantization and overwrite the approximate
  // row results before downstream semantics consume them. This is explicit
  // compiler precision policy; the runtime never guesses modality rows.
  std::vector<std::uint32_t> h3_convrot_bf16_correction_rows;
  // Candidate prepared lowering for ConvRot H3 blocks: consume compact
  // creator AdaLN tables directly inside RMSNorm/ConvRot encode and residual
  // epilogues. This removes the six sequence-expanded modulation tensors and
  // two normalized activation tensors per block without changing DiffIR
  // semantics. Off by default until exact boundary and decoded quality gates
  // admit it.
  bool h3_int8_compact_adaln{false};
  // Accepted MiniMax-H3 groupwise INT8 weight-only route. The Serenity cache
  // stores transformed I8 projection weights with compact F16 group scales;
  // the backend dequantizes each projection into shared BF16 scratch below
  // the unchanged DiffIR semantic boundary.
  std::filesystem::path h3_groupwise_cache;
  std::uint32_t h3_groupwise_layer{};
  std::uint32_t h3_groupwise_layers{
      std::numeric_limits<std::uint32_t>::max()};
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
  // Explicit architecture-tagged H3 attention DSO route. Project-owned dense
  // INT8 and legacy development-oracle ABIs are identified separately in the
  // run receipt; an empty path retains the exact DiffIR-selected
  // implementation. A production admission must use project-owned code.
  std::filesystem::path h3_ck_attention_dso;
  // Contiguous transformer-block Attention range replaced in program order by
  // the selected H3 INT8 DSO. Operations before and after the range remain on
  // exact cuDNN. This is explicit compiler policy for quality/performance
  // sweeps and is reported through the admitted primitive list.
  std::uint32_t h3_int8_attention_first_layer{};
  std::uint32_t h3_int8_attention_layers{
      std::numeric_limits<std::uint32_t>::max()};
  std::uint32_t linear_tuning_warmups{3};
  std::uint32_t linear_tuning_iterations{10};
  std::uint32_t linear_tuning_sessions{3};
  bool expand_linear_algorithms{false};
  // Persist cuBLASLt Linear algorithm selections under cache_directory
  // (PTX-cache style), keyed on the full problem identity (shape, storage
  // and compute types, bias form, preference workspace, cuBLASLt version,
  // device arch) with distinct namespaces for passively selected and tuned
  // algorithms. Restores are validated through cublasLtMatmulAlgoCheck and
  // fail open to fresh heuristics. Default OFF; enabling is an execution-
  // policy choice and enters difopt candidate identity.
  bool persist_linear_heuristics{false};
  bool trace_operations{false};
  bool overlap_streaming{true};
  bool profile_pipeline{false};
  // Collect attributed TraceEvent records at the centralized submission
  // sites (launches, library dispatches, copies, waits, staging) for this
  // run. Host-side timestamps only; never alters what is submitted. The
  // DIF_TRACE_FILE environment variable enables the same collection for any
  // tool and appends one runtime-trace document per run() to that file.
  bool trace_events{false};
  // Push an NVTX range per semantic operation and per phase so Nsight
  // Systems can correlate kernels with DiffIR operations. Requires the NVTX
  // headers at build time; silently unavailable otherwise.
  bool nvtx_ranges{false};
  // W1-R streamed-transport policy. Default preserves the historical
  // behavior exactly: mapped streamed constants madvise(MADV_DONTNEED)
  // their source ranges after every staging copy. This drops process page
  // tables without intentionally evicting the shared file-cache pages; a
  // later pass may incur minor refaults but not a deliberate disk reread.
  // When false, mappings remain populated through the run. Outputs are byte-
  // identical either way; only host paging behavior changes. Any non-default
  // choice must enter difopt candidate identity.
  bool streamed_release_mapped_pages_per_copy{true};
  // Keep mapped streamed-weight pages populated across repeated run() calls
  // on the same PreparedExecution. This is the host-resident checkpoint mode
  // for iterative samplers: the file mapping and its page tables are reused
  // for every denoise step instead of being discarded at copy or run end.
  // It is mutually exclusive with release-per-copy and increases host RSS by
  // up to the streamed checkpoint payload. Default OFF preserves historical
  // behavior; enabling it is an explicit whole-system plan choice.
  bool streamed_keep_mapped_pages_between_runs{false};
  // Pinned staging ring for streamed constants and how many operations
  // ahead the overlapped scheduler prefetches. Defaults (2 buffers,
  // depth 1) are the historical double-buffer policy, byte-for-byte.
  // Depth is fixed at prepare time: the memory plan widens every streamed
  // interval by the same distance, which is what makes a deeper prefetch
  // hazard-free. Buffers must be >= 2 and >= depth + 1.
  std::uint32_t streamed_staging_buffers{2};
  std::uint32_t streamed_prefetch_depth{1};
  // Worker threads for the mmap->pinned staging copy (1 = the historical
  // single-threaded memcpy on the submitting thread).
  std::uint32_t streamed_stage_threads{1};
  // Upload compiler-promoted reusable constants and explicit resident H3 INT8
  // plan weights through a bounded two-slot pinned staging pipeline during
  // preparation. This overlaps mmap-to-pinned host copies with the previous
  // H2D transfer; policy remains explicit and the pinned footprint is checked
  // against streamed_pinned_budget_bytes.
  bool pipelined_resident_upload{false};
  // Allocate selected persistent constants at prepare, but populate their
  // dedicated device storage at first semantic use. The first execution can
  // overlap this one-time upload with model compute; later executions reuse
  // the populated storage without another transfer.
  bool lazy_resident_upload{false};
  // Host file-cache policy for weights that became GPU resident. Default
  // (true) preserves the historical behavior: after a resident upload the
  // mapped range is released with posix_fadvise(DONTNEED), so every fresh
  // process rereads those bytes from storage. False keeps the clean,
  // reclaimable file-cache pages (the mapping itself is still dropped), so
  // a later process on the same host stages them from cache. Outputs are
  // byte-identical either way; only host paging changes. The runtime fails
  // closed to eviction when the process' cgroup memory limit cannot hold the
  // resident bytes (RESIDENT_HOST_PAGES diagnostic on stderr). Any
  // non-default choice must enter difopt candidate identity.
  bool resident_evict_host_pages{true};
  // Upper bound on pinned host memory the streamed staging ring may
  // allocate. The historical two-buffer footprint is always admitted; a
  // larger ring that would exceed the budget fails closed. This host has
  // 62 GiB and a documented host-OOM incident: keep this modest.
  std::uint64_t streamed_pinned_budget_bytes{2ULL * 1024ULL * 1024ULL *
                                             1024ULL};
  // Route the reusable W8A8 tail weight uploads over the copy stream with
  // event fences instead of the compute stream (default false = historical
  // compute-stream uploads).
  bool h3_w8a8_tail_uploads_on_copy_stream{false};
  // Stage dynamic input uploads and output readbacks through a pinned
  // bounce buffer sized at prepare (default false = historical pageable
  // transfers). Must be requested at prepare time; counted against
  // streamed_pinned_budget_bytes.
  bool pinned_io_staging{false};
};

struct OperationTiming {
  std::uint32_t operation_id{};
  ir::Opcode opcode{};
  double mean_milliseconds{};
  double minimum_milliseconds{};
  double maximum_milliseconds{};
  // Non-empty when the backend executed a fused plan at this operation's
  // slot instead of the operation's own opcode (for example the H3 INT8 QKV
  // projection launched where the QKV weight layout op sits). Consumers
  // must not classify such a timing by opcode alone.
  std::string plan;
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

struct LinearBiasFusionResult {
  std::uint32_t linear_operation_id{};
  std::uint32_t bias_operation_id{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::string implementation;
};

struct ConvRotInt8LinearResult {
  std::uint32_t operation_id{};
  std::uint32_t weight_tensor_id{};
  std::uint64_t rows{};
  std::uint64_t output_columns{};
  std::uint64_t contraction{};
  std::uint64_t quantized_weight_bytes{};
  std::string classification;
  std::string implementation;
  std::string cache_path;
};

// Counters for the persisted cuBLASLt Linear algorithm store: how many plans
// restored a validated cached selection, how many cached entries failed
// AlgoCheck and fell open to fresh heuristics, and how many selections were
// written (passive namespace at plan build, tuned namespace after tuning).
struct LinearHeuristicCacheStats {
  std::uint64_t restored{};
  std::uint64_t rejected{};
  std::uint64_t saved_passive{};
  std::uint64_t saved_tuned{};
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
  std::uint64_t cudnn_convolution_dispatches{};
  std::uint64_t cutlass_launches{};
  std::uint64_t ck_attention_dispatches{};
  std::uint64_t h2d_copies{};
  std::uint64_t h2d_bytes{};
  std::uint64_t d2h_copies{};
  std::uint64_t d2h_bytes{};
  std::uint64_t d2d_copies{};
  std::uint64_t d2d_bytes{};
  std::uint64_t event_records{};
  std::uint64_t stream_wait_events{};
  std::uint64_t host_event_synchronizes{};
  std::uint64_t host_stream_synchronizes{};
  std::uint64_t device_mem_allocs{};
  std::uint64_t pinned_mem_allocs{};
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

// One attributed runtime event recorded at a centralized submission site.
// Host timestamps are milliseconds since the start of the run() (or
// prepare()) call that produced the event; a submission that does not block
// the host has host_end_ms equal to host_start_ms. Category names come from
// dif/telemetry/vocabulary.hpp.
struct TraceEvent {
  std::string category;
  std::string name;
  // Semantic operation being executed when the event was submitted; zero
  // when the event belongs to no operation (input upload, output readback,
  // preparation).
  std::uint32_t operation_id{};
  std::string opcode;
  double host_start_ms{};
  double host_end_ms{};
  std::uint64_t bytes{};
  std::string stream;
};

struct RunResult {
  target::TargetProfile target_profile;
  target::RuntimeBudget runtime_budget;
  TensorMap outputs;
  TensorMap captured_intermediates;
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
  std::vector<LinearBiasFusionResult> linear_bias_fusions;
  std::vector<ConvRotInt8LinearResult> convrot_int8_linears;
  LinearHeuristicCacheStats linear_heuristic_cache;
  std::vector<GemmPrimitiveResult> gemm_primitives;
  std::vector<H3W8A8MlpResult> h3_w8a8_mlps;
  std::vector<H3W8A8AttentionResult> h3_w8a8_attentions;
  std::vector<H3CKAttentionResult> h3_ck_attentions;
  std::vector<H3GroupwiseInt8Result> h3_groupwise_int8;
  std::vector<H3ModulationCacheResult> h3_modulation_caches;
  std::uint32_t repeated_invariant_operation_count{};
  std::uint64_t repeated_invariant_persistent_bytes{};
  bool repeated_invariant_cache_hit{};
  PipelineProfile pipeline_profile;
  // CUDA backend only; other executors leave these zeroed. The preparation
  // phase counters describe the one-time prepare() of the executing plan;
  // the run counters describe exactly this run() call (warmups, timed
  // iterations, input upload, and output readback included).
  LaunchTelemetry preparation_telemetry;
  LaunchTelemetry run_telemetry;
  // Populated only when RunOptions::trace_events (or DIF_TRACE_FILE) is set.
  std::vector<TraceEvent> preparation_trace_events;
  std::vector<TraceEvent> trace_events;
  double preparation_trace_milliseconds{};
  double trace_milliseconds{};
  // True for the first run() result after prepare(); later results of the
  // same prepared execution describe the same one-time preparation, so
  // report consumers must count preparation once. CPU executors leave this
  // true on every result.
  bool preparation_reported{true};
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
