#include "dif/runtime/executor.hpp"

#include "dif/compiler/compiler.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#if DIF_HAS_CUDNN
#include "dif/runtime/cudnn_attention.hpp"
#include "dif/runtime/cudnn_conv.hpp"
#endif
#if DIF_HAS_CUTLASS
#include "dif/runtime/cutlass_gemm.hpp"
#include "dif/runtime/device_probe.hpp"
#endif
#if DIF_HAS_H3_OWNED_ATTENTION
#include "dif/runtime/h3_owned_attention.hpp"
#endif
#if DIF_HAS_FLASH_ATTENTION
#include "dif/runtime/flash_attention.hpp"
#endif
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/telemetry/trace_sink.hpp"
#include "dif/telemetry/vocabulary.hpp"
#include "dif/weights/safetensors.hpp"
#if DIF_HAS_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>

// cuBLASLt block-scaled (MXFP8) matmul descriptors exist from cuBLAS 12.8.
// Older toolkits compile without the plan; a program that carries
// LinearFp8BlockScaled then fails closed at prepare with the library facts.
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 120800
#define DIF_HAS_CUBLASLT_BLOCK_SCALE 1
#else
#define DIF_HAS_CUBLASLT_BLOCK_SCALE 0
#endif
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>
#include <unistd.h>

namespace dif::runtime {
namespace {

void check(CUresult result, const char *action) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  const char *description = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &description);
  fail(std::string(action) + ": " + (name ? name : "CUDA_ERROR") + " (" +
       (description ? description : "no description") + ")");
}

void check(nvrtcResult result, const char *action) {
  if (result != NVRTC_SUCCESS)
    fail(std::string(action) + ": " + nvrtcGetErrorString(result));
}

void check(cublasStatus_t result, const char *action) {
  if (result == CUBLAS_STATUS_SUCCESS)
    return;
  fail(std::string(action) + ": " + cublasGetStatusString(result));
}

// --- Launch/transfer telemetry ---------------------------------------------
// One accumulator is active per execution phase (preparation, or one run).
// CUDA work submission in this backend is single-threaded, so a plain
// pointer suffices. Every wrapper forwards unchanged to the driver/library
// call it counts; telemetry can never alter execution.
LaunchTelemetry *active_telemetry = nullptr;

class TelemetryScope {
public:
  explicit TelemetryScope(LaunchTelemetry &telemetry)
      : previous_(active_telemetry) {
    active_telemetry = &telemetry;
  }
  ~TelemetryScope() { active_telemetry = previous_; }
  TelemetryScope(const TelemetryScope &) = delete;
  TelemetryScope &operator=(const TelemetryScope &) = delete;

private:
  LaunchTelemetry *previous_;
};

// --- Attributed trace events ----------------------------------------------
// Sibling of the launch counters. When a Tracer is active the same wrappers
// also append a TraceEvent naming the semantic operation that submitted the
// work, so an agent can attribute GEMM, attention, generated-kernel, copy,
// staging, wait, and synchronization submissions to DiffIR operations.
// Collection is host-side bookkeeping only; it never changes what is
// submitted or in which order. NVTX ranges use the same operation scope so
// Nsight Systems can correlate device kernels with DiffIR operations.
bool nvtx_enabled = false;

void nvtx_push(const char *name) {
#if DIF_HAS_NVTX
  if (nvtx_enabled)
    nvtxRangePushA(name);
#else
  (void)name;
#endif
}

void nvtx_pop() {
#if DIF_HAS_NVTX
  if (nvtx_enabled)
    nvtxRangePop();
#endif
}

bool layout_opcode(ir::Opcode opcode) {
  switch (opcode) {
  case ir::Opcode::Reshape:
  case ir::Opcode::BroadcastTo:
  case ir::Opcode::Slice:
  case ir::Opcode::Permute:
  case ir::Opcode::Concat:
  case ir::Opcode::Patchify3D:
  case ir::Opcode::Unpatchify3D:
  case ir::Opcode::H3DeinterleaveQkv:
  case ir::Opcode::H3DeinterleaveQkvWeight:
  case ir::Opcode::SelectRowChunks:
  case ir::Opcode::GatherRows:
  case ir::Opcode::IndexedUpdateRows:
  case ir::Opcode::PadConstant:
  case ir::Opcode::PadReflect:
    return true;
  default:
    return false;
  }
}

struct Tracer {
  std::vector<TraceEvent> events;
  std::chrono::steady_clock::time_point origin{
      std::chrono::steady_clock::now()};
  bool in_operation{};
  std::uint32_t operation_id{};
  ir::Opcode opcode{};
  // Context for submissions outside an operation or for naming a specific
  // mechanism inside one (streamed staging, resident upload, readback).
  const char *label{nullptr};

  double now_ms() const {
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - origin)
        .count();
  }

  void record(std::string_view category, std::string_view api,
              std::uint64_t bytes, std::string_view stream, double start_ms,
              double end_ms) {
    TraceEvent event;
    if (category == telemetry::category::generated_kernel && in_operation &&
        layout_opcode(opcode))
      category = telemetry::category::layout;
    event.category = std::string(category);
    event.name = label ? std::string(label) + ":" + std::string(api)
                       : std::string(api);
    event.operation_id = in_operation ? operation_id : 0U;
    if (in_operation)
      event.opcode = std::string(ir::opcode_name(opcode));
    event.host_start_ms = start_ms;
    event.host_end_ms = end_ms;
    event.bytes = bytes;
    event.stream = std::string(stream);
    events.push_back(std::move(event));
  }
};

Tracer *active_tracer = nullptr;

class TracerScope {
public:
  explicit TracerScope(Tracer &tracer) : previous_(active_tracer) {
    active_tracer = &tracer;
  }
  ~TracerScope() { active_tracer = previous_; }
  TracerScope(const TracerScope &) = delete;
  TracerScope &operator=(const TracerScope &) = delete;

private:
  Tracer *previous_;
};

class TraceLabelScope {
public:
  explicit TraceLabelScope(const char *label) {
    if (active_tracer) {
      previous_ = active_tracer->label;
      active_tracer->label = label;
    }
  }
  ~TraceLabelScope() {
    if (active_tracer)
      active_tracer->label = previous_;
  }
  TraceLabelScope(const TraceLabelScope &) = delete;
  TraceLabelScope &operator=(const TraceLabelScope &) = delete;

private:
  const char *previous_{nullptr};
};

// Attributes every submission inside a semantic operation to that operation
// and records the operation's own host submission span.
class TraceOperationScope {
public:
  explicit TraceOperationScope(const ir::Operation &operation)
      : active_(active_tracer != nullptr) {
    if (active_) {
      previous_in_operation_ = active_tracer->in_operation;
      previous_id_ = active_tracer->operation_id;
      previous_opcode_ = active_tracer->opcode;
      active_tracer->in_operation = true;
      active_tracer->operation_id = operation.id;
      active_tracer->opcode = operation.opcode;
      start_ms_ = active_tracer->now_ms();
    }
    if (nvtx_enabled) {
      name_ = "op" + std::to_string(operation.id) + " " +
              std::string(ir::opcode_name(operation.opcode));
      nvtx_push(name_.c_str());
    }
  }
  ~TraceOperationScope() {
    if (nvtx_enabled)
      nvtx_pop();
    if (active_) {
      active_tracer->record(telemetry::category::operation, "submit", 0U,
                            "compute", start_ms_, active_tracer->now_ms());
      active_tracer->in_operation = previous_in_operation_;
      active_tracer->operation_id = previous_id_;
      active_tracer->opcode = previous_opcode_;
    }
  }
  TraceOperationScope(const TraceOperationScope &) = delete;
  TraceOperationScope &operator=(const TraceOperationScope &) = delete;

private:
  bool active_;
  bool previous_in_operation_{};
  std::uint32_t previous_id_{};
  ir::Opcode previous_opcode_{};
  double start_ms_{};
  std::string name_;
};

void trace_submit(std::string_view category, std::string_view api,
                  std::uint64_t bytes = 0U, std::string_view stream = "") {
  if (!active_tracer)
    return;
  const auto now = active_tracer->now_ms();
  active_tracer->record(category, api, bytes, stream, now, now);
}

// Records a host-blocking call as a wait spanning its actual duration.
class TraceWaitScope {
public:
  explicit TraceWaitScope(std::string_view api) : api_(api) {
    if (active_tracer)
      start_ms_ = active_tracer->now_ms();
  }
  ~TraceWaitScope() {
    if (active_tracer)
      active_tracer->record(telemetry::category::wait, api_, 0U, "host",
                            start_ms_, active_tracer->now_ms());
  }
  TraceWaitScope(const TraceWaitScope &) = delete;
  TraceWaitScope &operator=(const TraceWaitScope &) = delete;

private:
  std::string_view api_;
  double start_ms_{};
};

CUresult counted_launch_kernel(CUfunction function, unsigned grid_x,
                               unsigned grid_y, unsigned grid_z,
                               unsigned block_x, unsigned block_y,
                               unsigned block_z, unsigned shared_bytes,
                               CUstream stream, void **parameters,
                               void **extra) {
  if (active_telemetry)
    ++active_telemetry->kernel_launches;
  trace_submit(telemetry::category::generated_kernel, "cuLaunchKernel");
  return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y,
                        block_z, shared_bytes, stream, parameters, extra);
}

CUresult counted_memcpy_htod(CUdeviceptr destination, const void *source,
                             std::size_t bytes, CUstream stream) {
  if (active_telemetry) {
    ++active_telemetry->h2d_copies;
    active_telemetry->h2d_bytes += bytes;
  }
  trace_submit(telemetry::category::h2d, "cuMemcpyHtoDAsync", bytes);
  return cuMemcpyHtoDAsync(destination, source, bytes, stream);
}

CUresult counted_memcpy_dtoh(void *destination, CUdeviceptr source,
                             std::size_t bytes, CUstream stream) {
  if (active_telemetry) {
    ++active_telemetry->d2h_copies;
    active_telemetry->d2h_bytes += bytes;
  }
  trace_submit(telemetry::category::d2h, "cuMemcpyDtoHAsync", bytes);
  return cuMemcpyDtoHAsync(destination, source, bytes, stream);
}

CUresult counted_memcpy_dtod(CUdeviceptr destination, CUdeviceptr source,
                             std::size_t bytes, CUstream stream) {
  if (active_telemetry) {
    ++active_telemetry->d2d_copies;
    active_telemetry->d2d_bytes += bytes;
  }
  trace_submit(telemetry::category::d2d, "cuMemcpyDtoDAsync", bytes);
  return cuMemcpyDtoDAsync(destination, source, bytes, stream);
}

CUresult counted_event_record(CUevent event, CUstream stream) {
  if (active_telemetry)
    ++active_telemetry->event_records;
  trace_submit(telemetry::category::synchronization, "cuEventRecord");
  return cuEventRecord(event, stream);
}

CUresult counted_stream_wait_event(CUstream stream, CUevent event,
                                   unsigned flags) {
  if (active_telemetry)
    ++active_telemetry->stream_wait_events;
  trace_submit(telemetry::category::synchronization, "cuStreamWaitEvent");
  return cuStreamWaitEvent(stream, event, flags);
}

CUresult counted_event_synchronize(CUevent event) {
  if (active_telemetry)
    ++active_telemetry->host_event_synchronizes;
  TraceWaitScope wait("cuEventSynchronize");
  return cuEventSynchronize(event);
}

CUresult counted_stream_synchronize(CUstream stream) {
  if (active_telemetry)
    ++active_telemetry->host_stream_synchronizes;
  TraceWaitScope wait("cuStreamSynchronize");
  return cuStreamSynchronize(stream);
}

CUresult counted_mem_alloc(CUdeviceptr *pointer, std::size_t bytes) {
  if (active_telemetry)
    ++active_telemetry->device_mem_allocs;
  trace_submit(telemetry::category::allocation, "cuMemAlloc", bytes);
  return cuMemAlloc(pointer, bytes);
}

CUresult counted_mem_host_alloc(void **pointer, std::size_t bytes,
                                unsigned flags) {
  if (active_telemetry)
    ++active_telemetry->pinned_mem_allocs;
  trace_submit(telemetry::category::allocation, "cuMemHostAlloc", bytes);
  return cuMemHostAlloc(pointer, bytes, flags);
}

void count_cublaslt_matmul() {
  if (active_telemetry)
    ++active_telemetry->cublaslt_matmuls;
  trace_submit(telemetry::category::gemm, "cublasLtMatmul");
}

void count_cudnn_attention_dispatch() {
  if (active_telemetry)
    ++active_telemetry->cudnn_attention_dispatches;
  trace_submit(telemetry::category::attention, "cudnn-sdpa");
}

void count_cudnn_convolution_dispatch() {
  if (active_telemetry)
    ++active_telemetry->cudnn_convolution_dispatches;
  trace_submit(telemetry::category::convolution, "cudnn-convolution");
}

void count_cutlass_launch() {
  if (active_telemetry)
    ++active_telemetry->cutlass_launches;
  trace_submit(telemetry::category::gemm, "cutlass");
}

// One dispatch = the DSO's quantize-QK + quantize-V + attend launcher trio.
void count_ck_attention_dispatch() {
  if (active_telemetry)
    ++active_telemetry->ck_attention_dispatches;
  trace_submit(telemetry::category::attention, "ck-int8-dso");
}

template <typename... Arguments>
cublasStatus_t counted_cublas_gemm_ex(Arguments &&...arguments) {
  if (active_telemetry)
    ++active_telemetry->cublas_gemms;
  trace_submit(telemetry::category::gemm, "cublasGemmEx");
  return cublasGemmEx(std::forward<Arguments>(arguments)...);
}

class Context {
public:
  explicit Context(int ordinal) : ordinal_(ordinal) {
    check(cuInit(0), "cuInit");
    int count = 0;
    check(cuDeviceGetCount(&count), "cuDeviceGetCount");
    if (ordinal < 0 || ordinal >= count)
      fail("CUDA device ordinal is out of range");
    check(cuDeviceGet(&device_, ordinal), "cuDeviceGet");
    check(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
    check(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    check(cuStreamCreate(&copy_stream_, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate copy");
    check(cublasCreate(&cublas_), "cublasCreate");
    check(cublasSetStream(cublas_, reinterpret_cast<cudaStream_t>(stream_)),
          "cublasSetStream");
    // Correctness admission is strict. TF32 can later be exposed as an
    // explicit candidate, but is never enabled silently.
    check(cublasSetMathMode(cublas_, CUBLAS_PEDANTIC_MATH),
          "cublasSetMathMode");
    check(cublasLtCreate(&cublas_lt_), "cublasLtCreate");
  }

  ~Context() {
    if (cublas_lt_)
      (void)cublasLtDestroy(cublas_lt_);
    if (cublas_)
      (void)cublasDestroy(cublas_);
    if (stream_)
      (void)cuStreamDestroy(stream_);
    if (copy_stream_)
      (void)cuStreamDestroy(copy_stream_);
    (void)cuCtxSetCurrent(previous_);
    if (context_)
      (void)cuDevicePrimaryCtxRelease(device_);
  }

  Context(const Context &) = delete;
  Context &operator=(const Context &) = delete;

  CUdevice device() const { return device_; }
  int ordinal() const { return ordinal_; }
  CUstream stream() const { return stream_; }
  CUstream copy_stream() const { return copy_stream_; }
  cublasHandle_t cublas() const { return cublas_; }
  cublasLtHandle_t cublas_lt() const { return cublas_lt_; }

private:
  int ordinal_{};
  CUdevice device_{};
  CUcontext context_{};
  CUcontext previous_{};
  CUstream stream_{};
  CUstream copy_stream_{};
  cublasHandle_t cublas_{};
  cublasLtHandle_t cublas_lt_{};
};

class Stream {
public:
  Stream() {
    check(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING),
          "cuStreamCreate auxiliary compute");
  }
  ~Stream() {
    if (stream_)
      (void)cuStreamDestroy(stream_);
  }
  Stream(const Stream &) = delete;
  Stream &operator=(const Stream &) = delete;
  CUstream get() const { return stream_; }

private:
  CUstream stream_{};
};

class Module {
public:
  explicit Module(const std::string &ptx) {
    // Capture the driver JIT log so an invalid-PTX failure names the
    // instruction and line instead of only the error code.
    std::array<char, 4096> error_log{};
    std::array<CUjit_option, 2> option_keys = {CU_JIT_ERROR_LOG_BUFFER,
                                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
    std::array<void *, 2> option_values = {
        static_cast<void *>(error_log.data()),
        reinterpret_cast<void *>(static_cast<std::uintptr_t>(error_log.size()))};
    const auto status = cuModuleLoadDataEx(
        &module_, ptx.data(), static_cast<unsigned>(option_keys.size()),
        option_keys.data(), option_values.data());
    if (status != CUDA_SUCCESS) {
      const std::string what =
          std::string("cuModuleLoadDataEx: ") + error_log.data();
      check(status, what.c_str());
    }
  }
  ~Module() {
    if (module_)
      (void)cuModuleUnload(module_);
  }
  Module(const Module &) = delete;
  Module &operator=(const Module &) = delete;
  CUmodule get() const { return module_; }

private:
  CUmodule module_{};
};

class Event {
public:
  explicit Event(unsigned flags = CU_EVENT_DEFAULT) {
    check(cuEventCreate(&event_, flags), "cuEventCreate");
  }
  ~Event() {
    if (event_)
      (void)cuEventDestroy(event_);
  }
  CUevent get() const { return event_; }

private:
  CUevent event_{};
};

// One device reservation backing every prepare-time allocation (memory-plan
// slots plus the feature workspaces). Carved front-to-back at 256-byte
// granularity; nothing is freed or relocated until the prepared execution
// is destroyed (CUTLASS freezes device pointers at plan build, so all
// carving happens before any plan is built against the pointers).
class DeviceArena {
public:
  explicit DeviceArena(std::uint64_t capacity) : capacity_(capacity) {
    if (capacity_ != 0U)
      check(counted_mem_alloc(&base_, static_cast<std::size_t>(capacity_)),
            "cuMemAlloc device arena");
  }
  ~DeviceArena() {
    if (base_)
      (void)cuMemFree(base_);
  }
  DeviceArena(const DeviceArena &) = delete;
  DeviceArena &operator=(const DeviceArena &) = delete;

  CUdeviceptr take(std::uint64_t bytes, const char *label) {
    if (bytes > std::numeric_limits<std::uint64_t>::max() - 255U)
      fail("device arena alignment overflow");
    const auto aligned = (bytes + 255U) & ~std::uint64_t{255U};
    if (aligned > capacity_ - offset_)
      fail(std::string("device arena exhausted taking ") + label +
           ": requested=" + std::to_string(bytes) +
           " used=" + std::to_string(offset_) +
           " capacity=" + std::to_string(capacity_));
    const auto pointer = base_ + offset_;
    offset_ += aligned;
    return pointer;
  }

  std::uint64_t capacity() const { return capacity_; }
  std::uint64_t used() const { return offset_; }

private:
  CUdeviceptr base_{};
  std::uint64_t capacity_{};
  std::uint64_t offset_{};
};

class DeviceBuffers {
public:
  ~DeviceBuffers() {
    if (!owns_)
      return;
    for (const auto pointer : allocations_) {
      if (pointer)
        (void)cuMemFree(pointer);
    }
  }

  void allocate(
      const ir::Program &program, const compiler::MemoryPlan &plan,
      const std::unordered_set<std::uint32_t> &excluded_tensors = {},
      DeviceArena *arena = nullptr) {
    owns_ = arena == nullptr;
    allocations_.resize(plan.slots.size());
    for (const auto &slot : plan.slots) {
      CUdeviceptr pointer{};
      if (arena)
        pointer = arena->take(slot.bytes, "memory-plan slot");
      else
        check(counted_mem_alloc(&pointer, static_cast<std::size_t>(slot.bytes)),
              "cuMemAlloc");
      allocations_.at(slot.id) = pointer;
    }
    for (const auto &desc : program.tensors) {
      if (excluded_tensors.contains(desc.id))
        continue;
      const auto *assignment = plan.assignment(desc.id);
      if (!assignment)
        fail("memory plan omitted tensor " + std::to_string(desc.id));
      pointers_.emplace(desc.id, allocations_.at(assignment->slot_id));
    }
  }

  CUdeviceptr &at(std::uint32_t id) { return pointers_.at(id); }
  const CUdeviceptr &at(std::uint32_t id) const { return pointers_.at(id); }
  bool contains(std::uint32_t id) const { return pointers_.contains(id); }
  void bind_external(std::uint32_t id, CUdeviceptr pointer) {
    if (pointer == 0U || pointers_.contains(id))
      fail("invalid external CUDA tensor binding " + std::to_string(id));
    pointers_.emplace(id, pointer);
    external_.insert(id);
  }
  void rebind_external(std::uint32_t id, CUdeviceptr pointer) {
    if (pointer == 0U || !external_.contains(id))
      fail("invalid external CUDA tensor rebinding " + std::to_string(id));
    pointers_.at(id) = pointer;
  }

private:
  std::vector<CUdeviceptr> allocations_;
  std::unordered_map<std::uint32_t, CUdeviceptr> pointers_;
  std::unordered_set<std::uint32_t> external_;
  bool owns_{true};
};

class Workspace {
public:
  explicit Workspace(std::size_t bytes, DeviceArena *arena = nullptr)
      : bytes_(bytes) {
    if (bytes_ == 0U)
      return;
    if (arena) {
      pointer_ = arena->take(bytes_, "feature workspace");
      owns_ = false;
    } else {
      check(counted_mem_alloc(&pointer_, bytes_),
            "cuMemAlloc cuBLASLt workspace");
    }
  }
  ~Workspace() {
    if (pointer_ && owns_)
      (void)cuMemFree(pointer_);
  }
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  void *data() const { return reinterpret_cast<void *>(pointer_); }
  CUdeviceptr pointer() const { return pointer_; }
  std::size_t size() const { return bytes_; }

private:
  CUdeviceptr pointer_{};
  std::size_t bytes_{};
  bool owns_{true};
};

class PinnedHostWorkspace {
public:
  explicit PinnedHostWorkspace(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0U)
      check(counted_mem_host_alloc(&pointer_, bytes_, CU_MEMHOSTALLOC_PORTABLE),
            "cuMemHostAlloc streamed staging");
  }
  ~PinnedHostWorkspace() {
    if (pointer_)
      (void)cuMemFreeHost(pointer_);
  }
  PinnedHostWorkspace(const PinnedHostWorkspace &) = delete;
  PinnedHostWorkspace &operator=(const PinnedHostWorkspace &) = delete;
  void *data() const { return pointer_; }
  std::size_t size() const { return bytes_; }

private:
  void *pointer_{};
  std::size_t bytes_{};
};

// Exact backend lowering for large unbatched F32 attention.  DiffIR keeps one
// Attention semantic; implementation 3 is only an execution-plan choice for
// shapes that the generated S<=4096 kernel and cuDNN BF16/F16 paths cannot
// admit.  Scores are intentionally materialized so no approximate/online
// attention math is introduced at the source-faithful parity gate.
struct MaterializedF32AttentionPlan {
  std::uint32_t operation{};
  int sequence{};
  int head_dim{};
  float scale{};

  std::uint64_t score_bytes() const {
    return static_cast<std::uint64_t>(sequence) *
           static_cast<std::uint64_t>(sequence) * sizeof(float);
  }

  void execute(const ir::Operation &op, const DeviceBuffers &buffers,
               cublasHandle_t cublas, CUfunction softmax,
               CUdeviceptr scores, CUstream stream) const {
    constexpr float zero = 0.0F;
    constexpr float one = 1.0F;
    const auto query = buffers.at(op.inputs.at(0));
    const auto key = buffers.at(op.inputs.at(1));
    const auto value = buffers.at(op.inputs.at(2));
    const auto output = buffers.at(op.outputs.at(0));

    // DiffIR tensors are row-major [S,D].  cuBLAS sees those same bytes as
    // column-major [D,S], so C_col = K_row * Q_row^T stores
    // C_col == scores_row^T without a transpose materialization.
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, sequence, sequence,
              head_dim, &scale, reinterpret_cast<const void *>(key),
              CUDA_R_32F, head_dim,
              reinterpret_cast<const void *>(query), CUDA_R_32F, head_dim,
              &zero, reinterpret_cast<void *>(scores), CUDA_R_32F, sequence,
              CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cuBLAS materialized F32 attention QK");

    auto rows = sequence;
    auto columns = sequence;
    std::array<void *, 3> softmax_arguments = {&scores, &rows, &columns};
    check(counted_launch_kernel(
              softmax, static_cast<unsigned>(sequence), 1U, 1U, 256U, 1U,
              1U, 256U * sizeof(float), stream, softmax_arguments.data(),
              nullptr),
          "cuLaunchKernel materialized F32 attention softmax");

    // O_row^T = V_row^T * P_row^T, again requiring no layout conversion.
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, sequence, sequence,
              &one, reinterpret_cast<const void *>(value), CUDA_R_32F,
              head_dim, reinterpret_cast<const void *>(scores), CUDA_R_32F,
              sequence, &zero, reinterpret_cast<void *>(output), CUDA_R_32F,
              head_dim, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
          "cuBLAS materialized F32 attention PV");
  }
};

std::string materialized_f32_attention_source(bool enabled) {
  if (!enabled)
    return {};
  return R"CUDA(
extern "C" __global__ void dif_materialized_f32_attention_softmax(
    float* scores, int rows, int columns) {
  const int row = (int)blockIdx.x;
  const int lane = (int)threadIdx.x;
  if (row >= rows) return;
  extern __shared__ float shared[];
  float maximum = -3.402823466e+38F;
  const unsigned long long offset =
      (unsigned long long)row * (unsigned long long)columns;
  for (int column = lane; column < columns; column += (int)blockDim.x)
    maximum = fmaxf(maximum, scores[offset + (unsigned long long)column]);
  shared[lane] = maximum;
  __syncthreads();
  for (int stride = (int)blockDim.x / 2; stride != 0; stride >>= 1) {
    if (lane < stride)
      shared[lane] = fmaxf(shared[lane], shared[lane + stride]);
    __syncthreads();
  }
  maximum = shared[0];
  float sum = 0.0F;
  for (int column = lane; column < columns; column += (int)blockDim.x) {
    const float probability =
        expf(scores[offset + (unsigned long long)column] - maximum);
    scores[offset + (unsigned long long)column] = probability;
    sum += probability;
  }
  shared[lane] = sum;
  __syncthreads();
  for (int stride = (int)blockDim.x / 2; stride != 0; stride >>= 1) {
    if (lane < stride) shared[lane] += shared[lane + stride];
    __syncthreads();
  }
  const float inverse_sum = 1.0F / shared[0];
  for (int column = lane; column < columns; column += (int)blockDim.x)
    scores[offset + (unsigned long long)column] *= inverse_sum;
}
)CUDA";
}

#if DIF_HAS_CUDNN
struct CudnnAttentionKey {
  ir::DType dtype{};
  std::uint64_t batch{};
  std::uint64_t sequence{};
  std::uint64_t heads{};
  std::uint64_t kv_heads{};
  std::uint64_t head_dim{};
  std::uint64_t scale_bits{};
  bool causal{};
  bool additive_bias{};
  std::uint32_t heuristic{};

  bool operator==(const CudnnAttentionKey &) const = default;
};

struct CudnnAttentionKeyHash {
  std::size_t operator()(const CudnnAttentionKey &key) const noexcept {
    std::size_t result = 0xcbf29ce484222325ULL;
    auto mix = [&](std::uint64_t value) {
      result ^= static_cast<std::size_t>(value);
      result *= 0x100000001b3ULL;
    };
    mix(static_cast<std::uint64_t>(key.dtype));
    mix(key.batch);
    mix(key.sequence);
    mix(key.heads);
    mix(key.kv_heads);
    mix(key.head_dim);
    mix(key.scale_bits);
    mix(key.causal ? 1U : 0U);
    mix(key.additive_bias ? 1U : 0U);
    mix(key.heuristic);
    return result;
  }
};

struct CudnnConv2dKey {
  ir::DType dtype{};
  std::array<std::uint64_t, 4> input{};
  std::array<std::uint64_t, 4> weight{};
  std::array<std::uint64_t, 4> output{};
  std::uint64_t stride_h{}, stride_w{};
  std::uint64_t pad_h{}, pad_w{};
  std::uint64_t dilation_h{}, dilation_w{};
  std::uint64_t groups{};
  std::uint64_t workspace_limit{};
  bool biased{};
  bool deterministic{};

  bool operator==(const CudnnConv2dKey &) const = default;
};

struct CudnnConv2dKeyHash {
  std::size_t operator()(const CudnnConv2dKey &key) const noexcept {
    std::size_t result = 0xcbf29ce484222325ULL;
    auto mix = [&](std::uint64_t value) {
      result ^= static_cast<std::size_t>(value);
      result *= 0x100000001b3ULL;
    };
    mix(static_cast<std::uint64_t>(key.dtype));
    for (const auto value : key.input) mix(value);
    for (const auto value : key.weight) mix(value);
    for (const auto value : key.output) mix(value);
    mix(key.stride_h); mix(key.stride_w); mix(key.pad_h); mix(key.pad_w);
    mix(key.dilation_h); mix(key.dilation_w); mix(key.groups);
    mix(key.workspace_limit); mix(key.biased ? 1U : 0U);
    mix(key.deterministic ? 1U : 0U);
    return result;
  }
};

struct CudnnConv3dKey {
  ir::DType dtype{};
  std::array<std::uint64_t, 5> input{};
  std::array<std::uint64_t, 5> weight{};
  std::array<std::uint64_t, 5> output{};
  std::uint64_t stride_t{}, stride_h{}, stride_w{};
  std::uint64_t pad_t{}, pad_h{}, pad_w{};
  std::uint64_t dilation_t{}, dilation_h{}, dilation_w{};
  std::uint64_t groups{};
  std::uint64_t workspace_limit{};
  bool biased{};
  bool deterministic{};

  bool operator==(const CudnnConv3dKey &) const = default;
};

struct CudnnConv3dKeyHash {
  std::size_t operator()(const CudnnConv3dKey &key) const noexcept {
    std::size_t result = 0xcbf29ce484222325ULL;
    auto mix = [&](std::uint64_t value) {
      result ^= static_cast<std::size_t>(value);
      result *= 0x100000001b3ULL;
    };
    mix(static_cast<std::uint64_t>(key.dtype));
    for (const auto value : key.input) mix(value);
    for (const auto value : key.weight) mix(value);
    for (const auto value : key.output) mix(value);
    mix(key.stride_t); mix(key.stride_h); mix(key.stride_w);
    mix(key.pad_t); mix(key.pad_h); mix(key.pad_w);
    mix(key.dilation_t); mix(key.dilation_h); mix(key.dilation_w);
    mix(key.groups); mix(key.workspace_limit); mix(key.biased ? 1U : 0U);
    mix(key.deterministic ? 1U : 0U);
    return result;
  }
};
#endif

class LinearPlan {
public:
  LinearPlan(const ir::Program &program, const ir::Operation &op,
             const DeviceBuffers &buffers, cublasLtHandle_t handle,
             std::size_t workspace_bytes, bool expand_algorithms, int major,
             int minor, const std::filesystem::path &cache_directory,
             bool persist, bool allow_restore,
             LinearHeuristicCacheStats *cache_stats,
             CUfunction addmm_prefill = nullptr,
             bool deterministic_algorithms = false) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    const auto *output = program.tensor(op.outputs.at(0));
    if (!input || !weight || !output || input->dims.empty() ||
        weight->dims.size() != 2U)
      fail("invalid Linear descriptors while building cuBLASLt plan");
    const auto m64 = input->dims[0];
    const auto k64 = input->element_count() / m64;
    const auto n64 = weight->dims[0];
    if (m64 > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        n64 > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
        k64 > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
      fail("Linear shape is not representable by cuBLASLt");
    const auto m = static_cast<std::int64_t>(m64);
    const auto n = static_cast<std::int64_t>(n64);
    const auto k = static_cast<std::int64_t>(k64);
    const auto implementation = op.u64(ir::AttrKey::Implementation, 1U);
    const auto storage = input->dtype == ir::DType::BF16
                             ? CUDA_R_16BF
                             : input->dtype == ir::DType::F16 ? CUDA_R_16F
                                                               : CUDA_R_32F;
    const auto compute = implementation == 2U ? CUBLAS_COMPUTE_32F_FAST_TF32
                                               : CUBLAS_COMPUTE_32F;
    check(cublasLtMatmulDescCreate(&operation_, compute, CUDA_R_32F),
          "cublasLtMatmulDescCreate");
    has_bias_ = op.inputs.size() == 3U;
    bias_mode_ = static_cast<ir::LinearBiasMode>(op.u64(
        ir::AttrKey::LinearBiasMode,
        static_cast<std::uint64_t>(ir::LinearBiasMode::Epilogue)));
    if (bias_mode_ == ir::LinearBiasMode::Addmm) {
      if (!has_bias_)
        fail("cuBLASLt addmm bias mode requires a bias input");
      if (!addmm_prefill)
        fail("cuBLASLt addmm bias mode is missing its broadcast prefill kernel");
      addmm_prefill_ = addmm_prefill;
      addmm_prefill_grid_ = static_cast<unsigned>(
          (output->element_count() + 255U) / 256U);
    }
    const cublasOperation_t trans_a =
        has_bias_ ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t trans_b =
        has_bias_ ? CUBLAS_OP_N : CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSA,
                                         &trans_a, sizeof(trans_a)),
          "cublasLtMatmulDescSetAttribute trans A");
    check(cublasLtMatmulDescSetAttribute(operation_, CUBLASLT_MATMUL_DESC_TRANSB,
                                         &trans_b, sizeof(trans_b)),
          "cublasLtMatmulDescSetAttribute trans B");
    if (has_bias_ && bias_mode_ == ir::LinearBiasMode::Epilogue) {
      constexpr cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue,
                sizeof(epilogue)),
            "cublasLtMatmulDescSetAttribute bias epilogue");
      const auto bias_pointer =
          reinterpret_cast<const void *>(buffers.at(op.inputs[2]));
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                sizeof(bias_pointer)),
            "cublasLtMatmulDescSetAttribute bias pointer");
    }
    if (has_bias_) {
      // Match PyTorch addmm's supported cuBLASLt bias orientation: reinterpret
      // row-major X/W/Y as column-major X^T/W^T/Y^T and compute
      // Y^T = W * X^T.
      check(cublasLtMatrixLayoutCreate(&input_, storage, k, m, k),
            "cublasLtMatrixLayoutCreate transposed input");
      check(cublasLtMatrixLayoutCreate(&weight_, storage, k, n, k),
            "cublasLtMatrixLayoutCreate transposed weight");
      check(cublasLtMatrixLayoutCreate(&output_, storage, n, m, n),
            "cublasLtMatrixLayoutCreate transposed output");
    } else {
      check(cublasLtMatrixLayoutCreate(&input_, storage, m, k, k),
            "cublasLtMatrixLayoutCreate input");
      check(cublasLtMatrixLayoutCreate(&weight_, storage, n, k, k),
            "cublasLtMatrixLayoutCreate weight");
      check(cublasLtMatrixLayoutCreate(&output_, storage, m, n, n),
            "cublasLtMatrixLayoutCreate output");
      constexpr cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
      for (auto layout : {input_, weight_, output_}) {
        check(cublasLtMatrixLayoutSetAttribute(
                  layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                  sizeof(row_major)),
              "cublasLtMatrixLayoutSetAttribute row major");
      }
    }
    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulPreferenceCreate(&preference),
          "cublasLtMatmulPreferenceCreate");
    try {
      const auto default_workspace =
          input->dtype == ir::DType::F32 ? 1U * 1024U * 1024U
                                         : workspace_bytes;
      workspace_limit_bytes_ = static_cast<std::size_t>(op.u64(
          ir::AttrKey::WorkspaceLimitBytes, default_workspace));
      const auto preference_workspace =
          std::min(default_workspace, workspace_limit_bytes_);
      check(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &preference_workspace, sizeof(preference_workspace)),
            "cublasLtMatmulPreferenceSetAttribute workspace");
      if (input->dtype == ir::DType::F32) {
        const auto pointer_alignment = [](CUdeviceptr pointer) {
          std::uint32_t alignment = 256U;
          while (alignment > 1U && pointer % alignment != 0U)
            alignment /= 2U;
          return alignment;
        };
        const auto matrix_a_id = has_bias_ ? op.inputs[1] : op.inputs[0];
        const auto matrix_b_id = has_bias_ ? op.inputs[0] : op.inputs[1];
        const std::array<std::pair<cublasLtMatmulPreferenceAttributes_t,
                                   std::uint32_t>,
                         4>
            alignments{{
                {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                 pointer_alignment(buffers.at(matrix_a_id))},
                {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                 pointer_alignment(buffers.at(matrix_b_id))},
                {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                 pointer_alignment(buffers.at(op.outputs[0]))},
                {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                 has_bias_ ? pointer_alignment(buffers.at(op.inputs[2]))
                           : 256U},
            }};
        for (const auto &[attribute, alignment] : alignments)
          check(cublasLtMatmulPreferenceSetAttribute(
                    preference, attribute, &alignment, sizeof(alignment)),
                "cublasLtMatmulPreferenceSetAttribute alignment");
      }
      const auto matrix_a = has_bias_ ? weight_ : input_;
      const auto matrix_b = has_bias_ ? input_ : weight_;
      persist_ = persist;
      cache_stats_ = cache_stats;
      if (persist) {
        // PTX-cache-style keyed store: the key is the full problem identity,
        // so any environment change (library, arch, workspace policy) misses
        // and falls open to fresh heuristics.
        const std::string key_material =
            "linear-algo-v1\nm=" + std::to_string(m) +
            " n=" + std::to_string(n) + " k=" + std::to_string(k) +
            " storage=" + std::to_string(static_cast<int>(storage)) +
            " compute=" + std::to_string(static_cast<int>(compute)) +
            " bias=" + std::to_string(has_bias_ ? 1 : 0) +
            " bias_mode=" +
            std::to_string(static_cast<std::uint64_t>(bias_mode_)) +
            " workspace=" + std::to_string(preference_workspace) +
            " ltver=" + std::to_string(cublasLtGetVersion()) +
            " arch=sm_" + std::to_string(major) + std::to_string(minor);
        const auto key_bytes = std::span<const std::uint8_t>(
            reinterpret_cast<const std::uint8_t *>(key_material.data()),
            key_material.size());
        const auto digest = hex_digest(sha256(key_bytes));
        const auto directory =
            cache_directory.empty()
                ? std::filesystem::temp_directory_path() / "dif-ptx-cache"
                : cache_directory;
        std::filesystem::create_directories(directory);
        tuned_cache_file_ =
            directory / ("linear-algo-tuned-" + digest + ".txt");
        passive_cache_file_ =
            directory / ("linear-algo-passive-" + digest + ".txt");
      }
      bool restored = false;
      if (persist && allow_restore) {
        for (const auto *path : {&tuned_cache_file_, &passive_cache_file_}) {
          PersistedAlgorithm stored;
          if (!read_persisted_algorithm(*path, stored))
            continue;
          cublasLtMatmulHeuristicResult_t candidate{};
          if (cublasLtMatmulAlgoInit(handle, compute, CUDA_R_32F, storage,
                                     storage, storage, storage,
                                     stored.algorithm_id, &candidate.algo) !=
                  CUBLAS_STATUS_SUCCESS ||
              !apply_persisted_configuration(candidate.algo, stored)) {
            if (cache_stats_)
              ++cache_stats_->rejected;
            continue;
          }
          cublasLtMatmulHeuristicResult_t checked{};
          if (cublasLtMatmulAlgoCheck(handle, operation_, matrix_a, matrix_b,
                                      output_, output_, &candidate.algo,
                                      &checked) != CUBLAS_STATUS_SUCCESS ||
              checked.workspaceSize > preference_workspace) {
            if (cache_stats_)
              ++cache_stats_->rejected;
            continue;
          }
          candidate.workspaceSize = checked.workspaceSize;
          candidate.wavesCount = checked.wavesCount;
          candidate.state = CUBLAS_STATUS_SUCCESS;
          heuristics_.assign(1U, candidate);
          heuristic_ = candidate;
          restored = true;
          if (cache_stats_)
            ++cache_stats_->restored;
          break;
        }
      }
      if (!restored) {
      constexpr int requested_algorithms = 32;
      std::array<cublasLtMatmulHeuristicResult_t, requested_algorithms>
          heuristics{};
      int returned = 0;
      check(cublasLtMatmulAlgoGetHeuristic(handle, operation_, matrix_a,
                                            matrix_b, output_, output_,
                                            preference, requested_algorithms,
                                            heuristics.data(), &returned),
            "cublasLtMatmulAlgoGetHeuristic");
      if (returned == 0)
        fail("cuBLASLt found no admitted Linear algorithm");
      heuristics_.assign(heuristics.begin(), heuristics.begin() + returned);
      if (expand_algorithms) {
        constexpr int maximum_algorithm_ids = 64;
        std::array<int, maximum_algorithm_ids> algorithm_ids{};
        int algorithm_count = 0;
        check(cublasLtMatmulAlgoGetIds(
                  handle, compute, CUDA_R_32F, storage, storage, storage,
                  storage, maximum_algorithm_ids, algorithm_ids.data(),
                  &algorithm_count),
              "cublasLtMatmulAlgoGetIds");
        constexpr std::uint32_t limited =
            static_cast<std::uint32_t>(CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID);
        check(cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_SEARCH_MODE, &limited,
                  sizeof(limited)),
              "cublasLtMatmulPreferenceSetAttribute search mode");
        for (int index = 0; index < algorithm_count; ++index) {
          cublasLtMatmulHeuristicResult_t candidate{};
          const auto init_status = cublasLtMatmulAlgoInit(
              handle, compute, CUDA_R_32F, storage, storage, storage, storage,
              algorithm_ids[index], &candidate.algo);
          if (init_status != CUBLAS_STATUS_SUCCESS)
            continue;
          int candidate_count = 0;
          const auto heuristic_status = cublasLtMatmulAlgoGetHeuristic(
              handle, operation_, matrix_a, matrix_b, output_, output_,
              preference, 1, &candidate, &candidate_count);
          if (heuristic_status != CUBLAS_STATUS_SUCCESS ||
              candidate_count != 1 || candidate.state != CUBLAS_STATUS_SUCCESS)
            continue;
          const auto duplicate = std::any_of(
              heuristics_.begin(), heuristics_.end(),
              [&](const cublasLtMatmulHeuristicResult_t &existing) {
                return same_algorithm_config(existing, candidate);
              });
          if (!duplicate)
            heuristics_.push_back(candidate);
        }
      }
      heuristic_ = heuristics_.front();
      if (deterministic_algorithms) {
        const auto deterministic = std::find_if(
            heuristics_.begin(), heuristics_.end(),
            [&](const cublasLtMatmulHeuristicResult_t &candidate) {
              return algorithm_config<std::int32_t>(
                         candidate, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) <= 1 &&
                     algorithm_config<std::uint32_t>(
                         candidate,
                         CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME) == 0U;
            });
        if (deterministic == heuristics_.end())
          fail("cuBLASLt found no no-split deterministic Linear algorithm");
        heuristic_ = *deterministic;
      }
      if (persist)
        save_persisted_algorithm(passive_cache_file_, heuristic_,
                                 cache_stats_ ? &cache_stats_->saved_passive
                                              : nullptr);
      }
    } catch (...) {
      (void)cublasLtMatmulPreferenceDestroy(preference);
      throw;
    }
    check(cublasLtMatmulPreferenceDestroy(preference),
          "cublasLtMatmulPreferenceDestroy");
  }

  ~LinearPlan() {
    if (output_)
      (void)cublasLtMatrixLayoutDestroy(output_);
    if (weight_)
      (void)cublasLtMatrixLayoutDestroy(weight_);
    if (input_)
      (void)cublasLtMatrixLayoutDestroy(input_);
    if (operation_)
      (void)cublasLtMatmulDescDestroy(operation_);
  }
  LinearPlan(const LinearPlan &) = delete;
  LinearPlan &operator=(const LinearPlan &) = delete;

  LinearTuningResult tune(std::uint32_t operation_id,
                          const ir::Operation &op,
                          const DeviceBuffers &buffers,
                          cublasLtHandle_t handle,
                          const Workspace &workspace, CUstream stream,
                          std::uint32_t warmups,
                          std::uint32_t iterations,
                          std::uint32_t sessions) {
    if (iterations == 0U || sessions < 2U)
      fail("cuBLASLt tuning requires nonzero iterations and at least two sessions");
    const auto tuning_start = std::chrono::steady_clock::now();
    std::vector<std::vector<double>> session_means(heuristics_.size());
    std::vector<bool> admitted(heuristics_.size(), true);
    std::vector<std::size_t> order(heuristics_.size());
    std::iota(order.begin(), order.end(), 0U);
    for (std::uint32_t session = 0U; session < sessions; ++session) {
      if (session != 0U)
        std::reverse(order.begin(), order.end());
      for (const auto index : order) {
        const auto &candidate = heuristics_[index];
        if (!admitted[index] || candidate.state != CUBLAS_STATUS_SUCCESS ||
            candidate.workspaceSize > workspace.size()) {
          admitted[index] = false;
          continue;
        }
        for (std::uint32_t warmup = 0U; warmup < warmups; ++warmup) {
          if (launch_candidate(op, buffers, handle, workspace, stream,
                               candidate) != CUBLAS_STATUS_SUCCESS) {
            admitted[index] = false;
            break;
          }
        }
        if (!admitted[index])
          continue;
        Event start;
        Event stop;
        check(counted_event_record(start.get(), stream),
              "cuEventRecord Linear tuning start");
        for (std::uint32_t iteration = 0U; iteration < iterations;
             ++iteration) {
          if (launch_candidate(op, buffers, handle, workspace, stream,
                               candidate) != CUBLAS_STATUS_SUCCESS) {
            admitted[index] = false;
            break;
          }
        }
        check(counted_event_record(stop.get(), stream),
              "cuEventRecord Linear tuning stop");
        check(counted_event_synchronize(stop.get()),
              "cuEventSynchronize Linear tuning");
        if (!admitted[index])
          continue;
        float elapsed = 0.0F;
        check(cuEventElapsedTime(&elapsed, start.get(), stop.get()),
              "cuEventElapsedTime Linear tuning");
        session_means[index].push_back(
            static_cast<double>(elapsed) / static_cast<double>(iterations));
      }
    }

    LinearTuningResult result;
    result.operation_id = operation_id;
    std::vector<double> means(heuristics_.size(),
                              std::numeric_limits<double>::infinity());
    std::vector<double> spreads(heuristics_.size(), 0.0);
    for (std::size_t index = 0U; index < heuristics_.size(); ++index) {
      const auto &samples = session_means[index];
      if (!admitted[index] || samples.size() != sessions)
        continue;
      means[index] =
          std::accumulate(samples.begin(), samples.end(), 0.0) /
          static_cast<double>(samples.size());
      const auto [minimum, maximum] =
          std::minmax_element(samples.begin(), samples.end());
      spreads[index] = *maximum - *minimum;
      result.candidates.push_back(
          {static_cast<std::uint32_t>(index),
           algorithm_id(heuristics_[index]),
           algorithm_config<std::uint32_t>(heuristics_[index],
                                           CUBLASLT_ALGO_CONFIG_TILE_ID),
           algorithm_config<std::uint32_t>(heuristics_[index],
                                           CUBLASLT_ALGO_CONFIG_STAGES_ID),
           algorithm_config<std::int32_t>(heuristics_[index],
                                          CUBLASLT_ALGO_CONFIG_SPLITK_NUM),
           algorithm_config<std::uint32_t>(
               heuristics_[index], CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME),
           algorithm_config<std::uint32_t>(
               heuristics_[index], CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING),
           algorithm_config<std::uint32_t>(
               heuristics_[index], CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION),
           heuristics_[index].wavesCount,
           heuristics_[index].workspaceSize,
           means[index], *minimum, *maximum});
    }
    if (!std::isfinite(means.front()))
      fail("default cuBLASLt Linear algorithm failed controlled tuning");
    const auto best = std::min_element(means.begin(), means.end());
    if (best == means.end() || !std::isfinite(*best))
      fail("cuBLASLt Linear tuning found no runnable algorithm");
    const auto best_index =
        static_cast<std::size_t>(std::distance(means.begin(), best));
    result.default_mean_milliseconds = means.front();
    result.observed_noise_milliseconds =
        std::max(spreads.front(), spreads[best_index]);
    const auto improvement = means.front() - means[best_index];
    auto selected = std::size_t{0U};
    if (best_index == 0U) {
      result.decision = "default_fastest";
    } else if (improvement <= result.observed_noise_milliseconds) {
      result.decision = "candidate_within_observed_noise";
    } else {
      selected = best_index;
      result.changed_from_default = true;
      result.decision = "candidate_faster_outside_observed_noise";
    }
    heuristic_ = heuristics_[selected];
    result.selected_heuristic_index = static_cast<std::uint32_t>(selected);
    result.selected_algorithm_id = algorithm_id(heuristic_);
    result.selected_mean_milliseconds = means[selected];
    if (persist_)
      save_persisted_algorithm(tuned_cache_file_, heuristic_,
                               cache_stats_ ? &cache_stats_->saved_tuned
                                            : nullptr);
    result.tuning_milliseconds =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - tuning_start)
            .count();
    return result;
  }

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              cublasLtHandle_t handle, const Workspace &workspace,
              CUstream stream) const {
    check(launch_candidate(op, buffers, handle, workspace, stream, heuristic_),
          "cublasLtMatmul Linear");
  }

  void select_heuristic(std::uint32_t rank) {
    if (rank >= heuristics_.size())
      fail("cuBLASLt Linear heuristic rank is out of range");
    if (heuristics_[rank].state != CUBLAS_STATUS_SUCCESS)
      fail("cuBLASLt Linear heuristic rank is not admitted");
    heuristic_ = heuristics_[rank];
  }

  std::string selected_algorithm_description() const {
    return "algorithm=" + std::to_string(algorithm_id(heuristic_)) +
           " tile=" +
           std::to_string(algorithm_config<std::uint32_t>(
               heuristic_, CUBLASLT_ALGO_CONFIG_TILE_ID)) +
           " stages=" +
           std::to_string(algorithm_config<std::uint32_t>(
               heuristic_, CUBLASLT_ALGO_CONFIG_STAGES_ID)) +
           " split_k=" +
           std::to_string(algorithm_config<std::int32_t>(
               heuristic_, CUBLASLT_ALGO_CONFIG_SPLITK_NUM)) +
           " reduction=" +
           std::to_string(algorithm_config<std::uint32_t>(
               heuristic_, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)) +
           " workspace=" + std::to_string(heuristic_.workspaceSize);
  }

private:
  cublasStatus_t launch_candidate(
      const ir::Operation &op, const DeviceBuffers &buffers,
      cublasLtHandle_t handle, const Workspace &workspace, CUstream stream,
      const cublasLtMatmulHeuristicResult_t &heuristic) const {
    constexpr float alpha = 1.0F;
    const float beta = bias_mode_ == ir::LinearBiasMode::Addmm ? 1.0F : 0.0F;
    const auto input_pointer = reinterpret_cast<const void *>(buffers.at(op.inputs[0]));
    const auto weight_pointer = reinterpret_cast<const void *>(buffers.at(op.inputs[1]));
    const auto output_pointer = reinterpret_cast<void *>(buffers.at(op.outputs[0]));
    if (has_bias_ && bias_mode_ == ir::LinearBiasMode::Epilogue) {
      const auto bias_pointer =
          reinterpret_cast<const void *>(buffers.at(op.inputs[2]));
      const auto status = cublasLtMatmulDescSetAttribute(
          operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
          sizeof(bias_pointer));
      if (status != CUBLAS_STATUS_SUCCESS)
        return status;
    }
    if (bias_mode_ == ir::LinearBiasMode::Addmm) {
      auto x = buffers.at(op.inputs[0]);
      auto weight = buffers.at(op.inputs[1]);
      auto bias = buffers.at(op.inputs[2]);
      auto output = buffers.at(op.outputs[0]);
      std::array<void *, 4> arguments{&x, &weight, &bias, &output};
      const auto prefill_status = counted_launch_kernel(
          addmm_prefill_, addmm_prefill_grid_, 1U, 1U, 256U, 1U, 1U, 0U,
          stream, arguments.data(), nullptr);
      if (prefill_status != CUDA_SUCCESS)
        return CUBLAS_STATUS_EXECUTION_FAILED;
    }
    const auto matrix_a_pointer = has_bias_ ? weight_pointer : input_pointer;
    const auto matrix_b_pointer = has_bias_ ? input_pointer : weight_pointer;
    const auto matrix_a = has_bias_ ? weight_ : input_;
    const auto matrix_b = has_bias_ ? input_ : weight_;
    count_cublaslt_matmul();
    return cublasLtMatmul(
        handle, operation_, &alpha, matrix_a_pointer, matrix_a,
        matrix_b_pointer, matrix_b, &beta, output_pointer, output_,
        output_pointer, output_, &heuristic.algo, workspace.data(),
        std::min(workspace.size(), workspace_limit_bytes_),
        reinterpret_cast<cudaStream_t>(stream));
  }

  static std::int32_t
  algorithm_id(const cublasLtMatmulHeuristicResult_t &heuristic) {
    std::int32_t result = -1;
    std::size_t written = 0U;
    check(cublasLtMatmulAlgoConfigGetAttribute(
              &heuristic.algo, CUBLASLT_ALGO_CONFIG_ID, &result,
              sizeof(result), &written),
          "cublasLtMatmulAlgoConfigGetAttribute algorithm id");
    return result;
  }

  template <typename T>
  static T algorithm_config(
      const cublasLtMatmulHeuristicResult_t &heuristic,
      cublasLtMatmulAlgoConfigAttributes_t attribute) {
    T result{};
    std::size_t written = 0U;
    check(cublasLtMatmulAlgoConfigGetAttribute(
              &heuristic.algo, attribute, &result, sizeof(result), &written),
          "cublasLtMatmulAlgoConfigGetAttribute");
    return result;
  }

  static bool same_algorithm_config(
      const cublasLtMatmulHeuristicResult_t &left,
      const cublasLtMatmulHeuristicResult_t &right) {
    return algorithm_id(left) == algorithm_id(right) &&
           algorithm_config<std::uint32_t>(
               left, CUBLASLT_ALGO_CONFIG_TILE_ID) ==
               algorithm_config<std::uint32_t>(
                   right, CUBLASLT_ALGO_CONFIG_TILE_ID) &&
           algorithm_config<std::uint32_t>(
               left, CUBLASLT_ALGO_CONFIG_STAGES_ID) ==
               algorithm_config<std::uint32_t>(
                   right, CUBLASLT_ALGO_CONFIG_STAGES_ID) &&
           algorithm_config<std::int32_t>(
               left, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) ==
               algorithm_config<std::int32_t>(
                   right, CUBLASLT_ALGO_CONFIG_SPLITK_NUM) &&
           algorithm_config<std::uint32_t>(
               left, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) ==
               algorithm_config<std::uint32_t>(
                   right, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING) &&
           algorithm_config<std::uint32_t>(
               left, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION) ==
               algorithm_config<std::uint32_t>(
                   right, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
  }

  struct PersistedAlgorithm {
    std::int32_t algorithm_id{};
    std::uint32_t tile_id{};
    std::uint32_t stages_id{};
    std::int32_t split_k{};
    std::uint32_t reduction_scheme{};
    std::uint32_t cta_swizzle{};
    std::uint32_t custom_option{};
  };

  static bool read_persisted_algorithm(const std::filesystem::path &path,
                                       PersistedAlgorithm &value) {
    std::ifstream input(path);
    if (!input)
      return false;
    input >> value.algorithm_id >> value.tile_id >> value.stages_id >>
        value.split_k >> value.reduction_scheme >> value.cta_swizzle >>
        value.custom_option;
    return static_cast<bool>(input);
  }

  static bool apply_persisted_configuration(cublasLtMatmulAlgo_t &algo,
                                            const PersistedAlgorithm &value) {
    const auto set = [&](cublasLtMatmulAlgoConfigAttributes_t attribute,
                         const void *data, std::size_t bytes) {
      return cublasLtMatmulAlgoConfigSetAttribute(&algo, attribute, data,
                                                  bytes) ==
             CUBLAS_STATUS_SUCCESS;
    };
    return set(CUBLASLT_ALGO_CONFIG_TILE_ID, &value.tile_id,
               sizeof(value.tile_id)) &&
           set(CUBLASLT_ALGO_CONFIG_STAGES_ID, &value.stages_id,
               sizeof(value.stages_id)) &&
           set(CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &value.split_k,
               sizeof(value.split_k)) &&
           set(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
               &value.reduction_scheme, sizeof(value.reduction_scheme)) &&
           set(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &value.cta_swizzle,
               sizeof(value.cta_swizzle)) &&
           set(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, &value.custom_option,
               sizeof(value.custom_option));
  }

  // Cache writes fail open: a filesystem problem must never fail a prepare.
  static void save_persisted_algorithm(
      const std::filesystem::path &path,
      const cublasLtMatmulHeuristicResult_t &heuristic,
      std::uint64_t *counter) {
    std::ofstream output(path, std::ios::trunc);
    if (!output)
      return;
    output << algorithm_id(heuristic) << ' '
           << algorithm_config<std::uint32_t>(heuristic,
                                              CUBLASLT_ALGO_CONFIG_TILE_ID)
           << ' '
           << algorithm_config<std::uint32_t>(heuristic,
                                              CUBLASLT_ALGO_CONFIG_STAGES_ID)
           << ' '
           << algorithm_config<std::int32_t>(heuristic,
                                             CUBLASLT_ALGO_CONFIG_SPLITK_NUM)
           << ' '
           << algorithm_config<std::uint32_t>(
                  heuristic, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME)
           << ' '
           << algorithm_config<std::uint32_t>(
                  heuristic, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING)
           << ' '
           << algorithm_config<std::uint32_t>(
                  heuristic, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION)
           << '\n';
    if (!output)
      return;
    if (counter)
      ++*counter;
  }

  cublasLtMatmulDesc_t operation_{};
  cublasLtMatrixLayout_t input_{};
  cublasLtMatrixLayout_t weight_{};
  cublasLtMatrixLayout_t output_{};
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics_;
  cublasLtMatmulHeuristicResult_t heuristic_{};
  bool has_bias_{};
  ir::LinearBiasMode bias_mode_{ir::LinearBiasMode::Epilogue};
  CUfunction addmm_prefill_{};
  unsigned addmm_prefill_grid_{};
  std::size_t workspace_limit_bytes_{};
  bool persist_{};
  LinearHeuristicCacheStats *cache_stats_{};
  std::filesystem::path tuned_cache_file_;
  std::filesystem::path passive_cache_file_;
};

struct H3Int8GemmKey {
  std::uint32_t rows{};
  std::uint32_t columns{};
  std::uint32_t contraction{};
  bool operator==(const H3Int8GemmKey &) const = default;
};

struct H3Int8GemmKeyHash {
  std::size_t operator()(const H3Int8GemmKey &key) const noexcept {
    auto result = static_cast<std::size_t>(key.rows);
    result = result * 0x9e3779b185ebca87ULL + key.columns;
    result = result * 0x9e3779b185ebca87ULL + key.contraction;
    return result;
  }
};

class H3Int8GemmPlan {
public:
  H3Int8GemmPlan(cublasLtHandle_t handle, H3Int8GemmKey key,
                 std::size_t workspace_bytes, std::uint32_t heuristic_rank)
      : key_(key), workspace_bytes_(workspace_bytes) {
    if (key.rows == 0U || key.columns == 0U || key.contraction == 0U)
      fail("H3 INT8 cuBLASLt plan requires positive dimensions");
    check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32I,
                                   CUDA_R_32I),
          "cublasLtMatmulDescCreate H3 INT8");
    constexpr cublasOperation_t trans_a = CUBLAS_OP_N;
    constexpr cublasOperation_t trans_b = CUBLAS_OP_T;
    check(cublasLtMatmulDescSetAttribute(operation_,
                                         CUBLASLT_MATMUL_DESC_TRANSA,
                                         &trans_a, sizeof(trans_a)),
          "cublasLtMatmulDescSetAttribute H3 INT8 trans A");
    check(cublasLtMatmulDescSetAttribute(operation_,
                                         CUBLASLT_MATMUL_DESC_TRANSB,
                                         &trans_b, sizeof(trans_b)),
          "cublasLtMatmulDescSetAttribute H3 INT8 trans B");
    check(cublasLtMatrixLayoutCreate(&activation_, CUDA_R_8I, key.rows,
                                     key.contraction, key.contraction),
          "cublasLtMatrixLayoutCreate H3 INT8 activation");
    check(cublasLtMatrixLayoutCreate(&weight_, CUDA_R_8I, key.columns,
                                     key.contraction, key.contraction),
          "cublasLtMatrixLayoutCreate H3 INT8 weight");
    check(cublasLtMatrixLayoutCreate(&output_, CUDA_R_32I, key.rows,
                                     key.columns, key.columns),
          "cublasLtMatrixLayoutCreate H3 INT8 output");
    constexpr cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
    for (auto layout : {activation_, weight_, output_})
      check(cublasLtMatrixLayoutSetAttribute(
                layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                sizeof(row_major)),
            "cublasLtMatrixLayoutSetAttribute H3 INT8 row major");

    cublasLtMatmulPreference_t preference{};
    check(cublasLtMatmulPreferenceCreate(&preference),
          "cublasLtMatmulPreferenceCreate H3 INT8");
    try {
      check(cublasLtMatmulPreferenceSetAttribute(
                preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_bytes_, sizeof(workspace_bytes_)),
            "cublasLtMatmulPreferenceSetAttribute H3 INT8 workspace");
      constexpr int requested = 32;
      std::array<cublasLtMatmulHeuristicResult_t, requested> candidates{};
      int returned = 0;
      check(cublasLtMatmulAlgoGetHeuristic(
                handle, operation_, activation_, weight_, output_, output_,
                preference, requested, candidates.data(), &returned),
            "cublasLtMatmulAlgoGetHeuristic H3 INT8");
      if (returned == 0)
        fail("cuBLASLt found no H3 INT8 projection algorithm");
      if (heuristic_rank >= static_cast<std::uint32_t>(returned))
        fail("H3 INT8 cuBLASLt heuristic rank " +
             std::to_string(heuristic_rank) + " is unavailable for shape " +
             std::to_string(key.rows) + "x" +
             std::to_string(key.columns) + "x" +
             std::to_string(key.contraction));
      heuristics_.assign(candidates.begin(), candidates.begin() + returned);
      heuristic_ = heuristics_.at(heuristic_rank);
    } catch (...) {
      (void)cublasLtMatmulPreferenceDestroy(preference);
      throw;
    }
    check(cublasLtMatmulPreferenceDestroy(preference),
          "cublasLtMatmulPreferenceDestroy H3 INT8");
  }

  ~H3Int8GemmPlan() {
    if (output_)
      (void)cublasLtMatrixLayoutDestroy(output_);
    if (weight_)
      (void)cublasLtMatrixLayoutDestroy(weight_);
    if (activation_)
      (void)cublasLtMatrixLayoutDestroy(activation_);
    if (operation_)
      (void)cublasLtMatmulDescDestroy(operation_);
  }
  H3Int8GemmPlan(const H3Int8GemmPlan &) = delete;
  H3Int8GemmPlan &operator=(const H3Int8GemmPlan &) = delete;

  void launch(cublasLtHandle_t handle, CUdeviceptr activation,
              CUdeviceptr weight, CUdeviceptr output,
              const Workspace &workspace, CUstream stream) const {
    constexpr std::int32_t alpha = 1;
    constexpr std::int32_t beta = 0;
    count_cublaslt_matmul();
    check(cublasLtMatmul(
              handle, operation_, &alpha,
              reinterpret_cast<const void *>(activation), activation_,
              reinterpret_cast<const void *>(weight), weight_, &beta,
              reinterpret_cast<const void *>(output), output_,
              reinterpret_cast<void *>(output), output_, &heuristic_.algo,
              workspace.data(),
              std::min(workspace.size(), workspace_bytes_),
              reinterpret_cast<cudaStream_t>(stream)),
          "cublasLtMatmul H3 INT8 projection");
  }

  std::pair<std::uint32_t, double>
  tune(cublasLtHandle_t handle, CUdeviceptr activation, CUdeviceptr weight,
       CUdeviceptr output, const Workspace &workspace, CUstream stream) {
    auto best_rank = std::uint32_t{0U};
    auto best_ms = std::numeric_limits<double>::infinity();
    for (std::size_t index = 0U; index < heuristics_.size(); ++index) {
      const auto launch_candidate = [&]() {
        constexpr std::int32_t alpha = 1;
        constexpr std::int32_t beta = 0;
        count_cublaslt_matmul();
        return cublasLtMatmul(
            handle, operation_, &alpha,
            reinterpret_cast<const void *>(activation), activation_,
            reinterpret_cast<const void *>(weight), weight_, &beta,
            reinterpret_cast<const void *>(output), output_,
            reinterpret_cast<void *>(output), output_,
            &heuristics_[index].algo, workspace.data(),
            std::min(workspace.size(), workspace_bytes_),
            reinterpret_cast<cudaStream_t>(stream));
      };
      if (launch_candidate() != CUBLAS_STATUS_SUCCESS)
        continue;
      Event start;
      Event stop;
      check(counted_event_record(start.get(), stream),
            "cuEventRecord H3 INT8 tuning start");
      constexpr std::uint32_t iterations = 2U;
      auto admitted = true;
      for (std::uint32_t iteration = 0U; iteration < iterations; ++iteration)
        if (launch_candidate() != CUBLAS_STATUS_SUCCESS) {
          admitted = false;
          break;
        }
      if (!admitted)
        continue;
      check(counted_event_record(stop.get(), stream),
            "cuEventRecord H3 INT8 tuning stop");
      check(counted_event_synchronize(stop.get()),
            "cuEventSynchronize H3 INT8 tuning");
      float elapsed = 0.0F;
      check(cuEventElapsedTime(&elapsed, start.get(), stop.get()),
            "cuEventElapsedTime H3 INT8 tuning");
      const auto mean = static_cast<double>(elapsed) / iterations;
      if (mean < best_ms) {
        best_ms = mean;
        best_rank = static_cast<std::uint32_t>(index);
      }
    }
    if (!std::isfinite(best_ms))
      fail("H3 INT8 cuBLASLt tuning found no runnable algorithm");
    heuristic_ = heuristics_.at(best_rank);
    return {best_rank, best_ms};
  }

private:
  H3Int8GemmKey key_{};
  std::size_t workspace_bytes_{};
  cublasLtMatmulDesc_t operation_{};
  cublasLtMatrixLayout_t activation_{};
  cublasLtMatrixLayout_t weight_{};
  cublasLtMatrixLayout_t output_{};
  cublasLtMatmulHeuristicResult_t heuristic_{};
  std::vector<cublasLtMatmulHeuristicResult_t> heuristics_;
};

class H3Int8GemmRegistry {
public:
  H3Int8GemmRegistry(cublasLtHandle_t handle, std::size_t workspace_bytes,
                     std::uint32_t heuristic_rank)
      : handle_(handle), workspace_bytes_(workspace_bytes),
        heuristic_rank_(heuristic_rank) {}

  void add(H3Int8GemmKey key) {
    if (!plans_.contains(key))
      plans_.emplace(key, std::make_unique<H3Int8GemmPlan>(
                              handle_, key, workspace_bytes_,
                              heuristic_rank_));
  }

  void launch(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
              CUdeviceptr output, const Workspace &workspace,
              CUstream stream) const {
    const auto found = plans_.find(key);
    if (found == plans_.end())
      fail("missing prepared H3 INT8 cuBLASLt shape");
    found->second->launch(handle_, activation, weight, output, workspace,
                          stream);
  }

  void tune(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
            CUdeviceptr output, const Workspace &workspace, CUstream stream) {
    if (!tuned_.insert(key).second)
      return;
    const auto found = plans_.find(key);
    if (found == plans_.end())
      fail("missing prepared H3 INT8 cuBLASLt tuning shape");
    const auto [rank, milliseconds] = found->second->tune(
        handle_, activation, weight, output, workspace, stream);
    std::cerr << "H3_INT8_CUBLASLT_TUNED rows=" << key.rows
              << " columns=" << key.columns
              << " contraction=" << key.contraction << " rank=" << rank
              << " mean_ms=" << milliseconds << '\n';
  }

  std::size_t size() const { return plans_.size(); }

private:
  cublasLtHandle_t handle_{};
  std::size_t workspace_bytes_{};
  std::uint32_t heuristic_rank_{};
  std::unordered_map<H3Int8GemmKey, std::unique_ptr<H3Int8GemmPlan>,
                     H3Int8GemmKeyHash>
      plans_;
  std::unordered_set<H3Int8GemmKey, H3Int8GemmKeyHash> tuned_;
};

class Fp8ScaledLinearPlan {
public:
  Fp8ScaledLinearPlan(const ir::Program &program, const ir::Operation &op,
                      cublasLtHandle_t handle, std::size_t workspace_bytes)
      : workspace_bytes_(workspace_bytes) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    const auto *output = program.tensor(op.outputs.at(0));
    if (!input || !weight || !output || input->dims.empty() ||
        weight->dims.size() != 2U)
      fail("invalid scaled FP8 Linear descriptors");
    const auto rows64 = input->element_count() / input->dims.back();
    const auto inner64 = input->dims.back();
    const auto columns64 = weight->dims.front();
    if (rows64 > static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()) ||
        inner64 > static_cast<std::uint64_t>(
                      std::numeric_limits<std::int64_t>::max()) ||
        columns64 > static_cast<std::uint64_t>(
                        std::numeric_limits<std::int64_t>::max()))
      fail("scaled FP8 Linear shape is not representable by cuBLASLt");
    const auto rows = static_cast<std::int64_t>(rows64);
    const auto inner = static_cast<std::int64_t>(inner64);
    const auto columns = static_cast<std::int64_t>(columns64);
    try {
      check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F),
            "cublasLtMatmulDescCreate scaled FP8");
      constexpr cublasOperation_t transpose_weight = CUBLAS_OP_T;
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_weight,
                sizeof(transpose_weight)),
            "cublasLtMatmulDescSetAttribute scaled FP8 trans B");
      check(cublasLtMatrixLayoutCreate(&input_, CUDA_R_8F_E4M3, rows, inner,
                                       inner),
            "cublasLtMatrixLayoutCreate scaled FP8 input");
      check(cublasLtMatrixLayoutCreate(&weight_, CUDA_R_8F_E4M3, columns,
                                       inner, inner),
            "cublasLtMatrixLayoutCreate scaled FP8 weight");
      check(cublasLtMatrixLayoutCreate(&output_, CUDA_R_16BF, rows, columns,
                                       columns),
            "cublasLtMatrixLayoutCreate scaled FP8 output");
      constexpr cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
      for (auto layout : {input_, weight_, output_})
        check(cublasLtMatrixLayoutSetAttribute(
                  layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                  sizeof(row_major)),
              "cublasLtMatrixLayoutSetAttribute scaled FP8 row major");
      cublasLtMatmulPreference_t preference{};
      check(cublasLtMatmulPreferenceCreate(&preference),
            "cublasLtMatmulPreferenceCreate scaled FP8");
      try {
        check(cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_bytes_, sizeof(workspace_bytes_)),
              "cublasLtMatmulPreferenceSetAttribute scaled FP8 workspace");
        constexpr int requested = 32;
        std::array<cublasLtMatmulHeuristicResult_t, requested> candidates{};
        int returned = 0;
        check(cublasLtMatmulAlgoGetHeuristic(
                  handle, operation_, input_, weight_, output_, output_,
                  preference, requested, candidates.data(), &returned),
              "cublasLtMatmulAlgoGetHeuristic scaled FP8");
        if (returned == 0)
          fail("cuBLASLt found no admitted scaled FP8 Linear algorithm");
        heuristic_ = candidates.front();
      } catch (...) {
        (void)cublasLtMatmulPreferenceDestroy(preference);
        throw;
      }
      check(cublasLtMatmulPreferenceDestroy(preference),
            "cublasLtMatmulPreferenceDestroy scaled FP8");
    } catch (...) {
      destroy();
      throw;
    }
  }

  ~Fp8ScaledLinearPlan() { destroy(); }

  Fp8ScaledLinearPlan(const Fp8ScaledLinearPlan &) = delete;
  Fp8ScaledLinearPlan &operator=(const Fp8ScaledLinearPlan &) = delete;

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              cublasLtHandle_t handle, const Workspace &workspace,
              CUstream stream) const {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    count_cublaslt_matmul();
    check(cublasLtMatmul(
              handle, operation_, &alpha,
              reinterpret_cast<const void *>(buffers.at(op.inputs.at(0))),
              input_,
              reinterpret_cast<const void *>(buffers.at(op.inputs.at(1))),
              weight_, &beta,
              reinterpret_cast<const void *>(buffers.at(op.outputs.at(0))),
              output_,
              reinterpret_cast<void *>(buffers.at(op.outputs.at(0))), output_,
              &heuristic_.algo, workspace.data(),
              std::min(workspace.size(), workspace_bytes_),
              reinterpret_cast<cudaStream_t>(stream)),
          "cublasLtMatmul scaled FP8 Linear");
  }

private:
  void destroy() {
    if (output_)
      (void)cublasLtMatrixLayoutDestroy(output_);
    if (weight_)
      (void)cublasLtMatrixLayoutDestroy(weight_);
    if (input_)
      (void)cublasLtMatrixLayoutDestroy(input_);
    if (operation_)
      (void)cublasLtMatmulDescDestroy(operation_);
    output_ = nullptr;
    weight_ = nullptr;
    input_ = nullptr;
    operation_ = nullptr;
  }

  std::size_t workspace_bytes_{};
  cublasLtMatmulDesc_t operation_{};
  cublasLtMatrixLayout_t input_{};
  cublasLtMatrixLayout_t weight_{};
  cublasLtMatrixLayout_t output_{};
  cublasLtMatmulHeuristicResult_t heuristic_{};
};

#if DIF_HAS_CUBLASLT_BLOCK_SCALE
class Fp8BlockScaledLinearPlan {
public:
  Fp8BlockScaledLinearPlan(const ir::Program &program,
                           const ir::Operation &op,
                           const DeviceBuffers &buffers,
                           cublasLtHandle_t handle,
                           std::size_t workspace_bytes)
      : workspace_bytes_(workspace_bytes) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    const auto *output = program.tensor(op.outputs.at(0));
    if (!input || !weight || !output || input->dims.empty() ||
        weight->dims.size() != 2U)
      fail("invalid MXFP8 Linear descriptors");
    const auto rows = static_cast<std::int64_t>(
        input->element_count() / input->dims.back());
    const auto inner = static_cast<std::int64_t>(input->dims.back());
    const auto columns = static_cast<std::int64_t>(weight->dims.front());
    try {
      check(cublasLtMatmulDescCreate(&operation_, CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F),
            "cublasLtMatmulDescCreate MXFP8");
      constexpr cublasOperation_t transpose_weight = CUBLAS_OP_T;
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_TRANSB, &transpose_weight,
                sizeof(transpose_weight)),
            "cublasLtMatmulDescSetAttribute MXFP8 trans B");
      constexpr cublasLtMatmulMatrixScale_t block_scale =
          CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &block_scale,
                sizeof(block_scale)),
            "cublasLtMatmulDescSetAttribute MXFP8 A scale mode");
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &block_scale,
                sizeof(block_scale)),
            "cublasLtMatmulDescSetAttribute MXFP8 B scale mode");
      const auto *input_scales =
          reinterpret_cast<const void *>(buffers.at(op.inputs.at(2)));
      const auto *weight_scales =
          reinterpret_cast<const void *>(buffers.at(op.inputs.at(3)));
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                &input_scales, sizeof(input_scales)),
            "cublasLtMatmulDescSetAttribute MXFP8 A scale pointer");
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                &weight_scales, sizeof(weight_scales)),
            "cublasLtMatmulDescSetAttribute MXFP8 B scale pointer");
      check(cublasLtMatrixLayoutCreate(&input_, CUDA_R_8F_E4M3, rows, inner,
                                       inner),
            "cublasLtMatrixLayoutCreate MXFP8 input");
      check(cublasLtMatrixLayoutCreate(&weight_, CUDA_R_8F_E4M3, columns,
                                       inner, inner),
            "cublasLtMatrixLayoutCreate MXFP8 weight");
      check(cublasLtMatrixLayoutCreate(&output_, CUDA_R_16BF, rows, columns,
                                       columns),
            "cublasLtMatrixLayoutCreate MXFP8 output");
      constexpr cublasLtOrder_t row_major = CUBLASLT_ORDER_ROW;
      for (auto layout : {input_, weight_, output_})
        check(cublasLtMatrixLayoutSetAttribute(
                  layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &row_major,
                  sizeof(row_major)),
              "cublasLtMatrixLayoutSetAttribute MXFP8 row major");
      cublasLtMatmulPreference_t preference{};
      check(cublasLtMatmulPreferenceCreate(&preference),
            "cublasLtMatmulPreferenceCreate MXFP8");
      try {
        check(cublasLtMatmulPreferenceSetAttribute(
                  preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                  &workspace_bytes_, sizeof(workspace_bytes_)),
              "cublasLtMatmulPreferenceSetAttribute MXFP8 workspace");
        constexpr int requested = 32;
        std::array<cublasLtMatmulHeuristicResult_t, requested> candidates{};
        int returned = 0;
        check(cublasLtMatmulAlgoGetHeuristic(
                  handle, operation_, input_, weight_, output_, output_,
                  preference, requested, candidates.data(), &returned),
              "cublasLtMatmulAlgoGetHeuristic MXFP8");
        if (returned == 0)
          fail("cuBLASLt found no admitted MXFP8 Linear algorithm");
        heuristic_ = candidates.front();
      } catch (...) {
        (void)cublasLtMatmulPreferenceDestroy(preference);
        throw;
      }
      check(cublasLtMatmulPreferenceDestroy(preference),
            "cublasLtMatmulPreferenceDestroy MXFP8");
    } catch (...) {
      destroy();
      throw;
    }
  }

  ~Fp8BlockScaledLinearPlan() { destroy(); }

  Fp8BlockScaledLinearPlan(const Fp8BlockScaledLinearPlan &) = delete;
  Fp8BlockScaledLinearPlan &
  operator=(const Fp8BlockScaledLinearPlan &) = delete;

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              cublasLtHandle_t handle, const Workspace &workspace,
              CUstream stream) const {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    count_cublaslt_matmul();
    check(cublasLtMatmul(
              handle, operation_, &alpha,
              reinterpret_cast<const void *>(buffers.at(op.inputs.at(0))),
              input_,
              reinterpret_cast<const void *>(buffers.at(op.inputs.at(1))),
              weight_, &beta,
              reinterpret_cast<const void *>(buffers.at(op.outputs.at(0))),
              output_,
              reinterpret_cast<void *>(buffers.at(op.outputs.at(0))), output_,
              &heuristic_.algo, workspace.data(),
              std::min(workspace.size(), workspace_bytes_),
              reinterpret_cast<cudaStream_t>(stream)),
          "cublasLtMatmul MXFP8 Linear");
  }

private:
  void destroy() {
    if (output_)
      (void)cublasLtMatrixLayoutDestroy(output_);
    if (weight_)
      (void)cublasLtMatrixLayoutDestroy(weight_);
    if (input_)
      (void)cublasLtMatrixLayoutDestroy(input_);
    if (operation_)
      (void)cublasLtMatmulDescDestroy(operation_);
    output_ = nullptr;
    weight_ = nullptr;
    input_ = nullptr;
    operation_ = nullptr;
  }

  std::size_t workspace_bytes_{};
  cublasLtMatmulDesc_t operation_{};
  cublasLtMatrixLayout_t input_{};
  cublasLtMatrixLayout_t weight_{};
  cublasLtMatrixLayout_t output_{};
  cublasLtMatmulHeuristicResult_t heuristic_{};
};
#endif // DIF_HAS_CUBLASLT_BLOCK_SCALE

#if DIF_HAS_CUTLASS
struct CutlassInt8ScaledGemmDeleter {
  void operator()(CutlassInt8ScaledGemmHandle *handle) const {
    destroy_cutlass_int8_scaled_gemm(handle);
  }
};

struct CutlassInt8WeightGemmDeleter {
  void operator()(CutlassInt8WeightGemmHandle *handle) const {
    destroy_cutlass_int8_weight_gemm(handle);
  }
};

class H3Int8ScaledGemmPlan {
public:
  H3Int8ScaledGemmPlan(H3Int8GemmKey key, CUdeviceptr activation,
                      CUdeviceptr weight, CUdeviceptr row_scale,
                      CUdeviceptr column_scale, CUdeviceptr output,
                      CUstream stream)
      : key_(key) {
    std::array<char, 512> error{};
    handle_.reset(create_cutlass_int8_scaled_gemm(
        key.rows, key.columns, key.contraction,
        static_cast<std::uintptr_t>(activation),
        static_cast<std::uintptr_t>(weight),
        static_cast<std::uintptr_t>(row_scale),
        static_cast<std::uintptr_t>(column_scale),
        static_cast<std::uintptr_t>(output),
        reinterpret_cast<std::uintptr_t>(stream), error.data(), error.size()));
    if (!handle_)
      fail(std::string("CUTLASS H3 scaled INT8 plan creation failed: ") +
           error.data());
  }

  void launch(CUdeviceptr activation, CUdeviceptr weight,
              CUdeviceptr row_scale, CUdeviceptr column_scale,
              CUdeviceptr output, CUstream stream) const {
    std::array<char, 512> error{};
    count_cutlass_launch();
    if (!launch_cutlass_int8_scaled_gemm(
            handle_.get(), static_cast<std::uintptr_t>(activation),
            static_cast<std::uintptr_t>(weight),
            static_cast<std::uintptr_t>(row_scale),
            static_cast<std::uintptr_t>(column_scale),
            static_cast<std::uintptr_t>(output),
            reinterpret_cast<std::uintptr_t>(stream), error.data(),
            error.size()))
      fail(std::string("CUTLASS H3 scaled INT8 launch failed: ") +
           error.data());
  }

private:
  H3Int8GemmKey key_{};
  std::unique_ptr<CutlassInt8ScaledGemmHandle,
                  CutlassInt8ScaledGemmDeleter>
      handle_;
};

struct H3Int8ScaledGemmKey {
  H3Int8GemmKey gemm{};
  CUdeviceptr activation{};
  CUdeviceptr weight{};
  CUdeviceptr row_scale{};
  CUdeviceptr column_scale{};
  CUdeviceptr output{};

  bool operator==(const H3Int8ScaledGemmKey &) const = default;
};

struct H3Int8ScaledGemmKeyHash {
  std::size_t operator()(const H3Int8ScaledGemmKey &key) const noexcept {
    auto result = H3Int8GemmKeyHash{}(key.gemm);
    const auto mix = [&result](CUdeviceptr pointer) {
      result ^= std::hash<CUdeviceptr>{}(pointer) + 0x9e3779b97f4a7c15ULL +
                (result << 6U) + (result >> 2U);
    };
    mix(key.activation);
    mix(key.weight);
    mix(key.row_scale);
    mix(key.column_scale);
    mix(key.output);
    return result;
  }
};

class H3Int8ScaledGemmRegistry {
public:
  void add(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
           CUdeviceptr row_scale, CUdeviceptr column_scale,
           CUdeviceptr output, CUstream stream) {
    const auto prepared = H3Int8ScaledGemmKey{
        key, activation, weight, row_scale, column_scale, output};
    if (!plans_.contains(prepared))
      plans_.emplace(prepared, std::make_unique<H3Int8ScaledGemmPlan>(
                                   key, activation, weight, row_scale,
                                   column_scale, output, stream));
  }

  void launch(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
              CUdeviceptr row_scale, CUdeviceptr column_scale,
              CUdeviceptr output, CUstream stream) const {
    const auto prepared = H3Int8ScaledGemmKey{
        key, activation, weight, row_scale, column_scale, output};
    const auto found = plans_.find(prepared);
    if (found == plans_.end())
      fail("missing pointer-stable H3 CUTLASS scaled INT8 plan");
    found->second->launch(activation, weight, row_scale, column_scale, output,
                          stream);
  }

  std::size_t size() const { return plans_.size(); }

private:
  std::unordered_map<H3Int8ScaledGemmKey,
                     std::unique_ptr<H3Int8ScaledGemmPlan>,
                     H3Int8ScaledGemmKeyHash>
      plans_;
};

struct CutlassInt8ScaledF16GemmDeleter {
  void operator()(CutlassInt8ScaledF16GemmHandle *handle) const {
    destroy_cutlass_int8_scaled_f16_gemm(handle);
  }
};

struct Int8ScaledF16GemmKey {
  H3Int8GemmKey gemm{};
  CUdeviceptr activation{};
  CUdeviceptr weight{};
  CUdeviceptr row_scale{};
  CUdeviceptr column_scale{};
  CUdeviceptr bias{};
  CUdeviceptr output{};

  bool operator==(const Int8ScaledF16GemmKey &) const = default;
};

struct Int8ScaledF16GemmKeyHash {
  std::size_t operator()(const Int8ScaledF16GemmKey &key) const noexcept {
    auto result = H3Int8GemmKeyHash{}(key.gemm);
    const auto mix = [&result](CUdeviceptr pointer) {
      result ^= std::hash<CUdeviceptr>{}(pointer) + 0x9e3779b97f4a7c15ULL +
                (result << 6U) + (result >> 2U);
    };
    mix(key.activation);
    mix(key.weight);
    mix(key.row_scale);
    mix(key.column_scale);
    mix(key.bias);
    mix(key.output);
    return result;
  }
};

class Int8ScaledF16GemmPlan {
public:
  Int8ScaledF16GemmPlan(H3Int8GemmKey key, CUdeviceptr activation,
                       CUdeviceptr weight, CUdeviceptr row_scale,
                       CUdeviceptr column_scale, CUdeviceptr bias,
                       CUdeviceptr output, CUstream stream)
      : key_(key) {
    std::array<char, 512> error{};
    handle_.reset(create_cutlass_int8_scaled_f16_gemm(
        key.rows, key.columns, key.contraction,
        static_cast<std::uintptr_t>(activation),
        static_cast<std::uintptr_t>(weight),
        static_cast<std::uintptr_t>(row_scale),
        static_cast<std::uintptr_t>(column_scale),
        static_cast<std::uintptr_t>(bias),
        static_cast<std::uintptr_t>(output),
        reinterpret_cast<std::uintptr_t>(stream), error.data(), error.size()));
    if (!handle_)
      fail(std::string("CUTLASS scaled INT8 F16 plan creation failed: ") +
           error.data());
  }

  void launch(CUdeviceptr activation, CUdeviceptr weight,
              CUdeviceptr row_scale, CUdeviceptr column_scale,
              CUdeviceptr bias, CUdeviceptr output, CUstream stream) const {
    std::array<char, 512> error{};
    count_cutlass_launch();
    if (!launch_cutlass_int8_scaled_f16_gemm(
            handle_.get(), static_cast<std::uintptr_t>(activation),
            static_cast<std::uintptr_t>(weight),
            static_cast<std::uintptr_t>(row_scale),
            static_cast<std::uintptr_t>(column_scale),
            static_cast<std::uintptr_t>(bias),
            static_cast<std::uintptr_t>(output),
            reinterpret_cast<std::uintptr_t>(stream), error.data(),
            error.size()))
      fail(std::string("CUTLASS scaled INT8 F16 launch failed: ") +
           error.data());
  }

private:
  H3Int8GemmKey key_{};
  std::unique_ptr<CutlassInt8ScaledF16GemmHandle,
                  CutlassInt8ScaledF16GemmDeleter>
      handle_;
};

class Int8ScaledF16GemmRegistry {
public:
  void add(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
           CUdeviceptr row_scale, CUdeviceptr column_scale,
           CUdeviceptr bias, CUdeviceptr output, CUstream stream) {
    const auto prepared = Int8ScaledF16GemmKey{
        key, activation, weight, row_scale, column_scale, bias, output};
    if (!plans_.contains(prepared))
      plans_.emplace(prepared, std::make_unique<Int8ScaledF16GemmPlan>(
                                   key, activation, weight, row_scale,
                                   column_scale, bias, output, stream));
  }

  void launch(H3Int8GemmKey key, CUdeviceptr activation, CUdeviceptr weight,
              CUdeviceptr row_scale, CUdeviceptr column_scale,
              CUdeviceptr bias, CUdeviceptr output, CUstream stream) const {
    const auto prepared = Int8ScaledF16GemmKey{
        key, activation, weight, row_scale, column_scale, bias, output};
    const auto found = plans_.find(prepared);
    if (found == plans_.end())
      fail("missing pointer-stable CUTLASS scaled INT8 F16 plan");
    found->second->launch(activation, weight, row_scale, column_scale, bias,
                          output, stream);
  }

private:
  std::unordered_map<Int8ScaledF16GemmKey,
                     std::unique_ptr<Int8ScaledF16GemmPlan>,
                     Int8ScaledF16GemmKeyHash>
      plans_;
};

class Int8ScaledLinearPlan {
public:
  Int8ScaledLinearPlan(const ir::Program &program, const ir::Operation &op,
                       const DeviceBuffers &buffers, CUstream stream)
      : operation_id_(op.id) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    if (!input || !weight)
      fail("scaled INT8 Linear plan references a missing tensor");
    const auto rows = input->element_count() / input->dims.back();
    const auto columns = weight->dims.front();
    const auto inner = input->dims.back();
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        columns > std::numeric_limits<std::uint32_t>::max() ||
        inner > std::numeric_limits<std::uint32_t>::max())
      fail("scaled INT8 Linear shape exceeds the CUTLASS primitive ABI");
    std::array<char, 512> error{};
    handle_.reset(create_cutlass_int8_scaled_gemm(
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(columns),
        static_cast<std::uint32_t>(inner),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(0))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(1))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(2))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(3))),
        static_cast<std::uintptr_t>(buffers.at(op.outputs.at(0))),
        reinterpret_cast<std::uintptr_t>(stream), error.data(), error.size()));
    if (!handle_)
      fail(std::string("CUTLASS scaled INT8 Linear plan creation failed: ") +
           error.data());
  }

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              CUstream stream) const {
    std::array<char, 512> error{};
    count_cutlass_launch();
    if (!launch_cutlass_int8_scaled_gemm(
            handle_.get(),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(1))),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(2))),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(3))),
            static_cast<std::uintptr_t>(buffers.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(stream), error.data(),
            error.size()))
      fail(std::string("CUTLASS scaled INT8 Linear launch failed: ") +
           error.data());
  }

  std::uint32_t operation_id() const { return operation_id_; }

private:
  std::uint32_t operation_id_{};
  std::unique_ptr<CutlassInt8ScaledGemmHandle,
                  CutlassInt8ScaledGemmDeleter>
      handle_;
};

class Int8WeightLinearPlan {
public:
  Int8WeightLinearPlan(const ir::Program &program, const ir::Operation &op,
                       const DeviceBuffers &buffers, CUstream stream)
      : operation_id_(op.id) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    if (!input || !weight)
      fail("INT8 weight Linear plan references a missing tensor");
    const auto rows = input->element_count() / input->dims.back();
    const auto columns = weight->dims.front();
    const auto inner = input->dims.back();
    if (rows > std::numeric_limits<std::uint32_t>::max() ||
        columns > std::numeric_limits<std::uint32_t>::max() ||
        inner > std::numeric_limits<std::uint32_t>::max())
      fail("INT8 weight Linear shape exceeds the CUTLASS primitive ABI");
    std::array<char, 512> error{};
    handle_.reset(create_cutlass_int8_weight_gemm(
        static_cast<std::uint32_t>(rows),
        static_cast<std::uint32_t>(columns),
        static_cast<std::uint32_t>(inner),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(0))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(1))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(2))),
        static_cast<std::uintptr_t>(buffers.at(op.outputs.at(0))),
        reinterpret_cast<std::uintptr_t>(stream), error.data(), error.size()));
    if (!handle_)
      fail(std::string("CUTLASS INT8 weight Linear plan creation failed: ") +
           error.data());
  }

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              CUstream stream) const {
    std::array<char, 512> error{};
    count_cutlass_launch();
    if (!launch_cutlass_int8_weight_gemm(
            handle_.get(),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(1))),
            static_cast<std::uintptr_t>(buffers.at(op.inputs.at(2))),
            static_cast<std::uintptr_t>(buffers.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(stream), error.data(),
            error.size()))
      fail(std::string("CUTLASS INT8 weight Linear launch failed: ") +
           error.data());
  }

private:
  std::uint32_t operation_id_{};
  std::unique_ptr<CutlassInt8WeightGemmHandle, CutlassInt8WeightGemmDeleter>
      handle_;
};

class CutlassLinearPlan {
public:
  CutlassLinearPlan(const ir::Program &program, const ir::Operation &op,
                    const DeviceBuffers &buffers, std::uint32_t schedule,
                    CUstream stream)
      : operation_id_(op.id), schedule_(schedule) {
    const auto *input = program.tensor(op.inputs.at(0));
    const auto *weight = program.tensor(op.inputs.at(1));
    const auto *output = program.tensor(op.outputs.at(0));
    if (!input || !weight || !output || op.inputs.size() != 2U ||
        input->dtype != ir::DType::BF16 || weight->dtype != ir::DType::BF16 ||
        output->dtype != ir::DType::BF16 || input->dims.empty() ||
        weight->dims.size() != 2U)
      fail("CUTLASS GEMM requires an unbiased BF16 Linear");
    const auto m = input->dims.front();
    const auto k = input->element_count() / m;
    const auto n = weight->dims.front();
    if (m > std::numeric_limits<std::uint32_t>::max() ||
        n > std::numeric_limits<std::uint32_t>::max() ||
        k > std::numeric_limits<std::uint32_t>::max())
      fail("CUTLASS GEMM shape exceeds the primitive ABI");
    std::array<char, 512> error{};
    handle_ = create_cutlass_gemm(
        schedule, static_cast<std::uint32_t>(m),
        static_cast<std::uint32_t>(n), static_cast<std::uint32_t>(k),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(0))),
        static_cast<std::uintptr_t>(buffers.at(op.inputs.at(1))),
        static_cast<std::uintptr_t>(buffers.at(op.outputs.at(0))),
        reinterpret_cast<std::uintptr_t>(stream), error.data(), error.size());
    if (!handle_)
      fail(std::string("CUTLASS GEMM plan creation failed: ") + error.data());
    resources_ = cutlass_gemm_resources(handle_);
  }

  ~CutlassLinearPlan() { destroy_cutlass_gemm(handle_); }
  CutlassLinearPlan(const CutlassLinearPlan &) = delete;
  CutlassLinearPlan &operator=(const CutlassLinearPlan &) = delete;

  void launch(CUstream stream) const {
    std::array<char, 512> error{};
    if (!launch_cutlass_gemm(handle_, reinterpret_cast<std::uintptr_t>(stream),
                             error.data(), error.size()))
      fail(std::string("CUTLASS GEMM launch failed: ") + error.data());
  }

  GemmPrimitiveResult result() const {
    return {operation_id_,
            schedule_,
            resources_.name ? resources_.name : "unknown_cutlass_schedule",
            resources_.threadblock_m,
            resources_.threadblock_n,
            resources_.threadblock_k,
            resources_.warp_m,
            resources_.warp_n,
            resources_.warp_k,
            resources_.stages,
            resources_.threads_per_block,
            resources_.registers_per_thread,
            resources_.static_shared_bytes,
            resources_.dynamic_shared_bytes,
            resources_.maximum_dynamic_shared_bytes};
  }

private:
  std::uint32_t operation_id_{};
  std::uint32_t schedule_{};
  CutlassGemmHandle *handle_{};
  CutlassGemmResources resources_{};
};
#endif

struct FusedLinearSwiGluPlan {
  std::uint32_t linear_operation{};
  std::uint32_t swiglu_operation{};
  std::uint32_t input_tensor{};
  std::uint32_t weight_tensor{};
  std::uint32_t intermediate_tensor{};
  std::uint32_t output_tensor{};
  std::uint64_t rows{};
  std::uint64_t inner{};
  std::uint64_t width{};
  bool gate_first{};
  std::string entrypoint;
  CUfunction function{};
};

std::vector<FusedLinearSwiGluPlan> find_fused_linear_swiglu_plans(
    const ir::Program &program, const RunOptions &options, int major) {
  std::vector<FusedLinearSwiGluPlan> result;
  std::unordered_set<std::uint32_t> requested;
  for (const auto operation_id : options.fuse_linear_swiglu_operations) {
    if (!requested.insert(operation_id).second)
      continue;
    const auto linear = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const ir::Operation &operation) {
          return operation.id == operation_id;
        });
    if (linear == program.operations.end() ||
        linear->opcode != ir::Opcode::Linear || linear->inputs.size() != 2U ||
        linear->u64(ir::AttrKey::Implementation, 1U) != 1U)
      fail("fused Linear->SwiGlu requires an unbiased native Linear id: " +
           std::to_string(operation_id));
    const auto intermediate = linear->outputs.at(0);
    const ir::Operation *swiglu = nullptr;
    std::size_t uses = 0U;
    for (const auto &operation : program.operations) {
      for (const auto input : operation.inputs) {
        if (input != intermediate)
          continue;
        ++uses;
        if (operation.opcode == ir::Opcode::SwiGlu)
          swiglu = &operation;
      }
    }
    if (uses != 1U || !swiglu || swiglu->inputs.at(0) != intermediate)
      fail("fused Linear->SwiGlu requires one exclusive SwiGlu consumer");
    const auto *input = program.tensor(linear->inputs.at(0));
    const auto *weight = program.tensor(linear->inputs.at(1));
    const auto *projected = program.tensor(intermediate);
    const auto *output = program.tensor(swiglu->outputs.at(0));
    if (!input || !weight || !projected || !output ||
        input->dtype != ir::DType::BF16 || weight->dtype != ir::DType::BF16 ||
        projected->dtype != ir::DType::BF16 ||
        output->dtype != ir::DType::BF16 || input->dims.empty() ||
        weight->dims.size() != 2U || output->dims.empty())
      fail("fused Linear->SwiGlu currently requires BF16 tensors");
    const auto rows = input->dims[0];
    const auto inner = input->element_count() / rows;
    const auto width = output->element_count() / output->dims[0];
    if (weight->dims[0] != 2U * width || weight->dims[1] != inner ||
        output->dims[0] != rows)
      fail("fused Linear->SwiGlu shapes are inconsistent");
    if (projected->element_count() != rows * 2U * width ||
        output->element_count() != rows * width)
      fail("fused Linear->SwiGlu flattened shapes are inconsistent");
    if (major < 8)
      fail("fused BF16 Linear->SwiGlu requires an Ampere-or-newer CUDA GPU");
    result.push_back(
        {linear->id, swiglu->id, linear->inputs.at(0), linear->inputs.at(1),
         intermediate, swiglu->outputs.at(0), rows, inner, width,
         swiglu->boolean(ir::AttrKey::GateFirst, false),
         "dif_fused_linear_swiglu_" + std::to_string(linear->id), nullptr});
  }
  return result;
}

std::string fused_linear_swiglu_source(
    const std::vector<FusedLinearSwiGluPlan> &plans) {
  if (plans.empty())
    return {};
  std::ostringstream out;
  out << "#include <cuda_bf16.h>\n#include <mma.h>\n";
  for (const auto &plan : plans) {
    const auto gate_offset = plan.gate_first ? 0U : plan.width;
    const auto value_offset = plan.gate_first ? plan.width : 0U;
    out << "extern \"C\" __global__ void " << plan.entrypoint
        << "(const dif_bf16* x,const dif_bf16* weight,dif_bf16* y){\n"
        << "  using namespace nvcuda;\n"
           "  __shared__ __nv_bfloat16 tile_a[64*16];\n"
           "  __shared__ __nv_bfloat16 tile_gate_weight[16*64];\n"
           "  __shared__ __nv_bfloat16 tile_value_weight[16*64];\n"
           "  __shared__ float tile_gate[64*64];\n"
           "  __shared__ float tile_value[64*64];\n"
           "  unsigned warp=threadIdx.x>>5U,warp_n=warp&3U;\n"
           "  unsigned warp_m0=warp>>2U,warp_m1=warp_m0+2U;\n"
           "  unsigned long long row0=(unsigned long long)blockIdx.y*64ULL;\n"
           "  unsigned long long col0=(unsigned long long)blockIdx.x*64ULL;\n"
           "  wmma::fragment<wmma::accumulator,16,16,16,float> gate0,gate1,value0,value1;\n"
           "  wmma::fill_fragment(gate0,0.0f);wmma::fill_fragment(gate1,0.0f);\n"
           "  wmma::fill_fragment(value0,0.0f);wmma::fill_fragment(value1,0.0f);\n"
           "  const __nv_bfloat16* xb=(const __nv_bfloat16*)x;\n"
           "  const __nv_bfloat16* wb=(const __nv_bfloat16*)weight;\n"
           "  for(unsigned long long k0=0ULL;k0<"
        << plan.inner
        << "ULL;k0+=16ULL){\n"
           "    for(unsigned i=threadIdx.x;i<64U*16U;i+=blockDim.x){\n"
           "      unsigned r=i>>4U,c=i&15U;unsigned long long gr=row0+r,gk=k0+c;\n"
           "      tile_a[i]=(gr<"
        << plan.rows << "ULL&&gk<" << plan.inner
        << "ULL)?xb[gr*" << plan.inner
        << "ULL+gk]:__float2bfloat16_rn(0.0f);\n"
           "    }\n"
           "    for(unsigned i=threadIdx.x;i<16U*64U;i+=blockDim.x){\n"
           "      unsigned kr=i&15U,n=i>>4U;unsigned long long gn=col0+n,gk=k0+kr;\n"
           "      tile_gate_weight[i]=(gn<"
        << plan.width << "ULL&&gk<" << plan.inner << "ULL)?wb[(gn+"
        << gate_offset << "ULL)*" << plan.inner
        << "ULL+gk]:__float2bfloat16_rn(0.0f);\n"
           "      tile_value_weight[i]=(gn<"
        << plan.width << "ULL&&gk<" << plan.inner << "ULL)?wb[(gn+"
        << value_offset << "ULL)*" << plan.inner
        << "ULL+gk]:__float2bfloat16_rn(0.0f);\n"
           "    }\n"
           "    __syncthreads();\n"
           "    wmma::fragment<wmma::matrix_a,16,16,16,__nv_bfloat16,wmma::row_major> a0,a1;\n"
           "    wmma::fragment<wmma::matrix_b,16,16,16,__nv_bfloat16,wmma::col_major> bg,bv;\n"
           "    wmma::load_matrix_sync(a0,tile_a+warp_m0*16U*16U,16);\n"
           "    wmma::load_matrix_sync(a1,tile_a+warp_m1*16U*16U,16);\n"
           "    wmma::load_matrix_sync(bg,tile_gate_weight+warp_n*16U*16U,16);\n"
           "    wmma::load_matrix_sync(bv,tile_value_weight+warp_n*16U*16U,16);\n"
           "    wmma::mma_sync(gate0,a0,bg,gate0);wmma::mma_sync(gate1,a1,bg,gate1);\n"
           "    wmma::mma_sync(value0,a0,bv,value0);wmma::mma_sync(value1,a1,bv,value1);\n"
           "    __syncthreads();\n"
           "  }\n"
           "  float* g0=tile_gate+warp_m0*16U*64U+warp_n*16U;\n"
           "  float* g1=tile_gate+warp_m1*16U*64U+warp_n*16U;\n"
           "  float* v0=tile_value+warp_m0*16U*64U+warp_n*16U;\n"
           "  float* v1=tile_value+warp_m1*16U*64U+warp_n*16U;\n"
           "  wmma::store_matrix_sync(g0,gate0,64,wmma::mem_row_major);\n"
           "  wmma::store_matrix_sync(g1,gate1,64,wmma::mem_row_major);\n"
           "  wmma::store_matrix_sync(v0,value0,64,wmma::mem_row_major);\n"
           "  wmma::store_matrix_sync(v1,value1,64,wmma::mem_row_major);\n"
           "  __syncthreads();\n"
           "  for(unsigned i=threadIdx.x;i<64U*64U;i+=blockDim.x){\n"
           "    unsigned r=i>>6U,c=i&63U;unsigned long long gr=row0+r,gc=col0+c;\n"
           "    if(gr<"
        << plan.rows << "ULL&&gc<" << plan.width
        << "ULL){float gate=dif_round_bf16(tile_gate[i]);"
           "float value=dif_round_bf16(tile_value[i]);"
           "float activated=dif_round_bf16(dif_silu(gate));"
           "dif_store_bf16(y,gr*"
        << plan.width << "ULL+gc,value*activated);}\n"
           "  }\n"
           "}\n";
  }
  return out.str();
}

void launch_fused_linear_swiglu(const FusedLinearSwiGluPlan &plan,
                                const DeviceBuffers &buffers,
                                CUstream stream) {
  auto input = buffers.at(plan.input_tensor);
  auto weight = buffers.at(plan.weight_tensor);
  auto output = buffers.at(plan.output_tensor);
  std::array<void *, 3> arguments = {&input, &weight, &output};
  const auto grid_x = static_cast<unsigned>((plan.width + 63U) / 64U);
  const auto grid_y = static_cast<unsigned>((plan.rows + 63U) / 64U);
  check(counted_launch_kernel(plan.function, grid_x, grid_y, 1U, 256U, 1U, 1U, 0U,
                       stream, arguments.data(), nullptr),
        "cuLaunchKernel fused Linear->SwiGlu");
}

struct AbsorbedLinearBiasPlan {
  std::uint32_t linear_operation{};
  std::uint32_t bias_operation{};
  std::uint32_t intermediate_tensor{};
  std::uint32_t output_tensor{};
  std::uint64_t eliminated_intermediate_bytes{};
  // The Linear operation re-expressed in the biased form the LinearPlan
  // builds and launches: the Linear's input and weight plus the BiasAdd's
  // bias, writing the BiasAdd's output.
  ir::Operation launch_operation;
};

// Absorb an unbiased Linear's exclusive, immediately-following BiasAdd into
// the cuBLASLt bias epilogue: one library launch, no materialized
// intermediate.  Explicit ids fail closed on any malformed pattern.  The
// absorbed launch runs at the Linear's position but writes the BiasAdd's
// output one position early, so every tensor it reads must be provably safe
// against memory-plan slot reuse (the write-early hazard recorded by the
// arena audit): each of input/weight/bias must hold a dedicated slot, stay
// semantically live through the BiasAdd's position, or be impossible for
// the planner to hand to the early-written output (no streamed constants in
// the program AND either the output is dedicated or the read tensor's
// aligned slot is smaller than the output needs).
std::vector<AbsorbedLinearBiasPlan> find_absorbed_linear_bias_plans(
    const ir::Program &program, const RunOptions &options) {
  std::vector<AbsorbedLinearBiasPlan> result;
  if (options.absorb_linear_bias_operations.empty())
    return result;
  bool streamed_program = false;
  for (const auto &tensor : program.tensors)
    if (tensor.has_role(ir::TensorRole::Streamed))
      streamed_program = true;
  const auto dedicated = [](const ir::TensorDesc &tensor) {
    return tensor.has_role(ir::TensorRole::Input) ||
           tensor.has_role(ir::TensorRole::Output) ||
           (tensor.has_role(ir::TensorRole::Constant) &&
            !tensor.has_role(ir::TensorRole::Streamed));
  };
  const auto align256 = [](std::uint64_t bytes) {
    return (bytes + 255U) & ~static_cast<std::uint64_t>(255U);
  };
  std::unordered_set<std::uint32_t> requested;
  for (const auto operation_id : options.absorb_linear_bias_operations) {
    if (!requested.insert(operation_id).second)
      continue;
    auto position = program.operations.size();
    for (std::size_t index = 0; index < program.operations.size(); ++index)
      if (program.operations[index].id == operation_id)
        position = index;
    if (position >= program.operations.size())
      fail("absorbed Linear id does not exist: " +
           std::to_string(operation_id));
    const auto &linear = program.operations[position];
    if (linear.opcode != ir::Opcode::Linear || linear.inputs.size() != 2U ||
        linear.u64(ir::AttrKey::Implementation, 1U) == 3U)
      fail("bias absorption requires an unbiased cuBLASLt Linear id: " +
           std::to_string(operation_id));
    const auto intermediate = linear.outputs.at(0);
    if (position + 1U >= program.operations.size())
      fail("absorbed Linear has no following BiasAdd: " +
           std::to_string(operation_id));
    const auto &bias_add = program.operations[position + 1U];
    if (bias_add.opcode != ir::Opcode::BiasAdd ||
        bias_add.inputs.size() != 2U || bias_add.inputs[0] != intermediate)
      fail("bias absorption requires the BiasAdd to immediately follow its "
           "Linear: " +
           std::to_string(operation_id));
    std::size_t uses = 0U;
    for (const auto &operation : program.operations)
      for (const auto input : operation.inputs)
        if (input == intermediate)
          ++uses;
    if (uses != 1U || program.tensor(intermediate)->roles != 0U)
      fail("bias absorption requires an exclusive internal Linear "
           "intermediate: " +
           std::to_string(operation_id));
    const auto *weight = program.tensor(linear.inputs[1]);
    const auto *bias = program.tensor(bias_add.inputs[1]);
    const auto *output = program.tensor(bias_add.outputs.at(0));
    if (!weight || !bias || !output || weight->dims.size() != 2U ||
        bias->dims.size() != 1U || bias->dims[0] != weight->dims[0])
      fail("bias absorption bias width must match the Linear output "
           "width: " +
           std::to_string(operation_id));
    for (const auto tensor_id :
         {linear.inputs[0], linear.inputs[1], bias_add.inputs[1]}) {
      const auto *description = program.tensor(tensor_id);
      if (dedicated(*description))
        continue;
      std::size_t last_use = 0U;
      for (std::size_t index = 0; index < program.operations.size(); ++index)
        for (const auto input : program.operations[index].inputs)
          if (input == tensor_id)
            last_use = index;
      if (last_use >= position + 1U)
        continue;
      if (!streamed_program &&
          (dedicated(*output) || align256(description->byte_count()) <
                                     align256(output->byte_count())))
        continue;
      fail("bias absorption would read tensor " + std::to_string(tensor_id) +
           " past its planned lifetime (write-early hazard); refusing "
           "Linear id " +
           std::to_string(operation_id));
    }
    AbsorbedLinearBiasPlan plan;
    plan.linear_operation = linear.id;
    plan.bias_operation = bias_add.id;
    plan.intermediate_tensor = intermediate;
    plan.output_tensor = bias_add.outputs.at(0);
    plan.eliminated_intermediate_bytes =
        program.tensor(intermediate)->byte_count();
    plan.launch_operation = linear;
    plan.launch_operation.inputs.push_back(bias_add.inputs[1]);
    plan.launch_operation.outputs = bias_add.outputs;
    result.push_back(std::move(plan));
  }
  return result;
}

constexpr std::uint32_t kH3W8A8ProjectionChunkRows = 4096U;

struct H3CompactAdaLNBinding {
  bool enabled{};
  std::uint32_t norm_operation{};
  std::uint32_t norm_input_tensor{};
  std::uint32_t norm_weight_tensor{};
  std::uint32_t modulation_tensor{};
  std::uint32_t indices_tensor{};
  std::uint32_t scale_lane{};
  std::uint32_t shift_lane{};
  std::uint32_t gate_lane{};
  float epsilon{};
};

struct H3CompactAdaLNPlan {
  std::uint32_t select_operation{};
  std::array<std::uint32_t, 6> expanded_tensors{};
  std::array<std::uint32_t, 2> norm_operations{};
  std::array<std::uint32_t, 2> normalized_tensors{};
  std::uint32_t layer{};
};

struct H3W8A8MlpPlan {
  std::uint32_t fc1_operation{};
  std::uint32_t swiglu_operation{};
  std::uint32_t fc2_operation{};
  std::uint32_t residual_operation{};
  std::uint32_t input_tensor{};
  std::uint32_t residual_tensor{};
  std::uint32_t gate_tensor{};
  std::uint32_t output_tensor{};
  std::uint32_t layer{};
  std::uint32_t chunk_rows{};
  std::uint64_t rows{};
  std::uint64_t hidden{};
  std::uint64_t ffn{};
  std::uint64_t packed_ffn{};
  std::vector<std::uint32_t> excluded_tensors;
  std::vector<std::uint32_t> replaced_constant_tensors;
  // Position in the merged resident upload order (attention then MLP per
  // layer, the checkpoint's physical order); drives the read-ahead window.
  std::size_t upload_order{};
  Tensor fc1_weight;
  Tensor fc1_scale;
  std::uint32_t fc1_weight_scale_groups{1U};
  Tensor fc2_weight;
  Tensor fc2_scale;
  std::uint32_t fc2_weight_scale_groups{1U};
  std::unique_ptr<Workspace> weight_storage;
  CUdeviceptr fc1_weight_device{};
  CUdeviceptr fc1_scale_device{};
  CUdeviceptr fc2_weight_device{};
  CUdeviceptr fc2_scale_device{};
  CUdeviceptr input_scale_device{};
  CUdeviceptr input_i8_device{};
  CUdeviceptr fc1_accumulator_device{};
  CUdeviceptr fc1_aggregate_device{};
  CUdeviceptr activation_device{};
  CUdeviceptr activation_scale_device{};
  CUdeviceptr activation_i8_device{};
  CUdeviceptr fc2_accumulator_device{};
  CUdeviceptr fc2_aggregate_device{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t weight_storage_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::filesystem::path cache_path;
  bool resident{};
  bool uploaded{};
  bool convrot{};
  bool cutlass_scaled_fc1{};
  bool cutlass_scaled_fc2{};
  std::uint32_t convrot_scale_chunk{};
  bool convrot_global_activation_scale{};
  H3CompactAdaLNBinding compact_adaln;
};

struct H3W8A8AttentionPlan {
  bool has_qkv_projection{};
  bool has_output_projection{};
  std::uint32_t qkv_layout_operation{};
  std::array<std::uint32_t, 3> qkv_linear_operations{};
  std::uint32_t output_linear_operation{};
  std::uint32_t residual_operation{};
  std::uint32_t attention_input_tensor{};
  std::array<std::uint32_t, 3> qkv_output_tensors{};
  std::uint32_t output_input_tensor{};
  std::uint32_t residual_tensor{};
  std::uint32_t gate_tensor{};
  std::uint32_t output_tensor{};
  std::vector<std::uint32_t> excluded_tensors;
  std::vector<std::uint32_t> replaced_constant_tensors;
  std::size_t upload_order{};
  std::uint32_t layer{};
  std::uint64_t rows{};
  std::uint64_t hidden{};
  std::uint64_t inner{};
  std::uint64_t packed_inner{};
  std::uint64_t head_dim{};
  Tensor qkv_weight;
  Tensor qkv_scale;
  std::uint32_t qkv_weight_scale_groups{1U};
  Tensor output_weight;
  Tensor output_scale;
  std::uint32_t output_weight_scale_groups{1U};
  std::unique_ptr<Workspace> weight_storage;
  CUdeviceptr qkv_weight_device{};
  CUdeviceptr qkv_scale_device{};
  CUdeviceptr output_weight_device{};
  CUdeviceptr output_scale_device{};
  CUdeviceptr activation_scale_device{};
  CUdeviceptr activation_i8_device{};
  CUdeviceptr accumulator_device{};
  CUdeviceptr aggregate_device{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t weight_storage_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::filesystem::path cache_path;
  bool resident{};
  bool uploaded{};
  bool convrot{};
  bool cutlass_scaled{};
  std::uint32_t convrot_scale_chunk{};
  bool convrot_global_activation_scale{};
  H3CompactAdaLNBinding compact_adaln;
};

struct H3W8A8Functions {
  CUfunction rowscale{};
  CUfunction encode{};
  CUfunction qkv{};
  CUfunction swiglu{};
  CUfunction residual{};
};

struct H3ConvRotFunctions {
  CUfunction encode{};
  CUfunction chunked_encode{};
  CUfunction generic_encode{};
  CUfunction generic_bf16_rotate{};
  CUfunction generic_weight_dequant{};
  CUfunction compact_encode{};
  CUfunction compact_chunked_encode{};
  CUfunction chunk_accumulate{};
  CUfunction qkv{};
  CUfunction qkv_bf16{};
  CUfunction qkv_f32{};
  CUfunction swiglu{};
  CUfunction swiglu_encode{};
  CUfunction swiglu_bf16_encode{};
  CUfunction swiglu_f32{};
  CUfunction compact_residual{};
  CUfunction compact_residual_bf16{};
  CUfunction compact_residual_f32{};
  CUfunction residual_f32{};
  CUfunction bf16_rotate_gather{};
  CUfunction compact_bf16_rotate_gather{};
  CUfunction qkv_bf16_scatter{};
  CUfunction swiglu_bf16{};
  CUfunction compact_residual_bf16_scatter{};
  CUfunction residual_bf16_scatter{};
};

struct H3ConvRotBf16Correction {
  std::vector<std::uint32_t> rows;
  std::unique_ptr<Workspace> storage;
  CUdeviceptr indices{};
  CUdeviceptr weight{};
  CUdeviceptr activation{};
  CUdeviceptr projected{};
  CUdeviceptr auxiliary{};
  std::uint64_t storage_bytes{};
  std::uint64_t weight_bytes{};
  std::uint64_t activation_bytes{};
  std::uint64_t projected_bytes{};
  std::uint64_t auxiliary_bytes{};
};

struct ConvRotInt8LinearPlan {
  std::uint32_t operation{};
  std::uint32_t input_tensor{};
  std::uint32_t weight_tensor{};
  std::uint32_t output_tensor{};
  std::uint32_t bias_tensor{};
  ir::DType dtype{};
  std::uint64_t rows{};
  std::uint64_t columns{};
  std::uint64_t contraction{};
  Tensor weight;
  Tensor scale;
  std::filesystem::path cache_path;
  CUdeviceptr weight_device{};
  CUdeviceptr scale_device{};
};

std::uint64_t align_256(std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - 255U)
    fail("H3 W8A8 storage alignment overflow");
  return (value + 255U) & ~std::uint64_t{255U};
}

// Architecture-tagged development DSO adapter. The project-owned ABI and the
// legacy Serenity oracle ABI are detected and reported separately. The loaded
// DSO contains raw CUDA launchers plus ABI/target-SM metadata; no Mojo, Python,
// or external model weights participate in prepared execution.
class CkAttentionLibrary {
public:
  using QuantQK = void (*)(
      const void *, void *, void *, const void *, void *, void *, int, int,
      int, int, int, int, int, int, int, int, std::int64_t, std::int64_t,
      std::int64_t, std::int64_t, std::int64_t, std::int64_t, int, void *,
      void *);
  using QuantV = void (*)(const void *, void *, void *, int, int, int, int,
                          int, std::int64_t, std::int64_t, std::int64_t, int,
                          void *);
  using Attend = void (*)(
      const void *, const void *, const void *, void *, const void *,
      const void *, const void *, const void *, std::int64_t, std::int64_t,
      std::int64_t, std::int64_t, int, int, int, int, int, int, int,
      int, int, int, int, int, int, int, int, int, int, int, int, int,
      float, int, void *);
  using MetadataInt = int (*)();
  using OwnedQuantKv = int (*)(
      const void *, const void *, void *, void *, void *, void *, int, int,
      int, int, std::int64_t, std::int64_t, std::int64_t, std::int64_t,
      std::int64_t, std::int64_t, void *);
  using OwnedAttend = int (*)(
      const void *, const void *, const void *, const void *, const void *,
      void *, int, int, int, int, float, std::int64_t, std::int64_t,
      std::int64_t, std::int64_t, std::int64_t, std::int64_t, void *);
  using OwnedError = const char *(*)(int);
  using OwnedQuantKvCentered = int (*)(
      const void *, const void *, void *, void *, void *, void *, void *,
      void *, int, int, int, int, std::int64_t, std::int64_t, std::int64_t,
      std::int64_t, std::int64_t, std::int64_t, void *);

  struct InTree {};
  // In-tree owned kernel: no dlopen, the same ABI v4 entry points bound
  // directly. Fails closed when the build carries no CUDA sources or the
  // running device is not the kernel's SM.
  CkAttentionLibrary(InTree, int current_sm, bool center_k)
      : path_("in-tree"), center_k_(center_k) {
#if DIF_HAS_H3_OWNED_ATTENTION
    owned_abi_version_ = h3_owned_attention::abi_version();
    target_sm_ = h3_owned_attention::target_sm();
    if (owned_abi_version_ != 4 || target_sm_ != current_sm)
      fail("owned H3 attention kernel admission failed: ABI=" +
           std::to_string(owned_abi_version_) + " target_sm=" +
           std::to_string(target_sm_) + " current_sm=" +
           std::to_string(current_sm));
    owned_quant_kv_ = &h3_owned_attention::quantize_kv_bf16;
    owned_attend_ = &h3_owned_attention::attention_bf16;
    owned_error_ = &h3_owned_attention::cuda_error;
    owned_quant_kv_centered_ = &h3_owned_attention::quantize_kv_centered_bf16;
    owned_dense_ = true;
    in_tree_ = true;
#else
    (void)current_sm;
    fail("this build has no in-tree owned H3 attention kernel (CUDA sources "
         "compile only when a CUDA compiler and CUTLASS are configured)");
#endif
  }

  CkAttentionLibrary(const std::filesystem::path &path, int current_sm)
      : path_(path) {
    const auto path_string = path.string();
    void *candidate = dlopen(path_string.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!candidate) {
      const char *reason = dlerror();
      fail("cannot load H3 CK attention DSO " + path_string + ": " +
           (reason ? reason : "unknown dlopen error"));
    }
    auto optional_symbol = [&](const char *name) {
      dlerror();
      auto *address = dlsym(candidate, name);
      const char *reason = dlerror();
      return reason ? nullptr : address;
    };
    auto symbol = [&](const char *name) {
      auto *address = optional_symbol(name);
      if (!address) {
        const char *reason = dlerror();
        const auto message = std::string("H3 CK attention DSO is missing ") +
                             name + ": " +
                             (reason ? reason : "unknown dlsym error");
        dlclose(candidate);
        fail(message);
      }
      return address;
    };
    if (auto *owned_abi_address =
            optional_symbol("codealexx_h3_dense_abi_version")) {
      const auto abi = reinterpret_cast<MetadataInt>(owned_abi_address);
      const auto target = reinterpret_cast<MetadataInt>(
          symbol("codealexx_h3_dense_target_sm"));
      const auto abi_version = abi();
      target_sm_ = target();
      if ((abi_version != 3 && abi_version != 4) ||
          target_sm_ != current_sm) {
        const auto message =
            "owned H3 dense attention DSO admission failed: ABI=" +
            std::to_string(abi_version) + " target_sm=" +
            std::to_string(target_sm_) + " current_sm=" +
            std::to_string(current_sm);
        dlclose(candidate);
        fail(message);
      }
      owned_quant_kv_ = reinterpret_cast<OwnedQuantKv>(
          symbol("codealexx_h3_dense_quantize_kv_bf16"));
      owned_attend_ = reinterpret_cast<OwnedAttend>(
          symbol("codealexx_h3_dense_attention_bf16"));
      owned_error_ = reinterpret_cast<OwnedError>(
          symbol("codealexx_h3_dense_cuda_error"));
      owned_abi_version_ = abi_version;
      owned_dense_ = true;
      handle_ = candidate;
      return;
    }
    auto *ck_int8_abi_address =
        optional_symbol("ck_int8_kernel_abi_version");
    const auto abi = reinterpret_cast<MetadataInt>(
        ck_int8_abi_address
            ? ck_int8_abi_address
            : symbol("serenity_ck_attention_abi_version"));
    const auto target = reinterpret_cast<MetadataInt>(
        ck_int8_abi_address
            ? symbol("ck_int8_kernel_target_sm")
            : symbol("serenity_ck_attention_target_sm"));
    const auto abi_version = abi();
    target_sm_ = target();
    if (abi_version != 1 || target_sm_ != current_sm) {
      const auto message =
          std::string(ck_int8_abi_address ? "CodeAlexx CK-INT8" : "H3 CK") +
          " attention DSO admission failed: ABI=" +
          std::to_string(abi_version) + " target_sm=" +
          std::to_string(target_sm_) + " current_sm=" +
          std::to_string(current_sm);
      dlclose(candidate);
      fail(message);
    }
    quant_qk_ = reinterpret_cast<QuantQK>(
        symbol("launch_quant_qk_per_thread_int8"));
    quant_v_ = reinterpret_cast<QuantV>(
        symbol("launch_quant_v_int8_kernel"));
    attend_ = reinterpret_cast<Attend>(symbol("launch_sage_attn_kernel"));
    codealexx_ck_int8_ = ck_int8_abi_address != nullptr;
    handle_ = candidate;
  }

  ~CkAttentionLibrary() {
    if (handle_)
      dlclose(handle_);
  }
  CkAttentionLibrary(const CkAttentionLibrary &) = delete;
  CkAttentionLibrary &operator=(const CkAttentionLibrary &) = delete;

  void launch(CUdeviceptr query, CUdeviceptr key, CUdeviceptr value,
              CUdeviceptr output, CUdeviceptr q_int8, CUdeviceptr q_scale,
              CUdeviceptr k_int8, CUdeviceptr k_scale, CUdeviceptr v_int8,
              CUdeviceptr v_scale, CUdeviceptr anchor_indices, int sequence,
              int heads, float scale, CUstream stream,
              CUdeviceptr k_mean_partials = 0, CUdeviceptr k_mean = 0) const {
    constexpr int head_dim = 128;
    constexpr int cta_q = 128;
    constexpr int warp_q = 32;
    constexpr int cta_k = 128;
    constexpr int warp_k = 128;
    constexpr int bf16_code = 2;
    constexpr int batch = 1;
    const int padded_sequence =
        (sequence + cta_k - 1) / cta_k * cta_k;
    const auto in_sb = static_cast<std::int64_t>(sequence) * heads * head_dim;
    constexpr std::int64_t in_sh = head_dim;
    const auto in_sn = static_cast<std::int64_t>(heads) * head_dim;
    auto pointer = [](CUdeviceptr value) {
      return reinterpret_cast<void *>(value);
    };
    if (owned_dense_) {
      const auto padded_sequence =
          (sequence + cta_k - 1) / cta_k * cta_k;
      auto status =
          center_k_
              ? owned_quant_kv_centered_(
                    pointer(key), pointer(value), pointer(k_int8),
                    pointer(k_scale), pointer(v_int8), pointer(v_scale),
                    pointer(k_mean_partials), pointer(k_mean), batch, heads,
                    sequence, padded_sequence, in_sb, in_sh, in_sn, in_sb,
                    in_sh, in_sn, reinterpret_cast<void *>(stream))
              : owned_quant_kv_(
                    pointer(key), pointer(value), pointer(k_int8),
                    pointer(k_scale), pointer(v_int8), pointer(v_scale), batch,
                    heads, sequence, padded_sequence, in_sb, in_sh, in_sn,
                    in_sb, in_sh, in_sn, reinterpret_cast<void *>(stream));
      if (status == 0)
        status = owned_attend_(
            pointer(query), pointer(k_int8), pointer(k_scale), pointer(v_int8),
            pointer(v_scale), pointer(output), batch, heads, sequence,
            padded_sequence, scale, in_sb, in_sh, in_sn, in_sb, in_sh, in_sn,
            reinterpret_cast<void *>(stream));
      if (status != 0) {
        const char *reason = owned_error_ ? owned_error_(status) : nullptr;
        fail("owned H3 dense attention launch failed: " +
             std::string(reason ? reason : "unknown CUDA error"));
      }
      return;
    }
    quant_qk_(pointer(query), pointer(q_int8), pointer(q_scale), pointer(key),
              pointer(k_int8), pointer(k_scale), batch, heads, sequence, heads,
              sequence, head_dim, cta_q, warp_q, cta_k, warp_k, in_sb, in_sh,
              in_sn, in_sb, in_sh, in_sn, bf16_code,
              pointer(anchor_indices), reinterpret_cast<void *>(stream));
    quant_v_(pointer(value), pointer(v_int8), pointer(v_scale), batch, heads,
             sequence, head_dim, padded_sequence, in_sb, in_sh, in_sn,
             bf16_code, reinterpret_cast<void *>(stream));

    const int q_b = heads * sequence * head_dim;
    constexpr int q_n = head_dim;
    const int q_h = sequence * head_dim;
    const int v_b = heads * head_dim * padded_sequence;
    const int v_h = head_dim * padded_sequence;
    const int v_d = padded_sequence;
    const int o_b = sequence * heads * head_dim;
    const int o_n = heads * head_dim;
    constexpr int o_h = head_dim;
    attend_(pointer(q_int8), pointer(k_int8), pointer(v_int8), pointer(output),
            pointer(q_scale), pointer(k_scale), pointer(v_scale), nullptr, 0, 0,
            0, 0, 0, cta_k, batch, sequence, sequence, heads, heads, head_dim,
            q_b, q_n, q_h, q_b, q_n, q_h, v_b, v_h, v_d, o_b, o_n, o_h,
            scale, bf16_code, reinterpret_cast<void *>(stream));
  }

  int target_sm() const { return target_sm_; }
  bool owned_dense() const { return owned_dense_; }
  bool codealexx_ck_int8() const { return codealexx_ck_int8_; }
  int owned_abi_version() const { return owned_abi_version_; }
  std::string classification() const {
    return owned_dense_ ? "approximate_owned_h3_dense_int8_gate"
           : codealexx_ck_int8_
               ? "approximate_codealexx_ck_int8_established_h3_gate"
               : "approximate_ck_int8_established_h3_gate";
  }
  bool in_tree() const { return in_tree_; }
  bool center_k() const { return center_k_; }
  std::string implementation() const {
    return owned_dense_
               ? (in_tree_ ? (center_k_
                                  ? "owned_h3_dense_int8_v4_in_tree_center_k"
                                  : "owned_h3_dense_int8_v4_in_tree")
                           : "codealexx_h3_dense_int8_v" +
                                 std::to_string(owned_abi_version_))
           : codealexx_ck_int8_
               ? "codealexx_ck_int8_comfy_kitchen_sage_bf16"
               : "serenity_comfy_kitchen_sage_bf16";
  }
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
  void *handle_{};
  QuantQK quant_qk_{};
  QuantV quant_v_{};
  Attend attend_{};
  OwnedQuantKv owned_quant_kv_{};
  OwnedAttend owned_attend_{};
  OwnedError owned_error_{};
  int target_sm_{};
  int owned_abi_version_{};
  bool owned_dense_{};
  bool codealexx_ck_int8_{};
  bool in_tree_{};
  bool center_k_{};
  OwnedQuantKvCentered owned_quant_kv_centered_{};
};

class CkAttentionPlan {
public:
  CkAttentionPlan(const ir::Program &program, const ir::Operation &operation,
                  std::shared_ptr<CkAttentionLibrary> library)
      : operation_id_(operation.id), library_(std::move(library)) {
    const auto *query = program.tensor(operation.inputs.at(0));
    if (!query || query->dtype != ir::DType::BF16 || query->dims.size() != 3U ||
        query->dims.at(2) != 128U ||
        operation.boolean(ir::AttrKey::Causal, false))
      fail("H3 CK attention requires noncausal BF16 [S,H,128]");
    if (query->dims.at(0) > static_cast<std::uint64_t>(
                                std::numeric_limits<int>::max()) ||
        query->dims.at(1) > static_cast<std::uint64_t>(
                                std::numeric_limits<int>::max()))
      fail("H3 CK attention geometry exceeds launcher integer range");
    sequence_ = static_cast<int>(query->dims.at(0));
    heads_ = static_cast<int>(query->dims.at(1));
    scale_ = static_cast<float>(operation.f64(
        ir::AttrKey::AttentionScale, 1.0 / std::sqrt(128.0)));
    const auto sequence = static_cast<std::uint64_t>(sequence_);
    const auto heads = static_cast<std::uint64_t>(heads_);
    const auto padded_sequence = (sequence + 127U) / 128U * 128U;
    const auto elements = sequence * heads * 128U;
    const auto q_scale_elements = heads * ((sequence + 127U) / 128U) * 32U;
    const auto k_scale_elements =
        library_->owned_dense()
            ? (library_->owned_abi_version() == 3
                   ? heads * padded_sequence
                   : heads * (padded_sequence / 128U))
            : heads * ((sequence + 127U) / 128U) * 4U;
    const auto v_scale_elements =
        library_->owned_dense() && library_->owned_abi_version() == 3
            ? heads * (padded_sequence / 128U) * 128U
            : heads * 128U;
    const auto v_elements = heads * 128U * padded_sequence;
    // The owned kernels write K as [B,H,S_pad,128]: every padded row is
    // stored (zero-filled past the sequence), so the INT8 K scratch is sized
    // by the padded sequence. The DSO route's K stays [B,H,S,128].
    const auto k_elements =
        library_->owned_dense() ? heads * 128U * padded_sequence : elements;
    if (library_->owned_dense()) {
      scratch_bytes_ = align_256(k_elements) +
                       align_256(k_scale_elements * sizeof(float)) +
                       align_256(v_elements) +
                       align_256(v_scale_elements * sizeof(float));
      if (library_->center_k())
        scratch_bytes_ +=
            align_256(heads * (padded_sequence / 128U) * 128U * sizeof(float)) +
            align_256(heads * 128U * sizeof(float));
    } else {
      scratch_bytes_ = align_256(elements) + align_256(elements) +
                       align_256(q_scale_elements * sizeof(float)) +
                       align_256(k_scale_elements * sizeof(float)) +
                       align_256(v_elements) +
                       align_256(heads * 128U * sizeof(float)) +
                       align_256(heads * sizeof(std::int32_t));
    }
  }

  void allocate(DeviceArena *arena) {
    if (storage_)
      return;
    const auto sequence = static_cast<std::uint64_t>(sequence_);
    const auto heads = static_cast<std::uint64_t>(heads_);
    const auto padded_sequence = (sequence + 127U) / 128U * 128U;
    const auto elements = sequence * heads * 128U;
    const auto q_scale_elements = heads * ((sequence + 127U) / 128U) * 32U;
    const auto k_scale_elements =
        library_->owned_dense()
            ? (library_->owned_abi_version() == 3
                   ? heads * padded_sequence
                   : heads * (padded_sequence / 128U))
            : heads * ((sequence + 127U) / 128U) * 4U;
    const auto v_scale_elements =
        library_->owned_dense() && library_->owned_abi_version() == 3
            ? heads * (padded_sequence / 128U) * 128U
            : heads * 128U;
    const auto v_elements = heads * 128U * padded_sequence;
    const auto k_elements =
        library_->owned_dense() ? heads * 128U * padded_sequence : elements;
    storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(scratch_bytes_), arena);
    auto offset = std::uint64_t{0U};
    auto assign = [&](CUdeviceptr &target, std::uint64_t bytes) {
      target = storage_->pointer() + offset;
      offset += align_256(bytes);
    };
    if (library_->owned_dense()) {
      assign(k_int8_, k_elements);
      assign(k_scale_, k_scale_elements * sizeof(float));
      assign(v_int8_, v_elements);
      assign(v_scale_, v_scale_elements * sizeof(float));
      if (library_->center_k()) {
        assign(k_mean_partials_,
               heads * (padded_sequence / 128U) * 128U * sizeof(float));
        assign(k_mean_, heads * 128U * sizeof(float));
      }
    } else {
      assign(q_int8_, elements);
      assign(k_int8_, elements);
      assign(q_scale_, q_scale_elements * sizeof(float));
      assign(k_scale_, k_scale_elements * sizeof(float));
      assign(v_int8_, v_elements);
      assign(v_scale_, heads * 128U * sizeof(float));
      assign(anchor_indices_, heads * sizeof(std::int32_t));
    }
    if (offset != scratch_bytes_)
      fail("H3 CK attention scratch layout mismatch");
  }

  void execute(const ir::Operation &operation, DeviceBuffers &buffers,
               CUstream stream) const {
    library_->launch(buffers.at(operation.inputs.at(0)),
                     buffers.at(operation.inputs.at(1)),
                     buffers.at(operation.inputs.at(2)),
                     buffers.at(operation.outputs.at(0)), q_int8_, q_scale_,
                     k_int8_, k_scale_, v_int8_, v_scale_, anchor_indices_,
                     sequence_, heads_, scale_, stream, k_mean_partials_,
                     k_mean_);
  }

  bool compatible(const ir::Program &program,
                  const ir::Operation &operation) const {
    const auto *query = program.tensor(operation.inputs.at(0));
    const auto scale = static_cast<float>(operation.f64(
        ir::AttrKey::AttentionScale, 1.0 / std::sqrt(128.0)));
    return query && query->dtype == ir::DType::BF16 &&
           query->dims == std::vector<std::uint64_t>{
                              static_cast<std::uint64_t>(sequence_),
                              static_cast<std::uint64_t>(heads_), 128U} &&
           !operation.boolean(ir::AttrKey::Causal, false) && scale == scale_;
  }

  std::uint32_t operation_id() const { return operation_id_; }
  std::uint64_t scratch_bytes() const { return scratch_bytes_; }
  int target_sm() const { return library_->target_sm(); }
  bool owned_dense() const { return library_->owned_dense(); }
  bool codealexx_ck_int8() const { return library_->codealexx_ck_int8(); }
  const std::string classification() const { return library_->classification(); }
  const std::string implementation() const {
    return library_->implementation();
  }
  const std::filesystem::path &path() const { return library_->path(); }

private:
  std::uint32_t operation_id_{};
  std::shared_ptr<CkAttentionLibrary> library_;
  std::unique_ptr<Workspace> storage_;
  CUdeviceptr q_int8_{};
  CUdeviceptr q_scale_{};
  CUdeviceptr k_int8_{};
  CUdeviceptr k_scale_{};
  CUdeviceptr v_int8_{};
  CUdeviceptr k_mean_partials_{};
  CUdeviceptr k_mean_{};
  CUdeviceptr v_scale_{};
  CUdeviceptr anchor_indices_{};
  int sequence_{};
  int heads_{};
  float scale_{};
  std::uint64_t scratch_bytes_{};
};

struct H3ModulationCachePlan {
  std::uint32_t linear_operation{};
  std::uint32_t select_operation{};
  std::uint32_t projected_tensor{};
  std::uint32_t input_tensor{};
  std::uint32_t layer{};
  std::uint32_t slices{1U};
  std::uint32_t tables{};
  std::vector<std::uint32_t> replaced_constant_tensors;
  Tensor modulation;
  CUdeviceptr modulation_device{};
  std::uint64_t storage_bytes{};
  std::uint64_t slice_bytes{};
  std::uint64_t replaced_weight_bytes{};
  std::filesystem::path cache_path;
  bool final{};
};

std::uint64_t h3_metadata_u64(const weights::SafeTensorFile &file,
                              std::string_view name) {
  const auto *entry = file.find_metadata(name);
  if (!entry || entry->dtype != "I64" || entry->dims !=
                                              std::vector<std::uint64_t>{1U})
    fail("H3 runtime cache has invalid metadata: " + std::string(name));
  const auto bytes = weights::read_safetensor_metadata(file, name);
  if (bytes.size() != 8U)
    fail("H3 runtime cache I64 metadata has invalid width");
  auto value = std::uint64_t{0U};
  for (unsigned index = 0U; index < 8U; ++index)
    value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
  return value;
}

std::string h3_metadata_string(const weights::SafeTensorFile &file,
                               std::string_view name) {
  const auto *entry = file.find_metadata(name);
  if (!entry || entry->dtype != "U8" || entry->dims.size() != 1U)
    fail("H3 runtime cache has invalid string metadata: " +
         std::string(name));
  const auto bytes = weights::read_safetensor_metadata(file, name);
  return {reinterpret_cast<const char *>(bytes.data()), bytes.size()};
}

std::optional<std::string>
h3_optional_metadata_string(const weights::SafeTensorFile &file,
                            std::string_view name) {
  const auto *entry = file.find_metadata(name);
  if (!entry)
    return std::nullopt;
  if (entry->dtype != "U8" || entry->dims.size() != 1U)
    fail("H3 runtime cache has invalid string metadata: " +
         std::string(name));
  const auto bytes = weights::read_safetensor_metadata(file, name);
  return std::string(reinterpret_cast<const char *>(bytes.data()),
                     bytes.size());
}

void validate_h3_modulation_cache_metadata(
    const weights::SafeTensorFile &cache, const RunOptions &options,
    std::uint32_t slices, std::uint32_t tables, std::uint32_t blocks) {
  if (options.h3_modulation_source_index.empty() ||
      options.h3_modulation_steps == 0U)
    fail("schedule modulation cache requires its source index and step count");
  const auto row_layout =
      h3_optional_metadata_string(cache, "__meta__.row_layout");
  constexpr std::string_view local_layout =
      "per-evaluation-sorted-unique-padded-v1";
  // Legacy two-table T2VA caches happen to use the same value order as the
  // creator's sorted local table. Conditioned global caches do not: their
  // collapsed first evaluation repeats video/audio before condition rows.
  // Reinterpreting those bytes as local slices silently selects the wrong
  // modulation, so conditioned caches must declare the native layout.
  if ((row_layout && *row_layout != local_layout) ||
      (!row_layout && tables > 2U))
    fail("conditioned H3 modulation cache lacks compatible local row-layout provenance");
  if (h3_metadata_u64(cache, "__meta__.version") != 1U ||
      h3_metadata_string(cache, "__meta__.kind") != "adaln-modulation" ||
      h3_metadata_string(cache, "__meta__.src_path") !=
          options.h3_modulation_source_index.string() ||
      h3_metadata_u64(cache, "__meta__.steps") !=
          options.h3_modulation_steps ||
      h3_metadata_u64(cache, "__meta__.distinct_timesteps") !=
          static_cast<std::uint64_t>(tables) * slices ||
      h3_metadata_u64(cache, "__meta__.nblocks") != blocks)
    fail("H3 schedule modulation cache metadata does not match this run");
  struct stat source_stat {};
  if (::stat(options.h3_modulation_source_index.c_str(), &source_stat) != 0)
    fail("cannot stat H3 modulation source index");
  if (h3_metadata_u64(cache, "__meta__.src_size") !=
          static_cast<std::uint64_t>(source_stat.st_size) ||
      h3_metadata_u64(cache, "__meta__.src_mtime") !=
          static_cast<std::uint64_t>(source_stat.st_mtime))
    fail("H3 modulation source index changed after cache creation");
}

std::vector<H3ModulationCachePlan> find_h3_modulation_cache_plans(
    const ir::Program &program, const RunOptions &options) {
  if (options.h3_modulation_cache.empty()) {
    if (!options.h3_modulation_input.empty() ||
        !options.h3_modulation_source_index.empty() ||
        options.h3_modulation_steps != 0U ||
        options.h3_modulation_layer != 0U ||
        options.h3_modulation_total_layers != 0U)
      fail("H3 modulation input/layer requires an H3 modulation cache");
    return {};
  }
  const auto cache = weights::read_safetensors(options.h3_modulation_cache);
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs) {
      if (!producer.emplace(output, &operation).second)
        fail("H3 modulation cache found a tensor with multiple producers");
    }
  }

  std::vector<H3ModulationCachePlan> result;
  std::uint32_t common_input = 0U;
  for (const auto &select : program.operations) {
    if (select.opcode != ir::Opcode::H3AdaLNSelect)
      continue;
    if (select.inputs.size() != 2U || select.outputs.size() != 6U)
      fail("H3 modulation cache encountered a malformed AdaLN select");
    const auto found = producer.find(select.inputs.at(0));
    if (found == producer.end() || found->second->opcode != ir::Opcode::Linear)
      fail("H3 modulation cache requires Linear -> H3AdaLNSelect");
    const auto &linear = *found->second;
    if (linear.inputs.size() != 3U || linear.outputs.size() != 1U ||
        linear.outputs.at(0) != select.inputs.at(0))
      fail("H3 modulation cache requires a biased AdaLN Linear");
    const auto *input = program.tensor(linear.inputs.at(0));
    const auto *weight = program.tensor(linear.inputs.at(1));
    const auto *bias = program.tensor(linear.inputs.at(2));
    const auto *projected = program.tensor(linear.outputs.at(0));
    const auto *selected = program.tensor(select.outputs.at(0));
    if (!input || !weight || !bias || !projected || !selected ||
        input->dtype != ir::DType::BF16 || input->dims.size() != 2U ||
        weight->dtype != ir::DType::BF16 || weight->dims.size() != 2U ||
        bias->dtype != ir::DType::BF16 || bias->dims.size() != 1U ||
        projected->dtype != ir::DType::BF16 || projected->dims.size() != 2U ||
        selected->dtype != ir::DType::BF16 || selected->dims.size() != 2U ||
        !weight->has_role(ir::TensorRole::Constant) ||
        !bias->has_role(ir::TensorRole::Constant) ||
        projected->roles != static_cast<std::uint32_t>(ir::TensorRole::Internal))
      fail("H3 modulation cache semantic chain has incompatible tensors");
    const auto tables = input->dims.at(0);
    const auto time_width = input->dims.at(1);
    const auto hidden = selected->dims.at(1);
    if (projected->dims != std::vector<std::uint64_t>{tables, 18U * hidden} ||
        weight->dims !=
            std::vector<std::uint64_t>{18U * hidden, time_width} ||
        bias->dims != std::vector<std::uint64_t>{18U * hidden})
      fail("H3 modulation cache does not match released AdaLN geometry");
    for (const auto output : select.outputs) {
      const auto *description = program.tensor(output);
      if (!description || description->dtype != ir::DType::BF16 ||
          description->dims != selected->dims)
        fail("H3 modulation cache select outputs disagree");
    }
    if (common_input == 0U)
      common_input = linear.inputs.at(0);
    else if (common_input != linear.inputs.at(0))
      fail("H3 modulation cache requires one shared activated timestep input");

    const auto layer = options.h3_modulation_layer +
                       static_cast<std::uint32_t>(result.size());
    auto modulation =
        weights::map_safetensor(cache, "block." + std::to_string(layer));
    const auto cache_rows_per_slice = 3U * tables;
    const auto cache_columns = 6U * hidden;
    std::uint32_t slices = 1U;
    if (modulation.dtype != ir::DType::BF16)
      fail("H3 modulation cache block must be BF16");
    if (modulation.dims == projected->dims) {
      slices = 1U;
    } else if (modulation.dims.size() == 2U &&
               modulation.dims.at(1) == cache_columns &&
               modulation.dims.at(0) % cache_rows_per_slice == 0U &&
               modulation.dims.at(0) / cache_rows_per_slice <=
                   std::numeric_limits<std::uint32_t>::max()) {
      slices = static_cast<std::uint32_t>(modulation.dims.at(0) /
                                          cache_rows_per_slice);
    } else {
      fail("H3 modulation cache block does not match semantic AdaLN output");
    }
    if (modulation.byte_size() != projected->byte_count() * slices)
      fail("H3 modulation cache block slice size mismatch");
    H3ModulationCachePlan plan;
    plan.linear_operation = linear.id;
    plan.select_operation = select.id;
    plan.projected_tensor = projected->id;
    plan.input_tensor = input->id;
    plan.layer = layer;
    plan.slices = slices;
    plan.tables = static_cast<std::uint32_t>(tables);
    plan.replaced_constant_tensors = {weight->id, bias->id};
    plan.modulation = std::move(modulation);
    plan.storage_bytes = align_256(plan.modulation.byte_size());
    plan.slice_bytes = projected->byte_count();
    plan.replaced_weight_bytes = weight->byte_count() + bias->byte_count();
    plan.cache_path = options.h3_modulation_cache;
    result.push_back(std::move(plan));
  }
  if (result.empty())
    fail("H3 modulation cache requested but no AdaLN chain was found");

  const auto block_count = static_cast<std::uint32_t>(result.size());
  for (const auto &select : program.operations) {
    if (select.opcode != ir::Opcode::SelectRowChunks ||
        select.inputs.size() != 2U || select.outputs.size() != 2U)
      continue;
    const auto found = producer.find(select.inputs.at(0));
    if (found == producer.end() || found->second->opcode != ir::Opcode::Linear)
      continue;
    const auto &linear = *found->second;
    if (linear.inputs.size() != 3U || linear.outputs.size() != 1U)
      continue;
    const auto *input = program.tensor(linear.inputs.at(0));
    const auto *weight = program.tensor(linear.inputs.at(1));
    const auto *bias = program.tensor(linear.inputs.at(2));
    const auto *projected = program.tensor(linear.outputs.at(0));
    const auto *selected = program.tensor(select.outputs.at(0));
    if (!input || !weight || !bias || !projected || !selected ||
        input->dtype != ir::DType::BF16 || input->dims.size() != 2U ||
        input->id != common_input || weight->dtype != ir::DType::BF16 ||
        bias->dtype != ir::DType::BF16 ||
        projected->dtype != ir::DType::BF16 ||
        projected->dims.size() != 2U || selected->dtype != ir::DType::BF16 ||
        selected->dims.size() != 2U)
      continue;
    const auto tables = input->dims.at(0);
    const auto hidden = selected->dims.at(1);
    if (projected->dims !=
            std::vector<std::uint64_t>{tables, 2U * hidden} ||
        weight->dims !=
            std::vector<std::uint64_t>{2U * hidden, input->dims.at(1)} ||
        bias->dims != std::vector<std::uint64_t>{2U * hidden})
      continue;
    auto modulation = weights::map_safetensor(cache, "final");
    if (modulation.dtype != ir::DType::BF16 ||
        modulation.dims.size() != 2U ||
        modulation.dims.at(1) != 2U * hidden ||
        modulation.dims.at(0) % tables != 0U ||
        modulation.dims.at(0) / tables != result.front().slices ||
        modulation.byte_size() !=
            projected->byte_count() * result.front().slices)
      fail("H3 modulation cache final tensor does not match semantic output");
    H3ModulationCachePlan plan;
    plan.linear_operation = linear.id;
    plan.select_operation = select.id;
    plan.projected_tensor = projected->id;
    plan.input_tensor = input->id;
    plan.layer = block_count;
    plan.slices = result.front().slices;
    plan.tables = result.front().tables;
    plan.replaced_constant_tensors = {weight->id, bias->id};
    plan.modulation = std::move(modulation);
    plan.storage_bytes = align_256(plan.modulation.byte_size());
    plan.slice_bytes = projected->byte_count();
    plan.replaced_weight_bytes = weight->byte_count() + bias->byte_count();
    plan.cache_path = options.h3_modulation_cache;
    plan.final = true;
    result.push_back(std::move(plan));
    break;
  }
  const auto slices = result.front().slices;
  if (std::any_of(result.begin(), result.end(), [&](const auto &plan) {
        return plan.slices != slices;
      }))
    fail("H3 modulation cache tensors disagree on schedule slices");
  if (slices == 1U) {
    if (options.h3_modulation_input.empty())
      fail("single-slice modulation cache requires its exact source input");
  } else {
    if (options.h3_modulation_total_layers == 0U) {
      if (result.size() != static_cast<std::size_t>(block_count) + 1U)
        fail("schedule modulation cache requires the final output-head tensor");
      validate_h3_modulation_cache_metadata(
          cache, options, slices, result.front().tables, block_count);
    } else {
      if (options.h3_modulation_layer + block_count >
          options.h3_modulation_total_layers)
        fail("H3 diagnostic modulation slice exceeds total layer count");
      validate_h3_modulation_cache_metadata(
          cache, options, slices, result.front().tables,
          options.h3_modulation_total_layers);
    }
  }
  return result;
}

void assign_h3_modulation_storage(
    std::vector<H3ModulationCachePlan> &plans, CUdeviceptr base,
    DeviceBuffers &buffers) {
  auto offset = std::uint64_t{0U};
  for (auto &plan : plans) {
    plan.modulation_device = base + offset;
    buffers.bind_external(plan.projected_tensor, plan.modulation_device);
    offset += plan.storage_bytes;
  }
}

void select_h3_modulation_slice(
    const std::vector<H3ModulationCachePlan> &plans, std::uint32_t slice,
    DeviceBuffers &buffers) {
  for (const auto &plan : plans) {
    if (slice >= plan.slices)
      fail("H3 modulation cache slice is outside the prepared schedule");
    buffers.rebind_external(plan.projected_tensor,
                            plan.modulation_device + slice * plan.slice_bytes);
  }
}

void upload_h3_modulation_cache(
    const std::vector<H3ModulationCachePlan> &plans, CUstream stream) {
  for (const auto &plan : plans)
    check(counted_memcpy_htod(plan.modulation_device, plan.modulation.data(),
                            plan.modulation.byte_size(), stream),
          "cuMemcpyHtoDAsync H3 modulation cache");
}

struct H3GroupwiseProjection {
  Tensor weight;
  Tensor scale;
  CUdeviceptr weight_device{};
  CUdeviceptr scale_device{};
  std::uint64_t rows{};
  std::uint64_t columns{};
  std::uint32_t group_size{};
};

struct H3GroupwiseBlockPlan {
  std::uint32_t qkv_layout_operation{};
  std::uint32_t output_linear_operation{};
  std::uint32_t fc1_operation{};
  std::uint32_t fc2_operation{};
  std::array<std::uint32_t, 3> qkv_weight_tensors{};
  std::uint32_t output_weight_tensor{};
  std::uint32_t fc1_weight_tensor{};
  std::uint32_t fc2_weight_tensor{};
  std::uint32_t layer{};
  std::vector<std::uint32_t> excluded_tensors;
  std::vector<std::uint32_t> replaced_constant_tensors;
  std::array<H3GroupwiseProjection, 4> projections;
  std::unique_ptr<Workspace> weight_storage;
  std::uint64_t weight_storage_bytes{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t scratch_bytes{};
  std::filesystem::path cache_path;
};

std::vector<H3GroupwiseBlockPlan> find_h3_groupwise_plans(
    const ir::Program &program, const RunOptions &options) {
  if (options.h3_groupwise_cache.empty()) {
    if (options.h3_groupwise_layer != 0U)
      fail("H3 groupwise layer requires an H3 groupwise cache");
    return {};
  }
  if (!options.h3_w8a8_cache.empty() ||
      !options.h3_convrot_int8_checkpoint.empty())
    fail("H3 direct INT8 and groupwise INT8 precision routes are mutually exclusive");
  const auto cache = weights::read_safetensors(options.h3_groupwise_cache);
  std::unordered_map<std::uint32_t, const ir::Operation *> producers;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> consumers;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs)
      producers.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::vector<H3GroupwiseBlockPlan> result;
  for (std::size_t layout_index = 0U;
       layout_index < program.operations.size(); ++layout_index) {
    const auto &layout = program.operations[layout_index];
    if (layout.opcode != ir::Opcode::H3DeinterleaveQkvWeight ||
        layout.inputs.size() != 1U || layout.outputs.size() != 3U)
      continue;
    if (result.size() >= options.h3_groupwise_layers)
      break;
    const auto *packed = program.tensor(layout.inputs.at(0));
    if (!packed || packed->dtype != ir::DType::BF16 ||
        packed->dims.size() != 2U || packed->dims.at(0) % 3U != 0U)
      fail("H3 groupwise route found an incompatible QKV layout");
    const auto packed_inner = packed->dims.at(0);
    const auto inner = packed_inner / 3U;
    const auto hidden = packed->dims.at(1);
    std::array<const ir::Operation *, 3> qkv_linears{};
    for (std::size_t output_index = 0U;
         output_index < layout.outputs.size(); ++output_index) {
      const auto output = layout.outputs[output_index];
      const auto *description = program.tensor(output);
      const auto found = consumers.find(output);
      if (!description || description->dtype != ir::DType::BF16 ||
          description->dims != std::vector<std::uint64_t>{inner, hidden} ||
          found == consumers.end() || found->second.size() != 1U ||
          found->second.front()->opcode != ir::Opcode::Linear ||
          found->second.front()->inputs.size() != 2U ||
          found->second.front()->inputs.at(1) != output)
        fail("H3 groupwise route requires the source-faithful split QKV chain");
      qkv_linears[output_index] = found->second.front();
    }
    const auto q_consumers = consumers.find(qkv_linears[0]->outputs.at(0));
    const auto k_consumers = consumers.find(qkv_linears[1]->outputs.at(0));
    const auto transformer_qk =
        q_consumers != consumers.end() && k_consumers != consumers.end() &&
        q_consumers->second.size() == 1U && k_consumers->second.size() == 1U &&
        q_consumers->second.front()->opcode == ir::Opcode::QkNormPartialRope &&
        k_consumers->second.front()->opcode == ir::Opcode::QkNormPartialRope;
    if (!transformer_qk)
      continue;

    auto next_layout = program.operations.size();
    for (std::size_t index = layout_index + 1U;
         index < program.operations.size(); ++index) {
      if (program.operations[index].opcode ==
          ir::Opcode::H3DeinterleaveQkvWeight) {
        next_layout = index;
        break;
      }
    }
    const ir::Operation *swiglu = nullptr;
    for (std::size_t index = layout_index + 1U; index < next_layout; ++index) {
      if (program.operations[index].opcode != ir::Opcode::SwiGlu)
        continue;
      if (swiglu)
        fail("H3 groupwise route found multiple SwiGlu operations in one block");
      swiglu = &program.operations[index];
    }
    if (!swiglu || swiglu->inputs.size() != 1U ||
        swiglu->outputs.size() != 1U ||
        !swiglu->boolean(ir::AttrKey::GateFirst, false))
      fail("H3 groupwise route requires the creator gate-first SwiGLU chain");
    const auto fc1_found = producers.find(swiglu->inputs.at(0));
    const auto fc2_found = consumers.find(swiglu->outputs.at(0));
    if (fc1_found == producers.end() ||
        fc1_found->second->opcode != ir::Opcode::Linear ||
        fc1_found->second->inputs.size() != 2U ||
        fc2_found == consumers.end() || fc2_found->second.size() != 1U ||
        fc2_found->second.front()->opcode != ir::Opcode::Linear ||
        fc2_found->second.front()->inputs.size() != 2U)
      fail("H3 groupwise route requires Linear -> SwiGlu -> Linear");
    const auto &fc1 = *fc1_found->second;
    const auto &fc2 = *fc2_found->second.front();
    const auto *fc1_weight = program.tensor(fc1.inputs.at(1));
    const auto *fc2_weight = program.tensor(fc2.inputs.at(1));
    if (!fc1_weight || !fc2_weight ||
        fc1_weight->dtype != ir::DType::BF16 ||
        fc2_weight->dtype != ir::DType::BF16 ||
        fc1_weight->dims.size() != 2U ||
        fc1_weight->dims.at(0) % 2U != 0U ||
        fc1_weight->dims.at(1) != hidden)
      fail("H3 groupwise route found incompatible MLP weights");
    const auto ffn = fc1_weight->dims.at(0) / 2U;
    if (fc2_weight->dims != std::vector<std::uint64_t>{hidden, ffn})
      fail("H3 groupwise route found incompatible FC2 geometry");

    const ir::Operation *output_linear = nullptr;
    for (std::size_t index = layout_index + 1U; index < next_layout; ++index) {
      const auto &candidate = program.operations[index];
      if (candidate.opcode != ir::Opcode::Linear ||
          candidate.inputs.size() != 2U)
        continue;
      const auto *weight = program.tensor(candidate.inputs.at(1));
      const auto *input = program.tensor(candidate.inputs.at(0));
      if (!weight || !input ||
          weight->dims != std::vector<std::uint64_t>{hidden, inner} ||
          input->dims.size() != 3U)
        continue;
      if (output_linear)
        fail("H3 groupwise route found multiple output projections in one block");
      output_linear = &candidate;
    }
    if (!output_linear)
      fail("H3 groupwise route could not locate the output projection");
    const auto *output_weight = program.tensor(output_linear->inputs.at(1));

    H3GroupwiseBlockPlan plan;
    plan.qkv_layout_operation = layout.id;
    plan.output_linear_operation = output_linear->id;
    plan.fc1_operation = fc1.id;
    plan.fc2_operation = fc2.id;
    for (std::size_t index = 0U; index < layout.outputs.size(); ++index)
      plan.qkv_weight_tensors[index] = layout.outputs[index];
    plan.output_weight_tensor = output_linear->inputs.at(1);
    plan.fc1_weight_tensor = fc1.inputs.at(1);
    plan.fc2_weight_tensor = fc2.inputs.at(1);
    plan.layer = options.h3_groupwise_layer +
                 static_cast<std::uint32_t>(result.size());
    plan.excluded_tensors = layout.outputs;
    plan.replaced_constant_tensors = {
        layout.inputs.at(0), output_linear->inputs.at(1), fc1.inputs.at(1),
        fc2.inputs.at(1)};
    plan.cache_path = options.h3_groupwise_cache;
    const auto prefix = "block." + std::to_string(plan.layer);
    const std::array<const ir::TensorDesc *, 4> semantic_weights = {
        packed, output_weight, fc1_weight, fc2_weight};
    for (std::size_t index = 0U; index < plan.projections.size(); ++index) {
      auto &projection = plan.projections[index];
      projection.weight = weights::map_safetensor(
          cache, prefix + ".weight." + std::to_string(index));
      projection.scale = weights::map_safetensor(
          cache, prefix + ".scale." + std::to_string(index));
      const auto *semantic = semantic_weights[index];
      if (!semantic || projection.weight.dtype != ir::DType::I8 ||
          projection.weight.dims != semantic->dims ||
          (projection.scale.dtype != ir::DType::F16 &&
           projection.scale.dtype != ir::DType::F32) ||
          projection.scale.dims.size() != 2U ||
          projection.scale.dims.at(0) != semantic->dims.at(0) ||
          projection.scale.dims.at(1) == 0U ||
          semantic->dims.at(1) % projection.scale.dims.at(1) != 0U)
        fail("H3 groupwise cache tensors do not match semantic projection shapes");
      projection.rows = semantic->dims.at(0);
      projection.columns = semantic->dims.at(1);
      projection.group_size = static_cast<std::uint32_t>(
          projection.columns / projection.scale.dims.at(1));
      if (projection.group_size > projection.columns)
        fail("H3 groupwise cache has an invalid group size");
      plan.quantized_weight_bytes += projection.weight.byte_size() +
                                     projection.scale.byte_size();
      plan.weight_storage_bytes += align_256(projection.weight.byte_size()) +
                                   align_256(projection.scale.byte_size());
    }
    plan.scratch_bytes = align_256(std::max(
        {packed->byte_count(), output_weight->byte_count(),
         fc1_weight->byte_count(), fc2_weight->byte_count()}));
    result.push_back(std::move(plan));
  }
  if (result.empty())
    fail("H3 groupwise cache requested but no supported H3 block was found");
  return result;
}

std::string h3_groupwise_source(bool enabled) {
  if (!enabled)
    return {};
  return R"CUDA(
extern "C" __global__ void dif_h3_groupwise_dequant(
    const signed char* weight, const void* scale, dif_bf16* output,
    unsigned long long rows, unsigned long long columns,
    unsigned group_size, unsigned swap_row_halves, unsigned scale_f32) {
  unsigned long long target =
      (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned long long stride =
      (unsigned long long)gridDim.x * blockDim.x;
  const unsigned long long elements = rows * columns;
  while (target < elements) {
    const unsigned long long target_row = target / columns;
    const unsigned long long column = target - target_row * columns;
    unsigned long long source_row = target_row;
    if (swap_row_halves != 0U) {
      const unsigned long long half = rows / 2ULL;
      source_row = target_row < half ? target_row + half : target_row - half;
    }
    const unsigned long long source = source_row * columns + column;
    const unsigned long long scale_index = source / group_size;
    const float scale_value = scale_f32 != 0U
        ? ((const float*)scale)[scale_index]
        : dif_load_f16((const dif_f16*)scale, scale_index);
    const float value = (float)weight[source] * scale_value;
    dif_store_bf16(output, target, value);
    target += stride;
  }
}
)CUDA";
}

void allocate_h3_groupwise_weights(H3GroupwiseBlockPlan &plan,
                                   DeviceArena *arena) {
  plan.weight_storage =
      std::make_unique<Workspace>(plan.weight_storage_bytes, arena);
  auto pointer = plan.weight_storage->pointer();
  for (auto &projection : plan.projections) {
    projection.weight_device = pointer;
    pointer += align_256(projection.weight.byte_size());
    projection.scale_device = pointer;
    pointer += align_256(projection.scale.byte_size());
  }
}

void assign_h3_groupwise_scratch(H3GroupwiseBlockPlan &plan,
                                 CUdeviceptr scratch,
                                 DeviceBuffers &buffers) {
  const auto qkv_segment =
      plan.projections.at(0).columns *
      (plan.projections.at(0).rows / 3U) * sizeof(std::uint16_t);
  for (std::size_t index = 0U; index < plan.qkv_weight_tensors.size(); ++index)
    buffers.bind_external(plan.qkv_weight_tensors[index],
                          scratch + index * qkv_segment);
  buffers.bind_external(plan.output_weight_tensor, scratch);
  buffers.bind_external(plan.fc1_weight_tensor, scratch);
  buffers.bind_external(plan.fc2_weight_tensor, scratch);
}

void upload_h3_groupwise_weights(const H3GroupwiseBlockPlan &plan,
                                 CUstream stream) {
  for (const auto &projection : plan.projections) {
    check(counted_memcpy_htod(projection.weight_device, projection.weight.data(),
                            projection.weight.byte_size(), stream),
          "cuMemcpyHtoDAsync H3 groupwise INT8 weight");
    check(counted_memcpy_htod(projection.scale_device, projection.scale.data(),
                            projection.scale.byte_size(), stream),
          "cuMemcpyHtoDAsync H3 groupwise F16 scale");
  }
}

void launch_h3_groupwise_dequant(const H3GroupwiseProjection &projection,
                                 CUdeviceptr scratch, bool swap_row_halves,
                                 CUfunction function, CUstream stream) {
  auto weight = projection.weight_device;
  auto scale = projection.scale_device;
  auto output = scratch;
  auto rows = projection.rows;
  auto columns = projection.columns;
  auto group_size = projection.group_size;
  auto swap = static_cast<unsigned>(swap_row_halves);
  auto scale_f32 = static_cast<unsigned>(projection.scale.dtype == ir::DType::F32);
  const auto elements = rows * columns;
  const auto grid = static_cast<unsigned>(std::min<std::uint64_t>(
      65535U, (elements + 255U) / 256U));
  std::array<void *, 8> arguments = {&weight, &scale, &output, &rows,
                                     &columns, &group_size, &swap,
                                     &scale_f32};
  check(counted_launch_kernel(function, grid, 1U, 1U, 256U, 1U, 1U, 0U, stream,
                       arguments.data(), nullptr),
        "cuLaunchKernel H3 groupwise INT8 dequant");
}

struct H3DirectInt8Config {
  std::filesystem::path path;
  std::uint32_t layer{};
  std::uint32_t layers{};
  std::uint32_t resident_layers{};
  bool convrot{};
};

H3DirectInt8Config h3_direct_int8_config(const RunOptions &options,
                                         std::uint32_t convrot_layers) {
  if (!options.h3_w8a8_cache.empty() &&
      !options.h3_convrot_int8_checkpoint.empty())
    fail("H3 W8A8 and ConvRot INT8 precision routes are mutually exclusive");
  if (!options.h3_convrot_int8_checkpoint.empty())
    return {options.h3_convrot_int8_checkpoint,
            options.h3_convrot_int8_layer,
            convrot_layers,
            options.h3_convrot_int8_resident_layers, true};
  return {options.h3_w8a8_cache, options.h3_w8a8_layer,
          std::numeric_limits<std::uint32_t>::max(),
          options.h3_w8a8_resident_layers, false};
}

bool h3_channel_scale_shape(const Tensor &scale, std::uint64_t rows) {
  return scale.dtype == ir::DType::F32 &&
         (scale.dims == std::vector<std::uint64_t>{rows} ||
          (scale.dims.size() == 2U && scale.dims[0] == rows &&
           scale.dims[1] != 0U));
}

std::uint32_t h3_channel_scale_groups(const Tensor &scale,
                                      std::uint64_t rows) {
  if (!h3_channel_scale_shape(scale, rows))
    fail("H3 channel scale tensor has invalid shape");
  const auto groups = scale.dims.size() == 1U ? 1U : scale.dims[1];
  if (groups > std::numeric_limits<std::uint32_t>::max())
    fail("H3 channel scale group count exceeds U32");
  return static_cast<std::uint32_t>(groups);
}

void validate_h3_convrot_scale_policy(std::uint32_t weight_scale_groups,
                                      std::uint64_t contraction,
                                      std::uint32_t scale_chunk) {
  if (scale_chunk == 0U) {
    if (weight_scale_groups != 1U)
      fail("chunk-scaled H3 ConvRot cache requires an explicit scale-chunk policy");
    return;
  }
  const auto expected =
      (contraction + scale_chunk - 1U) / scale_chunk;
  if (weight_scale_groups != 1U && weight_scale_groups != expected)
    fail("H3 ConvRot cache scale groups do not match the execution policy");
}

void validate_h3_convrot_metadata(const weights::SafeTensorFile &cache,
                                  std::uint32_t required_layer) {
  constexpr std::uint32_t magic = 0x44494643U; // "DIFC"
  constexpr std::uint32_t version = 1U;
  constexpr std::uint32_t chunk_scaled_version = 2U;
  constexpr std::uint32_t group_size = 256U;
  constexpr std::uint32_t qkv_layout_contiguous = 1U;
  constexpr std::uint32_t projection_count = 4U;
  const auto metadata =
      weights::map_safetensor(cache, "__meta__.h3_convrot");
  if (metadata.dtype != ir::DType::I32 || metadata.dims.size() != 1U ||
      (metadata.dims[0] != 14U && metadata.dims[0] != 15U) ||
      metadata.byte_size() != metadata.dims[0] * sizeof(std::uint32_t))
    fail("H3 ConvRot cache has an invalid identity record");
  std::array<std::uint32_t, 15> identity{};
  std::memcpy(identity.data(), metadata.data(), metadata.byte_size());
  if (identity[0] != magic ||
      (identity[1] != version && identity[1] != chunk_scaled_version) ||
      identity[2] != group_size || identity[3] != qkv_layout_contiguous ||
      identity[5] != projection_count)
    fail("H3 ConvRot cache has unsupported native format metadata");
  if (identity[1] == version && metadata.dims[0] != 14U)
    fail("H3 ConvRot v1 cache has an invalid metadata length");
  if (identity[1] == chunk_scaled_version &&
      (metadata.dims[0] != 15U || identity[6] < group_size ||
       identity[6] > 2048U || identity[6] % group_size != 0U))
    fail("H3 chunk-scaled ConvRot cache has invalid scale metadata");
  if (required_layer >= identity[4])
    fail("H3 ConvRot cache does not cover requested block " +
         std::to_string(required_layer));
}

std::vector<ConvRotInt8LinearPlan> find_convrot_int8_linear_plans(
    const ir::Program &program, const RunOptions &options) {
  if (options.convrot_int8_checkpoint.empty())
    return {};
#if !DIF_HAS_CUTLASS
  fail("generic ConvRot INT8 requires a CUTLASS-enabled NVIDIA backend");
#endif
  constexpr std::uint32_t magic = 0x31525643U;
  constexpr std::uint32_t version = 1U;
  constexpr std::uint32_t group = 256U;
  const auto cache =
      weights::read_safetensors(options.convrot_int8_checkpoint);
  const auto metadata =
      weights::map_safetensor(cache, "__meta__.convrot_int8");
  if (metadata.dtype != ir::DType::I32 ||
      metadata.dims != std::vector<std::uint64_t>{20U} ||
      metadata.byte_size() != 20U * sizeof(std::uint32_t))
    fail("generic ConvRot cache has an invalid identity record");
  std::array<std::uint32_t, 20> identity{};
  std::memcpy(identity.data(), metadata.data(), metadata.byte_size());
  if (identity[0] != magic || identity[1] != version || identity[2] != group)
    fail("generic ConvRot cache has unsupported format metadata");
  const auto fingerprint = ir::fingerprint(program);
  for (std::size_t word = 0U; word < 8U; ++word) {
    const auto expected =
        static_cast<std::uint32_t>(fingerprint[word * 4U]) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 1U]) << 8U) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 2U]) << 16U) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 3U]) << 24U);
    if (identity[4U + word] != expected)
      fail("generic ConvRot cache DiffIR fingerprint mismatch");
  }

  std::vector<ConvRotInt8LinearPlan> result;
  std::unordered_set<std::uint32_t> seen_weights;
  std::uint32_t eligible_count = 0U;
  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::Linear ||
        (operation.inputs.size() != 2U && operation.inputs.size() != 3U) ||
        operation.outputs.size() != 1U)
      continue;
    const auto *input = program.tensor(operation.inputs[0]);
    const auto *weight = program.tensor(operation.inputs[1]);
    const auto *output = program.tensor(operation.outputs[0]);
    const auto *bias = operation.inputs.size() == 3U
                           ? program.tensor(operation.inputs[2])
                           : nullptr;
    const auto eligible_dtype =
        input && (input->dtype == ir::DType::BF16 ||
                  input->dtype == ir::DType::F16);
    if (!input || !weight || !output || !eligible_dtype ||
        weight->dtype != input->dtype || output->dtype != input->dtype ||
        input->dims.empty() || weight->dims.size() != 2U ||
        weight->dims[1] == 0U || weight->dims[1] % group != 0U)
      continue;
    if (bias &&
        (input->dtype != ir::DType::F16 || bias->dtype != input->dtype ||
         bias->dims != std::vector<std::uint64_t>{weight->dims[0]} ||
         static_cast<ir::LinearBiasMode>(operation.u64(
             ir::AttrKey::LinearBiasMode,
             static_cast<std::uint64_t>(ir::LinearBiasMode::Epilogue))) !=
             ir::LinearBiasMode::Epilogue))
      continue;
    const auto rows = input->element_count() / weight->dims[1];
    if (rows == 0U || rows * weight->dims[1] != input->element_count() ||
        output->element_count() != rows * weight->dims[0])
      fail("generic ConvRot Linear has inconsistent flattened geometry");
    ++eligible_count;
    const auto weight_name =
        "linear." + std::to_string(weight->id) + ".weight";
    const auto scale_name =
        "linear." + std::to_string(weight->id) + ".scale";
    const auto *weight_entry = cache.find(weight_name);
    const auto *scale_entry = cache.find(scale_name);
    if (!weight_entry || !scale_entry)
      fail("generic ConvRot cache does not cover Linear weight tensor " +
           std::to_string(weight->id));
    auto quantized = weights::map_safetensor(cache, weight_name);
    auto scale = weights::map_safetensor(cache, scale_name);
    if (quantized.dtype != ir::DType::I8 || quantized.dims != weight->dims ||
        scale.dtype != ir::DType::F32 ||
        scale.dims != std::vector<std::uint64_t>{weight->dims[0]})
      fail("generic ConvRot cache tensor shape does not match Linear weight " +
           std::to_string(weight->id));
    seen_weights.insert(weight->id);
    if (options.convrot_int8_linear_count == 0U ||
        result.size() < options.convrot_int8_linear_count)
      result.push_back({operation.id,
                        input->id,
                        weight->id,
                        output->id,
                        bias ? bias->id : 0U,
                        input->dtype,
                        rows,
                        weight->dims[0],
                        weight->dims[1],
                        std::move(quantized),
                        std::move(scale),
                        options.convrot_int8_checkpoint});
  }
  if (result.empty())
    fail("generic ConvRot checkpoint requested but no eligible Linear was found");
  if (identity[3] != seen_weights.size())
    fail("generic ConvRot cache projection count does not match the program");
  if (options.convrot_int8_linear_count > eligible_count)
    fail("generic ConvRot Linear count exceeds eligible program Linears");
  return result;
}

std::vector<H3W8A8MlpPlan> find_h3_w8a8_mlp_plans(
    const ir::Program &program, const RunOptions &options) {
  const auto config = h3_direct_int8_config(
      options, options.h3_convrot_int8_mlp_layers);
  if (config.path.empty())
    return {};
  if (options.h3_int8_mlp_chunk_rows == 0U)
    fail("H3 direct INT8 MLP chunk rows must be positive");
  const auto cache = weights::read_safetensors(config.path);
  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> consumers;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::vector<H3W8A8MlpPlan> result;
  for (const auto &swiglu : program.operations) {
    if (result.size() >= config.layers)
      break;
    if (swiglu.opcode != ir::Opcode::SwiGlu || swiglu.inputs.size() != 1U ||
        swiglu.outputs.size() != 1U)
      continue;
    const auto fc1_found = producer.find(swiglu.inputs.at(0));
    if (fc1_found == producer.end() ||
        fc1_found->second->opcode != ir::Opcode::Linear ||
        consumers[swiglu.inputs.at(0)].size() != 1U)
      continue;
    const auto &fc1 = *fc1_found->second;
    // DiffIR and the official checkpoint both bind FC1 as [gate|value].
    if (fc1.inputs.size() != 2U || fc1.outputs.size() != 1U ||
        !swiglu.boolean(ir::AttrKey::GateFirst, false))
      continue;
    const auto activation_consumers = consumers[swiglu.outputs.at(0)];
    if (activation_consumers.size() != 1U ||
        activation_consumers.front()->opcode != ir::Opcode::Linear)
      continue;
    const auto &fc2 = *activation_consumers.front();
    if (fc2.inputs.size() != 2U || fc2.outputs.size() != 1U)
      continue;
    const auto projected_consumers = consumers[fc2.outputs.at(0)];
    if (projected_consumers.size() != 1U ||
        projected_consumers.front()->opcode != ir::Opcode::ResidualGate)
      continue;
    const auto &residual = *projected_consumers.front();
    if (residual.inputs.size() != 3U || residual.outputs.size() != 1U ||
        residual.inputs.at(1) != fc2.outputs.at(0))
      continue;

    const auto *input = program.tensor(fc1.inputs.at(0));
    const auto *fc1_weight = program.tensor(fc1.inputs.at(1));
    const auto *activation = program.tensor(swiglu.outputs.at(0));
    const auto *fc2_weight = program.tensor(fc2.inputs.at(1));
    const auto *residual_input = program.tensor(residual.inputs.at(0));
    const auto *gate = program.tensor(residual.inputs.at(2));
    const auto *output = program.tensor(residual.outputs.at(0));
    if (!input || !fc1_weight || !activation || !fc2_weight ||
        !residual_input || !gate || !output ||
        input->dtype != ir::DType::BF16 ||
        fc1_weight->dtype != ir::DType::BF16 ||
        fc2_weight->dtype != ir::DType::BF16 ||
        residual_input->dtype != ir::DType::BF16 ||
        gate->dtype != ir::DType::BF16 || output->dtype != ir::DType::BF16 ||
        fc1_weight->dims.size() != 2U || fc2_weight->dims.size() != 2U ||
        input->dims.size() != 2U || activation->dims.size() != 2U ||
        input->dims != residual_input->dims || input->dims != gate->dims ||
        input->dims != output->dims || fc1_weight->dims.at(0) % 2U != 0U)
      continue;
    const auto hidden = input->dims.at(1);
    const auto packed_ffn = fc1_weight->dims.at(0);
    const auto ffn = packed_ffn / 2U;
    if (fc1_weight->dims.at(1) != hidden ||
        activation->dims != std::vector<std::uint64_t>{input->dims.at(0), ffn} ||
        fc2_weight->dims != std::vector<std::uint64_t>{hidden, ffn})
      continue;
    for (const auto tensor_id :
         {fc1.outputs.at(0), swiglu.outputs.at(0), fc2.outputs.at(0)}) {
      const auto *tensor = program.tensor(tensor_id);
      if (!tensor || tensor->roles !=
                         static_cast<std::uint32_t>(ir::TensorRole::Internal))
        fail("H3 W8A8 MLP requires unobservable exclusive intermediates");
    }

    const auto layer = config.layer +
                       static_cast<std::uint32_t>(result.size());
    const auto prefix = "block." + std::to_string(layer);
    H3W8A8MlpPlan plan;
    plan.fc1_operation = fc1.id;
    plan.swiglu_operation = swiglu.id;
    plan.fc2_operation = fc2.id;
    plan.residual_operation = residual.id;
    plan.input_tensor = fc1.inputs.at(0);
    plan.residual_tensor = residual.inputs.at(0);
    plan.gate_tensor = residual.inputs.at(2);
    plan.output_tensor = residual.outputs.at(0);
    plan.layer = layer;
    plan.chunk_rows = options.h3_int8_mlp_chunk_rows;
    plan.resident = layer - config.layer < config.resident_layers;
    plan.convrot = config.convrot;
    plan.cutlass_scaled_fc1 = config.convrot &&
                              (options.h3_int8_cutlass_scaled_fc1 ||
                               options.h3_int8_cutlass_scaled_all);
    plan.cutlass_scaled_fc2 =
        config.convrot && options.h3_int8_cutlass_scaled_all;
    plan.convrot_scale_chunk =
        config.convrot ? options.h3_int8_convrot_scale_chunk : 0U;
    plan.convrot_global_activation_scale =
        config.convrot && options.h3_int8_convrot_global_activation_scale;
    plan.rows = input->dims.at(0);
    plan.hidden = hidden;
    plan.ffn = ffn;
    plan.packed_ffn = packed_ffn;
    plan.excluded_tensors = {fc1.outputs.at(0), swiglu.outputs.at(0),
                             fc2.outputs.at(0)};
    plan.replaced_constant_tensors = {fc1.inputs.at(1), fc2.inputs.at(1)};
    if (config.convrot) {
      validate_h3_convrot_metadata(cache, layer);
      plan.fc1_weight =
          weights::map_safetensor(cache, prefix + ".convrot_weight.2");
      plan.fc1_scale =
          weights::map_safetensor(cache, prefix + ".convrot_scale.2");
      plan.fc2_weight =
          weights::map_safetensor(cache, prefix + ".convrot_weight.3");
      plan.fc2_scale =
          weights::map_safetensor(cache, prefix + ".convrot_scale.3");
    } else {
      plan.fc1_weight =
          weights::map_safetensor(cache, prefix + ".weight.2");
      plan.fc1_scale =
          weights::map_safetensor(cache, prefix + ".scale.2");
      plan.fc2_weight =
          weights::map_safetensor(cache, prefix + ".weight.3");
      plan.fc2_scale =
          weights::map_safetensor(cache, prefix + ".scale.3");
    }
    if (plan.fc1_weight.dtype != ir::DType::I8 ||
        plan.fc1_weight.dims != fc1_weight->dims ||
        !h3_channel_scale_shape(plan.fc1_scale, packed_ffn) ||
        plan.fc2_weight.dtype != ir::DType::I8 ||
        plan.fc2_weight.dims != fc2_weight->dims ||
        !h3_channel_scale_shape(plan.fc2_scale, hidden))
      fail("H3 W8A8 cache tensors do not match the semantic MLP shapes");
    plan.fc1_weight_scale_groups =
        h3_channel_scale_groups(plan.fc1_scale, packed_ffn);
    plan.fc2_weight_scale_groups =
        h3_channel_scale_groups(plan.fc2_scale, hidden);
    validate_h3_convrot_scale_policy(plan.fc1_weight_scale_groups, hidden,
                                     plan.convrot_scale_chunk);
    validate_h3_convrot_scale_policy(plan.fc2_weight_scale_groups, ffn,
                                     plan.convrot_scale_chunk);
    plan.quantized_weight_bytes =
        plan.fc1_weight.byte_size() + plan.fc1_scale.byte_size() +
        plan.fc2_weight.byte_size() + plan.fc2_scale.byte_size();
    plan.weight_storage_bytes =
        align_256(plan.fc1_weight.byte_size()) +
        align_256(plan.fc1_scale.byte_size()) +
        align_256(plan.fc2_weight.byte_size()) +
        align_256(plan.fc2_scale.byte_size());
    for (const auto tensor_id : plan.excluded_tensors)
      plan.eliminated_intermediate_bytes +=
          program.tensor(tensor_id)->byte_count();
    const auto chunk_rows =
        std::min<std::uint64_t>(plan.rows, plan.chunk_rows);
    const auto input_scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                        ? 1U
                                        : (hidden + plan.convrot_scale_chunk - 1U) /
                                              plan.convrot_scale_chunk;
    const auto activation_scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                             ? 1U
                                             : (ffn + plan.convrot_scale_chunk - 1U) /
                                                   plan.convrot_scale_chunk;
    plan.scratch_bytes =
        align_256(chunk_rows * input_scale_groups * sizeof(float)) +
        align_256(chunk_rows * hidden) +
        align_256(chunk_rows * packed_ffn *
                  (plan.cutlass_scaled_fc1 ? sizeof(std::uint16_t)
                                           : sizeof(std::int32_t))) +
        (plan.convrot_scale_chunk == 0U
             ? 0U
             : align_256(chunk_rows * packed_ffn * sizeof(float))) +
        align_256(chunk_rows * ffn * sizeof(std::uint16_t)) +
        align_256(chunk_rows * activation_scale_groups * sizeof(float)) +
        align_256(chunk_rows * ffn) +
        align_256(chunk_rows * hidden *
                  (plan.cutlass_scaled_fc2 ? sizeof(std::uint16_t)
                                           : sizeof(std::int32_t)));
    if (plan.convrot_scale_chunk != 0U)
      plan.scratch_bytes +=
          align_256(chunk_rows * hidden * sizeof(float));
    plan.cache_path = config.path;
    result.push_back(std::move(plan));
  }
  return result;
}

std::vector<H3W8A8AttentionPlan> find_h3_w8a8_attention_plans(
    const ir::Program &program, const RunOptions &options) {
  const auto config = h3_direct_int8_config(
      options, options.h3_convrot_int8_attention_layers);
  if (config.path.empty())
    return {};
  const auto cache = weights::read_safetensors(config.path);
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> consumers;
  for (const auto &operation : program.operations) {
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::vector<H3W8A8AttentionPlan> result;
  std::unordered_set<std::uint32_t> planned_output_linears;
  for (const auto &layout : program.operations) {
    if (result.size() >= config.layers)
      break;
    if (layout.opcode != ir::Opcode::H3DeinterleaveQkvWeight ||
        layout.inputs.size() != 1U || layout.outputs.size() != 3U)
      continue;
    std::array<const ir::Operation *, 3> qkv_linears{};
    bool valid = true;
    for (std::size_t index = 0U; index < qkv_linears.size(); ++index) {
      const auto found = consumers.find(layout.outputs.at(index));
      if (found == consumers.end() || found->second.size() != 1U ||
          found->second.front()->opcode != ir::Opcode::Linear ||
          found->second.front()->inputs.size() != 2U ||
          found->second.front()->outputs.size() != 1U ||
          found->second.front()->inputs.at(1) != layout.outputs.at(index)) {
        valid = false;
        break;
      }
      qkv_linears[index] = found->second.front();
    }
    if (!valid || qkv_linears[0]->inputs.at(0) != qkv_linears[1]->inputs.at(0) ||
        qkv_linears[0]->inputs.at(0) != qkv_linears[2]->inputs.at(0))
      continue;

    // A full H3 denoiser places context-refiner QKV chains before the creator
    // transformer blocks. They use plain RMSNorm and Add residuals, so mapping
    // block.0 W8A8 weights onto them would be a silent semantic corruption.
    // Projection-only toolbox programs remain admitted when their QKV outputs
    // are explicitly observable.
    const auto q_consumers = consumers.find(qkv_linears[0]->outputs.at(0));
    const auto k_consumers = consumers.find(qkv_linears[1]->outputs.at(0));
    const auto transformer_qk =
        q_consumers != consumers.end() && k_consumers != consumers.end() &&
        q_consumers->second.size() == 1U && k_consumers->second.size() == 1U &&
        q_consumers->second.front()->opcode ==
            ir::Opcode::QkNormPartialRope &&
        k_consumers->second.front()->opcode ==
            ir::Opcode::QkNormPartialRope;
    const auto exposed_projection = std::all_of(
        qkv_linears.begin(), qkv_linears.end(), [&](const auto *linear) {
          const auto *description = program.tensor(linear->outputs.at(0));
          return description && description->has_role(ir::TensorRole::Output);
        });
    if (!transformer_qk && !exposed_projection)
      continue;

    const auto *attention_input = program.tensor(qkv_linears[0]->inputs.at(0));
    const auto *packed_weight = program.tensor(layout.inputs.at(0));
    const auto *q_output = program.tensor(qkv_linears[0]->outputs.at(0));
    if (!attention_input || !packed_weight || !q_output ||
        attention_input->dtype != ir::DType::BF16 ||
        packed_weight->dtype != ir::DType::BF16 ||
        q_output->dtype != ir::DType::BF16 ||
        attention_input->dims.size() != 2U || packed_weight->dims.size() != 2U ||
        q_output->dims.size() != 3U)
      continue;
    const auto rows = attention_input->dims.at(0);
    const auto hidden = attention_input->dims.at(1);
    const auto inner = q_output->dims.at(1) * q_output->dims.at(2);
    const auto packed_inner = 3U * inner;
    if (packed_weight->dims !=
        std::vector<std::uint64_t>{packed_inner, hidden})
      continue;
    for (std::size_t index = 0U; index < qkv_linears.size(); ++index) {
      const auto *weight = program.tensor(layout.outputs.at(index));
      const auto *output = program.tensor(qkv_linears[index]->outputs.at(0));
      if (!weight || !output || weight->dtype != ir::DType::BF16 ||
          weight->dims != std::vector<std::uint64_t>{inner, hidden} ||
          output->dtype != ir::DType::BF16 || output->dims != q_output->dims) {
        valid = false;
        break;
      }
    }
    if (!valid)
      continue;

    const ir::Operation *output_linear = nullptr;
    const ir::Operation *residual = nullptr;
    const auto q_norm_found =
        consumers.find(qkv_linears[0]->outputs.at(0));
    const auto k_norm_found =
        consumers.find(qkv_linears[1]->outputs.at(0));
    if (q_norm_found != consumers.end() &&
        k_norm_found != consumers.end() &&
        q_norm_found->second.size() == 1U &&
        k_norm_found->second.size() == 1U &&
        q_norm_found->second.front()->opcode ==
            ir::Opcode::QkNormPartialRope &&
        k_norm_found->second.front()->opcode ==
            ir::Opcode::QkNormPartialRope &&
        q_norm_found->second.front()->outputs.size() == 1U &&
        k_norm_found->second.front()->outputs.size() == 1U) {
      const auto q_attention_found =
          consumers.find(q_norm_found->second.front()->outputs.at(0));
      const auto k_attention_found =
          consumers.find(k_norm_found->second.front()->outputs.at(0));
      const auto v_attention_found =
          consumers.find(qkv_linears[2]->outputs.at(0));
      if (q_attention_found != consumers.end() &&
          k_attention_found != consumers.end() &&
          v_attention_found != consumers.end() &&
          q_attention_found->second.size() == 1U &&
          k_attention_found->second.size() == 1U &&
          v_attention_found->second.size() == 1U &&
          q_attention_found->second.front() ==
              k_attention_found->second.front() &&
          q_attention_found->second.front() ==
              v_attention_found->second.front() &&
          q_attention_found->second.front()->opcode == ir::Opcode::Attention &&
          q_attention_found->second.front()->outputs.size() == 1U) {
        const auto output_found = consumers.find(
            q_attention_found->second.front()->outputs.at(0));
        if (output_found != consumers.end() &&
            output_found->second.size() == 1U &&
            output_found->second.front()->opcode == ir::Opcode::Linear &&
            output_found->second.front()->inputs.size() == 2U &&
            output_found->second.front()->outputs.size() == 1U) {
          const auto *candidate = output_found->second.front();
          const auto *weight = program.tensor(candidate->inputs.at(1));
          const auto *output = program.tensor(candidate->outputs.at(0));
          const auto residual_found =
              consumers.find(candidate->outputs.at(0));
          if (weight && output &&
              weight->dims ==
                  std::vector<std::uint64_t>{hidden, inner} &&
              output->dims == attention_input->dims &&
              residual_found != consumers.end() &&
              residual_found->second.size() == 1U &&
              residual_found->second.front()->opcode ==
                  ir::Opcode::ResidualGate) {
            output_linear = candidate;
            residual = residual_found->second.front();
          }
        }
      }
    }
    const auto has_output_projection = output_linear && residual;
    if (has_output_projection &&
        (residual->inputs.size() != 3U || residual->outputs.size() != 1U ||
         residual->inputs.at(1) != output_linear->outputs.at(0)))
      continue;
    if (has_output_projection) {
      const auto *residual_input = program.tensor(residual->inputs.at(0));
      const auto *gate = program.tensor(residual->inputs.at(2));
      const auto *block_output = program.tensor(residual->outputs.at(0));
      if (!residual_input || !gate || !block_output ||
          residual_input->dtype != ir::DType::BF16 ||
          gate->dtype != ir::DType::BF16 ||
          block_output->dtype != ir::DType::BF16 ||
          residual_input->dims != attention_input->dims ||
          gate->dims != attention_input->dims ||
          block_output->dims != attention_input->dims)
        continue;
    }

    const auto layer = config.layer +
                       static_cast<std::uint32_t>(result.size());
    const auto prefix = "block." + std::to_string(layer);
    H3W8A8AttentionPlan plan;
    plan.has_qkv_projection = true;
    plan.has_output_projection = has_output_projection;
    plan.qkv_layout_operation = layout.id;
    for (std::size_t index = 0U; index < qkv_linears.size(); ++index) {
      plan.qkv_linear_operations[index] = qkv_linears[index]->id;
      plan.qkv_output_tensors[index] = qkv_linears[index]->outputs.at(0);
    }
    plan.attention_input_tensor = qkv_linears[0]->inputs.at(0);
    plan.excluded_tensors = {layout.outputs.at(0), layout.outputs.at(1),
                             layout.outputs.at(2)};
    plan.replaced_constant_tensors = {layout.inputs.at(0)};
    if (has_output_projection) {
      plan.output_linear_operation = output_linear->id;
      plan.residual_operation = residual->id;
      plan.output_input_tensor = output_linear->inputs.at(0);
      plan.residual_tensor = residual->inputs.at(0);
      plan.gate_tensor = residual->inputs.at(2);
      plan.output_tensor = residual->outputs.at(0);
      plan.excluded_tensors.push_back(output_linear->outputs.at(0));
      plan.replaced_constant_tensors.push_back(output_linear->inputs.at(1));
      planned_output_linears.insert(output_linear->id);
    }
    plan.layer = layer;
    plan.resident = layer - config.layer < config.resident_layers;
    plan.convrot = config.convrot;
    plan.cutlass_scaled =
        config.convrot && options.h3_int8_cutlass_scaled_all;
    plan.convrot_scale_chunk =
        config.convrot ? options.h3_int8_convrot_scale_chunk : 0U;
    plan.convrot_global_activation_scale =
        config.convrot && options.h3_int8_convrot_global_activation_scale;
    plan.rows = rows;
    plan.hidden = hidden;
    plan.inner = inner;
    plan.packed_inner = packed_inner;
    plan.head_dim = q_output->dims.at(2);
    if (config.convrot) {
      validate_h3_convrot_metadata(cache, layer);
      plan.qkv_weight =
          weights::map_safetensor(cache, prefix + ".convrot_weight.0");
      plan.qkv_scale =
          weights::map_safetensor(cache, prefix + ".convrot_scale.0");
    } else {
      plan.qkv_weight =
          weights::map_safetensor(cache, prefix + ".weight.0");
      plan.qkv_scale =
          weights::map_safetensor(cache, prefix + ".scale.0");
    }
    if (plan.qkv_weight.dtype != ir::DType::I8 ||
        plan.qkv_weight.dims != packed_weight->dims ||
        !h3_channel_scale_shape(plan.qkv_scale, packed_inner))
      fail("H3 W8A8 cache tensors do not match the semantic QKV shapes");
    plan.qkv_weight_scale_groups =
        h3_channel_scale_groups(plan.qkv_scale, packed_inner);
    validate_h3_convrot_scale_policy(plan.qkv_weight_scale_groups, hidden,
                                     plan.convrot_scale_chunk);
    plan.quantized_weight_bytes =
        plan.qkv_weight.byte_size() + plan.qkv_scale.byte_size();
    plan.weight_storage_bytes = align_256(plan.qkv_weight.byte_size()) +
                                align_256(plan.qkv_scale.byte_size());
    if (has_output_projection) {
      if (config.convrot) {
        plan.output_weight =
            weights::map_safetensor(cache, prefix + ".convrot_weight.1");
        plan.output_scale =
            weights::map_safetensor(cache, prefix + ".convrot_scale.1");
      } else {
        plan.output_weight =
            weights::map_safetensor(cache, prefix + ".weight.1");
        plan.output_scale =
            weights::map_safetensor(cache, prefix + ".scale.1");
      }
      if (plan.output_weight.dtype != ir::DType::I8 ||
          plan.output_weight.dims !=
              std::vector<std::uint64_t>{hidden, inner} ||
          !h3_channel_scale_shape(plan.output_scale, hidden))
        fail("H3 W8A8 cache tensors do not match output projection shapes");
      plan.output_weight_scale_groups =
          h3_channel_scale_groups(plan.output_scale, hidden);
      validate_h3_convrot_scale_policy(plan.output_weight_scale_groups, inner,
                                       plan.convrot_scale_chunk);
      plan.quantized_weight_bytes +=
          plan.output_weight.byte_size() + plan.output_scale.byte_size();
      plan.weight_storage_bytes +=
          align_256(plan.output_weight.byte_size()) +
          align_256(plan.output_scale.byte_size());
    }
    for (const auto tensor_id : plan.excluded_tensors)
      plan.eliminated_intermediate_bytes +=
          program.tensor(tensor_id)->byte_count();
    const auto chunk_rows = std::min<std::uint64_t>(
        rows, kH3W8A8ProjectionChunkRows);
    const auto qkv_scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                      ? 1U
                                      : (hidden + plan.convrot_scale_chunk - 1U) /
                                            plan.convrot_scale_chunk;
    plan.scratch_bytes = align_256(rows * qkv_scale_groups * sizeof(float)) +
                         align_256(rows * hidden) +
                         align_256(chunk_rows * packed_inner *
                                   (plan.cutlass_scaled
                                        ? sizeof(std::uint16_t)
                                        : sizeof(std::int32_t)));
    if (plan.convrot_scale_chunk != 0U)
      plan.scratch_bytes +=
          align_256(chunk_rows * packed_inner * sizeof(float));
    plan.cache_path = config.path;
    result.push_back(std::move(plan));
  }
  for (const auto &output_linear : program.operations) {
    if (result.size() >= config.layers)
      break;
    if (output_linear.opcode != ir::Opcode::Linear ||
        output_linear.inputs.size() != 2U ||
        output_linear.outputs.size() != 1U ||
        planned_output_linears.contains(output_linear.id))
      continue;
    const auto found = consumers.find(output_linear.outputs.at(0));
    if (found == consumers.end() || found->second.size() != 1U ||
        found->second.front()->opcode != ir::Opcode::ResidualGate)
      continue;
    const auto &residual = *found->second.front();
    if (residual.inputs.size() != 3U || residual.outputs.size() != 1U ||
        residual.inputs.at(1) != output_linear.outputs.at(0))
      continue;
    const auto *input = program.tensor(output_linear.inputs.at(0));
    const auto *weight = program.tensor(output_linear.inputs.at(1));
    const auto *projected = program.tensor(output_linear.outputs.at(0));
    const auto *residual_input = program.tensor(residual.inputs.at(0));
    const auto *gate = program.tensor(residual.inputs.at(2));
    const auto *output = program.tensor(residual.outputs.at(0));
    if (!input || !weight || !projected || !residual_input || !gate ||
        !output || input->dtype != ir::DType::BF16 ||
        weight->dtype != ir::DType::BF16 ||
        projected->dtype != ir::DType::BF16 ||
        residual_input->dtype != ir::DType::BF16 ||
        gate->dtype != ir::DType::BF16 || output->dtype != ir::DType::BF16 ||
        input->dims.size() != 3U || weight->dims.size() != 2U ||
        projected->dims.size() != 2U ||
        projected->dims != residual_input->dims ||
        projected->dims != gate->dims || projected->dims != output->dims)
      continue;
    const auto rows = input->dims.at(0);
    const auto inner = input->dims.at(1) * input->dims.at(2);
    const auto hidden = projected->dims.at(1);
    if (projected->dims.at(0) != rows ||
        weight->dims != std::vector<std::uint64_t>{hidden, inner})
      continue;
    const auto layer = config.layer +
                       static_cast<std::uint32_t>(result.size());
    const auto prefix = "block." + std::to_string(layer);
    H3W8A8AttentionPlan plan;
    plan.has_output_projection = true;
    plan.output_linear_operation = output_linear.id;
    plan.residual_operation = residual.id;
    plan.output_input_tensor = output_linear.inputs.at(0);
    plan.residual_tensor = residual.inputs.at(0);
    plan.gate_tensor = residual.inputs.at(2);
    plan.output_tensor = residual.outputs.at(0);
    plan.excluded_tensors = {output_linear.outputs.at(0)};
    plan.replaced_constant_tensors = {output_linear.inputs.at(1)};
    plan.layer = layer;
    plan.resident = layer - config.layer < config.resident_layers;
    plan.convrot = config.convrot;
    plan.cutlass_scaled =
        config.convrot && options.h3_int8_cutlass_scaled_all;
    plan.convrot_scale_chunk =
        config.convrot ? options.h3_int8_convrot_scale_chunk : 0U;
    plan.convrot_global_activation_scale =
        config.convrot && options.h3_int8_convrot_global_activation_scale;
    plan.rows = rows;
    plan.hidden = hidden;
    plan.inner = inner;
    plan.packed_inner = 3U * inner;
    if (config.convrot) {
      validate_h3_convrot_metadata(cache, layer);
      plan.output_weight =
          weights::map_safetensor(cache, prefix + ".convrot_weight.1");
      plan.output_scale =
          weights::map_safetensor(cache, prefix + ".convrot_scale.1");
    } else {
      plan.output_weight =
          weights::map_safetensor(cache, prefix + ".weight.1");
      plan.output_scale =
          weights::map_safetensor(cache, prefix + ".scale.1");
    }
    if (plan.output_weight.dtype != ir::DType::I8 ||
        plan.output_weight.dims != weight->dims ||
        !h3_channel_scale_shape(plan.output_scale, hidden))
      fail("H3 W8A8 cache tensors do not match output projection shapes");
    plan.output_weight_scale_groups =
        h3_channel_scale_groups(plan.output_scale, hidden);
    validate_h3_convrot_scale_policy(plan.output_weight_scale_groups, inner,
                                     plan.convrot_scale_chunk);
    plan.quantized_weight_bytes =
        plan.output_weight.byte_size() + plan.output_scale.byte_size();
    plan.weight_storage_bytes = align_256(plan.output_weight.byte_size()) +
                                align_256(plan.output_scale.byte_size());
    plan.eliminated_intermediate_bytes = projected->byte_count();
    const auto chunk_rows = std::min<std::uint64_t>(
        rows, kH3W8A8ProjectionChunkRows);
    const auto output_scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                         ? 1U
                                         : (inner + plan.convrot_scale_chunk - 1U) /
                                               plan.convrot_scale_chunk;
    plan.scratch_bytes = align_256(chunk_rows * output_scale_groups * sizeof(float)) +
                         align_256(chunk_rows * inner) +
                         align_256(chunk_rows * hidden *
                                   (plan.cutlass_scaled
                                        ? sizeof(std::uint16_t)
                                        : sizeof(std::int32_t)));
    if (plan.convrot_scale_chunk != 0U)
      plan.scratch_bytes +=
          align_256(chunk_rows * hidden * sizeof(float));
    plan.cache_path = config.path;
    result.push_back(std::move(plan));
  }
  return result;
}

std::vector<H3CompactAdaLNPlan> find_h3_compact_adaln_plans(
    const ir::Program &program, const RunOptions &options,
    std::vector<H3W8A8AttentionPlan> &attention_plans,
    std::vector<H3W8A8MlpPlan> &mlp_plans) {
  if (!options.h3_int8_compact_adaln)
    return {};
  if (options.h3_convrot_int8_checkpoint.empty() ||
      options.h3_modulation_cache.empty())
    fail("H3 compact AdaLN requires ConvRot INT8 and a prepared modulation cache");
  if (attention_plans.empty() || mlp_plans.empty())
    fail("H3 compact AdaLN requires complete direct INT8 attention and MLP plans");

  std::unordered_map<std::uint32_t, const ir::Operation *> producer;
  std::unordered_map<std::uint32_t, const ir::Operation *> operation_by_id;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> consumers;
  for (const auto &operation : program.operations) {
    operation_by_id.emplace(operation.id, &operation);
    for (const auto output : operation.outputs)
      producer.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }
  const auto single_consumer_is = [&](std::uint32_t tensor,
                                      const ir::Operation *operation) {
    const auto found = consumers.find(tensor);
    return found != consumers.end() && found->second.size() == 1U &&
           found->second.front() == operation;
  };

  std::vector<H3CompactAdaLNPlan> result;
  for (const auto &select : program.operations) {
    if (select.opcode != ir::Opcode::H3AdaLNSelect)
      continue;
    if (select.inputs.size() != 2U || select.outputs.size() != 6U)
      fail("H3 compact AdaLN found a malformed select operation");
    auto attention = std::find_if(
        attention_plans.begin(), attention_plans.end(),
        [&](const H3W8A8AttentionPlan &plan) {
          return plan.has_qkv_projection && plan.has_output_projection &&
                 plan.gate_tensor == select.outputs.at(2);
        });
    auto mlp = std::find_if(mlp_plans.begin(), mlp_plans.end(),
                            [&](const H3W8A8MlpPlan &plan) {
                              return plan.gate_tensor == select.outputs.at(5);
                            });
    if (attention == attention_plans.end() || mlp == mlp_plans.end())
      continue;
    if (!attention->convrot || !mlp->convrot || attention->layer != mlp->layer)
      fail("H3 compact AdaLN matched inconsistent ConvRot block plans");

    const auto attention_norm_found = producer.find(attention->attention_input_tensor);
    const auto mlp_norm_found = producer.find(mlp->input_tensor);
    if (attention_norm_found == producer.end() ||
        mlp_norm_found == producer.end())
      fail("H3 compact AdaLN normalized input has no producer");
    const auto *attention_norm = attention_norm_found->second;
    const auto *mlp_norm = mlp_norm_found->second;
    const auto valid_norm = [&](const ir::Operation *norm,
                                std::uint32_t scale,
                                std::uint32_t shift) {
      if (!norm || norm->opcode != ir::Opcode::RmsNormModulate ||
          norm->inputs.size() != 4U || norm->outputs.size() != 1U ||
          norm->inputs.at(2) != scale || norm->inputs.at(3) != shift ||
          norm->u64(ir::AttrKey::ModulationLayout,
                    static_cast<std::uint64_t>(
                        ir::ModulationLayout::ExplicitScaleShift)) !=
              static_cast<std::uint64_t>(
                  ir::ModulationLayout::ExplicitScaleShift) ||
          norm->u64(ir::AttrKey::BlockSize, 256U) != 256U ||
          !(norm->f64(ir::AttrKey::Epsilon, 1.0e-5) > 0.0))
        return false;
      const auto *input = program.tensor(norm->inputs.at(0));
      const auto *weight = program.tensor(norm->inputs.at(1));
      const auto *output = program.tensor(norm->outputs.at(0));
      return input && weight && output && input->dtype == ir::DType::BF16 &&
             output->dtype == ir::DType::BF16 && input->dims == output->dims &&
             input->dims == std::vector<std::uint64_t>{attention->rows,
                                                        attention->hidden} &&
             weight->dtype == ir::DType::BF16 &&
             weight->dims == std::vector<std::uint64_t>{attention->hidden};
    };
    if (!valid_norm(attention_norm, select.outputs.at(1),
                    select.outputs.at(0)) ||
        !valid_norm(mlp_norm, select.outputs.at(4), select.outputs.at(3)))
      fail("H3 compact AdaLN requires the creator BF16 RMSNorm/modulation chain");
    if (attention_norm->outputs.at(0) != attention->attention_input_tensor ||
        mlp_norm->outputs.at(0) != mlp->input_tensor ||
        !operation_by_id.contains(attention->residual_operation) ||
        !operation_by_id.contains(mlp->residual_operation) ||
        !single_consumer_is(select.outputs.at(0), attention_norm) ||
        !single_consumer_is(select.outputs.at(1), attention_norm) ||
        !single_consumer_is(select.outputs.at(2),
                            operation_by_id.at(attention->residual_operation)) ||
        !single_consumer_is(select.outputs.at(3), mlp_norm) ||
        !single_consumer_is(select.outputs.at(4), mlp_norm) ||
        !single_consumer_is(select.outputs.at(5),
                            operation_by_id.at(mlp->residual_operation)))
      fail("H3 compact AdaLN select outputs have an unsupported extra consumer");

    const auto bind = [&](H3CompactAdaLNBinding &binding,
                          const ir::Operation &norm, std::uint32_t scale_lane,
                          std::uint32_t shift_lane, std::uint32_t gate_lane) {
      binding.enabled = true;
      binding.norm_operation = norm.id;
      binding.norm_input_tensor = norm.inputs.at(0);
      binding.norm_weight_tensor = norm.inputs.at(1);
      binding.modulation_tensor = select.inputs.at(0);
      binding.indices_tensor = select.inputs.at(1);
      binding.scale_lane = scale_lane;
      binding.shift_lane = shift_lane;
      binding.gate_lane = gate_lane;
      binding.epsilon = static_cast<float>(
          norm.f64(ir::AttrKey::Epsilon, 1.0e-5));
    };
    bind(attention->compact_adaln, *attention_norm, 1U, 0U, 2U);
    bind(mlp->compact_adaln, *mlp_norm, 4U, 3U, 5U);

    H3CompactAdaLNPlan compact;
    compact.select_operation = select.id;
    std::copy(select.outputs.begin(), select.outputs.end(),
              compact.expanded_tensors.begin());
    compact.norm_operations = {attention_norm->id, mlp_norm->id};
    compact.normalized_tensors = {attention_norm->outputs.at(0),
                                  mlp_norm->outputs.at(0)};
    compact.layer = attention->layer;
    result.push_back(compact);
  }
  const auto complete_attention_count = static_cast<std::size_t>(std::count_if(
      attention_plans.begin(), attention_plans.end(),
      [](const H3W8A8AttentionPlan &plan) {
        return plan.has_qkv_projection && plan.has_output_projection;
      }));
  if (result.size() != mlp_plans.size() ||
      result.size() != complete_attention_count)
    fail("H3 compact AdaLN did not cover every direct INT8 transformer block");
  return result;
}

std::string h3_w8a8_source(bool enabled) {
  if (!enabled)
    return {};
  return R"CUDA(
extern "C" __global__ void dif_h3_w8a8_rowscale(
    const dif_bf16* x,float* scale,int row_start,int cols,int rows){
  __shared__ float values[256];int row=(int)blockIdx.x,tid=(int)threadIdx.x;
  if(row>=rows)return;float maximum=0.0f;unsigned long long base=
      (unsigned long long)(row_start+row)*(unsigned long long)cols;
  for(int col=tid;col<cols;col+=256){float value=dif_load_bf16(x,base+col);
    float magnitude=value>=0.0f?value:-value;if(magnitude>maximum)maximum=magnitude;}
  values[tid]=maximum;__syncthreads();for(int active=128;active>0;active>>=1){
    if(tid<active&&values[tid+active]>values[tid])values[tid]=values[tid+active];
    __syncthreads();}if(tid==0){float value=values[0]/127.0f;
    scale[row]=value<1.0e-30f?1.0e-30f:value;}}
extern "C" __global__ void dif_h3_w8a8_encode(
    const dif_bf16* x,const float* scale,signed char* output,
    int row_start,int rows,int cols){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*cols;while(i<total){
    int row=(int)(i/(unsigned long long)cols);float value=dif_load_bf16(
      x,(unsigned long long)row_start*cols+i);int q=(int)rintf(value/scale[row]);
    q=q>127?127:(q<-127?-127:q);output[i]=(signed char)q;i+=stride;}}
extern "C" __global__ void dif_h3_w8a8_qkv(
    const int* accumulator,const float* x_scale,const float* w_scale,
    dif_bf16* q,dif_bf16* k,dif_bf16* v,
    int row_start,int rows,int inner){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*inner;int packed=3*inner;
  while(i<total){int row=(int)(i/(unsigned long long)inner),col=(int)(i%(unsigned long long)inner);
    unsigned long long oi=(unsigned long long)(row_start+row)*inner+col;
    float xs=x_scale[row_start+row];
    dif_store_bf16(q,oi,(float)accumulator[(unsigned long long)row*packed+col]*xs*w_scale[col]);
    dif_store_bf16(k,oi,(float)accumulator[(unsigned long long)row*packed+inner+col]*xs*w_scale[inner+col]);
    dif_store_bf16(v,oi,(float)accumulator[(unsigned long long)row*packed+2*inner+col]*xs*w_scale[2*inner+col]);
    i+=stride;}}
extern "C" __global__ void dif_h3_w8a8_swiglu(
    const int* accumulator,const float* x_scale,const float* w_scale,
    dif_bf16* output,int rows,int ffn){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*ffn;int packed=2*ffn;
  while(i<total){int row=(int)(i/(unsigned long long)ffn),col=(int)(i%(unsigned long long)ffn);
    float xs=x_scale[row];float value=dif_round_bf16((float)accumulator[(unsigned long long)row*packed+col]*xs*w_scale[col]);
    float gate=dif_round_bf16((float)accumulator[(unsigned long long)row*packed+ffn+col]*xs*w_scale[ffn+col]);
    float activated=dif_round_bf16(gate/(1.0f+expf(-gate)));
    dif_store_bf16(output,i,activated*value);i+=stride;}}
extern "C" __global__ void dif_h3_w8a8_residual(
    const int* accumulator,const float* x_scale,const float* w_scale,
    const dif_bf16* residual,const dif_bf16* gate,dif_bf16* output,
    int row_start,int rows,int hidden){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long oi=(unsigned long long)(row_start+row)*hidden+col;
    float projected=dif_round_bf16((float)accumulator[i]*x_scale[row]*w_scale[col]);
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(gate,oi)*projected;
    dif_store_bf16(output,oi,result);i+=stride;}}
)CUDA";
}

std::string h3_convrot_source(bool enabled) {
  if (!enabled)
    return {};
  // Source-faithful H3 ConvRot activation contract: normalized
  // H256 = H4 kron H4 kron H4 kron H4, online in 256-wide groups, followed
  // by dynamic per-row INT8 quantization. The checkpoint already contains
  // the matching offline-rotated per-output-channel INT8 weights.
  return R"CUDA(
template<int S> __device__ __forceinline__ void dif_convrot_stage64(
    const float* src,float* dst,int lane){
  int base=(lane%S)+(lane/S)*(4*S);float x0=src[base],x1=src[base+S];
  float x2=src[base+2*S],x3=src[base+3*S];
  dst[base]=0.5f*(x0+x1+x2-x3);dst[base+S]=0.5f*(x0+x1-x2+x3);
  dst[base+2*S]=0.5f*(x0-x1+x2+x3);dst[base+3*S]=0.5f*(-x0+x1+x2+x3);}
template<int S> __device__ __forceinline__ float dif_convrot_stage64_store(
    const float* src,float* output,int lane){
  int base=(lane%S)+(lane/S)*(4*S);float x0=src[base],x1=src[base+S];
  float x2=src[base+2*S],x3=src[base+3*S];
  float y0=0.5f*(x0+x1+x2-x3),y1=0.5f*(x0+x1-x2+x3);
  float y2=0.5f*(x0-x1+x2+x3),y3=0.5f*(-x0+x1+x2+x3);
  output[base]=y0;output[base+S]=y1;output[base+2*S]=y2;output[base+3*S]=y3;
  return fmaxf(fmaxf(fabsf(y0),fabsf(y1)),fmaxf(fabsf(y2),fabsf(y3)));}
extern "C" __global__ void dif_convrot_int8_encode(
    const dif_bf16* x,signed char* q,float* scales,int row_start,int rows,int K){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* row_buf=smem;float* tmp=smem+K;
  __shared__ float warp_max[32];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;float local_max=0.0f;
  unsigned long long input_base=(unsigned long long)(row_start+row)*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?dif_load_bf16(x,input_base+col):0.0f;
    float x1=active?dif_load_bf16(x,input_base+col+1):0.0f;
    float x2=active?dif_load_bf16(x,input_base+col+2):0.0f;
    float x3=active?dif_load_bf16(x,input_base+col+3):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=fmaxf(local_max,
      dif_convrot_stage64_store<64>(b1,row_buf+gc,lane));__syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){block_max=v;float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_bf16=dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int col=tid;col<K;col+=blockDim.x){float value=dif_round_bf16(row_buf[col]);
    float scaled=dif_round_bf16(value/scale_bf16);int v=(int)nearbyintf(scaled);
    v=v>127?127:(v<-128?-128:v);q[output_base+col]=(signed char)v;}}
extern "C" __global__ void dif_convrot_int8_encode_cached(
    const void* x,signed char* q,float* scales,int row_start,int rows,int K,
    int input_f16){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* row_buf=smem;float* tmp=smem+K;
  __shared__ float warp_max[16];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;float local_max=0.0f;
  unsigned long long input_base=(unsigned long long)(row_start+row)*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col):dif_load_bf16((const dif_bf16*)x,input_base+col)):0.0f;
    float x1=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+1):dif_load_bf16((const dif_bf16*)x,input_base+col+1)):0.0f;
    float x2=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+2):dif_load_bf16((const dif_bf16*)x,input_base+col+2)):0.0f;
    float x3=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+3):dif_load_bf16((const dif_bf16*)x,input_base+col+3)):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=fmaxf(local_max,
      dif_convrot_stage64_store<64>(b1,row_buf+gc,lane));__syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){block_max=v;float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_lowp=input_f16?dif_round_f16(scales[row]):dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int col=tid;col<K;col+=blockDim.x){
    float value=input_f16?dif_round_f16(row_buf[col]):dif_round_bf16(row_buf[col]);
    float scaled=input_f16?dif_round_f16(value/scale_lowp):dif_round_bf16(value/scale_lowp);
    int v=(int)nearbyintf(scaled);v=v>127?127:(v<-128?-128:v);
    q[output_base+col]=(signed char)v;}}
extern "C" __global__ void dif_convrot_int8_encode_streamed(
    const void* x,signed char* q,float* scales,int row_start,int rows,int K,
    int input_f16){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float tmp[];__shared__ float warp_max[32];
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;float local_max=0.0f;
  unsigned long long input_base=(unsigned long long)(row_start+row)*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col):dif_load_bf16((const dif_bf16*)x,input_base+col)):0.0f;
    float x1=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+1):dif_load_bf16((const dif_bf16*)x,input_base+col+1)):0.0f;
    float x2=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+2):dif_load_bf16((const dif_bf16*)x,input_base+col+2)):0.0f;
    float x3=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+3):dif_load_bf16((const dif_bf16*)x,input_base+col+3)):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    int base=lane;float y0=0.5f*(b1[base]+b1[base+64]+b1[base+128]-b1[base+192]);
    float y1=0.5f*(b1[base]+b1[base+64]-b1[base+128]+b1[base+192]);
    float y2=0.5f*(b1[base]-b1[base+64]+b1[base+128]+b1[base+192]);
    float y3=0.5f*(-b1[base]+b1[base+64]+b1[base+128]+b1[base+192]);
    if(active)local_max=fmaxf(local_max,fmaxf(fmaxf(fabsf(y0),fabsf(y1)),
                                               fmaxf(fabsf(y2),fabsf(y3))));
    __syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_lowp=input_f16?dif_round_f16(scales[row]):dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col):dif_load_bf16((const dif_bf16*)x,input_base+col)):0.0f;
    float x1=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+1):dif_load_bf16((const dif_bf16*)x,input_base+col+1)):0.0f;
    float x2=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+2):dif_load_bf16((const dif_bf16*)x,input_base+col+2)):0.0f;
    float x3=active?(input_f16?dif_load_f16((const dif_f16*)x,input_base+col+3):dif_load_bf16((const dif_bf16*)x,input_base+col+3)):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    int base=lane;float y0=0.5f*(b1[base]+b1[base+64]+b1[base+128]-b1[base+192]);
    float y1=0.5f*(b1[base]+b1[base+64]-b1[base+128]+b1[base+192]);
    float y2=0.5f*(b1[base]-b1[base+64]+b1[base+128]+b1[base+192]);
    float y3=0.5f*(-b1[base]+b1[base+64]+b1[base+128]+b1[base+192]);
    if(active){float values[4]={y0,y1,y2,y3};for(int j=0;j<4;++j){
      float value=input_f16?dif_round_f16(values[j]):dif_round_bf16(values[j]);
      float scaled=input_f16?dif_round_f16(value/scale_lowp):dif_round_bf16(value/scale_lowp);
      int v=(int)nearbyintf(scaled);v=v>127?127:(v<-128?-128:v);
      q[output_base+gc+lane+j*64]=(signed char)v;}}
    __syncthreads();}}
extern "C" __global__ void dif_convrot_int8_encode_chunked(
    const dif_bf16* x,signed char* q,float* scales,int row_start,int rows,
    int K,int scale_chunk){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* row_buf=smem;float* tmp=smem+scale_chunk;
  __shared__ float warp_max[32];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;
  int groups_per_chunk=scale_chunk/group;
  int chunks=(n_groups+groups_per_chunk-1)/groups_per_chunk;
  unsigned long long input_base=(unsigned long long)(row_start+row)*K;
  unsigned long long output_base=(unsigned long long)row*K;
  for(int chunk=0;chunk<chunks;++chunk){
    int first_group=chunk*groups_per_chunk;
    int active_groups=min(groups_per_chunk,n_groups-first_group);
    int chunk_columns=active_groups*group;float local_max=0.0f;
    int g=first_group+sub;bool active=sub<active_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?dif_load_bf16(x,input_base+col):0.0f;
    float x1=active?dif_load_bf16(x,input_base+col+1):0.0f;
    float x2=active?dif_load_bf16(x,input_base+col+2):0.0f;
    float x3=active?dif_load_bf16(x,input_base+col+3):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=dif_convrot_stage64_store<64>(
        b1,row_buf+sub*group,lane);__syncthreads();
    for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
        local_max,__shfl_down_sync(0xffffffff,local_max,offset));
    int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
    __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
      for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
      if(lane32==0){float s=v*(1.0f/127.0f);block_max=s<1.0e-30f?1.0e-30f:s;
        scales[(unsigned long long)row*chunks+chunk]=block_max;}}
    __syncthreads();float scale_bf16=dif_round_bf16(block_max);
    int chunk_start=first_group*group;
    for(int local_col=tid;local_col<chunk_columns;local_col+=blockDim.x){
      float value=dif_round_bf16(row_buf[local_col]);
      float scaled=dif_round_bf16(value/scale_bf16);
      int v=(int)nearbyintf(scaled);v=v>127?127:(v<-128?-128:v);
      q[output_base+chunk_start+local_col]=(signed char)v;}
    __syncthreads();}}
extern "C" __global__ void dif_convrot_bf16_rotate_streamed(
    const dif_bf16* x,dif_bf16* y,int row_start,int rows,int K){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float tmp[];
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;
  unsigned long long input_base=(unsigned long long)(row_start+row)*K;
  unsigned long long output_base=(unsigned long long)row*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?dif_load_bf16(x,input_base+col):0.0f;
    float x1=active?dif_load_bf16(x,input_base+col+1):0.0f;
    float x2=active?dif_load_bf16(x,input_base+col+2):0.0f;
    float x3=active?dif_load_bf16(x,input_base+col+3):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    int base=lane;float values[4]={
      0.5f*(b1[base]+b1[base+64]+b1[base+128]-b1[base+192]),
      0.5f*(b1[base]+b1[base+64]-b1[base+128]+b1[base+192]),
      0.5f*(b1[base]-b1[base+64]+b1[base+128]+b1[base+192]),
      0.5f*(-b1[base]+b1[base+64]+b1[base+128]+b1[base+192])};
    if(active)for(int j=0;j<4;++j)
      dif_store_bf16(y,output_base+gc+lane+j*64,values[j]);
    __syncthreads();}}
extern "C" __global__ void dif_convrot_bf16_rotate_gather(
    const dif_bf16* x,const unsigned int* indices,dif_bf16* y,int rows,int K){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float tmp[];
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;
  unsigned long long input_base=(unsigned long long)indices[row]*K;
  unsigned long long output_base=(unsigned long long)row*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float x0=active?dif_load_bf16(x,input_base+col):0.0f;
    float x1=active?dif_load_bf16(x,input_base+col+1):0.0f;
    float x2=active?dif_load_bf16(x,input_base+col+2):0.0f;
    float x3=active?dif_load_bf16(x,input_base+col+3):0.0f;
    b1[b]=0.5f*(x0+x1+x2-x3);b1[b+1]=0.5f*(x0+x1-x2+x3);
    b1[b+2]=0.5f*(x0-x1+x2+x3);b1[b+3]=0.5f*(-x0+x1+x2+x3);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    int base=lane;float values[4]={
      0.5f*(b1[base]+b1[base+64]+b1[base+128]-b1[base+192]),
      0.5f*(b1[base]+b1[base+64]-b1[base+128]+b1[base+192]),
      0.5f*(b1[base]-b1[base+64]+b1[base+128]+b1[base+192]),
      0.5f*(-b1[base]+b1[base+64]+b1[base+128]+b1[base+192])};
    if(active)for(int j=0;j<4;++j)
      dif_store_bf16(y,output_base+gc+lane+j*64,values[j]);
    __syncthreads();}}
extern "C" __global__ void dif_convrot_weight_dequant_bf16(
    const signed char* weight,const float* scale,dif_bf16* output,
    unsigned long long rows,unsigned long long columns){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=rows*columns;
  while(i<total){dif_store_bf16(output,i,(float)weight[i]*scale[i/columns]);
    i+=stride;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_encode(
    const dif_bf16* x,const dif_bf16* weight,const dif_bf16* modulation,
    const int* indices,signed char* q,float* scales,int row_start,int rows,
    int K,int scale_lane,int shift_lane,float epsilon){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* reduction=smem;float* row_buf=smem;
  float* tmp=smem+K;__shared__ float warp_max[32];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;int global_row=row_start+row;
  float local=0.0f;if(tid<256){for(int col=tid;col<K;col+=256){
    float value=dif_load_bf16(x,(unsigned long long)global_row*K+col);
    local=__fadd_rn(local,__fmul_rn(value,value));}reduction[tid]=local;}
  __syncthreads();for(unsigned active=128U;active>0U;active>>=1U){
    if(tid<(int)active)reduction[tid]=__fadd_rn(reduction[tid],reduction[tid+active]);
    __syncthreads();}
  float inv=rsqrtf(__fadd_rn(__fdiv_rn(reduction[0],(float)K),epsilon));
  unsigned long long table=(unsigned long long)indices[global_row];
  unsigned long long scale_base=(table*6ULL+(unsigned long long)scale_lane)*K;
  unsigned long long shift_base=(table*6ULL+(unsigned long long)shift_lane)*K;
  float local_max=0.0f;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j){float value=0.0f;if(active){int c=col+j;
      unsigned long long i=(unsigned long long)global_row*K+c;
      float normed=dif_round_bf16(dif_load_bf16(x,i)*inv*dif_load_bf16(weight,c));
      float result=(1.0f+dif_load_bf16(modulation,scale_base+c))*normed+
                   dif_load_bf16(modulation,shift_base+c);
      value=dif_round_bf16(result);}values[j]=value;}
    b1[b]=0.5f*(values[0]+values[1]+values[2]-values[3]);
    b1[b+1]=0.5f*(values[0]+values[1]-values[2]+values[3]);
    b1[b+2]=0.5f*(values[0]-values[1]+values[2]+values[3]);
    b1[b+3]=0.5f*(-values[0]+values[1]+values[2]+values[3]);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=fmaxf(local_max,
      dif_convrot_stage64_store<64>(b1,row_buf+gc,lane));__syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){block_max=v;float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_bf16=dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int col=tid;col<K;col+=blockDim.x){float value=dif_round_bf16(row_buf[col]);
    float scaled=dif_round_bf16(value/scale_bf16);int v=(int)nearbyintf(scaled);
    v=v>127?127:(v<-128?-128:v);q[output_base+col]=(signed char)v;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_encode_chunked(
    const dif_bf16* x,const dif_bf16* weight,const dif_bf16* modulation,
    const int* indices,signed char* q,float* scales,int row_start,int rows,
    int K,int scale_lane,int shift_lane,float epsilon,int scale_chunk){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* reduction=smem;float* row_buf=smem;
  float* tmp=smem+scale_chunk;__shared__ float warp_max[32];
  __shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group,global_row=row_start+row;
  float local=0.0f;if(tid<256){for(int col=tid;col<K;col+=256){
    float value=dif_load_bf16(x,(unsigned long long)global_row*K+col);
    local=__fadd_rn(local,__fmul_rn(value,value));}reduction[tid]=local;}
  __syncthreads();for(unsigned active=128U;active>0U;active>>=1U){
    if(tid<(int)active)reduction[tid]=__fadd_rn(reduction[tid],reduction[tid+active]);
    __syncthreads();}
  float inv=rsqrtf(__fadd_rn(__fdiv_rn(reduction[0],(float)K),epsilon));
  unsigned long long table=(unsigned long long)indices[global_row];
  unsigned long long scale_base=(table*6ULL+(unsigned long long)scale_lane)*K;
  unsigned long long shift_base=(table*6ULL+(unsigned long long)shift_lane)*K;
  int groups_per_chunk=scale_chunk/group;
  int chunks=(n_groups+groups_per_chunk-1)/groups_per_chunk;
  unsigned long long output_base=(unsigned long long)row*K;
  for(int chunk=0;chunk<chunks;++chunk){
    int first_group=chunk*groups_per_chunk;
    int active_groups=min(groups_per_chunk,n_groups-first_group);
    int chunk_columns=active_groups*group;float local_max=0.0f;
    int g=first_group+sub;bool active=sub<active_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j){float value=0.0f;if(active){int c=col+j;
      unsigned long long i=(unsigned long long)global_row*K+c;
      float normed=dif_round_bf16(dif_load_bf16(x,i)*inv*dif_load_bf16(weight,c));
      float result=(1.0f+dif_load_bf16(modulation,scale_base+c))*normed+
                   dif_load_bf16(modulation,shift_base+c);
      value=dif_round_bf16(result);}values[j]=value;}
    b1[b]=0.5f*(values[0]+values[1]+values[2]-values[3]);
    b1[b+1]=0.5f*(values[0]+values[1]-values[2]+values[3]);
    b1[b+2]=0.5f*(values[0]-values[1]+values[2]+values[3]);
    b1[b+3]=0.5f*(-values[0]+values[1]+values[2]+values[3]);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=dif_convrot_stage64_store<64>(
        b1,row_buf+sub*group,lane);__syncthreads();
    for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
        local_max,__shfl_down_sync(0xffffffff,local_max,offset));
    int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
    __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
      for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
      if(lane32==0){float s=v*(1.0f/127.0f);block_max=s<1.0e-30f?1.0e-30f:s;
        scales[(unsigned long long)row*chunks+chunk]=block_max;}}
    __syncthreads();float scale_bf16=dif_round_bf16(block_max);
    int chunk_start=first_group*group;
    for(int local_col=tid;local_col<chunk_columns;local_col+=blockDim.x){
      float value=dif_round_bf16(row_buf[local_col]);
      float scaled=dif_round_bf16(value/scale_bf16);
      int v=(int)nearbyintf(scaled);v=v>127?127:(v<-128?-128:v);
      q[output_base+chunk_start+local_col]=(signed char)v;}
    __syncthreads();}}
extern "C" __global__ void dif_h3_convrot_chunk_accumulate(
    const int* partial,const float* x_scale,const float* w_scale,
    float* aggregate,int rows,int columns,int scale_groups,
    int activation_scale_groups,
    int weight_scale_groups,int group_index){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*columns;
  while(i<total){int row=(int)(i/(unsigned long long)columns);
    int col=(int)(i%(unsigned long long)columns);
    int wg=weight_scale_groups==1?0:group_index;
    int xg=activation_scale_groups==1?0:group_index;
    float value=(float)partial[i]*x_scale[(unsigned long long)row*activation_scale_groups+xg]*
                w_scale[(unsigned long long)col*weight_scale_groups+wg];
    aggregate[i]=group_index==0?value:__fadd_rn(aggregate[i],value);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_bf16_rotate_gather(
    const dif_bf16* x,const dif_bf16* weight,const dif_bf16* modulation,
    const int* adaln_indices,const unsigned int* correction_indices,
    dif_bf16* output,int rows,int K,int scale_lane,int shift_lane,float epsilon){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float tmp[];float* reduction=tmp;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;
  int global_row=(int)correction_indices[row];
  float local=0.0f;if(tid<256){for(int col=tid;col<K;col+=256){
    float value=dif_load_bf16(x,(unsigned long long)global_row*K+col);
    local=__fadd_rn(local,__fmul_rn(value,value));}reduction[tid]=local;}
  __syncthreads();for(unsigned active=128U;active>0U;active>>=1U){
    if(tid<(int)active)reduction[tid]=__fadd_rn(reduction[tid],reduction[tid+active]);
    __syncthreads();}
  float inv=rsqrtf(__fadd_rn(__fdiv_rn(reduction[0],(float)K),epsilon));
  unsigned long long table=(unsigned long long)adaln_indices[global_row];
  unsigned long long scale_base=(table*6ULL+(unsigned long long)scale_lane)*K;
  unsigned long long shift_base=(table*6ULL+(unsigned long long)shift_lane)*K;
  unsigned long long output_base=(unsigned long long)row*K;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j){float value=0.0f;if(active){int c=col+j;
      unsigned long long i=(unsigned long long)global_row*K+c;
      float normed=dif_round_bf16(dif_load_bf16(x,i)*inv*dif_load_bf16(weight,c));
      float result=(1.0f+dif_load_bf16(modulation,scale_base+c))*normed+
                   dif_load_bf16(modulation,shift_base+c);
      value=dif_round_bf16(result);}values[j]=value;}
    b1[b]=0.5f*(values[0]+values[1]+values[2]-values[3]);
    b1[b+1]=0.5f*(values[0]+values[1]-values[2]+values[3]);
    b1[b+2]=0.5f*(values[0]-values[1]+values[2]+values[3]);
    b1[b+3]=0.5f*(-values[0]+values[1]+values[2]+values[3]);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    int base=lane;float rotated[4]={
      0.5f*(b1[base]+b1[base+64]+b1[base+128]-b1[base+192]),
      0.5f*(b1[base]+b1[base+64]-b1[base+128]+b1[base+192]),
      0.5f*(b1[base]-b1[base+64]+b1[base+128]+b1[base+192]),
      0.5f*(-b1[base]+b1[base+64]+b1[base+128]+b1[base+192])};
    if(active)for(int j=0;j<4;++j)
      dif_store_bf16(output,output_base+gc+lane+j*64,rotated[j]);
    __syncthreads();}}
extern "C" __global__ void dif_h3_convrot_qkv(
    const int* accumulator,const float* x_scale,const float* w_scale,
    dif_bf16* q,dif_bf16* k,dif_bf16* v,int row_start,int rows,int inner){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*inner;
  int packed=3*inner;while(i<total){int row=(int)(i/(unsigned long long)inner),col=(int)(i%(unsigned long long)inner);
    unsigned long long oi=(unsigned long long)(row_start+row)*inner+col;
    float xs=x_scale[row_start+row];unsigned long long base=(unsigned long long)row*packed;
    dif_store_bf16(q,oi,(float)accumulator[base+col]*xs*w_scale[col]);
    dif_store_bf16(k,oi,(float)accumulator[base+inner+col]*xs*w_scale[inner+col]);
    dif_store_bf16(v,oi,(float)accumulator[base+2*inner+col]*xs*w_scale[2*inner+col]);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_qkv_bf16(
    const dif_bf16* projected,dif_bf16* q,dif_bf16* k,dif_bf16* v,
    int row_start,int rows,int inner){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*inner;
  int packed=3*inner;while(i<total){int row=(int)(i/(unsigned long long)inner),col=(int)(i%(unsigned long long)inner);
    unsigned long long oi=(unsigned long long)(row_start+row)*inner+col;
    unsigned long long base=(unsigned long long)row*packed;
    dif_store_bf16(q,oi,dif_load_bf16(projected,base+col));
    dif_store_bf16(k,oi,dif_load_bf16(projected,base+inner+col));
    dif_store_bf16(v,oi,dif_load_bf16(projected,base+2*inner+col));i+=stride;}}
extern "C" __global__ void dif_h3_convrot_qkv_f32(
    const float* projected,dif_bf16* q,dif_bf16* k,dif_bf16* v,
    int row_start,int rows,int inner){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*inner;
  int packed=3*inner;while(i<total){int row=(int)(i/(unsigned long long)inner),col=(int)(i%(unsigned long long)inner);
    unsigned long long oi=(unsigned long long)(row_start+row)*inner+col;
    unsigned long long base=(unsigned long long)row*packed;
    dif_store_bf16(q,oi,projected[base+col]);
    dif_store_bf16(k,oi,projected[base+inner+col]);
    dif_store_bf16(v,oi,projected[base+2*inner+col]);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_qkv_bf16_scatter(
    const dif_bf16* projected,const unsigned int* indices,
    dif_bf16* q,dif_bf16* k,dif_bf16* v,int rows,int inner){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*inner;
  int packed=3*inner;while(i<total){int row=(int)(i/(unsigned long long)inner),col=(int)(i%(unsigned long long)inner);
    unsigned long long oi=(unsigned long long)indices[row]*inner+col;
    unsigned long long base=(unsigned long long)row*packed;
    dif_store_bf16(q,oi,dif_load_bf16(projected,base+col));
    dif_store_bf16(k,oi,dif_load_bf16(projected,base+inner+col));
    dif_store_bf16(v,oi,dif_load_bf16(projected,base+2*inner+col));i+=stride;}}
extern "C" __global__ void dif_h3_convrot_swiglu(
    const int* accumulator,const float* x_scale,const float* w_scale,
    dif_bf16* output,int rows,int ffn){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*ffn;
  int packed=2*ffn;while(i<total){int row=(int)(i/(unsigned long long)ffn),col=(int)(i%(unsigned long long)ffn);
    float xs=x_scale[row];float gate=dif_round_bf16((float)accumulator[(unsigned long long)row*packed+col]*xs*w_scale[col]);
    float value=dif_round_bf16((float)accumulator[(unsigned long long)row*packed+ffn+col]*xs*w_scale[ffn+col]);
    float activated=dif_round_bf16(gate/(1.0f+expf(-gate)));
    dif_store_bf16(output,i,activated*value);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_swiglu_f32(
    const float* projected,dif_bf16* output,int rows,int K){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*K;
  int packed=2*K;while(i<total){int row=(int)(i/(unsigned long long)K),col=(int)(i%(unsigned long long)K);
    unsigned long long base=(unsigned long long)row*packed;
    float gate=dif_round_bf16(projected[base+col]);
    float value=dif_round_bf16(projected[base+K+col]);
    float activated=dif_round_bf16(gate/(1.0f+expf(-gate)));
    dif_store_bf16(output,i,activated*value);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_swiglu_bf16(
    const dif_bf16* projected,dif_bf16* output,int rows,int K){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x,total=(unsigned long long)rows*K;
  int packed=2*K;while(i<total){int row=(int)(i/(unsigned long long)K),col=(int)(i%(unsigned long long)K);
    unsigned long long base=(unsigned long long)row*packed;
    float gate=dif_load_bf16(projected,base+col);
    float value=dif_load_bf16(projected,base+K+col);
    float activated=dif_round_bf16(gate/(1.0f+expf(-gate)));
    dif_store_bf16(output,i,activated*value);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_swiglu_int8_encode(
    const int* accumulator,const float* x_scale,const float* w_scale,
    signed char* q,float* scales,int rows,int K){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* row_buf=smem;float* tmp=smem+K;
  __shared__ float warp_max[32];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;float local_max=0.0f;
  int packed=2*K;unsigned long long input_base=(unsigned long long)row*packed;
  float xs=x_scale[row];
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j){int c=col+j;float gate=0.0f,value=0.0f;
      if(active){gate=dif_round_bf16((float)accumulator[input_base+c]*xs*w_scale[c]);
        value=dif_round_bf16((float)accumulator[input_base+K+c]*xs*w_scale[K+c]);}
      values[j]=(gate/(1.0f+expf(-gate)))*value;}
    b1[b]=0.5f*(values[0]+values[1]+values[2]-values[3]);
    b1[b+1]=0.5f*(values[0]+values[1]-values[2]+values[3]);
    b1[b+2]=0.5f*(values[0]-values[1]+values[2]+values[3]);
    b1[b+3]=0.5f*(-values[0]+values[1]+values[2]+values[3]);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=fmaxf(local_max,
      dif_convrot_stage64_store<64>(b1,row_buf+gc,lane));__syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){block_max=v;float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_bf16=dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int col=tid;col<K;col+=blockDim.x){float value=dif_round_bf16(row_buf[col]);
    float scaled=dif_round_bf16(value/scale_bf16);int v=(int)nearbyintf(scaled);
    v=v>127?127:(v<-128?-128:v);q[output_base+col]=(signed char)v;}}
extern "C" __global__ void dif_h3_convrot_swiglu_bf16_int8_encode(
    const dif_bf16* projected,signed char* q,float* scales,int rows,int K){
  constexpr int group=256,group_threads=64,groups_in_flight=8;
  extern __shared__ float smem[];float* row_buf=smem;float* tmp=smem+K;
  __shared__ float warp_max[32];__shared__ float block_max;
  int row=(int)blockIdx.x,tid=(int)threadIdx.x,sub=tid/group_threads;
  int lane=tid%group_threads,n_groups=K/group;float local_max=0.0f;
  int packed=2*K;unsigned long long input_base=(unsigned long long)row*packed;
  for(int it=0;it<(n_groups+groups_in_flight-1)/groups_in_flight;++it){
    int g=it*groups_in_flight+sub;bool active=g<n_groups;int b=lane*4;
    int gc=g*group,col=gc+b;float* b0=tmp+sub*(2*group);float* b1=b0+group;
    float values[4];
    #pragma unroll
    for(int j=0;j<4;++j){int c=col+j;float gate=0.0f,value=0.0f;
      if(active){gate=dif_load_bf16(projected,input_base+c);
        value=dif_load_bf16(projected,input_base+K+c);}
      values[j]=(gate/(1.0f+expf(-gate)))*value;}
    b1[b]=0.5f*(values[0]+values[1]+values[2]-values[3]);
    b1[b+1]=0.5f*(values[0]+values[1]-values[2]+values[3]);
    b1[b+2]=0.5f*(values[0]-values[1]+values[2]+values[3]);
    b1[b+3]=0.5f*(-values[0]+values[1]+values[2]+values[3]);
    __syncthreads();dif_convrot_stage64<4>(b1,b0,lane);__syncthreads();
    dif_convrot_stage64<16>(b0,b1,lane);__syncthreads();
    if(active)local_max=fmaxf(local_max,
      dif_convrot_stage64_store<64>(b1,row_buf+gc,lane));__syncthreads();}
  for(int offset=16;offset>0;offset>>=1)local_max=fmaxf(
      local_max,__shfl_down_sync(0xffffffff,local_max,offset));
  int warp=tid>>5,lane32=tid&31;if(lane32==0)warp_max[warp]=local_max;
  __syncthreads();if(warp==0){float v=lane32<(blockDim.x>>5)?warp_max[lane32]:0.0f;
    for(int offset=16;offset>0;offset>>=1)v=fmaxf(v,__shfl_down_sync(0xffffffff,v,offset));
    if(lane32==0){block_max=v;float s=v*(1.0f/127.0f);scales[row]=s<1.0e-30f?1.0e-30f:s;}}
  __syncthreads();float scale_bf16=dif_round_bf16(scales[row]);
  unsigned long long output_base=(unsigned long long)row*K;
  for(int col=tid;col<K;col+=blockDim.x){float value=dif_round_bf16(row_buf[col]);
    float scaled=dif_round_bf16(value/scale_bf16);int v=(int)nearbyintf(scaled);
    v=v>127?127:(v<-128?-128:v);q[output_base+col]=(signed char)v;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_residual(
    const int* accumulator,const float* x_scale,const float* w_scale,
    const dif_bf16* residual,const dif_bf16* modulation,const int* indices,
    dif_bf16* output,int row_start,int rows,int hidden,int gate_lane){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long output_row=(unsigned long long)(row_start+row);
    unsigned long long oi=output_row*hidden+col;
    unsigned long long table=(unsigned long long)indices[output_row];
    unsigned long long gi=(table*6ULL+(unsigned long long)gate_lane)*hidden+col;
    float projected=dif_round_bf16((float)accumulator[i]*x_scale[row]*w_scale[col]);
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(modulation,gi)*projected;
    dif_store_bf16(output,oi,result);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_residual_bf16(
    const dif_bf16* projected,const dif_bf16* residual,
    const dif_bf16* modulation,const int* indices,dif_bf16* output,
    int row_start,int rows,int hidden,int gate_lane){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long output_row=(unsigned long long)(row_start+row);
    unsigned long long oi=output_row*hidden+col;
    unsigned long long table=(unsigned long long)indices[output_row];
    unsigned long long gi=(table*6ULL+(unsigned long long)gate_lane)*hidden+col;
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(modulation,gi)*
                 dif_load_bf16(projected,i);
    dif_store_bf16(output,oi,result);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_compact_adaln_residual_f32(
    const float* projected,const dif_bf16* residual,
    const dif_bf16* modulation,const int* indices,dif_bf16* output,
    int row_start,int rows,int hidden,int gate_lane){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long output_row=(unsigned long long)(row_start+row);
    unsigned long long oi=output_row*hidden+col;
    unsigned long long table=(unsigned long long)indices[output_row];
    unsigned long long gi=(table*6ULL+(unsigned long long)gate_lane)*hidden+col;
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(modulation,gi)*
                 dif_round_bf16(projected[i]);
    dif_store_bf16(output,oi,result);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_residual_f32(
    const float* projected,const dif_bf16* residual,const dif_bf16* gate,
    dif_bf16* output,int row_start,int rows,int hidden){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long oi=(unsigned long long)(row_start+row)*hidden+col;
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(gate,oi)*
                 dif_round_bf16(projected[i]);
    dif_store_bf16(output,oi,result);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_compact_residual_bf16_scatter(
    const dif_bf16* projected,const unsigned int* correction_indices,
    const dif_bf16* residual,const dif_bf16* modulation,
    const int* adaln_indices,dif_bf16* output,int rows,int hidden,
    int gate_lane){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long output_row=(unsigned long long)correction_indices[row];
    unsigned long long oi=output_row*hidden+col;
    unsigned long long table=(unsigned long long)adaln_indices[output_row];
    unsigned long long gi=(table*6ULL+(unsigned long long)gate_lane)*hidden+col;
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(modulation,gi)*
                 dif_load_bf16(projected,i);
    dif_store_bf16(output,oi,result);i+=stride;}}
extern "C" __global__ void dif_h3_convrot_residual_bf16_scatter(
    const dif_bf16* projected,const unsigned int* correction_indices,
    const dif_bf16* residual,const dif_bf16* gate,dif_bf16* output,
    int rows,int hidden){
  unsigned long long i=(unsigned long long)blockIdx.x*blockDim.x+threadIdx.x;
  unsigned long long stride=(unsigned long long)gridDim.x*blockDim.x;
  unsigned long long total=(unsigned long long)rows*hidden;while(i<total){
    int row=(int)(i/(unsigned long long)hidden),col=(int)(i%(unsigned long long)hidden);
    unsigned long long oi=(unsigned long long)correction_indices[row]*hidden+col;
    float result=dif_load_bf16(residual,oi)+dif_load_bf16(gate,oi)*
                 dif_load_bf16(projected,i);
    dif_store_bf16(output,oi,result);i+=stride;}}
)CUDA";
}

void assign_h3_w8a8_weights(H3W8A8MlpPlan &plan, CUdeviceptr base) {
  auto weight_offset = std::uint64_t{0U};
  auto assign_weight = [&](CUdeviceptr &pointer, const Tensor &tensor) {
    pointer = base + weight_offset;
    weight_offset += align_256(tensor.byte_size());
  };
  assign_weight(plan.fc1_weight_device, plan.fc1_weight);
  assign_weight(plan.fc1_scale_device, plan.fc1_scale);
  assign_weight(plan.fc2_weight_device, plan.fc2_weight);
  assign_weight(plan.fc2_scale_device, plan.fc2_scale);
  if (weight_offset != plan.weight_storage_bytes)
    fail("H3 W8A8 weight-storage layout mismatch");
}

void allocate_h3_w8a8_weights(H3W8A8MlpPlan &plan, DeviceArena *arena) {
  plan.weight_storage = std::make_unique<Workspace>(
      static_cast<std::size_t>(plan.weight_storage_bytes), arena);
  assign_h3_w8a8_weights(plan, plan.weight_storage->pointer());
}

void assign_h3_w8a8_scratch(H3W8A8MlpPlan &plan, CUdeviceptr scratch) {
  const auto chunk_rows = std::min<std::uint64_t>(
      plan.rows, plan.chunk_rows);
  auto scratch_offset = std::uint64_t{0U};
  auto assign_scratch = [&](CUdeviceptr &pointer, std::uint64_t bytes) {
    pointer = scratch + scratch_offset;
    scratch_offset += align_256(bytes);
  };
  const auto input_scale_groups =
      plan.convrot_scale_chunk == 0U || plan.convrot_global_activation_scale
                                      ? 1U
                                      : (plan.hidden + plan.convrot_scale_chunk - 1U) /
                                            plan.convrot_scale_chunk;
  const auto activation_scale_groups =
      plan.convrot_scale_chunk == 0U || plan.convrot_global_activation_scale
                                           ? 1U
                                           : (plan.ffn + plan.convrot_scale_chunk - 1U) /
                                                 plan.convrot_scale_chunk;
  assign_scratch(plan.input_scale_device,
                 chunk_rows * input_scale_groups * sizeof(float));
  assign_scratch(plan.input_i8_device, chunk_rows * plan.hidden);
  assign_scratch(plan.fc1_accumulator_device,
                 chunk_rows * plan.packed_ffn *
                     (plan.cutlass_scaled_fc1 ? sizeof(std::uint16_t)
                                              : sizeof(std::int32_t)));
  if (plan.convrot_scale_chunk != 0U)
    assign_scratch(plan.fc1_aggregate_device,
                   chunk_rows * plan.packed_ffn * sizeof(float));
  assign_scratch(plan.activation_device,
                 chunk_rows * plan.ffn * sizeof(std::uint16_t));
  assign_scratch(plan.activation_scale_device,
                 chunk_rows * activation_scale_groups * sizeof(float));
  assign_scratch(plan.activation_i8_device, chunk_rows * plan.ffn);
  assign_scratch(plan.fc2_accumulator_device,
                 chunk_rows * plan.hidden *
                     (plan.cutlass_scaled_fc2 ? sizeof(std::uint16_t)
                                              : sizeof(std::int32_t)));
  if (plan.convrot_scale_chunk != 0U)
    assign_scratch(plan.fc2_aggregate_device,
                   chunk_rows * plan.hidden * sizeof(float));
  if (scratch_offset != plan.scratch_bytes)
    fail("H3 W8A8 scratch-storage layout mismatch");
}

void upload_h3_w8a8_weights(const H3W8A8MlpPlan &plan, CUstream stream) {
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    check(counted_memcpy_htod(destination, tensor.data(), tensor.byte_size(),
                            stream),
          "cuMemcpyHtoDAsync H3 W8A8 weight");
  };
  upload(plan.fc1_weight_device, plan.fc1_weight);
  upload(plan.fc1_scale_device, plan.fc1_scale);
  upload(plan.fc2_weight_device, plan.fc2_weight);
  upload(plan.fc2_scale_device, plan.fc2_scale);
}

// Stages one mapped weight tensor into host staging memory.
using H3HostCopy = std::function<void(std::uint8_t *, const Tensor &)>;

template <class T>
void release_resident_host_pages(const T &tensor, bool evict) {
  if (evict)
    tensor.evict_mapped_pages();
  else
    tensor.discard_mapped_pages();
}

void evict_h3_w8a8_weights(const H3W8A8MlpPlan &plan, bool evict) {
  release_resident_host_pages(plan.fc1_weight, evict);
  release_resident_host_pages(plan.fc1_scale, evict);
  release_resident_host_pages(plan.fc2_weight, evict);
  release_resident_host_pages(plan.fc2_scale, evict);
}

std::uint64_t stage_h3_w8a8_weights(const H3W8A8MlpPlan &plan,
                                    void *staging,
                                    std::uint64_t staging_bytes,
                                    CUstream stream,
                                    const H3HostCopy &host_copy) {
  auto *base = static_cast<std::uint8_t *>(staging);
  auto cursor = std::uint64_t{0U};
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    if (tensor.byte_size() > staging_bytes - cursor)
      fail("H3 W8A8 MLP tail staging overflow");
    host_copy(base + cursor, tensor);
    check(counted_memcpy_htod(destination, base + cursor, tensor.byte_size(),
                            stream),
          "cuMemcpyHtoDAsync H3 W8A8 MLP tail weight");
    cursor += tensor.byte_size();
  };
  upload(plan.fc1_weight_device, plan.fc1_weight);
  upload(plan.fc1_scale_device, plan.fc1_scale);
  upload(plan.fc2_weight_device, plan.fc2_weight);
  upload(plan.fc2_scale_device, plan.fc2_scale);
  return cursor;
}

void assign_h3_w8a8_weights(H3W8A8AttentionPlan &plan, CUdeviceptr base) {
  auto offset = std::uint64_t{0U};
  auto assign = [&](CUdeviceptr &pointer, const Tensor &tensor) {
    pointer = base + offset;
    offset += align_256(tensor.byte_size());
  };
  if (plan.has_qkv_projection) {
    assign(plan.qkv_weight_device, plan.qkv_weight);
    assign(plan.qkv_scale_device, plan.qkv_scale);
  }
  if (plan.has_output_projection) {
    assign(plan.output_weight_device, plan.output_weight);
    assign(plan.output_scale_device, plan.output_scale);
  }
  if (offset != plan.weight_storage_bytes)
    fail("H3 W8A8 attention weight-storage layout mismatch");
}

void allocate_h3_w8a8_weights(H3W8A8AttentionPlan &plan,
                              DeviceArena *arena) {
  plan.weight_storage = std::make_unique<Workspace>(
      static_cast<std::size_t>(plan.weight_storage_bytes), arena);
  assign_h3_w8a8_weights(plan, plan.weight_storage->pointer());
}

void assign_h3_w8a8_scratch(H3W8A8AttentionPlan &plan,
                             CUdeviceptr scratch) {
  const auto chunk_rows = std::min<std::uint64_t>(
      plan.rows, kH3W8A8ProjectionChunkRows);
  auto offset = std::uint64_t{0U};
  auto assign = [&](CUdeviceptr &pointer, std::uint64_t bytes) {
    pointer = scratch + offset;
    offset += align_256(bytes);
  };
  if (plan.has_qkv_projection) {
    const auto scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                  ? 1U
                                  : (plan.hidden + plan.convrot_scale_chunk - 1U) /
                                        plan.convrot_scale_chunk;
    assign(plan.activation_scale_device,
           plan.rows * scale_groups * sizeof(float));
    assign(plan.activation_i8_device, plan.rows * plan.hidden);
    assign(plan.accumulator_device,
           chunk_rows * plan.packed_inner *
               (plan.cutlass_scaled ? sizeof(std::uint16_t)
                                    : sizeof(std::int32_t)));
    if (plan.convrot_scale_chunk != 0U)
      assign(plan.aggregate_device,
             chunk_rows * plan.packed_inner * sizeof(float));
  } else {
    const auto scale_groups =
        plan.convrot_scale_chunk == 0U ||
                plan.convrot_global_activation_scale
                                  ? 1U
                                  : (plan.inner + plan.convrot_scale_chunk - 1U) /
                                        plan.convrot_scale_chunk;
    assign(plan.activation_scale_device,
           chunk_rows * scale_groups * sizeof(float));
    assign(plan.activation_i8_device, chunk_rows * plan.inner);
    assign(plan.accumulator_device,
           chunk_rows * plan.hidden *
               (plan.cutlass_scaled ? sizeof(std::uint16_t)
                                    : sizeof(std::int32_t)));
    if (plan.convrot_scale_chunk != 0U)
      assign(plan.aggregate_device,
             chunk_rows * plan.hidden * sizeof(float));
  }
  if (offset != plan.scratch_bytes)
    fail("H3 W8A8 attention scratch-storage layout mismatch");
}

void upload_h3_w8a8_weights(const H3W8A8AttentionPlan &plan,
                             CUstream stream) {
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    check(counted_memcpy_htod(destination, tensor.data(), tensor.byte_size(),
                            stream),
          "cuMemcpyHtoDAsync H3 W8A8 attention weight");
  };
  if (plan.has_qkv_projection) {
    upload(plan.qkv_weight_device, plan.qkv_weight);
    upload(plan.qkv_scale_device, plan.qkv_scale);
  }
  if (plan.has_output_projection) {
    upload(plan.output_weight_device, plan.output_weight);
    upload(plan.output_scale_device, plan.output_scale);
  }
}

void evict_h3_w8a8_weights(const H3W8A8AttentionPlan &plan, bool evict) {
  if (plan.has_qkv_projection) {
    release_resident_host_pages(plan.qkv_weight, evict);
    release_resident_host_pages(plan.qkv_scale, evict);
  }
  if (plan.has_output_projection) {
    release_resident_host_pages(plan.output_weight, evict);
    release_resident_host_pages(plan.output_scale, evict);
  }
}

std::uint64_t h3_w8a8_weight_bytes(const H3W8A8MlpPlan &plan) {
  return plan.fc1_weight.byte_size() + plan.fc1_scale.byte_size() +
         plan.fc2_weight.byte_size() + plan.fc2_scale.byte_size();
}

std::uint64_t h3_w8a8_weight_bytes(const H3W8A8AttentionPlan &plan) {
  auto bytes = std::uint64_t{0U};
  if (plan.has_qkv_projection)
    bytes += plan.qkv_weight.byte_size() + plan.qkv_scale.byte_size();
  if (plan.has_output_projection)
    bytes += plan.output_weight.byte_size() + plan.output_scale.byte_size();
  return bytes;
}

void prefetch_h3_w8a8_weights(const H3W8A8MlpPlan &plan) {
  plan.fc1_weight.prefetch_mapped_pages();
  plan.fc1_scale.prefetch_mapped_pages();
  plan.fc2_weight.prefetch_mapped_pages();
  plan.fc2_scale.prefetch_mapped_pages();
}

void prefetch_h3_w8a8_weights(const H3W8A8AttentionPlan &plan) {
  if (plan.has_qkv_projection) {
    plan.qkv_weight.prefetch_mapped_pages();
    plan.qkv_scale.prefetch_mapped_pages();
  }
  if (plan.has_output_projection) {
    plan.output_weight.prefetch_mapped_pages();
    plan.output_scale.prefetch_mapped_pages();
  }
}

std::uint64_t stage_h3_w8a8_weights(const H3W8A8AttentionPlan &plan,
                                    void *staging,
                                    std::uint64_t staging_bytes,
                                    CUstream stream,
                                    const H3HostCopy &host_copy) {
  auto *base = static_cast<std::uint8_t *>(staging);
  auto cursor = std::uint64_t{0U};
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    if (tensor.byte_size() > staging_bytes - cursor)
      fail("H3 W8A8 attention tail staging overflow");
    host_copy(base + cursor, tensor);
    check(counted_memcpy_htod(destination, base + cursor, tensor.byte_size(),
                            stream),
          "cuMemcpyHtoDAsync H3 W8A8 attention tail weight");
    cursor += tensor.byte_size();
  };
  if (plan.has_qkv_projection) {
    upload(plan.qkv_weight_device, plan.qkv_weight);
    upload(plan.qkv_scale_device, plan.qkv_scale);
  }
  if (plan.has_output_projection) {
    upload(plan.output_weight_device, plan.output_weight);
    upload(plan.output_scale_device, plan.output_scale);
  }
  return cursor;
}

unsigned h3_w8a8_grid(std::uint64_t elements) {
  constexpr std::uint64_t block = 256U;
  return static_cast<unsigned>(
      std::min<std::uint64_t>((elements + block - 1U) / block, 65535U));
}

void launch_h3_w8a8_kernel(CUfunction function, unsigned grid,
                           std::array<void *, 5> &arguments,
                           CUstream stream, const char *label) {
  check(counted_launch_kernel(function, grid, 1U, 1U, 256U, 1U, 1U, 0U, stream,
                       arguments.data(), nullptr),
        label);
}

void launch_h3_convrot_encode(CUfunction function, CUdeviceptr input,
                              CUdeviceptr encoded, CUdeviceptr scale,
                              int row_start, int rows, int columns,
                              CUstream stream, const char *label) {
  constexpr int group_size = 256;
  constexpr int block_threads = 512;
  constexpr std::uint64_t temporary_floats = 8U * 2U * group_size;
  if (columns <= 0 || columns % group_size != 0 || rows <= 0)
    fail("H3 ConvRot requires positive rows and K divisible by 256");
  const auto shared_bytes =
      (static_cast<std::uint64_t>(columns) + temporary_floats) *
      sizeof(float);
  std::array<void *, 6> arguments = {
      &input, &encoded, &scale, &row_start, &rows, &columns};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads, 1U,
            1U, static_cast<unsigned>(shared_bytes), stream, arguments.data(),
            nullptr),
        label);
}

void launch_h3_convrot_compact_adaln_encode(
    CUfunction function, const H3CompactAdaLNBinding &binding,
    const DeviceBuffers &buffers, CUdeviceptr encoded, CUdeviceptr scale,
    int row_start, int rows, int columns, CUstream stream,
    const char *label) {
  constexpr int group_size = 256;
  constexpr int block_threads = 512;
  constexpr std::uint64_t temporary_floats = 8U * 2U * group_size;
  if (!binding.enabled || columns <= 0 || columns % group_size != 0 ||
      rows <= 0)
    fail("H3 compact AdaLN ConvRot requires an admitted binding and aligned geometry");
  auto input = buffers.at(binding.norm_input_tensor);
  auto weight = buffers.at(binding.norm_weight_tensor);
  auto modulation = buffers.at(binding.modulation_tensor);
  auto indices = buffers.at(binding.indices_tensor);
  auto scale_lane = static_cast<int>(binding.scale_lane);
  auto shift_lane = static_cast<int>(binding.shift_lane);
  auto epsilon = binding.epsilon;
  const auto shared_bytes =
      (static_cast<std::uint64_t>(columns) + temporary_floats) *
      sizeof(float);
  std::array<void *, 12> arguments = {
      &input,      &weight,     &modulation, &indices,    &encoded,
      &scale,      &row_start,  &rows,       &columns,    &scale_lane,
      &shift_lane, &epsilon};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads, 1U,
            1U, static_cast<unsigned>(shared_bytes), stream, arguments.data(),
            nullptr),
        label);
}

void launch_h3_convrot_chunked_encode(
    CUfunction function, CUdeviceptr input, CUdeviceptr encoded,
    CUdeviceptr scale, int row_start, int rows, int columns,
    int scale_chunk, CUstream stream, const char *label) {
  constexpr int group_size = 256;
  constexpr int block_threads = 512;
  constexpr std::uint64_t temporary_floats = 8U * 2U * group_size;
  if (columns <= 0 || columns % group_size != 0 || rows <= 0 ||
      scale_chunk < group_size || scale_chunk > 2048 ||
      scale_chunk % group_size != 0)
    fail("H3 chunk-scaled ConvRot requires aligned positive geometry");
  const auto shared_bytes =
      (static_cast<std::uint64_t>(scale_chunk) + temporary_floats) *
      sizeof(float);
  std::array<void *, 7> arguments = {
      &input, &encoded, &scale, &row_start, &rows, &columns, &scale_chunk};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads, 1U,
            1U, static_cast<unsigned>(shared_bytes), stream, arguments.data(),
            nullptr),
        label);
}

void launch_h3_convrot_compact_adaln_chunked_encode(
    CUfunction function, const H3CompactAdaLNBinding &binding,
    const DeviceBuffers &buffers, CUdeviceptr encoded, CUdeviceptr scale,
    int row_start, int rows, int columns, int scale_chunk, CUstream stream,
    const char *label) {
  constexpr int group_size = 256;
  constexpr int block_threads = 512;
  constexpr std::uint64_t temporary_floats = 8U * 2U * group_size;
  if (!binding.enabled || columns <= 0 || columns % group_size != 0 ||
      rows <= 0 || scale_chunk < group_size || scale_chunk > 2048 ||
      scale_chunk % group_size != 0)
    fail("H3 compact AdaLN chunk-scaled ConvRot has invalid geometry");
  auto input = buffers.at(binding.norm_input_tensor);
  auto weight = buffers.at(binding.norm_weight_tensor);
  auto modulation = buffers.at(binding.modulation_tensor);
  auto indices = buffers.at(binding.indices_tensor);
  auto scale_lane = static_cast<int>(binding.scale_lane);
  auto shift_lane = static_cast<int>(binding.shift_lane);
  auto epsilon = binding.epsilon;
  const auto shared_bytes =
      (static_cast<std::uint64_t>(scale_chunk) + temporary_floats) *
      sizeof(float);
  std::array<void *, 13> arguments = {
      &input,      &weight,     &modulation, &indices,     &encoded,
      &scale,      &row_start,  &rows,       &columns,     &scale_lane,
      &shift_lane, &epsilon,    &scale_chunk};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads, 1U,
            1U, static_cast<unsigned>(shared_bytes), stream, arguments.data(),
            nullptr),
        label);
}

void launch_h3_direct_int8_gemm(
    cublasHandle_t cublas, const H3Int8GemmRegistry *registry,
    const Workspace *workspace, CUdeviceptr activation, CUdeviceptr weight,
    CUdeviceptr accumulator, int rows, int columns, int contraction,
    CUstream stream, const char *label) {
  if (registry) {
    if (!workspace)
      fail("H3 INT8 cuBLASLt projection requires prepared workspace");
    registry->launch(
        {static_cast<std::uint32_t>(rows),
         static_cast<std::uint32_t>(columns),
         static_cast<std::uint32_t>(contraction)},
        activation, weight, accumulator, *workspace, stream);
    return;
  }
  constexpr std::int32_t alpha = 1;
  constexpr std::int32_t beta = 0;
  check(counted_cublas_gemm_ex(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, contraction,
            &alpha, reinterpret_cast<const void *>(weight), CUDA_R_8I,
            contraction, reinterpret_cast<const void *>(activation),
            CUDA_R_8I, contraction, &beta,
            reinterpret_cast<void *>(accumulator), CUDA_R_32I, columns,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        label);
}

void launch_h3_convrot_chunked_gemm(
    cublasHandle_t cublas, CUfunction accumulate_function,
    CUdeviceptr activation, CUdeviceptr weight, CUdeviceptr activation_scale,
    CUdeviceptr weight_scale, CUdeviceptr partial, CUdeviceptr aggregate,
    int rows, int columns, int contraction, int scale_chunk,
    int activation_scale_groups, int weight_scale_groups, CUstream stream,
    const char *label) {
  if (rows <= 0 || columns <= 0 || contraction <= 0 ||
      contraction % 256 != 0 || scale_chunk < 256 ||
      scale_chunk > 2048 || scale_chunk % 256 != 0)
    fail("H3 chunk-scaled ConvRot GEMM has invalid geometry");
  auto scale_groups =
      (contraction + scale_chunk - 1) / scale_chunk;
  if (weight_scale_groups != 1 && weight_scale_groups != scale_groups)
    fail("H3 ConvRot weight/activation scale-group count mismatch");
  if (activation_scale_groups != 1 &&
      activation_scale_groups != scale_groups)
    fail("H3 ConvRot activation scale-group count mismatch");
  constexpr std::int32_t alpha = 1;
  constexpr std::int32_t beta = 0;
  for (int group = 0; group < scale_groups; ++group) {
    const auto offset = group * scale_chunk;
    const auto chunk = std::min(scale_chunk, contraction - offset);
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, chunk,
              &alpha, reinterpret_cast<const void *>(weight + offset),
              CUDA_R_8I, contraction,
              reinterpret_cast<const void *>(activation + offset),
              CUDA_R_8I, contraction, &beta,
              reinterpret_cast<void *>(partial), CUDA_R_32I, columns,
              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          label);
    std::array<void *, 10> arguments = {
        &partial, &activation_scale, &weight_scale, &aggregate,
        &rows, &columns, &scale_groups, &activation_scale_groups,
        &weight_scale_groups, &group};
    check(counted_launch_kernel(
              accumulate_function,
              h3_w8a8_grid(static_cast<std::uint64_t>(rows) * columns),
              1U, 1U, 256U, 1U, 1U, 0U, stream, arguments.data(), nullptr),
          "cuLaunchKernel H3 ConvRot chunk accumulation");
  }
}

void launch_h3_convrot_bf16_weight_dequant(
    CUfunction function, CUdeviceptr weight, CUdeviceptr scale,
    CUdeviceptr output, std::uint64_t rows, std::uint64_t columns,
    CUstream stream) {
  std::array<void *, 5> arguments = {
      &weight, &scale, &output, &rows, &columns};
  check(counted_launch_kernel(
            function, h3_w8a8_grid(rows * columns), 1U, 1U, 256U, 1U, 1U,
            0U, stream, arguments.data(), nullptr),
        "cuLaunchKernel H3 ConvRot BF16 correction weight dequant");
}

void launch_h3_convrot_bf16_rotate_gather(
    CUfunction function, CUdeviceptr input, CUdeviceptr indices,
    CUdeviceptr output, int rows, int columns, CUstream stream,
    const char *label) {
  constexpr int block_threads = 512;
  constexpr std::uint64_t shared_floats = 8U * 2U * 256U;
  std::array<void *, 5> arguments = {
      &input, &indices, &output, &rows, &columns};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads,
            1U, 1U, static_cast<unsigned>(shared_floats * sizeof(float)),
            stream, arguments.data(), nullptr),
        label);
}

void launch_h3_convrot_compact_bf16_rotate_gather(
    CUfunction function, const H3CompactAdaLNBinding &binding,
    const DeviceBuffers &buffers, CUdeviceptr correction_indices,
    CUdeviceptr output, int rows, int columns, CUstream stream,
    const char *label) {
  constexpr int block_threads = 512;
  constexpr std::uint64_t shared_floats = 8U * 2U * 256U;
  auto input = buffers.at(binding.norm_input_tensor);
  auto weight = buffers.at(binding.norm_weight_tensor);
  auto modulation = buffers.at(binding.modulation_tensor);
  auto adaln_indices = buffers.at(binding.indices_tensor);
  auto scale_lane = static_cast<int>(binding.scale_lane);
  auto shift_lane = static_cast<int>(binding.shift_lane);
  auto epsilon = binding.epsilon;
  std::array<void *, 11> arguments = {
      &input,       &weight,     &modulation, &adaln_indices,
      &correction_indices, &output, &rows,       &columns,
      &scale_lane,  &shift_lane, &epsilon};
  check(counted_launch_kernel(
            function, static_cast<unsigned>(rows), 1U, 1U, block_threads,
            1U, 1U, static_cast<unsigned>(shared_floats * sizeof(float)),
            stream, arguments.data(), nullptr),
        label);
}

void launch_h3_convrot_bf16_gemm(
    cublasHandle_t cublas, CUdeviceptr activation, CUdeviceptr weight,
    CUdeviceptr output, int rows, int columns, int contraction,
    CUstream stream, const char *label) {
  constexpr float alpha = 1.0F;
  constexpr float beta = 0.0F;
  check(counted_cublas_gemm_ex(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, contraction,
            &alpha, reinterpret_cast<const void *>(weight), CUDA_R_16BF,
            contraction, reinterpret_cast<const void *>(activation),
            CUDA_R_16BF, contraction, &beta,
            reinterpret_cast<void *>(output), CUDA_R_16BF, columns,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
        label);
  (void)stream;
}

void launch_h3_convrot_qkv_bf16_correction(
    const H3W8A8AttentionPlan &plan,
    const H3ConvRotFunctions &functions,
    const H3ConvRotBf16Correction *correction,
    const DeviceBuffers &buffers, cublasHandle_t cublas, CUstream stream) {
  if (!correction || correction->rows.empty())
    return;
  if (plan.qkv_weight_scale_groups != 1U)
    fail("H3 BF16 row correction currently requires one ConvRot weight scale per output channel");
  auto rows = static_cast<int>(correction->rows.size());
  const auto hidden = static_cast<int>(plan.hidden);
  const auto packed_inner = static_cast<int>(plan.packed_inner);
  auto inner = static_cast<int>(plan.inner);
  if (plan.compact_adaln.enabled)
    launch_h3_convrot_compact_bf16_rotate_gather(
        functions.compact_bf16_rotate_gather, plan.compact_adaln, buffers,
        correction->indices, correction->activation, rows, hidden, stream,
        "cuLaunchKernel H3 compact AdaLN BF16 QKV correction gather");
  else
    launch_h3_convrot_bf16_rotate_gather(
        functions.bf16_rotate_gather,
        buffers.at(plan.attention_input_tensor), correction->indices,
        correction->activation, rows, hidden, stream,
        "cuLaunchKernel H3 BF16 QKV correction gather");
  launch_h3_convrot_bf16_weight_dequant(
      functions.generic_weight_dequant, plan.qkv_weight_device,
      plan.qkv_scale_device, correction->weight, plan.packed_inner,
      plan.hidden, stream);
  launch_h3_convrot_bf16_gemm(
      cublas, correction->activation, correction->weight,
      correction->projected, rows, packed_inner, hidden, stream,
      "cublasGemmEx H3 ConvRot BF16 QKV correction");
  auto projected = correction->projected;
  auto indices = correction->indices;
  auto q = buffers.at(plan.qkv_output_tensors.at(0));
  auto k = buffers.at(plan.qkv_output_tensors.at(1));
  auto v = buffers.at(plan.qkv_output_tensors.at(2));
  std::array<void *, 7> arguments = {
      &projected, &indices, &q, &k, &v, &rows, &inner};
  check(counted_launch_kernel(
            functions.qkv_bf16_scatter,
            h3_w8a8_grid(static_cast<std::uint64_t>(rows) * plan.inner),
            1U, 1U, 256U, 1U, 1U, 0U, stream, arguments.data(), nullptr),
        "cuLaunchKernel H3 ConvRot BF16 QKV correction scatter");
}

void launch_h3_convrot_residual_bf16_correction(
    const H3CompactAdaLNBinding &binding,
    const H3ConvRotFunctions &functions,
    const H3ConvRotBf16Correction *correction,
    const DeviceBuffers &buffers, CUdeviceptr projected,
    CUdeviceptr residual, CUdeviceptr gate, CUdeviceptr output,
    std::uint64_t hidden, CUstream stream, const char *label) {
  if (!correction || correction->rows.empty())
    return;
  auto rows = static_cast<int>(correction->rows.size());
  auto hidden_i32 = static_cast<int>(hidden);
  auto indices = correction->indices;
  const auto grid = h3_w8a8_grid(
      static_cast<std::uint64_t>(rows) * hidden);
  if (binding.enabled) {
    auto modulation = buffers.at(binding.modulation_tensor);
    auto adaln_indices = buffers.at(binding.indices_tensor);
    auto gate_lane = static_cast<int>(binding.gate_lane);
    std::array<void *, 9> arguments = {
        &projected, &indices, &residual, &modulation, &adaln_indices,
        &output, &rows, &hidden_i32, &gate_lane};
    check(counted_launch_kernel(
              functions.compact_residual_bf16_scatter, grid, 1U, 1U, 256U,
              1U, 1U, 0U, stream, arguments.data(), nullptr),
          label);
  } else {
    std::array<void *, 7> arguments = {
        &projected, &indices, &residual, &gate, &output, &rows,
        &hidden_i32};
    check(counted_launch_kernel(
              functions.residual_bf16_scatter, grid, 1U, 1U, 256U, 1U, 1U,
              0U, stream, arguments.data(), nullptr),
          label);
  }
}

void launch_h3_convrot_output_bf16_correction(
    const H3W8A8AttentionPlan &plan,
    const H3ConvRotFunctions &functions,
    const H3ConvRotBf16Correction *correction,
    const DeviceBuffers &buffers, cublasHandle_t cublas, CUstream stream) {
  if (!correction || correction->rows.empty())
    return;
  if (plan.output_weight_scale_groups != 1U)
    fail("H3 BF16 row correction currently requires one ConvRot weight scale per output channel");
  const auto rows = static_cast<int>(correction->rows.size());
  const auto inner = static_cast<int>(plan.inner);
  const auto hidden = static_cast<int>(plan.hidden);
  launch_h3_convrot_bf16_rotate_gather(
      functions.bf16_rotate_gather, buffers.at(plan.output_input_tensor),
      correction->indices, correction->activation, rows, inner, stream,
      "cuLaunchKernel H3 BF16 output correction gather");
  launch_h3_convrot_bf16_weight_dequant(
      functions.generic_weight_dequant, plan.output_weight_device,
      plan.output_scale_device, correction->weight, plan.hidden, plan.inner,
      stream);
  launch_h3_convrot_bf16_gemm(
      cublas, correction->activation, correction->weight,
      correction->projected, rows, hidden, inner, stream,
      "cublasGemmEx H3 ConvRot BF16 output correction");
  launch_h3_convrot_residual_bf16_correction(
      plan.compact_adaln, functions, correction, buffers,
      correction->projected, buffers.at(plan.residual_tensor),
      plan.compact_adaln.enabled ? CUdeviceptr{} : buffers.at(plan.gate_tensor),
      buffers.at(plan.output_tensor), plan.hidden,
      stream, "cuLaunchKernel H3 ConvRot BF16 output correction scatter");
}

void launch_h3_convrot_mlp_bf16_correction(
    const H3W8A8MlpPlan &plan, const H3ConvRotFunctions &functions,
    const H3ConvRotBf16Correction *correction,
    const DeviceBuffers &buffers, cublasHandle_t cublas, CUstream stream) {
  if (!correction || correction->rows.empty())
    return;
  if (plan.fc1_weight_scale_groups != 1U ||
      plan.fc2_weight_scale_groups != 1U)
    fail("H3 BF16 row correction currently requires one ConvRot weight scale per output channel");
  auto rows = static_cast<int>(correction->rows.size());
  auto hidden = static_cast<int>(plan.hidden);
  auto ffn = static_cast<int>(plan.ffn);
  auto packed_ffn = static_cast<int>(plan.packed_ffn);
  if (plan.compact_adaln.enabled)
    launch_h3_convrot_compact_bf16_rotate_gather(
        functions.compact_bf16_rotate_gather, plan.compact_adaln, buffers,
        correction->indices, correction->activation, rows, hidden, stream,
        "cuLaunchKernel H3 compact AdaLN BF16 MLP correction gather");
  else
    launch_h3_convrot_bf16_rotate_gather(
        functions.bf16_rotate_gather, buffers.at(plan.input_tensor),
        correction->indices, correction->activation, rows, hidden, stream,
        "cuLaunchKernel H3 BF16 MLP correction gather");
  launch_h3_convrot_bf16_weight_dequant(
      functions.generic_weight_dequant, plan.fc1_weight_device,
      plan.fc1_scale_device, correction->weight, plan.packed_ffn,
      plan.hidden, stream);
  launch_h3_convrot_bf16_gemm(
      cublas, correction->activation, correction->weight,
      correction->projected, rows, packed_ffn, hidden, stream,
      "cublasGemmEx H3 ConvRot BF16 FC1 correction");
  auto projected = correction->projected;
  auto activation = correction->auxiliary;
  std::array<void *, 4> swiglu_arguments = {
      &projected, &activation, &rows, &ffn};
  check(counted_launch_kernel(
            functions.swiglu_bf16,
            h3_w8a8_grid(static_cast<std::uint64_t>(rows) * plan.ffn),
            1U, 1U, 256U, 1U, 1U, 0U, stream, swiglu_arguments.data(),
            nullptr),
        "cuLaunchKernel H3 ConvRot BF16 SwiGLU correction");
  auto row_start = 0;
  auto rotated_activation = correction->activation;
  constexpr std::uint64_t shared_floats = 8U * 2U * 256U;
  std::array<void *, 5> rotate_arguments = {
      &activation, &rotated_activation, &row_start, &rows, &ffn};
  check(counted_launch_kernel(
            functions.generic_bf16_rotate, static_cast<unsigned>(rows), 1U,
            1U, 512U, 1U, 1U,
            static_cast<unsigned>(shared_floats * sizeof(float)), stream,
            rotate_arguments.data(), nullptr),
        "cuLaunchKernel H3 ConvRot BF16 SwiGLU correction rotate");
  launch_h3_convrot_bf16_weight_dequant(
      functions.generic_weight_dequant, plan.fc2_weight_device,
      plan.fc2_scale_device, correction->weight, plan.hidden, plan.ffn,
      stream);
  launch_h3_convrot_bf16_gemm(
      cublas, correction->activation, correction->weight,
      correction->projected, rows, hidden, ffn, stream,
      "cublasGemmEx H3 ConvRot BF16 FC2 correction");
  launch_h3_convrot_residual_bf16_correction(
      plan.compact_adaln, functions, correction, buffers,
      correction->projected, buffers.at(plan.residual_tensor),
      plan.compact_adaln.enabled ? CUdeviceptr{} : buffers.at(plan.gate_tensor),
      buffers.at(plan.output_tensor), plan.hidden,
      stream, "cuLaunchKernel H3 ConvRot BF16 MLP correction scatter");
}

void launch_h3_w8a8_qkv(const H3W8A8AttentionPlan &plan,
                         const H3W8A8Functions &functions,
                         const H3ConvRotFunctions &convrot_functions,
                         const DeviceBuffers &buffers,
                         const H3ConvRotBf16Correction *correction,
                         cublasHandle_t cublas,
                         const H3Int8GemmRegistry *registry,
#if DIF_HAS_CUTLASS
                         const H3Int8ScaledGemmRegistry *scaled_registry,
#endif
                         const Workspace *workspace, CUstream stream) {
  auto scale = plan.activation_scale_device;
  auto zero = 0;
  auto rows = static_cast<int>(plan.rows);
  auto hidden = static_cast<int>(plan.hidden);
  auto encoded = plan.activation_i8_device;
  if (plan.convrot_scale_chunk != 0U &&
      !plan.convrot_global_activation_scale && plan.compact_adaln.enabled)
    launch_h3_convrot_compact_adaln_chunked_encode(
        convrot_functions.compact_chunked_encode, plan.compact_adaln, buffers,
        encoded, scale, zero, rows, hidden,
        static_cast<int>(plan.convrot_scale_chunk), stream,
        "cuLaunchKernel H3 compact AdaLN chunk-scaled ConvRot QKV encode");
  else if (plan.convrot_scale_chunk != 0U &&
           !plan.convrot_global_activation_scale)
    launch_h3_convrot_chunked_encode(
        convrot_functions.chunked_encode,
        buffers.at(plan.attention_input_tensor), encoded, scale, zero, rows,
        hidden, static_cast<int>(plan.convrot_scale_chunk), stream,
        "cuLaunchKernel H3 chunk-scaled ConvRot QKV encode");
  else if (plan.compact_adaln.enabled)
    launch_h3_convrot_compact_adaln_encode(
        convrot_functions.compact_encode, plan.compact_adaln, buffers,
        encoded, scale, zero, rows, hidden, stream,
        "cuLaunchKernel H3 compact AdaLN ConvRot QKV encode");
  else if (plan.convrot)
    launch_h3_convrot_encode(convrot_functions.encode,
                             buffers.at(plan.attention_input_tensor), encoded,
                             scale, zero, rows, hidden, stream,
                             "cuLaunchKernel H3 ConvRot QKV encode");
  else {
    auto input = buffers.at(plan.attention_input_tensor);
    std::array<void *, 5> rowscale_arguments = {
        &input, &scale, &zero, &hidden, &rows};
    launch_h3_w8a8_kernel(functions.rowscale, static_cast<unsigned>(rows),
                           rowscale_arguments, stream,
                           "cuLaunchKernel H3 W8A8 QKV rowscale");
    std::array<void *, 6> encode_arguments = {
        &input, &scale, &encoded, &zero, &rows, &hidden};
    check(counted_launch_kernel(functions.encode,
                         h3_w8a8_grid(plan.rows * plan.hidden), 1U, 1U, 256U,
                         1U, 1U, 0U, stream, encode_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 QKV encode");
  }

  auto packed_inner = static_cast<int>(plan.packed_inner);
  auto inner = static_cast<int>(plan.inner);
  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += kH3W8A8ProjectionChunkRows) {
    auto chunk_rows = static_cast<int>(std::min<std::uint64_t>(
        kH3W8A8ProjectionChunkRows, plan.rows - row_start));
    auto chunk_input = plan.activation_i8_device + row_start * plan.hidden;
    if (plan.convrot_scale_chunk != 0U) {
      const auto activation_scale_groups =
          plan.convrot_global_activation_scale
              ? 1U
              : (plan.hidden + plan.convrot_scale_chunk - 1U) /
                    plan.convrot_scale_chunk;
      launch_h3_convrot_chunked_gemm(
          cublas, convrot_functions.chunk_accumulate, chunk_input,
          plan.qkv_weight_device,
          plan.activation_scale_device +
              row_start * activation_scale_groups * sizeof(float),
          plan.qkv_scale_device, plan.accumulator_device,
          plan.aggregate_device, chunk_rows, packed_inner, hidden,
          static_cast<int>(plan.convrot_scale_chunk),
          static_cast<int>(activation_scale_groups),
          static_cast<int>(plan.qkv_weight_scale_groups), stream,
          "cublasGemmEx H3 chunk-scaled ConvRot QKV");
    }
#if DIF_HAS_CUTLASS
    else if (plan.cutlass_scaled) {
      if (!scaled_registry)
        fail("H3 CUTLASS scaled QKV plan is missing its prepared registry");
      scaled_registry->launch(
          {static_cast<std::uint32_t>(chunk_rows),
           static_cast<std::uint32_t>(packed_inner),
           static_cast<std::uint32_t>(hidden)},
          chunk_input, plan.qkv_weight_device,
          plan.activation_scale_device + row_start * sizeof(float),
          plan.qkv_scale_device, plan.accumulator_device, stream);
    } else
#endif
      launch_h3_direct_int8_gemm(
          cublas, registry, workspace, chunk_input, plan.qkv_weight_device,
          plan.accumulator_device, chunk_rows, packed_inner, hidden, stream,
          "cublasGemmEx H3 W8A8 QKV");
    auto accumulator = plan.accumulator_device;
    auto weight_scale = plan.qkv_scale_device;
    auto q = buffers.at(plan.qkv_output_tensors.at(0));
    auto k = buffers.at(plan.qkv_output_tensors.at(1));
    auto v = buffers.at(plan.qkv_output_tensors.at(2));
    auto row_start_i32 = static_cast<int>(row_start);
    const auto grid = h3_w8a8_grid(
        static_cast<std::uint64_t>(chunk_rows) * plan.inner);
    if (plan.convrot_scale_chunk != 0U) {
      auto projected = plan.aggregate_device;
      std::array<void *, 7> arguments = {
          &projected, &q, &k, &v, &row_start_i32, &chunk_rows, &inner};
      check(counted_launch_kernel(convrot_functions.qkv_f32, grid, 1U, 1U,
                                  256U, 1U, 1U, 0U, stream,
                                  arguments.data(), nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 QKV split");
    } else if (plan.cutlass_scaled) {
      std::array<void *, 7> arguments = {
          &accumulator, &q, &k, &v, &row_start_i32, &chunk_rows, &inner};
      check(counted_launch_kernel(convrot_functions.qkv_bf16, grid, 1U, 1U,
                                  256U, 1U, 1U, 0U, stream,
                                  arguments.data(), nullptr),
            "cuLaunchKernel H3 ConvRot BF16 QKV split");
    } else if (plan.convrot) {
      std::array<void *, 9> arguments = {
          &accumulator, &scale, &weight_scale, &q, &k, &v,
          &row_start_i32, &chunk_rows, &inner};
      check(counted_launch_kernel(convrot_functions.qkv, grid, 1U, 1U, 256U,
                                  1U, 1U, 0U, stream, arguments.data(),
                                  nullptr),
            "cuLaunchKernel H3 ConvRot QKV dequant");
    } else {
      std::array<void *, 9> arguments = {
          &accumulator, &scale, &weight_scale, &q, &k, &v,
          &row_start_i32, &chunk_rows, &inner};
      check(counted_launch_kernel(functions.qkv, grid, 1U, 1U, 256U, 1U, 1U,
                                  0U, stream, arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 QKV dequant");
    }
  }
  if (plan.convrot)
    launch_h3_convrot_qkv_bf16_correction(
        plan, convrot_functions, correction, buffers, cublas, stream);
}

void launch_h3_w8a8_output(const H3W8A8AttentionPlan &plan,
                            const H3W8A8Functions &functions,
                            const H3ConvRotFunctions &convrot_functions,
                            const DeviceBuffers &buffers,
                            const H3ConvRotBf16Correction *correction,
                            cublasHandle_t cublas,
                            const H3Int8GemmRegistry *registry,
#if DIF_HAS_CUTLASS
                            const H3Int8ScaledGemmRegistry *scaled_registry,
#endif
                            const Workspace *workspace, CUstream stream) {
  auto input = buffers.at(plan.output_input_tensor);
  auto residual = buffers.at(plan.residual_tensor);
  auto output = buffers.at(plan.output_tensor);
  auto inner = static_cast<int>(plan.inner);
  auto hidden = static_cast<int>(plan.hidden);
  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += kH3W8A8ProjectionChunkRows) {
    auto rows = static_cast<int>(std::min<std::uint64_t>(
        kH3W8A8ProjectionChunkRows, plan.rows - row_start));
    auto row_start_i32 = static_cast<int>(row_start);
    auto scale = plan.activation_scale_device;
    auto encoded = plan.activation_i8_device;
    if (plan.convrot_scale_chunk != 0U &&
        !plan.convrot_global_activation_scale)
      launch_h3_convrot_chunked_encode(
          convrot_functions.chunked_encode, input, encoded, scale,
          row_start_i32, rows, inner,
          static_cast<int>(plan.convrot_scale_chunk), stream,
          "cuLaunchKernel H3 chunk-scaled ConvRot output encode");
    else if (plan.convrot)
      launch_h3_convrot_encode(convrot_functions.encode, input, encoded, scale,
                               row_start_i32, rows, inner, stream,
                               "cuLaunchKernel H3 ConvRot output encode");
    else {
      std::array<void *, 5> rowscale_arguments = {
          &input, &scale, &row_start_i32, &inner, &rows};
      launch_h3_w8a8_kernel(functions.rowscale, static_cast<unsigned>(rows),
                             rowscale_arguments, stream,
                             "cuLaunchKernel H3 W8A8 output rowscale");
      std::array<void *, 6> encode_arguments = {
          &input, &scale, &encoded, &row_start_i32, &rows, &inner};
      check(counted_launch_kernel(functions.encode,
                           h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                       plan.inner),
                           1U, 1U, 256U, 1U, 1U, 0U, stream,
                           encode_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 output encode");
    }
    if (plan.convrot_scale_chunk != 0U) {
      launch_h3_convrot_chunked_gemm(
          cublas, convrot_functions.chunk_accumulate,
          plan.activation_i8_device, plan.output_weight_device,
          plan.activation_scale_device, plan.output_scale_device,
          plan.accumulator_device, plan.aggregate_device, rows, hidden, inner,
          static_cast<int>(plan.convrot_scale_chunk),
          plan.convrot_global_activation_scale
              ? 1
              : static_cast<int>((plan.inner + plan.convrot_scale_chunk - 1U) /
                                 plan.convrot_scale_chunk),
          static_cast<int>(plan.output_weight_scale_groups), stream,
          "cublasGemmEx H3 chunk-scaled ConvRot output projection");
    }
#if DIF_HAS_CUTLASS
    else if (plan.cutlass_scaled) {
      if (!scaled_registry)
        fail("H3 CUTLASS scaled output plan is missing its prepared registry");
      scaled_registry->launch(
          {static_cast<std::uint32_t>(rows),
           static_cast<std::uint32_t>(hidden),
           static_cast<std::uint32_t>(inner)},
          plan.activation_i8_device, plan.output_weight_device,
          plan.activation_scale_device, plan.output_scale_device,
          plan.accumulator_device, stream);
    } else
#endif
      launch_h3_direct_int8_gemm(
          cublas, registry, workspace, plan.activation_i8_device,
          plan.output_weight_device, plan.accumulator_device, rows, hidden,
          inner, stream, "cublasGemmEx H3 W8A8 output projection");
    auto accumulator = plan.accumulator_device;
    auto weight_scale = plan.output_scale_device;
    const auto grid = h3_w8a8_grid(
        static_cast<std::uint64_t>(rows) * plan.hidden);
    if (plan.convrot_scale_chunk != 0U && plan.compact_adaln.enabled) {
      auto projected = plan.aggregate_device;
      auto modulation = buffers.at(plan.compact_adaln.modulation_tensor);
      auto indices = buffers.at(plan.compact_adaln.indices_tensor);
      auto gate_lane = static_cast<int>(plan.compact_adaln.gate_lane);
      std::array<void *, 9> residual_arguments = {
          &projected, &residual, &modulation, &indices, &output,
          &row_start_i32, &rows, &hidden, &gate_lane};
      check(counted_launch_kernel(
                convrot_functions.compact_residual_f32, grid, 1U, 1U,
                256U, 1U, 1U, 0U, stream, residual_arguments.data(),
                nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 output residual");
    } else if (plan.convrot_scale_chunk != 0U) {
      auto projected = plan.aggregate_device;
      auto gate = buffers.at(plan.gate_tensor);
      std::array<void *, 7> residual_arguments = {
          &projected, &residual, &gate, &output,
          &row_start_i32, &rows, &hidden};
      check(counted_launch_kernel(
                convrot_functions.residual_f32, grid, 1U, 1U, 256U, 1U,
                1U, 0U, stream, residual_arguments.data(), nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 output residual");
    } else if (plan.compact_adaln.enabled) {
      auto modulation = buffers.at(plan.compact_adaln.modulation_tensor);
      auto indices = buffers.at(plan.compact_adaln.indices_tensor);
      auto gate_lane = static_cast<int>(plan.compact_adaln.gate_lane);
      if (plan.cutlass_scaled) {
        std::array<void *, 9> residual_arguments = {
            &accumulator, &residual, &modulation, &indices, &output,
            &row_start_i32, &rows, &hidden, &gate_lane};
        check(counted_launch_kernel(
                  convrot_functions.compact_residual_bf16, grid, 1U, 1U,
                  256U, 1U, 1U, 0U, stream, residual_arguments.data(),
                  nullptr),
              "cuLaunchKernel H3 compact AdaLN BF16 output residual");
      } else {
        std::array<void *, 11> residual_arguments = {
            &accumulator, &scale, &weight_scale, &residual, &modulation,
            &indices, &output, &row_start_i32, &rows, &hidden, &gate_lane};
        check(counted_launch_kernel(
                  convrot_functions.compact_residual, grid, 1U, 1U, 256U,
                  1U, 1U, 0U, stream, residual_arguments.data(), nullptr),
              "cuLaunchKernel H3 compact AdaLN output residual");
      }
    } else {
      auto gate = buffers.at(plan.gate_tensor);
      std::array<void *, 9> residual_arguments = {
          &accumulator, &scale, &weight_scale, &residual, &gate, &output,
          &row_start_i32, &rows, &hidden};
      check(counted_launch_kernel(functions.residual, grid, 1U, 1U, 256U,
                                  1U, 1U, 0U, stream,
                                  residual_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 output residual");
    }
  }
  if (plan.convrot)
    launch_h3_convrot_output_bf16_correction(
        plan, convrot_functions, correction, buffers, cublas, stream);
}

void launch_h3_w8a8_mlp(const H3W8A8MlpPlan &plan,
                         const H3W8A8Functions &functions,
                         const H3ConvRotFunctions &convrot_functions,
                         const DeviceBuffers &buffers,
                         const H3ConvRotBf16Correction *correction,
                         cublasHandle_t cublas,
                         const H3Int8GemmRegistry *registry,
#if DIF_HAS_CUTLASS
                         const H3Int8ScaledGemmRegistry *scaled_registry,
#endif
                         const Workspace *workspace, CUstream stream) {
  const auto residual = buffers.at(plan.residual_tensor);
  const auto output = buffers.at(plan.output_tensor);
  auto hidden = static_cast<int>(plan.hidden);
  auto ffn = static_cast<int>(plan.ffn);
  auto packed_ffn = static_cast<int>(plan.packed_ffn);
  auto gemm = [&](CUdeviceptr activation, CUdeviceptr weight,
                  CUdeviceptr accumulator, int rows, int columns,
                  int contraction) {
    launch_h3_direct_int8_gemm(
        cublas, registry, workspace, activation, weight, accumulator, rows,
        columns, contraction, stream, "cublasGemmEx H3 W8A8");
  };

  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += plan.chunk_rows) {
    auto rows = static_cast<int>(std::min<std::uint64_t>(
        plan.chunk_rows, plan.rows - row_start));
    auto row_start_i32 = static_cast<int>(row_start);

    auto input_scale = plan.input_scale_device;
    auto input_i8 = plan.input_i8_device;
    if (plan.convrot_scale_chunk != 0U &&
        !plan.convrot_global_activation_scale && plan.compact_adaln.enabled)
      launch_h3_convrot_compact_adaln_chunked_encode(
          convrot_functions.compact_chunked_encode, plan.compact_adaln,
          buffers, input_i8, input_scale, row_start_i32, rows, hidden,
          static_cast<int>(plan.convrot_scale_chunk), stream,
          "cuLaunchKernel H3 compact AdaLN chunk-scaled ConvRot MLP input encode");
    else if (plan.convrot_scale_chunk != 0U &&
             !plan.convrot_global_activation_scale)
      launch_h3_convrot_chunked_encode(
          convrot_functions.chunked_encode, buffers.at(plan.input_tensor),
          input_i8, input_scale, row_start_i32, rows, hidden,
          static_cast<int>(plan.convrot_scale_chunk), stream,
          "cuLaunchKernel H3 chunk-scaled ConvRot MLP input encode");
    else if (plan.compact_adaln.enabled)
      launch_h3_convrot_compact_adaln_encode(
          convrot_functions.compact_encode, plan.compact_adaln, buffers,
          input_i8, input_scale, row_start_i32, rows, hidden, stream,
          "cuLaunchKernel H3 compact AdaLN ConvRot MLP input encode");
    else if (plan.convrot)
      launch_h3_convrot_encode(convrot_functions.encode,
                               buffers.at(plan.input_tensor), input_i8,
                               input_scale, row_start_i32, rows, hidden,
                               stream,
                               "cuLaunchKernel H3 ConvRot MLP input encode");
    else {
      auto input = buffers.at(plan.input_tensor);
      std::array<void *, 5> rowscale_arguments = {
          &input, &input_scale, &row_start_i32, &hidden, &rows};
      launch_h3_w8a8_kernel(functions.rowscale,
                             static_cast<unsigned>(rows), rowscale_arguments,
                             stream, "cuLaunchKernel H3 W8A8 input rowscale");
      std::array<void *, 6> full_encode_arguments = {
          &input, &input_scale, &input_i8, &row_start_i32, &rows, &hidden};
      check(counted_launch_kernel(functions.encode,
                           h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                       plan.hidden),
                           1U, 1U, 256U, 1U, 1U, 0U, stream,
                           full_encode_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 input encode");
    }

    if (plan.convrot_scale_chunk != 0U) {
      launch_h3_convrot_chunked_gemm(
          cublas, convrot_functions.chunk_accumulate, plan.input_i8_device,
          plan.fc1_weight_device, plan.input_scale_device,
          plan.fc1_scale_device, plan.fc1_accumulator_device,
          plan.fc1_aggregate_device, rows, packed_ffn, hidden,
          static_cast<int>(plan.convrot_scale_chunk),
          plan.convrot_global_activation_scale
              ? 1
              : static_cast<int>((plan.hidden + plan.convrot_scale_chunk - 1U) /
                                 plan.convrot_scale_chunk),
          static_cast<int>(plan.fc1_weight_scale_groups), stream,
          "cublasGemmEx H3 chunk-scaled ConvRot FC1");
    }
#if DIF_HAS_CUTLASS
    else if (plan.cutlass_scaled_fc1) {
      if (!scaled_registry)
        fail("H3 CUTLASS scaled FC1 plan is missing its prepared registry");
      scaled_registry->launch(
          {static_cast<std::uint32_t>(rows),
           static_cast<std::uint32_t>(packed_ffn),
           static_cast<std::uint32_t>(hidden)},
          plan.input_i8_device, plan.fc1_weight_device,
          plan.input_scale_device, plan.fc1_scale_device,
          plan.fc1_accumulator_device, stream);
    } else
#endif
      gemm(plan.input_i8_device, plan.fc1_weight_device,
           plan.fc1_accumulator_device, rows, packed_ffn, hidden);

    auto rowscale_activation = plan.activation_device;
    auto rowscale_activation_output = plan.activation_scale_device;
    auto zero = 0;
    auto activation_scale = plan.activation_scale_device;
    auto activation_i8 = plan.activation_i8_device;
    auto swiglu_accumulator = plan.fc1_accumulator_device;
    auto swiglu_input_scale = plan.input_scale_device;
    auto swiglu_weight_scale = plan.fc1_scale_device;
    if (plan.convrot_scale_chunk != 0U) {
      auto projected = plan.fc1_aggregate_device;
      auto activation = plan.activation_device;
      const auto swiglu_grid = h3_w8a8_grid(
          static_cast<std::uint64_t>(rows) * plan.ffn);
      std::array<void *, 4> swiglu_arguments = {
          &projected, &activation, &rows, &ffn};
      check(counted_launch_kernel(
                convrot_functions.swiglu_f32, swiglu_grid, 1U, 1U, 256U,
                1U, 1U, 0U, stream, swiglu_arguments.data(), nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 SwiGLU");
      if (plan.convrot_global_activation_scale)
        launch_h3_convrot_encode(
            convrot_functions.encode, activation, activation_i8,
            activation_scale, zero, rows, ffn, stream,
            "cuLaunchKernel H3 global-scale ConvRot SwiGLU encode");
      else
        launch_h3_convrot_chunked_encode(
            convrot_functions.chunked_encode, activation, activation_i8,
            activation_scale, zero, rows, ffn,
            static_cast<int>(plan.convrot_scale_chunk), stream,
            "cuLaunchKernel H3 chunk-scaled ConvRot SwiGLU encode");
    } else if (plan.convrot) {
      constexpr std::uint64_t temporary_floats = 8U * 2U * 256U;
      const auto shared_bytes =
          (plan.ffn + temporary_floats) * sizeof(float);
      if (plan.cutlass_scaled_fc1) {
        std::array<void *, 5> arguments = {
            &swiglu_accumulator, &activation_i8, &activation_scale, &rows,
            &ffn};
        check(counted_launch_kernel(
                  convrot_functions.swiglu_bf16_encode,
                  static_cast<unsigned>(rows), 1U, 1U, 512U, 1U, 1U,
                  static_cast<unsigned>(shared_bytes), stream,
                  arguments.data(), nullptr),
              "cuLaunchKernel H3 ConvRot BF16 SwiGLU encode");
      } else {
        std::array<void *, 7> arguments = {
            &swiglu_accumulator, &swiglu_input_scale, &swiglu_weight_scale,
            &activation_i8, &activation_scale, &rows, &ffn};
        check(counted_launch_kernel(
                  convrot_functions.swiglu_encode,
                  static_cast<unsigned>(rows), 1U, 1U, 512U, 1U, 1U,
                  static_cast<unsigned>(shared_bytes), stream,
                  arguments.data(), nullptr),
              "cuLaunchKernel H3 ConvRot SwiGLU encode");
      }
    } else {
      auto swiglu_output = plan.activation_device;
      std::array<void *, 6> swiglu_arguments = {
          &swiglu_accumulator, &swiglu_input_scale, &swiglu_weight_scale,
          &swiglu_output, &rows, &ffn};
      check(counted_launch_kernel(functions.swiglu,
                           h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                       plan.ffn),
                           1U, 1U, 256U, 1U, 1U, 0U, stream,
                           swiglu_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 SwiGLU");
      std::array<void *, 5> activation_rowscale_arguments = {
          &rowscale_activation, &rowscale_activation_output, &zero, &ffn,
          &rows};
      launch_h3_w8a8_kernel(
          functions.rowscale, static_cast<unsigned>(rows),
          activation_rowscale_arguments, stream,
          "cuLaunchKernel H3 W8A8 activation rowscale");
      std::array<void *, 6> activation_encode_arguments = {
          &rowscale_activation, &activation_scale, &activation_i8,
          &zero, &rows, &ffn};
      check(counted_launch_kernel(functions.encode,
                           h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                       plan.ffn),
                           1U, 1U, 256U, 1U, 1U, 0U, stream,
                           activation_encode_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 activation encode");
    }

    if (plan.convrot_scale_chunk != 0U) {
      launch_h3_convrot_chunked_gemm(
          cublas, convrot_functions.chunk_accumulate,
          plan.activation_i8_device, plan.fc2_weight_device,
          plan.activation_scale_device, plan.fc2_scale_device,
          plan.fc2_accumulator_device, plan.fc2_aggregate_device, rows,
          hidden, ffn, static_cast<int>(plan.convrot_scale_chunk),
          plan.convrot_global_activation_scale
              ? 1
              : static_cast<int>((plan.ffn + plan.convrot_scale_chunk - 1U) /
                                 plan.convrot_scale_chunk),
          static_cast<int>(plan.fc2_weight_scale_groups), stream,
          "cublasGemmEx H3 chunk-scaled ConvRot FC2");
    }
#if DIF_HAS_CUTLASS
    else if (plan.cutlass_scaled_fc2) {
      if (!scaled_registry)
        fail("H3 CUTLASS scaled FC2 plan is missing its prepared registry");
      scaled_registry->launch(
          {static_cast<std::uint32_t>(rows),
           static_cast<std::uint32_t>(hidden),
           static_cast<std::uint32_t>(ffn)},
          plan.activation_i8_device, plan.fc2_weight_device,
          plan.activation_scale_device, plan.fc2_scale_device,
          plan.fc2_accumulator_device, stream);
    } else
#endif
      gemm(plan.activation_i8_device, plan.fc2_weight_device,
           plan.fc2_accumulator_device, rows, hidden, ffn);

    auto residual_accumulator = plan.fc2_accumulator_device;
    auto residual_scale = plan.activation_scale_device;
    auto residual_weight_scale = plan.fc2_scale_device;
    auto residual_input = residual;
    auto residual_output = output;
    const auto grid = h3_w8a8_grid(
        static_cast<std::uint64_t>(rows) * plan.hidden);
    if (plan.convrot_scale_chunk != 0U && plan.compact_adaln.enabled) {
      auto projected = plan.fc2_aggregate_device;
      auto modulation = buffers.at(plan.compact_adaln.modulation_tensor);
      auto indices = buffers.at(plan.compact_adaln.indices_tensor);
      auto gate_lane = static_cast<int>(plan.compact_adaln.gate_lane);
      std::array<void *, 9> residual_arguments = {
          &projected, &residual_input, &modulation, &indices,
          &residual_output, &row_start_i32, &rows, &hidden, &gate_lane};
      check(counted_launch_kernel(
                convrot_functions.compact_residual_f32, grid, 1U, 1U,
                256U, 1U, 1U, 0U, stream, residual_arguments.data(),
                nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 MLP residual");
    } else if (plan.convrot_scale_chunk != 0U) {
      auto projected = plan.fc2_aggregate_device;
      auto residual_gate = buffers.at(plan.gate_tensor);
      std::array<void *, 7> residual_arguments = {
          &projected, &residual_input, &residual_gate, &residual_output,
          &row_start_i32, &rows, &hidden};
      check(counted_launch_kernel(
                convrot_functions.residual_f32, grid, 1U, 1U, 256U, 1U,
                1U, 0U, stream, residual_arguments.data(), nullptr),
            "cuLaunchKernel H3 chunk-scaled ConvRot F32 MLP residual");
    } else if (plan.compact_adaln.enabled) {
      auto modulation = buffers.at(plan.compact_adaln.modulation_tensor);
      auto indices = buffers.at(plan.compact_adaln.indices_tensor);
      auto gate_lane = static_cast<int>(plan.compact_adaln.gate_lane);
      if (plan.cutlass_scaled_fc2) {
        std::array<void *, 9> residual_arguments = {
            &residual_accumulator, &residual_input, &modulation, &indices,
            &residual_output, &row_start_i32, &rows, &hidden, &gate_lane};
        check(counted_launch_kernel(
                  convrot_functions.compact_residual_bf16, grid, 1U, 1U,
                  256U, 1U, 1U, 0U, stream, residual_arguments.data(),
                  nullptr),
              "cuLaunchKernel H3 compact AdaLN BF16 MLP residual");
      } else {
        std::array<void *, 11> residual_arguments = {
            &residual_accumulator, &residual_scale, &residual_weight_scale,
            &residual_input, &modulation, &indices, &residual_output,
            &row_start_i32, &rows, &hidden, &gate_lane};
        check(counted_launch_kernel(
                  convrot_functions.compact_residual, grid, 1U, 1U, 256U,
                  1U, 1U, 0U, stream, residual_arguments.data(), nullptr),
              "cuLaunchKernel H3 compact AdaLN MLP residual");
      }
    } else {
      auto residual_gate = buffers.at(plan.gate_tensor);
      std::array<void *, 9> residual_arguments = {
          &residual_accumulator, &residual_scale, &residual_weight_scale,
          &residual_input, &residual_gate, &residual_output,
          &row_start_i32, &rows, &hidden};
      check(counted_launch_kernel(functions.residual, grid, 1U, 1U, 256U,
                                  1U, 1U, 0U, stream,
                                  residual_arguments.data(), nullptr),
            "cuLaunchKernel H3 W8A8 residual");
    }
  }
  if (plan.convrot)
    launch_h3_convrot_mlp_bf16_correction(
        plan, convrot_functions, correction, buffers, cublas, stream);
}

std::string read_text(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return {};
  const auto end = input.tellg();
  if (end < 0)
    return {};
  std::string text(static_cast<std::size_t>(end), '\0');
  input.seekg(0);
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  return input ? text : std::string{};
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output)
    fail("cannot create PTX cache file: " + path.string());
  output.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!output)
    fail("cannot write PTX cache file: " + path.string());
}

std::string compile_ptx(const std::string &source, int major, int minor,
                        const std::filesystem::path &cache_directory,
                        std::string &source_hash) {
  // The cache key covers everything that changes the PTX: the source, the
  // architecture, the NVRTC version (two toolkits coexist on this host), and
  // the exact option list (Flame lesson: flags are part of the numerics
  // contract; a cache keyed on name/arch alone served stale PTX).
  int nvrtc_major = 0;
  int nvrtc_minor = 0;
  (void)nvrtcVersion(&nvrtc_major, &nvrtc_minor);
  const std::string key_material =
      source + "\ncompute_" + std::to_string(major) + std::to_string(minor) +
      "\nnvrtc=" + std::to_string(nvrtc_major) + "." +
      std::to_string(nvrtc_minor) +
      "\noptions=--std=c++17;--gpu-architecture;--include-path;--restrict" +
      "\nnvrtc-v2";
  const auto key_bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(key_material.data()), key_material.size());
  source_hash = hex_digest(sha256(key_bytes));
  const auto directory = cache_directory.empty()
                             ? std::filesystem::temp_directory_path() / "dif-ptx-cache"
                             : cache_directory;
  std::filesystem::create_directories(directory);
  const auto cache_file = directory / (source_hash + ".ptx");
  auto cached = read_text(cache_file);
  if (!cached.empty())
    return cached;

  nvrtcProgram program{};
  check(nvrtcCreateProgram(&program, source.c_str(), "candidate.cu", 0, nullptr,
                           nullptr),
        "nvrtcCreateProgram");
  const std::string architecture = "--gpu-architecture=compute_" +
                                   std::to_string(major) + std::to_string(minor);
  const std::string include_path =
      std::string("--include-path=") + DIF_CUDA_INCLUDE_DIR;
  const std::array<const char *, 4> options = {
      "--std=c++17", architecture.c_str(), include_path.c_str(), "--restrict"};
  const auto compile_result =
      nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());
  std::size_t log_size = 0;
  check(nvrtcGetProgramLogSize(program, &log_size), "nvrtcGetProgramLogSize");
  std::string log(log_size, '\0');
  if (log_size > 1U)
    check(nvrtcGetProgramLog(program, log.data()), "nvrtcGetProgramLog");
  if (compile_result != NVRTC_SUCCESS) {
    (void)nvrtcDestroyProgram(&program);
    fail("NVRTC compilation failed: " + log);
  }
  std::size_t ptx_size = 0;
  check(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
  std::string ptx(ptx_size, '\0');
  check(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
  check(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
  write_text(cache_file, ptx);
  return ptx;
}

void validate_inputs(const ir::Program &program, const TensorMap &inputs,
                     const TensorMap &constants) {
  const auto &bound = [&](std::uint32_t id) -> const Tensor & {
    if (const auto found = inputs.find(id); found != inputs.end())
      return found->second;
    return constants.at(id);
  };
  for (const auto &desc : program.tensors) {
    if (!desc.has_role(ir::TensorRole::Input) &&
        !desc.has_role(ir::TensorRole::Constant))
      continue;
    const auto &tensor = bound(desc.id);
    tensor.validate();
    if (tensor.dtype != desc.dtype || tensor.dims != desc.dims)
      fail("bound tensor shape/dtype mismatch for id " + std::to_string(desc.id));
  }
  for (const auto &op : program.operations) {
    if (op.opcode == ir::Opcode::H3AdaLNSelect) {
      const auto &projected = *program.tensor(op.inputs[0]);
      const auto &indices = bound(op.inputs[1]);
      const auto table_rows = projected.dims[0] * 3U;
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= table_rows)
          fail("h3_adaln_select index is out of range");
      }
    } else if (op.opcode == ir::Opcode::SelectRowChunks) {
      const auto rows = program.tensor(op.inputs[0])->dims[0];
      const auto &indices = bound(op.inputs[1]);
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= rows)
          fail("select_row_chunks index is out of range");
      }
    } else if (op.opcode == ir::Opcode::GatherRows) {
      const auto rows = program.tensor(op.inputs[0])->dims[0];
      const auto &indices = bound(op.inputs[1]);
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= rows)
          fail("gather_rows index is out of range");
      }
    } else if (op.opcode == ir::Opcode::IndexedUpdateRows) {
      const auto update_rows = program.tensor(op.inputs[1])->dims[0];
      const auto &map = bound(op.inputs[2]);
      for (std::uint64_t row = 0; row < map.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, map.data() + row * sizeof(value), sizeof(value));
        if (value < -1 ||
            (value >= 0 && static_cast<std::uint64_t>(value) >= update_rows))
          fail("indexed_update_rows map is out of range");
      }
    }
  }
}

// Fixed staging granule: page-aligned, large enough that per-chunk atomic
// overhead vanishes, small enough that participants stay load-balanced.
constexpr std::size_t kStageChunkBytes = 4U * 1024U * 1024U;

// One mmap->pinned staging copy split across a persistent worker pool.
// Each copy() publishes an immutable job object; workers pull 4 MiB chunks
// through the job's atomic cursor. A straggler that wakes late only ever
// sees its own job's exhausted cursor, so consecutive jobs cannot tear.
// With one thread (the default) copy() degrades to plain memcpy — the
// historical behavior.
class StagingPool {
public:
  explicit StagingPool(std::uint32_t threads) {
    for (std::uint32_t index = 1U; index < threads; ++index)
      workers_.emplace_back([this] { work(); });
  }

  ~StagingPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    available_.notify_all();
    for (auto &worker : workers_)
      worker.join();
  }

  StagingPool(const StagingPool &) = delete;
  StagingPool &operator=(const StagingPool &) = delete;

  void copy(std::uint8_t *destination, const std::uint8_t *source,
            std::size_t bytes) {
    if (workers_.empty() || bytes < 2U * kStageChunkBytes) {
      std::memcpy(destination, source, bytes);
      return;
    }
    auto job = std::make_shared<StageJob>();
    job->destination = destination;
    job->source = source;
    job->bytes = bytes;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      job_ = job;
      ++generation_;
    }
    available_.notify_all();
    participate(*job);
    while (job->completed_bytes.load(std::memory_order_acquire) != bytes)
      std::this_thread::yield();
  }

private:
  struct StageJob {
    std::uint8_t *destination{};
    const std::uint8_t *source{};
    std::size_t bytes{};
    std::atomic<std::size_t> next_chunk{0U};
    std::atomic<std::size_t> completed_bytes{0U};
  };

  static void participate(StageJob &job) {
    while (true) {
      const auto chunk =
          job.next_chunk.fetch_add(1U, std::memory_order_relaxed);
      const auto offset = chunk * kStageChunkBytes;
      if (offset >= job.bytes)
        return;
      const auto count = std::min(kStageChunkBytes, job.bytes - offset);
      std::memcpy(job.destination + offset, job.source + offset, count);
      job.completed_bytes.fetch_add(count, std::memory_order_release);
    }
  }

  void work() {
    std::uint64_t seen = 0U;
    while (true) {
      std::shared_ptr<StageJob> job;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        available_.wait(lock,
                        [&] { return stopping_ || generation_ != seen; });
        if (stopping_)
          return;
        seen = generation_;
        job = job_;
      }
      if (job)
        participate(*job);
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable available_;
  std::shared_ptr<StageJob> job_;
  std::uint64_t generation_{};
  bool stopping_{};
};

class StreamedPrefetcher {
public:
  // resident_overrides: streamed constants the caller promoted to dedicated
  // resident storage (uploaded and fenced at preparation) because a fused
  // plan reads them at an earlier operation than their semantic consumer.
  // The prefetcher must not stage, wait on, or track them.
  StreamedPrefetcher(const ir::Program &program, const TensorMap &constants,
                     const compiler::MemoryPlan &plan, DeviceBuffers &buffers,
                     Context &context, std::uint32_t staging_buffers,
                     std::uint32_t stage_threads,
                     std::uint64_t pinned_budget_bytes,
                     std::unordered_set<std::uint32_t> resident_overrides,
                     std::unordered_set<std::uint32_t> lazy_resident_overrides,
                     bool direct_io, bool warm_page_cache,
                     bool describe_plan)
      : program_(program), constants_(constants), plan_(plan), buffers_(buffers),
        context_(context), staging_pool_(stage_threads),
        resident_overrides_(std::move(resident_overrides)),
        lazy_resident_overrides_(std::move(lazy_resident_overrides)),
        direct_io_(direct_io), warm_page_cache_(warm_page_cache) {
    if (warm_page_cache_) {
      const auto cgroup = probe_host_cgroup_memory();
      warm_page_cache_ = cgroup.limit_bytes == 0U ||
                         cgroup.limit_bytes >
                             cgroup.current_bytes + (4ULL << 30U);
    }
    std::uint64_t maximum = 0U;
    for (const auto &op : program_.operations) {
      std::uint64_t bytes = 0U;
      for (const auto id : op.inputs) {
        const auto *desc = program_.tensor(id);
        if (!desc || !desc->has_role(ir::TensorRole::Streamed) ||
            !buffers_.contains(id) || !active(id))
          continue;
        if (bytes > std::numeric_limits<std::uint64_t>::max() -
                        desc->byte_count())
          fail("streamed prefetch staging size overflow");
        bytes += desc->byte_count();
      }
      maximum = std::max(maximum, bytes);
    }
    if (maximum > std::numeric_limits<std::size_t>::max())
      fail("streamed prefetch staging size is not representable");
    // The historical two-buffer footprint is always admitted; growing the
    // ring beyond it must fit the pinned budget (fail-closed: this host
    // has a documented host-OOM incident, pinned staging stays bounded).
    if (staging_buffers < 2U)
      fail("streamed staging ring needs at least two buffers");
    if (staging_buffers > 2U && maximum != 0U &&
        static_cast<std::uint64_t>(staging_buffers) >
            pinned_budget_bytes / maximum)
      fail("streamed staging ring exceeds the pinned budget: buffers=" +
           std::to_string(staging_buffers) + " buffer_bytes=" +
           std::to_string(maximum) + " budget_bytes=" +
           std::to_string(pinned_budget_bytes));
    staging_.resize(staging_buffers);
    copy_done_.resize(staging_buffers);
    for (auto &staging : staging_)
      staging = std::make_unique<PinnedHostWorkspace>(
          static_cast<std::size_t>(maximum));
    for (auto &event : copy_done_)
      event = std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
    copy_recorded_.assign(staging_buffers, false);
    ready_events_.reserve(program_.operations.size());
    completion_events_.reserve(program_.operations.size());
    for (std::size_t index = 0; index < program_.operations.size(); ++index) {
      ready_events_.push_back(
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING));
      completion_events_.push_back(
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING));
      for (const auto id : program_.operations[index].inputs) {
        const auto *desc = program_.tensor(id);
        if (desc && desc->has_role(ir::TensorRole::Streamed) &&
            buffers_.contains(id) && active(id))
          first_consumer_.try_emplace(id, index);
      }
    }
    completion_recorded_.resize(program_.operations.size(), false);
    for (const auto &[id, first] : first_consumer_) {
      if (lazy_resident_overrides_.contains(id)) {
        overwrite_wait_operation_.emplace(
            id, std::numeric_limits<std::size_t>::max());
        continue;
      }
      const auto *target = plan_.assignment(id);
      if (!target)
        fail("streamed tensor lacks a memory-plan assignment: " +
             std::to_string(id));
      auto wait = std::numeric_limits<std::size_t>::max();
      for (const auto &candidate : plan_.assignments) {
        if (candidate.tensor_id == id || candidate.slot_id != target->slot_id ||
            candidate.last_operation >= first)
          continue;
        wait = wait == std::numeric_limits<std::size_t>::max()
                   ? static_cast<std::size_t>(candidate.last_operation)
                   : std::max(wait,
                              static_cast<std::size_t>(candidate.last_operation));
      }
      overwrite_wait_operation_.emplace(id, wait);
    }
    if (describe_plan) {
      auto total = std::uint64_t{0U};
      for (const auto &[id, first] : first_consumer_) {
        (void)first;
        total += constants_.at(id).byte_size();
      }
      std::cerr << "STREAMED_PREFETCH_PLAN tensors=" << first_consumer_.size()
                << " bytes_per_iteration=" << total << '\n';
      for (const auto &[id, first] : first_consumer_) {
        const auto bytes = constants_.at(id).byte_size();
        if (bytes >= 64ULL * 1024ULL * 1024ULL)
          std::cerr << "STREAMED_PREFETCH_TENSOR id=" << id
                    << " bytes=" << bytes << " first_operation=" << first
                    << '\n';
      }
    }
  }

  void begin_profile(
      std::uint32_t iterations,
      const std::unordered_set<std::uint32_t> &repeated_invariant_operations,
      std::uint32_t repeated_invariant_executions) {
    profiling_ = true;
    streamed_bytes_ = 0U;
    host_stage_milliseconds_ = 0.0;
    direct_read_bytes_ = 0U;
    host_wait_milliseconds_ = 0.0;
    profiled_copy_counts_.clear();
    next_copy_timing_ = 0U;
    copy_timings_.clear();
    std::size_t count = 0U;
    for (const auto &[id, first] : first_consumer_) {
      if (!active(id))
        continue;
      if (lazy_resident_overrides_.contains(id)) {
        count += 1U;
        continue;
      }
      const auto invariant = repeated_invariant_operations.contains(
          program_.operations.at(first).id);
      count += invariant ? repeated_invariant_executions
                         : static_cast<std::size_t>(iterations);
    }
    copy_timings_.reserve(count);
    for (std::size_t index = 0U; index < count; ++index)
      copy_timings_.push_back({std::make_unique<Event>(),
                               std::make_unique<Event>()});
  }

  void finish_profile(PipelineProfile &profile) {
    if (!profiling_)
      return;
    if (next_copy_timing_ != copy_timings_.size())
      fail("streamed pipeline profile did not observe every timed weight copy");
    double h2d_milliseconds = 0.0;
    for (const auto &timing : copy_timings_) {
      float milliseconds = 0.0F;
      check(cuEventElapsedTime(&milliseconds, timing.start->get(),
                               timing.stop->get()),
            "cuEventElapsedTime streamed H2D");
      h2d_milliseconds += milliseconds;
    }
    profile.streamed_weight_bytes = streamed_bytes_;
    profile.streamed_host_stage_milliseconds = host_stage_milliseconds_;
    profile.streamed_direct_read_bytes = direct_read_bytes_;
    profile.streamed_host_wait_milliseconds = host_wait_milliseconds_;
    profile.streamed_h2d_milliseconds = h2d_milliseconds;
    std::vector<std::pair<std::uint32_t, std::size_t>> copied;
    copied.reserve(profiled_copy_counts_.size());
    for (const auto &[id, copies] : profiled_copy_counts_)
      copied.emplace_back(id, copies);
    std::sort(copied.begin(), copied.end());
    auto copied_bytes = std::uint64_t{0U};
    for (const auto &[id, copies] : copied) {
      (void)copies;
      copied_bytes += constants_.at(id).byte_size();
    }
    if (!copied.empty()) {
      std::cerr << "STREAMED_COPY_PROFILE tensors=" << copied.size()
                << " bytes=" << copied_bytes
                << '\n';
    }
    profiling_ = false;
  }

  bool prefetch(std::size_t operation_index, bool force = false) {
    const auto parity = operation_index % staging_.size();
    const auto &op = program_.operations.at(operation_index);
    bool has_streamed = false;
    for (const auto id : op.inputs) {
      const auto *desc = program_.tensor(id);
      has_streamed = has_streamed ||
                     (desc && desc->has_role(ir::TensorRole::Streamed) &&
                      buffers_.contains(id) &&
                      active(id) &&
                      (force || first_consumer_.at(id) == operation_index));
    }
    if (!has_streamed)
      return false;
    const auto host_wait_start = std::chrono::steady_clock::now();
    if (copy_recorded_[parity]) {
      TraceLabelScope reuse_label("streamed-staging-reuse");
      check(counted_event_synchronize(copy_done_[parity]->get()),
            "cuEventSynchronize streamed staging reuse");
    }
    if (profiling_)
      host_wait_milliseconds_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - host_wait_start)
              .count();

    for (const auto id : op.inputs) {
      const auto *desc = program_.tensor(id);
      if (!desc || !desc->has_role(ir::TensorRole::Streamed) ||
          !buffers_.contains(id) || !active(id))
        continue;
      if (!force && first_consumer_.at(id) != operation_index)
        continue;
      const auto wait = overwrite_wait_operation_.at(id);
      if (wait != std::numeric_limits<std::size_t>::max() &&
          completion_recorded_.at(wait))
        check(counted_stream_wait_event(
                  context_.copy_stream(), completion_events_.at(wait)->get(), 0U),
              "cuStreamWaitEvent streamed slot release");
    }

    std::size_t offset = 0U;
    for (const auto id : op.inputs) {
      const auto *desc = program_.tensor(id);
      if (!desc || !desc->has_role(ir::TensorRole::Streamed) ||
          !buffers_.contains(id) || !active(id))
        continue;
      if (!force && first_consumer_.at(id) != operation_index)
        continue;
      const auto &tensor = constants_.at(id);
      if (tensor.byte_size() > staging_[parity]->size() - offset)
        fail("streamed tensor exceeds prefetch staging capacity");
      auto *destination = static_cast<std::uint8_t *>(staging_[parity]->data()) +
                          offset;
      const auto host_stage_start = std::chrono::steady_clock::now();
      const auto trace_stage_start =
          active_tracer ? active_tracer->now_ms() : 0.0;
      if (direct_io_ && tensor.mapped_resident_fraction() < 0.9 &&
          tensor.read_direct_into(destination)) {
        direct_read_bytes_ += tensor.byte_size();
        if (warm_page_cache_)
          tensor.prefetch_mapped_pages();
      } else {
        staging_pool_.copy(destination, tensor.data(), tensor.byte_size());
      }
      if (active_tracer)
        active_tracer->record(telemetry::category::staging,
                              "streamed-constant:host-stage",
                              tensor.byte_size(), "host", trace_stage_start,
                              active_tracer->now_ms());
      TraceLabelScope streamed_label("streamed-constant");
      if (profiling_) {
        host_stage_milliseconds_ +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_stage_start)
                .count();
        streamed_bytes_ += tensor.byte_size();
        ++profiled_copy_counts_[id];
        if (next_copy_timing_ >= copy_timings_.size())
          fail("streamed pipeline profile observed an unexpected weight copy");
        auto &timing = copy_timings_.at(next_copy_timing_++);
        check(counted_event_record(timing.start->get(), context_.copy_stream()),
              "cuEventRecord streamed H2D start");
        check(counted_memcpy_htod(buffers_.at(id), destination,
                                tensor.byte_size(), context_.copy_stream()),
              "cuMemcpyHtoDAsync profiled constant");
        check(counted_event_record(timing.stop->get(), context_.copy_stream()),
              "cuEventRecord streamed H2D stop");
      } else {
        check(counted_memcpy_htod(buffers_.at(id), destination,
                                tensor.byte_size(), context_.copy_stream()),
              "cuMemcpyHtoDAsync prefetched constant");
      }
      offset += tensor.byte_size();
      if (release_mapped_pages_per_copy_)
        tensor.discard_mapped_pages();
    }
    check(counted_event_record(ready_events_.at(operation_index)->get(),
                        context_.copy_stream()),
          "cuEventRecord streamed readiness");
    check(counted_event_record(copy_done_[parity]->get(), context_.copy_stream()),
          "cuEventRecord streamed staging copy");
    copy_recorded_[parity] = true;
    return true;
  }

  void set_release_mapped_pages_per_copy(bool release_per_copy) {
    release_mapped_pages_per_copy_ = release_per_copy;
  }

  // Run-end page drop for the release-at-end policy: return every mapped
  // streamed constant's pages to the kernel once, instead of after each
  // staging copy inside the run.
  void release_mapped_pages() const {
    for (const auto &[id, first] : first_consumer_) {
      (void)first;
      constants_.at(id).discard_mapped_pages();
    }
  }

  void wait(std::size_t operation_index, bool ready,
            CUstream stream = nullptr) {
    if (!ready)
      return;
    check(counted_stream_wait_event(stream ? stream : context_.stream(),
                            ready_events_.at(operation_index)->get(), 0U),
          "cuStreamWaitEvent prefetched constant");
  }

  void complete(std::size_t operation_index, CUstream stream = nullptr) {
    check(counted_event_record(completion_events_.at(operation_index)->get(),
                        stream ? stream : context_.stream()),
          "cuEventRecord operation completion");
    completion_recorded_.at(operation_index) = true;
  }

  void complete_iteration() {
    loaded_lazy_residents_.insert(lazy_resident_overrides_.begin(),
                                  lazy_resident_overrides_.end());
  }

private:
  bool active(std::uint32_t id) const {
    if (!resident_overrides_.contains(id))
      return true;
    return lazy_resident_overrides_.contains(id) &&
           !loaded_lazy_residents_.contains(id);
  }

  struct CopyTiming {
    std::unique_ptr<Event> start;
    std::unique_ptr<Event> stop;
  };

  const ir::Program &program_;
  const TensorMap &constants_;
  const compiler::MemoryPlan &plan_;
  DeviceBuffers &buffers_;
  Context &context_;
  StagingPool staging_pool_;
  std::unordered_set<std::uint32_t> resident_overrides_;
  std::unordered_set<std::uint32_t> lazy_resident_overrides_;
  bool direct_io_{true};
  bool warm_page_cache_{true};
  std::uint64_t direct_read_bytes_{};
  std::unordered_set<std::uint32_t> loaded_lazy_residents_;
  std::vector<std::unique_ptr<PinnedHostWorkspace>> staging_;
  std::vector<std::unique_ptr<Event>> copy_done_;
  std::vector<bool> copy_recorded_;
  std::vector<std::unique_ptr<Event>> ready_events_;
  std::vector<std::unique_ptr<Event>> completion_events_;
  std::vector<bool> completion_recorded_;
  std::unordered_map<std::uint32_t, std::size_t> first_consumer_;
  std::unordered_map<std::uint32_t, std::size_t> overwrite_wait_operation_;
  bool release_mapped_pages_per_copy_{true};
  bool profiling_{};
  std::uint64_t streamed_bytes_{};
  double host_stage_milliseconds_{};
  double host_wait_milliseconds_{};
  std::size_t next_copy_timing_{};
  std::vector<CopyTiming> copy_timings_;
  std::unordered_map<std::uint32_t, std::size_t> profiled_copy_counts_;
};

void upload_resident_constants(const ir::Program &program,
                               const TensorMap &inputs,
                               DeviceBuffers &buffers, CUstream stream) {
  for (const auto &desc : program.tensors) {
    if (!desc.has_role(ir::TensorRole::Constant))
      continue;
    if (desc.has_role(ir::TensorRole::Streamed))
      continue;
    if (!buffers.contains(desc.id))
      continue;
    const auto &tensor = inputs.at(desc.id);
    check(counted_memcpy_htod(buffers.at(desc.id), tensor.data(),
                            tensor.byte_size(), stream),
          "cuMemcpyHtoDAsync");
  }
}

void upload_dynamic_inputs(const ir::Program &program, const TensorMap &inputs,
                           DeviceBuffers &buffers, CUstream stream) {
  for (const auto &desc : program.tensors) {
    if (!desc.has_role(ir::TensorRole::Input))
      continue;
    const auto &tensor = inputs.at(desc.id);
    check(counted_memcpy_htod(buffers.at(desc.id), tensor.data(),
                            tensor.byte_size(), stream),
          "cuMemcpyHtoDAsync dynamic input");
  }
}

void launch(const ir::Program &program, const ir::Operation &op, CUfunction function,
            DeviceBuffers &buffers, CUstream stream,
            const std::vector<std::uint32_t> *input_override = nullptr) {
  if (op.opcode == ir::Opcode::Barrier) {
    check(counted_stream_synchronize(stream), "cuStreamSynchronize barrier");
    return;
  }
  std::array<CUdeviceptr, 16> pointers{};
  std::array<void *, 16> arguments{};
  std::size_t argument_count = 0;
  const auto &inputs = input_override ? *input_override : op.inputs;
  for (const auto input : inputs) {
    pointers[argument_count] = buffers.at(input);
    arguments[argument_count] = &pointers[argument_count];
    ++argument_count;
  }
  for (const auto output : op.outputs) {
    pointers[argument_count] = buffers.at(output);
    arguments[argument_count] = &pointers[argument_count];
    ++argument_count;
  }

  unsigned block = static_cast<unsigned>(op.u64(ir::AttrKey::BlockSize, 256));
  unsigned grid = 1;
  unsigned shared = 0;
  if (input_override && op.opcode == ir::Opcode::Linear) {
    block = 256U;
    grid = static_cast<unsigned>(program.tensor(op.inputs[1])->dims[0]);
    const auto *input = program.tensor(op.inputs[0]);
    const auto *weight = program.tensor(op.inputs[1]);
    const auto flattened_rows =
        input->element_count() / weight->dims[1];
    shared = static_cast<unsigned>(8U * flattened_rows * sizeof(float));
  } else if (op.opcode == ir::Opcode::RmsNormModulate ||
             op.opcode == ir::Opcode::RmsNorm ||
             op.opcode == ir::Opcode::LayerNorm ||
             op.opcode == ir::Opcode::LayerNormModulate) {
    const auto *input = program.tensor(op.inputs[0]);
    grid = static_cast<unsigned>(input->element_count() / input->dims.back());
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::QuantizeInt8Rows) {
    const auto *quantized = program.tensor(op.outputs[0]);
    grid = static_cast<unsigned>(quantized->element_count() /
                                 quantized->dims.back());
    block = 256U;
    const auto implementation = static_cast<ir::Int8RowQuantization>(op.u64(
        ir::AttrKey::Implementation,
        static_cast<std::uint64_t>(ir::Int8RowQuantization::Direct)));
    if (ir::is_convrot_int8_row_quantization(implementation))
      shared = static_cast<unsigned>(quantized->dims.back() * sizeof(float));
  } else if (op.opcode == ir::Opcode::QuantizeFp8Rows) {
    const auto *input = program.tensor(op.inputs[0]);
    grid = static_cast<unsigned>(input->element_count() / input->dims.back());
    block = 256U;
  } else if (op.opcode == ir::Opcode::QuantizeFp8Blocks32) {
    const auto *input = program.tensor(op.inputs[0]);
    grid = static_cast<unsigned>(input->element_count() / input->dims.back());
    block = 256U;
  } else if (op.opcode == ir::Opcode::QkNormPartialRope) {
    const auto &dims = program.tensor(op.inputs[0])->dims;
    const auto *table = program.tensor(op.inputs[2]);
    if (program.tensor(op.inputs[0])->dtype == ir::DType::BF16 &&
        dims.back() == 128U &&
        (table->dims.back() ==
             op.u64(ir::AttrKey::RotaryDim, dims.back()) ||
         table->dims.back() * 2U ==
             op.u64(ir::AttrKey::RotaryDim, dims.back())))
      block = 128U;
    grid = static_cast<unsigned>(program.tensor(op.inputs[0])->element_count() /
                                 dims.back());
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::ChannelRmsNorm) {
    const auto *input = program.tensor(op.inputs[0]);
    const auto axis = op.u64(ir::AttrKey::Axis, 1U);
    grid = static_cast<unsigned>(input->element_count() / input->dims[axis]);
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::GroupNorm) {
    const auto *input = program.tensor(op.inputs[0]);
    grid = static_cast<unsigned>(input->dims[0] *
                                 op.u64(ir::AttrKey::Groups, 1U));
    shared = 2U * block * sizeof(float);
  } else if (op.opcode == ir::Opcode::Attention) {
    const auto &dims = program.tensor(op.inputs[0])->dims;
    const auto batched = dims.size() == 4U;
    const auto batch = batched ? dims[0] : 1U;
    const auto sequence = dims[batched ? 1U : 0U];
    const auto heads = dims[batched ? 2U : 1U];
    block = std::min<unsigned>(block, 256U);
    grid = static_cast<unsigned>(batch * sequence * heads);
    shared = static_cast<unsigned>((block + sequence) * sizeof(float));
  } else {
    const auto count = program.tensor(op.outputs[0])->element_count();
    grid = static_cast<unsigned>((count + block - 1U) / block);
  }

  check(counted_launch_kernel(function, grid, 1, 1, block, 1, 1, shared, stream,
                       arguments.data(), nullptr),
        "cuLaunchKernel");
}

struct RepeatedInvariantPlan {
  std::unordered_set<std::uint32_t> operations;
  std::unordered_set<std::uint32_t> persistent_tensors;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      outputs_by_producer;
  std::vector<std::uint32_t> input_tensors;
  std::uint64_t persistent_bytes{};
};

RepeatedInvariantPlan find_repeated_invariant_plan(
    const ir::Program &program,
    const std::vector<std::uint32_t> &requested_operations) {
  RepeatedInvariantPlan result;
  if (requested_operations.empty())
    return result;

  std::unordered_map<std::uint32_t, const ir::Operation *> operations;
  std::unordered_map<std::uint32_t, const ir::Operation *> producers;
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>>
      consumers;
  for (const auto &operation : program.operations) {
    if (!operations.emplace(operation.id, &operation).second)
      fail("repeated-invariant plan found a duplicate operation id");
    for (const auto output : operation.outputs)
      producers.emplace(output, &operation);
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }
  for (const auto operation_id : requested_operations) {
    if (!operations.contains(operation_id))
      fail("repeated-invariant plan references missing operation " +
           std::to_string(operation_id));
    if (!result.operations.insert(operation_id).second)
      fail("repeated-invariant plan contains duplicate operation " +
           std::to_string(operation_id));
  }

  std::unordered_set<std::uint32_t> external_inputs;
  for (const auto operation_id : result.operations) {
    const auto &operation = *operations.at(operation_id);
    if (operation.opcode == ir::Opcode::Barrier)
      fail("repeated-invariant plan may contain only pure tensor operations");
    for (const auto input : operation.inputs) {
      const auto produced = producers.find(input);
      if (produced != producers.end()) {
        if (!result.operations.contains(produced->second->id))
          fail("repeated-invariant plan is not dependency-closed at tensor " +
               std::to_string(input));
        continue;
      }
      const auto *description = program.tensor(input);
      if (!description ||
          (!description->has_role(ir::TensorRole::Input) &&
           !description->has_role(ir::TensorRole::Constant)))
        fail("repeated-invariant plan depends on an unbound tensor " +
             std::to_string(input));
      if (description->has_role(ir::TensorRole::Input))
        external_inputs.insert(input);
      if (description->has_role(ir::TensorRole::Streamed)) {
        for (const auto *consumer : consumers[input])
          if (!result.operations.contains(consumer->id))
            fail("repeated-invariant streamed tensor has a consumer outside "
                 "the cached region: " +
                 std::to_string(input));
      }
    }
    for (const auto output : operation.outputs) {
      const auto *description = program.tensor(output);
      if (!description)
        fail("repeated-invariant operation has a missing output tensor");
      const auto crossing =
          description->has_role(ir::TensorRole::Output) ||
          std::any_of(consumers[output].begin(), consumers[output].end(),
                      [&](const auto *consumer) {
                        return !result.operations.contains(consumer->id);
                      });
      if (!crossing)
        continue;
      if (description->roles ==
          static_cast<std::uint32_t>(ir::TensorRole::Internal)) {
        result.persistent_tensors.insert(output);
        result.outputs_by_producer[operation.id].push_back(output);
        result.persistent_bytes += description->byte_count();
      }
    }
  }
  if (result.persistent_tensors.empty())
    fail("repeated-invariant plan has no internal output crossing its boundary");
  result.input_tensors.assign(external_inputs.begin(), external_inputs.end());
  std::sort(result.input_tensors.begin(), result.input_tensors.end());
  return result;
}

class CudaPreparedExecution final : public PreparedExecution {
public:
  CudaPreparedExecution(ir::Program program, const TensorMap &bindings,
                        const RunOptions &options,
                        std::shared_ptr<Context> context)
      : program_(std::move(program)), context_owner_(std::move(context)),
        context_(*context_owner_) {
    const auto preparation_start = std::chrono::steady_clock::now();
    TelemetryScope telemetry_scope(preparation_telemetry_);
    std::optional<TracerScope> tracer_scope;
    if (telemetry::trace_events_requested(options)) {
      preparation_tracer_.origin = preparation_start;
      tracer_scope.emplace(preparation_tracer_);
    }
    nvtx_enabled = telemetry::nvtx_ranges_requested(options);
    TraceLabelScope preparation_label("prepare");
    nvtx_push("dif::prepare");
    if (options.lazy_resident_upload && options.pipelined_resident_upload)
      fail("lazy and pipelined resident upload are mutually exclusive");
    if (options.cudnn_attention_heuristic > 4U)
      fail("cuDNN attention heuristic must be 0 (A), 1 (B), 2 (FALLBACK), 3 (autotune), or 4 (deterministic)");
    cudnn_attention_heuristic_ = options.cudnn_attention_heuristic;
    lazy_resident_upload_ = options.lazy_resident_upload;
    ir::verify(program_);
    // Grad-flow gate (Flame lesson, three silent zero-LoRA-B incidents):
    // fused inference plans replace semantic operations with kernels that
    // declare no backward. A differentiated program must not be executed
    // through them, or the training graph silently loses gradient flow.
    {
      const char *training_opcode = nullptr;
      for (const auto &operation : program_.operations) {
        switch (operation.opcode) {
        case ir::Opcode::MseLossBackward:
        case ir::Opcode::LinearBackwardInput:
        case ir::Opcode::LinearBackwardWeight:
        case ir::Opcode::BiasBackward:
        case ir::Opcode::SiLUBackward:
        case ir::Opcode::AdamWUpdate:
        case ir::Opcode::RmsNormBackward:
        case ir::Opcode::RmsNormModulateBackward:
        case ir::Opcode::SwiGluBackward:
        case ir::Opcode::ResidualGateBackward:
        case ir::Opcode::LayerNormBackward:
        case ir::Opcode::QkNormPartialRopeBackward:
        case ir::Opcode::AttentionLse:
        case ir::Opcode::AttentionBackward:
          training_opcode = ir::opcode_name(operation.opcode).data();
          break;
        default:
          break;
        }
        if (training_opcode)
          break;
      }
      if (training_opcode) {
        std::string fused;
        const auto note = [&](bool set, const char *name) {
          if (set)
            fused += (fused.empty() ? "" : ", ") + std::string(name);
        };
        // Linear->SwiGlu fusion eliminates the fc1 output that
        // SwiGluBackward consumes: illegal. Absorbing a BiasAdd into the
        // cuBLASLt epilogue eliminates only the unbiased intermediate, which
        // no backward opcode reads (BiasBackward and the Linear backwards
        // take grad_output, x and W), and dif_epilogue_tests gates its
        // byte-identity on the MLP training graph: allowed.
        note(!options.fuse_linear_swiglu_operations.empty(),
             "fuse_linear_swiglu_operations");
        note(!options.convrot_int8_checkpoint.empty(),
             "convrot_int8_checkpoint");
        note(!options.h3_w8a8_cache.empty(), "h3_w8a8_cache");
        note(!options.h3_convrot_int8_checkpoint.empty(),
             "h3_convrot_int8_checkpoint");
        note(!options.h3_groupwise_cache.empty(), "h3_groupwise_cache");
        note(!options.h3_modulation_cache.empty(), "h3_modulation_cache");
        note(!options.h3_ck_attention_dso.empty(), "h3_ck_attention_dso");
        note(options.h3_owned_attention, "h3_owned_attention");
        note(options.h3_owned_attention_center_k,
             "h3_owned_attention_center_k");
        if (!fused.empty())
          fail(std::string("training program (contains ") + training_opcode +
               ") requested fused inference plans (" + fused +
               ") that declare no backward; gradient flow through a fused "
               "plan is undefined, run the differentiated program without "
               "fusion options");
      }
    }
    auto repeated_invariant = find_repeated_invariant_plan(
        program_, options.repeated_invariant_operations);
    repeated_invariant_operations_ = std::move(repeated_invariant.operations);
    repeated_invariant_persistent_tensors_ =
        std::move(repeated_invariant.persistent_tensors);
    repeated_invariant_outputs_by_producer_ =
        std::move(repeated_invariant.outputs_by_producer);
    repeated_invariant_input_tensors_ =
        std::move(repeated_invariant.input_tensors);
    repeated_invariant_persistent_bytes_ =
        repeated_invariant.persistent_bytes;
    capture_intermediate_tensors_ = options.capture_intermediate_tensors;
    std::sort(capture_intermediate_tensors_.begin(),
              capture_intermediate_tensors_.end());
    if (std::adjacent_find(capture_intermediate_tensors_.begin(),
                           capture_intermediate_tensors_.end()) !=
        capture_intermediate_tensors_.end())
      fail("intermediate capture plan contains a duplicate tensor id");
    for (const auto tensor_id : capture_intermediate_tensors_) {
      const auto *description = program_.tensor(tensor_id);
      if (!description ||
          description->has_role(ir::TensorRole::Input) ||
          description->has_role(ir::TensorRole::Constant))
        fail("intermediate capture requires a produced tensor " +
             std::to_string(tensor_id));
      const auto producer = std::find_if(
          program_.operations.begin(), program_.operations.end(),
          [&](const auto &operation) {
            return std::find(operation.outputs.begin(), operation.outputs.end(),
                             tensor_id) != operation.outputs.end();
          });
      if (producer == program_.operations.end())
        fail("intermediate capture tensor has no producer " +
             std::to_string(tensor_id));
      captured_tensors_by_producer_[producer->id].push_back(tensor_id);
    }
    std::vector<std::uint32_t> repeated_persistent_ids(
        repeated_invariant_persistent_tensors_.begin(),
        repeated_invariant_persistent_tensors_.end());
    std::sort(repeated_persistent_ids.begin(), repeated_persistent_ids.end());
    for (const auto tensor_id : repeated_persistent_ids) {
      repeated_invariant_cache_offsets_.emplace(
          tensor_id, repeated_invariant_cache_bytes_);
      repeated_invariant_cache_bytes_ +=
          align_256(program_.tensor(tensor_id)->byte_count());
    }
    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Constant))
        continue;
      const auto found = bindings.find(desc.id);
      if (found == bindings.end())
        fail("missing CUDA constant tensor " + std::to_string(desc.id));
      found->second.validate();
      if (found->second.dtype != desc.dtype || found->second.dims != desc.dims)
        fail("CUDA constant shape/dtype mismatch for id " +
             std::to_string(desc.id));
      constants_.emplace(desc.id, found->second);
    }
    target_profile_ = probe_target(ProbeBackend::Cuda, context_.ordinal());
    BudgetRequest budget_request;
    budget_request.reserved_device_memory_bytes = options.minimum_free_bytes;
    budget_request.pinned_host_memory_budget_bytes =
        options.streamed_pinned_budget_bytes;
    budget_request.staging_budget_bytes = options.streamed_pinned_budget_bytes;
    runtime_budget_ = probe_runtime_budget(target_profile_, budget_request);
    const auto major = static_cast<int>(target_profile_.compute_major);
    const auto minor = static_cast<int>(target_profile_.compute_minor);
    device_name_ = target_profile_.product_name;
    free_bytes_before_ = runtime_budget_.free_device_memory_bytes;

    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::Attention ||
          operation.u64(ir::AttrKey::Implementation, 1U) != 3U)
        continue;
      const auto *query = program_.tensor(operation.inputs.at(0));
      if (!query || query->dims.size() != 3U || query->dims.at(1) != 1U)
        fail("materialized F32 attention found an invalid verified query");
      if (query->dims.at(0) >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()) ||
          query->dims.at(2) >
              static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
        fail("materialized F32 attention shape exceeds cuBLAS integer limits");
      MaterializedF32AttentionPlan plan{
          operation.id,
          static_cast<int>(query->dims.at(0)),
          static_cast<int>(query->dims.at(2)),
          static_cast<float>(operation.f64(
              ir::AttrKey::AttentionScale,
              1.0 / std::sqrt(static_cast<double>(query->dims.at(2))))),
      };
      if (!(plan.scale > 0.0F))
        fail("materialized F32 attention scale must be positive");
      materialized_f32_attention_score_bytes_ = std::max(
          materialized_f32_attention_score_bytes_, plan.score_bytes());
      materialized_f32_attention_plans_.emplace(operation.id, plan);
    }
    flash_attention_workspace_bytes_ = 0U;
    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::Attention ||
          operation.u64(ir::AttrKey::Implementation, 1U) != 4U)
        continue;
#if DIF_HAS_FLASH_ATTENTION
      const auto *query = program_.tensor(operation.inputs.at(0));
      if (!query ||
          (query->dims.size() != 3U && query->dims.size() != 4U))
        fail("native FlashAttention found an invalid verified query");
      const auto batched = query->dims.size() == 4U;
      const auto batch = batched ? query->dims.at(0) : 1U;
      const auto sequence = query->dims.at(batched ? 1U : 0U);
      const auto heads = query->dims.at(batched ? 2U : 1U);
      if (batch > std::numeric_limits<std::uint64_t>::max() / heads ||
          batch * heads > std::numeric_limits<std::uint64_t>::max() /
                              sequence ||
          batch * heads * sequence >
              std::numeric_limits<std::uint64_t>::max() / sizeof(float))
        fail("native FlashAttention workspace size overflow");
      flash_attention_workspace_bytes_ = std::max(
          flash_attention_workspace_bytes_,
          batch * heads * sequence * sizeof(float));
#else
      fail("DiffIR requests native FlashAttention but this CUDA backend was "
           "built without it");
#endif
    }

    fused_linear_swiglu_plans_ =
        find_fused_linear_swiglu_plans(program_, options, major);
    absorbed_linear_bias_plans_ =
        find_absorbed_linear_bias_plans(program_, options);
    h3_modulation_cache_plans_ =
        find_h3_modulation_cache_plans(program_, options);
    if (!h3_modulation_cache_plans_.empty()) {
      h3_modulation_slices_ = h3_modulation_cache_plans_.front().slices;
      if (!options.h3_modulation_input.empty()) {
        h3_modulation_expected_input_ =
            read_tensor(options.h3_modulation_input);
        h3_modulation_input_path_ = options.h3_modulation_input;
        const auto input_id = h3_modulation_cache_plans_.front().input_tensor;
        const auto *description = program_.tensor(input_id);
        const auto found = bindings.find(input_id);
        if (!description || found == bindings.end() ||
            h3_modulation_expected_input_.dtype != description->dtype ||
            h3_modulation_expected_input_.dims != description->dims ||
            found->second.byte_size() !=
                h3_modulation_expected_input_.byte_size() ||
            std::memcmp(found->second.data(),
                        h3_modulation_expected_input_.data(),
                        found->second.byte_size()) != 0)
          fail("H3 modulation cache input does not match its source capture");
      }
    }
    h3_groupwise_plans_ = find_h3_groupwise_plans(program_, options);
    h3_w8a8_mlp_plans_ = find_h3_w8a8_mlp_plans(program_, options);
    h3_w8a8_attention_plans_ =
        find_h3_w8a8_attention_plans(program_, options);
    convrot_int8_linear_plans_ =
        find_convrot_int8_linear_plans(program_, options);
    if (options.convrot_int8_resident &&
        convrot_int8_linear_plans_.empty())
      fail("resident generic ConvRot requires a ConvRot checkpoint");
    if (options.convrot_int8_resident &&
        options.convrot_int8_weight_only_quality)
      fail("resident generic ConvRot does not support weight-only quality mode");
    if (options.convrot_int8_weight_only_quality &&
        std::any_of(convrot_int8_linear_plans_.begin(),
                    convrot_int8_linear_plans_.end(), [](const auto &plan) {
                      return plan.dtype != ir::DType::BF16;
                    }))
      fail("generic ConvRot weight-only quality mode supports BF16 Linears only");
    if (!convrot_int8_linear_plans_.empty() &&
        (!h3_w8a8_mlp_plans_.empty() ||
         !h3_w8a8_attention_plans_.empty()))
      fail("generic and H3-chain ConvRot policies are mutually exclusive");
    h3_compact_adaln_plans_ = find_h3_compact_adaln_plans(
        program_, options, h3_w8a8_attention_plans_, h3_w8a8_mlp_plans_);
    if ((!options.h3_w8a8_cache.empty() ||
         !options.h3_convrot_int8_checkpoint.empty()) &&
        h3_w8a8_mlp_plans_.empty() && h3_w8a8_attention_plans_.empty())
      fail("H3 direct INT8 checkpoint requested but no supported H3 chain was found");
    if (options.h3_int8_cutlass_scaled_fc1) {
#if DIF_HAS_CUTLASS
      if (h3_w8a8_mlp_plans_.empty() ||
          !std::all_of(h3_w8a8_mlp_plans_.begin(),
                       h3_w8a8_mlp_plans_.end(), [](const auto &plan) {
                         return plan.convrot && plan.cutlass_scaled_fc1;
                       }))
        fail("H3 CUTLASS scaled FC1 requires a ConvRot INT8 MLP plan");
#else
      fail("H3 CUTLASS scaled FC1 requested without CUTLASS support");
#endif
    }
    if (options.h3_int8_convrot_scale_chunk != 0U) {
      if (options.h3_convrot_int8_checkpoint.empty())
        fail("H3 ConvRot scale-chunk policy requires a ConvRot checkpoint");
      if (options.h3_int8_convrot_scale_chunk < 256U ||
          options.h3_int8_convrot_scale_chunk > 2048U ||
          options.h3_int8_convrot_scale_chunk % 256U != 0U)
        fail("H3 ConvRot scale chunk must be a multiple of 256 in [256,2048]");
      if (options.h3_int8_cutlass_scaled_fc1 ||
          options.h3_int8_cutlass_scaled_all || options.h3_int8_cublaslt)
        fail("H3 ConvRot scale-chunk and scaled/CuBLASLt projection policies are mutually exclusive");
    }
    if (options.h3_int8_convrot_global_activation_scale &&
        options.h3_int8_convrot_scale_chunk == 0U)
      fail("H3 ConvRot global activation scale requires grouped weight scale chunks");
    if (options.h3_int8_cutlass_scaled_all) {
#if DIF_HAS_CUTLASS
      const auto complete_mlp =
          h3_w8a8_mlp_plans_.empty() ||
          std::all_of(h3_w8a8_mlp_plans_.begin(),
                      h3_w8a8_mlp_plans_.end(), [](const auto &plan) {
                        return plan.convrot && plan.cutlass_scaled_fc1 &&
                               plan.cutlass_scaled_fc2 &&
                               plan.compact_adaln.enabled;
                      });
      const auto complete_attention =
          h3_w8a8_attention_plans_.empty() ||
          std::all_of(h3_w8a8_attention_plans_.begin(),
                      h3_w8a8_attention_plans_.end(), [](const auto &plan) {
                        return plan.convrot && plan.cutlass_scaled &&
                               plan.compact_adaln.enabled;
                      });
      if ((!complete_mlp || !complete_attention) ||
          (h3_w8a8_mlp_plans_.empty() &&
           h3_w8a8_attention_plans_.empty()))
        fail("H3 CUTLASS scaled-all requires every selected compact-AdaLN "
             "ConvRot INT8 attention/MLP plan to be complete");
#else
      fail("H3 CUTLASS scaled-all requested without CUTLASS support");
#endif
    }
    if (!options.h3_convrot_bf16_correction_rows.empty()) {
      if (h3_w8a8_mlp_plans_.empty() &&
          h3_w8a8_attention_plans_.empty())
        fail("H3 ConvRot BF16 row correction requires an admitted projection chain");
      auto correction = std::make_unique<H3ConvRotBf16Correction>();
      correction->rows = options.h3_convrot_bf16_correction_rows;
      if (!std::is_sorted(correction->rows.begin(), correction->rows.end()) ||
          std::adjacent_find(correction->rows.begin(),
                             correction->rows.end()) != correction->rows.end())
        fail("H3 ConvRot BF16 correction rows must be sorted and unique");
      auto sequence = std::uint64_t{0U};
      for (const auto &plan : h3_w8a8_mlp_plans_)
        sequence = std::max(sequence, plan.rows);
      for (const auto &plan : h3_w8a8_attention_plans_)
        sequence = std::max(sequence, plan.rows);
      if (correction->rows.empty() || correction->rows.back() >= sequence)
        fail("H3 ConvRot BF16 correction row exceeds the sequence");
      const auto correction_rows =
          static_cast<std::uint64_t>(correction->rows.size());
      auto maximum_weight_elements = std::uint64_t{0U};
      auto maximum_activation_width = std::uint64_t{0U};
      auto maximum_projected_width = std::uint64_t{0U};
      auto maximum_auxiliary_width = std::uint64_t{0U};
      for (const auto &plan : h3_w8a8_attention_plans_) {
        if (!plan.convrot)
          continue;
        if (plan.qkv_weight_scale_groups != 1U ||
            plan.output_weight_scale_groups != 1U)
          fail("H3 ConvRot BF16 correction does not accept grouped weight scales");
        if (plan.has_qkv_projection) {
          maximum_weight_elements = std::max(
              maximum_weight_elements, plan.packed_inner * plan.hidden);
          maximum_activation_width =
              std::max(maximum_activation_width, plan.hidden);
          maximum_projected_width =
              std::max(maximum_projected_width, plan.packed_inner);
        }
        if (plan.has_output_projection) {
          maximum_weight_elements = std::max(
              maximum_weight_elements, plan.hidden * plan.inner);
          maximum_activation_width =
              std::max(maximum_activation_width, plan.inner);
          maximum_projected_width =
              std::max(maximum_projected_width, plan.hidden);
        }
      }
      for (const auto &plan : h3_w8a8_mlp_plans_) {
        if (!plan.convrot)
          continue;
        if (plan.fc1_weight_scale_groups != 1U ||
            plan.fc2_weight_scale_groups != 1U)
          fail("H3 ConvRot BF16 correction does not accept grouped MLP weight scales");
        maximum_weight_elements = std::max(
            {maximum_weight_elements, plan.packed_ffn * plan.hidden,
             plan.hidden * plan.ffn});
        maximum_activation_width = std::max(
            {maximum_activation_width, plan.hidden, plan.ffn});
        maximum_projected_width =
            std::max(maximum_projected_width, plan.packed_ffn);
        maximum_auxiliary_width =
            std::max(maximum_auxiliary_width, plan.ffn);
      }
      const auto checked_bytes = [](std::uint64_t a, std::uint64_t b,
                                    const char *label) {
        if (a != 0U && b > std::numeric_limits<std::uint64_t>::max() / a)
          fail(std::string(label) + " overflows U64");
        return a * b;
      };
      correction->weight_bytes = checked_bytes(
          maximum_weight_elements, sizeof(std::uint16_t),
          "H3 ConvRot correction weight bytes");
      correction->activation_bytes = checked_bytes(
          checked_bytes(correction_rows, maximum_activation_width,
                        "H3 ConvRot correction activation elements"),
          sizeof(std::uint16_t), "H3 ConvRot correction activation bytes");
      correction->projected_bytes = checked_bytes(
          checked_bytes(correction_rows, maximum_projected_width,
                        "H3 ConvRot correction projected elements"),
          sizeof(std::uint16_t), "H3 ConvRot correction projected bytes");
      correction->auxiliary_bytes = checked_bytes(
          checked_bytes(correction_rows, maximum_auxiliary_width,
                        "H3 ConvRot correction auxiliary elements"),
          sizeof(std::uint16_t), "H3 ConvRot correction auxiliary bytes");
      correction->storage_bytes =
          align_256(correction_rows * sizeof(std::uint32_t)) +
          align_256(correction->weight_bytes) +
          align_256(correction->activation_bytes) +
          align_256(correction->projected_bytes) +
          align_256(correction->auxiliary_bytes);
      h3_convrot_bf16_correction_ = std::move(correction);
    }
    if (options.h3_owned_attention && !options.h3_ck_attention_dso.empty())
      fail("h3_owned_attention and h3_ck_attention_dso select two H3 attention "
           "routes; choose one");
    if (options.h3_owned_attention_center_k && !options.h3_owned_attention)
      fail("h3_owned_attention_center_k requires h3_owned_attention");
    if (options.h3_owned_attention || !options.h3_ck_attention_dso.empty()) {
      auto library =
          options.h3_owned_attention
              ? std::make_shared<CkAttentionLibrary>(
                    CkAttentionLibrary::InTree{}, major * 10 + minor,
                    options.h3_owned_attention_center_k)
              : std::make_shared<CkAttentionLibrary>(
                    options.h3_ck_attention_dso, major * 10 + minor);
      std::uint64_t maximum_sequence = 0U;
      for (const auto &operation : program_.operations) {
        if (operation.opcode != ir::Opcode::Attention ||
            operation.inputs.empty())
          continue;
        const auto *query = program_.tensor(operation.inputs.front());
        if (!query || query->dims.empty())
          fail("H3 CK attention found an invalid query tensor");
        maximum_sequence = std::max(maximum_sequence, query->dims.front());
      }
      auto transformer_attention_index = std::uint32_t{0U};
      for (const auto &operation : program_.operations) {
        if (operation.opcode != ir::Opcode::Attention)
          continue;
        const auto *query = program_.tensor(operation.inputs.front());
        if (!query || query->dims.front() != maximum_sequence)
          continue;
        const auto layer_index = transformer_attention_index++;
        if (layer_index < options.h3_int8_attention_first_layer)
          continue;
        if (h3_ck_attention_plans_.size() >=
            options.h3_int8_attention_layers)
          continue;
        if (operation.u64(ir::AttrKey::Implementation, 1U) != 2U)
          fail("H3 CK attention can replace only an exact backend Attention");
        if (!h3_ck_attention_plan_)
          h3_ck_attention_plan_ = std::make_shared<CkAttentionPlan>(
              program_, operation, library);
        else if (!h3_ck_attention_plan_->compatible(program_, operation))
          fail("H3 CK attention scratch reuse requires one H3 geometry");
        h3_ck_attention_layer_.emplace(
            operation.id,
            static_cast<std::uint32_t>(h3_ck_attention_plans_.size()));
        h3_ck_attention_plans_.emplace(operation.id, h3_ck_attention_plan_);
      }
      if (h3_ck_attention_plans_.empty())
        fail("H3 INT8 attention route requested but no transformer Attention "
             "operation was selected");
      ck_attention_scratch_bytes_ =
          h3_ck_attention_plan_->scratch_bytes();
    }
    {
      // Hybrid sub-range: which routed operations follow the per-run
      // h3_int8_attention_active switch.
      const auto routed =
          static_cast<std::uint32_t>(h3_ck_attention_plans_.size());
      const bool explicit_subrange =
          options.h3_int8_attention_hybrid_first_layer != 0U ||
          options.h3_int8_attention_hybrid_layers !=
              std::numeric_limits<std::uint32_t>::max();
      if (explicit_subrange && !options.h3_int8_attention_hybrid)
        fail("h3_int8_attention_hybrid_first_layer/layers require "
             "h3_int8_attention_hybrid");
      if (explicit_subrange &&
          (routed == 0U ||
           options.h3_int8_attention_hybrid_first_layer >= routed ||
           options.h3_int8_attention_hybrid_layers == 0U ||
           options.h3_int8_attention_hybrid_layers >
               routed - options.h3_int8_attention_hybrid_first_layer))
        fail("H3 INT8 attention hybrid sub-range leaves the routed "
             "attention operations (" + std::to_string(routed) + ")");
      h3_int8_attention_hybrid_first_layer_ =
          options.h3_int8_attention_hybrid_first_layer;
      h3_int8_attention_hybrid_layers_ =
          routed == 0U
              ? 0U
              : std::min(options.h3_int8_attention_hybrid_layers,
                         routed - options.h3_int8_attention_hybrid_first_layer);
    }
    {
      std::unordered_map<std::uint32_t, std::uint32_t> direct_aliases;
      for (const auto operation_id : options.alias_reshape_operations) {
        if (!reshape_alias_operations_.insert(operation_id).second)
          fail("duplicate reshape-alias operation id: " +
               std::to_string(operation_id));
        const auto operation = std::find_if(
            program_.operations.begin(), program_.operations.end(),
            [&](const ir::Operation &value) {
              return value.id == operation_id;
            });
        if (operation == program_.operations.end() ||
            operation->opcode != ir::Opcode::Reshape ||
            operation->inputs.size() != 1U ||
            operation->outputs.size() != 1U)
          fail("reshape-alias id is not a unary Reshape: " +
               std::to_string(operation_id));
        const auto *input = program_.tensor(operation->inputs.front());
        const auto *output = program_.tensor(operation->outputs.front());
        if (!input || !output || output->roles != ir::TensorRole::Internal ||
            input->dtype != output->dtype ||
            input->byte_count() != output->byte_count())
          fail("reshape-alias operation is not an equal-storage internal view: " +
               std::to_string(operation_id));
        if (!direct_aliases
                 .emplace(output->id, input->id)
                 .second)
          fail("multiple reshape aliases target tensor " +
               std::to_string(output->id));
      }
      for (const auto &[output, input] : direct_aliases) {
        auto root = input;
        std::unordered_set<std::uint32_t> seen{output};
        while (direct_aliases.contains(root)) {
          if (!seen.insert(root).second)
            fail("reshape aliases contain a cycle");
          root = direct_aliases.at(root);
        }
        reshape_aliases_.emplace(output, root);
      }
    }
    auto generated = compiler::emit_cuda(program_);
    generated.skipped_operations.insert(reshape_alias_operations_.begin(),
                                        reshape_alias_operations_.end());
    generated.source +=
        fused_linear_swiglu_source(fused_linear_swiglu_plans_);
    generated.source += h3_w8a8_source(
        !h3_w8a8_mlp_plans_.empty() ||
        !h3_w8a8_attention_plans_.empty());
    const auto has_h3_convrot =
        std::any_of(h3_w8a8_mlp_plans_.begin(), h3_w8a8_mlp_plans_.end(),
                    [](const auto &plan) { return plan.convrot; }) ||
        std::any_of(h3_w8a8_attention_plans_.begin(),
                    h3_w8a8_attention_plans_.end(),
                    [](const auto &plan) { return plan.convrot; });
    const auto has_convrot =
        has_h3_convrot || !convrot_int8_linear_plans_.empty();
    generated.source += h3_convrot_source(has_convrot);
    generated.source += h3_groupwise_source(!h3_groupwise_plans_.empty());
    generated.source += materialized_f32_attention_source(
        !materialized_f32_attention_plans_.empty());
    std::unordered_set<std::uint32_t> fused_linear_operations;
    std::unordered_set<std::uint32_t> excluded_tensors;
    std::unordered_set<std::uint32_t> replaced_constant_tensors;
    for (const auto &operation : program_.operations) {
      if (!generated.skipped_operations.contains(operation.id))
        continue;
      for (const auto output : operation.outputs)
        excluded_tensors.insert(output);
    }
    for (const auto &fusion : fused_linear_swiglu_plans_) {
      fused_linear_operations.insert(fusion.linear_operation);
      excluded_tensors.insert(fusion.intermediate_tensor);
    }
    for (const auto &plan : absorbed_linear_bias_plans_)
      excluded_tensors.insert(plan.intermediate_tensor);
    for (const auto &plan : h3_modulation_cache_plans_) {
      fused_linear_operations.insert(plan.linear_operation);
      excluded_tensors.insert(plan.projected_tensor);
      replaced_constant_tensors.insert(plan.replaced_constant_tensors.begin(),
                                       plan.replaced_constant_tensors.end());
    }
    for (const auto &plan : h3_groupwise_plans_) {
      excluded_tensors.insert(plan.excluded_tensors.begin(),
                              plan.excluded_tensors.end());
      replaced_constant_tensors.insert(plan.replaced_constant_tensors.begin(),
                                       plan.replaced_constant_tensors.end());
    }
    for (const auto &plan : h3_w8a8_mlp_plans_) {
      fused_linear_operations.insert(plan.fc1_operation);
      fused_linear_operations.insert(plan.fc2_operation);
      excluded_tensors.insert(plan.excluded_tensors.begin(),
                              plan.excluded_tensors.end());
      replaced_constant_tensors.insert(plan.replaced_constant_tensors.begin(),
                                       plan.replaced_constant_tensors.end());
    }
    for (const auto &plan : h3_w8a8_attention_plans_) {
      if (plan.has_qkv_projection)
        fused_linear_operations.insert(plan.qkv_linear_operations.begin(),
                                       plan.qkv_linear_operations.end());
      if (plan.has_output_projection)
        fused_linear_operations.insert(plan.output_linear_operation);
      excluded_tensors.insert(plan.excluded_tensors.begin(),
                              plan.excluded_tensors.end());
      replaced_constant_tensors.insert(plan.replaced_constant_tensors.begin(),
                                       plan.replaced_constant_tensors.end());
    }
    for (const auto &plan : convrot_int8_linear_plans_) {
      if (!options.convrot_int8_weight_only_quality)
        fused_linear_operations.insert(plan.operation);
      replaced_constant_tensors.insert(plan.weight_tensor);
    }
    for (const auto &plan : h3_compact_adaln_plans_) {
      excluded_tensors.insert(plan.expanded_tensors.begin(),
                              plan.expanded_tensors.end());
      excluded_tensors.insert(plan.normalized_tensors.begin(),
                              plan.normalized_tensors.end());
    }
    // Root cause of the W8A8+streamed nondeterminism: a fused W8A8 plan
    // launches its kernels at an earlier operation than the semantic ops
    // it skips, but the streamed prefetcher's readiness wait for a
    // streamed constant is issued at the SKIPPED consumer's index - so a
    // streamed constant consumed only by a skipped op (the ResidualGate's
    // gate/residual inputs, or the QKV chain's attention input) is read
    // by an already-queued kernel with no ordering edge against its
    // copy-stream upload. Promote every such constant to dedicated
    // resident storage uploaded (and fenced) at preparation. Programs
    // without such constants (all production H3 graphs: gates/residuals
    // are activations) are byte-for-byte unaffected.
    auto promote_streamed_constant = [&](std::uint32_t id, bool required) {
      const auto *description = program_.tensor(id);
      if (!description ||
          !description->has_role(ir::TensorRole::Constant) ||
          !description->has_role(ir::TensorRole::Streamed)) {
        if (required)
          fail("resident streamed-constant id is not a streamed constant: " +
               std::to_string(id));
        return;
      }
      if (!constants_.contains(id)) {
        if (!required)
          return;
        fail("resident streamed-constant id has no bound tensor: " +
             std::to_string(id));
      }
      if (std::find(promoted_streamed_constants_.begin(),
                    promoted_streamed_constants_.end(),
                    id) != promoted_streamed_constants_.end())
        return;
      promoted_streamed_constants_.push_back(id);
      const auto bytes = align_256(description->byte_count());
      if (promoted_constant_bytes_ >
          std::numeric_limits<std::uint64_t>::max() - bytes)
        fail("promoted streamed-constant storage overflow");
      promoted_constant_bytes_ += bytes;
    };
    for (const auto id : options.resident_streamed_constants)
      promote_streamed_constant(id, true);
    for (const auto &plan : h3_w8a8_mlp_plans_) {
      promote_streamed_constant(plan.residual_tensor, false);
      promote_streamed_constant(plan.gate_tensor, false);
    }
    for (const auto &plan : h3_w8a8_attention_plans_) {
      if (plan.has_qkv_projection)
        promote_streamed_constant(plan.attention_input_tensor, false);
      if (plan.has_output_projection) {
        promote_streamed_constant(plan.residual_tensor, false);
        promote_streamed_constant(plan.gate_tensor, false);
      }
    }
    for (const auto &plan : h3_w8a8_attention_plans_)
      if (plan.compact_adaln.enabled)
        promote_streamed_constant(plan.compact_adaln.norm_weight_tensor, true);
    for (const auto &plan : h3_w8a8_mlp_plans_)
      if (plan.compact_adaln.enabled)
        promote_streamed_constant(plan.compact_adaln.norm_weight_tensor, true);
    for (const auto id : promoted_streamed_constants_)
      replaced_constant_tensors.insert(id);
    constexpr std::size_t linear_workspace_bytes = 64U * 1024U * 1024U;
    const bool contains_linear = std::any_of(
        program_.operations.begin(), program_.operations.end(),
        [&](const ir::Operation &op) {
          return op.opcode == ir::Opcode::Linear &&
                 !generated.launch_inputs.contains(op.id) &&
                 !fused_linear_operations.contains(op.id);
        });
    workspace_bytes_ = contains_linear ? linear_workspace_bytes : 0U;
    workspace_bytes_ = std::max<std::size_t>(
        workspace_bytes_,
        static_cast<std::size_t>(flash_attention_workspace_bytes_));
    if (options.h3_int8_cublaslt) {
      if (h3_w8a8_mlp_plans_.empty() &&
          h3_w8a8_attention_plans_.empty())
        fail("H3 INT8 cuBLASLt policy requested without a direct INT8 plan");
      workspace_bytes_ = std::max(workspace_bytes_, linear_workspace_bytes);
      h3_int8_gemm_registry_ = std::make_unique<H3Int8GemmRegistry>(
          context_.cublas_lt(), workspace_bytes_,
          options.h3_int8_cublaslt_heuristic_rank);
      for (const auto &plan : h3_w8a8_mlp_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += plan.chunk_rows) {
          const auto rows = static_cast<std::uint32_t>(
              std::min<std::uint64_t>(plan.chunk_rows,
                                      plan.rows - row_start));
          h3_int8_gemm_registry_->add(
              {rows, static_cast<std::uint32_t>(plan.packed_ffn),
               static_cast<std::uint32_t>(plan.hidden)});
          h3_int8_gemm_registry_->add(
              {rows, static_cast<std::uint32_t>(plan.hidden),
               static_cast<std::uint32_t>(plan.ffn)});
        }
      }
      for (const auto &plan : h3_w8a8_attention_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += kH3W8A8ProjectionChunkRows) {
          const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(
              kH3W8A8ProjectionChunkRows, plan.rows - row_start));
          if (plan.has_qkv_projection)
            h3_int8_gemm_registry_->add(
                {rows, static_cast<std::uint32_t>(plan.packed_inner),
                 static_cast<std::uint32_t>(plan.hidden)});
          if (plan.has_output_projection)
            h3_int8_gemm_registry_->add(
                {rows, static_cast<std::uint32_t>(plan.hidden),
                 static_cast<std::uint32_t>(plan.inner)});
        }
      }
    }
    {
      std::unordered_set<std::uint32_t> scheduled;
      const std::unordered_set<std::uint32_t> resident_streamed(
          options.resident_streamed_constants.begin(),
          options.resident_streamed_constants.end());
      std::size_t maximum_width = 1U;
      for (const auto &group : options.parallel_linear_groups) {
        if (group.size() < 2U)
          fail("parallel Linear group requires at least two operations");
        std::vector<std::size_t> indices;
        indices.reserve(group.size());
        std::optional<std::uint32_t> activation;
        for (const auto operation_id : group) {
          if (!scheduled.insert(operation_id).second)
            fail("parallel Linear operation appears in multiple groups");
          const auto found = std::find_if(
              program_.operations.begin(), program_.operations.end(),
              [&](const ir::Operation &operation) {
                return operation.id == operation_id;
              });
          if (found == program_.operations.end() ||
              found->opcode != ir::Opcode::Linear ||
              found->inputs.size() != 2U || found->outputs.size() != 1U ||
              generated.launch_inputs.contains(operation_id) ||
              fused_linear_operations.contains(operation_id))
            fail("parallel Linear group requires unfused unbiased Linears");
          if (!activation)
            activation = found->inputs[0];
          else if (*activation != found->inputs[0])
            fail("parallel Linear group operations must share one input");
          const auto *weight = program_.tensor(found->inputs[1]);
          if (!weight || !weight->has_role(ir::TensorRole::Constant) ||
              (weight->has_role(ir::TensorRole::Streamed) &&
               !resident_streamed.contains(weight->id)))
            fail("parallel Linear group weights must be resident");
          indices.push_back(static_cast<std::size_t>(
              std::distance(program_.operations.begin(), found)));
        }
        std::sort(indices.begin(), indices.end());
        for (std::size_t index = 1U; index < indices.size(); ++index)
          if (indices[index] != indices.front() + index)
            fail("parallel Linear group operations must be contiguous");
        parallel_linear_followups_.insert(indices.begin() + 1U,
                                          indices.end());
        parallel_linear_groups_.emplace(indices.front(), std::move(indices));
        maximum_width = std::max(maximum_width, group.size());
      }
      if (maximum_width > 1U) {
        if (workspace_bytes_ > std::numeric_limits<std::uint64_t>::max() /
                                   (maximum_width - 1U))
          fail("parallel Linear workspace size overflow");
        parallel_workspace_bytes_ =
            workspace_bytes_ * (maximum_width - 1U);
      }
    }
    cudnn_workspace_bytes_ = 0U;
#if DIF_HAS_CUDNN
    std::unordered_map<CudnnAttentionKey,
                       std::shared_ptr<CudnnAttentionPlan>,
                       CudnnAttentionKeyHash>
        cudnn_plan_cache;
    std::unordered_map<CudnnConv2dKey, std::shared_ptr<CudnnConv2dPlan>,
                       CudnnConv2dKeyHash>
        cudnn_conv_plan_cache;
    std::unordered_map<CudnnConv3dKey, std::shared_ptr<CudnnConv3dPlan>,
                       CudnnConv3dKeyHash>
        cudnn_conv3d_plan_cache;
    h3_int8_attention_hybrid_ = options.h3_int8_attention_hybrid;
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::Attention ||
          op.u64(ir::AttrKey::Implementation, 1U) != 2U ||
          (h3_ck_attention_plans_.contains(op.id) &&
           !options.h3_int8_attention_hybrid))
        continue;
      const auto *query = program_.tensor(op.inputs.at(0));
      if (!query)
        fail("cuDNN attention references a missing query tensor");
      const bool batched = query->dims.size() == 4U;
      const auto batch = batched ? query->dims.at(0) : 1U;
      const auto sequence = query->dims.at(batched ? 1U : 0U);
      const auto heads = query->dims.at(batched ? 2U : 1U);
      const auto head_dim = query->dims.at(batched ? 3U : 2U);
      const CudnnAttentionKey key{
          query->dtype,
          batch,
          sequence,
          heads,
          op.u64(ir::AttrKey::KvHeads, heads),
          head_dim,
          std::bit_cast<std::uint64_t>(op.f64(
              ir::AttrKey::AttentionScale,
              1.0 / std::sqrt(static_cast<double>(head_dim)))),
          op.boolean(ir::AttrKey::Causal, false),
          op.inputs.size() == 4U,
          options.cudnn_attention_heuristic,
      };
      auto found = cudnn_plan_cache.find(key);
      if (found == cudnn_plan_cache.end()) {
        auto plan = std::make_shared<CudnnAttentionPlan>(
            *query, key.kv_heads, std::bit_cast<double>(key.scale_bits),
            key.causal, key.additive_bias, key.heuristic);
        found = cudnn_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_attention_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
      uses_cudnn_attention_ = true;
    }
    // Exact query-row overlay for the INT8 attention route: one cuDNN plan
    // per range (query rows = the range, K/V rows = the full sequence),
    // shared by every routed operation because the route admits exactly one
    // H3 attention geometry.
    if (!options.h3_int8_attention_exact_query_ranges.empty()) {
      if (h3_ck_attention_plans_.empty())
        fail("h3_int8_attention_exact_query_ranges require an H3 INT8 "
             "attention route");
      auto ranges = options.h3_int8_attention_exact_query_ranges;
      std::sort(ranges.begin(), ranges.end(),
                [](const auto &a, const auto &b) { return a.begin < b.begin; });
      const ir::Operation *routed_operation = nullptr;
      for (const auto &op : program_.operations)
        if (h3_ck_attention_plans_.contains(op.id)) {
          routed_operation = &op;
          break;
        }
      const auto *query = program_.tensor(routed_operation->inputs.at(0));
      if (!query || query->dims.size() != 3U)
        fail("H3 exact query-row overlay requires the [S,H,D] route geometry");
      const auto sequence = query->dims.at(0);
      std::uint64_t previous_end = 0U;
      for (const auto &range : ranges) {
        if (range.count == 0U ||
            static_cast<std::uint64_t>(range.begin) + range.count > sequence ||
            range.begin < previous_end)
          fail("H3 exact query-row range is empty, out of bounds, or "
               "overlaps another range: [" + std::to_string(range.begin) +
               ", +" + std::to_string(range.count) + ") of " +
               std::to_string(sequence));
        previous_end = static_cast<std::uint64_t>(range.begin) + range.count;
        ir::TensorDesc subset = *query;
        subset.dims = {range.count, query->dims.at(1), query->dims.at(2)};
        auto plan = std::make_shared<CudnnAttentionPlan>(
            subset,
            routed_operation->u64(ir::AttrKey::KvHeads, query->dims.at(1)),
            routed_operation->f64(
                ir::AttrKey::AttentionScale,
                1.0 / std::sqrt(static_cast<double>(query->dims.at(2)))),
            false, false, options.cudnn_attention_heuristic, sequence);
        cudnn_workspace_bytes_ =
            std::max(cudnn_workspace_bytes_, plan->workspace_bytes());
        h3_exact_query_range_plans_.push_back(std::move(plan));
        uses_cudnn_attention_ = true;
      }
      h3_exact_query_ranges_ = std::move(ranges);
    }
    // AttentionBackward implementation 2: cuDNN SDPA backward over the saved
    // logsumexp. One plan per geometry, shared across operations.
    std::unordered_map<CudnnAttentionKey,
                       std::shared_ptr<CudnnAttentionBackwardPlan>,
                       CudnnAttentionKeyHash>
        cudnn_backward_plan_cache;
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::AttentionBackward ||
          op.u64(ir::AttrKey::Implementation, 1U) != 2U)
        continue;
      const auto *query = program_.tensor(op.inputs.at(1));
      if (!query || query->dims.size() != 3U)
        fail("cuDNN attention backward requires an [S,H,D] query");
      const auto sequence = query->dims.at(0);
      const auto heads = query->dims.at(1);
      const auto head_dim = query->dims.at(2);
      const CudnnAttentionKey key{
          query->dtype,
          1U,
          sequence,
          heads,
          op.u64(ir::AttrKey::KvHeads, heads),
          head_dim,
          std::bit_cast<std::uint64_t>(op.f64(
              ir::AttrKey::AttentionScale,
              1.0 / std::sqrt(static_cast<double>(head_dim)))),
          op.boolean(ir::AttrKey::Causal, false),
          false,
          options.cudnn_attention_heuristic,
      };
      auto found = cudnn_backward_plan_cache.find(key);
      if (found == cudnn_backward_plan_cache.end()) {
        auto plan = std::make_shared<CudnnAttentionBackwardPlan>(
            *query, key.kv_heads, std::bit_cast<double>(key.scale_bits),
            key.causal, key.heuristic);
        found = cudnn_backward_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_attention_backward_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
      // Packed [H,S] F32 copy of the program's [S,H] logsumexp per operation.
      cudnn_backward_stats_bytes_ +=
          align_256(sequence * heads * sizeof(float));
      uses_cudnn_attention_ = true;
    }
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::Conv2d)
        continue;
      const auto *input = program_.tensor(op.inputs.at(0));
      const auto *weight = program_.tensor(op.inputs.at(1));
      const auto *output = program_.tensor(op.outputs.at(0));
      if (!input || !weight || !output)
        fail("cuDNN Conv2d references a missing tensor");
      const auto pad_top = op.u64(ir::AttrKey::PadTop, 0U);
      const auto pad_bottom = op.u64(ir::AttrKey::PadBottom, 0U);
      const auto pad_west = op.u64(ir::AttrKey::PadWest, 0U);
      const auto pad_east = op.u64(ir::AttrKey::PadEast, 0U);
      if (pad_top != pad_bottom || pad_west != pad_east)
        fail("cuDNN Conv2d currently requires symmetric spatial padding");
      auto dims4 = [](const std::vector<std::uint64_t> &dims) {
        return std::array<std::uint64_t, 4>{dims[0], dims[1], dims[2],
                                            dims[3]};
      };
      constexpr std::uint64_t default_workspace = 64ULL * 1024ULL * 1024ULL;
      const auto limit = op.u64(ir::AttrKey::WorkspaceLimitBytes,
                                default_workspace);
      if (limit == 0U || limit > std::numeric_limits<std::size_t>::max())
        fail("cuDNN Conv2d workspace limit is invalid");
      const CudnnConv2dKey key{
          input->dtype,
          dims4(input->dims),
          dims4(weight->dims),
          dims4(output->dims),
          op.u64(ir::AttrKey::StrideH, 1U),
          op.u64(ir::AttrKey::StrideW, 1U),
          pad_top,
          pad_west,
          op.u64(ir::AttrKey::DilationH, 1U),
          op.u64(ir::AttrKey::DilationW, 1U),
          op.u64(ir::AttrKey::Groups, 1U),
          limit,
          op.inputs.size() == 3U,
          options.deterministic_convolution_algorithms,
      };
      auto found = cudnn_conv_plan_cache.find(key);
      if (found == cudnn_conv_plan_cache.end()) {
        auto plan = std::make_shared<CudnnConv2dPlan>(
            *input, *weight, *output, key.stride_h, key.stride_w, key.pad_h,
            key.pad_w, key.dilation_h, key.dilation_w, key.groups, key.biased,
            static_cast<std::size_t>(key.workspace_limit), key.deterministic);
        found = cudnn_conv_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_conv_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
    }
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::Conv3d)
        continue;
      const auto *input = program_.tensor(op.inputs.at(0));
      const auto *weight = program_.tensor(op.inputs.at(1));
      const auto *output = program_.tensor(op.outputs.at(0));
      if (!input || !weight || !output)
        fail("cuDNN Conv3d references a missing tensor");
      const auto front = op.u64(ir::AttrKey::PadFront, 0U);
      const auto back = op.u64(ir::AttrKey::PadBack, 0U);
      const auto top = op.u64(ir::AttrKey::PadTop, 0U);
      const auto bottom = op.u64(ir::AttrKey::PadBottom, 0U);
      const auto west = op.u64(ir::AttrKey::PadWest, 0U);
      const auto east = op.u64(ir::AttrKey::PadEast, 0U);
      if (front != back || top != bottom || west != east)
        fail("cuDNN Conv3d currently requires symmetric padding; materialize asymmetric padding with PadConstant");
      auto dims5 = [](const std::vector<std::uint64_t> &dims) {
        return std::array<std::uint64_t, 5>{dims[0], dims[1], dims[2],
                                            dims[3], dims[4]};
      };
      constexpr std::uint64_t default_workspace = 64ULL * 1024ULL * 1024ULL;
      const auto limit = op.u64(ir::AttrKey::WorkspaceLimitBytes,
                                default_workspace);
      if (limit == 0U || limit > std::numeric_limits<std::size_t>::max())
        fail("cuDNN Conv3d workspace limit is invalid");
      const CudnnConv3dKey key{
          input->dtype,
          dims5(input->dims),
          dims5(weight->dims),
          dims5(output->dims),
          op.u64(ir::AttrKey::StrideT, 1U),
          op.u64(ir::AttrKey::StrideH, 1U),
          op.u64(ir::AttrKey::StrideW, 1U),
          front,
          top,
          west,
          op.u64(ir::AttrKey::DilationT, 1U),
          op.u64(ir::AttrKey::DilationH, 1U),
          op.u64(ir::AttrKey::DilationW, 1U),
          op.u64(ir::AttrKey::Groups, 1U),
          limit,
          op.inputs.size() == 3U,
          options.deterministic_convolution_algorithms,
      };
      auto found = cudnn_conv3d_plan_cache.find(key);
      if (found == cudnn_conv3d_plan_cache.end()) {
        auto plan = std::make_shared<CudnnConv3dPlan>(
            *input, *weight, *output, key.stride_t, key.stride_h,
            key.stride_w, key.pad_t, key.pad_h, key.pad_w,
            key.dilation_t, key.dilation_h, key.dilation_w, key.groups,
            key.biased, static_cast<std::size_t>(key.workspace_limit),
            key.deterministic);
        found = cudnn_conv3d_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_conv3d_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
    }
#else
    for (const auto &op : program_.operations) {
      if (op.opcode == ir::Opcode::Attention &&
          op.u64(ir::AttrKey::Implementation, 1U) == 2U &&
          !h3_ck_attention_plans_.contains(op.id))
        fail("DiffIR requests cuDNN attention but this CUDA backend was built without cuDNN");
      if (op.opcode == ir::Opcode::Conv2d)
        fail("DiffIR requests Conv2d but this CUDA backend was built without cuDNN");
      if (op.opcode == ir::Opcode::Conv3d)
        fail("DiffIR requests Conv3d but this CUDA backend was built without cuDNN");
    }
#endif
    if (options.streamed_prefetch_depth == 0U)
      fail("streamed prefetch depth must be nonzero");
    if (options.streamed_staging_buffers <
        options.streamed_prefetch_depth + 1U)
      fail("streamed staging ring must exceed the prefetch depth");
    streamed_prefetch_depth_ = options.streamed_prefetch_depth;
    // The plan widens every streamed interval by the prefetch depth, so a
    // prefetch issued depth operations ahead can only overwrite a slot
    // whose previous tenant has already been submitted (safety argument
    // in the run loop below).
    memory_plan_ = compiler::plan_memory(
        program_, 256U,
        options.overlap_streaming ? streamed_prefetch_depth_ : 0U,
        excluded_tensors, replaced_constant_tensors, reshape_aliases_, true,
        repeated_invariant_persistent_tensors_);
    const auto tensor_bytes = memory_plan_.total_bytes;
    if (tensor_bytes > std::numeric_limits<std::uint64_t>::max() - workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() -
                parallel_workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ + parallel_workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() - cudnn_workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ + parallel_workspace_bytes_ +
                cudnn_workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() -
                materialized_f32_attention_score_bytes_)
      fail("DiffIR allocation plus backend workspace overflow");
    auto h3_w8a8_weight_bytes = std::uint64_t{0U};
    auto h3_w8a8_scratch_bytes = std::uint64_t{0U};
    for (const auto &plan : h3_w8a8_mlp_plans_) {
      if (plan.resident) {
        if (h3_w8a8_weight_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                plan.weight_storage_bytes)
          fail("H3 W8A8 prepared storage overflow");
        h3_w8a8_weight_bytes += plan.weight_storage_bytes;
        if (options.lazy_resident_upload)
          h3_w8a8_tail_stage_half_bytes_ = std::max(
              h3_w8a8_tail_stage_half_bytes_, plan.quantized_weight_bytes);
      } else {
        h3_w8a8_tail_mlp_bytes_ =
            std::max(h3_w8a8_tail_mlp_bytes_, plan.weight_storage_bytes);
        h3_w8a8_tail_stage_half_bytes_ = std::max(
            h3_w8a8_tail_stage_half_bytes_, plan.quantized_weight_bytes);
      }
      h3_w8a8_scratch_bytes =
          std::max(h3_w8a8_scratch_bytes, plan.scratch_bytes);
    }
    for (const auto &plan : h3_w8a8_attention_plans_) {
      if (plan.resident) {
        if (h3_w8a8_weight_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                plan.weight_storage_bytes)
          fail("H3 W8A8 prepared storage overflow");
        h3_w8a8_weight_bytes += plan.weight_storage_bytes;
        if (options.lazy_resident_upload)
          h3_w8a8_tail_stage_half_bytes_ = std::max(
              h3_w8a8_tail_stage_half_bytes_, plan.quantized_weight_bytes);
      } else {
        h3_w8a8_tail_attention_bytes_ = std::max(
            h3_w8a8_tail_attention_bytes_, plan.weight_storage_bytes);
        h3_w8a8_tail_stage_half_bytes_ = std::max(
            h3_w8a8_tail_stage_half_bytes_, plan.quantized_weight_bytes);
      }
      h3_w8a8_scratch_bytes =
          std::max(h3_w8a8_scratch_bytes, plan.scratch_bytes);
    }
    if (h3_w8a8_tail_attention_bytes_ >
        std::numeric_limits<std::uint64_t>::max() -
            h3_w8a8_tail_mlp_bytes_)
      fail("H3 W8A8 reusable tail storage overflow");
    h3_w8a8_tail_weight_bytes_ =
        h3_w8a8_tail_attention_bytes_ + h3_w8a8_tail_mlp_bytes_;
    if (h3_w8a8_weight_bytes >
        std::numeric_limits<std::uint64_t>::max() -
            h3_w8a8_tail_weight_bytes_)
      fail("H3 W8A8 resident plus tail storage overflow");
    h3_w8a8_weight_bytes += h3_w8a8_tail_weight_bytes_;
    const auto h3_w8a8_bytes =
        h3_w8a8_weight_bytes + h3_w8a8_scratch_bytes;
    auto convrot_weight_bytes = std::uint64_t{0U};
    auto convrot_scale_bytes = std::uint64_t{0U};
    auto convrot_activation_bytes = std::uint64_t{0U};
    auto convrot_activation_scale_bytes = std::uint64_t{0U};
    auto convrot_quality_weight_bytes = std::uint64_t{0U};
    auto convrot_resident_weight_bytes = std::uint64_t{0U};
    for (const auto &plan : convrot_int8_linear_plans_) {
      convrot_weight_bytes =
          std::max(convrot_weight_bytes, plan.weight.byte_size());
      convrot_scale_bytes =
          std::max(convrot_scale_bytes, plan.scale.byte_size());
      const auto activation_elements = plan.rows * plan.contraction;
      convrot_activation_bytes = std::max(
          convrot_activation_bytes,
          activation_elements *
              (options.convrot_int8_weight_only_quality
                   ? sizeof(std::uint16_t)
                   : sizeof(std::int8_t)));
      if (!options.convrot_int8_weight_only_quality)
        convrot_activation_scale_bytes =
            std::max(convrot_activation_scale_bytes,
                     plan.rows * sizeof(float));
      else
        convrot_quality_weight_bytes = std::max(
            convrot_quality_weight_bytes,
            plan.weight.element_count() * sizeof(std::uint16_t));
      const auto resident_plan_bytes =
          align_256(plan.weight.byte_size()) +
          align_256(plan.scale.byte_size());
      if (convrot_resident_weight_bytes >
          std::numeric_limits<std::uint64_t>::max() - resident_plan_bytes)
        fail("resident generic ConvRot weight storage overflow");
      convrot_resident_weight_bytes += resident_plan_bytes;
    }
    const auto convrot_weight_slot_bytes =
        align_256(convrot_weight_bytes) + align_256(convrot_scale_bytes);
    const auto convrot_scratch_bytes =
        align_256(convrot_activation_bytes) +
        align_256(convrot_activation_scale_bytes) +
        align_256(convrot_quality_weight_bytes);
    if (!options.convrot_int8_resident &&
        convrot_weight_slot_bytes >
            std::numeric_limits<std::uint64_t>::max() / 2U)
      fail("generic ConvRot streamed slot storage overflow");
    const auto convrot_weight_storage_bytes =
        options.convrot_int8_resident
            ? convrot_resident_weight_bytes
            : 2U * convrot_weight_slot_bytes;
    if (convrot_weight_storage_bytes >
        std::numeric_limits<std::uint64_t>::max() - convrot_scratch_bytes)
      fail("generic ConvRot prepared storage overflow");
    const auto convrot_bytes =
        convrot_weight_storage_bytes + convrot_scratch_bytes;
    auto h3_groupwise_weight_bytes = std::uint64_t{0U};
    auto h3_groupwise_scratch_bytes = std::uint64_t{0U};
    for (const auto &plan : h3_groupwise_plans_) {
      if (h3_groupwise_weight_bytes >
          std::numeric_limits<std::uint64_t>::max() -
              plan.weight_storage_bytes)
        fail("H3 groupwise prepared storage overflow");
      h3_groupwise_weight_bytes += plan.weight_storage_bytes;
      h3_groupwise_scratch_bytes =
          std::max(h3_groupwise_scratch_bytes, plan.scratch_bytes);
    }
    const auto h3_groupwise_bytes =
        h3_groupwise_weight_bytes + h3_groupwise_scratch_bytes;
    auto h3_modulation_bytes = std::uint64_t{0U};
    for (const auto &plan : h3_modulation_cache_plans_) {
      if (h3_modulation_bytes >
          std::numeric_limits<std::uint64_t>::max() - plan.storage_bytes)
        fail("H3 modulation prepared storage overflow");
      h3_modulation_bytes += plan.storage_bytes;
    }
    const auto base_required = tensor_bytes + workspace_bytes_ +
                               parallel_workspace_bytes_ +
                               cudnn_workspace_bytes_ +
                               materialized_f32_attention_score_bytes_;
    if (base_required > std::numeric_limits<std::uint64_t>::max() -
                            ck_attention_scratch_bytes_ -
                            cudnn_backward_stats_bytes_)
      fail("DiffIR allocation plus attention scratch overflow");
    const auto base_with_attention =
        base_required + ck_attention_scratch_bytes_ +
        cudnn_backward_stats_bytes_;
    if (base_with_attention > std::numeric_limits<std::uint64_t>::max() -
                            h3_w8a8_bytes)
      fail("DiffIR allocation plus H3 W8A8 storage overflow");
    if (base_with_attention + h3_w8a8_bytes >
        std::numeric_limits<std::uint64_t>::max() - convrot_bytes)
      fail("DiffIR allocation plus generic ConvRot storage overflow");
    const auto base_with_convrot =
        base_with_attention + h3_w8a8_bytes + convrot_bytes;
    if (base_with_convrot >
        std::numeric_limits<std::uint64_t>::max() - h3_groupwise_bytes)
      fail("DiffIR allocation plus H3 groupwise storage overflow");
    const auto base_with_weights =
        base_with_convrot + h3_groupwise_bytes;
    if (base_with_weights >
        std::numeric_limits<std::uint64_t>::max() - h3_modulation_bytes)
      fail("DiffIR allocation plus H3 modulation storage overflow");
    const auto base_with_modulation =
        base_with_weights + h3_modulation_bytes;
    if (base_with_modulation > std::numeric_limits<std::uint64_t>::max() -
                                   promoted_constant_bytes_)
      fail("DiffIR allocation plus promoted constant storage overflow");
    const auto base_with_promoted =
        base_with_modulation + promoted_constant_bytes_;
    if (base_with_promoted > std::numeric_limits<std::uint64_t>::max() -
                                 repeated_invariant_cache_bytes_)
      fail("DiffIR allocation plus repeated-invariant cache overflow");
    const auto base_with_repeated =
        base_with_promoted + repeated_invariant_cache_bytes_;
    const auto h3_convrot_correction_bytes =
        h3_convrot_bf16_correction_
            ? h3_convrot_bf16_correction_->storage_bytes
            : 0U;
    if (base_with_repeated > std::numeric_limits<std::uint64_t>::max() -
                                 h3_convrot_correction_bytes)
      fail("DiffIR allocation plus H3 ConvRot correction storage overflow");
    const auto required =
        base_with_repeated + h3_convrot_correction_bytes;
    if (options.profile_pipeline) {
      std::cerr << "CUDA_MEMORY_PLAN tensor_bytes=" << tensor_bytes
                << " linear_workspace_bytes=" << workspace_bytes_
                << " parallel_linear_workspace_bytes="
                << parallel_workspace_bytes_
                << " attention_workspace_bytes=" << cudnn_workspace_bytes_
                << " materialized_f32_attention_score_bytes="
                << materialized_f32_attention_score_bytes_
                << " ck_attention_scratch_bytes="
                << ck_attention_scratch_bytes_
                << " h3_w8a8_weight_bytes=" << h3_w8a8_weight_bytes
                << " h3_w8a8_scratch_bytes=" << h3_w8a8_scratch_bytes
                << " h3_w8a8_tail_weight_bytes="
                << h3_w8a8_tail_weight_bytes_
                << " h3_w8a8_tail_stage_half_bytes="
                << h3_w8a8_tail_stage_half_bytes_
                << " convrot_int8_bytes=" << convrot_bytes
                << " h3_groupwise_bytes=" << h3_groupwise_bytes
                << " h3_modulation_bytes=" << h3_modulation_bytes
                << " promoted_streamed_constant_bytes="
                << promoted_constant_bytes_
                << " repeated_invariant_cache_bytes="
                << repeated_invariant_cache_bytes_
                << " h3_convrot_correction_bytes="
                << h3_convrot_correction_bytes
                << " required_bytes=" << required
                << " free_before_bytes=" << free_bytes_before_
                << " minimum_free_bytes=" << options.minimum_free_bytes
                << '\n';
      for (const auto &slot : memory_plan_.slots) {
        if (slot.bytes >= 64ULL * 1024ULL * 1024ULL)
          std::cerr << "CUDA_MEMORY_SLOT id=" << slot.id
                    << " bytes=" << slot.bytes << '\n';
      }
    }
    if (required > free_bytes_before_ ||
        free_bytes_before_ - required < options.minimum_free_bytes)
      fail("GPU pressure gate refused candidate: required=" +
           std::to_string(required) + " free_before=" +
           std::to_string(free_bytes_before_) + " minimum_free=" +
           std::to_string(options.minimum_free_bytes));
    resident_bytes_ = required;
    // Single device reservation backing every prepare-time allocation
    // below (memory-plan slots + all feature workspaces). The slack
    // absorbs the per-take 256-byte alignment padding; take() fails
    // closed if the accounting above ever under-covers.
    arena_ = std::make_unique<DeviceArena>(required + 64U * 1024U);

    fused_launch_inputs_ = generated.launch_inputs;
    skipped_operations_ = generated.skipped_operations;
    for (const auto &fusion : fused_linear_swiglu_plans_)
      skipped_operations_.insert(fusion.swiglu_operation);
    for (const auto &plan : absorbed_linear_bias_plans_)
      skipped_operations_.insert(plan.bias_operation);
    for (const auto &plan : h3_w8a8_mlp_plans_) {
      skipped_operations_.insert(plan.swiglu_operation);
      skipped_operations_.insert(plan.fc2_operation);
      skipped_operations_.insert(plan.residual_operation);
    }
    for (const auto &plan : h3_w8a8_attention_plans_) {
      if (plan.has_qkv_projection)
        skipped_operations_.insert(plan.qkv_linear_operations.begin(),
                                   plan.qkv_linear_operations.end());
      if (plan.has_output_projection)
        skipped_operations_.insert(plan.residual_operation);
    }
    for (const auto &plan : h3_compact_adaln_plans_) {
      skipped_operations_.insert(plan.select_operation);
      skipped_operations_.insert(plan.norm_operations.begin(),
                                 plan.norm_operations.end());
    }
    for (const auto &plan : h3_modulation_cache_plans_)
      skipped_operations_.insert(plan.linear_operation);
    // FP8 physical formats need FP8 tensor cores (sm_89+): the generated
    // e4m3 conversions do not assemble below that. Fail closed from the
    // probed target facts before compiling, naming the format and target.
    for (const auto &operation : program_.operations) {
      const bool fp8_opcode =
          operation.opcode == ir::Opcode::QuantizeFp8Rows ||
          operation.opcode == ir::Opcode::LinearFp8Scaled ||
          operation.opcode == ir::Opcode::QuantizeFp8Blocks32 ||
          operation.opcode == ir::Opcode::LinearFp8BlockScaled;
      if (fp8_opcode && !target_profile_.precision.fp8_tensor_cores)
        fail(std::string("operation ") + std::to_string(operation.id) + " " +
             std::string(ir::opcode_name(operation.opcode)) +
             " uses an FP8 physical format that is illegal on target " +
             std::string(target::architecture_name(
                 target_profile_.architecture)) +
             " (no FP8 tensor cores, compute " +
             std::to_string(target_profile_.compute_major) + "." +
             std::to_string(target_profile_.compute_minor) +
             "); see difopt --formats-table");
    }
    const auto ptx = compile_ptx(generated.source, major, minor,
                                 options.cache_directory, source_hash_);
    module_ = std::make_unique<Module>(ptx);
    if (!materialized_f32_attention_plans_.empty())
      check(cuModuleGetFunction(
                &materialized_f32_attention_softmax_, module_->get(),
                "dif_materialized_f32_attention_softmax"),
            "cuModuleGetFunction materialized F32 attention softmax");
    for (const auto &[operation, entrypoint] : generated.entrypoints) {
      CUfunction function{};
      check(cuModuleGetFunction(&function, module_->get(), entrypoint.c_str()),
            "cuModuleGetFunction");
      functions_.emplace(operation, function);
    }
    for (const auto &operation : program_.operations) {
      const auto implementation = static_cast<ir::Int8RowQuantization>(
          operation.u64(ir::AttrKey::Implementation,
                        static_cast<std::uint64_t>(
                            ir::Int8RowQuantization::Direct)));
      if (operation.opcode != ir::Opcode::QuantizeInt8Rows ||
          !ir::is_convrot_int8_row_quantization(implementation))
        continue;
      const auto found = functions_.find(operation.id);
      const auto *quantized = program_.tensor(operation.outputs.front());
      if (found == functions_.end() || !quantized)
        fail("H256 ConvRot quantization kernel was not compiled");
      const auto shared_bytes = quantized->dims.back() * sizeof(float);
      if (shared_bytes > static_cast<std::uint64_t>(
                             std::numeric_limits<int>::max()))
        fail("H256 ConvRot dynamic shared-memory request overflows CUDA int");
      check(cuFuncSetAttribute(
                found->second, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(shared_bytes)),
            "cuFuncSetAttribute H256 ConvRot dynamic shared memory");
    }
    for (auto &fusion : fused_linear_swiglu_plans_)
      check(cuModuleGetFunction(&fusion.function, module_->get(),
                                fusion.entrypoint.c_str()),
            "cuModuleGetFunction fused Linear->SwiGlu");
    if (!h3_w8a8_mlp_plans_.empty() ||
        !h3_w8a8_attention_plans_.empty()) {
      check(cuModuleGetFunction(&h3_w8a8_functions_.rowscale, module_->get(),
                                "dif_h3_w8a8_rowscale"),
            "cuModuleGetFunction H3 W8A8 rowscale");
      check(cuModuleGetFunction(&h3_w8a8_functions_.encode, module_->get(),
                                "dif_h3_w8a8_encode"),
            "cuModuleGetFunction H3 W8A8 encode");
      check(cuModuleGetFunction(&h3_w8a8_functions_.qkv, module_->get(),
                                "dif_h3_w8a8_qkv"),
            "cuModuleGetFunction H3 W8A8 QKV");
      check(cuModuleGetFunction(&h3_w8a8_functions_.swiglu, module_->get(),
                                "dif_h3_w8a8_swiglu"),
            "cuModuleGetFunction H3 W8A8 SwiGLU");
      check(cuModuleGetFunction(&h3_w8a8_functions_.residual, module_->get(),
                                "dif_h3_w8a8_residual"),
            "cuModuleGetFunction H3 W8A8 residual");
    }
    if (has_convrot) {
      check(cuModuleGetFunction(&h3_convrot_functions_.generic_encode,
                                module_->get(),
                                "dif_convrot_int8_encode_cached"),
            "cuModuleGetFunction generic ConvRot INT8 encode");
      if (!options.convrot_int8_weight_only_quality) {
        std::uint64_t maximum_k = 0U;
        for (const auto &plan : convrot_int8_linear_plans_)
          maximum_k = std::max(maximum_k, plan.contraction);
        constexpr std::uint64_t temporary_floats = 8U * 2U * 256U;
        const auto shared_bytes =
            (maximum_k + temporary_floats) * sizeof(float);
        if (shared_bytes > static_cast<std::uint64_t>(
                               std::numeric_limits<int>::max()))
          fail("generic ConvRot dynamic shared-memory request overflows CUDA int");
        check(cuFuncSetAttribute(
                  h3_convrot_functions_.generic_encode,
                  CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                  static_cast<int>(shared_bytes)),
              "cuFuncSetAttribute generic ConvRot dynamic shared memory");
      }
      if (options.convrot_int8_weight_only_quality) {
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.generic_bf16_rotate, module_->get(),
                  "dif_convrot_bf16_rotate_streamed"),
              "cuModuleGetFunction generic ConvRot BF16 rotate");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.generic_weight_dequant,
                  module_->get(), "dif_convrot_weight_dequant_bf16"),
              "cuModuleGetFunction generic ConvRot weight dequant");
      }
    }
    if (has_h3_convrot) {
      check(cuModuleGetFunction(&h3_convrot_functions_.encode, module_->get(),
                                "dif_convrot_int8_encode"),
            "cuModuleGetFunction ConvRot INT8 encode");
      check(cuModuleGetFunction(&h3_convrot_functions_.chunked_encode,
                                module_->get(),
                                "dif_convrot_int8_encode_chunked"),
            "cuModuleGetFunction chunk-scaled ConvRot INT8 encode");
      check(cuModuleGetFunction(&h3_convrot_functions_.compact_encode,
                                module_->get(),
                                "dif_h3_convrot_compact_adaln_encode"),
            "cuModuleGetFunction H3 compact AdaLN ConvRot encode");
      check(cuModuleGetFunction(
                &h3_convrot_functions_.compact_chunked_encode, module_->get(),
                "dif_h3_convrot_compact_adaln_encode_chunked"),
            "cuModuleGetFunction H3 compact AdaLN chunk-scaled ConvRot encode");
      check(cuModuleGetFunction(&h3_convrot_functions_.chunk_accumulate,
                                module_->get(),
                                "dif_h3_convrot_chunk_accumulate"),
            "cuModuleGetFunction H3 ConvRot chunk accumulation");
      check(cuModuleGetFunction(&h3_convrot_functions_.qkv, module_->get(),
                                "dif_h3_convrot_qkv"),
            "cuModuleGetFunction H3 ConvRot QKV");
      check(cuModuleGetFunction(&h3_convrot_functions_.qkv_bf16,
                                module_->get(),
                                "dif_h3_convrot_qkv_bf16"),
            "cuModuleGetFunction H3 ConvRot BF16 QKV");
      check(cuModuleGetFunction(&h3_convrot_functions_.qkv_f32,
                                module_->get(),
                                "dif_h3_convrot_qkv_f32"),
            "cuModuleGetFunction H3 ConvRot F32 QKV");
      check(cuModuleGetFunction(&h3_convrot_functions_.swiglu, module_->get(),
                                "dif_h3_convrot_swiglu"),
            "cuModuleGetFunction H3 ConvRot SwiGLU");
      check(cuModuleGetFunction(&h3_convrot_functions_.swiglu_encode,
                                module_->get(),
                                "dif_h3_convrot_swiglu_int8_encode"),
            "cuModuleGetFunction H3 ConvRot SwiGLU encode");
      check(cuModuleGetFunction(&h3_convrot_functions_.swiglu_bf16_encode,
                                module_->get(),
                                "dif_h3_convrot_swiglu_bf16_int8_encode"),
            "cuModuleGetFunction H3 ConvRot BF16 SwiGLU encode");
      check(cuModuleGetFunction(&h3_convrot_functions_.swiglu_f32,
                                module_->get(),
                                "dif_h3_convrot_swiglu_f32"),
            "cuModuleGetFunction H3 ConvRot F32 SwiGLU");
      check(cuModuleGetFunction(&h3_convrot_functions_.compact_residual,
                                module_->get(),
                                "dif_h3_convrot_compact_adaln_residual"),
            "cuModuleGetFunction H3 compact AdaLN residual");
      check(cuModuleGetFunction(
                &h3_convrot_functions_.compact_residual_bf16, module_->get(),
                "dif_h3_convrot_compact_adaln_residual_bf16"),
            "cuModuleGetFunction H3 compact AdaLN BF16 residual");
      check(cuModuleGetFunction(
                &h3_convrot_functions_.compact_residual_f32, module_->get(),
                "dif_h3_convrot_compact_adaln_residual_f32"),
            "cuModuleGetFunction H3 compact AdaLN F32 residual");
      check(cuModuleGetFunction(&h3_convrot_functions_.residual_f32,
                                module_->get(),
                                "dif_h3_convrot_residual_f32"),
            "cuModuleGetFunction H3 ConvRot F32 residual");
      if (h3_convrot_bf16_correction_) {
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.generic_bf16_rotate, module_->get(),
                  "dif_convrot_bf16_rotate_streamed"),
              "cuModuleGetFunction H3 correction BF16 rotate");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.generic_weight_dequant,
                  module_->get(), "dif_convrot_weight_dequant_bf16"),
              "cuModuleGetFunction H3 correction weight dequant");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.bf16_rotate_gather, module_->get(),
                  "dif_convrot_bf16_rotate_gather"),
              "cuModuleGetFunction H3 correction BF16 gather");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.compact_bf16_rotate_gather,
                  module_->get(),
                  "dif_h3_convrot_compact_adaln_bf16_rotate_gather"),
              "cuModuleGetFunction H3 compact correction BF16 gather");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.qkv_bf16_scatter, module_->get(),
                  "dif_h3_convrot_qkv_bf16_scatter"),
              "cuModuleGetFunction H3 correction QKV scatter");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.swiglu_bf16, module_->get(),
                  "dif_h3_convrot_swiglu_bf16"),
              "cuModuleGetFunction H3 correction SwiGLU");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.compact_residual_bf16_scatter,
                  module_->get(),
                  "dif_h3_convrot_compact_residual_bf16_scatter"),
              "cuModuleGetFunction H3 compact correction residual scatter");
        check(cuModuleGetFunction(
                  &h3_convrot_functions_.residual_bf16_scatter,
                  module_->get(),
                  "dif_h3_convrot_residual_bf16_scatter"),
              "cuModuleGetFunction H3 correction residual scatter");
      }
      std::uint64_t maximum_k = 0U;
      for (const auto &plan : h3_w8a8_mlp_plans_)
        if (plan.convrot)
          maximum_k = std::max({maximum_k, plan.hidden, plan.ffn});
      for (const auto &plan : h3_w8a8_attention_plans_)
        if (plan.convrot)
          maximum_k = std::max({maximum_k, plan.hidden, plan.inner});
      constexpr std::uint64_t temporary_floats = 8U * 2U * 256U;
      const auto shared_bytes = (maximum_k + temporary_floats) * sizeof(float);
      if (shared_bytes > static_cast<std::uint64_t>(
                             std::numeric_limits<int>::max()))
        fail("H3 ConvRot dynamic shared-memory request overflows CUDA int");
      check(cuFuncSetAttribute(
                h3_convrot_functions_.encode,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(shared_bytes)),
            "cuFuncSetAttribute H3 ConvRot dynamic shared memory");
      check(cuFuncSetAttribute(
                h3_convrot_functions_.compact_encode,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(shared_bytes)),
            "cuFuncSetAttribute H3 compact AdaLN dynamic shared memory");
      check(cuFuncSetAttribute(
                h3_convrot_functions_.swiglu_encode,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(shared_bytes)),
            "cuFuncSetAttribute H3 ConvRot SwiGLU dynamic shared memory");
      check(cuFuncSetAttribute(
                h3_convrot_functions_.swiglu_bf16_encode,
                CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                static_cast<int>(shared_bytes)),
            "cuFuncSetAttribute H3 ConvRot BF16 SwiGLU dynamic shared memory");
    }
    if (!h3_groupwise_plans_.empty())
      check(cuModuleGetFunction(&h3_groupwise_dequant_function_, module_->get(),
                                "dif_h3_groupwise_dequant"),
            "cuModuleGetFunction H3 groupwise INT8 dequant");
    excluded_tensors.insert(replaced_constant_tensors.begin(),
                            replaced_constant_tensors.end());
    buffers_.allocate(program_, memory_plan_, excluded_tensors, arena_.get());
    repeated_invariant_cache_storage_ = std::make_unique<Workspace>(
        repeated_invariant_cache_bytes_, arena_.get());
    workspace_ = std::make_unique<Workspace>(workspace_bytes_, arena_.get());
    if (parallel_workspace_bytes_ != 0U) {
      const auto auxiliary_count =
          parallel_workspace_bytes_ / workspace_bytes_;
      parallel_streams_.reserve(auxiliary_count);
      parallel_workspaces_.reserve(auxiliary_count);
      parallel_done_events_.reserve(auxiliary_count);
      for (std::size_t index = 0U; index < auxiliary_count; ++index) {
        parallel_streams_.push_back(std::make_unique<Stream>());
        parallel_workspaces_.push_back(
            std::make_unique<Workspace>(workspace_bytes_, arena_.get()));
        parallel_done_events_.push_back(
            std::make_unique<Event>(CU_EVENT_DISABLE_TIMING));
      }
      parallel_start_event_ =
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
    }
    cudnn_workspace_ =
        std::make_unique<Workspace>(cudnn_workspace_bytes_, arena_.get());
    materialized_f32_attention_scores_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(materialized_f32_attention_score_bytes_),
        arena_.get());
    h3_w8a8_scratch_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(h3_w8a8_scratch_bytes), arena_.get());
    if (h3_convrot_bf16_correction_) {
      auto &correction = *h3_convrot_bf16_correction_;
      correction.storage = std::make_unique<Workspace>(
          static_cast<std::size_t>(correction.storage_bytes), arena_.get());
      auto offset = std::uint64_t{0U};
      correction.indices = correction.storage->pointer() + offset;
      offset += align_256(correction.rows.size() * sizeof(std::uint32_t));
      correction.weight = correction.storage->pointer() + offset;
      offset += align_256(correction.weight_bytes);
      correction.activation = correction.storage->pointer() + offset;
      offset += align_256(correction.activation_bytes);
      correction.projected = correction.storage->pointer() + offset;
      offset += align_256(correction.projected_bytes);
      correction.auxiliary = correction.storage->pointer() + offset;
      offset += align_256(correction.auxiliary_bytes);
      if (offset != correction.storage_bytes)
        fail("H3 ConvRot correction storage layout mismatch");
      check(counted_memcpy_htod(
                correction.indices, correction.rows.data(),
                correction.rows.size() * sizeof(std::uint32_t),
                context_.stream()),
            "cuMemcpyHtoDAsync H3 ConvRot correction rows");
    }
    h3_w8a8_tail_weight_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(h3_w8a8_tail_weight_bytes_), arena_.get());
    if (h3_w8a8_tail_stage_half_bytes_ != 0U) {
      if (h3_w8a8_tail_stage_half_bytes_ >
          std::numeric_limits<std::size_t>::max() / 2U)
        fail("H3 W8A8 reusable tail host staging overflow");
      h3_w8a8_tail_stage_ = std::make_unique<PinnedHostWorkspace>(
          static_cast<std::size_t>(2U * h3_w8a8_tail_stage_half_bytes_));
      for (auto &event : h3_w8a8_tail_stage_events_)
        event = std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
      h3_w8a8_tail_order_event_ =
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
      h3_w8a8_tail_ready_event_ =
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
    }
    if (!h3_w8a8_mlp_plans_.empty() ||
        !h3_w8a8_attention_plans_.empty()) {
      if (options.streamed_stage_threads == 0U)
        fail("streamed staging worker count must be positive");
      h3_w8a8_staging_pool_ =
          std::make_unique<StagingPool>(options.streamed_stage_threads);
    }
    convrot_weight_slot_bytes_ = convrot_weight_slot_bytes;
    convrot_weight_bytes_ = convrot_weight_bytes;
    convrot_scale_bytes_ = convrot_scale_bytes;
    convrot_weight_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(convrot_weight_storage_bytes),
        arena_.get());
    convrot_scratch_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(convrot_scratch_bytes), arena_.get());
    if (!convrot_int8_linear_plans_.empty()) {
      convrot_resident_ = options.convrot_int8_resident;
      if (convrot_resident_) {
        auto offset = std::uint64_t{0U};
        for (auto &plan : convrot_int8_linear_plans_) {
          plan.weight_device = convrot_weight_storage_->pointer() + offset;
          offset += align_256(plan.weight.byte_size());
          plan.scale_device = convrot_weight_storage_->pointer() + offset;
          offset += align_256(plan.scale.byte_size());
        }
        if (offset != convrot_weight_storage_bytes)
          fail("resident generic ConvRot storage layout mismatch");
      } else {
        if (options.streamed_stage_threads == 0U)
          fail("generic ConvRot staging worker count must be positive");
        convrot_staging_ = std::make_unique<PinnedHostWorkspace>(
            static_cast<std::size_t>(2U * convrot_weight_slot_bytes_));
        convrot_staging_pool_ =
            std::make_unique<StagingPool>(options.streamed_stage_threads);
        for (auto &event : convrot_slot_done_)
          event = std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
      }
      convrot_activation_device_ = convrot_scratch_storage_->pointer();
      convrot_activation_scale_device_ =
          convrot_activation_device_ + align_256(convrot_activation_bytes);
      convrot_quality_weight_device_ =
          convrot_activation_scale_device_ +
          align_256(convrot_activation_scale_bytes);
      convrot_weight_only_quality_ =
          options.convrot_int8_weight_only_quality;
      if (convrot_weight_only_quality_)
        for (const auto &plan : convrot_int8_linear_plans_)
          buffers_.bind_external(plan.weight_tensor,
                                 convrot_quality_weight_device_);
    }
    h3_groupwise_scratch_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(h3_groupwise_scratch_bytes), arena_.get());
    h3_modulation_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(h3_modulation_bytes), arena_.get());
    assign_h3_modulation_storage(h3_modulation_cache_plans_,
                                 h3_modulation_storage_->pointer(), buffers_);
    promoted_constant_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(promoted_constant_bytes_), arena_.get());
    {
      auto promoted_offset = std::uint64_t{0U};
      for (const auto id : promoted_streamed_constants_) {
        buffers_.bind_external(
            id, promoted_constant_storage_->pointer() + promoted_offset);
        promoted_offset += align_256(constants_.at(id).byte_size());
      }
      if (promoted_offset != promoted_constant_bytes_)
        fail("promoted streamed-constant storage layout mismatch");
    }
    if (h3_ck_attention_plan_)
      h3_ck_attention_plan_->allocate(arena_.get());
#if DIF_HAS_CUDNN
    for (const auto &[operation_id, plan] : cudnn_attention_backward_plans_) {
      const auto &op = *std::find_if(
          program_.operations.begin(), program_.operations.end(),
          [&](const ir::Operation &candidate) {
            return candidate.id == operation_id;
          });
      const auto *query = program_.tensor(op.inputs.at(1));
      cudnn_backward_stats_.emplace(
          operation_id,
          std::make_unique<Workspace>(
              static_cast<std::size_t>(query->dims.at(0) * query->dims.at(1) *
                                       sizeof(float)),
              arena_.get()));
    }
#endif
    for (auto &plan : h3_w8a8_mlp_plans_) {
      if (plan.resident)
        allocate_h3_w8a8_weights(plan, arena_.get());
      else
        assign_h3_w8a8_weights(
            plan, h3_w8a8_tail_weight_storage_->pointer() +
                      h3_w8a8_tail_attention_bytes_);
      assign_h3_w8a8_scratch(plan, h3_w8a8_scratch_storage_->pointer());
    }
    for (auto &plan : h3_w8a8_attention_plans_) {
      if (plan.resident)
        allocate_h3_w8a8_weights(plan, arena_.get());
      else
        assign_h3_w8a8_weights(
            plan, h3_w8a8_tail_weight_storage_->pointer());
      assign_h3_w8a8_scratch(plan, h3_w8a8_scratch_storage_->pointer());
    }
#if DIF_HAS_CUTLASS
    if (options.h3_int8_cutlass_scaled_fc1 ||
        options.h3_int8_cutlass_scaled_all ||
        (!options.convrot_int8_weight_only_quality &&
         std::any_of(convrot_int8_linear_plans_.begin(),
                     convrot_int8_linear_plans_.end(), [](const auto &plan) {
                       return plan.dtype == ir::DType::BF16;
                     }))) {
      h3_int8_scaled_gemm_registry_ =
          std::make_unique<H3Int8ScaledGemmRegistry>();
      for (const auto &plan : h3_w8a8_mlp_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += plan.chunk_rows) {
          const auto rows = static_cast<std::uint32_t>(
              std::min<std::uint64_t>(plan.chunk_rows,
                                      plan.rows - row_start));
          h3_int8_scaled_gemm_registry_->add(
              {rows, static_cast<std::uint32_t>(plan.packed_ffn),
               static_cast<std::uint32_t>(plan.hidden)},
              plan.input_i8_device, plan.fc1_weight_device,
              plan.input_scale_device, plan.fc1_scale_device,
              plan.fc1_accumulator_device, context_.stream());
          if (plan.cutlass_scaled_fc2)
            h3_int8_scaled_gemm_registry_->add(
                {rows, static_cast<std::uint32_t>(plan.hidden),
                 static_cast<std::uint32_t>(plan.ffn)},
                plan.activation_i8_device, plan.fc2_weight_device,
                plan.activation_scale_device, plan.fc2_scale_device,
                plan.fc2_accumulator_device, context_.stream());
        }
      }
      for (const auto &plan : h3_w8a8_attention_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += kH3W8A8ProjectionChunkRows) {
          const auto rows = static_cast<std::uint32_t>(
              std::min<std::uint64_t>(kH3W8A8ProjectionChunkRows,
                                      plan.rows - row_start));
          if (plan.has_qkv_projection && plan.cutlass_scaled)
            h3_int8_scaled_gemm_registry_->add(
                {rows, static_cast<std::uint32_t>(plan.packed_inner),
                 static_cast<std::uint32_t>(plan.hidden)},
                plan.activation_i8_device + row_start * plan.hidden,
                plan.qkv_weight_device,
                plan.activation_scale_device + row_start * sizeof(float),
                plan.qkv_scale_device, plan.accumulator_device,
                context_.stream());
          if (plan.has_output_projection && plan.cutlass_scaled)
            h3_int8_scaled_gemm_registry_->add(
                {rows, static_cast<std::uint32_t>(plan.hidden),
                 static_cast<std::uint32_t>(plan.inner)},
                plan.activation_i8_device, plan.output_weight_device,
                plan.activation_scale_device, plan.output_scale_device,
                plan.accumulator_device, context_.stream());
        }
      }
      for (const auto &plan : convrot_int8_linear_plans_) {
        if (options.convrot_int8_weight_only_quality)
          break;
        if (plan.dtype != ir::DType::BF16)
          continue;
        const auto slots = convrot_resident_ ? 1U : 2U;
        for (std::size_t slot = 0U; slot < slots; ++slot) {
          const auto slot_base = convrot_weight_storage_->pointer() +
                                 slot * convrot_weight_slot_bytes_;
          const auto weight_device =
              convrot_resident_ ? plan.weight_device : slot_base;
          const auto scale_device =
              convrot_resident_
                  ? plan.scale_device
                  : slot_base + align_256(convrot_weight_bytes_);
          h3_int8_scaled_gemm_registry_->add(
              {static_cast<std::uint32_t>(plan.rows),
               static_cast<std::uint32_t>(plan.columns),
               static_cast<std::uint32_t>(plan.contraction)},
              convrot_activation_device_, weight_device,
              convrot_activation_scale_device_, scale_device,
              buffers_.at(plan.output_tensor), context_.stream());
        }
      }
    }
    if (!options.convrot_int8_weight_only_quality &&
        std::any_of(convrot_int8_linear_plans_.begin(),
                    convrot_int8_linear_plans_.end(), [](const auto &plan) {
                      return plan.dtype == ir::DType::F16;
                    })) {
      int8_scaled_f16_gemm_registry_ =
          std::make_unique<Int8ScaledF16GemmRegistry>();
      for (const auto &plan : convrot_int8_linear_plans_) {
        if (plan.dtype != ir::DType::F16)
          continue;
        const auto slots = convrot_resident_ ? 1U : 2U;
        for (std::size_t slot = 0U; slot < slots; ++slot) {
          const auto slot_base = convrot_weight_storage_->pointer() +
                                 slot * convrot_weight_slot_bytes_;
          const auto weight_device =
              convrot_resident_ ? plan.weight_device : slot_base;
          const auto scale_device =
              convrot_resident_
                  ? plan.scale_device
                  : slot_base + align_256(convrot_weight_bytes_);
          int8_scaled_f16_gemm_registry_->add(
              {static_cast<std::uint32_t>(plan.rows),
               static_cast<std::uint32_t>(plan.columns),
               static_cast<std::uint32_t>(plan.contraction)},
              convrot_activation_device_, weight_device,
              convrot_activation_scale_device_, scale_device,
              plan.bias_tensor ? buffers_.at(plan.bias_tensor) : 0U,
              buffers_.at(plan.output_tensor), context_.stream());
        }
      }
    }
#endif
    if (options.h3_int8_cublaslt_tune) {
      if (!h3_int8_gemm_registry_)
        fail("H3 INT8 cuBLASLt tuning requires the cuBLASLt projection path");
      for (const auto &plan : h3_w8a8_mlp_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += plan.chunk_rows) {
          const auto rows = static_cast<std::uint32_t>(
              std::min<std::uint64_t>(plan.chunk_rows,
                                      plan.rows - row_start));
          h3_int8_gemm_registry_->tune(
              {rows, static_cast<std::uint32_t>(plan.packed_ffn),
               static_cast<std::uint32_t>(plan.hidden)},
              plan.input_i8_device, plan.fc1_weight_device,
              plan.fc1_accumulator_device, *workspace_, context_.stream());
          h3_int8_gemm_registry_->tune(
              {rows, static_cast<std::uint32_t>(plan.hidden),
               static_cast<std::uint32_t>(plan.ffn)},
              plan.activation_i8_device, plan.fc2_weight_device,
              plan.fc2_accumulator_device, *workspace_, context_.stream());
        }
      }
      for (const auto &plan : h3_w8a8_attention_plans_) {
        for (std::uint64_t row_start = 0U; row_start < plan.rows;
             row_start += kH3W8A8ProjectionChunkRows) {
          const auto rows = static_cast<std::uint32_t>(std::min<std::uint64_t>(
              kH3W8A8ProjectionChunkRows, plan.rows - row_start));
          if (plan.has_qkv_projection)
            h3_int8_gemm_registry_->tune(
                {rows, static_cast<std::uint32_t>(plan.packed_inner),
                 static_cast<std::uint32_t>(plan.hidden)},
                plan.activation_i8_device, plan.qkv_weight_device,
                plan.accumulator_device, *workspace_, context_.stream());
          if (plan.has_output_projection)
            h3_int8_gemm_registry_->tune(
                {rows, static_cast<std::uint32_t>(plan.hidden),
                 static_cast<std::uint32_t>(plan.inner)},
                plan.activation_i8_device, plan.output_weight_device,
                plan.accumulator_device, *workspace_, context_.stream());
        }
      }
    }
    for (const auto &[output, root] : reshape_aliases_) {
      if (!buffers_.contains(root))
        fail("reshape alias root has no prepared device storage: " +
             std::to_string(root));
      buffers_.bind_external(output, buffers_.at(root));
    }
    for (auto &plan : h3_groupwise_plans_) {
      allocate_h3_groupwise_weights(plan, arena_.get());
      assign_h3_groupwise_scratch(
          plan, h3_groupwise_scratch_storage_->pointer(), buffers_);
    }
    streamed_prefetcher_ = std::make_unique<StreamedPrefetcher>(
        program_, constants_, memory_plan_, buffers_, context_,
        options.streamed_staging_buffers, options.streamed_stage_threads,
        options.streamed_pinned_budget_bytes,
        replaced_constant_tensors,
        options.lazy_resident_upload
            ? std::unordered_set<std::uint32_t>(
                  promoted_streamed_constants_.begin(),
                  promoted_streamed_constants_.end())
            : std::unordered_set<std::uint32_t>{},
        options.streamed_direct_io, options.direct_io_warm_page_cache,
        options.profile_pipeline);
    if (options.pinned_io_staging) {
      auto input_bytes = std::uint64_t{0U};
      auto output_bytes = std::uint64_t{0U};
      for (const auto &description : program_.tensors) {
        if (description.has_role(ir::TensorRole::Input)) {
          if (input_bytes > std::numeric_limits<std::uint64_t>::max() -
                                description.byte_count())
            fail("pinned I/O staging input size overflow");
          input_bytes += description.byte_count();
        }
        if (description.has_role(ir::TensorRole::Output)) {
          if (output_bytes > std::numeric_limits<std::uint64_t>::max() -
                                 description.byte_count())
            fail("pinned I/O staging output size overflow");
          output_bytes += description.byte_count();
        }
      }
      const auto io_bytes = std::max(input_bytes, output_bytes);
      if (io_bytes > options.streamed_pinned_budget_bytes)
        fail("pinned I/O staging exceeds the pinned budget: io_bytes=" +
             std::to_string(io_bytes) + " budget_bytes=" +
             std::to_string(options.streamed_pinned_budget_bytes));
      pinned_io_ = std::make_unique<PinnedHostWorkspace>(
          static_cast<std::size_t>(io_bytes));
    }
    {
      std::unordered_set<std::uint32_t> tuned_ids(
          options.tune_linear_operations.begin(),
          options.tune_linear_operations.end());
      std::unordered_set<std::uint32_t> ranked_ids;
      for (const auto &choice : options.linear_algorithm_choices)
        ranked_ids.insert(choice.operation_id);
      std::unordered_map<std::string, std::shared_ptr<LinearPlan>> shared_plans;
      std::uint64_t linear_operation_count = 0U;
      std::uint64_t isolated_plan_count = 0U;
      for (const auto &op : program_.operations) {
        if (op.opcode != ir::Opcode::Linear ||
            fused_launch_inputs_.contains(op.id) ||
            fused_linear_operations.contains(op.id))
          continue;
        const auto absorbed = std::find_if(
            absorbed_linear_bias_plans_.begin(),
            absorbed_linear_bias_plans_.end(),
            [&](const AbsorbedLinearBiasPlan &plan) {
              return plan.linear_operation == op.id;
            });
        // An absorbed Linear builds and launches the biased plan form:
        // input and weight plus the BiasAdd's bias, writing its output.
        const auto &plan_operation =
            absorbed != absorbed_linear_bias_plans_.end()
                ? absorbed->launch_operation
                : op;
        // A restored single-candidate plan cannot serve tuning, rank
        // selection, or an expanded search; those always take fresh
        // heuristics.
        const auto allow_restore = !options.expand_linear_algorithms &&
                                   !options.deterministic_linear_algorithms &&
                                   !tuned_ids.contains(op.id) &&
                                   !ranked_ids.contains(op.id);
        ++linear_operation_count;
        const auto *input = program_.tensor(plan_operation.inputs.at(0));
        const auto *weight = program_.tensor(plan_operation.inputs.at(1));
        const auto *output = program_.tensor(plan_operation.outputs.at(0));
        const auto shareable = allow_restore && input && weight && output &&
                               input->dtype == ir::DType::BF16;
        std::string key;
        if (shareable) {
          const auto rows = input->dims.at(0);
          const auto inner = input->element_count() / rows;
          const auto width = weight->dims.at(0);
          key = "bf16|m=" + std::to_string(rows) +
                "|n=" + std::to_string(width) +
                "|k=" + std::to_string(inner) +
                "|bias=" +
                std::to_string(plan_operation.inputs.size() == 3U ? 1U : 0U) +
                "|bias_mode=" +
                std::to_string(plan_operation.u64(
                    ir::AttrKey::LinearBiasMode,
                    static_cast<std::uint64_t>(ir::LinearBiasMode::Epilogue))) +
                "|implementation=" +
                std::to_string(plan_operation.u64(
                    ir::AttrKey::Implementation, 1U)) +
                "|workspace=" +
                std::to_string(plan_operation.u64(
                    ir::AttrKey::WorkspaceLimitBytes, workspace_bytes_)) +
                "|persist=" +
                std::to_string(options.persist_linear_heuristics ? 1U : 0U);
        }
        auto shared = shareable ? shared_plans.find(key) : shared_plans.end();
        if (shared != shared_plans.end()) {
          linear_plans_.emplace(op.id, shared->second);
          continue;
        }
        auto plan = std::make_shared<LinearPlan>(
            program_, plan_operation, buffers_, context_.cublas_lt(),
            workspace_bytes_, options.expand_linear_algorithms, major,
            minor, options.cache_directory,
            options.persist_linear_heuristics, allow_restore,
            &linear_heuristic_cache_stats_,
            plan_operation.inputs.size() == 3U &&
                    plan_operation.u64(
                        ir::AttrKey::LinearBiasMode,
                        static_cast<std::uint64_t>(
                            ir::LinearBiasMode::Epilogue)) ==
                        static_cast<std::uint64_t>(
                            ir::LinearBiasMode::Addmm)
                ? functions_.at(op.id)
                : nullptr,
            options.deterministic_linear_algorithms);
        linear_plans_.emplace(op.id, plan);
        if (shareable) {
          std::cerr << "CUDA_LINEAR_PLAN_CLASS operation=" << op.id
                    << " key=" << key << " "
                    << plan->selected_algorithm_description() << "\n";
          shared_plans.emplace(std::move(key), std::move(plan));
        } else
          ++isolated_plan_count;
      }
      const auto unique_plan_count =
          static_cast<std::uint64_t>(shared_plans.size()) +
          isolated_plan_count;
      std::cerr << "CUDA_LINEAR_PLAN_POOL operations="
                << linear_operation_count
                << " unique=" << unique_plan_count
                << " reused=" << linear_operation_count - unique_plan_count
                << " isolated=" << isolated_plan_count
                << "\n";
    }
    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::LinearFp8Scaled)
        continue;
      fp8_scaled_linear_plans_.emplace(
          operation.id, std::make_unique<Fp8ScaledLinearPlan>(
                            program_, operation, context_.cublas_lt(),
                            workspace_bytes_));
    }
    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::LinearFp8BlockScaled)
        continue;
      // Physical-format legality from discovered target facts, never from
      // product names: FP8 tensor cores plus a cuBLASLt with block-scaled
      // matmul (>= 12.8). Both are reported in the failure.
      const auto linked = target_profile_.cublaslt_version;
      const bool library_ok = DIF_HAS_CUBLASLT_BLOCK_SCALE && linked >= 120800U;
      if (!library_ok || !target_profile_.precision.fp8_tensor_cores)
        fail(std::string("LinearFp8BlockScaled (MXFP8) is illegal here: ") +
             (library_ok ? "" : "needs cuBLASLt >= 120800 block-scaled matmul (build cuBLAS " +
                                std::to_string(CUBLAS_VERSION) + ", linked cuBLASLt " +
                                std::to_string(linked) + "); ") +
             (target_profile_.precision.fp8_tensor_cores
                  ? ""
                  : std::string("target ") +
                        std::string(target::architecture_name(
                            target_profile_.architecture)) +
                        " lacks FP8 tensor cores; ") +
             "see difopt --formats-table (mxfp8-block-scaled)");
#if DIF_HAS_CUBLASLT_BLOCK_SCALE
      fp8_block_scaled_linear_plans_.emplace(
          operation.id, std::make_unique<Fp8BlockScaledLinearPlan>(
                            program_, operation, buffers_, context_.cublas_lt(),
                            workspace_bytes_));
#endif
    }
#if DIF_HAS_CUTLASS
    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::LinearInt8Scaled)
        continue;
      int8_scaled_linear_plans_.emplace(
          operation.id, std::make_unique<Int8ScaledLinearPlan>(
                            program_, operation, buffers_, context_.stream()));
    }
    for (const auto &operation : program_.operations) {
      if (operation.opcode != ir::Opcode::LinearInt8WeightScaled)
        continue;
      int8_weight_linear_plans_.emplace(
          operation.id, std::make_unique<Int8WeightLinearPlan>(
                            program_, operation, buffers_, context_.stream()));
    }
    std::unordered_set<std::uint32_t> cutlass_operations;
    for (const auto &choice : options.cutlass_linear_operations) {
      if (!cutlass_operations.insert(choice.operation_id).second)
        fail("duplicate CUTLASS Linear operation id: " +
             std::to_string(choice.operation_id));
      const auto operation = std::find_if(
          program_.operations.begin(), program_.operations.end(),
          [&](const ir::Operation &value) {
            return value.id == choice.operation_id;
          });
      if (operation == program_.operations.end() ||
          operation->opcode != ir::Opcode::Linear ||
          !linear_plans_.contains(choice.operation_id))
        fail("requested CUTLASS id is not an unfused Linear: " +
             std::to_string(choice.operation_id));
      cutlass_linear_plans_.emplace(
          choice.operation_id,
          std::make_unique<CutlassLinearPlan>(
              program_, *operation, buffers_, choice.schedule,
              context_.stream()));
    }
#else
    if (std::any_of(program_.operations.begin(), program_.operations.end(),
                    [](const ir::Operation &operation) {
                      return operation.opcode ==
                                 ir::Opcode::LinearInt8Scaled ||
                             operation.opcode ==
                                 ir::Opcode::LinearInt8WeightScaled;
                    }))
      fail("INT8 Linear requested but the backend was built without "
           "CUTLASS");
    if (!options.cutlass_linear_operations.empty())
      fail("CUTLASS Linear requested but the backend was built without CUTLASS");
#endif
    for (const auto &description : program_.tensors) {
      if (description.has_role(ir::TensorRole::Constant) &&
          !description.has_role(ir::TensorRole::Streamed) &&
          buffers_.contains(description.id))
        resident_weight_bytes_ += description.byte_count();
    }
    for (const auto &plan : h3_w8a8_mlp_plans_)
      if (plan.resident)
        resident_weight_bytes_ += plan.weight_storage_bytes;
    for (const auto &plan : h3_w8a8_attention_plans_)
      if (plan.resident)
        resident_weight_bytes_ += plan.weight_storage_bytes;
    resident_weight_bytes_ += h3_w8a8_tail_weight_bytes_;
    for (const auto &plan : h3_groupwise_plans_)
      resident_weight_bytes_ += plan.weight_storage_bytes;
    if (convrot_resident_)
      resident_weight_bytes_ += convrot_resident_weight_bytes;
    resident_weight_bytes_ += h3_modulation_bytes;
    for (const auto id : promoted_streamed_constants_)
      resident_weight_bytes_ += program_.tensor(id)->byte_count();
    std::unordered_set<std::uint32_t> selected_linear_operations;
    for (const auto &choice : options.linear_algorithm_choices) {
      if (!selected_linear_operations.insert(choice.operation_id).second)
        fail("duplicate selected cuBLASLt Linear operation id: " +
             std::to_string(choice.operation_id));
      if (!linear_plans_.contains(choice.operation_id))
        fail("selected cuBLASLt id is not an unfused Linear: " +
             std::to_string(choice.operation_id));
#if DIF_HAS_CUTLASS
      if (cutlass_linear_plans_.contains(choice.operation_id))
        fail("a Linear cannot select both cuBLASLt and CUTLASS implementations");
#endif
      linear_plans_.at(choice.operation_id)
          ->select_heuristic(choice.heuristic_rank);
      selected_linear_algorithms_.push_back(choice);
    }
    const auto resident_upload_start = std::chrono::steady_clock::now();
    if (options.profile_pipeline && resident_weight_bytes_ != 0U) {
      struct rusage before {};
      struct rusage after {};
      if (getrusage(RUSAGE_SELF, &before) != 0)
        fail("getrusage before resident prefault failed");
      const auto page_size = ::sysconf(_SC_PAGESIZE);
      if (page_size <= 0)
        fail("cannot determine host page size for resident profile");
      const auto prefault_start = std::chrono::steady_clock::now();
      std::uint64_t checksum = 0U;
      for (const auto &description : program_.tensors) {
        if (!description.has_role(ir::TensorRole::Constant) ||
            description.has_role(ir::TensorRole::Streamed) ||
            !buffers_.contains(description.id))
          continue;
        const auto &constant = constants_.at(description.id);
        const auto bytes = constant.byte_size();
        const auto *data = constant.data();
        const auto stride = static_cast<std::size_t>(page_size);
        for (std::size_t offset = 0U; offset < bytes; offset += stride)
          checksum += data[offset];
        if (bytes != 0U)
          checksum += data[bytes - 1U];
      }
      if (convrot_resident_)
        for (const auto &plan : convrot_int8_linear_plans_) {
          for (const auto *tensor : {&plan.weight, &plan.scale}) {
            const auto bytes = tensor->byte_size();
            const auto *data = tensor->data();
            const auto stride = static_cast<std::size_t>(page_size);
            for (std::size_t offset = 0U; offset < bytes; offset += stride)
              checksum += data[offset];
            if (bytes != 0U)
              checksum += data[bytes - 1U];
          }
        }
      resident_prefault_checksum_ = checksum;
      resident_host_prefault_milliseconds_ =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - prefault_start)
              .count();
      if (getrusage(RUSAGE_SELF, &after) != 0)
        fail("getrusage after resident prefault failed");
      resident_minor_page_faults_ = static_cast<std::uint64_t>(
          std::max<long>(0L, after.ru_minflt - before.ru_minflt));
      resident_major_page_faults_ = static_cast<std::uint64_t>(
          std::max<long>(0L, after.ru_majflt - before.ru_majflt));
    }
    const auto resident_h2d_start = std::chrono::steady_clock::now();
    upload_resident_constants(program_, constants_, buffers_, context_.stream());
    constexpr std::size_t h3_resident_staging_slots = 2U;
    std::array<std::unique_ptr<PinnedHostWorkspace>,
               h3_resident_staging_slots>
        h3_resident_staging;
    std::array<std::unique_ptr<Event>, h3_resident_staging_slots>
        h3_resident_copy_done;
    std::array<bool, h3_resident_staging_slots> h3_resident_copy_recorded{};
    auto h3_resident_staging_bytes = std::uint64_t{0U};
    if (options.pipelined_resident_upload) {
      for (const auto &plan : h3_w8a8_attention_plans_)
        if (plan.resident)
          h3_resident_staging_bytes =
              std::max(h3_resident_staging_bytes,
                       plan.quantized_weight_bytes);
      for (const auto &plan : h3_w8a8_mlp_plans_)
        if (plan.resident)
          h3_resident_staging_bytes =
              std::max(h3_resident_staging_bytes,
                       plan.quantized_weight_bytes);
      if (h3_resident_staging_bytes != 0U) {
        if (h3_resident_staging_bytes >
            options.streamed_pinned_budget_bytes /
                h3_resident_staging_slots)
          fail("pipelined H3 resident upload exceeds the pinned budget: slots=" +
               std::to_string(h3_resident_staging_slots) +
               " buffer_bytes=" +
               std::to_string(h3_resident_staging_bytes) + " budget_bytes=" +
               std::to_string(options.streamed_pinned_budget_bytes));
        for (std::size_t slot = 0U; slot < h3_resident_staging_slots; ++slot) {
          h3_resident_staging[slot] = std::make_unique<PinnedHostWorkspace>(
              static_cast<std::size_t>(h3_resident_staging_bytes));
          h3_resident_copy_done[slot] =
              std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
        }
      }
    }
    auto h3_resident_turn = std::size_t{0U};
    const H3HostCopy h3_resident_host_copy =
        [&](std::uint8_t *destination, const Tensor &tensor) {
          // Direct IO only when the pages are cold: a warm mapping copies at
          // 2-4 s per checkpoint, a fresh disk read costs 8-9 s.
          if (h3_resident_direct_io_ &&
              tensor.mapped_resident_fraction() < 0.9 &&
              tensor.read_direct_into(destination)) {
            h3_resident_direct_read_bytes_ += tensor.byte_size();
            if (direct_io_warm_page_cache_)
              h3_resident_warm_list_.push_back(&tensor);
            return;
          }
          h3_w8a8_staging_pool_->copy(destination, tensor.data(),
                                      tensor.byte_size());
        };
    const auto upload_h3_resident = [&](auto &plan) {
      if (!plan.resident)
        return;
      if (options.lazy_resident_upload)
        return;
      if (h3_resident_staging_bytes == 0U) {
        upload_h3_w8a8_weights(plan, context_.stream());
        plan.uploaded = true;
        return;
      }
      const auto slot = h3_resident_turn % h3_resident_staging_slots;
      if (h3_resident_copy_recorded[slot])
        check(counted_event_synchronize(h3_resident_copy_done[slot]->get()),
              "cuEventSynchronize H3 resident staging reuse");
      (void)stage_h3_w8a8_weights(
          plan, h3_resident_staging[slot]->data(),
          h3_resident_staging_bytes, context_.stream(),
          h3_resident_host_copy);
      check(counted_event_record(h3_resident_copy_done[slot]->get(),
                                 context_.stream()),
            "cuEventRecord H3 resident staging copy");
      h3_resident_copy_recorded[slot] = true;
      plan.uploaded = true;
      ++h3_resident_turn;
    };
    // The cache is written attention then MLP for every layer. Merge the two
    // prepared plan lists by layer so a cold upload walks the file once in
    // physical order instead of seeking through all MLPs and then all
    // attention projections.
    h3_resident_readahead_bytes_ = options.h3_resident_readahead_bytes;
    h3_resident_direct_io_ = options.h3_resident_direct_io;
    direct_io_warm_page_cache_ = options.direct_io_warm_page_cache;
    h3_w8a8_upload_order_.clear();
    {
      auto attention_index = std::size_t{0U};
      auto mlp_index = std::size_t{0U};
      while (attention_index < h3_w8a8_attention_plans_.size() ||
             mlp_index < h3_w8a8_mlp_plans_.size()) {
        if (attention_index < h3_w8a8_attention_plans_.size() &&
            (mlp_index == h3_w8a8_mlp_plans_.size() ||
             h3_w8a8_attention_plans_[attention_index].layer <=
                 h3_w8a8_mlp_plans_[mlp_index].layer)) {
          auto &plan = h3_w8a8_attention_plans_[attention_index++];
          if (plan.resident) {
            plan.upload_order = h3_w8a8_upload_order_.size();
            h3_w8a8_upload_order_.emplace_back(true, attention_index - 1U);
          }
        } else {
          auto &plan = h3_w8a8_mlp_plans_[mlp_index++];
          if (plan.resident) {
            plan.upload_order = h3_w8a8_upload_order_.size();
            h3_w8a8_upload_order_.emplace_back(false, mlp_index - 1U);
          }
        }
      }
    }
    for (std::size_t order = 0U; order < h3_w8a8_upload_order_.size();
         ++order) {
      const auto [attention, slot] = h3_w8a8_upload_order_[order];
      if (!options.lazy_resident_upload)
        advise_h3_resident_readahead(order);
      if (attention)
        upload_h3_resident(h3_w8a8_attention_plans_[slot]);
      else
        upload_h3_resident(h3_w8a8_mlp_plans_[slot]);
    }
    if (options.lazy_resident_upload && !h3_w8a8_upload_order_.empty())
      advise_h3_resident_readahead(0U); // first window before evaluation 0
    for (const auto &plan : h3_groupwise_plans_)
      upload_h3_groupwise_weights(plan, context_.stream());
    upload_h3_modulation_cache(h3_modulation_cache_plans_, context_.stream());
    if (convrot_resident_) {
      // Cold mapped weights go through pinned staging with direct IO (the
      // mapping copy would fault them in at page-cache speed); warm ones are
      // copied from the mapping as before.
      std::unique_ptr<PinnedHostWorkspace> convrot_stage;
      std::unique_ptr<Event> convrot_stage_done;
      bool convrot_stage_armed = false;
      const auto upload_convrot = [&](CUdeviceptr device, const Tensor &tensor,
                                      const char *label) {
        if (!(h3_resident_direct_io_ &&
              tensor.mapped_resident_fraction() < 0.9)) {
          check(counted_memcpy_htod(device, tensor.data(), tensor.byte_size(),
                                    context_.stream()),
                label);
          return;
        }
        if (!convrot_stage || convrot_stage->size() < tensor.byte_size()) {
          if (convrot_stage_armed)
            check(counted_event_synchronize(convrot_stage_done->get()),
                  "cuEventSynchronize generic ConvRot staging reuse");
          convrot_stage_armed = false;
          convrot_stage = std::make_unique<PinnedHostWorkspace>(
              std::max<std::size_t>(tensor.byte_size(), 256U << 20U));
          if (!convrot_stage_done)
            convrot_stage_done = std::make_unique<Event>();
        } else if (convrot_stage_armed) {
          check(counted_event_synchronize(convrot_stage_done->get()),
                "cuEventSynchronize generic ConvRot staging reuse");
          convrot_stage_armed = false;
        }
        auto *staging = static_cast<std::uint8_t *>(convrot_stage->data());
        if (!tensor.read_direct_into(staging)) {
          check(counted_memcpy_htod(device, tensor.data(), tensor.byte_size(),
                                    context_.stream()),
                label);
          return;
        }
        h3_resident_direct_read_bytes_ += tensor.byte_size();
        if (direct_io_warm_page_cache_)
          h3_resident_warm_list_.push_back(&tensor);
        check(counted_memcpy_htod(device, staging, tensor.byte_size(),
                                  context_.stream()),
              label);
        check(counted_event_record(convrot_stage_done->get(), context_.stream()),
              "cuEventRecord generic ConvRot staging copy");
        convrot_stage_armed = true;
      };
      for (const auto &plan : convrot_int8_linear_plans_) {
        upload_convrot(plan.weight_device, plan.weight,
                       "cuMemcpyHtoDAsync resident generic ConvRot weight");
        upload_convrot(plan.scale_device, plan.scale,
                       "cuMemcpyHtoDAsync resident generic ConvRot scale");
      }
      if (convrot_stage_armed)
        check(counted_event_synchronize(convrot_stage_done->get()),
              "cuEventSynchronize generic ConvRot staging drain");
      // Prepare-time uploads: warm the cache now, the plan is ready.
      warm_h3_resident_pages();
    }
    if (options.lazy_resident_upload) {
      // Dedicated storage is populated at first semantic use by the prepared
      // prefetcher, then remains valid for the lifetime of this execution.
    } else if (!options.pipelined_resident_upload) {
      for (const auto id : promoted_streamed_constants_) {
        const auto &tensor = constants_.at(id);
        check(counted_memcpy_htod(buffers_.at(id), tensor.data(),
                                  tensor.byte_size(), context_.stream()),
              "cuMemcpyHtoDAsync promoted streamed constant");
      }
    } else if (!promoted_streamed_constants_.empty()) {
      constexpr std::size_t staging_slots = 2U;
      std::size_t maximum_bytes = 0U;
      for (const auto id : promoted_streamed_constants_)
        maximum_bytes =
            std::max(maximum_bytes, constants_.at(id).byte_size());
      if (maximum_bytes > options.streamed_pinned_budget_bytes / staging_slots)
        fail("pipelined resident upload exceeds the pinned budget: slots=" +
             std::to_string(staging_slots) + " buffer_bytes=" +
             std::to_string(maximum_bytes) + " budget_bytes=" +
             std::to_string(options.streamed_pinned_budget_bytes));
      std::array<std::unique_ptr<PinnedHostWorkspace>, staging_slots> staging;
      std::array<std::unique_ptr<Event>, staging_slots> copy_done;
      std::array<bool, staging_slots> recorded{};
      for (std::size_t slot = 0U; slot < staging_slots; ++slot) {
        staging[slot] = std::make_unique<PinnedHostWorkspace>(maximum_bytes);
        copy_done[slot] =
            std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
      }
      StagingPool staging_pool(options.streamed_stage_threads);
      for (std::size_t index = 0U;
           index < promoted_streamed_constants_.size(); ++index) {
        const auto slot = index % staging_slots;
        if (recorded[slot])
          check(counted_event_synchronize(copy_done[slot]->get()),
                "cuEventSynchronize resident staging reuse");
        const auto id = promoted_streamed_constants_[index];
        const auto &tensor = constants_.at(id);
        staging_pool.copy(
            static_cast<std::uint8_t *>(staging[slot]->data()), tensor.data(),
            tensor.byte_size());
        check(counted_memcpy_htod(buffers_.at(id), staging[slot]->data(),
                                  tensor.byte_size(), context_.stream()),
              "cuMemcpyHtoDAsync pipelined promoted constant");
        check(counted_event_record(copy_done[slot]->get(), context_.stream()),
              "cuEventRecord resident staging copy");
        recorded[slot] = true;
      }
    }
    check(counted_stream_synchronize(context_.stream()),
          "resident constant upload synchronization");
    if (convrot_resident_)
      for (auto &plan : convrot_int8_linear_plans_) {
        plan.weight.discard_mapped_pages();
        plan.scale.discard_mapped_pages();
      }
    if (resident_weight_bytes_ != 0U) {
      resident_h2d_milliseconds_ =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - resident_h2d_start)
              .count();
      resident_upload_milliseconds_ =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - resident_upload_start)
              .count();
    }
    if (!options.tune_linear_operations.empty()) {
      upload_dynamic_inputs(program_, bindings, buffers_, context_.stream());
      check(counted_stream_synchronize(context_.stream()),
            "Linear tuning input upload synchronization");
      std::unordered_set<std::uint32_t> tuned;
      for (const auto operation_id : options.tune_linear_operations) {
        if (!tuned.insert(operation_id).second)
          continue;
        const auto operation = std::find_if(
            program_.operations.begin(), program_.operations.end(),
            [&](const ir::Operation &value) {
              return value.id == operation_id;
            });
        bool cutlass_override = false;
#if DIF_HAS_CUTLASS
        cutlass_override = cutlass_linear_plans_.contains(operation_id);
#endif
        if (operation == program_.operations.end() ||
            operation->opcode != ir::Opcode::Linear ||
            !linear_plans_.contains(operation_id) || cutlass_override)
          fail("requested cuBLASLt tuning id is not an unfused Linear: " +
               std::to_string(operation_id));
        const auto absorbed_tuned = std::find_if(
            absorbed_linear_bias_plans_.begin(),
            absorbed_linear_bias_plans_.end(),
            [&](const AbsorbedLinearBiasPlan &plan) {
              return plan.linear_operation == operation_id;
            });
        linear_tuning_results_.push_back(linear_plans_.at(operation_id)->tune(
            operation_id,
            absorbed_tuned != absorbed_linear_bias_plans_.end()
                ? absorbed_tuned->launch_operation
                : *operation,
            buffers_, context_.cublas_lt(),
            *workspace_, context_.stream(), options.linear_tuning_warmups,
            options.linear_tuning_iterations,
            options.linear_tuning_sessions));
      }
    }
    // Fused plans execute at the slot of an operation whose own opcode no
    // longer describes the work; label those timings so consumers do not
    // classify a projection GEMM as a layout op.
    operation_plan_labels_.clear();
    for (const auto &plan : h3_w8a8_attention_plans_)
      if (plan.has_qkv_projection)
        operation_plan_labels_[plan.qkv_layout_operation] =
            "h3-int8-qkv-projection";
    // Host file-cache policy for the weights that are now GPU resident.
    // The bytes below are what a later fresh process would otherwise reread
    // from storage; keeping them is only admitted when the process' cgroup
    // memory limit can hold the charge (page cache is charged to the cgroup).
    resident_evict_host_pages_ = options.resident_evict_host_pages;
    if (!resident_evict_host_pages_) {
      std::uint64_t resident_host_bytes = 0U;
      for (const auto &description : program_.tensors)
        if (description.has_role(ir::TensorRole::Constant) &&
            !description.has_role(ir::TensorRole::Streamed))
          resident_host_bytes += constants_.at(description.id).byte_size();
      for (const auto id : promoted_streamed_constants_)
        resident_host_bytes += constants_.at(id).byte_size();
      for (const auto &plan : h3_w8a8_mlp_plans_)
        if (plan.resident)
          resident_host_bytes += plan.fc1_weight.byte_size() +
                                 plan.fc2_weight.byte_size();
      for (const auto &plan : h3_w8a8_attention_plans_)
        if (plan.resident)
          resident_host_bytes +=
              (plan.has_qkv_projection ? plan.qkv_weight.byte_size() : 0U) +
              (plan.has_output_projection ? plan.output_weight.byte_size()
                                          : 0U);
      const auto cgroup = probe_host_cgroup_memory();
      constexpr std::uint64_t margin = 512ULL * 1024ULL * 1024ULL;
      const bool admitted =
          cgroup.limit_bytes == 0U ||
          cgroup.limit_bytes >= cgroup.current_bytes + resident_host_bytes +
                                    margin;
      if (!admitted)
        resident_evict_host_pages_ = true;
      std::cerr << "RESIDENT_HOST_PAGES policy=keep decision="
                << (admitted ? "keep" : "evict")
                << " reason=" << (admitted ? "admitted" : "cgroup_limit")
                << " resident_host_bytes=" << resident_host_bytes
                << " cgroup_limit_bytes=" << cgroup.limit_bytes
                << " cgroup_current_bytes=" << cgroup.current_bytes << '\n';
    }
    for (const auto &description : program_.tensors) {
      if (description.has_role(ir::TensorRole::Constant) &&
          !description.has_role(ir::TensorRole::Streamed))
        release_resident_host_pages(constants_.at(description.id),
                                    resident_evict_host_pages_);
    }
    if (!options.lazy_resident_upload)
      for (const auto id : promoted_streamed_constants_)
        release_resident_host_pages(constants_.at(id),
                                    resident_evict_host_pages_);
    for (auto &plan : h3_w8a8_mlp_plans_) {
      if (!plan.uploaded)
        continue;
      evict_h3_w8a8_weights(plan, resident_evict_host_pages_);
    }
    for (auto &plan : h3_w8a8_attention_plans_) {
      if (!plan.uploaded)
        continue;
      evict_h3_w8a8_weights(plan, resident_evict_host_pages_);
    }
    for (auto &plan : h3_groupwise_plans_) {
      for (auto &projection : plan.projections) {
        release_resident_host_pages(projection.weight,
                                    resident_evict_host_pages_);
        release_resident_host_pages(projection.scale,
                                    resident_evict_host_pages_);
      }
    }
    for (auto &plan : h3_modulation_cache_plans_)
      release_resident_host_pages(plan.modulation,
                                  resident_evict_host_pages_);
    const auto preparation_stop = std::chrono::steady_clock::now();
    preparation_milliseconds_ =
        std::chrono::duration<double, std::milli>(preparation_stop -
                                                  preparation_start)
            .count();
    nvtx_pop();
    if (tracer_scope)
      preparation_trace_milliseconds_ = preparation_milliseconds_;
  }

  RunResult run(const TensorMap &inputs, const RunOptions &options) override {
    if (options.iterations == 0)
      fail("run iterations must be nonzero");
    if (options.streamed_keep_mapped_pages_between_runs &&
        options.streamed_release_mapped_pages_per_copy)
      fail("streamed mapped pages cannot be kept between runs and released per copy");
    if (options.lazy_resident_upload != lazy_resident_upload_)
      fail("lazy resident upload is fixed when the plan is prepared");
    auto requested_captures = options.capture_intermediate_tensors;
    std::sort(requested_captures.begin(), requested_captures.end());
    if (requested_captures != capture_intermediate_tensors_)
      fail("intermediate capture tensors are fixed when the plan is prepared");
    if (!capture_intermediate_tensors_.empty() &&
        (options.warmups != 0U || options.iterations != 1U))
      fail("intermediate capture requires zero warmups and one iteration");
    LaunchTelemetry run_telemetry;
    TelemetryScope telemetry_scope(run_telemetry);
    h3_exact_query_row_dispatches_ = 0U;
    Tracer run_tracer;
    std::optional<TracerScope> tracer_scope;
    const bool tracing = telemetry::trace_events_requested(options);
    if (tracing)
      tracer_scope.emplace(run_tracer);
    nvtx_enabled = telemetry::nvtx_ranges_requested(options);
    nvtx_push("dif::run");
    streamed_prefetcher_->set_release_mapped_pages_per_copy(
        options.streamed_release_mapped_pages_per_copy);
    if (options.streamed_prefetch_depth != streamed_prefetch_depth_)
      fail("streamed prefetch depth is fixed when the plan is prepared");
    // Constants were fully validated and frozen into constants_ during
    // preparation. Re-copying that map here is cheap only for mmap-backed
    // tensors; an owned packed/quantized checkpoint would deep-copy every
    // byte on every denoise step. The run path needs host tensors only for
    // dynamic inputs, while streamed and resident constants are consumed
    // directly from constants_ by their prepared upload plans.
    TensorMap bindings;
    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Input))
        continue;
      const auto found = inputs.find(desc.id);
      if (found == inputs.end())
        fail("missing CUDA dynamic input tensor " + std::to_string(desc.id));
      found->second.validate();
      if (found->second.dtype != desc.dtype ||
          found->second.dims != desc.dims)
        fail("CUDA dynamic input shape/dtype mismatch for id " +
             std::to_string(desc.id));
      bindings.emplace(desc.id, found->second);
    }
    validate_inputs(program_, bindings, constants_);
    auto repeated_cache_ready = repeated_invariant_valid_;
    if (repeated_cache_ready) {
      for (const auto input_id : repeated_invariant_input_tensors_) {
        const auto &actual = bindings.at(input_id);
        const auto found = repeated_invariant_input_snapshots_.find(input_id);
        if (found == repeated_invariant_input_snapshots_.end() ||
            found->second.size() != actual.byte_size() ||
            std::memcmp(found->second.data(), actual.data(),
                        actual.byte_size()) != 0) {
          repeated_cache_ready = false;
          break;
        }
      }
    }
    if (!repeated_cache_ready && !repeated_invariant_operations_.empty()) {
      repeated_invariant_input_snapshots_.clear();
      for (const auto input_id : repeated_invariant_input_tensors_) {
        const auto &actual = bindings.at(input_id);
        repeated_invariant_input_snapshots_.emplace(
            input_id,
            std::vector<std::uint8_t>(actual.data(),
                                      actual.data() + actual.byte_size()));
      }
    }
    if (!h3_modulation_cache_plans_.empty() &&
        !h3_modulation_input_path_.empty()) {
      const auto &actual = bindings.at(
          h3_modulation_cache_plans_.front().input_tensor);
      if (actual.byte_size() != h3_modulation_expected_input_.byte_size() ||
          std::memcmp(actual.data(), h3_modulation_expected_input_.data(),
                      actual.byte_size()) != 0)
        fail("H3 modulation cache input changed after preparation");
    }
    if (!h3_modulation_cache_plans_.empty()) {
      if (options.h3_modulation_slice >= h3_modulation_slices_)
        fail("H3 modulation slice is outside the prepared schedule");
      select_h3_modulation_slice(h3_modulation_cache_plans_,
                                 options.h3_modulation_slice, buffers_);
    }
    if (options.pinned_io_staging && !pinned_io_)
      fail("pinned I/O staging must be requested when the plan is prepared");
    if (options.pinned_io_staging) {
      auto *base = static_cast<std::uint8_t *>(pinned_io_->data());
      std::size_t offset = 0U;
      for (const auto &desc : program_.tensors) {
        if (!desc.has_role(ir::TensorRole::Input))
          continue;
        const auto &tensor = bindings.at(desc.id);
        std::memcpy(base + offset, tensor.data(), tensor.byte_size());
        check(counted_memcpy_htod(buffers_.at(desc.id), base + offset,
                                  tensor.byte_size(), context_.stream()),
              "cuMemcpyHtoDAsync pinned dynamic input");
        offset += tensor.byte_size();
      }
    } else {
      upload_dynamic_inputs(program_, bindings, buffers_, context_.stream());
    }
    check(counted_stream_synchronize(context_.stream()),
          "dynamic input upload synchronization");

    auto h3_w8a8_tail_streamed_bytes = std::uint64_t{0U};
    auto h3_w8a8_tail_host_stage_milliseconds = 0.0;
    // Per-run receipt counters (the prepare-time upload's bytes are reported
    // by the first run).
    if (h3_resident_counters_reported_) {
      h3_resident_direct_read_bytes_ = 0U;
      h3_resident_readahead_advised_bytes_ = 0U;
    }
    h3_resident_counters_reported_ = true;
    auto convrot_streamed_bytes = std::uint64_t{0U};
    auto convrot_host_stage_milliseconds = 0.0;
    const H3HostCopy h3_tail_host_copy =
        [&](std::uint8_t *destination, const Tensor &tensor) {
          // Direct IO only when the pages are cold: a warm mapping copies at
          // 2-4 s per checkpoint, a fresh disk read costs 8-9 s.
          if (h3_resident_direct_io_ &&
              tensor.mapped_resident_fraction() < 0.9 &&
              tensor.read_direct_into(destination)) {
            h3_resident_direct_read_bytes_ += tensor.byte_size();
            if (direct_io_warm_page_cache_)
              h3_resident_warm_list_.push_back(&tensor);
            return;
          }
          h3_w8a8_staging_pool_->copy(destination, tensor.data(),
                                      tensor.byte_size());
        };
    auto stage_h3_w8a8_tail = [&](auto &plan, bool profile) {
      const auto populate_resident = plan.resident && !plan.uploaded;
      if (plan.resident && !populate_resident)
        return;
      if (populate_resident && !lazy_resident_upload_)
        fail("H3 resident projection was not populated during preparation");
      if (!h3_w8a8_tail_stage_ || h3_w8a8_tail_stage_half_bytes_ == 0U)
        fail("H3 W8A8 tail plan lacks reusable host staging");
      const auto half = h3_w8a8_tail_stage_turn_ % 2U;
      if (h3_w8a8_tail_stage_armed_.at(half))
        check(counted_event_synchronize(h3_w8a8_tail_stage_events_.at(half)->get()),
              "cuEventSynchronize H3 W8A8 tail staging reuse");
      const auto tail_copy_stream =
          options.h3_w8a8_tail_uploads_on_copy_stream;
      const auto tail_stream =
          tail_copy_stream ? context_.copy_stream() : context_.stream();
      if (tail_copy_stream && !populate_resident) {
        // The reusable tail device storage is shared by every tail layer:
        // the copy stream must not overwrite it before the previous tail
        // layer's kernels (already submitted on the compute stream) have
        // consumed it. cuStreamWaitEvent snapshots the record at issue
        // time, so one event object per fence direction suffices.
        check(counted_event_record(h3_w8a8_tail_order_event_->get(),
                                   context_.stream()),
              "cuEventRecord H3 W8A8 tail upload order");
        check(counted_stream_wait_event(context_.copy_stream(),
                                        h3_w8a8_tail_order_event_->get(), 0U),
              "cuStreamWaitEvent H3 W8A8 tail upload order");
      }
      const auto stage_start = std::chrono::steady_clock::now();
      if (populate_resident)
        advise_h3_resident_readahead(plan.upload_order);
      auto *staging = static_cast<std::uint8_t *>(
          h3_w8a8_tail_stage_->data()) +
          half * h3_w8a8_tail_stage_half_bytes_;
      const auto bytes = stage_h3_w8a8_weights(
          plan, staging, h3_w8a8_tail_stage_half_bytes_, tail_stream,
          h3_tail_host_copy);
      if (populate_resident) {
        evict_h3_w8a8_weights(plan, resident_evict_host_pages_);
        plan.uploaded = true;
      }
      if (profile) {
        h3_w8a8_tail_streamed_bytes += bytes;
        h3_w8a8_tail_host_stage_milliseconds +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - stage_start)
                .count();
      }
      check(counted_event_record(h3_w8a8_tail_stage_events_.at(half)->get(),
                          tail_stream),
            "cuEventRecord H3 W8A8 tail staging copy");
      if (tail_copy_stream) {
        check(counted_event_record(h3_w8a8_tail_ready_event_->get(),
                                   context_.copy_stream()),
              "cuEventRecord H3 W8A8 tail weights ready");
        check(counted_stream_wait_event(context_.stream(),
                                        h3_w8a8_tail_ready_event_->get(), 0U),
              "cuStreamWaitEvent H3 W8A8 tail weights ready");
      }
      h3_w8a8_tail_stage_armed_.at(half) = true;
      ++h3_w8a8_tail_stage_turn_;
    };

    auto launch_convrot_linear = [&](ConvRotInt8LinearPlan &plan,
                                     const ir::Operation &operation,
                                     bool profile) {
#if DIF_HAS_CUTLASS
      if (!convrot_resident_ &&
          (!convrot_staging_ || !convrot_staging_pool_))
        fail("generic ConvRot Linear plan is not fully prepared");
      if (convrot_weight_only_quality_ &&
          (!h3_convrot_functions_.generic_bf16_rotate ||
           !h3_convrot_functions_.generic_weight_dequant ||
           !linear_plans_.contains(plan.operation)))
        fail("generic ConvRot weight-only quality plan is not fully prepared");
      if (!convrot_weight_only_quality_ &&
          (!h3_convrot_functions_.generic_encode ||
           (plan.dtype == ir::DType::BF16 &&
            !h3_int8_scaled_gemm_registry_) ||
           (plan.dtype == ir::DType::F16 &&
            !int8_scaled_f16_gemm_registry_)))
        fail("generic ConvRot fast plan is not fully prepared");
      const auto slot = static_cast<std::size_t>(convrot_turn_ % 2U);
      if (!convrot_resident_ && convrot_slot_armed_.at(slot))
        check(counted_event_synchronize(convrot_slot_done_.at(slot)->get()),
              "cuEventSynchronize generic ConvRot slot reuse");
      const auto slot_base = convrot_weight_storage_->pointer() +
                             slot * convrot_weight_slot_bytes_;
      auto weight_device = convrot_resident_ ? plan.weight_device : slot_base;
      auto scale_device =
          convrot_resident_
              ? plan.scale_device
              : slot_base + align_256(convrot_weight_bytes_);
      if (!convrot_resident_) {
        auto *host_base =
            static_cast<std::uint8_t *>(convrot_staging_->data()) +
            slot * convrot_weight_slot_bytes_;
        const auto stage_start = std::chrono::steady_clock::now();
        convrot_staging_pool_->copy(host_base, plan.weight.data(),
                                    plan.weight.byte_size());
        convrot_staging_pool_->copy(
            host_base + align_256(convrot_weight_bytes_), plan.scale.data(),
            plan.scale.byte_size());
        if (profile)
          convrot_host_stage_milliseconds +=
              std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - stage_start)
                  .count();
        check(counted_memcpy_htod(weight_device, host_base,
                                  plan.weight.byte_size(), context_.stream()),
              "cuMemcpyHtoDAsync generic ConvRot weight");
        check(counted_memcpy_htod(
                  scale_device,
                  host_base + align_256(convrot_weight_bytes_),
                  plan.scale.byte_size(), context_.stream()),
              "cuMemcpyHtoDAsync generic ConvRot scale");
      }
      constexpr unsigned threads = 512U;
      constexpr std::uint64_t temporary_floats = 8U * 2U * 256U;
      const auto shared_floats = convrot_weight_only_quality_
                                     ? temporary_floats
                                     : plan.contraction + temporary_floats;
      const auto shared_bytes = shared_floats * sizeof(float);
      if (shared_bytes > std::numeric_limits<unsigned>::max())
        fail("generic ConvRot launch shared-memory request overflows CUDA unsigned");
      const auto shared = static_cast<unsigned>(shared_bytes);
      auto row_start = 0;
      auto rows = static_cast<int>(plan.rows);
      auto contraction = static_cast<int>(plan.contraction);
      auto input = buffers_.at(plan.input_tensor);
      auto activation = convrot_activation_device_;
      if (convrot_weight_only_quality_) {
        std::array<void *, 5> rotate_arguments = {
            &input, &activation, &row_start, &rows, &contraction};
        check(counted_launch_kernel(
                  h3_convrot_functions_.generic_bf16_rotate,
                  static_cast<unsigned>(plan.rows), 1U, 1U, threads, 1U, 1U,
                  shared, context_.stream(), rotate_arguments.data(), nullptr),
              "cuLaunchKernel generic ConvRot BF16 activation rotate");
        auto dequantized_weight = convrot_quality_weight_device_;
        auto weight_rows = plan.columns;
        auto weight_columns = plan.contraction;
        const auto weight_elements = weight_rows * weight_columns;
        const auto dequant_grid = static_cast<unsigned>(
            std::min<std::uint64_t>(65535U,
                (weight_elements + 255U) / 256U));
        std::array<void *, 5> dequant_arguments = {
            &weight_device, &scale_device, &dequantized_weight,
            &weight_rows, &weight_columns};
        check(counted_launch_kernel(
                  h3_convrot_functions_.generic_weight_dequant,
                  dequant_grid, 1U, 1U, 256U, 1U, 1U, 0U,
                  context_.stream(), dequant_arguments.data(), nullptr),
              "cuLaunchKernel generic ConvRot BF16 weight dequant");
        const auto original_input = buffers_.at(plan.input_tensor);
        buffers_.at(plan.input_tensor) = convrot_activation_device_;
        buffers_.rebind_external(plan.weight_tensor,
                                 convrot_quality_weight_device_);
        linear_plans_.at(plan.operation)->launch(
            operation, buffers_, context_.cublas_lt(), *workspace_,
            context_.stream());
        buffers_.at(plan.input_tensor) = original_input;
      } else {
        auto activation_scale = convrot_activation_scale_device_;
        auto input_f16 = static_cast<int>(plan.dtype == ir::DType::F16);
        std::array<void *, 7> encode_arguments = {
            &input, &activation, &activation_scale, &row_start, &rows,
            &contraction, &input_f16};
        check(counted_launch_kernel(
                  h3_convrot_functions_.generic_encode,
                  static_cast<unsigned>(plan.rows), 1U, 1U, threads, 1U, 1U,
                  shared, context_.stream(), encode_arguments.data(), nullptr),
              "cuLaunchKernel generic ConvRot activation encode");
        const auto key = H3Int8GemmKey{
            static_cast<std::uint32_t>(plan.rows),
            static_cast<std::uint32_t>(plan.columns),
            static_cast<std::uint32_t>(plan.contraction)};
        if (plan.dtype == ir::DType::F16)
          int8_scaled_f16_gemm_registry_->launch(
              key, convrot_activation_device_, weight_device,
              convrot_activation_scale_device_, scale_device,
              plan.bias_tensor ? buffers_.at(plan.bias_tensor) : 0U,
              buffers_.at(plan.output_tensor), context_.stream());
        else
          h3_int8_scaled_gemm_registry_->launch(
              key, convrot_activation_device_, weight_device,
              convrot_activation_scale_device_, scale_device,
              buffers_.at(plan.output_tensor), context_.stream());
      }
      if (!convrot_resident_) {
        check(counted_event_record(convrot_slot_done_.at(slot)->get(),
                                   context_.stream()),
              "cuEventRecord generic ConvRot slot completion");
        convrot_slot_armed_.at(slot) = true;
        ++convrot_turn_;
        if (profile)
          convrot_streamed_bytes +=
              plan.weight.byte_size() + plan.scale.byte_size();
        plan.weight.discard_mapped_pages();
        plan.scale.discard_mapped_pages();
      }
#else
      (void)plan;
      (void)profile;
      fail("generic ConvRot Linear requires CUTLASS");
#endif
    };

    auto execute_operation = [&](const ir::Operation &op, bool profile) {
      if (skipped_operations_.contains(op.id))
        return;
      TraceOperationScope operation_scope(op);
      const auto h3_groupwise_qkv = std::find_if(
          h3_groupwise_plans_.begin(), h3_groupwise_plans_.end(),
          [&](const H3GroupwiseBlockPlan &plan) {
            return plan.qkv_layout_operation == op.id;
          });
      if (h3_groupwise_qkv != h3_groupwise_plans_.end())
        launch_h3_groupwise_dequant(
            h3_groupwise_qkv->projections.at(0),
            h3_groupwise_scratch_storage_->pointer(), false,
            h3_groupwise_dequant_function_, context_.stream());
      else if (const auto h3_groupwise_linear = std::find_if(
                   h3_groupwise_plans_.begin(), h3_groupwise_plans_.end(),
                   [&](const H3GroupwiseBlockPlan &plan) {
                     return plan.output_linear_operation == op.id ||
                            plan.fc1_operation == op.id ||
                            plan.fc2_operation == op.id;
                   });
               h3_groupwise_linear != h3_groupwise_plans_.end()) {
        std::size_t projection = 1U;
        bool swap_fc1 = false;
        if (h3_groupwise_linear->fc1_operation == op.id) {
          projection = 2U;
          swap_fc1 = true;
        } else if (h3_groupwise_linear->fc2_operation == op.id) {
          projection = 3U;
        }
        launch_h3_groupwise_dequant(
            h3_groupwise_linear->projections.at(projection),
            h3_groupwise_scratch_storage_->pointer(), swap_fc1,
            h3_groupwise_dequant_function_, context_.stream());
        linear_plans_.at(op.id)->launch(op, buffers_, context_.cublas_lt(),
                                        *workspace_, context_.stream());
      } else {
      const auto h3_w8a8_qkv = std::find_if(
          h3_w8a8_attention_plans_.begin(),
          h3_w8a8_attention_plans_.end(),
          [&](const H3W8A8AttentionPlan &plan) {
            return plan.qkv_layout_operation == op.id;
          });
      if (h3_w8a8_qkv != h3_w8a8_attention_plans_.end()) {
        stage_h3_w8a8_tail(*h3_w8a8_qkv, profile);
        launch_h3_w8a8_qkv(*h3_w8a8_qkv, h3_w8a8_functions_,
                            h3_convrot_functions_, buffers_,
                            h3_convrot_bf16_correction_.get(),
                            context_.cublas(),
#if DIF_HAS_CUTLASS
                            h3_int8_gemm_registry_.get(),
                            h3_int8_scaled_gemm_registry_.get(),
#else
                            h3_int8_gemm_registry_.get(),
#endif
                            workspace_.get(),
                            context_.stream());
      }
      else if (const auto h3_w8a8_output = std::find_if(
                   h3_w8a8_attention_plans_.begin(),
                   h3_w8a8_attention_plans_.end(),
                   [&](const H3W8A8AttentionPlan &plan) {
                     return plan.output_linear_operation == op.id;
                   });
               h3_w8a8_output != h3_w8a8_attention_plans_.end()) {
        if (!h3_w8a8_output->has_qkv_projection)
          stage_h3_w8a8_tail(*h3_w8a8_output, profile);
        launch_h3_w8a8_output(*h3_w8a8_output, h3_w8a8_functions_,
                              h3_convrot_functions_, buffers_,
                              h3_convrot_bf16_correction_.get(),
                              context_.cublas(), h3_int8_gemm_registry_.get(),
#if DIF_HAS_CUTLASS
                              h3_int8_scaled_gemm_registry_.get(),
#endif
                              workspace_.get(), context_.stream());
      }
      else if (const auto h3_w8a8_mlp = std::find_if(
          h3_w8a8_mlp_plans_.begin(), h3_w8a8_mlp_plans_.end(),
          [&](const H3W8A8MlpPlan &plan) {
            return plan.fc1_operation == op.id;
          });
               h3_w8a8_mlp != h3_w8a8_mlp_plans_.end()) {
        stage_h3_w8a8_tail(*h3_w8a8_mlp, profile);
        launch_h3_w8a8_mlp(*h3_w8a8_mlp, h3_w8a8_functions_,
                            h3_convrot_functions_, buffers_,
                            h3_convrot_bf16_correction_.get(),
                            context_.cublas(),
#if DIF_HAS_CUTLASS
                            h3_int8_gemm_registry_.get(),
                            h3_int8_scaled_gemm_registry_.get(), workspace_.get(),
#else
                            h3_int8_gemm_registry_.get(), workspace_.get(),
#endif
                            context_.stream());
      }
      else if (const auto convrot = std::find_if(
          convrot_int8_linear_plans_.begin(),
          convrot_int8_linear_plans_.end(),
          [&](const ConvRotInt8LinearPlan &plan) {
            return plan.operation == op.id;
          }); convrot != convrot_int8_linear_plans_.end())
        launch_convrot_linear(*convrot, op, profile);
      else if (const auto fused_swiglu = std::find_if(
          fused_linear_swiglu_plans_.begin(),
          fused_linear_swiglu_plans_.end(),
          [&](const FusedLinearSwiGluPlan &plan) {
            return plan.linear_operation == op.id;
          });
               fused_swiglu != fused_linear_swiglu_plans_.end())
        launch_fused_linear_swiglu(*fused_swiglu, buffers_, context_.stream());
      else if (const auto fused = fused_launch_inputs_.find(op.id);
               fused != fused_launch_inputs_.end())
        launch(program_, op, functions_.at(op.id), buffers_, context_.stream(),
               &fused->second);
      else if (const auto scaled = fp8_scaled_linear_plans_.find(op.id);
               scaled != fp8_scaled_linear_plans_.end()) {
        scaled->second->launch(op, buffers_, context_.cublas_lt(), *workspace_,
                               context_.stream());
        launch(program_, op, functions_.at(op.id), buffers_,
               context_.stream());
      }
#if DIF_HAS_CUBLASLT_BLOCK_SCALE
      else if (const auto scaled = fp8_block_scaled_linear_plans_.find(op.id);
               scaled != fp8_block_scaled_linear_plans_.end())
        scaled->second->launch(op, buffers_, context_.cublas_lt(), *workspace_,
                               context_.stream());
#endif
#if DIF_HAS_CUTLASS
      else if (const auto scaled = int8_scaled_linear_plans_.find(op.id);
               scaled != int8_scaled_linear_plans_.end())
        scaled->second->launch(op, buffers_, context_.stream());
      else if (const auto weight_only = int8_weight_linear_plans_.find(op.id);
               weight_only != int8_weight_linear_plans_.end())
        weight_only->second->launch(op, buffers_, context_.stream());
      else if (const auto cutlass = cutlass_linear_plans_.find(op.id);
               cutlass != cutlass_linear_plans_.end()) {
        count_cutlass_launch();
        cutlass->second->launch(context_.stream());
      }
#endif
      else if (op.opcode == ir::Opcode::Linear) {
        const auto absorbed = std::find_if(
            absorbed_linear_bias_plans_.begin(),
            absorbed_linear_bias_plans_.end(),
            [&](const AbsorbedLinearBiasPlan &plan) {
              return plan.linear_operation == op.id;
            });
        linear_plans_.at(op.id)->launch(
            absorbed != absorbed_linear_bias_plans_.end()
                ? absorbed->launch_operation
                : op,
            buffers_, context_.cublas_lt(), *workspace_, context_.stream());
      } else if (op.opcode == ir::Opcode::Attention &&
               h3_ck_attention_plans_.contains(op.id) &&
               (options.h3_int8_attention_active ||
                !h3_hybrid_member(op.id) ||
                (!h3_int8_attention_hybrid_ &&
                 [&] {
                   fail("exact attention for this run requires "
                        "h3_int8_attention_hybrid at prepare");
                   return false;
                 }()))) {
        count_ck_attention_dispatch();
        h3_ck_attention_plans_.at(op.id)->execute(op, buffers_,
                                                  context_.stream());
#if DIF_HAS_CUDNN
        if (!h3_exact_query_range_plans_.empty()) {
          const auto *query = program_.tensor(op.inputs.at(0));
          const auto row_bytes =
              query->dims.at(1) * query->dims.at(2) *
              static_cast<std::uint64_t>(ir::dtype_size(query->dtype));
          for (std::size_t index = 0U; index < h3_exact_query_ranges_.size();
               ++index) {
            const auto offset = h3_exact_query_ranges_[index].begin * row_bytes;
            count_cudnn_attention_dispatch();
            ++h3_exact_query_row_dispatches_;
            h3_exact_query_range_plans_[index]->execute(
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))) +
                    offset,
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2))),
                0U,
                static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))) +
                    offset,
                reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
                reinterpret_cast<std::uintptr_t>(context_.stream()));
          }
        }
#endif
      } else if (op.opcode == ir::Opcode::Attention &&
                 op.u64(ir::AttrKey::Implementation, 1U) == 3U) {
        materialized_f32_attention_plans_.at(op.id).execute(
            op, buffers_, context_.cublas(),
            materialized_f32_attention_softmax_,
            materialized_f32_attention_scores_->pointer(), context_.stream());
      }
#if DIF_HAS_FLASH_ATTENTION
      else if (op.opcode == ir::Opcode::Attention &&
               op.u64(ir::AttrKey::Implementation, 1U) == 4U) {
        const auto *query = program_.tensor(op.inputs.at(0));
        const auto batched = query->dims.size() == 4U;
        const auto batch = static_cast<std::uint32_t>(
            batched ? query->dims.at(0) : 1U);
        const auto sequence = static_cast<std::uint32_t>(
            query->dims.at(batched ? 1U : 0U));
        const auto heads = static_cast<std::uint32_t>(
            query->dims.at(batched ? 2U : 1U));
        const auto head_dimension =
            static_cast<std::uint32_t>(query->dims.back());
        const auto key_value_heads = static_cast<std::uint32_t>(
            op.u64(ir::AttrKey::KvHeads, heads));
        const auto scale = static_cast<float>(op.f64(
            ir::AttrKey::AttentionScale,
            1.0 / std::sqrt(static_cast<double>(head_dimension))));
        if (const auto *error = flash_attention_bf16_forward(
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
                static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2))),
                static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
                reinterpret_cast<std::uintptr_t>(workspace_->data()), batch,
                sequence, heads, key_value_heads, head_dimension, scale,
                reinterpret_cast<std::uintptr_t>(context_.stream())))
          fail(std::string("native FlashAttention execution failed: ") + error);
      }
#endif
#if DIF_HAS_CUDNN
      else if (op.opcode == ir::Opcode::Conv2d) {
        count_cudnn_convolution_dispatch();
        cudnn_conv_plans_.at(op.id)->execute(
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
            op.inputs.size() == 3U
                ? static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2)))
                : 0U,
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
            reinterpret_cast<std::uintptr_t>(context_.stream()));
      }
      else if (op.opcode == ir::Opcode::Conv3d) {
        count_cudnn_convolution_dispatch();
        cudnn_conv3d_plans_.at(op.id)->execute(
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
            op.inputs.size() == 3U
                ? static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2)))
                : 0U,
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
            reinterpret_cast<std::uintptr_t>(context_.stream()));
      }
      else if (op.opcode == ir::Opcode::AttentionBackward &&
               op.u64(ir::AttrKey::Implementation, 1U) == 2U) {
        count_cudnn_attention_dispatch();
        // Transpose the program's F32 [S,H] logsumexp into the packed [H,S]
        // stats cuDNN reads: C(SxH, column-major, ld S) = A^T with A the
        // HxS column-major view of the row-major [S,H] buffer (ld H).
        const auto *query = program_.tensor(op.inputs.at(1));
        const auto stats_sequence = static_cast<int>(query->dims.at(0));
        const auto stats_heads = static_cast<int>(query->dims.at(1));
        const auto &stats = cudnn_backward_stats_.at(op.id);
        const float one = 1.0f;
        const float zero = 0.0f;
        const auto *lse =
            reinterpret_cast<const float *>(buffers_.at(op.inputs.at(5)));
        auto *packed = reinterpret_cast<float *>(stats->pointer());
        if (cublasSgeam(context_.cublas(), CUBLAS_OP_T, CUBLAS_OP_N,
                        stats_sequence, stats_heads, &one, lse, stats_heads,
                        &zero, packed, stats_sequence, packed,
                        stats_sequence) != CUBLAS_STATUS_SUCCESS)
          fail("cublasSgeam attention backward stats transpose failed");
        cudnn_attention_backward_plans_.at(op.id)->execute(
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(3))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(4))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(stats->pointer()),
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(1))),
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(2))),
            reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
            reinterpret_cast<std::uintptr_t>(context_.stream()));
      }
      else if (op.opcode == ir::Opcode::Attention &&
               op.u64(ir::AttrKey::Implementation, 1U) == 2U) {
        count_cudnn_attention_dispatch();
        cudnn_attention_plans_.at(op.id)->execute(
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2))),
            op.inputs.size() == 4U
                ? static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(3)))
                : 0U,
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
            reinterpret_cast<std::uintptr_t>(context_.stream()));
      }
#endif
      else if (op.opcode == ir::Opcode::Barrier)
        launch(program_, op, nullptr, buffers_, context_.stream());
      else
        launch(program_, op, functions_.at(op.id), buffers_, context_.stream());
      }
    };
    struct OperationEventPair {
      std::unique_ptr<Event> start;
      std::unique_ptr<Event> stop;
    };
    std::vector<OperationEventPair> profile_operation_events;
    std::vector<bool> profile_reused_invariant_operations;
    if (options.profile_pipeline) {
      const auto count = program_.operations.size() *
                         static_cast<std::size_t>(options.iterations);
      profile_operation_events.reserve(count);
      profile_reused_invariant_operations.assign(count, false);
      for (std::size_t index = 0U; index < count; ++index)
        profile_operation_events.push_back(
            {std::make_unique<Event>(), std::make_unique<Event>()});
    }
    TensorMap captured_intermediates;
    const auto capture_intermediate_outputs =
        [&](std::uint32_t operation_id, CUstream stream) {
          const auto outputs = captured_tensors_by_producer_.find(operation_id);
          if (outputs == captured_tensors_by_producer_.end())
            return;
          for (const auto tensor_id : outputs->second) {
            const auto *description = program_.tensor(tensor_id);
            if (!description)
              fail("captured intermediate tensor is missing");
            auto tensor = zeros(*description);
            check(counted_memcpy_dtoh(tensor.mutable_data(),
                                      buffers_.at(tensor_id),
                                      tensor.byte_size(), stream),
                  "cuMemcpyDtoHAsync captured intermediate");
            if (!captured_intermediates.emplace(tensor_id, std::move(tensor))
                     .second)
              fail("captured intermediate tensor was produced more than once");
          }
        };
    const auto preserve_repeated_invariant_outputs =
        [&](std::uint32_t operation_id, CUstream stream) {
          const auto outputs =
              repeated_invariant_outputs_by_producer_.find(operation_id);
          if (outputs == repeated_invariant_outputs_by_producer_.end())
            return;
          if (!repeated_invariant_cache_storage_ ||
              repeated_invariant_cache_storage_->pointer() == 0U)
            fail("repeated-invariant output cache is not allocated");
          for (const auto tensor_id : outputs->second) {
            const auto *description = program_.tensor(tensor_id);
            if (!description)
              fail("repeated-invariant output tensor is missing");
            const auto offset = repeated_invariant_cache_offsets_.at(tensor_id);
            check(counted_memcpy_dtod(
                      repeated_invariant_cache_storage_->pointer() + offset,
                      buffers_.at(tensor_id), description->byte_count(), stream),
                  "cuMemcpyDtoDAsync preserve repeated-invariant output");
          }
        };
    auto execute_parallel_group = [&](std::size_t first,
                                      std::uint32_t iteration,
                                      bool profile,
                                      const std::vector<bool> *prefetched,
                                      bool preserve_invariants) {
      const auto found = parallel_linear_groups_.find(first);
      if (found == parallel_linear_groups_.end())
        return false;
      const auto &indices = found->second;
      std::vector<bool> ready(indices.size(), false);
      for (std::size_t lane = 1U; lane < indices.size(); ++lane) {
        const auto operation_index = indices[lane];
        ready[lane] = prefetched ? prefetched->at(operation_index) : false;
        if (!ready[lane])
          ready[lane] = streamed_prefetcher_->prefetch(operation_index);
      }
      check(counted_event_record(parallel_start_event_->get(),
                                 context_.stream()),
            "cuEventRecord parallel Linear start");
      for (std::size_t lane = 0U; lane < indices.size(); ++lane) {
        const auto operation_index = indices[lane];
        const auto &operation = program_.operations[operation_index];
        const auto stream = lane == 0U
                                ? context_.stream()
                                : parallel_streams_.at(lane - 1U)->get();
        const auto &workspace =
            lane == 0U ? *workspace_ : *parallel_workspaces_.at(lane - 1U);
        if (lane != 0U)
          streamed_prefetcher_->wait(operation_index, ready[lane], stream);
        if (lane != 0U)
          check(counted_stream_wait_event(stream, parallel_start_event_->get(),
                                          0U),
                "cuStreamWaitEvent parallel Linear input");
        OperationEventPair *timing = nullptr;
        if (profile) {
          timing = &profile_operation_events.at(
              static_cast<std::size_t>(iteration) *
                      program_.operations.size() +
              operation_index);
          check(counted_event_record(timing->start->get(), stream),
                "cuEventRecord profiled parallel Linear start");
        }
        linear_plans_.at(operation.id)->launch(
            operation, buffers_, context_.cublas_lt(), workspace, stream);
        capture_intermediate_outputs(operation.id, stream);
        if (preserve_invariants)
          preserve_repeated_invariant_outputs(operation.id, stream);
        if (timing)
          check(counted_event_record(timing->stop->get(), stream),
                "cuEventRecord profiled parallel Linear stop");
        streamed_prefetcher_->complete(operation_index, stream);
        if (lane != 0U)
          check(counted_event_record(parallel_done_events_.at(lane - 1U)->get(),
                                     stream),
                "cuEventRecord parallel Linear completion");
      }
      for (std::size_t lane = 1U; lane < indices.size(); ++lane)
        check(counted_stream_wait_event(
                  context_.stream(), parallel_done_events_.at(lane - 1U)->get(),
                  0U),
              "cuStreamWaitEvent parallel Linear completion");
      return true;
    };
    auto execute = [&](std::uint32_t iteration, bool profile,
                       bool reuse_invariants) {
      if (program_.operations.empty())
        return;
      if (reuse_invariants) {
        if (!repeated_invariant_cache_storage_ ||
            repeated_invariant_cache_storage_->pointer() == 0U)
          fail("repeated-invariant output cache is not allocated");
        for (const auto &[tensor_id, offset] :
             repeated_invariant_cache_offsets_) {
          const auto *description = program_.tensor(tensor_id);
          if (!description)
            fail("repeated-invariant output tensor is missing");
          check(counted_memcpy_dtod(
                    buffers_.at(tensor_id),
                    repeated_invariant_cache_storage_->pointer() + offset,
                    description->byte_count(), context_.stream()),
                "cuMemcpyDtoDAsync restore repeated-invariant output");
        }
      }
      const auto reused = [&](std::size_t index) {
        return reuse_invariants && repeated_invariant_operations_.contains(
                                       program_.operations.at(index).id);
      };
      const auto mark_reused = [&](std::size_t index) {
        if (profile)
          profile_reused_invariant_operations.at(
              static_cast<std::size_t>(iteration) *
                  program_.operations.size() +
              index) = true;
      };
      const auto prefetch = [&](std::size_t index) {
        return !reused(index) && streamed_prefetcher_->prefetch(index);
      };
      if (!options.overlap_streaming) {
        for (std::size_t index = 0; index < program_.operations.size(); ++index) {
          if (reused(index)) {
            mark_reused(index);
            capture_intermediate_outputs(program_.operations.at(index).id,
                                         context_.stream());
            // Preserve the semantic operation timeline even when its pure
            // value is restored from the repeated-invariant cache. Streamed
            // slot overwrite planning names the latest prior operation by
            // index; without a current-iteration completion record, it can
            // wait on that operation's stale event from the previous
            // evaluation and overwrite a lower-index live tenant.
            streamed_prefetcher_->complete(index);
            continue;
          }
          if (parallel_linear_followups_.contains(index))
            continue;
          const auto &op = program_.operations[index];
          const auto ready = prefetch(index);
          streamed_prefetcher_->wait(index, ready);
          if (execute_parallel_group(index, iteration, profile, nullptr,
                                     !reuse_invariants))
            continue;
          OperationEventPair *timing = nullptr;
          if (profile && !skipped_operations_.contains(op.id)) {
            timing = &profile_operation_events.at(
                static_cast<std::size_t>(iteration) *
                    program_.operations.size() +
                index);
            check(counted_event_record(timing->start->get(), context_.stream()),
                  "cuEventRecord profiled operation start");
          }
          execute_operation(op, profile);
          capture_intermediate_outputs(op.id, context_.stream());
          if (!reuse_invariants)
            preserve_repeated_invariant_outputs(op.id, context_.stream());
          if (timing)
            check(counted_event_record(timing->stop->get(), context_.stream()),
                  "cuEventRecord profiled operation stop");
          streamed_prefetcher_->complete(index);
        }
        return;
      }
      const auto operation_count = program_.operations.size();
      const auto depth = static_cast<std::size_t>(streamed_prefetch_depth_);
      std::vector<bool> prefetched(operation_count, false);
      for (std::size_t ahead = 0U; ahead < depth && ahead < operation_count;
           ++ahead)
        prefetched[ahead] = prefetch(ahead);
      for (std::size_t index = 0; index < operation_count; ++index) {
        if (reused(index)) {
          mark_reused(index);
          capture_intermediate_outputs(program_.operations.at(index).id,
                                       context_.stream());
          // See the serial path above: skipped semantic positions still need
          // current-iteration completion fences for streamed-slot safety.
          streamed_prefetcher_->complete(index);
          if (index + depth < operation_count)
            prefetched[index + depth] = prefetch(index + depth);
          continue;
        }
        if (parallel_linear_followups_.contains(index)) {
          if (index + depth < operation_count)
            prefetched[index + depth] = prefetch(index + depth);
          continue;
        }
        const auto &op = program_.operations[index];
        streamed_prefetcher_->wait(index, prefetched[index]);
        if (execute_parallel_group(index, iteration, profile, &prefetched,
                                   !reuse_invariants)) {
          continue;
        }
        OperationEventPair *timing = nullptr;
        if (profile && !skipped_operations_.contains(op.id)) {
          timing = &profile_operation_events.at(
              static_cast<std::size_t>(iteration) *
                  program_.operations.size() +
              index);
          check(counted_event_record(timing->start->get(), context_.stream()),
                "cuEventRecord profiled operation start");
        }
        execute_operation(op, profile);
        capture_intermediate_outputs(op.id, context_.stream());
        if (!reuse_invariants)
          preserve_repeated_invariant_outputs(op.id, context_.stream());
        if (timing)
          check(counted_event_record(timing->stop->get(), context_.stream()),
                "cuEventRecord profiled operation stop");
        streamed_prefetcher_->complete(index);
        // The completion record above is what makes prefetching depth
        // operations ahead safe: the memory plan widened every streamed
        // interval by the same depth, so the overwrite-wait target of
        // operation index+depth is at most index and its completion event
        // is guaranteed recorded before the copy stream is asked to wait
        // on it.
        if (index + depth < operation_count)
          prefetched[index + depth] = prefetch(index + depth);
      }
    };

    for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
      execute(0U, false, repeated_cache_ready);
      if (!repeated_invariant_operations_.empty())
        repeated_cache_ready = true;
      streamed_prefetcher_->complete_iteration();
      check(counted_stream_synchronize(context_.stream()), "warmup synchronization");
    }
    const auto repeated_invariant_executions =
        repeated_invariant_operations_.empty() || repeated_cache_ready
            ? 0U
            : 1U;
    if (options.profile_pipeline) {
      auto profiled_reused_operations = repeated_invariant_operations_;
      streamed_prefetcher_->begin_profile(
          options.iterations, profiled_reused_operations,
          repeated_invariant_executions);
    }

    RunResult result;
    result.target_profile = target_profile_;
    result.runtime_budget = runtime_budget_;
    result.preparation_milliseconds = preparation_milliseconds_;
    result.resident_bytes = resident_bytes_;
    result.free_bytes_before = free_bytes_before_;
    result.backend_name = name();
    result.device_name = device_name_;
    result.generated_source_hash = source_hash_;
    result.repeated_invariant_operation_count =
        static_cast<std::uint32_t>(repeated_invariant_operations_.size());
    result.repeated_invariant_persistent_bytes =
        repeated_invariant_persistent_bytes_;
    result.linear_tuning_results = linear_tuning_results_;
    result.selected_linear_algorithms = selected_linear_algorithms_;
    result.primitive_fusions.reserve(fused_linear_swiglu_plans_.size());
    for (const auto &fusion : fused_linear_swiglu_plans_)
      result.primitive_fusions.push_back(
          {fusion.linear_operation, fusion.swiglu_operation,
           program_.tensor(fusion.intermediate_tensor)->byte_count(),
           "wmma_bf16_fc1_swiglu"});
    result.linear_bias_fusions.reserve(absorbed_linear_bias_plans_.size());
    for (const auto &plan : absorbed_linear_bias_plans_)
      result.linear_bias_fusions.push_back(
          {plan.linear_operation, plan.bias_operation,
           plan.eliminated_intermediate_bytes, "cublaslt-bias-epilogue"});
    result.linear_heuristic_cache = linear_heuristic_cache_stats_;
#if DIF_HAS_CUTLASS
    result.gemm_primitives.reserve(cutlass_linear_plans_.size());
    for (const auto &[operation_id, plan] : cutlass_linear_plans_) {
      (void)operation_id;
      result.gemm_primitives.push_back(plan->result());
    }
#endif
    result.h3_w8a8_mlps.reserve(h3_w8a8_mlp_plans_.size());
    for (const auto &plan : h3_w8a8_mlp_plans_)
      result.h3_w8a8_mlps.push_back(
          {plan.fc1_operation,
           plan.swiglu_operation,
           plan.fc2_operation,
           plan.residual_operation,
           plan.layer,
           plan.chunk_rows,
           plan.quantized_weight_bytes,
           plan.scratch_bytes,
           plan.eliminated_intermediate_bytes,
           plan.convrot ? "approximate_native_h256_convrot_int8_gate"
                        : "approximate_w8a8_established_h3_gate",
           plan.convrot_scale_chunk != 0U
               ? "native_h256_convrot_int8_chunk_scaled_f32_accumulation_chunked_mlp_residual"
               : plan.cutlass_scaled_fc2
               ? "native_h256_convrot_int8_cutlass_scaled_all_compact_adaln_chunked_mlp_residual"
               : plan.cutlass_scaled_fc1
               ? "native_h256_convrot_int8_cutlass_scaled_fc1_chunked_mlp_residual"
               : plan.compact_adaln.enabled
               ? "native_h256_convrot_int8_compact_adaln_chunked_mlp_residual"
               : (plan.convrot
                      ? "native_h256_convrot_int8_chunked_mlp_residual"
                      : "serenity_h3_w8a8_chunked_mlp_residual"),
           plan.cache_path.string(),
           plan.resident});
    result.h3_w8a8_attentions.reserve(h3_w8a8_attention_plans_.size());
    for (const auto &plan : h3_w8a8_attention_plans_)
      result.h3_w8a8_attentions.push_back(
          {plan.qkv_layout_operation,
           plan.qkv_linear_operations,
           plan.output_linear_operation,
           plan.residual_operation,
           plan.layer,
           kH3W8A8ProjectionChunkRows,
           plan.quantized_weight_bytes,
           plan.scratch_bytes,
           plan.eliminated_intermediate_bytes,
           plan.convrot ? "approximate_native_h256_convrot_int8_gate"
                        : "approximate_w8a8_established_h3_gate",
           plan.convrot_scale_chunk != 0U
               ? "native_h256_convrot_int8_chunk_scaled_f32_accumulation_direct_qkv_output_residual"
               : plan.cutlass_scaled
               ? "native_h256_convrot_int8_cutlass_scaled_all_compact_adaln_direct_qkv_output_residual"
               : plan.compact_adaln.enabled
               ? "native_h256_convrot_int8_compact_adaln_direct_qkv_output_residual"
               : plan.convrot
               ? "native_h256_convrot_int8_direct_qkv_output_residual"
               : "serenity_h3_w8a8_direct_qkv_output_residual",
           plan.cache_path.string(),
           plan.resident});
    result.h3_ck_attentions.reserve(h3_ck_attention_plans_.size());
    for (const auto &[operation_id, plan] : h3_ck_attention_plans_)
      result.h3_ck_attentions.push_back(
          {operation_id,
           static_cast<std::uint32_t>(plan->target_sm()),
           plan->scratch_bytes(),
           plan->classification(),
           plan->implementation(),
           plan->path().string()});
    result.h3_int8_attention_hybrid_first_layer =
        h3_int8_attention_hybrid_first_layer_;
    result.h3_int8_attention_hybrid_layers = h3_int8_attention_hybrid_layers_;
    result.h3_int8_attention_exact_query_ranges = h3_exact_query_ranges_;
    result.h3_groupwise_int8.reserve(h3_groupwise_plans_.size());
    for (const auto &plan : h3_groupwise_plans_)
      result.h3_groupwise_int8.push_back(
          {plan.qkv_layout_operation,
           plan.output_linear_operation,
           plan.fc1_operation,
           plan.fc2_operation,
           plan.layer,
           {plan.projections.at(0).group_size,
            plan.projections.at(1).group_size,
            plan.projections.at(2).group_size,
            plan.projections.at(3).group_size},
           plan.quantized_weight_bytes,
           plan.scratch_bytes,
           "approximate_groupwise_int8_established_h3_gate",
           "serenity_h3_projection_local_groupwise_dequant_bf16",
           plan.cache_path.string()});
    result.h3_modulation_caches.reserve(h3_modulation_cache_plans_.size());
    for (const auto &plan : h3_modulation_cache_plans_)
      result.h3_modulation_caches.push_back(
          {plan.linear_operation,
           plan.select_operation,
           plan.layer,
           plan.modulation.byte_size(),
           plan.replaced_weight_bytes,
           "exact_precomputed_bf16_established_h3_gate",
           "serenity_h3_adaln_modulation_cache",
           plan.cache_path.string(),
           h3_modulation_input_path_.string()});
    result.convrot_int8_linears.reserve(convrot_int8_linear_plans_.size());
    for (const auto &plan : convrot_int8_linear_plans_)
      result.convrot_int8_linears.push_back(
          {plan.operation,
           plan.weight_tensor,
           plan.rows,
           plan.columns,
           plan.contraction,
           plan.weight.byte_size() + plan.scale.byte_size(),
           options.convrot_int8_weight_only_quality
               ? "approximate_native_h256_convrot_int8_weight_only_gate"
               : options.convrot_int8_resident
                     ? "approximate_native_h256_convrot_int8_resident_gate"
                     : "approximate_native_h256_convrot_int8_gate",
           options.convrot_int8_weight_only_quality
               ? "generic_diffir_linear_bf16_rotated_weight_only_cublaslt"
               : plan.dtype == ir::DType::F16
                     ? "generic_diffir_linear_cutlass_scaled_f16"
                     : "generic_diffir_linear_cutlass_scaled_bf16",
           plan.cache_path.string()});
    std::vector<double> elapsed;
    elapsed.reserve(options.iterations);
    auto reused_invariant_during_measurement = false;
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        Event start;
        Event stop;
        check(counted_event_record(start.get(), context_.stream()), "cuEventRecord start");
        const auto reuse_invariants = repeated_cache_ready;
        reused_invariant_during_measurement =
            reused_invariant_during_measurement || reuse_invariants;
        execute(iteration, options.profile_pipeline, reuse_invariants);
        if (!repeated_invariant_operations_.empty())
          repeated_cache_ready = true;
        streamed_prefetcher_->complete_iteration();
        check(counted_event_record(stop.get(), context_.stream()), "cuEventRecord stop");
        check(counted_event_synchronize(stop.get()), "cuEventSynchronize");
        float milliseconds = 0.0F;
        check(cuEventElapsedTime(&milliseconds, start.get(), stop.get()),
              "cuEventElapsedTime");
        elapsed.push_back(milliseconds);
    }
    if (!repeated_invariant_operations_.empty())
      repeated_invariant_valid_ = true;
    result.repeated_invariant_cache_hit =
        reused_invariant_during_measurement;
    result.minimum_milliseconds = *std::min_element(elapsed.begin(), elapsed.end());
    result.maximum_milliseconds = *std::max_element(elapsed.begin(), elapsed.end());
    result.mean_milliseconds =
        std::accumulate(elapsed.begin(), elapsed.end(), 0.0) / elapsed.size();

    if (options.profile_pipeline) {
      result.pipeline_profile.enabled = true;
      result.pipeline_profile.measured_iterations = options.iterations;
      result.pipeline_profile.resident_weight_bytes = resident_weight_bytes_;
      result.pipeline_profile.resident_host_prefault_milliseconds =
          resident_host_prefault_milliseconds_;
      result.pipeline_profile.resident_minor_page_faults =
          resident_minor_page_faults_;
      result.pipeline_profile.resident_major_page_faults =
          resident_major_page_faults_;
      result.pipeline_profile.resident_h2d_milliseconds =
          resident_h2d_milliseconds_;
      warm_h3_resident_pages();
      result.pipeline_profile.resident_readahead_bytes =
          h3_resident_readahead_advised_bytes_;
      result.pipeline_profile.resident_direct_read_bytes =
          h3_resident_direct_read_bytes_;
      result.pipeline_profile.resident_upload_milliseconds =
          resident_upload_milliseconds_;
      double kernel_total = 0.0;
      double attention_total = 0.0;
      result.operation_timings.reserve(program_.operations.size());
      for (std::size_t operation_index = 0U;
           operation_index < program_.operations.size(); ++operation_index) {
        if (skipped_operations_.contains(
                program_.operations[operation_index].id)) {
          result.operation_timings.push_back(
              {program_.operations[operation_index].id,
               program_.operations[operation_index].opcode, 0.0, 0.0, 0.0, std::string{}});
          continue;
        }
        std::vector<double> operation_elapsed;
        operation_elapsed.reserve(options.iterations);
        for (std::uint32_t iteration = 0U; iteration < options.iterations;
             ++iteration) {
          if (profile_reused_invariant_operations.at(
                  static_cast<std::size_t>(iteration) *
                      program_.operations.size() +
                  operation_index)) {
            operation_elapsed.push_back(0.0);
            continue;
          }
          const auto &timing = profile_operation_events.at(
              static_cast<std::size_t>(iteration) *
                  program_.operations.size() +
              operation_index);
          float milliseconds = 0.0F;
          check(cuEventElapsedTime(&milliseconds, timing.start->get(),
                                   timing.stop->get()),
                "cuEventElapsedTime profiled operation");
          operation_elapsed.push_back(milliseconds);
          kernel_total += milliseconds;
          if (program_.operations[operation_index].opcode ==
              ir::Opcode::Attention)
            attention_total += milliseconds;
        }
        result.operation_timings.push_back(
            {program_.operations[operation_index].id,
             program_.operations[operation_index].opcode,
             std::accumulate(operation_elapsed.begin(), operation_elapsed.end(),
                             0.0) /
                 operation_elapsed.size(),
             *std::min_element(operation_elapsed.begin(),
                               operation_elapsed.end()),
             *std::max_element(operation_elapsed.begin(),
                               operation_elapsed.end()), std::string{}});
      }
      result.pipeline_profile.operation_kernel_milliseconds = kernel_total;
      result.pipeline_profile.attention_kernel_milliseconds = attention_total;
      const auto total_device_timeline =
          std::accumulate(elapsed.begin(), elapsed.end(), 0.0);
      result.pipeline_profile.non_kernel_device_timeline_milliseconds =
          std::max(0.0, total_device_timeline - kernel_total);
      streamed_prefetcher_->finish_profile(result.pipeline_profile);
      result.pipeline_profile.streamed_weight_bytes +=
          h3_w8a8_tail_streamed_bytes;
      result.pipeline_profile.streamed_host_stage_milliseconds +=
          h3_w8a8_tail_host_stage_milliseconds;
      result.pipeline_profile.streamed_weight_bytes +=
          convrot_streamed_bytes;
      result.pipeline_profile.streamed_host_stage_milliseconds +=
          convrot_host_stage_milliseconds;
    }

    if (options.trace_operations && !options.profile_pipeline) {
      for (const auto &op : program_.operations) {
          std::vector<double> operation_elapsed;
          operation_elapsed.reserve(options.iterations);
          for (std::uint32_t iteration = 0; iteration < options.iterations;
               ++iteration) {
            Event start;
            Event stop;
            check(counted_event_record(start.get(), context_.stream()),
                  "cuEventRecord operation start");
            const auto index = static_cast<std::size_t>(&op -
                program_.operations.data());
            const auto ready = streamed_prefetcher_->prefetch(index, true);
            streamed_prefetcher_->wait(index, ready);
            execute_operation(op, false);
            streamed_prefetcher_->complete(index);
            check(counted_event_record(stop.get(), context_.stream()),
                  "cuEventRecord operation stop");
            check(counted_event_synchronize(stop.get()),
                  "cuEventSynchronize operation");
            float milliseconds = 0.0F;
            check(cuEventElapsedTime(&milliseconds, start.get(), stop.get()),
                  "cuEventElapsedTime operation");
            operation_elapsed.push_back(milliseconds);
          }
        result.operation_timings.push_back(
              {op.id,
               op.opcode,
               std::accumulate(operation_elapsed.begin(), operation_elapsed.end(),
                               0.0) /
                   operation_elapsed.size(),
               *std::min_element(operation_elapsed.begin(),
                                 operation_elapsed.end()),
               *std::max_element(operation_elapsed.begin(),
                                 operation_elapsed.end()), std::string{}});
      }
    }

    if (options.pinned_io_staging) {
      auto *base = static_cast<std::uint8_t *>(pinned_io_->data());
      std::size_t offset = 0U;
      std::vector<std::pair<const ir::TensorDesc *, std::size_t>> staged;
      for (const auto &desc : program_.tensors) {
        if (!desc.has_role(ir::TensorRole::Output))
          continue;
        check(counted_memcpy_dtoh(base + offset, buffers_.at(desc.id),
                                  desc.byte_count(), context_.stream()),
              "cuMemcpyDtoHAsync pinned output");
        staged.emplace_back(&desc, offset);
        offset += desc.byte_count();
      }
      check(counted_stream_synchronize(context_.stream()),
            "output copy synchronization");
      for (const auto &[description, staged_offset] : staged) {
        auto tensor = zeros(*description);
        std::memcpy(tensor.mutable_data(), base + staged_offset,
                    tensor.byte_size());
        result.outputs.emplace(description->id, std::move(tensor));
      }
    } else {
      for (const auto &desc : program_.tensors) {
        if (!desc.has_role(ir::TensorRole::Output))
          continue;
        auto tensor = zeros(desc);
        check(counted_memcpy_dtoh(tensor.mutable_data(), buffers_.at(desc.id),
                                tensor.byte_size(), context_.stream()),
              "cuMemcpyDtoHAsync");
        result.outputs.emplace(desc.id, std::move(tensor));
      }
      check(counted_stream_synchronize(context_.stream()),
            "output copy synchronization");
    }

    std::size_t free_after = 0;
    std::size_t total = 0;
    check(cuMemGetInfo(&free_after, &total), "cuMemGetInfo after");
    result.free_bytes_after = free_after;
    if (!options.streamed_release_mapped_pages_per_copy &&
        !options.streamed_keep_mapped_pages_between_runs)
      streamed_prefetcher_->release_mapped_pages();
    result.preparation_telemetry = preparation_telemetry_;
    result.run_telemetry = run_telemetry;
    result.h3_int8_attention_exact_row_dispatches =
        h3_exact_query_row_dispatches_;
    result.captured_intermediates = std::move(captured_intermediates);
    if (options.profile_pipeline) {
      const auto print = [](const char *phase, const LaunchTelemetry &t) {
        std::cerr << "CUDA_LAUNCH_TELEMETRY phase=" << phase
                  << " kernel_launches=" << t.kernel_launches
                  << " cublaslt_matmuls=" << t.cublaslt_matmuls
                  << " cublas_gemms=" << t.cublas_gemms
                  << " cudnn_attention=" << t.cudnn_attention_dispatches
                  << " cudnn_convolution="
                  << t.cudnn_convolution_dispatches
                  << " cutlass=" << t.cutlass_launches
                  << " ck_attention=" << t.ck_attention_dispatches
                  << " h2d_copies=" << t.h2d_copies
                  << " h2d_bytes=" << t.h2d_bytes
                  << " d2h_copies=" << t.d2h_copies
                  << " d2h_bytes=" << t.d2h_bytes
                  << " d2d_copies=" << t.d2d_copies
                  << " d2d_bytes=" << t.d2d_bytes
                  << " event_records=" << t.event_records
                  << " stream_wait_events=" << t.stream_wait_events
                  << " host_event_syncs=" << t.host_event_synchronizes
                  << " host_stream_syncs=" << t.host_stream_synchronizes
                  << " device_mem_allocs=" << t.device_mem_allocs
                  << " pinned_mem_allocs=" << t.pinned_mem_allocs
                  << '\n';
      };
      print("prepare", preparation_telemetry_);
      print("run", run_telemetry);
    }
    nvtx_pop();
    for (auto &timing : result.operation_timings)
      if (const auto label = operation_plan_labels_.find(timing.operation_id);
          label != operation_plan_labels_.end())
        timing.plan = label->second;
    result.preparation_reported = !preparation_reported_;
    preparation_reported_ = true;
    if (tracing) {
      result.trace_milliseconds = run_tracer.now_ms();
      result.trace_events = std::move(run_tracer.events);
      if (result.preparation_reported) {
        result.preparation_trace_events = preparation_tracer_.events;
        result.preparation_trace_milliseconds =
            preparation_trace_milliseconds_;
      }
    }
    telemetry::append_runtime_trace(result, program_, options);
    return result;
  }

  std::string name() const override {
    auto result = std::string("cuda-nvrtc");
    if (!linear_plans_.empty())
      result += "-cublaslt";
    if (!fp8_scaled_linear_plans_.empty())
      result += "-scaled-fp8";
#if DIF_HAS_CUBLASLT_BLOCK_SCALE
    if (!fp8_block_scaled_linear_plans_.empty())
      result += "-mxfp8";
#endif
#if DIF_HAS_CUTLASS
    if (!cutlass_linear_plans_.empty())
      result += "-cutlass";
    if (!int8_scaled_linear_plans_.empty())
      result += "-scaled-int8";
    if (!int8_weight_linear_plans_.empty())
      result += "-int8-weight";
#endif
    if (!fused_linear_swiglu_plans_.empty())
      result += "-wmma-swiglu";
    const auto has_h3_convrot =
        std::any_of(h3_w8a8_mlp_plans_.begin(), h3_w8a8_mlp_plans_.end(),
                    [](const auto &plan) { return plan.convrot; }) ||
        std::any_of(h3_w8a8_attention_plans_.begin(),
                    h3_w8a8_attention_plans_.end(),
                    [](const auto &plan) { return plan.convrot; });
    if (has_h3_convrot)
      result += "-h3-convrot-int8";
    else if (!h3_w8a8_mlp_plans_.empty() ||
             !h3_w8a8_attention_plans_.empty())
      result += "-h3-w8a8";
    if (!convrot_int8_linear_plans_.empty())
      result += convrot_weight_only_quality_
                    ? "-convrot-int8-weight-only-quality"
                    : convrot_resident_ ? "-convrot-int8-resident"
                                        : "-convrot-int8";
    if (h3_int8_gemm_registry_)
      result += "-h3-int8-cublaslt";
#if DIF_HAS_CUTLASS
    if (h3_int8_scaled_gemm_registry_) {
      const auto scaled_all =
          std::any_of(h3_w8a8_mlp_plans_.begin(),
                      h3_w8a8_mlp_plans_.end(),
                      [](const auto &plan) { return plan.cutlass_scaled_fc2; });
      result += scaled_all ? "-h3-int8-scaled-all"
                           : "-h3-int8-scaled-fc1";
    }
#endif
    const auto has_h3_chunk_scaled_convrot =
        std::any_of(h3_w8a8_mlp_plans_.begin(), h3_w8a8_mlp_plans_.end(),
                    [](const auto &plan) {
                      return plan.convrot_scale_chunk != 0U;
                    }) ||
        std::any_of(h3_w8a8_attention_plans_.begin(),
                    h3_w8a8_attention_plans_.end(), [](const auto &plan) {
                      return plan.convrot_scale_chunk != 0U;
                    });
    if (has_h3_chunk_scaled_convrot)
      result += "-h3-chunk-scaled";
    if (!h3_compact_adaln_plans_.empty())
      result += "-h3-compact-adaln";
    if (!h3_groupwise_plans_.empty())
      result += "-h3-groupwise-int8";
    if (!h3_ck_attention_plans_.empty())
      result += h3_ck_attention_plan_ && h3_ck_attention_plan_->owned_dense()
                    ? "-owned-h3-dense-int8"
                : h3_ck_attention_plan_ &&
                          h3_ck_attention_plan_->codealexx_ck_int8()
                    ? "-codealexx-ck-int8"
                    : "-legacy-ck-int8";
    if (!h3_modulation_cache_plans_.empty())
      result += "-h3-modcache";
    if (!repeated_invariant_operations_.empty())
      result += "-repeat-cache";
    if (!fused_launch_inputs_.empty())
      result += "-packed-int5";
    if (uses_cudnn_attention_) {
      result += "-cudnn";
      if (cudnn_attention_heuristic_ == 1U)
        result += "-heur-b";
      else if (cudnn_attention_heuristic_ == 2U)
        result += "-heur-fallback";
    }
    if (!materialized_f32_attention_plans_.empty())
      result += "-materialized-f32-attention";
    if (flash_attention_workspace_bytes_ != 0U)
      result += "-native-flash-attention";
    return result;
  }
  double preparation_milliseconds() const override {
    return preparation_milliseconds_;
  }
  std::uint64_t resident_bytes() const override { return resident_bytes_; }

private:
  ir::Program program_;
  TensorMap constants_;
  // All prepared executables made by one Executor share its CUDA session.
  // Shape-specialized plans and storage remain independently owned, while
  // streams and vendor handles are created once and reused serially.
  std::shared_ptr<Context> context_owner_;
  Context &context_;
  std::unique_ptr<Module> module_;
  compiler::MemoryPlan memory_plan_;
  std::unique_ptr<DeviceArena> arena_;
  DeviceBuffers buffers_;
  std::unique_ptr<Workspace> workspace_;
  std::vector<std::unique_ptr<Stream>> parallel_streams_;
  std::vector<std::unique_ptr<Workspace>> parallel_workspaces_;
  std::unique_ptr<Event> parallel_start_event_;
  std::vector<std::unique_ptr<Event>> parallel_done_events_;
  std::unique_ptr<Workspace> cudnn_workspace_;
  std::unique_ptr<Workspace> materialized_f32_attention_scores_;
  std::unique_ptr<Workspace> h3_w8a8_scratch_storage_;
  std::unique_ptr<Workspace> h3_w8a8_tail_weight_storage_;
  std::unique_ptr<PinnedHostWorkspace> h3_w8a8_tail_stage_;
  std::unique_ptr<StagingPool> h3_w8a8_staging_pool_;
  std::array<std::unique_ptr<Event>, 2> h3_w8a8_tail_stage_events_;
  std::unique_ptr<Event> h3_w8a8_tail_order_event_;
  std::unique_ptr<Event> h3_w8a8_tail_ready_event_;
  std::unique_ptr<PinnedHostWorkspace> pinned_io_;
  std::array<bool, 2> h3_w8a8_tail_stage_armed_{};
  std::unique_ptr<Workspace> h3_groupwise_scratch_storage_;
  std::unique_ptr<Workspace> h3_modulation_storage_;
  std::unique_ptr<Workspace> repeated_invariant_cache_storage_;
  std::unique_ptr<StreamedPrefetcher> streamed_prefetcher_;
  std::unordered_map<std::uint32_t, CUfunction> functions_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      fused_launch_inputs_;
  std::unordered_set<std::uint32_t> skipped_operations_;
  std::unordered_set<std::uint32_t> repeated_invariant_operations_;
  std::unordered_set<std::uint32_t> repeated_invariant_persistent_tensors_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      repeated_invariant_outputs_by_producer_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      captured_tensors_by_producer_;
  std::unordered_map<std::uint32_t, std::uint64_t>
      repeated_invariant_cache_offsets_;
  std::vector<std::uint32_t> repeated_invariant_input_tensors_;
  std::vector<std::uint32_t> capture_intermediate_tensors_;
  std::unordered_map<std::uint32_t, std::vector<std::uint8_t>>
      repeated_invariant_input_snapshots_;
  std::unordered_map<std::uint32_t, std::shared_ptr<LinearPlan>> linear_plans_;
  std::unordered_map<std::uint32_t, std::unique_ptr<Fp8ScaledLinearPlan>>
      fp8_scaled_linear_plans_;
#if DIF_HAS_CUBLASLT_BLOCK_SCALE
  std::unordered_map<std::uint32_t,
                     std::unique_ptr<Fp8BlockScaledLinearPlan>>
      fp8_block_scaled_linear_plans_;
#endif
  std::unordered_map<std::size_t, std::vector<std::size_t>>
      parallel_linear_groups_;
  std::unordered_set<std::size_t> parallel_linear_followups_;
#if DIF_HAS_CUTLASS
  std::unordered_map<std::uint32_t, std::unique_ptr<CutlassLinearPlan>>
      cutlass_linear_plans_;
  std::unordered_map<std::uint32_t, std::unique_ptr<Int8ScaledLinearPlan>>
      int8_scaled_linear_plans_;
  std::unordered_map<std::uint32_t, std::unique_ptr<Int8WeightLinearPlan>>
      int8_weight_linear_plans_;
#endif
  std::vector<LinearTuningResult> linear_tuning_results_;
  std::vector<LinearAlgorithmChoice> selected_linear_algorithms_;
  std::vector<FusedLinearSwiGluPlan> fused_linear_swiglu_plans_;
  std::vector<AbsorbedLinearBiasPlan> absorbed_linear_bias_plans_;
  LinearHeuristicCacheStats linear_heuristic_cache_stats_;
  std::vector<H3W8A8MlpPlan> h3_w8a8_mlp_plans_;
  std::vector<H3W8A8AttentionPlan> h3_w8a8_attention_plans_;
  std::vector<ConvRotInt8LinearPlan> convrot_int8_linear_plans_;
  std::vector<H3CompactAdaLNPlan> h3_compact_adaln_plans_;
  std::unique_ptr<H3Int8GemmRegistry> h3_int8_gemm_registry_;
#if DIF_HAS_CUTLASS
  std::unique_ptr<H3Int8ScaledGemmRegistry> h3_int8_scaled_gemm_registry_;
  std::unique_ptr<Int8ScaledF16GemmRegistry>
      int8_scaled_f16_gemm_registry_;
#endif
  std::vector<H3GroupwiseBlockPlan> h3_groupwise_plans_;
  std::vector<H3ModulationCachePlan> h3_modulation_cache_plans_;
  Tensor h3_modulation_expected_input_;
  std::filesystem::path h3_modulation_input_path_;
  std::uint32_t h3_modulation_slices_{};
  H3W8A8Functions h3_w8a8_functions_;
  H3ConvRotFunctions h3_convrot_functions_;
  std::unique_ptr<H3ConvRotBf16Correction> h3_convrot_bf16_correction_;
  std::unique_ptr<Workspace> convrot_weight_storage_;
  std::unique_ptr<Workspace> convrot_scratch_storage_;
  std::unique_ptr<PinnedHostWorkspace> convrot_staging_;
  std::unique_ptr<StagingPool> convrot_staging_pool_;
  std::array<std::unique_ptr<Event>, 2> convrot_slot_done_;
  std::array<bool, 2> convrot_slot_armed_{};
  CUdeviceptr convrot_activation_device_{};
  CUdeviceptr convrot_activation_scale_device_{};
  CUdeviceptr convrot_quality_weight_device_{};
  bool convrot_weight_only_quality_{};
  bool resident_evict_host_pages_{true};
  // Resident checkpoint read-ahead: merged upload order (is_attention, slot),
  // the next order index to advise, bytes advised but not yet staged, the
  // window, and the receipt count.
  std::vector<std::pair<bool, std::size_t>> h3_w8a8_upload_order_;
  std::size_t h3_w8a8_readahead_cursor_{};
  std::uint64_t h3_w8a8_readahead_ahead_bytes_{};
  std::uint64_t h3_resident_readahead_bytes_{};
  std::uint64_t h3_resident_readahead_advised_bytes_{};
  bool h3_resident_direct_io_{true};
  std::uint64_t h3_resident_direct_read_bytes_{};
  bool h3_resident_counters_reported_{false};
  bool direct_io_warm_page_cache_{true};
  std::vector<const Tensor *> h3_resident_warm_list_;

  // Background page-cache read of everything staged with direct IO since the
  // last call, so the next process (or evaluation) takes the mapping copy.
  void warm_h3_resident_pages() {
    if (h3_resident_warm_list_.empty())
      return;
    // Same admission as the keep policy: page cache is charged to the
    // process' cgroup, so warm only when its limit can hold the bytes.
    std::uint64_t bytes = 0U;
    for (const auto *tensor : h3_resident_warm_list_)
      bytes += tensor->byte_size();
    const auto cgroup = probe_host_cgroup_memory();
    constexpr std::uint64_t margin = 512ULL * 1024ULL * 1024ULL;
    const bool admitted =
        cgroup.limit_bytes == 0U ||
        cgroup.limit_bytes >= cgroup.current_bytes + bytes + margin;
    if (admitted)
      for (const auto *tensor : h3_resident_warm_list_)
        tensor->prefetch_mapped_pages();
    h3_resident_warm_list_.clear();
  }

  std::uint64_t h3_w8a8_order_bytes(std::size_t order) const {
    const auto [attention, slot] = h3_w8a8_upload_order_[order];
    return attention ? h3_w8a8_weight_bytes(h3_w8a8_attention_plans_[slot])
                     : h3_w8a8_weight_bytes(h3_w8a8_mlp_plans_[slot]);
  }

  // Called right before the plan at `order` is staged from its mapping:
  // retire that plan's bytes from the in-flight window, then advise plans
  // further along the checkpoint until the window is full again.
  void advise_h3_resident_readahead(std::size_t order) {
    if (h3_resident_readahead_bytes_ == 0U || h3_w8a8_upload_order_.empty())
      return;
    if (order < h3_w8a8_readahead_cursor_)
      h3_w8a8_readahead_ahead_bytes_ -=
          std::min(h3_w8a8_order_bytes(order), h3_w8a8_readahead_ahead_bytes_);
    while (h3_w8a8_readahead_cursor_ < h3_w8a8_upload_order_.size() &&
           h3_w8a8_readahead_ahead_bytes_ < h3_resident_readahead_bytes_) {
      const auto [attention, slot] =
          h3_w8a8_upload_order_[h3_w8a8_readahead_cursor_];
      if (attention)
        prefetch_h3_w8a8_weights(h3_w8a8_attention_plans_[slot]);
      else
        prefetch_h3_w8a8_weights(h3_w8a8_mlp_plans_[slot]);
      const auto bytes = h3_w8a8_order_bytes(h3_w8a8_readahead_cursor_);
      h3_w8a8_readahead_ahead_bytes_ += bytes;
      h3_resident_readahead_advised_bytes_ += bytes;
      ++h3_w8a8_readahead_cursor_;
    }
  }
  bool preparation_reported_{false};
  std::unordered_map<std::uint32_t, std::string> operation_plan_labels_;
  bool convrot_resident_{};
  std::uint64_t convrot_weight_slot_bytes_{};
  std::uint64_t convrot_weight_bytes_{};
  std::uint64_t convrot_scale_bytes_{};
  std::uint64_t convrot_turn_{};
  CUfunction h3_groupwise_dequant_function_{};
  std::shared_ptr<CkAttentionPlan> h3_ck_attention_plan_;
  std::unordered_map<std::uint32_t, std::shared_ptr<CkAttentionPlan>>
      h3_ck_attention_plans_;
  // Route layer index (program order within the INT8 attention route) per
  // routed operation, the hybrid sub-range that follows the per-run switch,
  // and the exact query-row overlay ranges with one shared cuDNN plan each.
  bool h3_hybrid_member(std::uint32_t operation_id) const {
    const auto layer = h3_ck_attention_layer_.find(operation_id);
    return layer != h3_ck_attention_layer_.end() &&
           layer->second >= h3_int8_attention_hybrid_first_layer_ &&
           layer->second < h3_int8_attention_hybrid_first_layer_ +
                               h3_int8_attention_hybrid_layers_;
  }
  std::unordered_map<std::uint32_t, std::uint32_t> h3_ck_attention_layer_;
  std::uint32_t h3_int8_attention_hybrid_first_layer_{};
  std::uint32_t h3_int8_attention_hybrid_layers_{};
  std::vector<RunOptions::QueryRowRange> h3_exact_query_ranges_;
  std::uint64_t h3_exact_query_row_dispatches_{};
  std::unordered_map<std::uint32_t, MaterializedF32AttentionPlan>
      materialized_f32_attention_plans_;
  CUfunction materialized_f32_attention_softmax_{};
#if DIF_HAS_CUDNN
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnAttentionPlan>>
      cudnn_attention_plans_;
  std::vector<std::shared_ptr<CudnnAttentionPlan>> h3_exact_query_range_plans_;
  std::unordered_map<std::uint32_t,
                     std::shared_ptr<CudnnAttentionBackwardPlan>>
      cudnn_attention_backward_plans_;
  std::unordered_map<std::uint32_t, std::unique_ptr<Workspace>>
      cudnn_backward_stats_;
  std::uint64_t cudnn_backward_stats_bytes_{};
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnConv2dPlan>>
      cudnn_conv_plans_;
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnConv3dPlan>>
      cudnn_conv3d_plans_;
#endif
  std::string device_name_;
  target::TargetProfile target_profile_;
  target::RuntimeBudget runtime_budget_;
  std::string source_hash_;
  std::size_t workspace_bytes_{};
  std::uint64_t flash_attention_workspace_bytes_{};
  std::uint64_t parallel_workspace_bytes_{};
  std::size_t cudnn_workspace_bytes_{};
  std::uint64_t materialized_f32_attention_score_bytes_{};
  std::uint64_t ck_attention_scratch_bytes_{};
  bool h3_int8_attention_hybrid_{false};
  std::uint64_t h3_w8a8_tail_attention_bytes_{};
  std::uint64_t h3_w8a8_tail_mlp_bytes_{};
  std::uint64_t h3_w8a8_tail_weight_bytes_{};
  std::uint64_t h3_w8a8_tail_stage_half_bytes_{};
  std::uint64_t h3_w8a8_tail_stage_turn_{};
  std::uint64_t resident_bytes_{};
  std::uint64_t repeated_invariant_persistent_bytes_{};
  std::uint64_t repeated_invariant_cache_bytes_{};
  std::uint64_t resident_weight_bytes_{};
  std::uint64_t free_bytes_before_{};
  std::uint32_t streamed_prefetch_depth_{1};
  std::vector<std::uint32_t> promoted_streamed_constants_;
  std::unordered_set<std::uint32_t> reshape_alias_operations_;
  std::unordered_map<std::uint32_t, std::uint32_t> reshape_aliases_;
  bool lazy_resident_upload_{};
  bool repeated_invariant_valid_{};
  std::unique_ptr<Workspace> promoted_constant_storage_;
  std::uint64_t promoted_constant_bytes_{};
  LaunchTelemetry preparation_telemetry_;
  Tracer preparation_tracer_;
  double preparation_trace_milliseconds_{};
  double preparation_milliseconds_{};
  double resident_upload_milliseconds_{};
  double resident_host_prefault_milliseconds_{};
  double resident_h2d_milliseconds_{};
  std::uint64_t resident_minor_page_faults_{};
  std::uint64_t resident_major_page_faults_{};
  std::uint64_t resident_prefault_checksum_{};
  bool uses_cudnn_attention_{};
  std::uint32_t cudnn_attention_heuristic_{};
};

class CudaExecutor final : public Executor {
public:
  explicit CudaExecutor(int device)
      : context_(std::make_shared<Context>(device)) {}

  std::unique_ptr<PreparedExecution>
  prepare(const ir::Program &program, const TensorMap &bindings,
          const RunOptions &options) override {
    return std::make_unique<CudaPreparedExecution>(program, bindings, options,
                                                   context_);
  }

  std::string name() const override { return "cuda-nvrtc-cublaslt"; }

private:
  std::shared_ptr<Context> context_;
};

} // namespace

std::unique_ptr<Executor> make_cuda_executor(int device) {
  return std::make_unique<CudaExecutor>(device);
}

bool cuda_available() {
  if (cuInit(0) != CUDA_SUCCESS)
    return false;
  int count = 0;
  return cuDeviceGetCount(&count) == CUDA_SUCCESS && count > 0;
}

} // namespace dif::runtime
