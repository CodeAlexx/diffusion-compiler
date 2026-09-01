#include "dif/runtime/executor.hpp"

#include "dif/compiler/compiler.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/verify.hpp"
#if DIF_HAS_CUDNN
#include "dif/runtime/cudnn_attention.hpp"
#include "dif/runtime/cudnn_conv.hpp"
#endif
#if DIF_HAS_CUTLASS
#include "dif/runtime/cutlass_gemm.hpp"
#endif
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>
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
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <string>
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

CUresult counted_launch_kernel(CUfunction function, unsigned grid_x,
                               unsigned grid_y, unsigned grid_z,
                               unsigned block_x, unsigned block_y,
                               unsigned block_z, unsigned shared_bytes,
                               CUstream stream, void **parameters,
                               void **extra) {
  if (active_telemetry)
    ++active_telemetry->kernel_launches;
  return cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y,
                        block_z, shared_bytes, stream, parameters, extra);
}

CUresult counted_memcpy_htod(CUdeviceptr destination, const void *source,
                             std::size_t bytes, CUstream stream) {
  if (active_telemetry) {
    ++active_telemetry->h2d_copies;
    active_telemetry->h2d_bytes += bytes;
  }
  return cuMemcpyHtoDAsync(destination, source, bytes, stream);
}

CUresult counted_memcpy_dtoh(void *destination, CUdeviceptr source,
                             std::size_t bytes, CUstream stream) {
  if (active_telemetry) {
    ++active_telemetry->d2h_copies;
    active_telemetry->d2h_bytes += bytes;
  }
  return cuMemcpyDtoHAsync(destination, source, bytes, stream);
}

CUresult counted_event_record(CUevent event, CUstream stream) {
  if (active_telemetry)
    ++active_telemetry->event_records;
  return cuEventRecord(event, stream);
}

CUresult counted_stream_wait_event(CUstream stream, CUevent event,
                                   unsigned flags) {
  if (active_telemetry)
    ++active_telemetry->stream_wait_events;
  return cuStreamWaitEvent(stream, event, flags);
}

CUresult counted_event_synchronize(CUevent event) {
  if (active_telemetry)
    ++active_telemetry->host_event_synchronizes;
  return cuEventSynchronize(event);
}

CUresult counted_stream_synchronize(CUstream stream) {
  if (active_telemetry)
    ++active_telemetry->host_stream_synchronizes;
  return cuStreamSynchronize(stream);
}

CUresult counted_mem_alloc(CUdeviceptr *pointer, std::size_t bytes) {
  if (active_telemetry)
    ++active_telemetry->device_mem_allocs;
  return cuMemAlloc(pointer, bytes);
}

CUresult counted_mem_host_alloc(void **pointer, std::size_t bytes,
                                unsigned flags) {
  if (active_telemetry)
    ++active_telemetry->pinned_mem_allocs;
  return cuMemHostAlloc(pointer, bytes, flags);
}

void count_cublaslt_matmul() {
  if (active_telemetry)
    ++active_telemetry->cublaslt_matmuls;
}

void count_cudnn_attention_dispatch() {
  if (active_telemetry)
    ++active_telemetry->cudnn_attention_dispatches;
}

void count_cudnn_convolution_dispatch() {
  if (active_telemetry)
    ++active_telemetry->cudnn_convolution_dispatches;
}

void count_cutlass_launch() {
  if (active_telemetry)
    ++active_telemetry->cutlass_launches;
}

// One dispatch = the DSO's quantize-QK + quantize-V + attend launcher trio.
void count_ck_attention_dispatch() {
  if (active_telemetry)
    ++active_telemetry->ck_attention_dispatches;
}

template <typename... Arguments>
cublasStatus_t counted_cublas_gemm_ex(Arguments &&...arguments) {
  if (active_telemetry)
    ++active_telemetry->cublas_gemms;
  return cublasGemmEx(std::forward<Arguments>(arguments)...);
}

class Context {
public:
  explicit Context(int ordinal) {
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
  CUstream stream() const { return stream_; }
  CUstream copy_stream() const { return copy_stream_; }
  cublasHandle_t cublas() const { return cublas_; }
  cublasLtHandle_t cublas_lt() const { return cublas_lt_; }

private:
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
    check(cuModuleLoadDataEx(&module_, ptx.data(), 0, nullptr, nullptr),
          "cuModuleLoadDataEx");
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
             LinearHeuristicCacheStats *cache_stats) {
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
      if (m != 1)
        fail("cuBLASLt addmm bias mode currently requires one output row");
      bias_bytes_ = program.tensor(op.inputs[2])->byte_count();
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
      const auto copy_status = cuMemcpyDtoDAsync(
          static_cast<CUdeviceptr>(buffers.at(op.outputs[0])),
          static_cast<CUdeviceptr>(buffers.at(op.inputs[2])), bias_bytes_,
          stream);
      if (copy_status != CUDA_SUCCESS)
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
  std::uint64_t bias_bytes_{};
  std::size_t workspace_limit_bytes_{};
  bool persist_{};
  LinearHeuristicCacheStats *cache_stats_{};
  std::filesystem::path tuned_cache_file_;
  std::filesystem::path passive_cache_file_;
};

#if DIF_HAS_CUTLASS
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

constexpr std::uint32_t kH3W8A8MlpChunkRows = 1024U;
constexpr std::uint32_t kH3W8A8ProjectionChunkRows = 4096U;

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
  std::uint64_t rows{};
  std::uint64_t hidden{};
  std::uint64_t ffn{};
  std::uint64_t packed_ffn{};
  std::vector<std::uint32_t> excluded_tensors;
  std::vector<std::uint32_t> replaced_constant_tensors;
  Tensor fc1_weight;
  Tensor fc1_scale;
  Tensor fc2_weight;
  Tensor fc2_scale;
  std::unique_ptr<Workspace> weight_storage;
  CUdeviceptr fc1_weight_device{};
  CUdeviceptr fc1_scale_device{};
  CUdeviceptr fc2_weight_device{};
  CUdeviceptr fc2_scale_device{};
  CUdeviceptr input_scale_device{};
  CUdeviceptr input_i8_device{};
  CUdeviceptr fc1_accumulator_device{};
  CUdeviceptr activation_device{};
  CUdeviceptr activation_scale_device{};
  CUdeviceptr activation_i8_device{};
  CUdeviceptr fc2_accumulator_device{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t weight_storage_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::filesystem::path cache_path;
  bool resident{};
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
  std::uint32_t layer{};
  std::uint64_t rows{};
  std::uint64_t hidden{};
  std::uint64_t inner{};
  std::uint64_t packed_inner{};
  Tensor qkv_weight;
  Tensor qkv_scale;
  Tensor output_weight;
  Tensor output_scale;
  std::unique_ptr<Workspace> weight_storage;
  CUdeviceptr qkv_weight_device{};
  CUdeviceptr qkv_scale_device{};
  CUdeviceptr output_weight_device{};
  CUdeviceptr output_scale_device{};
  CUdeviceptr activation_scale_device{};
  CUdeviceptr activation_i8_device{};
  CUdeviceptr accumulator_device{};
  std::uint64_t quantized_weight_bytes{};
  std::uint64_t weight_storage_bytes{};
  std::uint64_t scratch_bytes{};
  std::uint64_t eliminated_intermediate_bytes{};
  std::filesystem::path cache_path;
  bool resident{};
};

struct H3W8A8Functions {
  CUfunction rowscale{};
  CUfunction encode{};
  CUfunction qkv{};
  CUfunction swiglu{};
  CUfunction residual{};
};

std::uint64_t align_256(std::uint64_t value) {
  if (value > std::numeric_limits<std::uint64_t>::max() - 255U)
    fail("H3 W8A8 storage alignment overflow");
  return (value + 255U) & ~std::uint64_t{255U};
}

// Direct port of the accepted Serenity Comfy Kitchen C ABI bridge. The loaded
// DSO contains only the pinned raw CUDA launchers plus ABI/target-SM metadata;
// no Mojo or Python runtime participates in prepared execution.
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

  CkAttentionLibrary(const std::filesystem::path &path, int current_sm)
      : path_(path) {
    const auto path_string = path.string();
    void *candidate = dlopen(path_string.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!candidate) {
      const char *reason = dlerror();
      fail("cannot load H3 CK attention DSO " + path_string + ": " +
           (reason ? reason : "unknown dlopen error"));
    }
    auto symbol = [&](const char *name) {
      dlerror();
      auto *address = dlsym(candidate, name);
      const char *reason = dlerror();
      if (!address || reason) {
        const auto message = std::string("H3 CK attention DSO is missing ") +
                             name + ": " +
                             (reason ? reason : "unknown dlsym error");
        dlclose(candidate);
        fail(message);
      }
      return address;
    };
    const auto abi = reinterpret_cast<MetadataInt>(
        symbol("serenity_ck_attention_abi_version"));
    const auto target = reinterpret_cast<MetadataInt>(
        symbol("serenity_ck_attention_target_sm"));
    const auto abi_version = abi();
    target_sm_ = target();
    if (abi_version != 1 || target_sm_ != current_sm) {
      const auto message =
          "H3 CK attention DSO admission failed: ABI=" +
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
              int heads, float scale, CUstream stream) const {
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
  const std::filesystem::path &path() const { return path_; }

private:
  std::filesystem::path path_;
  void *handle_{};
  QuantQK quant_qk_{};
  QuantV quant_v_{};
  Attend attend_{};
  int target_sm_{};
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
    const auto k_scale_elements = heads * ((sequence + 127U) / 128U) * 4U;
    const auto v_elements = heads * 128U * padded_sequence;
    scratch_bytes_ = align_256(elements) + align_256(elements) +
                     align_256(q_scale_elements * sizeof(float)) +
                     align_256(k_scale_elements * sizeof(float)) +
                     align_256(v_elements) +
                     align_256(heads * 128U * sizeof(float)) +
                     align_256(heads * sizeof(std::int32_t));
  }

  void allocate(DeviceArena *arena) {
    if (storage_)
      return;
    const auto sequence = static_cast<std::uint64_t>(sequence_);
    const auto heads = static_cast<std::uint64_t>(heads_);
    const auto padded_sequence = (sequence + 127U) / 128U * 128U;
    const auto elements = sequence * heads * 128U;
    const auto q_scale_elements = heads * ((sequence + 127U) / 128U) * 32U;
    const auto k_scale_elements = heads * ((sequence + 127U) / 128U) * 4U;
    const auto v_elements = heads * 128U * padded_sequence;
    storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(scratch_bytes_), arena);
    auto offset = std::uint64_t{0U};
    auto assign = [&](CUdeviceptr &target, std::uint64_t bytes) {
      target = storage_->pointer() + offset;
      offset += align_256(bytes);
    };
    assign(q_int8_, elements);
    assign(k_int8_, elements);
    assign(q_scale_, q_scale_elements * sizeof(float));
    assign(k_scale_, k_scale_elements * sizeof(float));
    assign(v_int8_, v_elements);
    assign(v_scale_, heads * 128U * sizeof(float));
    assign(anchor_indices_, heads * sizeof(std::int32_t));
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
                     sequence_, heads_, scale_, stream);
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

void validate_h3_modulation_cache_metadata(
    const weights::SafeTensorFile &cache, const RunOptions &options,
    std::uint32_t slices, std::uint32_t blocks) {
  if (options.h3_modulation_source_index.empty() ||
      options.h3_modulation_steps == 0U)
    fail("schedule modulation cache requires its source index and step count");
  if (h3_metadata_u64(cache, "__meta__.version") != 1U ||
      h3_metadata_string(cache, "__meta__.kind") != "adaln-modulation" ||
      h3_metadata_string(cache, "__meta__.src_path") !=
          options.h3_modulation_source_index.string() ||
      h3_metadata_u64(cache, "__meta__.steps") !=
          options.h3_modulation_steps ||
      h3_metadata_u64(cache, "__meta__.distinct_timesteps") != 2U * slices ||
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
      validate_h3_modulation_cache_metadata(cache, options, slices,
                                            block_count);
    } else {
      if (options.h3_modulation_layer + block_count >
          options.h3_modulation_total_layers)
        fail("H3 diagnostic modulation slice exceeds total layer count");
      validate_h3_modulation_cache_metadata(
          cache, options, slices, options.h3_modulation_total_layers);
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
  if (!options.h3_w8a8_cache.empty())
    fail("H3 W8A8 and groupwise INT8 caches are mutually exclusive");
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
    const auto *packed = program.tensor(layout.inputs.at(0));
    if (!packed || packed->dtype != ir::DType::BF16 ||
        packed->dims.size() != 2U || packed->dims.at(0) % 3U != 0U)
      fail("H3 groupwise route found an incompatible QKV layout");
    const auto packed_inner = packed->dims.at(0);
    const auto inner = packed_inner / 3U;
    const auto hidden = packed->dims.at(1);
    for (const auto output : layout.outputs) {
      const auto *description = program.tensor(output);
      const auto found = consumers.find(output);
      if (!description || description->dtype != ir::DType::BF16 ||
          description->dims != std::vector<std::uint64_t>{inner, hidden} ||
          found == consumers.end() || found->second.size() != 1U ||
          found->second.front()->opcode != ir::Opcode::Linear ||
          found->second.front()->inputs.size() != 2U ||
          found->second.front()->inputs.at(1) != output)
        fail("H3 groupwise route requires the source-faithful split QKV chain");
    }

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
    const std::array<std::uint32_t, 4> accepted_groups = {16U, 64U, 32U,
                                                          64U};
    for (std::size_t index = 0U; index < plan.projections.size(); ++index) {
      auto &projection = plan.projections[index];
      projection.weight = weights::map_safetensor(
          cache, prefix + ".weight." + std::to_string(index));
      projection.scale = weights::map_safetensor(
          cache, prefix + ".scale." + std::to_string(index));
      const auto *semantic = semantic_weights[index];
      if (!semantic || projection.weight.dtype != ir::DType::I8 ||
          projection.weight.dims != semantic->dims ||
          projection.scale.dtype != ir::DType::F16 ||
          projection.scale.dims.size() != 2U ||
          projection.scale.dims.at(0) != semantic->dims.at(0) ||
          projection.scale.dims.at(1) == 0U ||
          semantic->dims.at(1) % projection.scale.dims.at(1) != 0U)
        fail("H3 groupwise cache tensors do not match semantic projection shapes");
      projection.rows = semantic->dims.at(0);
      projection.columns = semantic->dims.at(1);
      projection.group_size = static_cast<std::uint32_t>(
          projection.columns / projection.scale.dims.at(1));
      if (projection.group_size != accepted_groups[index])
        fail("H3 groupwise cache does not use the accepted H3 group sizes");
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
    const signed char* weight, const dif_f16* scale, dif_bf16* output,
    unsigned long long rows, unsigned long long columns,
    unsigned group_size, unsigned swap_row_halves) {
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
    const float value = (float)weight[source] *
                        dif_load_f16(scale, source / group_size);
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
  const auto elements = rows * columns;
  const auto grid = static_cast<unsigned>(std::min<std::uint64_t>(
      65535U, (elements + 255U) / 256U));
  std::array<void *, 7> arguments = {&weight, &scale, &output, &rows,
                                     &columns, &group_size, &swap};
  check(counted_launch_kernel(function, grid, 1U, 1U, 256U, 1U, 1U, 0U, stream,
                       arguments.data(), nullptr),
        "cuLaunchKernel H3 groupwise INT8 dequant");
}

std::vector<H3W8A8MlpPlan> find_h3_w8a8_mlp_plans(
    const ir::Program &program, const RunOptions &options) {
  if (options.h3_w8a8_cache.empty())
    return {};
  const auto cache = weights::read_safetensors(options.h3_w8a8_cache);
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
    if (swiglu.opcode != ir::Opcode::SwiGlu || swiglu.inputs.size() != 1U ||
        swiglu.outputs.size() != 1U)
      continue;
    const auto fc1_found = producer.find(swiglu.inputs.at(0));
    if (fc1_found == producer.end() ||
        fc1_found->second->opcode != ir::Opcode::Linear ||
        consumers[swiglu.inputs.at(0)].size() != 1U)
      continue;
    const auto &fc1 = *fc1_found->second;
    // DiffIR binds the released [gate|value] FC1 tensor and therefore records
    // GateFirst=true. The accepted Serenity cache stores its runtime-swapped
    // [value|gate] compute copy; the W8A8 epilogue below consumes that copy in
    // its native order while preserving the same semantic result.
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

    const auto layer = options.h3_w8a8_layer +
                       static_cast<std::uint32_t>(result.size());
    const auto prefix = "block." + std::to_string(layer);
    auto mapped = [&](const std::string &suffix) {
      return weights::map_safetensor(cache, prefix + suffix);
    };
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
    plan.resident = layer - options.h3_w8a8_layer <
                    options.h3_w8a8_resident_layers;
    plan.rows = input->dims.at(0);
    plan.hidden = hidden;
    plan.ffn = ffn;
    plan.packed_ffn = packed_ffn;
    plan.excluded_tensors = {fc1.outputs.at(0), swiglu.outputs.at(0),
                             fc2.outputs.at(0)};
    plan.replaced_constant_tensors = {fc1.inputs.at(1), fc2.inputs.at(1)};
    plan.fc1_weight = mapped(".weight.2");
    plan.fc1_scale = mapped(".scale.2");
    plan.fc2_weight = mapped(".weight.3");
    plan.fc2_scale = mapped(".scale.3");
    if (plan.fc1_weight.dtype != ir::DType::I8 ||
        plan.fc1_weight.dims != fc1_weight->dims ||
        plan.fc1_scale.dtype != ir::DType::F32 ||
        plan.fc1_scale.dims != std::vector<std::uint64_t>{packed_ffn} ||
        plan.fc2_weight.dtype != ir::DType::I8 ||
        plan.fc2_weight.dims != fc2_weight->dims ||
        plan.fc2_scale.dtype != ir::DType::F32 ||
        plan.fc2_scale.dims != std::vector<std::uint64_t>{hidden})
      fail("H3 W8A8 cache tensors do not match the semantic MLP shapes");
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
    const auto chunk_rows = std::min<std::uint64_t>(
        plan.rows, kH3W8A8MlpChunkRows);
    plan.scratch_bytes =
        align_256(chunk_rows * sizeof(float)) +
        align_256(chunk_rows * hidden) +
        align_256(chunk_rows * packed_ffn * sizeof(std::int32_t)) +
        align_256(chunk_rows * ffn * sizeof(std::uint16_t)) +
        align_256(chunk_rows * sizeof(float)) +
        align_256(chunk_rows * ffn) +
        align_256(chunk_rows * hidden * sizeof(std::int32_t));
    plan.cache_path = options.h3_w8a8_cache;
    result.push_back(std::move(plan));
  }
  return result;
}

std::vector<H3W8A8AttentionPlan> find_h3_w8a8_attention_plans(
    const ir::Program &program, const RunOptions &options) {
  if (options.h3_w8a8_cache.empty())
    return {};
  const auto cache = weights::read_safetensors(options.h3_w8a8_cache);
  std::unordered_map<std::uint32_t, std::vector<const ir::Operation *>> consumers;
  for (const auto &operation : program.operations) {
    for (const auto input : operation.inputs)
      consumers[input].push_back(&operation);
  }

  std::vector<H3W8A8AttentionPlan> result;
  std::unordered_set<std::uint32_t> planned_output_linears;
  for (const auto &layout : program.operations) {
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

    const auto layer = options.h3_w8a8_layer +
                       static_cast<std::uint32_t>(result.size());
    const auto prefix = "block." + std::to_string(layer);
    auto mapped = [&](const std::string &suffix) {
      return weights::map_safetensor(cache, prefix + suffix);
    };
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
    plan.resident = layer - options.h3_w8a8_layer <
                    options.h3_w8a8_resident_layers;
    plan.rows = rows;
    plan.hidden = hidden;
    plan.inner = inner;
    plan.packed_inner = packed_inner;
    plan.qkv_weight = mapped(".weight.0");
    plan.qkv_scale = mapped(".scale.0");
    if (plan.qkv_weight.dtype != ir::DType::I8 ||
        plan.qkv_weight.dims != packed_weight->dims ||
        plan.qkv_scale.dtype != ir::DType::F32 ||
        plan.qkv_scale.dims != std::vector<std::uint64_t>{packed_inner})
      fail("H3 W8A8 cache tensors do not match the semantic QKV shapes");
    plan.quantized_weight_bytes =
        plan.qkv_weight.byte_size() + plan.qkv_scale.byte_size();
    plan.weight_storage_bytes = align_256(plan.qkv_weight.byte_size()) +
                                align_256(plan.qkv_scale.byte_size());
    if (has_output_projection) {
      plan.output_weight = mapped(".weight.1");
      plan.output_scale = mapped(".scale.1");
      if (plan.output_weight.dtype != ir::DType::I8 ||
          plan.output_weight.dims !=
              std::vector<std::uint64_t>{hidden, inner} ||
          plan.output_scale.dtype != ir::DType::F32 ||
          plan.output_scale.dims != std::vector<std::uint64_t>{hidden})
        fail("H3 W8A8 cache tensors do not match output projection shapes");
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
    plan.scratch_bytes = align_256(rows * sizeof(float)) +
                         align_256(rows * hidden) +
                         align_256(chunk_rows * packed_inner *
                                   sizeof(std::int32_t));
    plan.cache_path = options.h3_w8a8_cache;
    result.push_back(std::move(plan));
  }
  for (const auto &output_linear : program.operations) {
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
    const auto layer = options.h3_w8a8_layer +
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
    plan.resident = layer - options.h3_w8a8_layer <
                    options.h3_w8a8_resident_layers;
    plan.rows = rows;
    plan.hidden = hidden;
    plan.inner = inner;
    plan.packed_inner = 3U * inner;
    plan.output_weight =
        weights::map_safetensor(cache, prefix + ".weight.1");
    plan.output_scale =
        weights::map_safetensor(cache, prefix + ".scale.1");
    if (plan.output_weight.dtype != ir::DType::I8 ||
        plan.output_weight.dims != weight->dims ||
        plan.output_scale.dtype != ir::DType::F32 ||
        plan.output_scale.dims != std::vector<std::uint64_t>{hidden})
      fail("H3 W8A8 cache tensors do not match output projection shapes");
    plan.quantized_weight_bytes =
        plan.output_weight.byte_size() + plan.output_scale.byte_size();
    plan.weight_storage_bytes = align_256(plan.output_weight.byte_size()) +
                                align_256(plan.output_scale.byte_size());
    plan.eliminated_intermediate_bytes = projected->byte_count();
    const auto chunk_rows = std::min<std::uint64_t>(
        rows, kH3W8A8ProjectionChunkRows);
    plan.scratch_bytes = align_256(chunk_rows * sizeof(float)) +
                         align_256(chunk_rows * inner) +
                         align_256(chunk_rows * hidden *
                                   sizeof(std::int32_t));
    plan.cache_path = options.h3_w8a8_cache;
    result.push_back(std::move(plan));
  }
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
      plan.rows, kH3W8A8MlpChunkRows);
  auto scratch_offset = std::uint64_t{0U};
  auto assign_scratch = [&](CUdeviceptr &pointer, std::uint64_t bytes) {
    pointer = scratch + scratch_offset;
    scratch_offset += align_256(bytes);
  };
  assign_scratch(plan.input_scale_device, chunk_rows * sizeof(float));
  assign_scratch(plan.input_i8_device, chunk_rows * plan.hidden);
  assign_scratch(plan.fc1_accumulator_device,
                 chunk_rows * plan.packed_ffn * sizeof(std::int32_t));
  assign_scratch(plan.activation_device,
                 chunk_rows * plan.ffn * sizeof(std::uint16_t));
  assign_scratch(plan.activation_scale_device, chunk_rows * sizeof(float));
  assign_scratch(plan.activation_i8_device, chunk_rows * plan.ffn);
  assign_scratch(plan.fc2_accumulator_device,
                 chunk_rows * plan.hidden * sizeof(std::int32_t));
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

std::uint64_t stage_h3_w8a8_weights(const H3W8A8MlpPlan &plan,
                                    void *staging,
                                    std::uint64_t staging_bytes,
                                    CUstream stream) {
  auto *base = static_cast<std::uint8_t *>(staging);
  auto cursor = std::uint64_t{0U};
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    if (tensor.byte_size() > staging_bytes - cursor)
      fail("H3 W8A8 MLP tail staging overflow");
    std::memcpy(base + cursor, tensor.data(), tensor.byte_size());
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
    assign(plan.activation_scale_device, plan.rows * sizeof(float));
    assign(plan.activation_i8_device, plan.rows * plan.hidden);
    assign(plan.accumulator_device,
           chunk_rows * plan.packed_inner * sizeof(std::int32_t));
  } else {
    assign(plan.activation_scale_device, chunk_rows * sizeof(float));
    assign(plan.activation_i8_device, chunk_rows * plan.inner);
    assign(plan.accumulator_device,
           chunk_rows * plan.hidden * sizeof(std::int32_t));
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

std::uint64_t stage_h3_w8a8_weights(const H3W8A8AttentionPlan &plan,
                                    void *staging,
                                    std::uint64_t staging_bytes,
                                    CUstream stream) {
  auto *base = static_cast<std::uint8_t *>(staging);
  auto cursor = std::uint64_t{0U};
  const auto upload = [&](CUdeviceptr destination, const Tensor &tensor) {
    if (tensor.byte_size() > staging_bytes - cursor)
      fail("H3 W8A8 attention tail staging overflow");
    std::memcpy(base + cursor, tensor.data(), tensor.byte_size());
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

void launch_h3_w8a8_qkv(const H3W8A8AttentionPlan &plan,
                         const H3W8A8Functions &functions,
                         const DeviceBuffers &buffers, cublasHandle_t cublas,
                         CUstream stream) {
  auto input = buffers.at(plan.attention_input_tensor);
  auto scale = plan.activation_scale_device;
  auto zero = 0;
  auto rows = static_cast<int>(plan.rows);
  auto hidden = static_cast<int>(plan.hidden);
  std::array<void *, 5> rowscale_arguments = {
      &input, &scale, &zero, &hidden, &rows};
  launch_h3_w8a8_kernel(functions.rowscale, static_cast<unsigned>(rows),
                         rowscale_arguments, stream,
                         "cuLaunchKernel H3 W8A8 QKV rowscale");

  auto encoded = plan.activation_i8_device;
  std::array<void *, 6> encode_arguments = {
      &input, &scale, &encoded, &zero, &rows, &hidden};
  check(counted_launch_kernel(functions.encode,
                       h3_w8a8_grid(plan.rows * plan.hidden), 1U, 1U, 256U,
                       1U, 1U, 0U, stream, encode_arguments.data(), nullptr),
        "cuLaunchKernel H3 W8A8 QKV encode");

  constexpr std::int32_t alpha = 1;
  constexpr std::int32_t beta = 0;
  auto packed_inner = static_cast<int>(plan.packed_inner);
  auto inner = static_cast<int>(plan.inner);
  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += kH3W8A8ProjectionChunkRows) {
    auto chunk_rows = static_cast<int>(std::min<std::uint64_t>(
        kH3W8A8ProjectionChunkRows, plan.rows - row_start));
    auto chunk_input = plan.activation_i8_device + row_start * plan.hidden;
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, packed_inner, chunk_rows,
              hidden, &alpha,
              reinterpret_cast<const void *>(plan.qkv_weight_device),
              CUDA_R_8I, hidden,
              reinterpret_cast<const void *>(chunk_input), CUDA_R_8I, hidden,
              &beta, reinterpret_cast<void *>(plan.accumulator_device),
              CUDA_R_32I, packed_inner, CUBLAS_COMPUTE_32I,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "cublasGemmEx H3 W8A8 QKV");
    auto accumulator = plan.accumulator_device;
    auto weight_scale = plan.qkv_scale_device;
    auto q = buffers.at(plan.qkv_output_tensors.at(0));
    auto k = buffers.at(plan.qkv_output_tensors.at(1));
    auto v = buffers.at(plan.qkv_output_tensors.at(2));
    auto row_start_i32 = static_cast<int>(row_start);
    std::array<void *, 9> arguments = {
        &accumulator, &scale, &weight_scale, &q, &k, &v,
        &row_start_i32, &chunk_rows, &inner};
    check(counted_launch_kernel(functions.qkv,
                         h3_w8a8_grid(static_cast<std::uint64_t>(chunk_rows) *
                                     plan.inner),
                         1U, 1U, 256U, 1U, 1U, 0U, stream, arguments.data(),
                         nullptr),
          "cuLaunchKernel H3 W8A8 QKV dequant");
  }
}

void launch_h3_w8a8_output(const H3W8A8AttentionPlan &plan,
                            const H3W8A8Functions &functions,
                            const DeviceBuffers &buffers,
                            cublasHandle_t cublas, CUstream stream) {
  auto input = buffers.at(plan.output_input_tensor);
  auto residual = buffers.at(plan.residual_tensor);
  auto gate = buffers.at(plan.gate_tensor);
  auto output = buffers.at(plan.output_tensor);
  auto inner = static_cast<int>(plan.inner);
  auto hidden = static_cast<int>(plan.hidden);
  constexpr std::int32_t alpha = 1;
  constexpr std::int32_t beta = 0;
  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += kH3W8A8ProjectionChunkRows) {
    auto rows = static_cast<int>(std::min<std::uint64_t>(
        kH3W8A8ProjectionChunkRows, plan.rows - row_start));
    auto row_start_i32 = static_cast<int>(row_start);
    auto scale = plan.activation_scale_device;
    std::array<void *, 5> rowscale_arguments = {
        &input, &scale, &row_start_i32, &inner, &rows};
    launch_h3_w8a8_kernel(functions.rowscale, static_cast<unsigned>(rows),
                           rowscale_arguments, stream,
                           "cuLaunchKernel H3 W8A8 output rowscale");
    auto encoded = plan.activation_i8_device;
    std::array<void *, 6> encode_arguments = {
        &input, &scale, &encoded, &row_start_i32, &rows, &inner};
    check(counted_launch_kernel(functions.encode,
                         h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                     plan.inner),
                         1U, 1U, 256U, 1U, 1U, 0U, stream,
                         encode_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 output encode");
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, hidden, rows, inner, &alpha,
              reinterpret_cast<const void *>(plan.output_weight_device),
              CUDA_R_8I, inner,
              reinterpret_cast<const void *>(plan.activation_i8_device),
              CUDA_R_8I, inner, &beta,
              reinterpret_cast<void *>(plan.accumulator_device), CUDA_R_32I,
              hidden, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "cublasGemmEx H3 W8A8 output projection");
    auto accumulator = plan.accumulator_device;
    auto weight_scale = plan.output_scale_device;
    std::array<void *, 9> residual_arguments = {
        &accumulator, &scale, &weight_scale, &residual, &gate, &output,
        &row_start_i32, &rows, &hidden};
    check(counted_launch_kernel(functions.residual,
                         h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                     plan.hidden),
                         1U, 1U, 256U, 1U, 1U, 0U, stream,
                         residual_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 output residual");
  }
}

void launch_h3_w8a8_mlp(const H3W8A8MlpPlan &plan,
                         const H3W8A8Functions &functions,
                         const DeviceBuffers &buffers, cublasHandle_t cublas,
                         CUstream stream) {
  const auto input = buffers.at(plan.input_tensor);
  const auto residual = buffers.at(plan.residual_tensor);
  const auto gate = buffers.at(plan.gate_tensor);
  const auto output = buffers.at(plan.output_tensor);
  auto hidden = static_cast<int>(plan.hidden);
  auto ffn = static_cast<int>(plan.ffn);
  auto packed_ffn = static_cast<int>(plan.packed_ffn);
  constexpr std::int32_t alpha = 1;
  constexpr std::int32_t beta = 0;

  auto gemm = [&](CUdeviceptr activation, CUdeviceptr weight,
                  CUdeviceptr accumulator, int rows, int columns,
                  int contraction) {
    check(counted_cublas_gemm_ex(
              cublas, CUBLAS_OP_T, CUBLAS_OP_N, columns, rows, contraction,
              &alpha, reinterpret_cast<const void *>(weight), CUDA_R_8I,
              contraction, reinterpret_cast<const void *>(activation),
              CUDA_R_8I, contraction, &beta,
              reinterpret_cast<void *>(accumulator), CUDA_R_32I, columns,
              CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP),
          "cublasGemmEx H3 W8A8");
  };

  for (std::uint64_t row_start = 0U; row_start < plan.rows;
       row_start += kH3W8A8MlpChunkRows) {
    auto rows = static_cast<int>(std::min<std::uint64_t>(
        kH3W8A8MlpChunkRows, plan.rows - row_start));
    auto row_start_i32 = static_cast<int>(row_start);

    auto rowscale_input = input;
    auto rowscale_output = plan.input_scale_device;
    std::array<void *, 5> rowscale_arguments = {
        &rowscale_input, &rowscale_output, &row_start_i32, &hidden, &rows};
    launch_h3_w8a8_kernel(functions.rowscale,
                           static_cast<unsigned>(rows), rowscale_arguments,
                           stream, "cuLaunchKernel H3 W8A8 input rowscale");

    auto encode_input = input;
    auto encode_scale = plan.input_scale_device;
    auto encode_output = plan.input_i8_device;
    std::array<void *, 6> full_encode_arguments = {
        &encode_input, &encode_scale, &encode_output,
        &row_start_i32, &rows, &hidden};
    check(counted_launch_kernel(functions.encode,
                         h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                     plan.hidden),
                         1U, 1U, 256U, 1U, 1U, 0U, stream,
                         full_encode_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 input encode");

    gemm(plan.input_i8_device, plan.fc1_weight_device,
         plan.fc1_accumulator_device, rows, packed_ffn, hidden);

    auto swiglu_accumulator = plan.fc1_accumulator_device;
    auto swiglu_input_scale = plan.input_scale_device;
    auto swiglu_weight_scale = plan.fc1_scale_device;
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

    auto rowscale_activation = plan.activation_device;
    auto rowscale_activation_output = plan.activation_scale_device;
    auto zero = 0;
    std::array<void *, 5> activation_rowscale_arguments = {
        &rowscale_activation, &rowscale_activation_output, &zero, &ffn, &rows};
    launch_h3_w8a8_kernel(
        functions.rowscale, static_cast<unsigned>(rows),
        activation_rowscale_arguments, stream,
        "cuLaunchKernel H3 W8A8 activation rowscale");

    auto activation_scale = plan.activation_scale_device;
    auto activation_i8 = plan.activation_i8_device;
    std::array<void *, 6> activation_encode_arguments = {
        &rowscale_activation, &activation_scale, &activation_i8,
        &zero, &rows, &ffn};
    check(counted_launch_kernel(functions.encode,
                         h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                     plan.ffn),
                         1U, 1U, 256U, 1U, 1U, 0U, stream,
                         activation_encode_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 activation encode");

    gemm(plan.activation_i8_device, plan.fc2_weight_device,
         plan.fc2_accumulator_device, rows, hidden, ffn);

    auto residual_accumulator = plan.fc2_accumulator_device;
    auto residual_scale = plan.activation_scale_device;
    auto residual_weight_scale = plan.fc2_scale_device;
    auto residual_input = residual;
    auto residual_gate = gate;
    auto residual_output = output;
    std::array<void *, 9> residual_arguments = {
        &residual_accumulator, &residual_scale, &residual_weight_scale,
        &residual_input, &residual_gate, &residual_output,
        &row_start_i32, &rows, &hidden};
    check(counted_launch_kernel(functions.residual,
                         h3_w8a8_grid(static_cast<std::uint64_t>(rows) *
                                     plan.hidden),
                         1U, 1U, 256U, 1U, 1U, 0U, stream,
                         residual_arguments.data(), nullptr),
          "cuLaunchKernel H3 W8A8 residual");
  }
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
  const std::string key_material = source + "\ncompute_" + std::to_string(major) +
                                   std::to_string(minor) + "\nnvrtc-v1";
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

void validate_inputs(const ir::Program &program, const TensorMap &inputs) {
  for (const auto &desc : program.tensors) {
    if (!desc.has_role(ir::TensorRole::Input) &&
        !desc.has_role(ir::TensorRole::Constant))
      continue;
    const auto it = inputs.find(desc.id);
    if (it == inputs.end())
      fail("missing input tensor " + std::to_string(desc.id));
    it->second.validate();
    if (it->second.dtype != desc.dtype || it->second.dims != desc.dims)
      fail("bound tensor shape/dtype mismatch for id " + std::to_string(desc.id));
  }
  for (const auto &op : program.operations) {
    if (op.opcode == ir::Opcode::H3AdaLNSelect) {
      const auto &projected = *program.tensor(op.inputs[0]);
      const auto &indices = inputs.at(op.inputs[1]);
      const auto table_rows = projected.dims[0] * 3U;
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= table_rows)
          fail("h3_adaln_select index is out of range");
      }
    } else if (op.opcode == ir::Opcode::SelectRowChunks) {
      const auto rows = program.tensor(op.inputs[0])->dims[0];
      const auto &indices = inputs.at(op.inputs[1]);
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= rows)
          fail("select_row_chunks index is out of range");
      }
    } else if (op.opcode == ir::Opcode::GatherRows) {
      const auto rows = program.tensor(op.inputs[0])->dims[0];
      const auto &indices = inputs.at(op.inputs[1]);
      for (std::uint64_t row = 0; row < indices.element_count(); ++row) {
        std::int32_t value = 0;
        std::memcpy(&value, indices.data() + row * sizeof(value), sizeof(value));
        if (value < 0 || static_cast<std::uint64_t>(value) >= rows)
          fail("gather_rows index is out of range");
      }
    } else if (op.opcode == ir::Opcode::IndexedUpdateRows) {
      const auto update_rows = program.tensor(op.inputs[1])->dims[0];
      const auto &map = inputs.at(op.inputs[2]);
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
                     std::unordered_set<std::uint32_t> lazy_resident_overrides)
      : program_(program), constants_(constants), plan_(plan), buffers_(buffers),
        context_(context), staging_pool_(stage_threads),
        resident_overrides_(std::move(resident_overrides)),
        lazy_resident_overrides_(std::move(lazy_resident_overrides)) {
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
        fail("streamed tensor lacks a memory-plan assignment");
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
  }

  void begin_profile(std::uint32_t iterations) {
    profiling_ = true;
    streamed_bytes_ = 0U;
    host_stage_milliseconds_ = 0.0;
    host_wait_milliseconds_ = 0.0;
    next_copy_timing_ = 0U;
    copy_timings_.clear();
    std::size_t count = 0U;
    for (const auto &[id, first] : first_consumer_) {
      (void)first;
      if (!active(id))
        continue;
      count += lazy_resident_overrides_.contains(id)
                   ? 1U
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
    profile.streamed_host_wait_milliseconds = host_wait_milliseconds_;
    profile.streamed_h2d_milliseconds = h2d_milliseconds;
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
    if (copy_recorded_[parity])
      check(counted_event_synchronize(copy_done_[parity]->get()),
            "cuEventSynchronize streamed staging reuse");
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
      staging_pool_.copy(destination, tensor.data(), tensor.byte_size());
      if (profiling_) {
        host_stage_milliseconds_ +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_stage_start)
                .count();
        streamed_bytes_ += tensor.byte_size();
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
             op.opcode == ir::Opcode::LayerNorm) {
    const auto *input = program.tensor(op.inputs[0]);
    grid = static_cast<unsigned>(input->element_count() / input->dims.back());
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::QkNormPartialRope) {
    const auto &dims = program.tensor(op.inputs[0])->dims;
    const auto *table = program.tensor(op.inputs[2]);
    if (program.tensor(op.inputs[0])->dtype == ir::DType::BF16 &&
        dims[2] == 128U &&
        table->dims[1] == op.u64(ir::AttrKey::RotaryDim, dims[2]))
      block = 128U;
    grid = static_cast<unsigned>(dims[0] * dims[1]);
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::ChannelRmsNorm) {
    const auto *input = program.tensor(op.inputs[0]);
    const auto axis = op.u64(ir::AttrKey::Axis, 1U);
    grid = static_cast<unsigned>(input->element_count() / input->dims[axis]);
    shared = block * sizeof(float);
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

class CudaPreparedExecution final : public PreparedExecution {
public:
  CudaPreparedExecution(ir::Program program, const TensorMap &bindings,
                        const RunOptions &options,
                        std::shared_ptr<Context> context)
      : program_(std::move(program)), context_owner_(std::move(context)),
        context_(*context_owner_) {
    const auto preparation_start = std::chrono::steady_clock::now();
    TelemetryScope telemetry_scope(preparation_telemetry_);
    if (options.lazy_resident_upload && options.pipelined_resident_upload)
      fail("lazy and pipelined resident upload are mutually exclusive");
    lazy_resident_upload_ = options.lazy_resident_upload;
    ir::verify(program_);
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
    int major = 0;
    int minor = 0;
    check(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               context_.device()),
          "cuDeviceGetAttribute major");
    check(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               context_.device()),
          "cuDeviceGetAttribute minor");
    std::array<char, 256> device_name{};
    check(cuDeviceGetName(device_name.data(), static_cast<int>(device_name.size()),
                          context_.device()),
          "cuDeviceGetName");
    device_name_ = device_name.data();

    std::size_t free_before = 0;
    std::size_t total = 0;
    check(cuMemGetInfo(&free_before, &total), "cuMemGetInfo before");
    free_bytes_before_ = free_before;
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
    if (!options.h3_w8a8_cache.empty() && h3_w8a8_mlp_plans_.empty() &&
        h3_w8a8_attention_plans_.empty())
      fail("H3 W8A8 cache requested but no supported H3 chain was found");
    if (!options.h3_ck_attention_dso.empty()) {
      auto library = std::make_shared<CkAttentionLibrary>(
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
      for (const auto &operation : program_.operations) {
        if (operation.opcode != ir::Opcode::Attention)
          continue;
        const auto *query = program_.tensor(operation.inputs.front());
        if (!query || query->dims.front() != maximum_sequence)
          continue;
        if (operation.u64(ir::AttrKey::Implementation, 1U) != 2U)
          fail("H3 CK attention can replace only an exact backend Attention");
        if (!h3_ck_attention_plan_)
          h3_ck_attention_plan_ = std::make_shared<CkAttentionPlan>(
              program_, operation, library);
        else if (!h3_ck_attention_plan_->compatible(program_, operation))
          fail("H3 CK attention scratch reuse requires one H3 geometry");
        h3_ck_attention_plans_.emplace(operation.id, h3_ck_attention_plan_);
      }
      if (h3_ck_attention_plans_.empty())
        fail("H3 CK attention DSO requested but no Attention operation exists");
      ck_attention_scratch_bytes_ =
          h3_ck_attention_plan_->scratch_bytes();
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
    generated.source += h3_groupwise_source(!h3_groupwise_plans_.empty());
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
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::Attention ||
          op.u64(ir::AttrKey::Implementation, 1U) != 2U ||
          h3_ck_attention_plans_.contains(op.id))
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
      };
      auto found = cudnn_plan_cache.find(key);
      if (found == cudnn_plan_cache.end()) {
        auto plan = std::make_shared<CudnnAttentionPlan>(
            *query, key.kv_heads, std::bit_cast<double>(key.scale_bits),
            key.causal, key.additive_bias);
        found = cudnn_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_attention_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
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
      };
      auto found = cudnn_conv_plan_cache.find(key);
      if (found == cudnn_conv_plan_cache.end()) {
        auto plan = std::make_shared<CudnnConv2dPlan>(
            *input, *weight, *output, key.stride_h, key.stride_w, key.pad_h,
            key.pad_w, key.dilation_h, key.dilation_w, key.groups, key.biased,
            static_cast<std::size_t>(key.workspace_limit));
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
      };
      auto found = cudnn_conv3d_plan_cache.find(key);
      if (found == cudnn_conv3d_plan_cache.end()) {
        auto plan = std::make_shared<CudnnConv3dPlan>(
            *input, *weight, *output, key.stride_t, key.stride_h,
            key.stride_w, key.pad_t, key.pad_h, key.pad_w,
            key.dilation_t, key.dilation_h, key.dilation_w, key.groups,
            key.biased, static_cast<std::size_t>(key.workspace_limit));
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
        excluded_tensors, replaced_constant_tensors, reshape_aliases_);
    const auto tensor_bytes = memory_plan_.total_bytes;
    if (tensor_bytes > std::numeric_limits<std::uint64_t>::max() - workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() -
                parallel_workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ + parallel_workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() - cudnn_workspace_bytes_)
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
                               cudnn_workspace_bytes_;
    if (base_required > std::numeric_limits<std::uint64_t>::max() -
                            ck_attention_scratch_bytes_)
      fail("DiffIR allocation plus H3 CK scratch overflow");
    const auto base_with_attention =
        base_required + ck_attention_scratch_bytes_;
    if (base_with_attention > std::numeric_limits<std::uint64_t>::max() -
                            h3_w8a8_bytes)
      fail("DiffIR allocation plus H3 W8A8 storage overflow");
    if (base_with_attention + h3_w8a8_bytes >
        std::numeric_limits<std::uint64_t>::max() - h3_groupwise_bytes)
      fail("DiffIR allocation plus H3 groupwise storage overflow");
    const auto base_with_weights =
        base_with_attention + h3_w8a8_bytes + h3_groupwise_bytes;
    if (base_with_weights >
        std::numeric_limits<std::uint64_t>::max() - h3_modulation_bytes)
      fail("DiffIR allocation plus H3 modulation storage overflow");
    if (base_with_weights + h3_modulation_bytes >
        std::numeric_limits<std::uint64_t>::max() - promoted_constant_bytes_)
      fail("DiffIR allocation plus promoted constant storage overflow");
    const auto required =
        base_with_weights + h3_modulation_bytes + promoted_constant_bytes_;
    if (options.profile_pipeline) {
      std::cerr << "CUDA_MEMORY_PLAN tensor_bytes=" << tensor_bytes
                << " linear_workspace_bytes=" << workspace_bytes_
                << " parallel_linear_workspace_bytes="
                << parallel_workspace_bytes_
                << " attention_workspace_bytes=" << cudnn_workspace_bytes_
                << " ck_attention_scratch_bytes="
                << ck_attention_scratch_bytes_
                << " h3_w8a8_weight_bytes=" << h3_w8a8_weight_bytes
                << " h3_w8a8_scratch_bytes=" << h3_w8a8_scratch_bytes
                << " h3_w8a8_tail_weight_bytes="
                << h3_w8a8_tail_weight_bytes_
                << " h3_w8a8_tail_stage_half_bytes="
                << h3_w8a8_tail_stage_half_bytes_
                << " h3_groupwise_bytes=" << h3_groupwise_bytes
                << " h3_modulation_bytes=" << h3_modulation_bytes
                << " promoted_streamed_constant_bytes="
                << promoted_constant_bytes_
                << " required_bytes=" << required
                << " free_before_bytes=" << free_before
                << " minimum_free_bytes=" << options.minimum_free_bytes
                << '\n';
    }
    if (required > free_before ||
        free_before - required < options.minimum_free_bytes)
      fail("GPU pressure gate refused candidate: required=" +
           std::to_string(required) + " free_before=" +
           std::to_string(free_before) + " minimum_free=" +
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
    for (const auto &plan : h3_modulation_cache_plans_)
      skipped_operations_.insert(plan.linear_operation);
    const auto ptx = compile_ptx(generated.source, major, minor,
                                 options.cache_directory, source_hash_);
    module_ = std::make_unique<Module>(ptx);
    for (const auto &[operation, entrypoint] : generated.entrypoints) {
      CUfunction function{};
      check(cuModuleGetFunction(&function, module_->get(), entrypoint.c_str()),
            "cuModuleGetFunction");
      functions_.emplace(operation, function);
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
    if (!h3_groupwise_plans_.empty())
      check(cuModuleGetFunction(&h3_groupwise_dequant_function_, module_->get(),
                                "dif_h3_groupwise_dequant"),
            "cuModuleGetFunction H3 groupwise INT8 dequant");
    excluded_tensors.insert(replaced_constant_tensors.begin(),
                            replaced_constant_tensors.end());
    buffers_.allocate(program_, memory_plan_, excluded_tensors, arena_.get());
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
    h3_w8a8_scratch_storage_ = std::make_unique<Workspace>(
        static_cast<std::size_t>(h3_w8a8_scratch_bytes), arena_.get());
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
        std::unordered_set<std::uint32_t>(
            promoted_streamed_constants_.begin(),
            promoted_streamed_constants_.end()),
        options.lazy_resident_upload
            ? std::unordered_set<std::uint32_t>(
                  promoted_streamed_constants_.begin(),
                  promoted_streamed_constants_.end())
            : std::unordered_set<std::uint32_t>{});
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
            &linear_heuristic_cache_stats_);
        linear_plans_.emplace(op.id, plan);
        if (shareable) {
          std::cout << "CUDA_LINEAR_PLAN_CLASS operation=" << op.id
                    << " key=" << key << "\n";
          shared_plans.emplace(std::move(key), std::move(plan));
        } else
          ++isolated_plan_count;
      }
      const auto unique_plan_count =
          static_cast<std::uint64_t>(shared_plans.size()) +
          isolated_plan_count;
      std::cout << "CUDA_LINEAR_PLAN_POOL operations="
                << linear_operation_count
                << " unique=" << unique_plan_count
                << " reused=" << linear_operation_count - unique_plan_count
                << " isolated=" << isolated_plan_count
                << "\n";
    }
#if DIF_HAS_CUTLASS
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
    for (const auto &plan : h3_w8a8_mlp_plans_)
      if (plan.resident)
        upload_h3_w8a8_weights(plan, context_.stream());
    for (const auto &plan : h3_w8a8_attention_plans_)
      if (plan.resident)
        upload_h3_w8a8_weights(plan, context_.stream());
    for (const auto &plan : h3_groupwise_plans_)
      upload_h3_groupwise_weights(plan, context_.stream());
    upload_h3_modulation_cache(h3_modulation_cache_plans_, context_.stream());
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
    for (const auto &description : program_.tensors) {
      if (description.has_role(ir::TensorRole::Constant) &&
          !description.has_role(ir::TensorRole::Streamed))
        constants_.at(description.id).discard_mapped_pages();
    }
    if (!options.lazy_resident_upload)
      for (const auto id : promoted_streamed_constants_)
        constants_.at(id).discard_mapped_pages();
    for (auto &plan : h3_w8a8_mlp_plans_) {
      plan.fc1_weight.discard_mapped_pages();
      plan.fc1_scale.discard_mapped_pages();
      plan.fc2_weight.discard_mapped_pages();
      plan.fc2_scale.discard_mapped_pages();
    }
    for (auto &plan : h3_w8a8_attention_plans_) {
      if (plan.has_qkv_projection) {
        plan.qkv_weight.discard_mapped_pages();
        plan.qkv_scale.discard_mapped_pages();
      }
      if (plan.has_output_projection) {
        plan.output_weight.discard_mapped_pages();
        plan.output_scale.discard_mapped_pages();
      }
    }
    for (auto &plan : h3_groupwise_plans_) {
      for (auto &projection : plan.projections) {
        projection.weight.discard_mapped_pages();
        projection.scale.discard_mapped_pages();
      }
    }
    for (auto &plan : h3_modulation_cache_plans_)
      plan.modulation.discard_mapped_pages();
    const auto preparation_stop = std::chrono::steady_clock::now();
    preparation_milliseconds_ =
        std::chrono::duration<double, std::milli>(preparation_stop -
                                                  preparation_start)
            .count();
  }

  RunResult run(const TensorMap &inputs, const RunOptions &options) override {
    if (options.iterations == 0)
      fail("run iterations must be nonzero");
    if (options.lazy_resident_upload != lazy_resident_upload_)
      fail("lazy resident upload is fixed when the plan is prepared");
    LaunchTelemetry run_telemetry;
    TelemetryScope telemetry_scope(run_telemetry);
    streamed_prefetcher_->set_release_mapped_pages_per_copy(
        options.streamed_release_mapped_pages_per_copy);
    if (options.streamed_prefetch_depth != streamed_prefetch_depth_)
      fail("streamed prefetch depth is fixed when the plan is prepared");
    TensorMap bindings = constants_;
    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Input))
        continue;
      const auto found = inputs.find(desc.id);
      if (found == inputs.end())
        fail("missing CUDA dynamic input tensor " + std::to_string(desc.id));
      bindings.insert_or_assign(desc.id, found->second);
    }
    validate_inputs(program_, bindings);
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
    auto stage_h3_w8a8_tail = [&](const auto &plan, bool profile) {
      if (plan.resident)
        return;
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
      if (tail_copy_stream) {
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
      auto *staging = static_cast<std::uint8_t *>(
          h3_w8a8_tail_stage_->data()) +
          half * h3_w8a8_tail_stage_half_bytes_;
      const auto bytes = stage_h3_w8a8_weights(
          plan, staging, h3_w8a8_tail_stage_half_bytes_, tail_stream);
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

    auto execute_operation = [&](const ir::Operation &op, bool profile) {
      if (skipped_operations_.contains(op.id))
        return;
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
        launch_h3_w8a8_qkv(*h3_w8a8_qkv, h3_w8a8_functions_, buffers_,
                            context_.cublas(), context_.stream());
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
        launch_h3_w8a8_output(*h3_w8a8_output, h3_w8a8_functions_, buffers_,
                              context_.cublas(), context_.stream());
      }
      else if (const auto h3_w8a8_mlp = std::find_if(
          h3_w8a8_mlp_plans_.begin(), h3_w8a8_mlp_plans_.end(),
          [&](const H3W8A8MlpPlan &plan) {
            return plan.fc1_operation == op.id;
          });
               h3_w8a8_mlp != h3_w8a8_mlp_plans_.end()) {
        stage_h3_w8a8_tail(*h3_w8a8_mlp, profile);
        launch_h3_w8a8_mlp(*h3_w8a8_mlp, h3_w8a8_functions_, buffers_,
                            context_.cublas(), context_.stream());
      }
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
#if DIF_HAS_CUTLASS
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
               h3_ck_attention_plans_.contains(op.id)) {
        count_ck_attention_dispatch();
        h3_ck_attention_plans_.at(op.id)->execute(op, buffers_,
                                                  context_.stream());
      }
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
    if (options.profile_pipeline) {
      const auto count = program_.operations.size() *
                         static_cast<std::size_t>(options.iterations);
      profile_operation_events.reserve(count);
      for (std::size_t index = 0U; index < count; ++index)
        profile_operation_events.push_back(
            {std::make_unique<Event>(), std::make_unique<Event>()});
    }
    auto execute_parallel_group = [&](std::size_t first,
                                      std::uint32_t iteration,
                                      bool profile,
                                      const std::vector<bool> *prefetched) {
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
    auto execute = [&](std::uint32_t iteration, bool profile) {
      if (program_.operations.empty())
        return;
      if (!options.overlap_streaming) {
        for (std::size_t index = 0; index < program_.operations.size(); ++index) {
          if (parallel_linear_followups_.contains(index))
            continue;
          const auto &op = program_.operations[index];
          const auto ready = streamed_prefetcher_->prefetch(index);
          streamed_prefetcher_->wait(index, ready);
          if (execute_parallel_group(index, iteration, profile, nullptr))
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
        prefetched[ahead] = streamed_prefetcher_->prefetch(ahead);
      for (std::size_t index = 0; index < operation_count; ++index) {
        if (parallel_linear_followups_.contains(index)) {
          if (index + depth < operation_count)
            prefetched[index + depth] =
                streamed_prefetcher_->prefetch(index + depth);
          continue;
        }
        const auto &op = program_.operations[index];
        streamed_prefetcher_->wait(index, prefetched[index]);
        if (execute_parallel_group(index, iteration, profile, &prefetched)) {
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
          prefetched[index + depth] =
              streamed_prefetcher_->prefetch(index + depth);
      }
    };

    for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
      execute(0U, false);
      streamed_prefetcher_->complete_iteration();
      check(counted_stream_synchronize(context_.stream()), "warmup synchronization");
    }
    if (options.profile_pipeline)
      streamed_prefetcher_->begin_profile(options.iterations);

    RunResult result;
    result.preparation_milliseconds = preparation_milliseconds_;
    result.resident_bytes = resident_bytes_;
    result.free_bytes_before = free_bytes_before_;
    result.backend_name = name();
    result.device_name = device_name_;
    result.generated_source_hash = source_hash_;
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
           kH3W8A8MlpChunkRows,
           plan.quantized_weight_bytes,
           plan.scratch_bytes,
           plan.eliminated_intermediate_bytes,
           "approximate_w8a8_established_h3_gate",
           "serenity_h3_w8a8_chunked_mlp_residual",
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
           "approximate_w8a8_established_h3_gate",
           "serenity_h3_w8a8_direct_qkv_output_residual",
           plan.cache_path.string(),
           plan.resident});
    result.h3_ck_attentions.reserve(h3_ck_attention_plans_.size());
    for (const auto &[operation_id, plan] : h3_ck_attention_plans_)
      result.h3_ck_attentions.push_back(
          {operation_id,
           static_cast<std::uint32_t>(plan->target_sm()),
           plan->scratch_bytes(),
           "approximate_ck_int8_established_h3_gate",
           "serenity_comfy_kitchen_sage_bf16",
           plan->path().string()});
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
    std::vector<double> elapsed;
    elapsed.reserve(options.iterations);
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        Event start;
        Event stop;
        check(counted_event_record(start.get(), context_.stream()), "cuEventRecord start");
        execute(iteration, options.profile_pipeline);
        streamed_prefetcher_->complete_iteration();
        check(counted_event_record(stop.get(), context_.stream()), "cuEventRecord stop");
        check(counted_event_synchronize(stop.get()), "cuEventSynchronize");
        float milliseconds = 0.0F;
        check(cuEventElapsedTime(&milliseconds, start.get(), stop.get()),
              "cuEventElapsedTime");
        elapsed.push_back(milliseconds);
    }
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
               program_.operations[operation_index].opcode, 0.0, 0.0, 0.0});
          continue;
        }
        std::vector<double> operation_elapsed;
        operation_elapsed.reserve(options.iterations);
        for (std::uint32_t iteration = 0U; iteration < options.iterations;
             ++iteration) {
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
                               operation_elapsed.end())});
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
                                 operation_elapsed.end())});
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
    if (!options.streamed_release_mapped_pages_per_copy)
      streamed_prefetcher_->release_mapped_pages();
    result.preparation_telemetry = preparation_telemetry_;
    result.run_telemetry = run_telemetry;
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
    return result;
  }

  std::string name() const override {
    auto result = std::string("cuda-nvrtc");
    if (!linear_plans_.empty())
      result += "-cublaslt";
#if DIF_HAS_CUTLASS
    if (!cutlass_linear_plans_.empty())
      result += "-cutlass";
#endif
    if (!fused_linear_swiglu_plans_.empty())
      result += "-wmma-swiglu";
    if (!h3_w8a8_mlp_plans_.empty() ||
        !h3_w8a8_attention_plans_.empty())
      result += "-h3-w8a8";
    if (!h3_groupwise_plans_.empty())
      result += "-h3-groupwise-int8";
    if (!h3_ck_attention_plans_.empty())
      result += "-ck-int8";
    if (!h3_modulation_cache_plans_.empty())
      result += "-h3-modcache";
    if (!fused_launch_inputs_.empty())
      result += "-packed-int5";
    if (uses_cudnn_attention_)
      result += "-cudnn";
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
  std::unique_ptr<Workspace> h3_w8a8_scratch_storage_;
  std::unique_ptr<Workspace> h3_w8a8_tail_weight_storage_;
  std::unique_ptr<PinnedHostWorkspace> h3_w8a8_tail_stage_;
  std::array<std::unique_ptr<Event>, 2> h3_w8a8_tail_stage_events_;
  std::unique_ptr<Event> h3_w8a8_tail_order_event_;
  std::unique_ptr<Event> h3_w8a8_tail_ready_event_;
  std::unique_ptr<PinnedHostWorkspace> pinned_io_;
  std::array<bool, 2> h3_w8a8_tail_stage_armed_{};
  std::unique_ptr<Workspace> h3_groupwise_scratch_storage_;
  std::unique_ptr<Workspace> h3_modulation_storage_;
  std::unique_ptr<StreamedPrefetcher> streamed_prefetcher_;
  std::unordered_map<std::uint32_t, CUfunction> functions_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      fused_launch_inputs_;
  std::unordered_set<std::uint32_t> skipped_operations_;
  std::unordered_map<std::uint32_t, std::shared_ptr<LinearPlan>> linear_plans_;
  std::unordered_map<std::size_t, std::vector<std::size_t>>
      parallel_linear_groups_;
  std::unordered_set<std::size_t> parallel_linear_followups_;
#if DIF_HAS_CUTLASS
  std::unordered_map<std::uint32_t, std::unique_ptr<CutlassLinearPlan>>
      cutlass_linear_plans_;
#endif
  std::vector<LinearTuningResult> linear_tuning_results_;
  std::vector<LinearAlgorithmChoice> selected_linear_algorithms_;
  std::vector<FusedLinearSwiGluPlan> fused_linear_swiglu_plans_;
  std::vector<AbsorbedLinearBiasPlan> absorbed_linear_bias_plans_;
  LinearHeuristicCacheStats linear_heuristic_cache_stats_;
  std::vector<H3W8A8MlpPlan> h3_w8a8_mlp_plans_;
  std::vector<H3W8A8AttentionPlan> h3_w8a8_attention_plans_;
  std::vector<H3GroupwiseBlockPlan> h3_groupwise_plans_;
  std::vector<H3ModulationCachePlan> h3_modulation_cache_plans_;
  Tensor h3_modulation_expected_input_;
  std::filesystem::path h3_modulation_input_path_;
  std::uint32_t h3_modulation_slices_{};
  H3W8A8Functions h3_w8a8_functions_;
  CUfunction h3_groupwise_dequant_function_{};
  std::shared_ptr<CkAttentionPlan> h3_ck_attention_plan_;
  std::unordered_map<std::uint32_t, std::shared_ptr<CkAttentionPlan>>
      h3_ck_attention_plans_;
#if DIF_HAS_CUDNN
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnAttentionPlan>>
      cudnn_attention_plans_;
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnConv2dPlan>>
      cudnn_conv_plans_;
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnConv3dPlan>>
      cudnn_conv3d_plans_;
#endif
  std::string device_name_;
  std::string source_hash_;
  std::size_t workspace_bytes_{};
  std::uint64_t parallel_workspace_bytes_{};
  std::size_t cudnn_workspace_bytes_{};
  std::uint64_t ck_attention_scratch_bytes_{};
  std::uint64_t h3_w8a8_tail_attention_bytes_{};
  std::uint64_t h3_w8a8_tail_mlp_bytes_{};
  std::uint64_t h3_w8a8_tail_weight_bytes_{};
  std::uint64_t h3_w8a8_tail_stage_half_bytes_{};
  std::uint64_t h3_w8a8_tail_stage_turn_{};
  std::uint64_t resident_bytes_{};
  std::uint64_t resident_weight_bytes_{};
  std::uint64_t free_bytes_before_{};
  std::uint32_t streamed_prefetch_depth_{1};
  std::vector<std::uint32_t> promoted_streamed_constants_;
  std::unordered_set<std::uint32_t> reshape_alias_operations_;
  std::unordered_map<std::uint32_t, std::uint32_t> reshape_aliases_;
  bool lazy_resident_upload_{};
  std::unique_ptr<Workspace> promoted_constant_storage_;
  std::uint64_t promoted_constant_bytes_{};
  LaunchTelemetry preparation_telemetry_;
  double preparation_milliseconds_{};
  double resident_upload_milliseconds_{};
  double resident_host_prefault_milliseconds_{};
  double resident_h2d_milliseconds_{};
  std::uint64_t resident_minor_page_faults_{};
  std::uint64_t resident_major_page_faults_{};
  std::uint64_t resident_prefault_checksum_{};
  bool uses_cudnn_attention_{};
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
