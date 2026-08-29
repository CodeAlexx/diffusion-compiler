#include "dif/runtime/executor.hpp"

#include "dif/compiler/compiler.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/verify.hpp"
#if DIF_HAS_CUDNN
#include "dif/runtime/cudnn_attention.hpp"
#endif
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <cuda.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <nvrtc.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

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

class DeviceBuffers {
public:
  ~DeviceBuffers() {
    for (const auto pointer : allocations_) {
      if (pointer)
        (void)cuMemFree(pointer);
    }
  }

  void allocate(
      const ir::Program &program, const compiler::MemoryPlan &plan,
      const std::unordered_set<std::uint32_t> &excluded_tensors = {}) {
    allocations_.resize(plan.slots.size());
    for (const auto &slot : plan.slots) {
      CUdeviceptr pointer{};
      check(cuMemAlloc(&pointer, static_cast<std::size_t>(slot.bytes)),
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

private:
  std::vector<CUdeviceptr> allocations_;
  std::unordered_map<std::uint32_t, CUdeviceptr> pointers_;
};

class Workspace {
public:
  explicit Workspace(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0U)
      check(cuMemAlloc(&pointer_, bytes_), "cuMemAlloc cuBLASLt workspace");
  }
  ~Workspace() {
    if (pointer_)
      (void)cuMemFree(pointer_);
  }
  Workspace(const Workspace &) = delete;
  Workspace &operator=(const Workspace &) = delete;
  void *data() const { return reinterpret_cast<void *>(pointer_); }
  std::size_t size() const { return bytes_; }

private:
  CUdeviceptr pointer_{};
  std::size_t bytes_{};
};

class PinnedHostWorkspace {
public:
  explicit PinnedHostWorkspace(std::size_t bytes) : bytes_(bytes) {
    if (bytes_ != 0U)
      check(cuMemHostAlloc(&pointer_, bytes_, CU_MEMHOSTALLOC_PORTABLE),
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
  std::uint64_t sequence{};
  std::uint64_t heads{};
  std::uint64_t head_dim{};
  std::uint64_t scale_bits{};
  bool causal{};

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
    mix(key.sequence);
    mix(key.heads);
    mix(key.head_dim);
    mix(key.scale_bits);
    mix(key.causal ? 1U : 0U);
    return result;
  }
};
#endif

class LinearPlan {
public:
  LinearPlan(const ir::Program &program, const ir::Operation &op,
             const DeviceBuffers &buffers, cublasLtHandle_t handle,
             std::size_t workspace_bytes) {
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
    if (has_bias_) {
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
      const auto preference_workspace =
          input->dtype == ir::DType::F32 ? 1U * 1024U * 1024U
                                         : workspace_bytes;
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
      int returned = 0;
      const auto matrix_a = has_bias_ ? weight_ : input_;
      const auto matrix_b = has_bias_ ? input_ : weight_;
      check(cublasLtMatmulAlgoGetHeuristic(handle, operation_, matrix_a,
                                            matrix_b, output_, output_,
                                            preference, 1, &heuristic_,
                                            &returned),
            "cublasLtMatmulAlgoGetHeuristic");
      if (returned != 1)
        fail("cuBLASLt found no admitted Linear algorithm");
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

  void launch(const ir::Operation &op, const DeviceBuffers &buffers,
              cublasLtHandle_t handle, const Workspace &workspace,
              CUstream stream) const {
    constexpr float alpha = 1.0F;
    constexpr float beta = 0.0F;
    const auto input_pointer = reinterpret_cast<const void *>(buffers.at(op.inputs[0]));
    const auto weight_pointer = reinterpret_cast<const void *>(buffers.at(op.inputs[1]));
    const auto output_pointer = reinterpret_cast<void *>(buffers.at(op.outputs[0]));
    if (has_bias_) {
      const auto bias_pointer =
          reinterpret_cast<const void *>(buffers.at(op.inputs[2]));
      check(cublasLtMatmulDescSetAttribute(
                operation_, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias_pointer,
                sizeof(bias_pointer)),
            "cublasLtMatmulDescSetAttribute bias pointer");
    }
    const auto matrix_a_pointer = has_bias_ ? weight_pointer : input_pointer;
    const auto matrix_b_pointer = has_bias_ ? input_pointer : weight_pointer;
    const auto matrix_a = has_bias_ ? weight_ : input_;
    const auto matrix_b = has_bias_ ? input_ : weight_;
    check(cublasLtMatmul(
              handle, operation_, &alpha, matrix_a_pointer, matrix_a,
              matrix_b_pointer, matrix_b, &beta, output_pointer, output_,
              output_pointer, output_, &heuristic_.algo, workspace.data(),
              workspace.size(), reinterpret_cast<cudaStream_t>(stream)),
          "cublasLtMatmul Linear");
  }

private:
  cublasLtMatmulDesc_t operation_{};
  cublasLtMatrixLayout_t input_{};
  cublasLtMatrixLayout_t weight_{};
  cublasLtMatrixLayout_t output_{};
  cublasLtMatmulHeuristicResult_t heuristic_{};
  bool has_bias_{};
};

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
  const std::array<const char *, 3> options = {
      "--std=c++17", architecture.c_str(), "--restrict"};
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

class StreamedPrefetcher {
public:
  StreamedPrefetcher(const ir::Program &program, const TensorMap &constants,
                     const compiler::MemoryPlan &plan, DeviceBuffers &buffers,
                     Context &context)
      : program_(program), constants_(constants), plan_(plan), buffers_(buffers),
        context_(context) {
    std::uint64_t maximum = 0U;
    for (const auto &op : program_.operations) {
      std::uint64_t bytes = 0U;
      for (const auto id : op.inputs) {
        const auto *desc = program_.tensor(id);
        if (!desc || !desc->has_role(ir::TensorRole::Streamed))
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
    for (auto &staging : staging_)
      staging = std::make_unique<PinnedHostWorkspace>(
          static_cast<std::size_t>(maximum));
    for (auto &event : copy_done_)
      event = std::make_unique<Event>(CU_EVENT_DISABLE_TIMING);
    copy_recorded_.fill(false);
    ready_events_.reserve(program_.operations.size());
    completion_events_.reserve(program_.operations.size());
    for (std::size_t index = 0; index < program_.operations.size(); ++index) {
      ready_events_.push_back(
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING));
      completion_events_.push_back(
          std::make_unique<Event>(CU_EVENT_DISABLE_TIMING));
      for (const auto id : program_.operations[index].inputs) {
        const auto *desc = program_.tensor(id);
        if (desc && desc->has_role(ir::TensorRole::Streamed))
          first_consumer_.try_emplace(id, index);
      }
    }
    completion_recorded_.resize(program_.operations.size(), false);
    for (const auto &[id, first] : first_consumer_) {
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
    const auto count = first_consumer_.size() *
                       static_cast<std::size_t>(iterations);
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
                      (force || first_consumer_.at(id) == operation_index));
    }
    if (!has_streamed)
      return false;
    const auto host_wait_start = std::chrono::steady_clock::now();
    if (copy_recorded_[parity])
      check(cuEventSynchronize(copy_done_[parity]->get()),
            "cuEventSynchronize streamed staging reuse");
    if (profiling_)
      host_wait_milliseconds_ +=
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - host_wait_start)
              .count();

    for (const auto id : op.inputs) {
      const auto *desc = program_.tensor(id);
      if (!desc || !desc->has_role(ir::TensorRole::Streamed))
        continue;
      if (!force && first_consumer_.at(id) != operation_index)
        continue;
      const auto wait = overwrite_wait_operation_.at(id);
      if (wait != std::numeric_limits<std::size_t>::max() &&
          completion_recorded_.at(wait))
        check(cuStreamWaitEvent(
                  context_.copy_stream(), completion_events_.at(wait)->get(), 0U),
              "cuStreamWaitEvent streamed slot release");
    }

    std::size_t offset = 0U;
    for (const auto id : op.inputs) {
      const auto *desc = program_.tensor(id);
      if (!desc || !desc->has_role(ir::TensorRole::Streamed))
        continue;
      if (!force && first_consumer_.at(id) != operation_index)
        continue;
      const auto &tensor = constants_.at(id);
      if (tensor.byte_size() > staging_[parity]->size() - offset)
        fail("streamed tensor exceeds prefetch staging capacity");
      auto *destination = static_cast<std::uint8_t *>(staging_[parity]->data()) +
                          offset;
      const auto host_stage_start = std::chrono::steady_clock::now();
      std::memcpy(destination, tensor.data(), tensor.byte_size());
      if (profiling_) {
        host_stage_milliseconds_ +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - host_stage_start)
                .count();
        streamed_bytes_ += tensor.byte_size();
        if (next_copy_timing_ >= copy_timings_.size())
          fail("streamed pipeline profile observed an unexpected weight copy");
        auto &timing = copy_timings_.at(next_copy_timing_++);
        check(cuEventRecord(timing.start->get(), context_.copy_stream()),
              "cuEventRecord streamed H2D start");
        check(cuMemcpyHtoDAsync(buffers_.at(id), destination,
                                tensor.byte_size(), context_.copy_stream()),
              "cuMemcpyHtoDAsync profiled constant");
        check(cuEventRecord(timing.stop->get(), context_.copy_stream()),
              "cuEventRecord streamed H2D stop");
      } else {
        check(cuMemcpyHtoDAsync(buffers_.at(id), destination,
                                tensor.byte_size(), context_.copy_stream()),
              "cuMemcpyHtoDAsync prefetched constant");
      }
      offset += tensor.byte_size();
      tensor.discard_mapped_pages();
    }
    check(cuEventRecord(ready_events_.at(operation_index)->get(),
                        context_.copy_stream()),
          "cuEventRecord streamed readiness");
    check(cuEventRecord(copy_done_[parity]->get(), context_.copy_stream()),
          "cuEventRecord streamed staging copy");
    copy_recorded_[parity] = true;
    return true;
  }

  void wait(std::size_t operation_index, bool ready) {
    if (!ready)
      return;
    check(cuStreamWaitEvent(context_.stream(),
                            ready_events_.at(operation_index)->get(), 0U),
          "cuStreamWaitEvent prefetched constant");
  }

  void complete(std::size_t operation_index) {
    check(cuEventRecord(completion_events_.at(operation_index)->get(),
                        context_.stream()),
          "cuEventRecord operation completion");
    completion_recorded_.at(operation_index) = true;
  }

private:
  struct CopyTiming {
    std::unique_ptr<Event> start;
    std::unique_ptr<Event> stop;
  };

  const ir::Program &program_;
  const TensorMap &constants_;
  const compiler::MemoryPlan &plan_;
  DeviceBuffers &buffers_;
  Context &context_;
  std::array<std::unique_ptr<PinnedHostWorkspace>, 2> staging_;
  std::array<std::unique_ptr<Event>, 2> copy_done_;
  std::array<bool, 2> copy_recorded_{};
  std::vector<std::unique_ptr<Event>> ready_events_;
  std::vector<std::unique_ptr<Event>> completion_events_;
  std::vector<bool> completion_recorded_;
  std::unordered_map<std::uint32_t, std::size_t> first_consumer_;
  std::unordered_map<std::uint32_t, std::size_t> overwrite_wait_operation_;
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
    const auto &tensor = inputs.at(desc.id);
    check(cuMemcpyHtoDAsync(buffers.at(desc.id), tensor.data(),
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
    check(cuMemcpyHtoDAsync(buffers.at(desc.id), tensor.data(),
                            tensor.byte_size(), stream),
          "cuMemcpyHtoDAsync dynamic input");
  }
}

void launch(const ir::Program &program, const ir::Operation &op, CUfunction function,
            DeviceBuffers &buffers, CUstream stream,
            const std::vector<std::uint32_t> *input_override = nullptr) {
  if (op.opcode == ir::Opcode::Barrier) {
    check(cuStreamSynchronize(stream), "cuStreamSynchronize barrier");
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
    grid = static_cast<unsigned>(dims[0] * dims[1]);
    shared = block * sizeof(float);
  } else if (op.opcode == ir::Opcode::Attention) {
    const auto &dims = program.tensor(op.inputs[0])->dims;
    block = std::min<unsigned>(block, 256U);
    grid = static_cast<unsigned>(dims[0] * dims[1]);
    shared = static_cast<unsigned>((block + dims[0]) * sizeof(float));
  } else {
    const auto count = program.tensor(op.outputs[0])->element_count();
    grid = static_cast<unsigned>((count + block - 1U) / block);
  }

  check(cuLaunchKernel(function, grid, 1, 1, block, 1, 1, shared, stream,
                       arguments.data(), nullptr),
        "cuLaunchKernel");
}

class CudaPreparedExecution final : public PreparedExecution {
public:
  CudaPreparedExecution(ir::Program program, const TensorMap &bindings,
                        const RunOptions &options, int device_ordinal)
      : program_(std::move(program)), context_(device_ordinal) {
    const auto preparation_start = std::chrono::steady_clock::now();
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
    const auto generated = compiler::emit_cuda(program_);
    std::unordered_set<std::uint32_t> excluded_tensors;
    for (const auto &operation : program_.operations) {
      if (!generated.skipped_operations.contains(operation.id))
        continue;
      for (const auto output : operation.outputs)
        excluded_tensors.insert(output);
    }
    constexpr std::size_t linear_workspace_bytes = 64U * 1024U * 1024U;
    const bool contains_linear = std::any_of(
        program_.operations.begin(), program_.operations.end(),
        [&](const ir::Operation &op) {
          return op.opcode == ir::Opcode::Linear &&
                 !generated.launch_inputs.contains(op.id);
        });
    workspace_bytes_ = contains_linear ? linear_workspace_bytes : 0U;
    cudnn_workspace_bytes_ = 0U;
#if DIF_HAS_CUDNN
    std::unordered_map<CudnnAttentionKey,
                       std::shared_ptr<CudnnAttentionPlan>,
                       CudnnAttentionKeyHash>
        cudnn_plan_cache;
    for (const auto &op : program_.operations) {
      if (op.opcode != ir::Opcode::Attention ||
          op.u64(ir::AttrKey::Implementation, 1U) != 2U)
        continue;
      const auto *query = program_.tensor(op.inputs.at(0));
      if (!query)
        fail("cuDNN attention references a missing query tensor");
      const CudnnAttentionKey key{
          query->dtype,
          query->dims.at(0),
          query->dims.at(1),
          query->dims.at(2),
          std::bit_cast<std::uint64_t>(op.f64(
              ir::AttrKey::AttentionScale,
              1.0 / std::sqrt(static_cast<double>(query->dims.at(2))))),
          op.boolean(ir::AttrKey::Causal, false),
      };
      auto found = cudnn_plan_cache.find(key);
      if (found == cudnn_plan_cache.end()) {
        auto plan = std::make_shared<CudnnAttentionPlan>(
            *query, std::bit_cast<double>(key.scale_bits), key.causal);
        found = cudnn_plan_cache.emplace(key, std::move(plan)).first;
      }
      cudnn_attention_plans_.emplace(op.id, found->second);
      cudnn_workspace_bytes_ =
          std::max(cudnn_workspace_bytes_, found->second->workspace_bytes());
      uses_cudnn_attention_ = true;
    }
#else
    for (const auto &op : program_.operations) {
      if (op.opcode == ir::Opcode::Attention &&
          op.u64(ir::AttrKey::Implementation, 1U) == 2U)
        fail("DiffIR requests cuDNN attention but this CUDA backend was built without cuDNN");
    }
#endif
    memory_plan_ = compiler::plan_memory(
        program_, 256U, options.overlap_streaming ? 1U : 0U,
        excluded_tensors);
    const auto tensor_bytes = memory_plan_.total_bytes;
    if (tensor_bytes > std::numeric_limits<std::uint64_t>::max() - workspace_bytes_ ||
        tensor_bytes + workspace_bytes_ >
            std::numeric_limits<std::uint64_t>::max() - cudnn_workspace_bytes_)
      fail("DiffIR allocation plus backend workspace overflow");
    const auto required = tensor_bytes + workspace_bytes_ + cudnn_workspace_bytes_;
    if (required > free_before || free_before - required < options.minimum_free_bytes)
      fail("GPU pressure gate refused candidate: insufficient free memory");
    resident_bytes_ = required;

    fused_launch_inputs_ = generated.launch_inputs;
    skipped_operations_ = generated.skipped_operations;
    const auto ptx = compile_ptx(generated.source, major, minor,
                                 options.cache_directory, source_hash_);
    module_ = std::make_unique<Module>(ptx);
    for (const auto &[operation, entrypoint] : generated.entrypoints) {
      CUfunction function{};
      check(cuModuleGetFunction(&function, module_->get(), entrypoint.c_str()),
            "cuModuleGetFunction");
      functions_.emplace(operation, function);
    }
    buffers_.allocate(program_, memory_plan_, excluded_tensors);
    workspace_ = std::make_unique<Workspace>(workspace_bytes_);
    cudnn_workspace_ = std::make_unique<Workspace>(cudnn_workspace_bytes_);
    streamed_prefetcher_ = std::make_unique<StreamedPrefetcher>(
        program_, constants_, memory_plan_, buffers_, context_);
    for (const auto &op : program_.operations) {
      if (op.opcode == ir::Opcode::Linear &&
          !fused_launch_inputs_.contains(op.id))
        linear_plans_.emplace(
            op.id,
            std::make_unique<LinearPlan>(program_, op, buffers_,
                                         context_.cublas_lt(), workspace_bytes_));
    }
    for (const auto &description : program_.tensors) {
      if (description.has_role(ir::TensorRole::Constant) &&
          !description.has_role(ir::TensorRole::Streamed))
        resident_weight_bytes_ += description.byte_count();
    }
    const auto resident_upload_start = std::chrono::steady_clock::now();
    upload_resident_constants(program_, constants_, buffers_, context_.stream());
    check(cuStreamSynchronize(context_.stream()),
          "resident constant upload synchronization");
    if (resident_weight_bytes_ != 0U)
      resident_upload_milliseconds_ =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - resident_upload_start)
              .count();
    for (const auto &description : program_.tensors) {
      if (description.has_role(ir::TensorRole::Constant) &&
          !description.has_role(ir::TensorRole::Streamed))
        constants_.at(description.id).discard_mapped_pages();
    }
    const auto preparation_stop = std::chrono::steady_clock::now();
    preparation_milliseconds_ =
        std::chrono::duration<double, std::milli>(preparation_stop -
                                                  preparation_start)
            .count();
  }

  RunResult run(const TensorMap &inputs, const RunOptions &options) override {
    if (options.iterations == 0)
      fail("run iterations must be nonzero");
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
    upload_dynamic_inputs(program_, bindings, buffers_, context_.stream());
    check(cuStreamSynchronize(context_.stream()),
          "dynamic input upload synchronization");

    auto execute_operation = [&](const ir::Operation &op) {
      if (skipped_operations_.contains(op.id))
        return;
      if (const auto fused = fused_launch_inputs_.find(op.id);
          fused != fused_launch_inputs_.end())
        launch(program_, op, functions_.at(op.id), buffers_, context_.stream(),
               &fused->second);
      else if (op.opcode == ir::Opcode::Linear)
        linear_plans_.at(op.id)->launch(op, buffers_, context_.cublas_lt(),
                                        *workspace_, context_.stream());
#if DIF_HAS_CUDNN
      else if (op.opcode == ir::Opcode::Attention &&
               op.u64(ir::AttrKey::Implementation, 1U) == 2U) {
        cudnn_attention_plans_.at(op.id)->execute(
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(0))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(1))),
            static_cast<std::uintptr_t>(buffers_.at(op.inputs.at(2))),
            static_cast<std::uintptr_t>(buffers_.at(op.outputs.at(0))),
            reinterpret_cast<std::uintptr_t>(cudnn_workspace_->data()),
            reinterpret_cast<std::uintptr_t>(context_.stream()));
      }
#endif
      else if (op.opcode == ir::Opcode::Barrier)
        launch(program_, op, nullptr, buffers_, context_.stream());
      else
        launch(program_, op, functions_.at(op.id), buffers_, context_.stream());
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
    auto execute = [&](std::uint32_t iteration, bool profile) {
      if (program_.operations.empty())
        return;
      if (!options.overlap_streaming) {
        for (std::size_t index = 0; index < program_.operations.size(); ++index) {
          const auto &op = program_.operations[index];
          const auto ready = streamed_prefetcher_->prefetch(index);
          streamed_prefetcher_->wait(index, ready);
          OperationEventPair *timing = nullptr;
          if (profile && !skipped_operations_.contains(op.id)) {
            timing = &profile_operation_events.at(
                static_cast<std::size_t>(iteration) *
                    program_.operations.size() +
                index);
            check(cuEventRecord(timing->start->get(), context_.stream()),
                  "cuEventRecord profiled operation start");
          }
          execute_operation(op);
          if (timing)
            check(cuEventRecord(timing->stop->get(), context_.stream()),
                  "cuEventRecord profiled operation stop");
          streamed_prefetcher_->complete(index);
        }
        return;
      }
      auto ready = streamed_prefetcher_->prefetch(0U);
      for (std::size_t index = 0; index < program_.operations.size(); ++index) {
        const auto &op = program_.operations[index];
        streamed_prefetcher_->wait(index, ready);
        OperationEventPair *timing = nullptr;
        if (profile && !skipped_operations_.contains(op.id)) {
          timing = &profile_operation_events.at(
              static_cast<std::size_t>(iteration) *
                  program_.operations.size() +
              index);
          check(cuEventRecord(timing->start->get(), context_.stream()),
                "cuEventRecord profiled operation start");
        }
        execute_operation(op);
        if (timing)
          check(cuEventRecord(timing->stop->get(), context_.stream()),
                "cuEventRecord profiled operation stop");
        streamed_prefetcher_->complete(index);
        ready = index + 1U < program_.operations.size()
                    ? streamed_prefetcher_->prefetch(index + 1U)
                    : false;
      }
    };

    for (std::uint32_t warmup = 0; warmup < options.warmups; ++warmup) {
      execute(0U, false);
      check(cuStreamSynchronize(context_.stream()), "warmup synchronization");
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
    std::vector<double> elapsed;
    elapsed.reserve(options.iterations);
    for (std::uint32_t iteration = 0; iteration < options.iterations; ++iteration) {
        Event start;
        Event stop;
        check(cuEventRecord(start.get(), context_.stream()), "cuEventRecord start");
        execute(iteration, options.profile_pipeline);
        check(cuEventRecord(stop.get(), context_.stream()), "cuEventRecord stop");
        check(cuEventSynchronize(stop.get()), "cuEventSynchronize");
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
    }

    if (options.trace_operations && !options.profile_pipeline) {
      for (const auto &op : program_.operations) {
          std::vector<double> operation_elapsed;
          operation_elapsed.reserve(options.iterations);
          for (std::uint32_t iteration = 0; iteration < options.iterations;
               ++iteration) {
            Event start;
            Event stop;
            check(cuEventRecord(start.get(), context_.stream()),
                  "cuEventRecord operation start");
            const auto index = static_cast<std::size_t>(&op -
                program_.operations.data());
            const auto ready = streamed_prefetcher_->prefetch(index, true);
            streamed_prefetcher_->wait(index, ready);
            execute_operation(op);
            streamed_prefetcher_->complete(index);
            check(cuEventRecord(stop.get(), context_.stream()),
                  "cuEventRecord operation stop");
            check(cuEventSynchronize(stop.get()),
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

    for (const auto &desc : program_.tensors) {
      if (!desc.has_role(ir::TensorRole::Output))
        continue;
      auto tensor = zeros(desc);
      check(cuMemcpyDtoHAsync(tensor.mutable_data(), buffers_.at(desc.id),
                              tensor.byte_size(), context_.stream()),
            "cuMemcpyDtoHAsync");
      result.outputs.emplace(desc.id, std::move(tensor));
    }
    check(cuStreamSynchronize(context_.stream()), "output copy synchronization");

    std::size_t free_after = 0;
    std::size_t total = 0;
    check(cuMemGetInfo(&free_after, &total), "cuMemGetInfo after");
    result.free_bytes_after = free_after;
    return result;
  }

  std::string name() const override {
    auto result = std::string("cuda-nvrtc");
    if (!linear_plans_.empty())
      result += "-cublaslt";
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
  Context context_;
  std::unique_ptr<Module> module_;
  compiler::MemoryPlan memory_plan_;
  DeviceBuffers buffers_;
  std::unique_ptr<Workspace> workspace_;
  std::unique_ptr<Workspace> cudnn_workspace_;
  std::unique_ptr<StreamedPrefetcher> streamed_prefetcher_;
  std::unordered_map<std::uint32_t, CUfunction> functions_;
  std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
      fused_launch_inputs_;
  std::unordered_set<std::uint32_t> skipped_operations_;
  std::unordered_map<std::uint32_t, std::unique_ptr<LinearPlan>> linear_plans_;
#if DIF_HAS_CUDNN
  std::unordered_map<std::uint32_t, std::shared_ptr<CudnnAttentionPlan>>
      cudnn_attention_plans_;
#endif
  std::string device_name_;
  std::string source_hash_;
  std::size_t workspace_bytes_{};
  std::size_t cudnn_workspace_bytes_{};
  std::uint64_t resident_bytes_{};
  std::uint64_t resident_weight_bytes_{};
  std::uint64_t free_bytes_before_{};
  double preparation_milliseconds_{};
  double resident_upload_milliseconds_{};
  bool uses_cudnn_attention_{};
};

class CudaExecutor final : public Executor {
public:
  explicit CudaExecutor(int device) : device_ordinal_(device) {}

  std::unique_ptr<PreparedExecution>
  prepare(const ir::Program &program, const TensorMap &bindings,
          const RunOptions &options) override {
    return std::make_unique<CudaPreparedExecution>(program, bindings, options,
                                                   device_ordinal_);
  }

  std::string name() const override { return "cuda-nvrtc-cublaslt"; }

private:
  int device_ordinal_{};
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
