// Native generic and MiniMax-H3 ConvRot INT8 projection-cache builder.
//
// Reads the official sharded BF16 transformer checkpoint, applies the H256
// block rotation and deterministic per-output-row INT8 quantization on the
// installed NVIDIA GPU, and emits only the four projection matrices used by
// each H3 block.  The resulting cache is a derived Diffusion Compiler artifact;
// it is not a ComfyUI checkpoint and the production runtime does not import
// Python, PyTorch, or Comfy Kitchen.

#include "dif/runtime/tensor.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#if DIF_HAS_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace {

constexpr std::uint32_t kFormatMagic = 0x44494643U; // "DIFC"
constexpr std::uint32_t kFormatVersion = 1U;
constexpr std::uint32_t kChunkScaledFormatVersion = 2U;
constexpr std::uint32_t kGroupSize = 256U;
constexpr std::uint32_t kQkvLayoutContiguous = 1U;
constexpr std::uint32_t kProjectionCount = 4U;
constexpr std::uint64_t kH3Hidden = 5376U;
constexpr std::uint64_t kH3Inner = 7168U;
constexpr std::uint64_t kH3HeadDim = 128U;
constexpr std::uint64_t kH3Ffn = 14336U;
constexpr std::uint32_t kGenericFormatMagic = 0x31525643U; // "CVR1"
constexpr std::uint32_t kGenericFormatVersion = 1U;
constexpr std::array<std::uint32_t, 4> kH3QualityGroups = {16U, 64U, 32U,
                                                            64U};

struct Projection {
  std::string source_name;
  std::string output_weight_name;
  std::string output_scale_name;
  std::vector<std::uint64_t> dims;
  bool reorder_qkv{};
  std::uint32_t group_size{};
  bool swap_row_halves{};
};

struct GenericProjection {
  std::uint32_t tensor_id{};
  std::uint32_t shard_index{};
  dif::ir::DType dtype{};
  std::string source_name;
  std::string output_weight_name;
  std::string output_scale_name;
  std::vector<std::uint64_t> dims;
};

class PartialFileCleanup {
public:
  explicit PartialFileCleanup(std::filesystem::path path)
      : path_(std::move(path)) {}
  ~PartialFileCleanup() {
    if (!armed_)
      return;
    std::error_code error;
    std::filesystem::remove(path_, error);
  }
  void disarm() { armed_ = false; }

private:
  std::filesystem::path path_;
  bool armed_{true};
};

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr << "difconvrot: " << message << "\n"
            << "usage: difconvrot --program P.difir --bundle B.difbind "
               "--output OUT.safetensors\n"
            << "       difconvrot --program P.difir --bundle B.difbind "
               "--rebind-cache CACHE.safetensors\n"
            << "       difh3convrot OFFICIAL.index.json OUT.safetensors "
               "[--layers N] [--convrot-scale-chunk N | --groupwise-quality] "
               "[--quality-groups QKV OUT FC1 FC2] [--quality-scale-f32]\n";
  std::exit(2);
}

std::uint32_t parse_layers(const std::string &text) {
  std::size_t consumed = 0U;
  const auto value = std::stoull(text, &consumed, 10);
  if (consumed != text.size() || value == 0U || value > 50U)
    usage_error("--layers must be in [1,50]");
  return static_cast<std::uint32_t>(value);
}

std::filesystem::path partial_path(const std::filesystem::path &path) {
  return path.string() + ".partial";
}

std::vector<Projection> projections(std::uint32_t layers) {
  std::vector<Projection> result;
  result.reserve(static_cast<std::size_t>(layers) * kProjectionCount);
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const auto source = "blocks." + std::to_string(layer);
    const auto output = "block." + std::to_string(layer);
    result.push_back({source + ".attn.qkv_proj.weight",
                      output + ".convrot_weight.0",
                      output + ".convrot_scale.0",
                      {3U * kH3Inner, kH3Hidden}, true});
    result.push_back({source + ".attn.out_proj.weight",
                      output + ".convrot_weight.1",
                      output + ".convrot_scale.1",
                      {kH3Hidden, kH3Inner}, false});
    result.push_back({source + ".mlp.fc1.weight",
                      output + ".convrot_weight.2",
                      output + ".convrot_scale.2",
                      {2U * kH3Ffn, kH3Hidden}, false});
    result.push_back({source + ".mlp.fc2.weight",
                      output + ".convrot_weight.3",
                      output + ".convrot_scale.3",
                      {kH3Hidden, kH3Ffn}, false});
  }
  return result;
}

std::vector<Projection>
groupwise_projections(std::uint32_t layers,
                      const std::array<std::uint32_t, 4> &groups) {
  std::vector<Projection> result;
  result.reserve(static_cast<std::size_t>(layers) * kProjectionCount);
  for (std::uint32_t layer = 0; layer < layers; ++layer) {
    const auto source = "blocks." + std::to_string(layer);
    const auto output = "block." + std::to_string(layer);
    result.push_back({source + ".attn.qkv_proj.weight",
                      output + ".weight.0", output + ".scale.0",
                      {3U * kH3Inner, kH3Hidden}, true,
                      groups[0], false});
    result.push_back({source + ".attn.out_proj.weight",
                      output + ".weight.1", output + ".scale.1",
                      {kH3Hidden, kH3Inner}, false, groups[1],
                      false});
    result.push_back({source + ".mlp.fc1.weight", output + ".weight.2",
                      output + ".scale.2", {2U * kH3Ffn, kH3Hidden}, false,
                      groups[2], true});
    result.push_back({source + ".mlp.fc2.weight", output + ".weight.3",
                      output + ".scale.3", {kH3Hidden, kH3Ffn}, false,
                      groups[3], false});
  }
  return result;
}

std::array<std::uint32_t, 14>
format_metadata(std::uint32_t layers, const dif::Sha256Digest &source_digest) {
  std::array<std::uint32_t, 14> values{};
  values[0] = kFormatMagic;
  values[1] = kFormatVersion;
  values[2] = kGroupSize;
  values[3] = kQkvLayoutContiguous;
  values[4] = layers;
  values[5] = kProjectionCount;
  for (std::size_t word = 0; word < 8U; ++word) {
    values[6U + word] = static_cast<std::uint32_t>(source_digest[word * 4U]) |
                        (static_cast<std::uint32_t>(source_digest[word * 4U + 1U]) << 8U) |
                        (static_cast<std::uint32_t>(source_digest[word * 4U + 2U]) << 16U) |
                        (static_cast<std::uint32_t>(source_digest[word * 4U + 3U]) << 24U);
  }
  return values;
}

std::array<std::uint32_t, 15> chunk_scaled_format_metadata(
    std::uint32_t layers, std::uint32_t scale_chunk,
    const dif::Sha256Digest &source_digest) {
  std::array<std::uint32_t, 15> values{};
  values[0] = kFormatMagic;
  values[1] = kChunkScaledFormatVersion;
  values[2] = kGroupSize;
  values[3] = kQkvLayoutContiguous;
  values[4] = layers;
  values[5] = kProjectionCount;
  values[6] = scale_chunk;
  for (std::size_t word = 0; word < 8U; ++word) {
    values[7U + word] =
        static_cast<std::uint32_t>(source_digest[word * 4U]) |
        (static_cast<std::uint32_t>(source_digest[word * 4U + 1U]) << 8U) |
        (static_cast<std::uint32_t>(source_digest[word * 4U + 2U]) << 16U) |
        (static_cast<std::uint32_t>(source_digest[word * 4U + 3U]) << 24U);
  }
  return values;
}

std::array<std::uint32_t, 20> generic_format_metadata(
    std::uint32_t projection_count, const dif::Sha256Digest &program_digest,
    const dif::Sha256Digest &index_digest) {
  std::array<std::uint32_t, 20> values{};
  values[0] = kGenericFormatMagic;
  values[1] = kGenericFormatVersion;
  values[2] = kGroupSize;
  values[3] = projection_count;
  const auto append_digest = [&](std::size_t start,
                                 const dif::Sha256Digest &digest) {
    for (std::size_t word = 0; word < 8U; ++word) {
      values[start + word] =
          static_cast<std::uint32_t>(digest[word * 4U]) |
          (static_cast<std::uint32_t>(digest[word * 4U + 1U]) << 8U) |
          (static_cast<std::uint32_t>(digest[word * 4U + 2U]) << 16U) |
          (static_cast<std::uint32_t>(digest[word * 4U + 3U]) << 24U);
    }
  };
  append_digest(4U, program_digest);
  append_digest(12U, index_digest);
  return values;
}

std::vector<GenericProjection> generic_projections(
    const dif::ir::Program &program,
    const dif::weights::WeightBundle &bundle) {
  std::map<std::uint32_t, const dif::weights::BundleBinding *> bindings;
  for (const auto &binding : bundle.bindings)
    bindings.emplace(binding.tensor_id, &binding);
  std::map<std::uint32_t, GenericProjection> unique;
  for (const auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Linear ||
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
        input && (input->dtype == dif::ir::DType::BF16 ||
                  input->dtype == dif::ir::DType::F16);
    if (!input || !weight || !output || !eligible_dtype ||
        weight->dtype != input->dtype || output->dtype != input->dtype ||
        weight->dims.size() != 2U ||
        weight->dims[1] == 0U || weight->dims[1] % kGroupSize != 0U)
      continue;
    if (bias &&
        (input->dtype != dif::ir::DType::F16 ||
         bias->dtype != input->dtype ||
         bias->dims != std::vector<std::uint64_t>{weight->dims[0]} ||
         static_cast<dif::ir::LinearBiasMode>(operation.u64(
             dif::ir::AttrKey::LinearBiasMode,
             static_cast<std::uint64_t>(
                 dif::ir::LinearBiasMode::Epilogue))) !=
             dif::ir::LinearBiasMode::Epilogue))
      continue;
    const auto found = bindings.find(weight->id);
    if (found == bindings.end())
      dif::fail("generic ConvRot Linear weight lacks a bundle binding: tensor " +
                std::to_string(weight->id));
    const auto &binding = *found->second;
    if (binding.dtype != weight->dtype || binding.dims != weight->dims ||
        binding.shard_index >= bundle.shards.size())
      dif::fail("generic ConvRot bundle binding does not match tensor " +
                std::to_string(weight->id));
    unique.try_emplace(
        weight->id,
        GenericProjection{weight->id,
                          binding.shard_index,
                          weight->dtype,
                          binding.tensor_name,
                          "linear." + std::to_string(weight->id) + ".weight",
                          "linear." + std::to_string(weight->id) + ".scale",
                          weight->dims});
  }
  std::vector<GenericProjection> result;
  result.reserve(unique.size());
  for (auto &[id, projection] : unique) {
    (void)id;
    result.push_back(std::move(projection));
  }
  if (result.empty())
    dif::fail("program has no BF16/F16 Linear weights eligible for ConvRot");
  return result;
}

#if DIF_HAS_CUDA

void check_cu(CUresult result, const char *action) {
  if (result == CUDA_SUCCESS)
    return;
  const char *message = nullptr;
  cuGetErrorString(result, &message);
  dif::fail(std::string(action) + ": " +
            (message ? message : "unknown CUDA error"));
}

void check_nvrtc(nvrtcResult result, const char *action) {
  if (result != NVRTC_SUCCESS)
    dif::fail(std::string(action) + ": " + nvrtcGetErrorString(result));
}

constexpr const char *kKernelSource = R"CUDA(
__device__ __forceinline__ float load_bf16(const unsigned short* p,
                                            unsigned long long i) {
  return __uint_as_float((unsigned int)p[i] << 16);
}
__device__ __forceinline__ unsigned short store_f16(float input) {
  unsigned short result;
  asm("cvt.rn.f16.f32 %0, %1;" : "=h"(result) : "f"(input));
  return result;
}
__device__ __forceinline__ float load_f16(unsigned short input) {
  float result;
  asm("cvt.f32.f16 %0, %1;" : "=f"(result) : "h"(input));
  return result;
}
__device__ __forceinline__ float round_bf16(float value) {
  unsigned int bits = __float_as_uint(value);
  unsigned int exponent = bits & 0x7f800000U;
  if (exponent == 0x7f800000U)
    return value;
  unsigned int lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  return __uint_as_float(bits & 0xffff0000U);
}
__device__ __forceinline__ float round_f16(float value) {
  return load_f16(store_f16(value));
}
template<int S> __device__ __forceinline__ void stage64(
    const float* src, float* dst, int lane) {
  int base = (lane % S) + (lane / S) * (4 * S);
  float x0 = src[base], x1 = src[base + S];
  float x2 = src[base + 2 * S], x3 = src[base + 3 * S];
  dst[base] = 0.5f * (x0 + x1 + x2 - x3);
  dst[base + S] = 0.5f * (x0 + x1 - x2 + x3);
  dst[base + 2 * S] = 0.5f * (x0 - x1 + x2 + x3);
  dst[base + 3 * S] = 0.5f * (-x0 + x1 + x2 + x3);
}
template<int S> __device__ __forceinline__ float stage64_store(
    const float* src, float* output, int lane) {
  int base = (lane % S) + (lane / S) * (4 * S);
  float x0 = src[base], x1 = src[base + S];
  float x2 = src[base + 2 * S], x3 = src[base + 3 * S];
  float y0 = 0.5f * (x0 + x1 + x2 - x3);
  float y1 = 0.5f * (x0 + x1 - x2 + x3);
  float y2 = 0.5f * (x0 - x1 + x2 + x3);
  float y3 = 0.5f * (-x0 + x1 + x2 + x3);
  output[base] = y0; output[base + S] = y1;
  output[base + 2 * S] = y2; output[base + 3 * S] = y3;
  return fmaxf(fmaxf(fabsf(y0), fabsf(y1)),
               fmaxf(fabsf(y2), fabsf(y3)));
}
extern "C" __global__ void quantize_h3_convrot(
    const unsigned short* input, signed char* output, float* scales,
    int rows, int columns, int reorder_qkv, int inner, int head_dim) {
  constexpr int group = 256, group_threads = 64, groups_in_flight = 16;
  extern __shared__ float smem[];
  float* row_buf = smem;
  float* temporary = smem + columns;
  __shared__ float warp_max[32];
  __shared__ float block_max;
  int output_row = (int)blockIdx.x;
  if (output_row >= rows)
    return;
  int source_row = output_row;
  if (reorder_qkv) {
    int part = output_row / inner;
    int within = output_row - part * inner;
    int head = within / head_dim;
    int d = within - head * head_dim;
    source_row = (head * 3 + part) * head_dim + d;
  }
  int tid = (int)threadIdx.x;
  int sub = tid / group_threads;
  int lane = tid % group_threads;
  int group_count = columns / group;
  float local_max = 0.0f;
  unsigned long long input_base = (unsigned long long)source_row * columns;
  for (int it = 0; it < (group_count + groups_in_flight - 1) /
                             groups_in_flight; ++it) {
    int g = it * groups_in_flight + sub;
    bool active = g < group_count;
    int b = lane * 4;
    int group_column = g * group;
    int column = group_column + b;
    float* b0 = temporary + sub * (2 * group);
    float* b1 = b0 + group;
    float x0 = active ? load_bf16(input, input_base + column) : 0.0f;
    float x1 = active ? load_bf16(input, input_base + column + 1) : 0.0f;
    float x2 = active ? load_bf16(input, input_base + column + 2) : 0.0f;
    float x3 = active ? load_bf16(input, input_base + column + 3) : 0.0f;
    b1[b] = 0.5f * (x0 + x1 + x2 - x3);
    b1[b + 1] = 0.5f * (x0 + x1 - x2 + x3);
    b1[b + 2] = 0.5f * (x0 - x1 + x2 + x3);
    b1[b + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    stage64<4>(b1, b0, lane);
    __syncthreads();
    stage64<16>(b0, b1, lane);
    __syncthreads();
    if (active)
      local_max = fmaxf(local_max,
                        stage64_store<64>(b1, row_buf + group_column, lane));
    __syncthreads();
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max,
                      __shfl_down_sync(0xffffffff, local_max, offset));
  int warp = tid >> 5;
  int lane32 = tid & 31;
  if (lane32 == 0)
    warp_max[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float value = lane32 < 32 ? warp_max[lane32] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      value = fmaxf(value,
                    __shfl_down_sync(0xffffffff, value, offset));
    if (lane32 == 0) {
      block_max = value;
      float scale = value * (1.0f / 127.0f);
      scales[output_row] = scale < 1.0e-30f ? 1.0e-30f : scale;
    }
  }
  __syncthreads();
  float scale_bf16 = round_bf16(scales[output_row]);
  unsigned long long output_base = (unsigned long long)output_row * columns;
  for (int column = tid; column < columns; column += 1024) {
    float value = round_bf16(row_buf[column]);
    float scaled = round_bf16(value / scale_bf16);
    int quantized = (int)nearbyintf(scaled);
    quantized = quantized > 127 ? 127 : (quantized < -128 ? -128 : quantized);
    output[output_base + column] = (signed char)quantized;
  }
}

extern "C" __global__ void quantize_h3_convrot_chunked(
    const unsigned short* input, signed char* output, float* scales,
    int rows, int columns, int reorder_qkv, int inner, int head_dim,
    int scale_chunk) {
  constexpr int group = 256, group_threads = 64, groups_in_flight = 8;
  extern __shared__ float smem[];
  float* row_buf = smem;
  float* temporary = smem + scale_chunk;
  __shared__ float warp_max[32];
  __shared__ float block_max;
  int output_row = (int)blockIdx.x;
  if (output_row >= rows)
    return;
  int source_row = output_row;
  if (reorder_qkv) {
    int part = output_row / inner;
    int within = output_row - part * inner;
    int head = within / head_dim;
    int d = within - head * head_dim;
    source_row = (head * 3 + part) * head_dim + d;
  }
  int tid = (int)threadIdx.x;
  int sub = tid / group_threads;
  int lane = tid % group_threads;
  int group_count = columns / group;
  int groups_per_chunk = scale_chunk / group;
  int chunks = (group_count + groups_per_chunk - 1) / groups_per_chunk;
  unsigned long long input_base = (unsigned long long)source_row * columns;
  unsigned long long output_base = (unsigned long long)output_row * columns;
  for (int chunk = 0; chunk < chunks; ++chunk) {
    int first_group = chunk * groups_per_chunk;
    int active_groups = min(groups_per_chunk, group_count - first_group);
    int chunk_columns = active_groups * group;
    float local_max = 0.0f;
    int g = first_group + sub;
    bool active = sub < active_groups;
    int b = lane * 4;
    int group_column = g * group;
    int column = group_column + b;
    float* b0 = temporary + sub * (2 * group);
    float* b1 = b0 + group;
    float x0 = active ? load_bf16(input, input_base + column) : 0.0f;
    float x1 = active ? load_bf16(input, input_base + column + 1) : 0.0f;
    float x2 = active ? load_bf16(input, input_base + column + 2) : 0.0f;
    float x3 = active ? load_bf16(input, input_base + column + 3) : 0.0f;
    b1[b] = 0.5f * (x0 + x1 + x2 - x3);
    b1[b + 1] = 0.5f * (x0 + x1 - x2 + x3);
    b1[b + 2] = 0.5f * (x0 - x1 + x2 + x3);
    b1[b + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    stage64<4>(b1, b0, lane);
    __syncthreads();
    stage64<16>(b0, b1, lane);
    __syncthreads();
    if (active)
      local_max = stage64_store<64>(b1, row_buf + sub * group, lane);
    __syncthreads();
    for (int offset = 16; offset > 0; offset >>= 1)
      local_max = fmaxf(local_max,
                        __shfl_down_sync(0xffffffff, local_max, offset));
    int warp = tid >> 5;
    int lane32 = tid & 31;
    if (lane32 == 0)
      warp_max[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
      float value = lane32 < (int)(blockDim.x >> 5) ? warp_max[lane32] : 0.0f;
      for (int offset = 16; offset > 0; offset >>= 1)
        value = fmaxf(value,
                      __shfl_down_sync(0xffffffff, value, offset));
      if (lane32 == 0) {
        float scale = value * (1.0f / 127.0f);
        block_max = scale < 1.0e-30f ? 1.0e-30f : scale;
        scales[(unsigned long long)output_row * chunks + chunk] = block_max;
      }
    }
    __syncthreads();
    float scale_bf16 = round_bf16(block_max);
    int chunk_start = first_group * group;
    for (int local_column = tid; local_column < chunk_columns;
         local_column += (int)blockDim.x) {
      float value = round_bf16(row_buf[local_column]);
      float scaled = round_bf16(value / scale_bf16);
      int quantized = (int)nearbyintf(scaled);
      quantized = quantized > 127 ? 127 :
                  (quantized < -128 ? -128 : quantized);
      output[output_base + chunk_start + local_column] =
          (signed char)quantized;
    }
    __syncthreads();
  }
}

// Generic form used for arbitrary DiffIR Linear weights.  Unlike the legacy
// H3 builder kernel it does not retain the complete rotated row in dynamic
// shared memory.  It deterministically recomputes each 256-wide group after
// the row maximum is known, which admits contractions such as Qwen's 25,600
// columns on SM86 while keeping the same H256 transform and source-dtype
// observable quantization boundaries.
extern "C" __global__ void quantize_generic_convrot(
    const unsigned short* input, signed char* output, float* scales,
    int rows, int columns, int input_f16) {
  constexpr int group = 256, group_threads = 64, groups_in_flight = 16;
  extern __shared__ float temporary[];
  __shared__ float warp_max[32];
  __shared__ float block_max;
  int output_row = (int)blockIdx.x;
  if (output_row >= rows)
    return;
  int tid = (int)threadIdx.x;
  int sub = tid / group_threads;
  int lane = tid % group_threads;
  int group_count = columns / group;
  float local_max = 0.0f;
  unsigned long long input_base = (unsigned long long)output_row * columns;
  for (int it = 0; it < (group_count + groups_in_flight - 1) /
                             groups_in_flight; ++it) {
    int g = it * groups_in_flight + sub;
    bool active = g < group_count;
    int b = lane * 4;
    int group_column = g * group;
    int column = group_column + b;
    float* b0 = temporary + sub * (2 * group);
    float* b1 = b0 + group;
    float x0 = active ? (input_f16 ? load_f16(input[input_base + column])
                                   : load_bf16(input, input_base + column)) : 0.0f;
    float x1 = active ? (input_f16 ? load_f16(input[input_base + column + 1])
                                   : load_bf16(input, input_base + column + 1)) : 0.0f;
    float x2 = active ? (input_f16 ? load_f16(input[input_base + column + 2])
                                   : load_bf16(input, input_base + column + 2)) : 0.0f;
    float x3 = active ? (input_f16 ? load_f16(input[input_base + column + 3])
                                   : load_bf16(input, input_base + column + 3)) : 0.0f;
    b1[b] = 0.5f * (x0 + x1 + x2 - x3);
    b1[b + 1] = 0.5f * (x0 + x1 - x2 + x3);
    b1[b + 2] = 0.5f * (x0 - x1 + x2 + x3);
    b1[b + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    stage64<4>(b1, b0, lane);
    __syncthreads();
    stage64<16>(b0, b1, lane);
    __syncthreads();
    int base = lane;
    float y0 = 0.5f * (b1[base] + b1[base + 64] +
                       b1[base + 128] - b1[base + 192]);
    float y1 = 0.5f * (b1[base] + b1[base + 64] -
                       b1[base + 128] + b1[base + 192]);
    float y2 = 0.5f * (b1[base] - b1[base + 64] +
                       b1[base + 128] + b1[base + 192]);
    float y3 = 0.5f * (-b1[base] + b1[base + 64] +
                       b1[base + 128] + b1[base + 192]);
    if (active)
      local_max = fmaxf(local_max, fmaxf(fmaxf(fabsf(y0), fabsf(y1)),
                                         fmaxf(fabsf(y2), fabsf(y3))));
    __syncthreads();
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    local_max = fmaxf(local_max,
                      __shfl_down_sync(0xffffffff, local_max, offset));
  int warp = tid >> 5;
  int lane32 = tid & 31;
  if (lane32 == 0)
    warp_max[warp] = local_max;
  __syncthreads();
  if (warp == 0) {
    float value = lane32 < 32 ? warp_max[lane32] : 0.0f;
    for (int offset = 16; offset > 0; offset >>= 1)
      value = fmaxf(value,
                    __shfl_down_sync(0xffffffff, value, offset));
    if (lane32 == 0) {
      block_max = value;
      float scale = value * (1.0f / 127.0f);
      scales[output_row] = scale < 1.0e-30f ? 1.0e-30f : scale;
    }
  }
  __syncthreads();
  float scale_lowp = input_f16 ? round_f16(scales[output_row])
                               : round_bf16(scales[output_row]);
  unsigned long long output_base = (unsigned long long)output_row * columns;
  for (int it = 0; it < (group_count + groups_in_flight - 1) /
                             groups_in_flight; ++it) {
    int g = it * groups_in_flight + sub;
    bool active = g < group_count;
    int b = lane * 4;
    int group_column = g * group;
    int column = group_column + b;
    float* b0 = temporary + sub * (2 * group);
    float* b1 = b0 + group;
    float x0 = active ? (input_f16 ? load_f16(input[input_base + column])
                                   : load_bf16(input, input_base + column)) : 0.0f;
    float x1 = active ? (input_f16 ? load_f16(input[input_base + column + 1])
                                   : load_bf16(input, input_base + column + 1)) : 0.0f;
    float x2 = active ? (input_f16 ? load_f16(input[input_base + column + 2])
                                   : load_bf16(input, input_base + column + 2)) : 0.0f;
    float x3 = active ? (input_f16 ? load_f16(input[input_base + column + 3])
                                   : load_bf16(input, input_base + column + 3)) : 0.0f;
    b1[b] = 0.5f * (x0 + x1 + x2 - x3);
    b1[b + 1] = 0.5f * (x0 + x1 - x2 + x3);
    b1[b + 2] = 0.5f * (x0 - x1 + x2 + x3);
    b1[b + 3] = 0.5f * (-x0 + x1 + x2 + x3);
    __syncthreads();
    stage64<4>(b1, b0, lane);
    __syncthreads();
    stage64<16>(b0, b1, lane);
    __syncthreads();
    int base = lane;
    float y[4];
    y[0] = 0.5f * (b1[base] + b1[base + 64] +
                    b1[base + 128] - b1[base + 192]);
    y[1] = 0.5f * (b1[base] + b1[base + 64] -
                    b1[base + 128] + b1[base + 192]);
    y[2] = 0.5f * (b1[base] - b1[base + 64] +
                    b1[base + 128] + b1[base + 192]);
    y[3] = 0.5f * (-b1[base] + b1[base + 64] +
                    b1[base + 128] + b1[base + 192]);
    if (active) {
      for (int j = 0; j < 4; ++j) {
        float value = input_f16 ? round_f16(y[j]) : round_bf16(y[j]);
        float scaled = input_f16 ? round_f16(value / scale_lowp)
                                 : round_bf16(value / scale_lowp);
        int quantized = (int)nearbyintf(scaled);
        quantized = quantized > 127 ? 127 :
                    (quantized < -128 ? -128 : quantized);
        output[output_base + group_column + lane + j * 64] =
            (signed char)quantized;
      }
    }
    __syncthreads();
  }
}

extern "C" __global__ void quantize_h3_groupwise_quality(
    const unsigned short* input, signed char* output, void* scales,
    int rows, int columns, int group_size, int reorder_qkv, int inner,
    int head_dim, int swap_row_halves, int scale_f32) {
  const int groups = columns / group_size;
  const unsigned long long linear_group = blockIdx.x;
  const int output_row = (int)(linear_group / groups);
  const int group = (int)(linear_group - (unsigned long long)output_row * groups);
  if (output_row >= rows)
    return;
  int source_row = output_row;
  if (reorder_qkv) {
    const int part = output_row / inner;
    const int within = output_row - part * inner;
    const int head = within / head_dim;
    const int d = within - head * head_dim;
    source_row = (head * 3 + part) * head_dim + d;
  } else if (swap_row_halves) {
    const int half = rows / 2;
    source_row = output_row < half ? output_row + half : output_row - half;
  }
  __shared__ float maximum[128];
  const int lane = (int)threadIdx.x;
  const unsigned long long source_base =
      (unsigned long long)source_row * columns + group * group_size;
  float local = 0.0f;
  for (int column = lane; column < group_size; column += (int)blockDim.x)
    local = fmaxf(local, fabsf(load_bf16(input, source_base + column)));
  maximum[lane] = local;
  __syncthreads();
  for (int offset = (int)blockDim.x / 2; offset > 0; offset >>= 1) {
    if (lane < offset)
      maximum[lane] = fmaxf(maximum[lane], maximum[lane + offset]);
    __syncthreads();
  }
  if (lane == 0) {
    float scale = maximum[0] * (1.0f / 127.0f);
    if (scale < 1.0e-30f)
      scale = 1.0e-30f;
    const unsigned long long scale_index =
        (unsigned long long)output_row * groups + group;
    if (scale_f32)
      ((float*)scales)[scale_index] = scale;
    else
      ((unsigned short*)scales)[scale_index] = store_f16(scale);
  }
  __syncthreads();
  const unsigned long long scale_index =
      (unsigned long long)output_row * groups + group;
  const float scale = scale_f32 ? ((float*)scales)[scale_index] :
      load_f16(((unsigned short*)scales)[scale_index]);
  const unsigned long long output_base =
      (unsigned long long)output_row * columns + group * group_size;
  for (int column = lane; column < group_size; column += (int)blockDim.x) {
    int quantized = (int)nearbyintf(
        load_bf16(input, source_base + column) / scale);
    quantized = quantized > 127 ? 127 :
                (quantized < -128 ? -128 : quantized);
    output[output_base + column] = (signed char)quantized;
  }
}
)CUDA";

class NativeConvRotQuantizer {
public:
  NativeConvRotQuantizer() {
    check_cu(cuInit(0), "cuInit");
    check_cu(cuDeviceGet(&device_, 0), "cuDeviceGet");
    check_cu(cuDevicePrimaryCtxRetain(&context_, device_),
             "cuDevicePrimaryCtxRetain");
    check_cu(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    int major = 0, minor = 0;
    check_cu(cuDeviceGetAttribute(
                 &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device_),
             "cuDeviceGetAttribute major");
    check_cu(cuDeviceGetAttribute(
                 &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device_),
             "cuDeviceGetAttribute minor");
    const auto architecture = "--gpu-architecture=compute_" +
                              std::to_string(major) + std::to_string(minor);
    nvrtcProgram program{};
    check_nvrtc(nvrtcCreateProgram(&program, kKernelSource,
                                   "difh3convrot.cu", 0, nullptr, nullptr),
                "nvrtcCreateProgram");
    const char *options[] = {architecture.c_str()};
    const auto compile = nvrtcCompileProgram(program, 1, options);
    if (compile != NVRTC_SUCCESS) {
      std::size_t log_size = 0U;
      nvrtcGetProgramLogSize(program, &log_size);
      std::string log(log_size, '\0');
      nvrtcGetProgramLog(program, log.data());
      (void)nvrtcDestroyProgram(&program);
      dif::fail("NVRTC compile failed: " + log);
    }
    std::size_t ptx_size = 0U;
    check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
    check_nvrtc(nvrtcDestroyProgram(&program), "nvrtcDestroyProgram");
    check_cu(cuModuleLoadDataEx(&module_, ptx.c_str(), 0, nullptr, nullptr),
             "cuModuleLoadDataEx");
    check_cu(cuModuleGetFunction(&function_, module_, "quantize_h3_convrot"),
             "cuModuleGetFunction");
    check_cu(cuModuleGetFunction(&chunked_function_, module_,
                                 "quantize_h3_convrot_chunked"),
             "cuModuleGetFunction chunk-scaled H3 ConvRot");
    check_cu(cuModuleGetFunction(&generic_function_, module_,
                                 "quantize_generic_convrot"),
             "cuModuleGetFunction generic ConvRot");
    check_cu(cuModuleGetFunction(&groupwise_function_, module_,
                                 "quantize_h3_groupwise_quality"),
             "cuModuleGetFunction H3 groupwise quality");
    constexpr std::uint64_t temporary_floats = 16U * 2U * kGroupSize;
    const auto maximum_shared =
        (kH3Ffn + temporary_floats) * sizeof(float);
    check_cu(cuFuncSetAttribute(
                 function_, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                 static_cast<int>(maximum_shared)),
             "cuFuncSetAttribute dynamic shared memory");
  }

  ~NativeConvRotQuantizer() {
    if (module_)
      (void)cuModuleUnload(module_);
    if (context_)
      (void)cuDevicePrimaryCtxRelease(device_);
  }

  std::pair<dif::runtime::Tensor, dif::runtime::Tensor>
  quantize(const dif::runtime::Tensor &input, bool reorder_qkv) const {
    if (input.dtype != dif::ir::DType::BF16 || input.dims.size() != 2U ||
        input.dims[1] % kGroupSize != 0U)
      dif::fail("H3 ConvRot source projection must be rank-2 BF16 with K divisible by 256");
    if (input.dims[0] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
        input.dims[1] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
      dif::fail("H3 ConvRot projection exceeds CUDA int geometry");
    dif::runtime::Tensor output{dif::ir::DType::I8, input.dims, {}};
    output.bytes.resize(static_cast<std::size_t>(input.element_count()));
    dif::runtime::Tensor scales{dif::ir::DType::F32, {input.dims[0]}, {}};
    scales.bytes.resize(static_cast<std::size_t>(input.dims[0]) * sizeof(float));

    CUdeviceptr device_input{}, device_output{}, device_scales{};
    check_cu(cuMemAlloc(&device_input, input.byte_size()),
             "cuMemAlloc source projection");
    try {
      check_cu(cuMemAlloc(&device_output, output.byte_size()),
               "cuMemAlloc quantized projection");
      try {
        check_cu(cuMemAlloc(&device_scales, scales.byte_size()),
                 "cuMemAlloc projection scales");
        check_cu(cuMemcpyHtoD(device_input, input.data(), input.byte_size()),
                 "cuMemcpyHtoD source projection");
        auto rows = static_cast<int>(input.dims[0]);
        auto columns = static_cast<int>(input.dims[1]);
        auto reorder = reorder_qkv ? 1 : 0;
        auto inner = static_cast<int>(kH3Inner);
        auto head_dim = static_cast<int>(kH3HeadDim);
        void *arguments[] = {&device_input, &device_output, &device_scales,
                             &rows, &columns, &reorder, &inner, &head_dim};
        constexpr unsigned block = 1024U;
        constexpr std::uint64_t temporary_floats = 16U * 2U * kGroupSize;
        const auto shared = static_cast<unsigned>(
            (input.dims[1] + temporary_floats) * sizeof(float));
        check_cu(cuLaunchKernel(function_, static_cast<unsigned>(rows), 1U, 1U,
                                block, 1U, 1U, shared, nullptr, arguments,
                                nullptr),
                 "cuLaunchKernel H3 ConvRot weight quantization");
        check_cu(cuMemcpyDtoH(output.mutable_data(), device_output,
                              output.byte_size()),
                 "cuMemcpyDtoH quantized projection");
        check_cu(cuMemcpyDtoH(scales.mutable_data(), device_scales,
                              scales.byte_size()),
                 "cuMemcpyDtoH projection scales");
        (void)cuMemFree(device_scales);
      } catch (...) {
        (void)cuMemFree(device_output);
        throw;
      }
      (void)cuMemFree(device_output);
    } catch (...) {
      (void)cuMemFree(device_input);
      throw;
    }
    (void)cuMemFree(device_input);
    return {std::move(output), std::move(scales)};
  }

  std::pair<dif::runtime::Tensor, dif::runtime::Tensor>
  quantize_chunked(const dif::runtime::Tensor &input, bool reorder_qkv,
                   std::uint32_t scale_chunk) const {
    if (input.dtype != dif::ir::DType::BF16 || input.dims.size() != 2U ||
        input.dims[1] % kGroupSize != 0U || scale_chunk < kGroupSize ||
        scale_chunk > 2048U || scale_chunk % kGroupSize != 0U)
      dif::fail("H3 chunk-scaled ConvRot source has invalid geometry");
    if (input.dims[0] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
        input.dims[1] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
      dif::fail("H3 chunk-scaled ConvRot projection exceeds CUDA int geometry");
    const auto chunks =
        (input.dims[1] + scale_chunk - 1U) / scale_chunk;
    dif::runtime::Tensor output{dif::ir::DType::I8, input.dims, {}};
    output.bytes.resize(static_cast<std::size_t>(input.element_count()));
    dif::runtime::Tensor scales{dif::ir::DType::F32,
                                {input.dims[0], chunks}, {}};
    scales.bytes.resize(static_cast<std::size_t>(input.dims[0] * chunks) *
                        sizeof(float));

    CUdeviceptr device_input{}, device_output{}, device_scales{};
    check_cu(cuMemAlloc(&device_input, input.byte_size()),
             "cuMemAlloc chunk-scaled source projection");
    try {
      check_cu(cuMemAlloc(&device_output, output.byte_size()),
               "cuMemAlloc chunk-scaled quantized projection");
      try {
        check_cu(cuMemAlloc(&device_scales, scales.byte_size()),
                 "cuMemAlloc chunk-scaled projection scales");
        check_cu(cuMemcpyHtoD(device_input, input.data(), input.byte_size()),
                 "cuMemcpyHtoD chunk-scaled source projection");
        auto rows = static_cast<int>(input.dims[0]);
        auto columns = static_cast<int>(input.dims[1]);
        auto reorder = reorder_qkv ? 1 : 0;
        auto inner = static_cast<int>(kH3Inner);
        auto head_dim = static_cast<int>(kH3HeadDim);
        auto chunk = static_cast<int>(scale_chunk);
        void *arguments[] = {&device_input, &device_output, &device_scales,
                             &rows, &columns, &reorder, &inner, &head_dim,
                             &chunk};
        constexpr unsigned block = 512U;
        constexpr std::uint64_t temporary_floats =
            8U * 2U * kGroupSize;
        const auto shared = static_cast<unsigned>(
            (scale_chunk + temporary_floats) * sizeof(float));
        check_cu(cuLaunchKernel(chunked_function_,
                                static_cast<unsigned>(rows), 1U, 1U, block,
                                1U, 1U, shared, nullptr, arguments, nullptr),
                 "cuLaunchKernel chunk-scaled H3 ConvRot quantization");
        check_cu(cuMemcpyDtoH(output.mutable_data(), device_output,
                              output.byte_size()),
                 "cuMemcpyDtoH chunk-scaled quantized projection");
        check_cu(cuMemcpyDtoH(scales.mutable_data(), device_scales,
                              scales.byte_size()),
                 "cuMemcpyDtoH chunk-scaled projection scales");
        (void)cuMemFree(device_scales);
      } catch (...) {
        (void)cuMemFree(device_output);
        throw;
      }
      (void)cuMemFree(device_output);
    } catch (...) {
      (void)cuMemFree(device_input);
      throw;
    }
    (void)cuMemFree(device_input);
    return {std::move(output), std::move(scales)};
  }

  std::pair<dif::runtime::Tensor, dif::runtime::Tensor>
  quantize_generic(const dif::runtime::Tensor &input) const {
    if ((input.dtype != dif::ir::DType::BF16 &&
         input.dtype != dif::ir::DType::F16) ||
        input.dims.size() != 2U ||
        input.dims[1] % kGroupSize != 0U)
      dif::fail("generic ConvRot source must be rank-2 BF16/F16 with K divisible by 256");
    if (input.dims[0] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
        input.dims[1] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
      dif::fail("generic ConvRot projection exceeds CUDA int geometry");
    dif::runtime::Tensor output{dif::ir::DType::I8, input.dims, {}};
    output.bytes.resize(static_cast<std::size_t>(input.element_count()));
    dif::runtime::Tensor scales{dif::ir::DType::F32, {input.dims[0]}, {}};
    scales.bytes.resize(static_cast<std::size_t>(input.dims[0]) *
                        sizeof(float));

    CUdeviceptr device_input{}, device_output{}, device_scales{};
    check_cu(cuMemAlloc(&device_input, input.byte_size()),
             "cuMemAlloc generic source projection");
    try {
      check_cu(cuMemAlloc(&device_output, output.byte_size()),
               "cuMemAlloc generic quantized projection");
      try {
        check_cu(cuMemAlloc(&device_scales, scales.byte_size()),
                 "cuMemAlloc generic projection scales");
        check_cu(cuMemcpyHtoD(device_input, input.data(), input.byte_size()),
                 "cuMemcpyHtoD generic source projection");
        auto rows = static_cast<int>(input.dims[0]);
        auto columns = static_cast<int>(input.dims[1]);
        auto input_f16 = static_cast<int>(input.dtype == dif::ir::DType::F16);
        void *arguments[] = {&device_input, &device_output, &device_scales,
                             &rows, &columns, &input_f16};
        constexpr unsigned block = 1024U;
        constexpr std::uint64_t temporary_floats =
            16U * 2U * kGroupSize;
        const auto shared = static_cast<unsigned>(temporary_floats *
                                                   sizeof(float));
        check_cu(cuLaunchKernel(generic_function_,
                                static_cast<unsigned>(rows), 1U, 1U, block,
                                1U, 1U, shared, nullptr, arguments, nullptr),
                 "cuLaunchKernel generic ConvRot weight quantization");
        check_cu(cuMemcpyDtoH(output.mutable_data(), device_output,
                              output.byte_size()),
                 "cuMemcpyDtoH generic quantized projection");
        check_cu(cuMemcpyDtoH(scales.mutable_data(), device_scales,
                              scales.byte_size()),
                 "cuMemcpyDtoH generic projection scales");
        (void)cuMemFree(device_scales);
      } catch (...) {
        (void)cuMemFree(device_output);
        throw;
      }
      (void)cuMemFree(device_output);
    } catch (...) {
      (void)cuMemFree(device_input);
      throw;
    }
    (void)cuMemFree(device_input);
    return {std::move(output), std::move(scales)};
  }

  std::pair<dif::runtime::Tensor, dif::runtime::Tensor>
  quantize_groupwise(const dif::runtime::Tensor &input,
                     std::uint32_t group_size, bool reorder_qkv,
                     bool swap_row_halves, bool scale_f32) const {
    if (input.dtype != dif::ir::DType::BF16 || input.dims.size() != 2U ||
        group_size == 0U || input.dims[1] % group_size != 0U)
      dif::fail("H3 groupwise source must be rank-2 BF16 with divisible K");
    if (input.dims[0] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
        input.dims[1] > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
      dif::fail("H3 groupwise projection exceeds CUDA int geometry");
    const auto groups = input.dims[1] / group_size;
    dif::runtime::Tensor output{dif::ir::DType::I8, input.dims, {}};
    output.bytes.resize(static_cast<std::size_t>(input.element_count()));
    dif::runtime::Tensor scales{scale_f32 ? dif::ir::DType::F32
                                          : dif::ir::DType::F16,
                                {input.dims[0], groups}, {}};
    scales.bytes.resize(static_cast<std::size_t>(input.dims[0] * groups) *
                        (scale_f32 ? sizeof(float) : sizeof(std::uint16_t)));

    CUdeviceptr device_input{}, device_output{}, device_scales{};
    check_cu(cuMemAlloc(&device_input, input.byte_size()),
             "cuMemAlloc groupwise source projection");
    try {
      check_cu(cuMemAlloc(&device_output, output.byte_size()),
               "cuMemAlloc groupwise quantized projection");
      try {
        check_cu(cuMemAlloc(&device_scales, scales.byte_size()),
                 "cuMemAlloc groupwise projection scales");
        check_cu(cuMemcpyHtoD(device_input, input.data(), input.byte_size()),
                 "cuMemcpyHtoD groupwise source projection");
        auto rows = static_cast<int>(input.dims[0]);
        auto columns = static_cast<int>(input.dims[1]);
        auto group = static_cast<int>(group_size);
        auto reorder = reorder_qkv ? 1 : 0;
        auto inner = static_cast<int>(kH3Inner);
        auto head_dim = static_cast<int>(kH3HeadDim);
        auto swap = swap_row_halves ? 1 : 0;
        auto f32 = scale_f32 ? 1 : 0;
        void *arguments[] = {&device_input, &device_output, &device_scales,
                             &rows, &columns, &group, &reorder, &inner,
                             &head_dim, &swap, &f32};
        const auto grid = input.dims[0] * groups;
        if (grid > std::numeric_limits<unsigned>::max())
          dif::fail("H3 groupwise projection exceeds CUDA grid geometry");
        check_cu(cuLaunchKernel(groupwise_function_,
                                static_cast<unsigned>(grid), 1U, 1U, 128U,
                                1U, 1U, 0U, nullptr, arguments, nullptr),
                 "cuLaunchKernel H3 groupwise quality quantization");
        check_cu(cuMemcpyDtoH(output.mutable_data(), device_output,
                              output.byte_size()),
                 "cuMemcpyDtoH groupwise quantized projection");
        check_cu(cuMemcpyDtoH(scales.mutable_data(), device_scales,
                              scales.byte_size()),
                 "cuMemcpyDtoH groupwise projection scales");
        (void)cuMemFree(device_scales);
      } catch (...) {
        (void)cuMemFree(device_output);
        throw;
      }
      (void)cuMemFree(device_output);
    } catch (...) {
      (void)cuMemFree(device_input);
      throw;
    }
    (void)cuMemFree(device_input);
    return {std::move(output), std::move(scales)};
  }

private:
  CUdevice device_{};
  CUcontext context_{};
  CUmodule module_{};
  CUfunction function_{};
  CUfunction chunked_function_{};
  CUfunction generic_function_{};
  CUfunction groupwise_function_{};
};

#endif

} // namespace

namespace {

int run_generic(int argc, char **argv) {
  std::filesystem::path program_path;
  std::filesystem::path bundle_path;
  std::filesystem::path output_path;
  std::filesystem::path rebind_cache_path;
  for (int argument = 1; argument < argc; ++argument) {
    const std::string option = argv[argument];
    const auto value = [&](const char *name) -> std::filesystem::path {
      if (argument + 1 >= argc)
        usage_error(std::string(name) + " requires a value");
      return argv[++argument];
    };
    if (option == "--program")
      program_path = value("--program");
    else if (option == "--bundle")
      bundle_path = value("--bundle");
    else if (option == "--output")
      output_path = value("--output");
    else if (option == "--rebind-cache")
      rebind_cache_path = value("--rebind-cache");
    else
      usage_error("unknown generic option " + option);
  }
  if (program_path.empty() || bundle_path.empty() ||
      (output_path.empty() == rebind_cache_path.empty()))
    usage_error("generic mode requires --program, --bundle, and exactly one of --output or --rebind-cache");
  program_path = std::filesystem::absolute(program_path).lexically_normal();
  bundle_path = std::filesystem::absolute(bundle_path).lexically_normal();
  if (!output_path.empty())
    output_path = std::filesystem::absolute(output_path).lexically_normal();
  if (!rebind_cache_path.empty())
    rebind_cache_path =
        std::filesystem::absolute(rebind_cache_path).lexically_normal();
  if (!std::filesystem::is_regular_file(program_path) ||
      !std::filesystem::is_regular_file(bundle_path))
    dif::fail("generic ConvRot program or bundle does not exist");
  const auto temporary = partial_path(output_path);
  if (std::filesystem::exists(output_path) ||
      std::filesystem::exists(temporary))
    dif::fail("refusing to overwrite output or partial file: " +
              output_path.string());
  if (!output_path.parent_path().empty() &&
      !std::filesystem::is_directory(output_path.parent_path()))
    dif::fail("output parent directory does not exist: " +
              output_path.parent_path().string());

  const auto program = dif::ir::read_file(program_path);
  dif::ir::verify(program);
  const auto bundle = dif::weights::read_weight_bundle(bundle_path);
  dif::weights::verify_weight_bundle(bundle, program, false);
  const auto program_digest = dif::ir::fingerprint(program);
  const auto plan = generic_projections(program, bundle);
  if (!rebind_cache_path.empty()) {
    if (!std::filesystem::is_regular_file(rebind_cache_path))
      dif::fail("generic ConvRot cache to rebind does not exist");
    const auto cache = dif::weights::read_safetensors(rebind_cache_path);
    const auto *metadata_entry = cache.find("__meta__.convrot_int8");
    if (!metadata_entry || metadata_entry->dtype != dif::ir::DType::I32 ||
        metadata_entry->dims != std::vector<std::uint64_t>{20U} ||
        metadata_entry->byte_count != 20U * sizeof(std::uint32_t))
      dif::fail("generic ConvRot rebind cache has invalid metadata");
    const auto metadata =
        dif::weights::map_safetensor(cache, "__meta__.convrot_int8");
    std::array<std::uint32_t, 20> identity{};
    std::memcpy(identity.data(), metadata.data(), metadata.byte_size());
    if (identity[0] != kGenericFormatMagic ||
        identity[1] != kGenericFormatVersion || identity[2] != kGroupSize ||
        identity[3] != plan.size())
      dif::fail("generic ConvRot rebind cache identity does not match the program projection set");
    for (std::size_t word = 0U; word < 8U; ++word) {
      const auto expected =
          static_cast<std::uint32_t>(bundle.index_fingerprint[word * 4U]) |
          (static_cast<std::uint32_t>(bundle.index_fingerprint[word * 4U + 1U])
           << 8U) |
          (static_cast<std::uint32_t>(bundle.index_fingerprint[word * 4U + 2U])
           << 16U) |
          (static_cast<std::uint32_t>(bundle.index_fingerprint[word * 4U + 3U])
           << 24U);
      if (identity[12U + word] != expected)
        dif::fail("generic ConvRot rebind cache targets a different checkpoint index");
    }
    for (const auto &projection : plan) {
      const auto *weight = cache.find(projection.output_weight_name);
      const auto *scale = cache.find(projection.output_scale_name);
      if (!weight || weight->dtype != dif::ir::DType::I8 ||
          weight->dims != projection.dims || !scale ||
          scale->dtype != dif::ir::DType::F32 ||
          scale->dims != std::vector<std::uint64_t>{projection.dims[0]})
        dif::fail("generic ConvRot rebind cache projection mismatch at tensor " +
                  std::to_string(projection.tensor_id));
    }
    const auto rebound = generic_format_metadata(
        static_cast<std::uint32_t>(plan.size()), program_digest,
        bundle.index_fingerprint);
    std::fstream output(rebind_cache_path,
                        std::ios::binary | std::ios::in | std::ios::out);
    output.seekp(static_cast<std::streamoff>(metadata_entry->file_offset));
    output.write(reinterpret_cast<const char *>(rebound.data()),
                 static_cast<std::streamsize>(rebound.size() *
                                              sizeof(std::uint32_t)));
    output.flush();
    if (!output)
      dif::fail("failed to write rebound generic ConvRot metadata");
    std::cout << "CONVROT_REBIND PASS program=" << program_path
              << " program_sha256=" << dif::hex_digest(program_digest)
              << " bundle=" << bundle_path << " cache=" << rebind_cache_path
              << " projections=" << plan.size() << "\n";
    return 0;
  }
  std::uint64_t expected_payload_bytes = 20U * sizeof(std::uint32_t);
  for (const auto &projection : plan) {
    if (projection.dims[0] >
        std::numeric_limits<std::uint64_t>::max() / projection.dims[1])
      dif::fail("generic ConvRot projection element count overflows");
    const auto elements = projection.dims[0] * projection.dims[1];
    if (expected_payload_bytes >
        std::numeric_limits<std::uint64_t>::max() - elements -
            projection.dims[0] * sizeof(float))
      dif::fail("generic ConvRot package size overflows");
    expected_payload_bytes +=
        elements + projection.dims[0] * sizeof(float);
  }
  constexpr std::uint64_t reserve_bytes = 8ULL << 30U;
  const auto available =
      std::filesystem::space(output_path.parent_path()).available;
  if (available < expected_payload_bytes + reserve_bytes)
    dif::fail("insufficient disk space for generic ConvRot cache plus 8 GiB reserve");

  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  specs.reserve(1U + plan.size() * 2U);
  specs.push_back(
      {"__meta__.convrot_int8", dif::ir::DType::I32, {20U}});
  for (const auto &projection : plan) {
    specs.push_back({projection.output_weight_name, dif::ir::DType::I8,
                     projection.dims});
    specs.push_back({projection.output_scale_name, dif::ir::DType::F32,
                     {projection.dims[0]}});
  }

  PartialFileCleanup cleanup(temporary);
  dif::weights::SafeTensorWriter writer(temporary, std::move(specs));
  const auto identity = generic_format_metadata(
      static_cast<std::uint32_t>(plan.size()), program_digest,
      bundle.index_fingerprint);
  writer.append(
      "__meta__.convrot_int8",
      {reinterpret_cast<const std::uint8_t *>(identity.data()),
       identity.size() * sizeof(std::uint32_t)});
#if DIF_HAS_CUDA
  NativeConvRotQuantizer quantizer;
#else
  dif::fail("this build has no CUDA support");
#endif
  std::map<std::filesystem::path, dif::weights::SafeTensorFile> shards;
  for (std::size_t projection_index = 0; projection_index < plan.size();
       ++projection_index) {
    const auto &projection = plan[projection_index];
    auto shard_path = bundle.shards.at(projection.shard_index).path;
    if (shard_path.is_relative())
      shard_path = bundle_path.parent_path() / shard_path;
    shard_path = std::filesystem::absolute(shard_path).lexically_normal();
    auto shard = shards.find(shard_path);
    if (shard == shards.end())
      shard = shards
                  .emplace(shard_path,
                           dif::weights::read_safetensors(shard_path))
                  .first;
    auto source =
        dif::weights::map_safetensor(shard->second, projection.source_name);
    if (source.dtype != projection.dtype ||
        source.dims != projection.dims)
      dif::fail("generic ConvRot source has unexpected dtype/shape: " +
                projection.source_name);
#if DIF_HAS_CUDA
    auto [quantized, scales] = quantizer.quantize_generic(source);
#endif
    writer.append(projection.output_weight_name,
                  {quantized.data(), quantized.byte_size()});
    writer.append(projection.output_scale_name,
                  {scales.data(), scales.byte_size()});
    source.discard_mapped_pages();
    std::cout << "CONVROT_QUANT tensor_id=" << projection.tensor_id
              << " source=" << projection.source_name
              << " rows=" << projection.dims[0]
              << " columns=" << projection.dims[1]
              << " progress=" << (projection_index + 1U) << "/"
              << plan.size() << "\n"
              << std::flush;
  }
  (void)writer.finish();
  const auto output_bytes = std::filesystem::file_size(temporary);
  const auto output_digest = dif::sha256_file(temporary);
  std::filesystem::rename(temporary, output_path);
  cleanup.disarm();
  std::cout << "CONVROT_PACKAGE PASS program=" << program_path
            << " program_sha256=" << dif::hex_digest(program_digest)
            << " bundle=" << bundle_path
            << " index_sha256="
            << dif::hex_digest(bundle.index_fingerprint)
            << " output=" << output_path
            << " projections=" << plan.size() << " group=" << kGroupSize
            << " bytes=" << output_bytes
            << " sha256=" << dif::hex_digest(output_digest) << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc >= 2 && std::string(argv[1]) == "--program")
      return run_generic(argc, argv);
    if (argc < 3)
      usage_error("official index and output path are required");
    const auto index_path =
        std::filesystem::absolute(argv[1]).lexically_normal();
    const auto output_path =
        std::filesystem::absolute(argv[2]).lexically_normal();
    std::uint32_t layers = 50U;
    bool groupwise_quality = false;
    bool quality_scale_f32 = false;
    std::uint32_t convrot_scale_chunk = 0U;
    auto quality_groups = kH3QualityGroups;
    for (int argument = 3; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--layers" && argument + 1 < argc)
        layers = parse_layers(argv[++argument]);
      else if (option == "--groupwise-quality")
        groupwise_quality = true;
      else if (option == "--convrot-scale-chunk" && argument + 1 < argc) {
        const auto parsed = std::stoull(argv[++argument]);
        if (parsed < kGroupSize || parsed > 2048U ||
            parsed % kGroupSize != 0U)
          usage_error("--convrot-scale-chunk must be a multiple of 256 in [256,2048]");
        convrot_scale_chunk = static_cast<std::uint32_t>(parsed);
      }
      else if (option == "--quality-scale-f32")
        quality_scale_f32 = true;
      else if (option == "--quality-groups" && argument + 4 < argc) {
        for (auto &group : quality_groups) {
          const auto parsed = std::stoull(argv[++argument]);
          if (parsed == 0U || parsed > kH3Hidden)
            usage_error("quality groups must be positive and bounded");
          group = static_cast<std::uint32_t>(parsed);
        }
      }
      else
        usage_error("unknown option " + option);
    }
    if (!std::filesystem::is_regular_file(index_path))
      dif::fail("official H3 index does not exist: " + index_path.string());
    if (groupwise_quality && convrot_scale_chunk != 0U)
      dif::fail("groupwise-quality and chunk-scaled ConvRot caches are mutually exclusive");
    const auto temporary = partial_path(output_path);
    if (std::filesystem::exists(output_path) ||
        std::filesystem::exists(temporary))
      dif::fail("refusing to overwrite output or partial file: " +
                output_path.string());
    if (!output_path.parent_path().empty() &&
        !std::filesystem::is_directory(output_path.parent_path()))
      dif::fail("output parent directory does not exist: " +
                output_path.parent_path().string());

    const auto index = dif::weights::read_safetensors_index(index_path);
    const auto source_digest = dif::sha256_file(index_path);
    const auto plan = groupwise_quality
                          ? groupwise_projections(layers, quality_groups)
                                        : projections(layers);
    for (const auto &projection : plan)
      if (groupwise_quality &&
          projection.dims[1] % projection.group_size != 0U)
        dif::fail("H3 groupwise profile does not divide projection width");
    std::uint64_t expected_payload_bytes = groupwise_quality
                                               ? 0U
                                               : (convrot_scale_chunk == 0U
                                                      ? 14U
                                                      : 15U) *
                                                     sizeof(std::uint32_t);
    for (const auto &projection : plan) {
      const auto elements = projection.dims[0] * projection.dims[1];
      const auto scale_bytes = groupwise_quality
          ? projection.dims[0] *
                (projection.dims[1] / projection.group_size) *
                (quality_scale_f32 ? sizeof(float) : sizeof(std::uint16_t))
          : projection.dims[0] *
                (convrot_scale_chunk == 0U
                     ? 1U
                     : (projection.dims[1] + convrot_scale_chunk - 1U) /
                           convrot_scale_chunk) *
                sizeof(float);
      expected_payload_bytes += elements + scale_bytes;
    }
    constexpr std::uint64_t reserve_bytes = 8ULL << 30U;
    const auto available =
        std::filesystem::space(output_path.parent_path()).available;
    if (available < expected_payload_bytes + reserve_bytes)
      dif::fail(
          "insufficient disk space for H3 ConvRot cache plus 8 GiB reserve");
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    specs.reserve((groupwise_quality ? 0U : 1U) + plan.size() * 2U);
    if (!groupwise_quality)
      specs.push_back({"__meta__.h3_convrot", dif::ir::DType::I32,
                       {convrot_scale_chunk == 0U ? 14U : 15U}});
    for (const auto &projection : plan) {
      specs.push_back({projection.output_weight_name, dif::ir::DType::I8,
                       projection.dims});
      if (groupwise_quality)
        specs.push_back({projection.output_scale_name,
                         quality_scale_f32 ? dif::ir::DType::F32
                                           : dif::ir::DType::F16,
                         {projection.dims[0],
                          projection.dims[1] / projection.group_size}});
      else if (convrot_scale_chunk != 0U)
        specs.push_back({projection.output_scale_name, dif::ir::DType::F32,
                         {projection.dims[0],
                          (projection.dims[1] + convrot_scale_chunk - 1U) /
                              convrot_scale_chunk}});
      else
        specs.push_back({projection.output_scale_name, dif::ir::DType::F32,
                         {projection.dims[0]}});
    }

    PartialFileCleanup cleanup(temporary);
    dif::weights::SafeTensorWriter writer(temporary, std::move(specs));
    if (!groupwise_quality) {
      if (convrot_scale_chunk != 0U) {
        const auto identity = chunk_scaled_format_metadata(
            layers, convrot_scale_chunk, source_digest);
        writer.append(
            "__meta__.h3_convrot",
            {reinterpret_cast<const std::uint8_t *>(identity.data()),
             identity.size() * sizeof(std::uint32_t)});
      } else {
        const auto identity = format_metadata(layers, source_digest);
        writer.append(
            "__meta__.h3_convrot",
            {reinterpret_cast<const std::uint8_t *>(identity.data()),
             identity.size() * sizeof(std::uint32_t)});
      }
    }

#if DIF_HAS_CUDA
    NativeConvRotQuantizer quantizer;
#else
    dif::fail("this build has no CUDA support");
#endif
    std::map<std::filesystem::path, dif::weights::SafeTensorFile> shards;
    for (std::size_t projection_index = 0; projection_index < plan.size();
         ++projection_index) {
      const auto &projection = plan[projection_index];
      const auto mapped = index.weight_map.find(projection.source_name);
      if (mapped == index.weight_map.end())
        dif::fail("official H3 index lacks " + projection.source_name);
      auto shard_path = mapped->second;
      if (shard_path.is_relative())
        shard_path = index_path.parent_path() / shard_path;
      shard_path = std::filesystem::absolute(shard_path).lexically_normal();
      auto shard = shards.find(shard_path);
      if (shard == shards.end())
        shard = shards.emplace(shard_path,
                               dif::weights::read_safetensors(shard_path))
                    .first;
      auto source =
          dif::weights::map_safetensor(shard->second, projection.source_name);
      if (source.dtype != dif::ir::DType::BF16 ||
          source.dims != projection.dims)
        dif::fail("official H3 projection has unexpected dtype/shape: " +
                  projection.source_name);
#if DIF_HAS_CUDA
      auto converted = groupwise_quality
                           ? quantizer.quantize_groupwise(
                                 source, projection.group_size,
                                 projection.reorder_qkv,
                                 projection.swap_row_halves,
                                 quality_scale_f32)
                           : (convrot_scale_chunk != 0U
                                  ? quantizer.quantize_chunked(
                                        source, projection.reorder_qkv,
                                        convrot_scale_chunk)
                                  : quantizer.quantize(
                                        source, projection.reorder_qkv));
      auto &quantized = converted.first;
      auto &scales = converted.second;
#endif
      writer.append(projection.output_weight_name,
                    {quantized.data(), quantized.byte_size()});
      writer.append(projection.output_scale_name,
                    {scales.data(), scales.byte_size()});
      source.discard_mapped_pages();
      std::cout << (groupwise_quality
                        ? "H3_GROUPWISE_QUANT projection="
                        : (convrot_scale_chunk != 0U
                               ? "H3_CONVROT_CHUNK_QUANT projection="
                               : "H3_CONVROT_QUANT projection="))
                << projection.source_name << " output="
                << projection.output_weight_name << " rows="
                << projection.dims[0] << " columns=" << projection.dims[1]
                << " progress=" << (projection_index + 1U) << "/"
                << plan.size() << "\n"
                << std::flush;
    }
    const auto metadata = writer.finish();
    (void)metadata;
    const auto output_bytes = std::filesystem::file_size(temporary);
    const auto output_digest = dif::sha256_file(temporary);
    std::filesystem::rename(temporary, output_path);
    cleanup.disarm();
    std::cout << (groupwise_quality
                      ? "H3_GROUPWISE_PACKAGE PASS source_index="
                      : (convrot_scale_chunk != 0U
                             ? "H3_CONVROT_CHUNK_PACKAGE PASS source_index="
                             : "H3_CONVROT_PACKAGE PASS source_index="))
              << index_path
              << " source_index_sha256=" << dif::hex_digest(source_digest)
              << " output=" << output_path << " layers=" << layers
              << " projections=" << plan.size()
              << (groupwise_quality ? " groups=q" : " group=256")
              << (groupwise_quality ? std::to_string(quality_groups[0]) : "")
              << (groupwise_quality ? "_o" : "")
              << (groupwise_quality ? std::to_string(quality_groups[1]) : "")
              << (groupwise_quality ? "_fc1_" : "")
              << (groupwise_quality ? std::to_string(quality_groups[2]) : "")
              << (groupwise_quality ? "_fc2_" : "")
              << (groupwise_quality ? std::to_string(quality_groups[3]) : "")
              << (groupwise_quality
                      ? (quality_scale_f32 ? "_scale_f32" : "_scale_f16")
                      : "")
              << (groupwise_quality ? "" : " qkv_layout=contiguous_q_k_v")
              << (convrot_scale_chunk == 0U
                      ? ""
                      : " scale_chunk=" +
                            std::to_string(convrot_scale_chunk))
              << " bytes=" << output_bytes
              << " sha256=" << dif::hex_digest(output_digest) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3convrot: " << error.what() << "\n";
    return 1;
  }
}
