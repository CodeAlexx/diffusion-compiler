// Native MiniMax-H3 ConvRot INT8 projection-cache builder.
//
// Reads the official sharded BF16 transformer checkpoint, applies the H256
// block rotation and deterministic per-output-row INT8 quantization on the
// installed NVIDIA GPU, and emits only the four projection matrices used by
// each H3 block.  The resulting cache is a derived Diffusion Compiler artifact;
// it is not a ComfyUI checkpoint and the production runtime does not import
// Python, PyTorch, or Comfy Kitchen.

#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <array>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <filesystem>
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
constexpr std::uint32_t kGroupSize = 256U;
constexpr std::uint32_t kQkvLayoutContiguous = 1U;
constexpr std::uint32_t kProjectionCount = 4U;
constexpr std::uint64_t kH3Hidden = 5376U;
constexpr std::uint64_t kH3Inner = 7168U;
constexpr std::uint64_t kH3HeadDim = 128U;
constexpr std::uint64_t kH3Ffn = 14336U;

struct Projection {
  std::string source_name;
  std::string output_weight_name;
  std::string output_scale_name;
  std::vector<std::uint64_t> dims;
  bool reorder_qkv{};
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
  std::cerr << "difh3convrot: " << message << "\n"
            << "usage: difh3convrot OFFICIAL.index.json OUT.safetensors "
               "[--layers N]\n";
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
__device__ __forceinline__ float round_bf16(float value) {
  unsigned int bits = __float_as_uint(value);
  unsigned int exponent = bits & 0x7f800000U;
  if (exponent == 0x7f800000U)
    return value;
  unsigned int lsb = (bits >> 16U) & 1U;
  bits += 0x7fffU + lsb;
  return __uint_as_float(bits & 0xffff0000U);
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

private:
  CUdevice device_{};
  CUcontext context_{};
  CUmodule module_{};
  CUfunction function_{};
};

#endif

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3)
      usage_error("official index and output path are required");
    const auto index_path =
        std::filesystem::absolute(argv[1]).lexically_normal();
    const auto output_path =
        std::filesystem::absolute(argv[2]).lexically_normal();
    std::uint32_t layers = 50U;
    for (int argument = 3; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--layers" && argument + 1 < argc)
        layers = parse_layers(argv[++argument]);
      else
        usage_error("unknown option " + option);
    }
    if (!std::filesystem::is_regular_file(index_path))
      dif::fail("official H3 index does not exist: " + index_path.string());
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
    const auto plan = projections(layers);
    std::uint64_t expected_payload_bytes = 14U * sizeof(std::uint32_t);
    for (const auto &projection : plan) {
      const auto elements = projection.dims[0] * projection.dims[1];
      expected_payload_bytes +=
          elements + projection.dims[0] * sizeof(float);
    }
    constexpr std::uint64_t reserve_bytes = 8ULL << 30U;
    const auto available =
        std::filesystem::space(output_path.parent_path()).available;
    if (available < expected_payload_bytes + reserve_bytes)
      dif::fail(
          "insufficient disk space for H3 ConvRot cache plus 8 GiB reserve");
    std::vector<dif::weights::SafeTensorWriteSpec> specs;
    specs.reserve(1U + plan.size() * 2U);
    specs.push_back({"__meta__.h3_convrot", dif::ir::DType::I32, {14U}});
    for (const auto &projection : plan) {
      specs.push_back({projection.output_weight_name, dif::ir::DType::I8,
                       projection.dims});
      specs.push_back({projection.output_scale_name, dif::ir::DType::F32,
                       {projection.dims[0]}});
    }

    PartialFileCleanup cleanup(temporary);
    dif::weights::SafeTensorWriter writer(temporary, std::move(specs));
    const auto identity = format_metadata(layers, source_digest);
    writer.append(
        "__meta__.h3_convrot",
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
      auto [quantized, scales] =
          quantizer.quantize(source, projection.reorder_qkv);
#endif
      writer.append(projection.output_weight_name,
                    {quantized.data(), quantized.byte_size()});
      writer.append(projection.output_scale_name,
                    {scales.data(), scales.byte_size()});
      source.discard_mapped_pages();
      std::cout << "H3_CONVROT_QUANT projection="
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
    std::cout << "H3_CONVROT_PACKAGE PASS source_index=" << index_path
              << " source_index_sha256=" << dif::hex_digest(source_digest)
              << " output=" << output_path << " layers=" << layers
              << " projections=" << plan.size()
              << " group=" << kGroupSize
              << " qkv_layout=contiguous_q_k_v"
              << " bytes=" << output_bytes
              << " sha256=" << dif::hex_digest(output_digest) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3convrot: " << error.what() << "\n";
    return 1;
  }
}
