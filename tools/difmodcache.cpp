// difmodcache — native MiniMax-H3 AdaLN modulation-cache builder.
//
// Replaces the silent Serenity dependency: the compiler consumed
// modcache_steps_*.safetensors that only serenitymojo could generate
// (serenitymojo/models/dit/minimax_h3_modcache.mojo + minimax_h3_runtime_cache
// .mojo + pipeline/minimax_h3_t2va.mojo section 5). This tool reproduces that
// generator natively, aiming for BYTE-IDENTICAL output:
//
//   timesteps[2i]   = 1.0f - video_sigmas[i]          (i in [0, points-1))
//   timesteps[2i+1] = 1.0f - audio_sigmas[i]
//   sinusoid  : freq = exp(-ln(10000)*(i/128)), cos block then sin block
//   temb      : sinusoid -> linear(proj_in) -> silu -> linear(proj_out), F32
//   activated : bf16_rne(silu(temb))
//   block N   : bf16_rne(activated @ adaln_w[N]^T (F32 accum) + f32(bias))
//   final     : same with final_layer.adaln_proj
//
// Exactness notes (from reading the Mojo sources, pinned Diffusers reference
// /home/alex/minimax_h3_ref/.../transformer_minimax_h3.py, and the Modular
// stdlib at /home/alex/modular):
//   * Mojo GPU exp(x) f32 == ex2.approx.ftz.f32(x * log2e_f32); cos/sin f32
//     == cos/sin.approx.ftz.f32; casts to bf16 are RNE (cvt.rn.bf16.f32).
//     The CUDA engine here uses the same PTX instructions inline.
//   * -ln(10000) is computed on the HOST by Mojo's _log_base[27] (Cephes
//     polynomial with FMA Horner), NOT libm; replicated in mojo_host_log_f32.
//   * GEMMs go through linalg.matmul.vendor.blas -> cublasGemmEx with
//     CUBLAS_DEFAULT_MATH, COMPUTE_32F, CUBLAS_GEMM_DEFAULT, and the
//     row-major swap (opB,opA / N,M,K). Serenity ran on the pixi env's
//     cuBLAS 13.6; --cublas selects the library to dlopen so the exact same
//     binary can be used.
//   * The SafeTensors writer mirrors serenitymojo/io/safetensors_writer.mojo:
//     compact JSON, insertion order, no padding, contiguous offsets, and the
//     __meta__ tensors of models/dit/minimax_h3_runtime_cache.mojo
//     (version, kind, src_path, src_size, src_mtime, steps,
//     distinct_timesteps, nblocks) followed by block.0..N-1 and final.
//
// The --verify-against mode compares the generated file to a recorded cache
// tensor-by-tensor and reports exact element mismatch counts.

#include "dif/runtime/tensor.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <sys/stat.h>

#if DIF_HAS_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace fs = std::filesystem;

namespace {

// ── Mojo host log (stdlib _log_base[27], Cephes + FMA Horner) ───────────────
float mojo_host_log_f32(float x) {
  int exponent_int = 0;
  float fraction = std::frexp(x, &exponent_int);
  float exponent = static_cast<float>(exponent_int);
  if (fraction < 0.70710678118654752440F) {
    exponent -= 1.0F;
    fraction = fraction + fraction;
  }
  const float x1 = fraction - 1.0F;
  const float x2 = x1 * x1;
  const float x3 = x2 * x1;
  const float coefficients[9] = {
      3.3333331174e-1F, -2.4999993993e-1F, 2.0000714765e-1F,
      -1.6668057665e-1F, 1.4249322787e-1F, -1.2420140846e-1F,
      1.1676998740e-1F, -1.1514610310e-1F, 7.0376836292e-2F};
  float horner = std::fmaf(x1, coefficients[8], coefficients[7]);
  for (int index = 6; index >= 0; --index)
    horner = std::fmaf(horner, x1, coefficients[index]);
  float y = horner * x3;
  y = x1 + std::fmaf(x2, -0.5F, y);
  return std::fmaf(exponent, 0.69314718055994530942F, y);
}

std::uint16_t bf16_rne_host(float value) {
  std::uint32_t bits;
  std::memcpy(&bits, &value, 4);
  const std::uint32_t lsb = (bits >> 16) & 1U;
  bits += 0x7FFFU + lsb;
  return static_cast<std::uint16_t>(bits >> 16);
}

float bf16_to_f32_host(std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16;
  float out;
  std::memcpy(&out, &bits, 4);
  return out;
}

struct Arguments {
  fs::path checkpoint_index;
  fs::path video_sigmas;
  fs::path audio_sigmas;
  fs::path output;
  fs::path verify_against;
  std::string source_index_string;
  std::string engine = "cuda";
  std::string cublas_path =
      "/home/alex/mojodiffusion/.pixi/envs/default/lib/libcublas.so.13";
  long long schedule_points = -1;
  long long steps = -1;
};

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr << "difmodcache: " << message << "\n"
            << "usage: difmodcache --checkpoint-index INDEX.json "
               "(--video-sigmas F.diftensor --audio-sigmas F.diftensor | "
               "--schedule-points N) [--steps N] --output FILE.safetensors "
               "[--engine cuda|cpu] [--cublas LIB.so] "
               "[--source-index-string STR] "
               "[--verify-against FILE.safetensors]\n";
  std::exit(2);
}

// ── minimal SafeTensors emitter mirroring serenity's writer ─────────────────
struct EmitTensor {
  std::string name;
  std::string dtype;
  std::vector<std::uint64_t> shape;
  std::vector<std::uint8_t> bytes;
};

EmitTensor meta_i64(const std::string &name, std::int64_t value) {
  EmitTensor tensor;
  tensor.name = name;
  tensor.dtype = "I64";
  tensor.shape = {1U};
  tensor.bytes.resize(8U);
  for (unsigned index = 0; index < 8U; ++index)
    tensor.bytes[index] =
        static_cast<std::uint8_t>(static_cast<std::uint64_t>(value) >> (8U * index));
  return tensor;
}

EmitTensor meta_u8(const std::string &name, const std::string &value) {
  EmitTensor tensor;
  tensor.name = name;
  tensor.dtype = "U8";
  tensor.shape = {static_cast<std::uint64_t>(value.size())};
  tensor.bytes.assign(value.begin(), value.end());
  return tensor;
}

void write_serenity_safetensors(const std::vector<EmitTensor> &tensors,
                                const fs::path &path) {
  std::ostringstream header;
  header << '{';
  std::uint64_t offset = 0;
  for (std::size_t index = 0; index < tensors.size(); ++index) {
    const auto &tensor = tensors[index];
    if (index)
      header << ',';
    header << '"' << tensor.name << "\":{\"dtype\":\"" << tensor.dtype
           << "\",\"shape\":[";
    for (std::size_t dim = 0; dim < tensor.shape.size(); ++dim)
      header << (dim ? "," : "") << tensor.shape[dim];
    header << "],\"data_offsets\":[" << offset << ','
           << offset + tensor.bytes.size() << "]}";
    offset += tensor.bytes.size();
  }
  header << '}';
  const auto header_text = header.str();

  const auto tmp_path = path.string() + ".tmp";
  std::ofstream output(tmp_path, std::ios::binary | std::ios::trunc);
  if (!output)
    dif::fail("cannot create cache file: " + tmp_path);
  const std::uint64_t header_length = header_text.size();
  char length_bytes[8];
  for (unsigned index = 0; index < 8U; ++index)
    length_bytes[index] = static_cast<char>(header_length >> (8U * index));
  output.write(length_bytes, 8);
  output.write(header_text.data(),
               static_cast<std::streamsize>(header_text.size()));
  for (const auto &tensor : tensors)
    output.write(reinterpret_cast<const char *>(tensor.bytes.data()),
                 static_cast<std::streamsize>(tensor.bytes.size()));
  if (!output)
    dif::fail("cannot write cache file: " + tmp_path);
  output.close();
  std::error_code rename_error;
  fs::rename(tmp_path, path, rename_error);
  if (rename_error)
    dif::fail("cannot rename cache file into place: " + rename_error.message());
}

// ── checkpoint access ───────────────────────────────────────────────────────
struct ShardCache {
  std::map<std::string, dif::weights::SafeTensorFile> files;
  const dif::weights::SafeTensorFile &open(const fs::path &path) {
    const auto key = path.string();
    auto found = files.find(key);
    if (found == files.end())
      found = files.emplace(key, dif::weights::read_safetensors(path)).first;
    return found->second;
  }
};

dif::runtime::Tensor load_checkpoint_tensor(
    const dif::weights::SafeTensorIndex &index, ShardCache &shards,
    const std::string &name, dif::ir::DType expected_dtype) {
  const auto location = index.weight_map.find(name);
  if (location == index.weight_map.end())
    dif::fail("checkpoint index is missing " + name);
  const auto &file = shards.open(location->second);
  auto tensor = dif::weights::map_safetensor(file, name);
  if (tensor.dtype != expected_dtype)
    dif::fail("checkpoint tensor has unexpected dtype: " + name);
  return tensor;
}

// ── CUDA engine ─────────────────────────────────────────────────────────────
#if DIF_HAS_CUDA

void check_cu(CUresult result, const char *what) {
  if (result != CUDA_SUCCESS) {
    const char *message = nullptr;
    cuGetErrorString(result, &message);
    dif::fail(std::string("CUDA error in ") + what + ": " +
              (message ? message : "unknown"));
  }
}

void check_nvrtc(nvrtcResult result, const char *what) {
  if (result != NVRTC_SUCCESS)
    dif::fail(std::string("NVRTC error in ") + what + ": " +
              nvrtcGetErrorString(result));
}

constexpr const char *kKernelSource = R"CUDA(
typedef unsigned short u16;
typedef unsigned int u32;
typedef long long i64;

__device__ __forceinline__ float ex2_ftz(float x) {
  float r; asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float cos_ap(float x) {
  float r; asm("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float sin_ap(float x) {
  float r; asm("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x)); return r;
}
__device__ __forceinline__ float exp_mojo(float x) {
  return ex2_ftz(x * __uint_as_float(0x3FB8AA3Bu));  // log2e as f32
}
__device__ __forceinline__ u16 bf16_rn(float v) {
  u16 r; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float bf16_up(u16 v) {
  return __uint_as_float(((u32)v) << 16);
}

extern "C" __global__ void timestep_embed(const float *t, float *o, i64 n_w,
                                          int dim, int half, float neg_ln_mp) {
  i64 total = n_w * (i64)half;
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int row = (int)(idx / half);
    int i = (int)(idx % half);
    float tv = t[row];
    float freq = exp_mojo(neg_ln_mp * ((float)i / (float)half));
    float angle = tv * freq;
    o[(i64)row * dim + i] = cos_ap(angle);
    o[(i64)row * dim + half + i] = sin_ap(angle);
  }
}

extern "C" __global__ void silu_f32(const float *x, float *o, i64 n) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    o[i] = v / (1.0f + exp_mojo(-v));
  }
}

extern "C" __global__ void bias_direct_f32(const float *c, const float *bias,
                                           float *o, int m, int out_dim) {
  i64 total = (i64)m * out_dim;
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int col = (int)(idx % out_dim);
    o[idx] = c[idx] + bias[col];
  }
}

extern "C" __global__ void bias_direct_bf16(const float *c, const u16 *bias,
                                            u16 *o, int m, int out_dim) {
  i64 total = (i64)m * out_dim;
  i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    int col = (int)(idx % out_dim);
    float v = c[idx] + bf16_up(bias[col]);
    o[idx] = bf16_rn(v);
  }
}

extern "C" __global__ void cast_f32_bf16(const float *x, u16 *o, i64 n) {
  i64 i = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    o[i] = bf16_rn(x[i]);
}
)CUDA";

// cuBLAS via dlopen so the EXACT library Serenity ran on can be used.
struct CublasApi {
  void *library = nullptr;
  void *handle = nullptr;
  int (*create)(void **) = nullptr;
  int (*destroy)(void *) = nullptr;
  int (*set_math_mode)(void *, int) = nullptr;
  int (*gemm_ex)(void *, int, int, int, int, int, const void *, const void *,
                 int, int, const void *, int, int, const void *, void *, int,
                 int, int, int) = nullptr;

  void open(const std::string &path) {
    const fs::path library_path(path);
    if (library_path.has_parent_path()) {
      const auto sibling = library_path.parent_path() / "libcublasLt.so.13";
      if (fs::exists(sibling))
        (void)dlopen(sibling.c_str(), RTLD_NOW | RTLD_GLOBAL);
    }
    library = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!library)
      dif::fail(std::string("cannot dlopen cuBLAS: ") + dlerror());
    create = reinterpret_cast<int (*)(void **)>(dlsym(library, "cublasCreate_v2"));
    destroy = reinterpret_cast<int (*)(void *)>(dlsym(library, "cublasDestroy_v2"));
    set_math_mode =
        reinterpret_cast<int (*)(void *, int)>(dlsym(library, "cublasSetMathMode"));
    gemm_ex = reinterpret_cast<
        int (*)(void *, int, int, int, int, int, const void *, const void *,
                int, int, const void *, int, int, const void *, void *, int,
                int, int, int)>(dlsym(library, "cublasGemmEx"));
    if (!create || !destroy || !set_math_mode || !gemm_ex)
      dif::fail("cuBLAS library is missing required symbols");
    if (create(&handle) != 0)
      dif::fail("cublasCreate failed");
    if (set_math_mode(handle, 0) != 0)  // CUBLAS_DEFAULT_MATH
      dif::fail("cublasSetMathMode failed");
  }

  // C[M,N] row-major = A[M,K] @ B[N,K]^T, F32 accumulate — the exact
  // c_row_major=True / transpose_b=True arm of Modular's _cublas_matmul.
  void gemm_rowmajor_bt(int m, int n, int k, CUdeviceptr a, int a_type,
                        CUdeviceptr b, int b_type, CUdeviceptr c) {
    const float alpha = 1.0F;
    const float beta = 0.0F;
    // 0=CUBLAS_OP_N, 1=CUBLAS_OP_T; CUDA_R_32F=0, CUDA_R_16BF=14;
    // CUBLAS_COMPUTE_32F=68; CUBLAS_GEMM_DEFAULT=-1.
    const int status = gemm_ex(
        handle, /*transa=*/1, /*transb=*/0, /*m=*/n, /*n=*/m, /*k=*/k, &alpha,
        reinterpret_cast<const void *>(b), b_type, /*ldb=*/k,
        reinterpret_cast<const void *>(a), a_type, /*lda=*/k, &beta,
        reinterpret_cast<void *>(c), /*Ctype=*/0, /*ldc=*/n,
        /*computeType=*/68, /*algo=*/-1);
    if (status != 0)
      dif::fail("cublasGemmEx failed with status " + std::to_string(status));
  }

  ~CublasApi() {
    if (handle && destroy)
      destroy(handle);
  }
};

struct CudaEngine {
  CUcontext context{};
  CUdevice device{};
  CUmodule module{};
  CUfunction fn_timestep{}, fn_silu{}, fn_bias_f32{}, fn_bias_bf16{}, fn_cast{};
  CublasApi cublas;

  void init(const std::string &cublas_path) {
    check_cu(cuInit(0), "cuInit");
    check_cu(cuDeviceGet(&device, 0), "cuDeviceGet");
    check_cu(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    check_cu(cuCtxSetCurrent(context), "cuCtxSetCurrent");

    int major = 0, minor = 0;
    check_cu(cuDeviceGetAttribute(
                 &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
             "cc major");
    check_cu(cuDeviceGetAttribute(
                 &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
             "cc minor");
    const auto arch = "--gpu-architecture=compute_" + std::to_string(major) +
                      std::to_string(minor);

    nvrtcProgram program{};
    check_nvrtc(nvrtcCreateProgram(&program, kKernelSource, "difmodcache.cu",
                                   0, nullptr, nullptr),
                "nvrtcCreateProgram");
    const char *options[] = {arch.c_str()};
    const auto compile_status = nvrtcCompileProgram(program, 1, options);
    if (compile_status != NVRTC_SUCCESS) {
      std::size_t log_size = 0;
      nvrtcGetProgramLogSize(program, &log_size);
      std::string log(log_size, '\0');
      nvrtcGetProgramLog(program, log.data());
      dif::fail("NVRTC compile failed: " + log);
    }
    std::size_t ptx_size = 0;
    check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "nvrtcGetPTXSize");
    std::string ptx(ptx_size, '\0');
    check_nvrtc(nvrtcGetPTX(program, ptx.data()), "nvrtcGetPTX");
    nvrtcDestroyProgram(&program);

    check_cu(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr),
             "cuModuleLoadDataEx");
    check_cu(cuModuleGetFunction(&fn_timestep, module, "timestep_embed"), "fn");
    check_cu(cuModuleGetFunction(&fn_silu, module, "silu_f32"), "fn");
    check_cu(cuModuleGetFunction(&fn_bias_f32, module, "bias_direct_f32"), "fn");
    check_cu(cuModuleGetFunction(&fn_bias_bf16, module, "bias_direct_bf16"), "fn");
    check_cu(cuModuleGetFunction(&fn_cast, module, "cast_f32_bf16"), "fn");

    cublas.open(cublas_path);
  }

  CUdeviceptr upload(const void *data, std::size_t bytes) {
    CUdeviceptr pointer{};
    check_cu(cuMemAlloc(&pointer, bytes), "cuMemAlloc");
    check_cu(cuMemcpyHtoD(pointer, data, bytes), "cuMemcpyHtoD");
    return pointer;
  }

  void launch1d(CUfunction function, std::int64_t total, void **arguments) {
    const unsigned block = 256;
    const auto grid =
        static_cast<unsigned>((total + block - 1) / static_cast<std::int64_t>(block));
    check_cu(cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr,
                            arguments, nullptr),
             "cuLaunchKernel");
  }
};

#endif // DIF_HAS_CUDA

// ── CPU diagnostic engine (double-accum GEMM + libm; NOT expected to be
//    byte-exact against the GPU generator — used only to bound mismatches) ──
std::vector<float> cpu_linear_f32(const std::vector<float> &input, int m, int k,
                                  const float *weight, const float *bias,
                                  int out_dim) {
  std::vector<float> output(static_cast<std::size_t>(m) * out_dim);
  for (int row = 0; row < m; ++row)
    for (int column = 0; column < out_dim; ++column) {
      double accumulator = 0.0;
      const float *w = weight + static_cast<std::size_t>(column) * k;
      const float *x = input.data() + static_cast<std::size_t>(row) * k;
      for (int inner = 0; inner < k; ++inner)
        accumulator += static_cast<double>(x[inner]) * w[inner];
      output[static_cast<std::size_t>(row) * out_dim + column] =
          static_cast<float>(accumulator) + bias[column];
    }
  return output;
}

} // namespace

int main(int argc, char **argv) {
  try {
    Arguments arguments;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      const auto value = [&](const char *name) -> std::string {
        if (i + 1 >= argc)
          usage_error(std::string("missing value for ") + name);
        return argv[++i];
      };
      if (option == "--checkpoint-index")
        arguments.checkpoint_index = value("--checkpoint-index");
      else if (option == "--video-sigmas")
        arguments.video_sigmas = value("--video-sigmas");
      else if (option == "--audio-sigmas")
        arguments.audio_sigmas = value("--audio-sigmas");
      else if (option == "--schedule-points")
        arguments.schedule_points =
            std::stoll(value("--schedule-points"));
      else if (option == "--steps")
        arguments.steps = std::stoll(value("--steps"));
      else if (option == "--output")
        arguments.output = value("--output");
      else if (option == "--engine")
        arguments.engine = value("--engine");
      else if (option == "--cublas")
        arguments.cublas_path = value("--cublas");
      else if (option == "--source-index-string")
        arguments.source_index_string = value("--source-index-string");
      else if (option == "--verify-against")
        arguments.verify_against = value("--verify-against");
      else
        usage_error("unknown option " + option);
    }
    if (arguments.checkpoint_index.empty() || arguments.output.empty())
      usage_error("--checkpoint-index and --output are required");
    if (arguments.engine != "cuda" && arguments.engine != "cpu")
      usage_error("--engine must be cuda or cpu");

    // ── schedule → interleaved model timesteps ──────────────────────────────
    std::vector<float> video_sigmas, audio_sigmas;
    if (!arguments.video_sigmas.empty() || !arguments.audio_sigmas.empty()) {
      if (arguments.video_sigmas.empty() || arguments.audio_sigmas.empty())
        usage_error("--video-sigmas and --audio-sigmas go together");
      auto video_tensor = dif::runtime::read_tensor(arguments.video_sigmas);
      auto audio_tensor = dif::runtime::read_tensor(arguments.audio_sigmas);
      if (video_tensor.dtype != dif::ir::DType::F32 ||
          audio_tensor.dtype != dif::ir::DType::F32 ||
          video_tensor.dims.size() != 1U ||
          audio_tensor.dims != video_tensor.dims)
        dif::fail("sigma tensors must be matching 1-D F32");
      const auto video_span = video_tensor.f32();
      const auto audio_span = audio_tensor.f32();
      video_sigmas.assign(video_span.begin(), video_span.end());
      audio_sigmas.assign(audio_span.begin(), audio_span.end());
    }
    if (arguments.schedule_points > 0) {
      const auto video_schedule = dif::sampling::make_exponential_shifted_schedule(
          static_cast<std::uint32_t>(arguments.schedule_points), 12.0F);
      const auto audio_schedule = dif::sampling::make_exponential_shifted_schedule(
          static_cast<std::uint32_t>(arguments.schedule_points), 3.0F);
      if (!video_sigmas.empty()) {
        if (video_sigmas != video_schedule.sigmas ||
            audio_sigmas != audio_schedule.sigmas)
          dif::fail("recorded sigmas disagree with the derived schedule");
        std::cout << "SCHEDULE_MATCH derived == recorded sigmas ("
                  << video_sigmas.size() << " points)\n";
      } else {
        video_sigmas = video_schedule.sigmas;
        audio_sigmas = audio_schedule.sigmas;
      }
    }
    if (video_sigmas.size() < 2U)
      usage_error("no schedule given (need sigmas files or --schedule-points)");
    const auto points = video_sigmas.size();
    if (video_sigmas.back() != 0.0F || audio_sigmas.back() != 0.0F)
      dif::fail("sigma schedules must end at zero");
    const auto evaluations = points - 1U;
    const auto distinct_timesteps = 2U * evaluations;
    if (arguments.steps < 0)
      arguments.steps = static_cast<long long>(points);

    std::vector<float> timesteps(distinct_timesteps);
    for (std::size_t i = 0; i < evaluations; ++i) {
      timesteps[2U * i] = 1.0F - video_sigmas[i];
      timesteps[2U * i + 1U] = 1.0F - audio_sigmas[i];
    }

    // ── checkpoint tensors ──────────────────────────────────────────────────
    const auto index =
        dif::weights::read_safetensors_index(arguments.checkpoint_index);
    ShardCache shards;
    const auto proj_in_w = load_checkpoint_tensor(
        index, shards, "time_embedder.proj_in.weight", dif::ir::DType::F32);
    const auto proj_in_b = load_checkpoint_tensor(
        index, shards, "time_embedder.proj_in.bias", dif::ir::DType::F32);
    const auto proj_out_w = load_checkpoint_tensor(
        index, shards, "time_embedder.proj_out.weight", dif::ir::DType::F32);
    const auto proj_out_b = load_checkpoint_tensor(
        index, shards, "time_embedder.proj_out.bias", dif::ir::DType::F32);
    if (proj_in_w.dims.size() != 2U || proj_out_w.dims.size() != 2U)
      dif::fail("time_embedder weights have unexpected rank");
    const auto freq_dim = static_cast<int>(proj_in_w.dims.at(1));       // 256
    const auto embed_hidden = static_cast<int>(proj_in_w.dims.at(0));   // 5376
    const auto time_embed_dim = static_cast<int>(proj_out_w.dims.at(0)); // 2688
    if (freq_dim % 2 != 0)
      dif::fail("freq_dim must be even");

    std::size_t num_blocks = 0;
    while (index.weight_map.find("blocks." + std::to_string(num_blocks) +
                                 ".adaln_proj.linear.weight") !=
           index.weight_map.end())
      ++num_blocks;
    if (num_blocks == 0)
      dif::fail("checkpoint has no adaln_proj blocks");

    const float neg_ln_mp = -mojo_host_log_f32(10000.0F);
    std::uint32_t neg_ln_mp_bits;
    std::memcpy(&neg_ln_mp_bits, &neg_ln_mp, 4);
    std::cout << "MODCACHE build: blocks=" << num_blocks
              << " distinct_timesteps=" << distinct_timesteps
              << " freq_dim=" << freq_dim << " hidden=" << embed_hidden
              << " time_embed_dim=" << time_embed_dim << " engine="
              << arguments.engine << " neg_ln_mp=" << std::hexfloat << neg_ln_mp
              << std::defaultfloat << " (bits 0x" << std::hex << neg_ln_mp_bits
              << std::dec << ")\n";

    const int rows = static_cast<int>(distinct_timesteps);
    std::vector<std::vector<std::uint8_t>> block_payloads(num_blocks);
    std::vector<std::uint8_t> final_payload;
    const auto adaln_out = 18U * (embed_hidden / 18U == 0 ? 0U : 0U); // unused
    (void)adaln_out;

    if (arguments.engine == "cuda") {
#if DIF_HAS_CUDA
      CudaEngine engine;
      engine.init(arguments.cublas_path);

      const auto d_timesteps =
          engine.upload(timesteps.data(), timesteps.size() * 4U);
      CUdeviceptr d_sinusoid{};
      check_cu(cuMemAlloc(&d_sinusoid,
                          static_cast<std::size_t>(rows) * freq_dim * 4U),
               "alloc sinusoid");
      {
        std::int64_t n = rows;
        int dim = freq_dim, half = freq_dim / 2;
        float neg = neg_ln_mp;
        void *arguments_list[] = {const_cast<CUdeviceptr *>(&d_timesteps),
                                  &d_sinusoid, &n, &dim, &half, &neg};
        engine.launch1d(engine.fn_timestep,
                        static_cast<std::int64_t>(rows) * (freq_dim / 2),
                        arguments_list);
      }

      const auto run_linear_f32 =
          [&](CUdeviceptr input, int k, const dif::runtime::Tensor &weight,
              const dif::runtime::Tensor &bias, int out_dim) -> CUdeviceptr {
        const auto d_weight = engine.upload(weight.data(), weight.byte_size());
        const auto d_bias = engine.upload(bias.data(), bias.byte_size());
        CUdeviceptr d_gemm{}, d_out{};
        check_cu(cuMemAlloc(&d_gemm,
                            static_cast<std::size_t>(rows) * out_dim * 4U),
                 "alloc gemm");
        check_cu(cuMemAlloc(&d_out,
                            static_cast<std::size_t>(rows) * out_dim * 4U),
                 "alloc out");
        engine.cublas.gemm_rowmajor_bt(rows, out_dim, k, input, /*f32*/ 0,
                                       d_weight, 0, d_gemm);
        int m = rows, n = out_dim;
        void *arguments_list[] = {&d_gemm, const_cast<CUdeviceptr *>(&d_bias),
                                  &d_out, &m, &n};
        engine.launch1d(engine.fn_bias_f32,
                        static_cast<std::int64_t>(rows) * out_dim,
                        arguments_list);
        cuMemFree(d_gemm);
        cuMemFree(d_weight);
        cuMemFree(d_bias);
        return d_out;
      };

      const auto run_silu = [&](CUdeviceptr input, std::int64_t count) {
        CUdeviceptr output{};
        check_cu(cuMemAlloc(&output, static_cast<std::size_t>(count) * 4U),
                 "alloc silu");
        std::int64_t n = count;
        void *arguments_list[] = {&input, &output, &n};
        engine.launch1d(engine.fn_silu, count, arguments_list);
        return output;
      };

      // temb = linear2(silu(linear1(sinusoid)))
      const auto d_hidden =
          run_linear_f32(d_sinusoid, freq_dim, proj_in_w, proj_in_b, embed_hidden);
      const auto d_hidden_act =
          run_silu(d_hidden, static_cast<std::int64_t>(rows) * embed_hidden);
      const auto d_temb = run_linear_f32(d_hidden_act, embed_hidden, proj_out_w,
                                         proj_out_b, time_embed_dim);
      // activated = bf16_rne(silu(temb))
      const auto d_temb_act =
          run_silu(d_temb, static_cast<std::int64_t>(rows) * time_embed_dim);
      CUdeviceptr d_activated{};
      check_cu(cuMemAlloc(&d_activated,
                          static_cast<std::size_t>(rows) * time_embed_dim * 2U),
               "alloc activated");
      {
        std::int64_t n = static_cast<std::int64_t>(rows) * time_embed_dim;
        void *arguments_list[] = {const_cast<CUdeviceptr *>(&d_temb_act),
                                  &d_activated, &n};
        engine.launch1d(engine.fn_cast, n, arguments_list);
      }

      const auto run_adaln = [&](const std::string &weight_name,
                                 const std::string &bias_name)
          -> std::vector<std::uint8_t> {
        const auto weight = load_checkpoint_tensor(index, shards, weight_name,
                                                   dif::ir::DType::BF16);
        const auto bias = load_checkpoint_tensor(index, shards, bias_name,
                                                 dif::ir::DType::BF16);
        const auto out_dim = static_cast<int>(weight.dims.at(0));
        if (weight.dims.at(1) != static_cast<std::uint64_t>(time_embed_dim) ||
            bias.dims.at(0) != weight.dims.at(0))
          dif::fail("adaln tensor geometry mismatch: " + weight_name);
        const auto d_weight = engine.upload(weight.data(), weight.byte_size());
        const auto d_bias = engine.upload(bias.data(), bias.byte_size());
        weight.discard_mapped_pages();
        CUdeviceptr d_gemm{}, d_out{};
        check_cu(cuMemAlloc(&d_gemm,
                            static_cast<std::size_t>(rows) * out_dim * 4U),
                 "alloc adaln gemm");
        check_cu(cuMemAlloc(&d_out,
                            static_cast<std::size_t>(rows) * out_dim * 2U),
                 "alloc adaln out");
        engine.cublas.gemm_rowmajor_bt(rows, out_dim, time_embed_dim,
                                       d_activated, /*bf16*/ 14, d_weight, 14,
                                       d_gemm);
        int m = rows, n = out_dim;
        void *arguments_list[] = {&d_gemm, const_cast<CUdeviceptr *>(&d_bias),
                                  &d_out, &m, &n};
        engine.launch1d(engine.fn_bias_bf16,
                        static_cast<std::int64_t>(rows) * out_dim,
                        arguments_list);
        std::vector<std::uint8_t> payload(
            static_cast<std::size_t>(rows) * out_dim * 2U);
        check_cu(cuMemcpyDtoH(payload.data(), d_out, payload.size()),
                 "cuMemcpyDtoH");
        cuMemFree(d_gemm);
        cuMemFree(d_out);
        cuMemFree(d_weight);
        cuMemFree(d_bias);
        return payload;
      };

      for (std::size_t block = 0; block < num_blocks; ++block) {
        const auto prefix = "blocks." + std::to_string(block) + ".adaln_proj.linear.";
        block_payloads[block] = run_adaln(prefix + "weight", prefix + "bias");
        if ((block + 1U) % 10U == 0U || block + 1U == num_blocks)
          std::cout << "  adaln block " << block + 1U << "/" << num_blocks
                    << "\n";
      }
      final_payload = run_adaln("final_layer.adaln_proj.linear.weight",
                                "final_layer.adaln_proj.linear.bias");
#else
      dif::fail("this build has no CUDA support; use --engine cpu");
#endif
    } else {
      // CPU diagnostic engine.
      const int half = freq_dim / 2;
      std::vector<float> sinusoid(static_cast<std::size_t>(rows) * freq_dim);
      for (int row = 0; row < rows; ++row)
        for (int i = 0; i < half; ++i) {
          const float freq =
              std::exp(neg_ln_mp * (static_cast<float>(i) / half));
          const float angle = timesteps[row] * freq;
          sinusoid[static_cast<std::size_t>(row) * freq_dim + i] = std::cos(angle);
          sinusoid[static_cast<std::size_t>(row) * freq_dim + half + i] =
              std::sin(angle);
        }
      auto hidden = cpu_linear_f32(
          sinusoid, rows, freq_dim,
          reinterpret_cast<const float *>(proj_in_w.data()),
          reinterpret_cast<const float *>(proj_in_b.data()), embed_hidden);
      for (auto &value : hidden)
        value = value / (1.0F + std::exp(-value));
      auto temb = cpu_linear_f32(
          hidden, rows, embed_hidden,
          reinterpret_cast<const float *>(proj_out_w.data()),
          reinterpret_cast<const float *>(proj_out_b.data()), time_embed_dim);
      std::vector<std::uint16_t> activated(temb.size());
      for (std::size_t i = 0; i < temb.size(); ++i)
        activated[i] = bf16_rne_host(temb[i] / (1.0F + std::exp(-temb[i])));

      const auto run_adaln_cpu = [&](const std::string &weight_name,
                                     const std::string &bias_name)
          -> std::vector<std::uint8_t> {
        const auto weight = load_checkpoint_tensor(index, shards, weight_name,
                                                   dif::ir::DType::BF16);
        const auto bias = load_checkpoint_tensor(index, shards, bias_name,
                                                 dif::ir::DType::BF16);
        const auto out_dim = static_cast<std::size_t>(weight.dims.at(0));
        const auto *weight_u16 =
            reinterpret_cast<const std::uint16_t *>(weight.data());
        const auto *bias_u16 =
            reinterpret_cast<const std::uint16_t *>(bias.data());
        std::vector<std::uint8_t> payload(
            static_cast<std::size_t>(rows) * out_dim * 2U);
        auto *out_u16 = reinterpret_cast<std::uint16_t *>(payload.data());
        for (int row = 0; row < rows; ++row)
          for (std::size_t column = 0; column < out_dim; ++column) {
            double accumulator = 0.0;
            const auto *w = weight_u16 + column * time_embed_dim;
            const auto *x = activated.data() +
                            static_cast<std::size_t>(row) * time_embed_dim;
            for (int inner = 0; inner < time_embed_dim; ++inner)
              accumulator += static_cast<double>(bf16_to_f32_host(x[inner])) *
                             bf16_to_f32_host(w[inner]);
            const float with_bias = static_cast<float>(accumulator) +
                                    bf16_to_f32_host(bias_u16[column]);
            out_u16[static_cast<std::size_t>(row) * out_dim + column] =
                bf16_rne_host(with_bias);
          }
        weight.discard_mapped_pages();
        return payload;
      };
      for (std::size_t block = 0; block < num_blocks; ++block) {
        const auto prefix = "blocks." + std::to_string(block) + ".adaln_proj.linear.";
        block_payloads[block] = run_adaln_cpu(prefix + "weight", prefix + "bias");
        std::cout << "  adaln block (cpu) " << block + 1U << "/" << num_blocks
                  << "\n";
      }
      final_payload = run_adaln_cpu("final_layer.adaln_proj.linear.weight",
                                    "final_layer.adaln_proj.linear.bias");
    }

    // ── emit the cache file (serenity writer format) ────────────────────────
    const auto source_string = arguments.source_index_string.empty()
                                   ? arguments.checkpoint_index.string()
                                   : arguments.source_index_string;
    struct stat source_stat {};
    if (::stat(source_string.c_str(), &source_stat) != 0)
      dif::fail("cannot stat modulation source index: " + source_string);

    std::vector<EmitTensor> tensors;
    tensors.push_back(meta_i64("__meta__.version", 1));
    tensors.push_back(meta_u8("__meta__.kind", "adaln-modulation"));
    tensors.push_back(meta_u8("__meta__.src_path", source_string));
    tensors.push_back(meta_i64("__meta__.src_size", source_stat.st_size));
    tensors.push_back(meta_i64("__meta__.src_mtime", source_stat.st_mtime));
    tensors.push_back(meta_i64("__meta__.steps", arguments.steps));
    tensors.push_back(meta_i64("__meta__.distinct_timesteps",
                               static_cast<std::int64_t>(distinct_timesteps)));
    tensors.push_back(
        meta_i64("__meta__.nblocks", static_cast<std::int64_t>(num_blocks)));
    const auto tags = 3U;
    const auto hidden_size = static_cast<std::uint64_t>(embed_hidden);
    for (std::size_t block = 0; block < num_blocks; ++block) {
      EmitTensor tensor;
      tensor.name = "block." + std::to_string(block);
      tensor.dtype = "BF16";
      tensor.shape = {static_cast<std::uint64_t>(distinct_timesteps) * tags,
                      6U * hidden_size};
      tensor.bytes = std::move(block_payloads[block]);
      tensors.push_back(std::move(tensor));
    }
    {
      EmitTensor tensor;
      tensor.name = "final";
      tensor.dtype = "BF16";
      tensor.shape = {static_cast<std::uint64_t>(distinct_timesteps),
                      2U * hidden_size};
      tensor.bytes = std::move(final_payload);
      tensors.push_back(std::move(tensor));
    }
    write_serenity_safetensors(tensors, arguments.output);
    std::cout << "MODCACHE_WRITTEN " << arguments.output.string() << " sha256="
              << dif::hex_digest(dif::sha256_file(arguments.output)) << "\n";

    // ── verification against a recorded cache ───────────────────────────────
    if (!arguments.verify_against.empty()) {
      const auto generated = dif::weights::read_safetensors(arguments.output);
      const auto recorded =
          dif::weights::read_safetensors(arguments.verify_against);
      std::uint64_t total_elements = 0, total_mismatched = 0;
      std::size_t mismatched_tensors = 0;
      const auto compare_tensor = [&](const std::string &name) {
        const auto ours = dif::weights::map_safetensor(generated, name);
        const auto theirs = dif::weights::map_safetensor(recorded, name);
        if (ours.dims != theirs.dims)
          dif::fail("verify: shape mismatch on " + name);
        const auto count = ours.byte_size() / 2U;
        const auto *a = reinterpret_cast<const std::uint16_t *>(ours.data());
        const auto *b = reinterpret_cast<const std::uint16_t *>(theirs.data());
        std::uint64_t mismatched = 0;
        for (std::uint64_t i = 0; i < count; ++i)
          mismatched += a[i] != b[i];
        total_elements += count;
        total_mismatched += mismatched;
        if (mismatched) {
          ++mismatched_tensors;
          std::cout << "  VERIFY " << name << ": " << mismatched << "/" << count
                    << " bf16 elements differ\n";
        }
      };
      for (std::size_t block = 0; block < num_blocks; ++block)
        compare_tensor("block." + std::to_string(block));
      compare_tensor("final");
      const auto generated_sha =
          dif::hex_digest(dif::sha256_file(arguments.output));
      const auto recorded_sha =
          dif::hex_digest(dif::sha256_file(arguments.verify_against));
      std::cout << "MODCACHE_VERIFY tensors=" << num_blocks + 1U
                << " mismatched_tensors=" << mismatched_tensors
                << " mismatched_elements=" << total_mismatched << "/"
                << total_elements << "\n";
      std::cout << "MODCACHE_VERIFY generated_sha=" << generated_sha << "\n";
      std::cout << "MODCACHE_VERIFY recorded_sha=" << recorded_sha << "\n";
      if (generated_sha == recorded_sha)
        std::cout << "MODCACHE_VERIFY BYTE-IDENTICAL FILES\n";
      else if (total_mismatched == 0)
        std::cout << "MODCACHE_VERIFY payloads identical, file bytes differ "
                     "(header/meta)\n";
      return total_mismatched == 0 && generated_sha == recorded_sha ? 0 : 3;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difmodcache: " << error.what() << "\n";
    return 1;
  }
}
