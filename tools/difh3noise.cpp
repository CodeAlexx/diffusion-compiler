// difh3noise — source-faithful H3 sampler-noise generation.
//
// Reproduces the Serenity sampler's deterministic initial-noise stream
// byte-exactly: Rust rand 0.8.5 StdRng semantics (seed_from_u64 -> PCG32-
// expanded 32-byte key -> ChaCha12 keystream -> Standard f32 pairs) followed
// by Box-Muller. The Box-Muller transcendentals run with the same GPU
// lowerings Mojo uses (sqrt.approx.ftz, lg2.approx.ftz * ln2_f32,
// cos/sin.approx.ftz), so the recorded H3 initial-state payloads (video seed
// 4242, audio seed 4243) regenerate bit-for-bit.
//
// The ChaCha key k0..k7 depends only on the seed; it is computed on the host
// (exact u64/u32 integer math) and passed to the kernel.

#include "dif/runtime/tensor.hpp"
#include "dif/frontend/h3_latents.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/support/torch_cpu_rng.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#if DIF_HAS_CUDA
#include <cuda.h>
#include <nvrtc.h>
#endif

namespace {

struct Arguments {
  long long rows = -1;
  long long cols = -1;
  unsigned long long seed = 0;
  unsigned long long skip_normal_values = 0;
  bool seed_set = false;
  std::string rng{"serenity"};
  std::string layout{"flat"};
  long long latent_frames = -1;
  long long latent_height = -1;
  long long latent_width = -1;
  long long audio_latents = -1;
  std::filesystem::path output;
  std::filesystem::path verify_against;
};

[[noreturn]] void usage_error(const std::string &message) {
  std::cerr << "difh3noise: " << message << "\n"
            << "usage: difh3noise --seed U64 "
               "--output FILE.diftensor [--rng serenity|torch-cpu] "
               "[--layout flat|h3-video|h3-audio] "
               "[--rows N --cols N] "
               "[--latent-frames N --latent-height N --latent-width N] "
               "[--audio-latents N] "
               "[--skip-normal-values N] "
               "[--verify-against FILE.diftensor]\n";
  std::exit(2);
}

std::uint32_t pcg32_word(std::uint64_t &state) {
  state = state * 6364136223846793005ULL + 11634580027462260723ULL;
  const auto xorshifted =
      static_cast<std::uint32_t>(((state >> 18) ^ state) >> 27);
  const auto rotation = static_cast<int>((state >> 59) & 31U);
  if (rotation == 0)
    return xorshifted;
  return (xorshifted >> rotation) | (xorshifted << (32 - rotation));
}

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
typedef unsigned int u32;
typedef unsigned long long u64;
typedef long long i64;

__device__ __forceinline__ u32 rotl32(u32 x, int n) {
  return (x << n) | (x >> (32 - n));
}

__device__ void quarter(u32 &a, u32 &b, u32 &c, u32 &d) {
  a += b; d = rotl32(d ^ a, 16);
  c += d; b = rotl32(b ^ c, 12);
  a += b; d = rotl32(d ^ a, 8);
  c += d; b = rotl32(b ^ c, 7);
}

__device__ u32 chacha12_word(const u32 *k, u64 block, int offset) {
  u32 s[16] = {0x61707865u, 0x3320646Eu, 0x79622D32u, 0x6B206574u,
               k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7],
               (u32)(block & 0xFFFFFFFFu), (u32)(block >> 32), 0u, 0u};
  u32 x[16];
  for (int i = 0; i < 16; ++i) x[i] = s[i];
  for (int round = 0; round < 6; ++round) {
    quarter(x[0], x[4], x[8], x[12]);
    quarter(x[1], x[5], x[9], x[13]);
    quarter(x[2], x[6], x[10], x[14]);
    quarter(x[3], x[7], x[11], x[15]);
    quarter(x[0], x[5], x[10], x[15]);
    quarter(x[1], x[6], x[11], x[12]);
    quarter(x[2], x[7], x[8], x[13]);
    quarter(x[3], x[4], x[9], x[14]);
  }
  return x[offset] + s[offset];
}

__device__ __forceinline__ float sqrt_ap(float v) {
  float r; asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float lg2_ap(float v) {
  float r; asm("lg2.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float cos_ap(float v) {
  float r; asm("cos.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
__device__ __forceinline__ float sin_ap(float v) {
  float r; asm("sin.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(v)); return r;
}
// Mojo std.math.log on NVIDIA f32: ln2_f32 * lg2.approx.ftz(x)
__device__ __forceinline__ float log_mojo(float v) {
  return __uint_as_float(0x3F317218u) * lg2_ap(v);
}

extern "C" __global__ void randn_f32(float *o, i64 n, u32 k0, u32 k1, u32 k2,
                                     u32 k3, u32 k4, u32 k5, u32 k6, u32 k7) {
  i64 pair = (i64)blockIdx.x * blockDim.x + threadIdx.x;
  i64 i = pair * 2;
  if (i >= n)
    return;
  u32 key[8] = {k0, k1, k2, k3, k4, k5, k6, k7};
  u64 word_pos = (u64)pair * 2u;
  u64 block = word_pos / 16u;
  int offset = (int)(word_pos % 16u);
  u32 w0 = chacha12_word(key, block, offset);
  u32 w1 = chacha12_word(key, block, offset + 1);
  // Standard f32: (word >> 8) / 2^24
  float u1 = (float)(w0 >> 8) * 5.9604644775390625e-8f;
  float u2 = (float)(w1 >> 8) * 5.9604644775390625e-8f;
  if (u1 < 1.0e-10f)
    u1 = 1.0e-10f;
  float r = sqrt_ap(-2.0f * log_mojo(u1));
  float theta = 6.2831853071795864769f * u2;  // Float32(TWO_PI)
  o[i] = r * cos_ap(theta);
  if (i + 1 < n)
    o[i + 1] = r * sin_ap(theta);
}
)CUDA";

std::vector<float> generate_cuda(std::int64_t count, std::uint64_t seed) {
  check_cu(cuInit(0), "cuInit");
  CUdevice device{};
  check_cu(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context{};
  check_cu(cuDevicePrimaryCtxRetain(&context, device), "primary ctx");
  check_cu(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  int major = 0, minor = 0;
  check_cu(cuDeviceGetAttribute(&major,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                                device),
           "cc");
  check_cu(cuDeviceGetAttribute(&minor,
                                CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                                device),
           "cc");
  const auto arch = "--gpu-architecture=compute_" + std::to_string(major) +
                    std::to_string(minor);
  nvrtcProgram program{};
  check_nvrtc(nvrtcCreateProgram(&program, kKernelSource, "difh3noise.cu", 0,
                                 nullptr, nullptr),
              "nvrtcCreateProgram");
  const char *options[] = {arch.c_str()};
  if (nvrtcCompileProgram(program, 1, options) != NVRTC_SUCCESS) {
    std::size_t log_size = 0;
    nvrtcGetProgramLogSize(program, &log_size);
    std::string log(log_size, '\0');
    nvrtcGetProgramLog(program, log.data());
    dif::fail("NVRTC compile failed: " + log);
  }
  std::size_t ptx_size = 0;
  check_nvrtc(nvrtcGetPTXSize(program, &ptx_size), "ptx size");
  std::string ptx(ptx_size, '\0');
  check_nvrtc(nvrtcGetPTX(program, ptx.data()), "ptx");
  nvrtcDestroyProgram(&program);
  CUmodule module{};
  check_cu(cuModuleLoadDataEx(&module, ptx.c_str(), 0, nullptr, nullptr),
           "module load");
  CUfunction function{};
  check_cu(cuModuleGetFunction(&function, module, "randn_f32"), "function");

  // ChaCha key from PCG32-expanded seed — host-side exact integer math.
  std::uint64_t state = seed;
  std::uint32_t key[8];
  for (auto &word : key)
    word = pcg32_word(state);

  CUdeviceptr d_output{};
  check_cu(cuMemAlloc(&d_output, static_cast<std::size_t>(count) * 4U),
           "cuMemAlloc");
  std::int64_t n = count;
  void *arguments[] = {&d_output, &n,      &key[0], &key[1], &key[2],
                       &key[3],   &key[4], &key[5], &key[6], &key[7]};
  const std::int64_t pairs = (count + 1) / 2;
  const unsigned block = 256;
  const auto grid = static_cast<unsigned>((pairs + block - 1) / block);
  check_cu(cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, nullptr,
                          arguments, nullptr),
           "launch");
  std::vector<float> host(static_cast<std::size_t>(count));
  check_cu(cuMemcpyDtoH(host.data(), d_output,
                        static_cast<std::size_t>(count) * 4U),
           "copy back");
  cuMemFree(d_output);
  return host;
}

#endif // DIF_HAS_CUDA

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
      if (option == "--rows")
        arguments.rows = std::stoll(value("--rows"));
      else if (option == "--cols")
        arguments.cols = std::stoll(value("--cols"));
      else if (option == "--seed") {
        arguments.seed = std::stoull(value("--seed"));
        arguments.seed_set = true;
      } else if (option == "--output")
        arguments.output = value("--output");
      else if (option == "--rng")
        arguments.rng = value("--rng");
      else if (option == "--layout")
        arguments.layout = value("--layout");
      else if (option == "--latent-frames")
        arguments.latent_frames = std::stoll(value("--latent-frames"));
      else if (option == "--latent-height")
        arguments.latent_height = std::stoll(value("--latent-height"));
      else if (option == "--latent-width")
        arguments.latent_width = std::stoll(value("--latent-width"));
      else if (option == "--audio-latents")
        arguments.audio_latents = std::stoll(value("--audio-latents"));
      else if (option == "--skip-normal-values")
        arguments.skip_normal_values =
            std::stoull(value("--skip-normal-values"));
      else if (option == "--verify-against")
        arguments.verify_against = value("--verify-against");
      else
        usage_error("unknown option " + option);
    }
    if (!arguments.seed_set || arguments.output.empty())
      usage_error("--seed and --output are required");
    if (arguments.rng != "serenity" && arguments.rng != "torch-cpu")
      usage_error("--rng must be serenity or torch-cpu");
    if (arguments.layout != "flat" && arguments.layout != "h3-video" &&
        arguments.layout != "h3-audio")
      usage_error("--layout must be flat, h3-video, or h3-audio");
    if (arguments.layout == "flat" &&
        (arguments.rows <= 0 || arguments.cols <= 0))
      usage_error("flat layout requires --rows and --cols");
    if (arguments.layout == "h3-video" &&
        (arguments.latent_frames <= 0 || arguments.latent_height <= 0 ||
         arguments.latent_width <= 0 || arguments.latent_height % 2 != 0 ||
         arguments.latent_width % 2 != 0))
      usage_error("h3-video requires positive T and even H/W latent geometry");
    if (arguments.layout == "h3-audio" && arguments.audio_latents <= 0)
      usage_error("h3-audio requires --audio-latents");
    if (arguments.rng == "serenity" && arguments.skip_normal_values != 0U)
      usage_error("--skip-normal-values is only valid with --rng torch-cpu");
    if (arguments.rng == "serenity" && arguments.layout != "flat")
      usage_error("H3 source packing is only valid with --rng torch-cpu");

    long long count = arguments.rows * arguments.cols;
    if (arguments.layout == "h3-video")
      count = 24LL * arguments.latent_frames * arguments.latent_height *
              arguments.latent_width;
    else if (arguments.layout == "h3-audio")
      count = 32LL * 2LL * arguments.audio_latents;
    std::vector<float> values;
    if (arguments.rng == "torch-cpu") {
      dif::TorchCpuMt19937 generator(arguments.seed);
      if (arguments.skip_normal_values != 0U)
        (void)dif::torch_cpu_normal(
            generator,
            static_cast<std::size_t>(arguments.skip_normal_values));
      values =
          dif::torch_cpu_normal(generator, static_cast<std::size_t>(count));
    } else {
#if DIF_HAS_CUDA
      values = generate_cuda(count, arguments.seed);
#else
      dif::fail("Serenity noise requires a CUDA build");
#endif
    }

    dif::runtime::Tensor tensor;
    if (arguments.layout == "h3-video") {
      dif::runtime::Tensor raw{
          dif::ir::DType::F32,
          {1U, 24U, static_cast<std::uint64_t>(arguments.latent_frames),
           static_cast<std::uint64_t>(arguments.latent_height),
           static_cast<std::uint64_t>(arguments.latent_width)},
          {}};
      raw.bytes.resize(static_cast<std::size_t>(count) * sizeof(float));
      std::memcpy(raw.bytes.data(), values.data(), raw.bytes.size());
      tensor = dif::frontend::pack_h3_video_latent(raw);
    } else if (arguments.layout == "h3-audio") {
      dif::runtime::Tensor raw{
          dif::ir::DType::F32,
          {1U, 32U, 2U,
           static_cast<std::uint64_t>(arguments.audio_latents)},
          {}};
      raw.bytes.resize(static_cast<std::size_t>(count) * sizeof(float));
      std::memcpy(raw.bytes.data(), values.data(), raw.bytes.size());
      tensor = dif::frontend::pack_h3_audio_latent(raw);
    } else {
      tensor = dif::runtime::Tensor{
          dif::ir::DType::F32,
          {static_cast<std::uint64_t>(arguments.rows),
           static_cast<std::uint64_t>(arguments.cols)},
          {}};
      tensor.bytes.resize(static_cast<std::size_t>(count) * sizeof(float));
      std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    }
    dif::runtime::write_tensor(tensor, arguments.output);
    const auto payload_sha = dif::hex_digest(
        dif::sha256({tensor.bytes.data(), tensor.bytes.size()}));
    std::cout << "NOISE_WRITTEN " << arguments.output.string()
              << " rows=" << tensor.dims[0] << " cols=" << tensor.dims[1]
              << " seed=" << arguments.seed << " rng=" << arguments.rng
              << " layout=" << arguments.layout
              << " skip_normal_values=" << arguments.skip_normal_values
              << " payload_sha256=" << payload_sha
              << "\n";

    if (!arguments.verify_against.empty()) {
      const auto recorded = dif::runtime::read_tensor(arguments.verify_against);
      if (recorded.dims != tensor.dims || recorded.dtype != tensor.dtype)
        dif::fail("verify: recorded tensor geometry differs");
      std::uint64_t mismatched = 0;
      const auto *ours =
          reinterpret_cast<const std::uint32_t *>(tensor.bytes.data());
      const auto *theirs =
          reinterpret_cast<const std::uint32_t *>(recorded.data());
      for (std::uint64_t i = 0; i < tensor.element_count(); ++i)
        mismatched += ours[i] != theirs[i];
      const auto recorded_payload_sha = dif::hex_digest(dif::sha256(
          {recorded.data(), static_cast<std::size_t>(recorded.byte_size())}));
      std::cout << "NOISE_VERIFY mismatched_elements=" << mismatched << "/"
                << tensor.element_count() << "\n";
      std::cout << "NOISE_VERIFY recorded_payload_sha256="
                << recorded_payload_sha << "\n";
      if (mismatched == 0 && payload_sha == recorded_payload_sha)
        std::cout << "NOISE_VERIFY BYTE-IDENTICAL PAYLOAD\n";
      return mismatched == 0 && payload_sha == recorded_payload_sha ? 0 : 3;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difh3noise: " << error.what() << "\n";
    return 1;
  }
}
