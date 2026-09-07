#include "dif/runtime/residual_cache.hpp"
#include "dif/support/error.hpp"

#include <limits>

namespace dif::runtime {
ResidualCacheDeviceLayout residual_cache_device_layout(const ResidualCacheRegion &r) {
  constexpr auto max = std::numeric_limits<std::uint64_t>::max();
  if (r.sequence == 0 || r.hidden == 0 || r.group_size != 32 ||
      r.hidden % r.group_size != 0 || r.sequence > max / r.hidden ||
      r.probe_rows.empty() || r.probe_rows.size() > 16 || r.audio_probe_rows.size() > 16 ||
      r.front_first_operation == 0 || r.front_first_operation >= r.middle_first_operation ||
      r.middle_first_operation >= r.back_first_operation ||
      r.before_front_tensor == 0 || r.after_front_tensor == 0 || r.after_middle_tensor == 0)
    fail("invalid explicit residual-cache region or group32 geometry");
  for (const auto *rows : {&r.probe_rows, &r.audio_probe_rows})
    for (const auto row : *rows)
      if (row < 0 || static_cast<std::uint64_t>(row) >= r.sequence)
        fail("residual-cache probe row out of bounds");
  ResidualCacheDeviceLayout d;
  d.elements = r.sequence * r.hidden;
  d.groups = d.elements / r.group_size;
  if (d.elements > max / 4 || r.hidden > max / 128)
    fail("residual-cache device workspace byte count overflow");
  d.main_probe_elements = r.probe_rows.size() * r.hidden;
  d.audio_probe_elements = r.audio_probe_rows.size() * r.hidden;
  const auto take = [&](std::uint64_t bytes, std::uint64_t &offset) {
    if (d.required_bytes > max - 255 || bytes > max - d.required_bytes - 255)
      fail("residual-cache arena size overflow");
    offset = (d.required_bytes + 255) & ~std::uint64_t{255};
    d.required_bytes = offset + bytes;
  };
  take((r.probe_rows.size() + r.audio_probe_rows.size()) * 4, d.indices);
  const auto probe_bytes = (d.main_probe_elements + d.audio_probe_elements) * 4;
  take(probe_bytes, d.before_probe); take(probe_bytes, d.after_probe);
  take(d.elements, d.front_codes); take(d.groups * 4, d.front_scales);
  take(d.elements, d.residual_codes); take(d.groups * 4, d.residual_scales);
  if (d.required_bytes > max - 255) fail("residual-cache final arena alignment overflow");
  d.required_bytes = (d.required_bytes + 255) & ~std::uint64_t{255};
  return d;
}

std::string_view residual_cache_cuda_source() {
  return R"CUDA(
// Generic compressed residual cache. BF16 boundaries are explicit and F32
// arithmetic is noncontracted. All kernels run on the executor's stream.
__device__ __forceinline__ float dif_rc_load(unsigned short bits) {
  return __uint_as_float((unsigned int)bits << 16);
}
__device__ __forceinline__ unsigned short dif_rc_store(float value) {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7f800000U) == 0x7f800000U && (bits & 0x007fffffU))
    return (unsigned short)((bits >> 16) | 0x0040U);
  return (unsigned short)((bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16);
}
extern "C" __global__ void dif_residual_cache_gather(
    const unsigned short *hidden, const int *rows, float *probe,
    unsigned long long width, unsigned long long count) {
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < count; i += (unsigned long long)blockDim.x * gridDim.x)
    probe[i] = dif_rc_load(hidden[(unsigned long long)rows[i / width] * width + i % width]);
}
extern "C" __global__ void dif_residual_cache_quantize(
    const unsigned short *hidden, signed char *codes, float *scales,
    unsigned long long groups) {
  for (unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       g < groups; g += (unsigned long long)blockDim.x * gridDim.x) {
    float maximum = 0.0f;
    for (unsigned int j = 0; j < 32; ++j)
      maximum = fmaxf(maximum, fabsf(dif_rc_load(hidden[g * 32 + j])));
    float scale = fmaxf(__fdiv_rn(maximum, 127.0f), 1.0e-30f);
    scales[g] = scale;
    for (unsigned int j = 0; j < 32; ++j) {
      int q = __float2int_rn(__fdiv_rn(dif_rc_load(hidden[g * 32 + j]), scale));
      q = q > 127 ? 127 : q < -127 ? -127 : q;
      codes[g * 32 + j] = (signed char)q;
    }
  }
}
extern "C" __global__ void dif_residual_cache_store(
    const unsigned short *middle, const signed char *front_codes,
    const float *front_scales, signed char *codes, float *scales,
    unsigned long long groups) {
  for (unsigned long long g = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       g < groups; g += (unsigned long long)blockDim.x * gridDim.x) {
    float residual[32];
    float maximum = 0.0f;
    for (unsigned int j = 0; j < 32; ++j) {
      float front = __fmul_rn((float)front_codes[g * 32 + j], front_scales[g]);
      residual[j] = dif_rc_load(dif_rc_store(__fsub_rn(dif_rc_load(middle[g * 32 + j]), front)));
      maximum = fmaxf(maximum, fabsf(residual[j]));
    }
    float scale = fmaxf(__fdiv_rn(maximum, 127.0f), 1.0e-30f);
    scales[g] = scale;
    for (unsigned int j = 0; j < 32; ++j) {
      int q = __float2int_rn(__fdiv_rn(residual[j], scale));
      q = q > 127 ? 127 : q < -127 ? -127 : q;
      codes[g * 32 + j] = (signed char)q;
    }
  }
}
extern "C" __global__ void dif_residual_cache_apply(
    const unsigned short *front, const signed char *codes, const float *scales,
    unsigned short *output, unsigned long long elements) {
  for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
       i < elements; i += (unsigned long long)blockDim.x * gridDim.x)
    output[i] = dif_rc_store(__fadd_rn(dif_rc_load(front[i]),
        __fmul_rn((float)codes[i], scales[i / 32])));
}
)CUDA";
}
} // namespace dif::runtime
