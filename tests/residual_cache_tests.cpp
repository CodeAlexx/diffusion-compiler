#include "dif/runtime/residual_cache.hpp"
#include "dif/runtime/scalar.hpp"
#include "../src/runtime/residual_cache_cuda.hpp"

#ifdef DIF_RESIDUAL_CACHE_NVRTC_TEST
#include <nvrtc.h>
#endif

#include <array>
#include <cfenv>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace dif::runtime;
int failures{};
std::size_t readback_bytes{}, launches{}, synchronizations{};
void expect(bool ok, const char *label) {
  if (!ok) { ++failures; std::cerr << "FAIL " << label << '\n'; }
}
template<class F> void rejects(F function, const char *label) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  expect(rejected, label);
}
template<class T> T *arg(void **args, std::size_t index) {
  return reinterpret_cast<T *>(*static_cast<CUdeviceptr *>(args[index]));
}
std::int8_t code(float value, float scale) {
  return static_cast<std::int8_t>(std::clamp(std::nearbyint(value / scale), -127.0F, 127.0F));
}
CUfunction kernel(std::uintptr_t value) { return reinterpret_cast<CUfunction>(value); }
// CPU oracle for the supplied CUDA dispatch, NOT an inference fallback. It
// executes the source formula and lets the real nonowning helper be tested
// without a CUDA context, GPU allocation, or model weights.
CUresult mock_launch(CUfunction function, unsigned, unsigned, unsigned,
                     unsigned, unsigned, unsigned, unsigned, CUstream,
                     void **args, void **) {
  ++launches;
  if (function == kernel(1)) {
    auto *hidden = arg<std::uint16_t>(args, 0);
    auto *rows = arg<std::int32_t>(args, 1);
    auto *out = arg<float>(args, 2);
    const auto width = *static_cast<std::uint64_t *>(args[3]);
    const auto count = *static_cast<std::uint64_t *>(args[4]);
    for (std::uint64_t i = 0; i < count; ++i)
      out[i] = bf16_to_float(hidden[rows[i / width] * width + i % width]);
  } else if (function == kernel(2) || function == kernel(3)) {
    const bool residual = function == kernel(3);
    auto *hidden = arg<std::uint16_t>(args, 0);
    auto *codes = arg<std::int8_t>(args, residual ? 3 : 1);
    auto *scales = arg<float>(args, residual ? 4 : 2);
    auto *front = residual ? arg<std::int8_t>(args, 1) : nullptr;
    auto *front_scales = residual ? arg<float>(args, 2) : nullptr;
    const auto groups = *static_cast<std::uint64_t *>(args[residual ? 5 : 3]);
    for (std::uint64_t g = 0; g < groups; ++g) {
      std::array<float, 32> values{};
      float maximum = 0;
      for (std::size_t j = 0; j < 32; ++j) {
        values[j] = bf16_to_float(hidden[g * 32 + j]);
        if (residual) {
          const volatile float dequantized = static_cast<float>(front[g * 32 + j]) * front_scales[g];
          values[j] = bf16_to_float(float_to_bf16(values[j] - dequantized));
        }
        maximum = std::max(maximum, std::abs(values[j]));
      }
      scales[g] = std::max(maximum / 127.0F, 1e-30F);
      for (std::size_t j = 0; j < 32; ++j) codes[g * 32 + j] = code(values[j], scales[g]);
    }
  } else if (function == kernel(4)) {
    auto *front = arg<std::uint16_t>(args, 0);
    auto *codes = arg<std::int8_t>(args, 1);
    auto *scales = arg<float>(args, 2);
    auto *out = arg<std::uint16_t>(args, 3);
    const auto n = *static_cast<std::uint64_t *>(args[4]);
    for (std::uint64_t i = 0; i < n; ++i) {
      const volatile float delta = static_cast<float>(codes[i]) * scales[i / 32];
      out[i] = float_to_bf16(bf16_to_float(front[i]) + delta);
    }
  } else return CUDA_ERROR_INVALID_VALUE;
  return CUDA_SUCCESS;
}
CUresult mock_htod(CUdeviceptr dest, const void *source, std::size_t bytes, CUstream) {
  std::memcpy(reinterpret_cast<void *>(dest), source, bytes); return CUDA_SUCCESS;
}
CUresult mock_dtoh(void *dest, CUdeviceptr source, std::size_t bytes, CUstream) {
  readback_bytes += bytes;
  std::memcpy(dest, reinterpret_cast<void *>(source), bytes); return CUDA_SUCCESS;
}
CUresult mock_sync(CUstream) { ++synchronizations; return CUDA_SUCCESS; }
ResidualCacheRegion region() {
  return {100, 200, 400, 10, 20, 30, 65, 64, 32, {0, 32, 64}, {9, 17}};
}
void helper_gate() {
  std::fesetround(FE_TONEAREST);
  auto r = region();
  const auto layout = residual_cache_device_layout(r);
  expect(layout.elements == 4160 && layout.groups == 130 &&
             layout.main_probe_elements == 192 && layout.audio_probe_elements == 128,
         "explicit geometry and separate probe-band lengths");
  expect(layout.required_bytes % 256 == 0 && layout.residual_scales + 130 * 4 <= layout.required_bytes,
         "complete group32 scratch accounted inside aligned shared arena");
  auto bad = r; bad.group_size = 128;
  rejects([&] { residual_cache_device_layout(bad); }, "source group32 is not silently replaced by docstring group128");
  bad = r; bad.audio_probe_rows = {-1};
  rejects([&] { residual_cache_device_layout(bad); }, "invalid modality row rejected");
  bad = r; bad.sequence = std::numeric_limits<std::uint64_t>::max();
  rejects([&] { residual_cache_device_layout(bad); }, "workspace overflow rejected");

  std::vector<std::uint8_t> storage(layout.required_bytes);
  std::vector<std::uint16_t> before(layout.elements), front(layout.elements), middle(layout.elements), output(layout.elements);
  for (std::uint64_t i = 0; i < layout.elements; ++i) {
    before[i] = float_to_bf16(static_cast<float>(i % 32));
    const std::array values{127.0F, 0.5F, 1.5F, 2.5F, -0.5F, -1.5F};
    const auto v = i % 32 < values.size() ? values[i % 32] : 0.0F;
    front[i] = float_to_bf16(v);
    // Group0: residual has exact scale1 and half-even ties. Other groups
    // also exercise the full packed sequence, including selected audio rows.
    const float f = static_cast<float>(code(v, 1.0F));
    middle[i] = float_to_bf16(f + v);
  }
  // A zero group exercises the source1e-30 scale floor, not epsilon1e-8.
  std::fill(front.end() - 32, front.end(), 0);
  std::fill(middle.end() - 32, middle.end(), 0);
  auto callbacks = std::make_shared<ResidualCacheCallbacks>();
  bool active = true;
  ResidualCacheAction next = ResidualCacheAction::Refresh;
  std::size_t decisions = 0, refreshed = 0;
  callbacks->needs_probes = [&] { return active; };
  callbacks->decide = [&](std::span<const float> b, std::span<const float> a,
                          std::span<const float> ab, std::span<const float> aa) {
    ++decisions;
    if (active) {
      expect(b.size() == 192 && a.size() == 192 && ab.size() == 128 && aa.size() == 128,
             "callbacks see independent whole-sequence and audio probes only");
      expect(b[0] == 0 && b[64 + 1] == 1 && ab[1] == 1 && a[0] == 127 && aa[0] == 127,
             "probes preserve BF16-to-F32 values and selected packed row order");
    } else expect(b.empty() && a.empty() && ab.empty() && aa.empty(), "disabled tail callback advances without readbacks");
    return next;
  };
  callbacks->refreshed = [&] { ++refreshed; };
  const ResidualCacheCudaDispatch dispatch{mock_launch, mock_htod, mock_dtoh, mock_sync};
  CudaResidualCache cache(r, reinterpret_cast<CUdeviceptr>(storage.data()), nullptr,
      {kernel(1),kernel(2),kernel(3),kernel(4)}, dispatch, callbacks);
  const auto before_ptr = reinterpret_cast<CUdeviceptr>(before.data());
  const auto front_ptr = reinterpret_cast<CUdeviceptr>(front.data());
  const auto middle_ptr = reinterpret_cast<CUdeviceptr>(middle.data());
  const auto output_ptr = reinterpret_cast<CUdeviceptr>(output.data());
  cache.before_front(before_ptr);
  expect(!cache.before_middle(front_ptr, output_ptr) && !cache.residual_ready(), "refresh snapshots front without bypassing middle");
  cache.before_back(middle_ptr);
  expect(cache.residual_ready() && refreshed == 1, "residual ready only after source middle store");
  const auto *front_q = reinterpret_cast<const std::int8_t *>(storage.data() + layout.front_codes);
  const auto *residual_q = reinterpret_cast<const std::int8_t *>(storage.data() + layout.residual_codes);
  const auto *scales = reinterpret_cast<const float *>(storage.data() + layout.residual_scales);
  expect(front_q[0] == 127 && front_q[1] == 0 && front_q[2] == 2 && front_q[3] == 2 &&
             front_q[4] == 0 && front_q[5] == -2, "snapshot signed INT8 round-to-nearest-even golden bytes");
  expect(residual_q[0] == 127 && residual_q[1] == 0 && residual_q[2] == 2 && residual_q[5] == -2 &&
             scales[0] == 1.0F && scales[layout.groups - 1] == 1e-30F,
         "BF16 residual boundary and source scale floor retained");
  next = ResidualCacheAction::Reuse;
  cache.before_front(before_ptr);
  expect(cache.before_middle(front_ptr, output_ptr), "reuse bypasses only middle region");
  cache.before_back(output_ptr);
  expect(refreshed == 1 && output[0] == float_to_bf16(254.0F) &&
             output[9 * 64] == float_to_bf16(254.0F), "residual applied to all mixed rows including audio; no audio mask");
  expect(readback_bytes == 2 * 2 * (192 + 128) * 4 && synchronizations == 2,
         "only sampled rows cross host boundary, one probe fence per decision");
  const auto previous_launches = launches;
  active = false; next = ResidualCacheAction::Exact;
  cache.before_front(before_ptr); cache.before_middle(front_ptr, output_ptr); cache.before_back(middle_ptr);
  expect(launches == previous_launches && !cache.residual_ready() && decisions == 3,
         "exact tail discards cached validity and does no probe/quantize/apply work");
  active = true; next = ResidualCacheAction::Reuse;
  cache.before_front(before_ptr);
  rejects([&] { cache.before_middle(front_ptr, output_ptr); }, "stale residual cannot cross invalidation boundary");
}
#ifdef DIF_RESIDUAL_CACHE_NVRTC_TEST
void compile_gate() {
  const auto source = residual_cache_cuda_source();
  nvrtcProgram program{};
  expect(nvrtcCreateProgram(&program, source.data(), "residual-cache.cu", 0, nullptr, nullptr) == NVRTC_SUCCESS,
         "NVRTC compile-only program created without a GPU context");
  const char *options[]{"--gpu-architecture=compute_86", "--std=c++17", "--fmad=false"};
  const auto status = nvrtcCompileProgram(program, 3, options);
  if (status != NVRTC_SUCCESS) {
    std::size_t bytes{}; nvrtcGetProgramLogSize(program, &bytes);
    std::string log(bytes, '\0'); nvrtcGetProgramLog(program, log.data()); std::cerr << log;
  }
  expect(status == NVRTC_SUCCESS, "actual residual CUDA source compiles to PTX, no device execution");
  std::size_t bytes{};
  expect(nvrtcGetPTXSize(program, &bytes) == NVRTC_SUCCESS && bytes > 0,
         "residual primitive has generated PTX");
  nvrtcDestroyProgram(&program);
}
#endif
} // namespace
int main() {
  try {
    helper_gate();
#ifdef DIF_RESIDUAL_CACHE_NVRTC_TEST
    compile_gate();
#endif
    if (failures) return 1;
    std::cout << "RESIDUAL_CACHE_CPU PASS dispatch=real-helper-with-CPU-oracle group=32"
              << " signed_INT8=RNE BF16_residual_boundary=preserved tiny_probes=only-readback"
#ifdef DIF_RESIDUAL_CACHE_NVRTC_TEST
              << " NVRTC=compile-only-pass"
#endif
              << " GPU_execution=not-run decoded_gate=not-run\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL exception " << error.what() << '\n'; return 1;
  }
}
