// fastload_test.cpp -- gate for the assembly kernels in fastload_x86.S.
//
// Every kernel is compared against a scalar C reference over inputs chosen
// to hit the rounding edges: exact ties, ties at the mantissa top (carry into
// the exponent), max finite, infinities, signalling and quiet NaNs, denormals,
// negative zero, plus a few million random bit patterns. Lengths sweep the
// vector/tail split (0..40) and a large odd length.
//
// Build + run (no CUDA needed):
//   g++ -O2 -std=c++17 fastload_test.cpp fastload_x86.S -o /tmp/fastload_test && /tmp/fastload_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "dif/runtime/fastload.h"

static uint16_t ref_f32_to_bf16(uint32_t bits) {
  if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return 0x7FC0;  // NaN -> canonical qNaN
  const uint32_t lsb = (bits >> 16) & 1u;
  return static_cast<uint16_t>((bits + 0x7FFFu + lsb) >> 16);
}

// IEEE half -> float bits, exact (the same mapping vcvtph2ps performs).
static uint32_t ref_f16_to_f32_bits(uint16_t h) {
  const uint32_t sign = (h >> 15) & 1u, exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
  if (exp == 0) {
    if (man == 0) return sign << 31;
    // denormal: normalize
    int e = -1;
    uint32_t m = man;
    do { ++e; m <<= 1; } while ((m & 0x400u) == 0);
    return (sign << 31) | ((127u - 15u - e) << 23) | ((m & 0x3FFu) << 13);
  }
  if (exp == 31) return (sign << 31) | 0x7F800000u | (man << 13);
  return (sign << 31) | ((exp + 127u - 15u) << 23) | (man << 13);
}

static int failures = 0;
static void expect(bool ok, const char *what, uint64_t detail) {
  if (!ok) {
    ++failures;
    if (failures <= 20) std::printf("FAIL %s (0x%llx)\n", what, (unsigned long long)detail);
  }
}

int main() {
  if (!serenity_fastload_cpu_supported()) {
    std::printf("SKIP: fastload requires AVX2 and F16C\n");
    return 77;
  }
  // Runtime admission is default-off until the loader is production-admitted.
  // No subordinate knob or malformed global value may silently enable it.
  unsetenv("DIF_FASTLOAD");
  unsetenv("DIF_FASTLOAD_STREAMED");
  expect(serenity_fastload_runtime_enabled() == 0,
         "fastload must default off", 0);
  expect(serenity_fastload_streamed_runtime_enabled() == 0,
         "streamed fastload must default off", 0);
  setenv("DIF_FASTLOAD_STREAMED", "1", 1);
  setenv("DIF_FASTLOAD_PREFETCH", "1", 1);
  expect(serenity_fastload_runtime_enabled() == 0,
         "subordinate knobs changed resident default", 0);
  expect(serenity_fastload_streamed_runtime_enabled() == 0,
         "subordinate knobs enabled streamed fastload without global opt-in", 0);
  for (const char *value : {"", "0", "true", "false", "2", "01", "1 "}) {
    setenv("DIF_FASTLOAD", value, 1);
    expect(serenity_fastload_runtime_enabled() == 0,
           "only exact global opt-in may enable resident fastload", 0);
    expect(serenity_fastload_streamed_runtime_enabled() == 0,
           "only exact global opt-in may enable streamed fastload", 0);
  }
  unsetenv("DIF_FASTLOAD_STREAMED");
  unsetenv("DIF_FASTLOAD_PREFETCH");
  setenv("DIF_FASTLOAD", "1", 1);
  expect(serenity_fastload_runtime_enabled() == 1,
         "explicit fastload enable ignored", 0);
  expect(serenity_fastload_streamed_runtime_enabled() == 1,
         "explicit enable did not admit streamed fastload", 0);
  setenv("DIF_FASTLOAD_STREAMED", "0", 1);
  expect(serenity_fastload_runtime_enabled() == 1,
         "streamed opt-out disabled resident opt-in", 0);
  expect(serenity_fastload_streamed_runtime_enabled() == 0,
         "streamed fastload opt-out ignored", 0);
  unsetenv("DIF_FASTLOAD_STREAMED");
  unsetenv("DIF_FASTLOAD");
  std::printf("fastload runtime policy: default-off + explicit opt-in checked\n");
  std::mt19937_64 rng(20260905);
  std::vector<uint32_t> edge = {
      0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,  // 0, -0, 1, -1
      0x3F808000u,  // 1 + 2^-8 exactly: tie, rounds to even (down)
      0x3F818000u,  // tie that rounds up (odd kept mantissa)
      0x3F7FFFFFu,  // just below 1: rounds up to 1.0
      0x7F7FFFFFu,  // max finite: rounds up to +inf in bf16
      0xFF7FFFFFu, 0x7F800000u, 0xFF800000u,  // -max, +inf, -inf
      0x7FC00000u, 0xFFC00000u, 0x7F800001u, 0x7FBFFFFFu,  // qNaN, -qNaN, sNaNs
      0x00000001u, 0x00400000u, 0x007FFFFFu, 0x80000001u,  // denormals
      0x00008000u, 0x00018000u,  // denormal ties
      0x4B7FFFFFu, 0x3EAAAAABu, 0x40490FDBu,
  };
  // f32 -> bf16: edges then random, at every length 0..40 and one long odd length.
  {
    std::vector<uint32_t> src_bits(edge);
    for (int i = 0; i < 4000000; ++i) src_bits.push_back(static_cast<uint32_t>(rng()));
    std::vector<float> src(src_bits.size());
    std::memcpy(src.data(), src_bits.data(), src_bits.size() * 4);
    std::vector<uint16_t> out(src.size() + 8, 0xABCD);
    for (uint64_t n = 0; n <= 40; ++n) {
      std::fill(out.begin(), out.end(), 0xABCD);
      serenity_f32_to_bf16_rne(src.data(), out.data(), n);
      for (uint64_t i = 0; i < n; ++i) expect(out[i] == ref_f32_to_bf16(src_bits[i]), "f32->bf16 short", src_bits[i]);
      expect(out[n] == 0xABCD, "f32->bf16 wrote past n", n);
    }
    const uint64_t n = src.size() - 3;
    serenity_f32_to_bf16_rne(src.data(), out.data(), n);
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; ++i) if (out[i] != ref_f32_to_bf16(src_bits[i])) { ++bad; if (bad < 4) expect(false, "f32->bf16 long", src_bits[i]); }
    expect(bad == 0, "f32->bf16 long mismatch count", bad);
    expect(out[n] == 0xABCD, "f32->bf16 long wrote past n", n);
    std::printf("f32->bf16: %llu elements checked\n", (unsigned long long)n);
  }
  // f16 -> bf16: every one of the 65536 half patterns, plus lengths 0..40.
  {
    std::vector<uint16_t> src(65536);
    for (uint32_t i = 0; i < 65536; ++i) src[i] = static_cast<uint16_t>(i);
    std::vector<uint16_t> out(src.size() + 8, 0xABCD);
    for (uint64_t n = 0; n <= 40; ++n) {
      std::fill(out.begin(), out.end(), 0xABCD);
      serenity_f16_to_bf16_rne(src.data(), out.data(), n);
      for (uint64_t i = 0; i < n; ++i) expect(out[i] == ref_f32_to_bf16(ref_f16_to_f32_bits(src[i])), "f16->bf16 short", src[i]);
      expect(out[n] == 0xABCD, "f16->bf16 wrote past n", n);
    }
    serenity_f16_to_bf16_rne(src.data(), out.data(), src.size());
    uint64_t bad = 0;
    for (uint32_t i = 0; i < 65536; ++i) if (out[i] != ref_f32_to_bf16(ref_f16_to_f32_bits(src[i]))) { ++bad; if (bad < 4) expect(false, "f16->bf16 full", i); }
    expect(bad == 0, "f16->bf16 full mismatch count", bad);
    std::printf("f16->bf16: all 65536 half patterns checked\n");
  }
  // nt_memcpy: every (dst misalignment 0..31) x (length 0..300) plus 64 MiB.
  {
    std::vector<uint8_t> src(64u << 20), dst((64u << 20) + 64);
    for (auto &b : src) b = static_cast<uint8_t>(rng());
    for (uint64_t mis = 0; mis < 32; ++mis) {
      for (uint64_t n = 0; n <= 300; ++n) {
        std::fill(dst.begin(), dst.begin() + 400, 0xEE);
        serenity_nt_memcpy(dst.data() + mis, src.data() + (n % 7), n);
        expect(std::memcmp(dst.data() + mis, src.data() + (n % 7), n) == 0, "nt_memcpy body", (mis << 32) | n);
        expect(dst[mis + n] == 0xEE, "nt_memcpy wrote past n", (mis << 32) | n);
        expect(mis == 0 || dst[mis - 1] == 0xEE, "nt_memcpy wrote before dst", (mis << 32) | n);
      }
    }
    serenity_nt_memcpy(dst.data(), src.data(), src.size());
    expect(std::memcmp(dst.data(), src.data(), src.size()) == 0, "nt_memcpy 64MiB", 0);
    std::printf("nt_memcpy: 32 alignments x 301 lengths + 64 MiB checked\n");
  }
  if (failures) { std::printf("FAILED: %d\n", failures); return 1; }
  std::printf("fastload_x86 kernels: PASS\n");
  return 0;
}
