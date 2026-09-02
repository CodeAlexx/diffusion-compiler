// The transcendental approximations below are an adapted, AVX2-only subset of
// PyTorch's aten/src/ATen/native/cpu/avx_mathfun.h. The original is based on
// Julien Pommier's sse_mathfun and is distributed under the zlib license:
// Copyright (C) 2012 Giovanni Garberoglio. This source is altered to remove
// ATen dependencies and retain only the log/sincos path used by normal_fill.

#include <immintrin.h>

namespace dif {
namespace {

using v8sf = __m256;
using v8si = __m256i;

v8sf f32(float value) { return _mm256_set1_ps(value); }
v8si i32(int value) { return _mm256_set1_epi32(value); }
v8sf f32_bits(int value) { return _mm256_castsi256_ps(i32(value)); }

v8sf log256_ps(v8sf x) {
  const auto one = f32(1.0F);
  const auto invalid = _mm256_cmp_ps(x, _mm256_setzero_ps(), _CMP_LE_OS);
  x = _mm256_max_ps(x, f32_bits(0x00800000));
  auto exponent = _mm256_srli_epi32(_mm256_castps_si256(x), 23);
  x = _mm256_and_ps(x, f32_bits(~0x7f800000));
  x = _mm256_or_ps(x, f32(0.5F));
  exponent = _mm256_sub_epi32(exponent, i32(0x7f));
  auto e = _mm256_add_ps(_mm256_cvtepi32_ps(exponent), one);
  const auto mask = _mm256_cmp_ps(x, f32(0.707106781186547524F), _CMP_LT_OS);
  const auto masked_x = _mm256_and_ps(x, mask);
  x = _mm256_sub_ps(x, one);
  e = _mm256_sub_ps(e, _mm256_and_ps(one, mask));
  x = _mm256_add_ps(x, masked_x);
  const auto z = _mm256_mul_ps(x, x);

  auto y = f32(7.0376836292E-2F);
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(-1.1514610310E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(1.1676998740E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(-1.2420140846E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(1.4249322787E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(-1.6668057665E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(2.0000714765E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(-2.4999993993E-1F));
  y = _mm256_add_ps(_mm256_mul_ps(y, x), f32(3.3333331174E-1F));
  y = _mm256_mul_ps(_mm256_mul_ps(y, x), z);
  y = _mm256_add_ps(y, _mm256_mul_ps(e, f32(-2.12194440e-4F)));
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, f32(0.5F)));
  x = _mm256_add_ps(x, y);
  x = _mm256_add_ps(x, _mm256_mul_ps(e, f32(0.693359375F)));
  return _mm256_or_ps(x, invalid);
}

void sincos256_ps(v8sf x, v8sf *sine, v8sf *cosine) {
  auto sign_bit_sine = x;
  x = _mm256_and_ps(x, f32_bits(~static_cast<int>(0x80000000U)));
  sign_bit_sine =
      _mm256_and_ps(sign_bit_sine, f32_bits(static_cast<int>(0x80000000U)));
  auto y = _mm256_mul_ps(x, f32(1.27323954473516F));
  auto j = _mm256_cvttps_epi32(y);
  j = _mm256_add_epi32(j, i32(1));
  j = _mm256_and_si256(j, i32(~1));
  y = _mm256_cvtepi32_ps(j);
  auto cosine_j = j;

  auto sine_swap = _mm256_and_si256(j, i32(4));
  sine_swap = _mm256_slli_epi32(sine_swap, 29);
  auto poly = _mm256_and_si256(j, i32(2));
  poly = _mm256_cmpeq_epi32(poly, i32(0));
  const auto sine_swap_bit = _mm256_castsi256_ps(sine_swap);
  const auto poly_mask = _mm256_castsi256_ps(poly);

  auto x1 = _mm256_mul_ps(y, f32(-0.78515625F));
  auto x2 = _mm256_mul_ps(y, f32(-2.4187564849853515625e-4F));
  auto x3 = _mm256_mul_ps(y, f32(-3.77489497744594108e-8F));
  x = _mm256_add_ps(x, x1);
  x = _mm256_add_ps(x, x2);
  x = _mm256_add_ps(x, x3);

  cosine_j = _mm256_sub_epi32(cosine_j, i32(2));
  cosine_j = _mm256_andnot_si256(cosine_j, i32(4));
  cosine_j = _mm256_slli_epi32(cosine_j, 29);
  const auto sign_bit_cosine = _mm256_castsi256_ps(cosine_j);
  sign_bit_sine = _mm256_xor_ps(sign_bit_sine, sine_swap_bit);

  const auto z = _mm256_mul_ps(x, x);
  y = f32(2.443315711809948E-005F);
  y = _mm256_add_ps(_mm256_mul_ps(y, z), f32(-1.388731625493765E-003F));
  y = _mm256_add_ps(_mm256_mul_ps(y, z), f32(4.166664568298827E-002F));
  y = _mm256_mul_ps(y, z);
  y = _mm256_mul_ps(y, z);
  y = _mm256_sub_ps(y, _mm256_mul_ps(z, f32(0.5F)));
  y = _mm256_add_ps(y, f32(1.0F));

  auto y2 = f32(-1.9515295891E-4F);
  y2 = _mm256_add_ps(_mm256_mul_ps(y2, z), f32(8.3321608736E-3F));
  y2 = _mm256_add_ps(_mm256_mul_ps(y2, z), f32(-1.6666654611E-1F));
  y2 = _mm256_mul_ps(y2, z);
  y2 = _mm256_mul_ps(y2, x);
  y2 = _mm256_add_ps(y2, x);

  const auto sine2 = _mm256_and_ps(poly_mask, y2);
  const auto sine1 = _mm256_andnot_ps(poly_mask, y);
  y2 = _mm256_sub_ps(y2, sine2);
  y = _mm256_sub_ps(y, sine1);
  const auto sine_value = _mm256_add_ps(sine1, sine2);
  const auto cosine_value = _mm256_add_ps(y, y2);
  *sine = _mm256_xor_ps(sine_value, sign_bit_sine);
  *cosine = _mm256_xor_ps(cosine_value, sign_bit_cosine);
}

} // namespace

bool torch_cpu_rng_avx2_available() {
#if defined(__GNUC__) || defined(__clang__)
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
#else
  return false;
#endif
}

void torch_cpu_normal_fill16_avx2(float *data) {
  const auto one = f32(1.0F);
  const auto minus_two = f32(-2.0F);
  const auto two_pi = f32(6.2831853071795864769F);
  const auto u1 = _mm256_sub_ps(one, _mm256_loadu_ps(data));
  const auto u2 = _mm256_loadu_ps(data + 8);
  const auto radius = _mm256_sqrt_ps(_mm256_mul_ps(minus_two, log256_ps(u1)));
  const auto theta = _mm256_mul_ps(two_pi, u2);
  v8sf sine{};
  v8sf cosine{};
  sincos256_ps(theta, &sine, &cosine);
  const auto n1 = _mm256_mul_ps(radius, cosine);
  const auto n2 = _mm256_mul_ps(radius, sine);
  _mm256_storeu_ps(data, _mm256_fmadd_ps(n1, one, _mm256_setzero_ps()));
  _mm256_storeu_ps(data + 8,
                   _mm256_fmadd_ps(n2, one, _mm256_setzero_ps()));
}

} // namespace dif
