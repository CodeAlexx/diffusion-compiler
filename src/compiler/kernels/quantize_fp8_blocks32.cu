// Blackwell MXFP8 quantization: per 32-column block, |max|/448 rounded UP to
// a power of two (UE8M0), values converted with cvt.rn.satfinite.e4m3x2;
// the scale byte lands in the cuBLASLt 128x4 tiled layout.
extern "C" __global__ void ${function}(const dif_bf16* x, unsigned char* q, unsigned char* scales) {
  unsigned long long row = blockIdx.x;
  unsigned tid = threadIdx.x;
  if (row >= ${rows}ULL) return;
  unsigned lane = tid & 31U;
  unsigned warp = tid >> 5U;
  for (unsigned long long block = warp; block < ${blocks}ULL; block += 8ULL) {
    unsigned long long column = block * 32ULL + lane;
    float value = column < ${columns}ULL ? dif_load_bf16(x, row * ${columns}ULL + column) : 0.0f;
    float maximum = fabsf(value);
    for (unsigned offset = 16U; offset > 0U; offset >>= 1U)
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, offset));
    maximum = __shfl_sync(0xffffffffU, maximum, 0U);
    float target = maximum / 448.0f;
    unsigned bits = __float_as_uint(target) & 0x7fffffffU;
    unsigned exponent = bits >> 23U;
    unsigned mantissa = bits & 0x7fffffU;
    unsigned encoded = (bits == 0U) ? 0U : ((exponent == 0U) ? (mantissa > 0x400000U ? 1U : 0U) : exponent + (mantissa != 0U));
    unsigned char encoded_scale = (unsigned char)(encoded > 254U ? 254U : encoded);
    float scale = ldexpf(1.0f, (int)encoded_scale - 127);
    if (lane == 0U) {
      unsigned long long tile_outer = row / 128ULL;
      unsigned long long tile_inner = (block / 4ULL) * 4ULL;
      unsigned long long within = (row % 32ULL) * 16ULL + ((row % 128ULL) / 32ULL) * 4ULL + block % 4ULL;
      unsigned long long scale_offset = (tile_inner + tile_outer * ${scale_inner}ULL) * 128ULL + within;
      scales[scale_offset] = encoded_scale;
    }
    if (column < ${columns}ULL) {
      float divided = value / scale;
      unsigned short pair;
      asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}\n" : "=h"(pair) : "f"(divided), "f"(0.0f));
      q[row * ${columns}ULL + column] = (unsigned char)pair;
    }
  }
}
