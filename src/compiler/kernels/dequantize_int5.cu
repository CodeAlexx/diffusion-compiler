// INT5 dequantization: 5-bit codes packed bit-contiguously along each row
// (row stride 5*K/8 bytes), one scale per `group` columns, optional per-column
// scales.
extern "C" __global__ void ${function}(${parameters}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, col = i % ${columns}ULL,
                       bit = col * 5ULL, bi = row * ${row_bytes}ULL + bit / 8ULL;
    unsigned int shift = (unsigned int)(bit & 7ULL);
    unsigned int word = packed[bi];
    if (shift + 5U > 8U)
      word |= ((unsigned int)packed[bi + 1ULL]) << 8U;
    unsigned int encoded = (word >> shift) & 31U;
    int q = encoded < 16U ? (int)encoded : (int)encoded - 32;
    float scale = dif_load(scales, row * ${groups}ULL + col / ${group}ULL);
    float value = (float)q * scale;
    ${column_scale}
    dif_store(y, i, value);
  }
}
