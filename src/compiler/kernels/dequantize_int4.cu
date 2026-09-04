// INT4 group dequantization: two codes per byte (low nibble = even column,
// two's complement), one scale per `group` adjacent columns, optional one
// outlier residual per group added at the recorded column.
extern "C" __global__ void ${function}(${parameters}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL, col = i % ${columns}ULL;
    unsigned char byte = packed[row * ${packed_columns}ULL + col / 2ULL];
    unsigned int nibble = (col & 1ULL) ? (byte >> 4U) : (byte & 15U);
    int q = nibble < 8U ? (int)nibble : (int)nibble - 16;
    float scale = dif_load(scales, row * ${groups}ULL + col / ${group}ULL);
    float value = (float)q * scale;
    ${outlier}
    dif_store(y, i, value);
  }
}
