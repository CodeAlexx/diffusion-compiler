// Epilogue for the FP8 GEMM: the raw BF16 product in y is rescaled in place
// by the row and column dequantization scales (x and w are the GEMM
// operands, unused here).
extern "C" __global__ void ${function}(const unsigned char* x, const unsigned char* w, const float* rs, const float* cs, dif_bf16* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL;
    unsigned long long column = i % ${columns}ULL;
    dif_store_bf16(y, i, dif_load_bf16(y, i) * rs[row] * cs[column]);
  }
}
