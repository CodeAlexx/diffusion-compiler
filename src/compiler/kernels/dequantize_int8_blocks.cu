// Row-major INT8 weights with one F32 scale per `block` adjacent K values.
extern "C" __global__ void ${function}(const signed char* x, const float* scales, dif_bf16* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${columns}ULL;
    unsigned long long column = i % ${columns}ULL;
    dif_store_bf16(y, i, (float)x[i] * scales[row * ${scale_columns}ULL + column / ${block}ULL]);
  }
}
