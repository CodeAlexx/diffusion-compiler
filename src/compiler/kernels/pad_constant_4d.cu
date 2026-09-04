// Constant padding of an NCHW tensor.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long ow = i % ${out_w}ULL, oh = (i / ${out_w}ULL) % ${out_h}ULL,
                       c = (i / ${out_hw}ULL) % ${out_c}ULL, b = i / ${out_chw}ULL;
    if (oh < ${top}ULL || oh >= ${bottom_edge}ULL || ow < ${west}ULL || ow >= ${east_edge}ULL) {
      dif_store(y, i, ${value}f);
      return;
    }
    unsigned long long source = ((b * ${in_c}ULL + c) * ${in_h}ULL + oh - ${top}ULL) * ${in_w}ULL + ow - ${west}ULL;
    dif_store(y, i, dif_load(x, source));
  }
}
