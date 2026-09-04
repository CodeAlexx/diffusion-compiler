// Integer nearest-neighbor spatial expansion of an NCHW tensor.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long ow = i % ${output_w}ULL, oh = (i / ${output_w}ULL) % ${output_h}ULL,
                       c = (i / ${output_hw}ULL) % ${channels}ULL, b = i / ${output_chw}ULL;
    unsigned long long source = ((b * ${channels}ULL + c) * ${input_h}ULL + oh / ${scale_h}ULL) * ${input_w}ULL + ow / ${scale_w}ULL;
    dif_store(y, i, dif_load(x, source));
  }
}
