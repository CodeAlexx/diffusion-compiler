// Edge-exclusive reflection padding of an NCDHW tensor.
extern "C" __global__ void ${function}(const dif_scalar* x, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long ow = i % ${out_w}ULL, oh = (i / ${out_w}ULL) % ${out_h}ULL,
                       ot = (i / ${out_hw}ULL) % ${out_t}ULL, c = (i / ${out_thw}ULL) % ${out_c}ULL,
                       b = i / ${out_cthw}ULL;
    unsigned long long st = ot < ${front}ULL ? ${front}ULL - ot : (ot - ${front}ULL < ${in_t}ULL ? ot - ${front}ULL : 2ULL * ${in_t}ULL - 2ULL - (ot - ${front}ULL));
    unsigned long long sy = oh < ${top}ULL ? ${top}ULL - oh : (oh - ${top}ULL < ${in_h}ULL ? oh - ${top}ULL : 2ULL * ${in_h}ULL - 2ULL - (oh - ${top}ULL));
    unsigned long long sx = ow < ${west}ULL ? ${west}ULL - ow : (ow - ${west}ULL < ${in_w}ULL ? ow - ${west}ULL : 2ULL * ${in_w}ULL - 2ULL - (ow - ${west}ULL));
    unsigned long long source = (((b * ${in_c}ULL + c) * ${in_t}ULL + st) * ${in_h}ULL + sy) * ${in_w}ULL + sx;
    dif_store(y, i, dif_load(x, source));
  }
}
