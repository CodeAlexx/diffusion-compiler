// Gradient of reflect padding. The forward is a gather: every output element
// reads exactly one input element, and the elements just inside each edge are
// read more than once. So the gradient is a scatter-add -- and it is written
// as a gather over the INPUT, one thread per input element summing the output
// positions that read it, rather than one thread per output racing with
// atomicAdd. The sum is then deterministic, which a training run depends on.
//
// Reflection is separable, and per axis an input coordinate is read from at
// most three output coordinates: the identity copy, the low reflection, and
// the high one. The bound is three per axis, so at most twenty-seven loads,
// whatever the padding is. A rank-4 tensor is the rank-5 form with a depth of
// one, which makes its flat layout identical and needs no separate kernel.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long w = i % ${in_w}ULL, rest = i / ${in_w}ULL;
    unsigned long long h = rest % ${in_h}ULL;
    rest /= ${in_h}ULL;
    unsigned long long t = rest % ${in_t}ULL, plane = rest / ${in_t}ULL;
    unsigned long long ot[3], oh[3], ow[3];
    int nt = 0, nh = 0, nw = 0;
    ot[nt++] = t + ${front}ULL;
    if (t >= 1ULL && t <= ${front}ULL) ot[nt++] = ${front}ULL - t;
    if (${back}ULL >= 1ULL && t + 1ULL + ${back}ULL >= ${in_t}ULL && t + 2ULL <= ${in_t}ULL) ot[nt++] = ${front}ULL + 2ULL * ${in_t}ULL - 2ULL - t;
    oh[nh++] = h + ${top}ULL;
    if (h >= 1ULL && h <= ${top}ULL) oh[nh++] = ${top}ULL - h;
    if (${bottom}ULL >= 1ULL && h + 1ULL + ${bottom}ULL >= ${in_h}ULL && h + 2ULL <= ${in_h}ULL) oh[nh++] = ${top}ULL + 2ULL * ${in_h}ULL - 2ULL - h;
    ow[nw++] = w + ${west}ULL;
    if (w >= 1ULL && w <= ${west}ULL) ow[nw++] = ${west}ULL - w;
    if (${east}ULL >= 1ULL && w + 1ULL + ${east}ULL >= ${in_w}ULL && w + 2ULL <= ${in_w}ULL) ow[nw++] = ${west}ULL + 2ULL * ${in_w}ULL - 2ULL - w;
    float accumulator = 0.0f;
    for (int a = 0; a < nt; ++a)
      for (int b = 0; b < nh; ++b)
        for (int c = 0; c < nw; ++c)
          accumulator += dif_load(grad_output, ((plane * ${out_t}ULL + ot[a]) * ${out_h}ULL + oh[b]) * ${out_w}ULL + ow[c]);
    dif_store(grad_input, i, accumulator);
  }
}
