// Gradient of RMS normalization across a channel axis, scaled by sqrt(C) and
// gamma. The fiber statistic is recomputed from the original input in F32
// rather than saved, as the other norm gradients here do.
//
// Two reductions with different axes share one kernel: the input gradient is
// per element and reduces along its own fiber, while the gamma gradient
// reduces across every fiber of a channel. Each output element is owned by
// exactly one thread, so both sum deterministically without atomics.
//
// The forward clamps the denominator at epsilon. Below that clamp the output
// is linear in the input, so the gradient is too -- and the branch here is
// the same branch the forward took, not an approximation of it.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, const dif_scalar* gamma, dif_scalar* grad_input${gamma_parameter}) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long trailing = i % ${inner}ULL, rest = i / ${inner}ULL;
    unsigned long long channel = rest % ${channels}ULL;
    unsigned long long fiber = rest / ${channels}ULL * ${channels}ULL * ${inner}ULL + trailing;
    float squared = 0.0f;
    for (unsigned long long c = 0ULL; c < ${channels}ULL; ++c) {
      float value = dif_load(x, fiber + c * ${inner}ULL);
      squared = fmaf(value, value, squared);
    }
    float length = sqrtf(squared);
    float denominator = fmaxf(length, ${epsilon}f);
    float own = dif_load(grad_output, i) * dif_load(gamma, channel) * ${scale}f;
    if (length > ${epsilon}f) {
      float projected = 0.0f;
      for (unsigned long long c = 0ULL; c < ${channels}ULL; ++c)
        projected = fmaf(dif_load(grad_output, fiber + c * ${inner}ULL) *
                             dif_load(gamma, c),
                         dif_load(x, fiber + c * ${inner}ULL), projected);
      projected *= ${scale}f;
      dif_store(grad_input, i,
                own / denominator -
                    dif_load(x, i) * projected / (denominator * denominator * denominator));
    } else {
      dif_store(grad_input, i, own / denominator);
    }
  }
  ${gamma_gradient}
}
