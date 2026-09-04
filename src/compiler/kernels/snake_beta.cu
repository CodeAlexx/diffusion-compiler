// Snake-beta activation (BigVGAN): y = x + (1 / (exp(lb) + eps)) * sin(exp(la) * x)^2,
// alpha/beta stored in log space per channel of a [B, C, L] tensor.
extern "C" __global__ void ${function}(const dif_scalar* x, const dif_scalar* la, const dif_scalar* lb, dif_scalar* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long c = (i / ${length}ULL) % ${channels}ULL;
    float alpha = expf(dif_load(la, c));
    float ib = 1.0f / (expf(dif_load(lb, c)) + ${epsilon}f);
    float xv = dif_load(x, i);
    float s = sinf(alpha * xv);
    dif_store(y, i, xv + ib * s * s);
  }
}
