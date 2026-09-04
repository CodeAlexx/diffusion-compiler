// Gradient of the saturating clamp. Inside the bounds the forward was the
// identity, so the upstream gradient passes through; outside, the output did
// not depend on the input and the gradient is zero. A value exactly on a
// bound counts as inside, matching the reference convention.
extern "C" __global__ void ${function}(const ${scalar}* g, const ${scalar}* x, ${scalar}* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float v = ${load}(x, i);
    ${store}(y, i, (v >= ${lower}f && v <= ${upper}f) ? ${load}(g, i) : 0.0f);
  }
}
