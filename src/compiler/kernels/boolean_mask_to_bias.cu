// Additive attention bias from a boolean mask: 0 where valid, -inf elsewhere.
extern "C" __global__ void ${function}(const unsigned char* mask, ${scalar}* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long key = i % ${sequence}ULL, query = (i / ${sequence}ULL) % ${sequence}ULL,
                       batch = i / (${sequence}ULL * ${sequence}ULL);
    bool valid = ${valid};
    ${store}(y, i, valid ? 0.0f : -__int_as_float(0x7f800000));
  }
}
