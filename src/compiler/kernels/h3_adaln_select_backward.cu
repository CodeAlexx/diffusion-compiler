// Gradient of the adaLN modulation selection. Every token reads six rows of
// the projected table by its table index, so the gradient sums the tokens
// that named each table row back into it. A gather over the TABLE, one thread
// per table element, so tokens sharing an index sum deterministically.
extern "C" __global__ void ${function}(const int* indices, const ${scalar}* g0, const ${scalar}* g1, const ${scalar}* g2, const ${scalar}* g3, const ${scalar}* g4, const ${scalar}* g5, ${scalar}* grad_projected) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long col = i % ${hidden}ULL, rest = i / ${hidden}ULL;
    unsigned long long chunk = rest % 6ULL, table = rest / 6ULL;
    const ${scalar}* source = chunk == 0ULL ? g0 : chunk == 1ULL ? g1 : chunk == 2ULL ? g2 : chunk == 3ULL ? g3 : chunk == 4ULL ? g4 : g5;
    float accumulator = 0.0f;
    for (unsigned long long s = 0ULL; s < ${rows}ULL; ++s)
      if ((unsigned long long)indices[s] == table)
        accumulator += ${load}(source, s * ${hidden}ULL + col);
    ${store}(grad_projected, i, accumulator);
  }
}
