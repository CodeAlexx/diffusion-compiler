// Gradient of the packed-QKV weight split: the inverse permutation. Every
// packed element comes from exactly one component, so nothing is summed and
// nothing is dropped.
extern "C" __global__ void ${function}(const ${scalar}* grad_q, const ${scalar}* grad_k, const ${scalar}* grad_v, ${scalar}* grad_packed) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long column = i % ${hidden}ULL, rest = i / ${hidden}ULL;
    unsigned long long d = rest % ${head_dim}ULL, rest2 = rest / ${head_dim}ULL;
    unsigned long long component = rest2 % 3ULL, head = rest2 / 3ULL;
    const ${scalar}* source = component == 0ULL ? grad_q : component == 1ULL ? grad_k : grad_v;
    ${store}(grad_packed, i,
             ${load}(source, (head * ${head_dim}ULL + d) * ${hidden}ULL + column));
  }
}
