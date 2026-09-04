// Gradient of the rotary embedding: the rotation transpose. A pair (first,
// second) rotated by (c,s) comes back through (c,-s; s,c), and the tail past
// the rotated range passes straight through. The pairing and the table dtype
// are generated from the operation's own attributes, so this differentiates
// the rotation that actually ran.
extern "C" __global__ void ${function}(const ${scalar}* grad_output, const ${table}* cosv, const ${table}* sinv, ${scalar}* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long d = i % ${dim}ULL, rest = i / ${dim}ULL;
    unsigned long long base = rest * ${dim}ULL, token = rest / ${heads}ULL;
    unsigned long long table = token * ${pairs}ULL;
    if (d >= ${rotated}ULL) {
      ${store}(grad_input, i, ${load}(grad_output, i));
    } else {
      unsigned long long pair = ${pair_index};
      bool leading = ${is_leading};
      unsigned long long partner = leading ? ${second_offset} : ${first_offset};
      float c = ${table_load}(cosv, table + pair);
      float s = ${table_load}(sinv, table + pair);
      float own = ${load}(grad_output, i);
      float other = ${load}(grad_output, base + partner);
      // Leading lane: g_first*c + g_second*s. Trailing lane: g_second*c - g_first*s.
      ${store}(grad_input, i, leading ? own * c + other * s : own * c - other * s);
    }
  }
}
