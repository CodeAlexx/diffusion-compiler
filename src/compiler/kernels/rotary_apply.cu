// Rotary embedding on [B,L,H,D] with f32 cos/sin tables [B,L,P]. The
// layout (interleaved or half-split) is folded into the pair/partner index
// expressions; every multiply-add is a separately rounded PTX op.
extern "C" __global__ void ${function}(const ${scalar}* x, const dif_f32* cosine, const dif_f32* sine, ${scalar}* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long d = i % ${dim}ULL, outer = i / ${dim}ULL, token = (outer / ${heads}ULL) % ${sequence}ULL,
                       batch = outer / (${heads}ULL * ${sequence}ULL);
    if (d < ${rotated}ULL) {
      unsigned long long pair = ${pair}, base = i - d, table = (batch * ${sequence}ULL + token) * ${pairs}ULL + pair;
      float even = ${load}(x, ${even_index}), odd = ${load}(x, ${odd_index}), c = dif_load_f32(cosine, table),
            s = dif_load_f32(sine, table), first, second, result;
      if (${is_second}) {
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(even), "f"(s));
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(odd), "f"(c));
        asm volatile("add.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
      } else {
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(first) : "f"(even), "f"(c));
        asm volatile("mul.rn.f32 %0,%1,%2;" : "=f"(second) : "f"(odd), "f"(s));
        asm volatile("sub.rn.f32 %0,%1,%2;" : "=f"(result) : "f"(first), "f"(second));
      }
      ${store}(y, i, result);
    } else ${store}(y, i, ${load}(x, i));
  }
}
