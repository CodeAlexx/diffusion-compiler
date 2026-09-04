// output = f * left + (1 - f) * right with round-to-nearest at every step.
extern "C" __global__ void ${function}(const dif_scalar* left, const dif_scalar* right, const dif_f32* factor, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float f = dif_load_f32(factor, 0ULL), left_value = dif_load(left, i),
          right_value = dif_load(right, i);
    float complement = __fsub_rn(1.0f, f);
    float weighted_left = __fmul_rn(f, left_value);
    float weighted_right = __fmul_rn(complement, right_value);
    dif_store(output, i, __fadd_rn(weighted_left, weighted_right));
  }
}
