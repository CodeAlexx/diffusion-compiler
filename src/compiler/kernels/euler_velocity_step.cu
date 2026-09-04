// output = sample + round_dtype((next - current) * velocity), each step
// rounded explicitly (the generic velocity-form Euler update).
extern "C" __global__ void ${function}(const dif_scalar* sample, const dif_scalar* velocity, const dif_f32* current, const dif_f32* next, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float dt = __fsub_rn(dif_load_f32(next, 0ULL), dif_load_f32(current, 0ULL));
    float scaled = dif_round(__fmul_rn(dt, dif_load(velocity, i)));
    dif_store(output, i, __fadd_rn(dif_load(sample, i), scaled));
  }
}
