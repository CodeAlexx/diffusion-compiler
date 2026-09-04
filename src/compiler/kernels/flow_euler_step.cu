// H3's data-ward Euler step: denoised = sample + (1 - t) * v,
// output = ratio * sample + (1 - ratio) * denoised with ratio = sigma_next / sigma,
// every operation rounded to nearest.
extern "C" __global__ void ${function}(const dif_scalar* sample, const dif_scalar* velocity, const dif_f32* timesteps, const dif_f32* sigmas, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float timestep = dif_load_f32(timesteps, ${step}ULL), sigma = dif_load_f32(sigmas, ${step}ULL),
          sigma_next = dif_load_f32(sigmas, ${next_step}ULL), sample_value = dif_load(sample, i),
          velocity_value = dif_load(velocity, i), sigma_from_timestep = __fsub_rn(1.0f, timestep),
          ratio = sigma_next / sigma;
    float velocity_delta = __fmul_rn(sigma_from_timestep, velocity_value);
    float denoised = __fadd_rn(sample_value, velocity_delta);
    float complement = __fsub_rn(1.0f, ratio);
    float weighted_sample = __fmul_rn(ratio, sample_value);
    float weighted_denoised = __fmul_rn(complement, denoised);
    dif_store(output, i, __fadd_rn(weighted_sample, weighted_denoised));
  }
}
