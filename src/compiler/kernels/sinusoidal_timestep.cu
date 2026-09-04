// Sinusoidal timestep embedding with the creator's frequency shift, scale,
// max period and sin/cos ordering folded into literals by the emitter.
extern "C" __global__ void ${function}(const dif_f32* timesteps, dif_f32* y) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${width}ULL, column = i % ${width}ULL;
    if (column >= ${paired}ULL) {
      dif_store_f32(y, i, 0.0f);
      return;
    }
    unsigned long long component = column % ${half}ULL;
    float exponent = (-${log_period}f * (float)component) / ${denominator}f;
    float frequency = expf(exponent);
    float scaled_timestep = dif_load_f32(timesteps, row) * ${scale}f;
    float angle = scaled_timestep * frequency;
    float s = sinf(angle), c = cosf(angle);
    dif_store_f32(y, i, ${select});
  }
}
