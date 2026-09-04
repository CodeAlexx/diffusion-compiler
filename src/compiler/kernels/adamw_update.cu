// Decoupled AdamW with typed parameter/gradient storage and F32 moments.
// The decay multiplies the parameter BEFORE the moment update is subtracted
// and never folds into the gradient (flame's LoRA-A runaway lesson).
extern "C" __global__ void ${function}(const ${parameter_scalar}* parameter, const ${gradient_scalar}* gradient, const dif_f32* first, const dif_f32* second, const int* completed_steps, ${parameter_scalar}* updated, dif_f32* updated_first, dif_f32* updated_second) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float step = (float)(completed_steps[0] + 1), beta1 = ${beta1}f, beta2 = ${beta2}f;
    float grad = ${gradient_load}(gradient, i) * ${clip_scale}f;
    float m = beta1 * dif_load_f32(first, i) + (1.0f - beta1) * grad;
    float v = beta2 * dif_load_f32(second, i) + (1.0f - beta2) * grad * grad;
    float bias1 = 1.0f - powf(beta1, step);
    float bias2_sqrt = sqrtf(1.0f - powf(beta2, step));
    float decayed = ${parameter_load}(parameter, i) * (1.0f - ${learning_rate}f * ${weight_decay}f);
    float denominator = sqrtf(v) / bias2_sqrt + ${epsilon}f;
    float value = decayed - (${learning_rate}f / bias1) * m / denominator;
    ${parameter_store}(updated, i, value);
    dif_store_f32(updated_first, i, m);
    dif_store_f32(updated_second, i, v);
  }
}
