// d loss / d prediction = 2 * grad_loss * (prediction - target) / N.
extern "C" __global__ void ${function}(const ${scalar}* prediction, const ${scalar}* target, const ${grad_loss_scalar}* grad_loss, ${grad_scalar}* grad_prediction) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    float factor = 2.0f * ${grad_loss_load}(grad_loss, 0ULL) / ${count}.0f;
    ${grad_store}(grad_prediction, i, (${load}(prediction, i) - ${load}(target, i)) * factor);
  }
}
