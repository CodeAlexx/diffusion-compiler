// Serial reference reduction on one thread: the training-loss oracle path.
extern "C" __global__ void ${function}(const ${scalar}* prediction, const ${scalar}* target, ${loss_scalar}* loss) {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    float sum = 0.0f;
    for (unsigned long long i = 0ULL; i < ${count}ULL; ++i) {
      float d = ${load}(prediction, i) - ${load}(target, i);
      sum = fmaf(d, d, sum);
    }
    ${store}(loss, 0ULL, sum / ${count}.0f);
  }
}
