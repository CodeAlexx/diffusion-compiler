// Generic multi-axis rotary tables: each output pair maps to (axis,
// component); omega = 1 / (theta * ntk) ^ (2 * component / axis_dim).
extern "C" __global__ void ${function}(const dif_f32* positions, const int* pair_axes, const int* pair_indices, const int* axis_dims, dif_f32* cosine, dif_f32* sine) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long pair = i % ${pairs}ULL, token = (i / ${pairs}ULL) % ${sequence}ULL,
                       batch = i / (${pairs}ULL * ${sequence}ULL);
    int axis = pair_axes[pair], component = pair_indices[pair], axis_dim = axis_dims[axis];
    float scale = (2.0f * (float)component) / (float)axis_dim;
    float omega = 1.0f / powf((float)(${theta} * ${ntk}), scale);
    float angle = dif_load_f32(positions, (batch * ${sequence}ULL + token) * ${axes}ULL + (unsigned long long)axis) * omega;
    dif_store_f32(cosine, i, cosf(angle));
    dif_store_f32(sine, i, sinf(angle));
  }
}
