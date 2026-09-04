// cos/sin tables for [S, 2*A*F]: angle = position[row, axis] * inv_freq[f],
// repeated twice along the last dimension.
extern "C" __global__ void ${function}(const dif_f32* positions, const dif_f32* inv_freq, ${output_type}* cosine, ${output_type}* sine) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${width}ULL, column = i % ${width}ULL,
                       component = column % ${unrepeated_width}ULL,
                       axis = component / ${frequencies}ULL, frequency = component % ${frequencies}ULL;
    float angle = dif_load_f32(positions, row * ${axes}ULL + axis) * dif_load_f32(inv_freq, frequency);
    ${store}(cosine, i, cosf(angle));
    ${store}(sine, i, sinf(angle));
  }
}
