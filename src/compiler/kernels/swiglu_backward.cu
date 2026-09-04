// Thread i owns one element of the packed [.., 2W] input gradient; the value
// half receives silu(gate)*g, the gate half dsilu(gate)*value*g; elements
// outside the packed window get zero.
extern "C" __global__ void ${function}(const dif_scalar* grad_output, const dif_scalar* x, dif_scalar* grad_input) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${input_width}ULL, col = i % ${input_width}ULL;
    if (col < ${start}ULL || col >= ${window_end}ULL) {
      dif_store(grad_input, i, 0.0f);
      return;
    }
    unsigned long long lane = col - ${start}ULL, cw = lane < ${width}ULL ? lane : lane - ${width}ULL,
                       base = row * ${input_width}ULL + ${start}ULL;
    float value = dif_load(x, base + ${value_offset}ULL + cw);
    float gate = dif_load(x, base + ${gate_offset}ULL + cw);
    float sigmoid = 1.0f / (1.0f + expf(-gate));
    float upstream = dif_load(grad_output, row * ${width}ULL + cw);
    int is_value_slot = lane ${value_slot_test} ${width}ULL;
    float gradient = is_value_slot ? gate * sigmoid * upstream : sigmoid * (1.0f + gate * (1.0f - sigmoid)) * value * upstream;
    dif_store(grad_input, i, gradient);
  }
}
