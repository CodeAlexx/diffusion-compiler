// [B,C,T,H,W] <-> rows [B*T/pt*H/ph*W/pw, C*pt*ph*pw]; the same index map
// serves patchify and unpatchify, only the transfer direction differs.
extern "C" __global__ void ${function}(const dif_scalar* input, dif_scalar* output) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ${count}ULL) {
    unsigned long long row = i / ${row_width}ULL, column = i % ${row_width}ULL, outer = row,
                       patch_x = outer % ${output_width}ULL;
    outer /= ${output_width}ULL;
    unsigned long long patch_y = outer % ${output_height}ULL;
    outer /= ${output_height}ULL;
    unsigned long long patch_frame = outer % ${output_frames}ULL, batch = outer / ${output_frames}ULL,
                       inner = column, offset_x = inner % ${patch_w}ULL;
    inner /= ${patch_w}ULL;
    unsigned long long offset_y = inner % ${patch_h}ULL;
    inner /= ${patch_h}ULL;
    unsigned long long offset_t = inner % ${patch_t}ULL;
    inner /= ${patch_t}ULL;
    unsigned long long channel = inner, frame = patch_frame * ${patch_t}ULL + offset_t,
                       y = patch_y * ${patch_h}ULL + offset_y, x = patch_x * ${patch_w}ULL + offset_x,
                       volume_index = ((((batch * ${channels}ULL + channel) * ${frames}ULL + frame) * ${height}ULL + y) * ${width}ULL + x);
    ${transfer}
  }
}
