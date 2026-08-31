#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/frontend/h3.hpp"
#include "dif/frontend/h3_vae.hpp"
#include "dif/frontend/training.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::uint64_t number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text, &end, 10);
  if (!text[0] || !end || *end != '\0')
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

double floating_number(const char *text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text, &end);
  if (!text[0] || !end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label + ": " + text);
  return value;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left)
    dif::fail(std::string(label) + " overflows");
  return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          const char *label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left)
    dif::fail(std::string(label) + " overflows");
  return left + right;
}

dif::ir::TensorDesc tensor(std::uint32_t id, std::uint32_t roles,
                           std::vector<std::uint64_t> dims) {
  return {id, dif::ir::DType::F32, roles, std::move(dims)};
}

dif::ir::Program make_rms(std::uint64_t rows, std::uint64_t cols,
                          std::uint64_t block) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      tensor(1, TensorRole::Input, {rows, cols}),
      tensor(2, TensorRole::Input, {rows, cols}),
      tensor(3, TensorRole::Input, {rows, cols}),
      tensor(4, TensorRole::Output, {rows, cols}),
  };
  program.operations = {{1,
                         Opcode::RmsNormModulate,
                         {1, 2, 3},
                         {4},
                         {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                          Attribute::u64(AttrKey::BlockSize, block)}}};
  return program;
}

dif::ir::DType float_dtype(const std::string &name) {
  if (name == "f32")
    return dif::ir::DType::F32;
  if (name == "bf16")
    return dif::ir::DType::BF16;
  if (name == "f16")
    return dif::ir::DType::F16;
  dif::fail("dtype must be f32, bf16, or f16");
}

dif::ir::Program make_linear_blend(std::uint64_t rows, std::uint64_t columns,
                                   dif::ir::DType dtype) {
  using namespace dif::ir;
  if (rows == 0U || columns == 0U)
    dif::fail("linear blend dimensions must be nonzero");
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, {rows, columns}},
      {2, dtype, TensorRole::Input, {rows, columns}},
      {3, DType::F32, TensorRole::Input, {1}},
      {4, dtype, TensorRole::Output, {rows, columns}},
  };
  program.operations = {{1, Opcode::LinearBlend, {1, 2, 3}, {4}, {}}};
  return program;
}

dif::ir::Program make_residual_gate(std::uint64_t rows,
                                    std::uint64_t columns,
                                    std::uint64_t block,
                                    dif::ir::DType dtype) {
  using namespace dif::ir;
  if (rows == 0U || columns == 0U)
    dif::fail("residual gate dimensions must be nonzero");
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, {rows, columns}},
      {2, dtype, TensorRole::Input, {rows, columns}},
      {3, dtype, TensorRole::Input, {rows, columns}},
      {4, dtype, TensorRole::Output, {rows, columns}},
  };
  program.operations = {
      {1,
       Opcode::ResidualGate,
       {1, 2, 3},
       {4},
       {Attribute::u64(AttrKey::BlockSize, block)}},
  };
  return program;
}

dif::ir::Program make_flow_euler_trajectory(std::uint64_t rows,
                                            std::uint64_t columns,
                                            std::uint64_t steps,
                                            dif::ir::DType dtype) {
  using namespace dif::ir;
  if (rows == 0U || columns == 0U || steps == 0U || steps > 100000U)
    dif::fail("flow trajectory requires nonzero dimensions and 1..100000 steps");
  if (steps > UINT32_MAX / 2U - 2U)
    dif::fail("flow trajectory tensor ids overflow");
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, {rows, columns}},
      {2, DType::F32, TensorRole::Input, {steps}},
      {3, DType::F32, TensorRole::Input, {steps + 1U}},
  };
  for (std::uint64_t step = 0; step < steps; ++step)
    program.tensors.push_back(
        {static_cast<std::uint32_t>(4U + step), dtype, TensorRole::Input,
         {rows, columns}});
  for (std::uint64_t step = 0; step < steps; ++step)
    program.tensors.push_back(
        {static_cast<std::uint32_t>(4U + steps + step), dtype,
         TensorRole::Output, {rows, columns}});
  for (std::uint64_t step = 0; step < steps; ++step) {
    const auto input = step == 0U ? 1U : 3U + steps + step;
    const auto velocity = 4U + step;
    const auto output = 4U + steps + step;
    program.operations.push_back(
        {static_cast<std::uint32_t>(step + 1U), Opcode::FlowEulerStep,
         {static_cast<std::uint32_t>(input),
          static_cast<std::uint32_t>(velocity), 2U, 3U},
         {static_cast<std::uint32_t>(output)},
         {Attribute::u64(AttrKey::StepIndex, step)}});
  }
  return program;
}

dif::ir::Program make_patchify_3d(std::uint64_t batch,
                                  std::uint64_t channels,
                                  std::uint64_t frames,
                                  std::uint64_t height,
                                  std::uint64_t width,
                                  std::uint64_t patch_t,
                                  std::uint64_t patch_h,
                                  std::uint64_t patch_w,
                                  dif::ir::DType dtype, bool inverse) {
  using namespace dif::ir;
  if (batch == 0U || channels == 0U || frames == 0U || height == 0U ||
      width == 0U || patch_t == 0U || patch_h == 0U || patch_w == 0U ||
      frames % patch_t != 0U || height % patch_h != 0U ||
      width % patch_w != 0U)
    dif::fail("patchify geometry must be nonzero and patch-divisible");
  const auto rows = checked_multiply(
      checked_multiply(
          checked_multiply(batch, frames / patch_t, "patchify rows"),
          height / patch_h, "patchify rows"),
      width / patch_w, "patchify rows");
  const auto columns = checked_multiply(
      checked_multiply(checked_multiply(channels, patch_t, "patchify columns"),
                       patch_h, "patchify columns"),
      patch_w, "patchify columns");
  const auto volume_dims =
      std::vector<std::uint64_t>{batch, channels, frames, height, width};
  const auto row_dims = std::vector<std::uint64_t>{rows, columns};
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, inverse ? row_dims : volume_dims},
      {2, dtype, TensorRole::Output, inverse ? volume_dims : row_dims},
  };
  program.operations = {
      {1, inverse ? Opcode::Unpatchify3D : Opcode::Patchify3D, {1}, {2},
       {Attribute::u64(AttrKey::PatchT, patch_t),
        Attribute::u64(AttrKey::PatchH, patch_h),
        Attribute::u64(AttrKey::PatchW, patch_w)}}};
  return program;
}

dif::ir::Program make_row_pack(std::uint64_t sequence,
                              std::uint64_t text_rows,
                              std::uint64_t video_rows,
                              std::uint64_t audio_rows,
                              std::uint64_t width,
                              dif::ir::DType dtype) {
  using namespace dif::ir;
  if (sequence == 0U || text_rows == 0U || video_rows == 0U ||
      audio_rows == 0U || width == 0U ||
      checked_add(checked_add(text_rows, video_rows, "row pack rows"),
                  audio_rows, "row pack rows") != sequence)
    dif::fail("row pack requires nonzero modality rows summing to sequence");
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, {text_rows, width}},
      {2, dtype, TensorRole::Input, {video_rows, width}},
      {3, dtype, TensorRole::Input, {audio_rows, width}},
      {4, DType::I32, TensorRole::Input, {sequence}},
      {5, DType::I32, TensorRole::Input, {sequence}},
      {6, DType::I32, TensorRole::Input, {sequence}},
      {7, dtype, TensorRole::Internal, {sequence, width}},
      {8, dtype, TensorRole::Internal, {sequence, width}},
      {9, dtype, TensorRole::Internal, {sequence, width}},
      {10, dtype, TensorRole::Output, {sequence, width}},
  };
  program.operations = {
      {1, Opcode::Fill, {}, {7}, {Attribute::f64(AttrKey::Value, 0.0)}},
      {2, Opcode::IndexedUpdateRows, {7, 1, 4}, {8}, {}},
      {3, Opcode::IndexedUpdateRows, {8, 2, 5}, {9}, {}},
      {4, Opcode::IndexedUpdateRows, {9, 3, 6}, {10}, {}},
  };
  return program;
}

dif::ir::Program make_h3_attention(std::uint64_t sequence, std::uint64_t heads,
                                   std::uint64_t dim, std::uint64_t rotary,
                                   std::uint64_t block) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      tensor(1, TensorRole::Input, {sequence, heads, dim}),
      tensor(2, TensorRole::Input, {sequence, heads, dim}),
      tensor(3, TensorRole::Input, {sequence, heads, dim}),
      tensor(4, TensorRole::Constant, {dim}),
      tensor(5, TensorRole::Constant, {dim}),
      tensor(6, TensorRole::Constant, {sequence, rotary / 2U}),
      tensor(7, TensorRole::Constant, {sequence, rotary / 2U}),
      tensor(8, TensorRole::Internal, {sequence, heads, dim}),
      tensor(9, TensorRole::Internal, {sequence, heads, dim}),
      tensor(10, TensorRole::Output, {sequence, heads, dim}),
  };
  const auto norm_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, dim),
      Attribute::u64(AttrKey::RotaryDim, rotary),
      Attribute::u64(AttrKey::BlockSize, block),
  };
  program.operations = {
      {1, Opcode::QkNormPartialRope, {1, 4, 6, 7}, {8}, norm_attrs},
      {2, Opcode::QkNormPartialRope, {2, 5, 6, 7}, {9}, norm_attrs},
      {3,
       Opcode::Attention,
       {8, 9, 3},
       {10},
       {Attribute::f64(AttrKey::AttentionScale,
                       1.0 / std::sqrt(static_cast<double>(dim))),
        Attribute::boolean(AttrKey::Causal, false),
        Attribute::u64(AttrKey::BlockSize, 64)}},
  };
  return program;
}

dif::ir::Program make_h3_block(std::uint64_t sequence, std::uint64_t hidden,
                               std::uint64_t heads, std::uint64_t dim,
                               std::uint64_t ffn, std::uint64_t rotary,
                               std::uint64_t block, dif::ir::DType dtype) {
  using namespace dif::ir;
  const auto inner = heads * dim;
  Program program;
  program.tensors = {
      tensor(1, TensorRole::Input, {sequence, hidden}),
      tensor(2, TensorRole::Input, {sequence, hidden}),
      tensor(3, TensorRole::Input, {sequence, hidden}),
      tensor(4, TensorRole::Input, {sequence, hidden}),
      tensor(5, TensorRole::Constant, {inner, hidden}),
      tensor(6, TensorRole::Constant, {inner, hidden}),
      tensor(7, TensorRole::Constant, {inner, hidden}),
      tensor(8, TensorRole::Constant, {dim}),
      tensor(9, TensorRole::Constant, {dim}),
      // Released MiniMax-H3 materializes both halves of the partial-RoPE
      // cosine/sine table. Imported graphs may still use compact tables.
      tensor(10, TensorRole::Constant, {sequence, rotary}),
      tensor(11, TensorRole::Constant, {sequence, rotary}),
      tensor(12, TensorRole::Constant, {hidden, inner}),
      tensor(13, TensorRole::Input, {sequence, hidden}),
      tensor(14, TensorRole::Input, {sequence, hidden}),
      tensor(15, TensorRole::Input, {sequence, hidden}),
      tensor(16, TensorRole::Constant, {2U * ffn, hidden}),
      tensor(17, TensorRole::Constant, {hidden, ffn}),
      tensor(18, TensorRole::Constant, {hidden}),
      tensor(19, TensorRole::Constant, {hidden}),
      tensor(20, TensorRole::Output, {sequence, hidden}),
      tensor(21, TensorRole::Internal, {sequence, heads, dim}),
      tensor(22, TensorRole::Internal, {sequence, heads, dim}),
      tensor(23, TensorRole::Internal, {sequence, heads, dim}),
      tensor(24, TensorRole::Output, {sequence, heads, dim}),
      tensor(25, TensorRole::Output, {sequence, heads, dim}),
      tensor(26, TensorRole::Output, {sequence, heads, dim}),
      tensor(27, TensorRole::Internal, {sequence, hidden}),
      tensor(28, TensorRole::Output, {sequence, hidden}),
      tensor(29, TensorRole::Output, {sequence, hidden}),
      tensor(30, TensorRole::Internal, {sequence, 2U * ffn}),
      tensor(31, TensorRole::Output, {sequence, ffn}),
      tensor(32, TensorRole::Output, {sequence, hidden}),
      tensor(33, TensorRole::Output, {sequence, hidden}),
  };
  for (auto &description : program.tensors)
    description.dtype = dtype;
  const auto rms_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::BlockSize, block),
  };
  const auto qk_attrs = std::vector<Attribute>{
      Attribute::f64(AttrKey::Epsilon, 1.0e-5),
      Attribute::u64(AttrKey::Heads, heads),
      Attribute::u64(AttrKey::HeadDim, dim),
      Attribute::u64(AttrKey::RotaryDim, rotary),
      Attribute::u64(AttrKey::BlockSize, block),
  };
  const auto linear_attrs = std::vector<Attribute>{
      Attribute::u64(AttrKey::BlockSize, block),
      Attribute::u64(AttrKey::Implementation, 1U),
  };
  program.operations = {
      {1, Opcode::RmsNormModulate, {1, 18, 2, 3}, {20}, rms_attrs},
      {2, Opcode::Linear, {20, 5}, {21}, linear_attrs},
      {3, Opcode::Linear, {20, 6}, {22}, linear_attrs},
      {4, Opcode::Linear, {20, 7}, {23}, linear_attrs},
      {5, Opcode::QkNormPartialRope, {21, 8, 10, 11}, {24}, qk_attrs},
      {6, Opcode::QkNormPartialRope, {22, 9, 10, 11}, {25}, qk_attrs},
      {7,
       Opcode::Attention,
       {24, 25, 23},
       {26},
       {Attribute::f64(AttrKey::AttentionScale,
                       1.0 / std::sqrt(static_cast<double>(dim))),
        Attribute::boolean(AttrKey::Causal, false),
        Attribute::u64(AttrKey::BlockSize, 64)}},
      {8, Opcode::Linear, {26, 12}, {27}, linear_attrs},
      {9, Opcode::ResidualGate, {1, 27, 4}, {28}, linear_attrs},
      {10, Opcode::RmsNormModulate, {28, 19, 13, 14}, {29}, rms_attrs},
      {11, Opcode::Linear, {29, 16}, {30}, linear_attrs},
      {12, Opcode::SwiGlu, {30}, {31}, linear_attrs},
      {13, Opcode::Linear, {31, 17}, {32}, linear_attrs},
      {14, Opcode::ResidualGate, {28, 32, 15}, {33}, linear_attrs},
  };
  return program;
}

void usage() {
  std::cerr << "usage:\n"
            << "  difc make-rms OUT.difir ROWS COLS BLOCK\n"
            << "  difc make-linear-blend OUT.difir ROWS COLS f32|bf16|f16\n"
            << "  difc make-residual-gate OUT.difir ROWS COLS BLOCK f32|bf16|f16\n"
            << "  difc make-flow-euler-trajectory OUT.difir ROWS COLS STEPS f32|bf16|f16\n"
            << "  difc make-patchify3d OUT.difir B C T H W PT PH PW f32|bf16|f16\n"
            << "  difc make-unpatchify3d OUT.difir B C T H W PT PH PW f32|bf16|f16\n"
            << "  difc make-row-pack OUT.difir SEQ TEXT VIDEO AUDIO WIDTH f32|bf16|f16\n"
            << "  difc make-h3-attention OUT.difir S HEADS DIM ROTARY BLOCK\n"
            << "  difc make-h3-block OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK\n"
            << "  difc make-h3-block-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK\n"
            << "  difc make-h3-stack-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY LAYERS BLOCK resident|streamed\n"
            << "  difc make-h3-transformer-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY LAYERS TABLES TIME_EMBED BLOCK resident|streamed [split|packed]\n"
            << "  difc make-h3-token-refiner-bf16 OUT.difir S HIDDEN HEADS DIM FFN LAYERS BLOCK resident|streamed\n"
            << "  difc make-h3-denoiser OUT.difir VIDEO_TOKENS AUDIO_TOKENS TEXT_TOKENS TIMESTEP_TABLES resident|streamed generated|cudnn [LAYERS REFINER_LAYERS]\n"
            << "  difc make-h3-video-vae OUT.difir LATENT_T LATENT_H LATENT_W LAYERS resident|streamed generated|cudnn\n"
            << "  difc make-mlp-training OUT.difir ROWS INPUT_WIDTH HIDDEN_WIDTH OUTPUT_WIDTH [LR BETA1 BETA2 EPS WEIGHT_DECAY]\n"
            << "  difc make-rectified-flow-training OUT.difir ROWS LATENT_WIDTH TIMESTEP_WIDTH HIDDEN_WIDTH ACCUMULATION_STEPS [LR BETA1 BETA2 EPS WEIGHT_DECAY]\n"
            << "  difc make-h3-block-raw-bf16 OUT.difir S HIDDEN HEADS DIM FFN ROTARY BLOCK resident|streamed\n"
            << "  difc set-linear-math IN.difir OUT.difir "
               "strict|tf32|direct-int5\n"
            << "  difc set-attention-implementation IN.difir OUT.difir generated|cudnn\n"
            << "  difc set-constant-residency IN.difir OUT.difir resident|streamed\n"
            << "  difc expose-tensors IN.difir OUT.difir ID [ID ...]\n"
            << "  difc verify FILE.difir\n"
            << "  difc fingerprint FILE.difir\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 3) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "make-rms" && argc == 6) {
      const auto program = make_rms(number(argv[3], "rows"), number(argv[4], "cols"),
                                    number(argv[5], "block size"));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(program))
                << "\n";
      return 0;
    }
    if (command == "make-linear-blend" && argc == 6) {
      const auto program = make_linear_blend(
          number(argv[3], "rows"), number(argv[4], "columns"),
          float_dtype(argv[5]));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " operation=linear_blend dtype=" << argv[5]
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-residual-gate" && argc == 7) {
      const auto program = make_residual_gate(
          number(argv[3], "rows"), number(argv[4], "columns"),
          number(argv[5], "block size"), float_dtype(argv[6]));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " operation=residual_gate dtype=" << argv[6]
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-flow-euler-trajectory" && argc == 7) {
      const auto program = make_flow_euler_trajectory(
          number(argv[3], "rows"), number(argv[4], "columns"),
          number(argv[5], "steps"), float_dtype(argv[6]));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " operation=flow_euler_trajectory steps=" << argv[5]
                << " dtype=" << argv[6] << " velocity=data-ward fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if ((command == "make-patchify3d" || command == "make-unpatchify3d") &&
        argc == 12) {
      const auto program = make_patchify_3d(
          number(argv[3], "batch"), number(argv[4], "channels"),
          number(argv[5], "frames"), number(argv[6], "height"),
          number(argv[7], "width"), number(argv[8], "patch t"),
          number(argv[9], "patch h"), number(argv[10], "patch w"),
          float_dtype(argv[11]), command == "make-unpatchify3d");
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2] << " operation="
                << (command == "make-unpatchify3d" ? "unpatchify_3d"
                                                     : "patchify_3d")
                << " dtype=" << argv[11] << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-row-pack" && argc == 9) {
      const auto program = make_row_pack(
          number(argv[3], "sequence"), number(argv[4], "text rows"),
          number(argv[5], "video rows"), number(argv[6], "audio rows"),
          number(argv[7], "width"), float_dtype(argv[8]));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " operation=row_pack dtype=" << argv[8]
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-h3-attention" && argc == 8) {
      const auto program = make_h3_attention(
          number(argv[3], "sequence"), number(argv[4], "heads"),
          number(argv[5], "head dimension"), number(argv[6], "rotary dimension"),
          number(argv[7], "block size"));
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(program))
                << "\n";
      return 0;
    }
    if ((command == "make-h3-block" || command == "make-h3-block-bf16") &&
        argc == 10) {
      const auto program = make_h3_block(
          number(argv[3], "sequence"), number(argv[4], "hidden"),
          number(argv[5], "heads"), number(argv[6], "head dimension"),
          number(argv[7], "ffn"), number(argv[8], "rotary dimension"),
          number(argv[9], "block size"),
          command == "make-h3-block-bf16" ? dif::ir::DType::BF16
                                           : dif::ir::DType::F32);
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(program))
                << "\n";
      return 0;
    }
    if (command == "make-h3-stack-bf16" && argc == 12) {
      const std::string residency = argv[11];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 stack residency must be resident or streamed");
      const auto program = dif::frontend::make_h3_stack_bf16(
          number(argv[3], "sequence"), number(argv[4], "hidden"),
          number(argv[5], "heads"), number(argv[6], "head dimension"),
          number(argv[7], "ffn"), number(argv[8], "rotary dimension"),
          number(argv[9], "layers"), number(argv[10], "block size"),
          residency == "streamed");
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2] << " layers=" << argv[9]
                << " constant_residency=" << residency << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-h3-transformer-bf16" &&
        (argc == 14 || argc == 15)) {
      const std::string residency = argv[13];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 transformer residency must be resident or streamed");
      const std::string qkv = argc == 15 ? argv[14] : "split";
      if (qkv != "split" && qkv != "packed")
        dif::fail("H3 transformer QKV mode must be split or packed");
      const auto program = dif::frontend::make_h3_transformer_bf16(
          number(argv[3], "sequence"), number(argv[4], "hidden"),
          number(argv[5], "heads"), number(argv[6], "head dimension"),
          number(argv[7], "ffn"), number(argv[8], "rotary dimension"),
          number(argv[9], "layers"), number(argv[10], "timestep tables"),
          number(argv[11], "time embedding dimension"),
          number(argv[12], "block size"), residency == "streamed",
          qkv == "split");
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2] << " layers=" << argv[9]
                << " constant_residency=" << residency << " qkv=" << qkv
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-h3-token-refiner-bf16" && argc == 11) {
      const std::string residency = argv[10];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 token-refiner residency must be resident or streamed");
      const auto program = dif::frontend::make_h3_token_refiner_bf16(
          number(argv[3], "sequence"), number(argv[4], "hidden"),
          number(argv[5], "heads"), number(argv[6], "head dimension"),
          number(argv[7], "ffn"), number(argv[8], "layers"),
          number(argv[9], "block size"), residency == "streamed");
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2] << " layers=" << argv[8]
                << " constant_residency=" << residency << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-h3-denoiser" && (argc == 9 || argc == 11)) {
      const std::string residency = argv[7];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 denoiser residency must be resident or streamed");
      const std::string attention = argv[8];
      if (attention != "generated" && attention != "cudnn")
        dif::fail("H3 denoiser attention must be generated or cudnn");
      dif::frontend::H3DenoiserConfig config;
      config.video_tokens = number(argv[3], "video tokens");
      config.audio_tokens = number(argv[4], "audio tokens");
      config.text_tokens = number(argv[5], "text tokens");
      config.timestep_tables = number(argv[6], "timestep tables");
      config.streamed_constants = residency == "streamed";
      config.attention_implementation = attention == "cudnn" ? 2U : 1U;
      if (argc == 11) {
        config.layers = number(argv[9], "layers");
        config.refiner_layers = number(argv[10], "refiner layers");
      }
      const auto program = dif::frontend::make_h3_denoiser(config);
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " video_tokens=" << config.video_tokens
                << " audio_tokens=" << config.audio_tokens
                << " text_tokens=" << config.text_tokens
                << " layers=" << config.layers
                << " refiner_layers=" << config.refiner_layers
                << " constant_residency=" << residency
                << " attention=" << attention << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "make-h3-video-vae" && argc == 9) {
      const std::string residency = argv[7];
      if (residency != "resident" && residency != "streamed")
        dif::fail("H3 video VAE residency must be resident or streamed");
      const std::string attention = argv[8];
      if (attention != "generated" && attention != "cudnn")
        dif::fail("H3 video VAE attention must be generated or cudnn");
      dif::frontend::H3VideoVaeConfig config;
      config.latent_frames = number(argv[3], "latent frames");
      config.latent_height = number(argv[4], "latent height");
      config.latent_width = number(argv[5], "latent width");
      config.layers = number(argv[6], "layers");
      config.streamed_constants = residency == "streamed";
      config.attention_implementation = attention == "cudnn" ? 2U : 1U;
      const auto build = dif::frontend::make_h3_video_vae_decoder(config);
      dif::ir::write_file(build.program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " latent=" << config.latent_frames << "x"
                << config.latent_height << "x" << config.latent_width
                << " layers=" << config.layers
                << " constant_residency=" << residency
                << " attention=" << attention
                << " raw_output=" << build.raw_output_id
                << " decoded_output=" << build.decoded_output_id
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(build.program))
                << "\n";
      return 0;
    }
    if (command == "make-mlp-training" && (argc == 7 || argc == 12)) {
      dif::frontend::MlpTrainingConfig config;
      config.rows = number(argv[3], "rows");
      config.input_width = number(argv[4], "input width");
      config.hidden_width = number(argv[5], "hidden width");
      config.output_width = number(argv[6], "output width");
      if (argc == 12) {
        config.learning_rate = floating_number(argv[7], "learning rate");
        config.beta1 = floating_number(argv[8], "beta1");
        config.beta2 = floating_number(argv[9], "beta2");
        config.epsilon = floating_number(argv[10], "epsilon");
        config.weight_decay = floating_number(argv[11], "weight decay");
      }
      const auto build = dif::frontend::make_mlp_training(config);
      dif::ir::write_file(build.program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " rows=" << config.rows
                << " input_width=" << config.input_width
                << " hidden_width=" << config.hidden_width
                << " output_width=" << config.output_width
                << " features_input=" << build.features_input
                << " target_input=" << build.target_input
                << " step_input=" << build.step_input
                << " prediction_output=" << build.prediction_output
                << " loss_output=" << build.loss_output
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(build.program))
                << "\n";
      for (const auto &binding : build.optimizer_bindings)
        std::cout << "OPTIMIZER parameter_input=" << binding.parameter_input
                  << " gradient_output=" << binding.gradient_output
                  << " first_moment_input=" << binding.first_moment_input
                  << " second_moment_input=" << binding.second_moment_input
                  << " parameter_output=" << binding.parameter_output
                  << " first_moment_output=" << binding.first_moment_output
                  << " second_moment_output=" << binding.second_moment_output
                  << "\n";
      return 0;
    }
    if (command == "make-rectified-flow-training" &&
        (argc == 8 || argc == 13)) {
      dif::frontend::RectifiedFlowTrainingConfig config;
      config.rows = number(argv[3], "rows");
      config.latent_width = number(argv[4], "latent width");
      config.timestep_width = number(argv[5], "timestep width");
      config.hidden_width = number(argv[6], "hidden width");
      config.accumulation_steps = number(argv[7], "accumulation steps");
      if (argc == 13) {
        config.learning_rate = floating_number(argv[8], "learning rate");
        config.beta1 = floating_number(argv[9], "beta1");
        config.beta2 = floating_number(argv[10], "beta2");
        config.epsilon = floating_number(argv[11], "epsilon");
        config.weight_decay = floating_number(argv[12], "weight decay");
      }
      const auto build = dif::frontend::make_rectified_flow_training(config);
      dif::ir::write_file(build.program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " objective=rectified-flow-data-ward"
                << " rows=" << config.rows
                << " latent_width=" << config.latent_width
                << " timestep_width=" << config.timestep_width
                << " hidden_width=" << config.hidden_width
                << " accumulation_steps=" << config.accumulation_steps
                << " loss_scale_tensor=" << build.loss_scale_tensor
                << " step_input=" << build.step_input
                << " loss_output=" << build.loss_output
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(build.program))
                << "\n";
      for (std::size_t index = 0U; index < build.microbatches.size(); ++index) {
        const auto &binding = build.microbatches[index];
        std::cout << "MICROBATCH index=" << index
                  << " clean_input=" << binding.clean_input
                  << " noise_input=" << binding.noise_input
                  << " clean_scale_input=" << binding.clean_scale_input
                  << " noise_scale_input=" << binding.noise_scale_input
                  << " timestep_features_input="
                  << binding.timestep_features_input
                  << " target_velocity_input="
                  << binding.target_velocity_input
                  << " prediction_output=" << binding.prediction_output
                  << " loss_tensor=" << binding.loss_tensor << "\n";
      }
      for (const auto &binding : build.optimizer_bindings)
        std::cout << "OPTIMIZER parameter_input=" << binding.parameter_input
                  << " gradient_output=" << binding.gradient_output
                  << " first_moment_input=" << binding.first_moment_input
                  << " second_moment_input=" << binding.second_moment_input
                  << " parameter_output=" << binding.parameter_output
                  << " first_moment_output=" << binding.first_moment_output
                  << " second_moment_output=" << binding.second_moment_output
                  << "\n";
      return 0;
    }
    if (command == "make-h3-block-raw-bf16" && argc == 11) {
      const std::string residency = argv[10];
      if (residency != "resident" && residency != "streamed")
        dif::fail("raw H3 block residency must be resident or streamed");
      const auto program = dif::frontend::make_h3_block_raw_bf16(
          number(argv[3], "sequence"), number(argv[4], "hidden"),
          number(argv[5], "heads"), number(argv[6], "head dimension"),
          number(argv[7], "ffn"), number(argv[8], "rotary dimension"),
          number(argv[9], "block size"), residency == "streamed");
      dif::ir::write_file(program, argv[2]);
      std::cout << "PROGRAM path=" << argv[2]
                << " checkpoint_layout=raw constant_residency=" << residency
                << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "set-linear-math" && argc == 5) {
      auto program = dif::ir::read_file(argv[2]);
      const std::string mode = argv[4];
      const std::uint64_t implementation =
          mode == "strict"        ? 1U
          : mode == "tf32"        ? 2U
          : mode == "direct-int5" ? 3U
                                     : 0U;
      if (implementation == 0U)
        dif::fail("linear math mode must be strict, tf32, or direct-int5");
      std::size_t changed = 0;
      for (auto &operation : program.operations) {
        if (operation.opcode != dif::ir::Opcode::Linear)
          continue;
        auto *attribute = const_cast<dif::ir::Attribute *>(
            operation.find(dif::ir::AttrKey::Implementation));
        if (attribute)
          *attribute = dif::ir::Attribute::u64(
              dif::ir::AttrKey::Implementation, implementation);
        else
          operation.attributes.push_back(dif::ir::Attribute::u64(
              dif::ir::AttrKey::Implementation, implementation));
        ++changed;
      }
      if (changed == 0U)
        dif::fail("program contains no Linear operations");
      dif::ir::verify(program);
      dif::ir::write_file(program, argv[3]);
      std::cout << "PROGRAM path=" << argv[3] << " linear_math=" << mode
                << " linear_ops=" << changed << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "set-attention-implementation" && argc == 5) {
      auto program = dif::ir::read_file(argv[2]);
      const std::string mode = argv[4];
      const std::uint64_t implementation =
          mode == "generated" ? 1U : mode == "cudnn" ? 2U : 0U;
      if (implementation == 0U)
        dif::fail("attention implementation must be generated or cudnn");
      std::size_t changed = 0;
      for (auto &operation : program.operations) {
        if (operation.opcode != dif::ir::Opcode::Attention)
          continue;
        auto *attribute = const_cast<dif::ir::Attribute *>(
            operation.find(dif::ir::AttrKey::Implementation));
        if (attribute)
          *attribute = dif::ir::Attribute::u64(
              dif::ir::AttrKey::Implementation, implementation);
        else
          operation.attributes.push_back(dif::ir::Attribute::u64(
              dif::ir::AttrKey::Implementation, implementation));
        ++changed;
      }
      if (changed == 0U)
        dif::fail("program contains no Attention operations");
      dif::ir::verify(program);
      dif::ir::write_file(program, argv[3]);
      std::cout << "PROGRAM path=" << argv[3]
                << " attention_implementation=" << mode
                << " attention_ops=" << changed << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "set-constant-residency" && argc == 5) {
      auto program = dif::ir::read_file(argv[2]);
      const std::string mode = argv[4];
      if (mode != "resident" && mode != "streamed")
        dif::fail("constant residency must be resident or streamed");
      std::size_t changed = 0;
      for (auto &description : program.tensors) {
        if (!description.has_role(dif::ir::TensorRole::Constant))
          continue;
        if (mode == "streamed")
          description.roles |= dif::ir::TensorRole::Streamed;
        else
          description.roles &= ~dif::ir::TensorRole::Streamed;
        ++changed;
      }
      if (changed == 0U)
        dif::fail("program contains no constants");
      dif::ir::verify(program);
      dif::ir::write_file(program, argv[3]);
      std::cout << "PROGRAM path=" << argv[3] << " constant_residency=" << mode
                << " constants=" << changed << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "expose-tensors" && argc >= 5) {
      auto program = dif::ir::read_file(argv[2]);
      for (int argument = 4; argument < argc; ++argument) {
        const auto raw_id = number(argv[argument], "tensor id");
        if (raw_id > UINT32_MAX)
          dif::fail("tensor id is outside the DiffIR range");
        const auto id = static_cast<std::uint32_t>(raw_id);
        auto found = false;
        for (auto &description : program.tensors) {
          if (description.id != id)
            continue;
          description.roles |= dif::ir::TensorRole::Output;
          found = true;
          break;
        }
        if (!found)
          dif::fail("unknown tensor id " + std::to_string(id));
      }
      dif::ir::verify(program);
      dif::ir::write_file(program, argv[3]);
      std::cout << "PROGRAM path=" << argv[3]
                << " exposed_tensors=" << (argc - 4) << " fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "verify" && argc == 3) {
      const auto program = dif::ir::read_file(argv[2]);
      dif::ir::verify(program);
      std::cout << "VERIFY PASS fingerprint="
                << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    if (command == "fingerprint" && argc == 3) {
      const auto program = dif::ir::read_file(argv[2]);
      std::cout << dif::hex_digest(dif::ir::fingerprint(program)) << "\n";
      return 0;
    }
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "difc: " << error.what() << "\n";
    return 1;
  }
}
