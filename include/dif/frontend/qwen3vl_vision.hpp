#pragma once

#include "dif/ir/ir.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/png.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dif::frontend {

// Shared Qwen3-VL vision-tower semantics. MiniMax-H3 supplies the released
// dimensions below; other Qwen3-VL frontends can provide another config
// without changing the runtime or backend.
struct Qwen3VlVisionConfig {
  std::uint64_t depth{27U};
  std::uint64_t hidden_size{1152U};
  std::uint64_t attention_heads{16U};
  std::uint64_t intermediate_size{4304U};
  std::uint64_t output_hidden_size{5120U};
  std::uint64_t patch_size{16U};
  std::uint64_t temporal_patch_size{2U};
  std::uint64_t spatial_merge_size{2U};
  std::uint64_t position_grid_side{48U};
  double layer_norm_epsilon{1.0e-6};
  // 1: compiler-generated exact attention, 2: exact cuDNN SDPA.
  std::uint64_t attention_implementation{2U};
  std::vector<std::uint64_t> deepstack_tap_blocks{8U, 16U, 24U};
  bool trace_outputs{};
};

struct Qwen3VlVisionBinding {
  std::uint32_t tensor_id{};
  std::string name;
};

struct Qwen3VlVisionBuild {
  ir::Program program;
  std::vector<Qwen3VlVisionBinding> bindings;
  runtime::TensorMap generated_constants;
  std::uint32_t pixel_patches_input_id{};
  std::uint32_t position_embeddings_input_id{};
  std::uint32_t embeds_output_id{};
  std::vector<std::uint32_t> deepstack_output_ids;
  std::vector<std::uint32_t> trace_output_ids;
  std::uint64_t linear_operations{};
  std::uint64_t attention_operations{};
};

Qwen3VlVisionBuild build_qwen3vl_vision_program(
    std::uint64_t grid_t, std::uint64_t grid_h, std::uint64_t grid_w,
    const Qwen3VlVisionConfig &config = {});

// The official image processor boundary: RGB8 canvas -> BF16 patch rows in
// merge-block order, each row laid out (channel, temporal, patch_h, patch_w).
runtime::Tensor qwen3vl_vision_image_patch_rows(
    const RgbImage &image, const Qwen3VlVisionConfig &config = {});

// Creator-faithful bilinear interpolation of the checkpoint's learned square
// position table, followed by the same merge-block permutation as patch rows.
runtime::Tensor qwen3vl_vision_position_embeddings(
    const runtime::Tensor &position_table, std::uint64_t grid_t,
    std::uint64_t grid_h, std::uint64_t grid_w,
    const Qwen3VlVisionConfig &config = {});

} // namespace dif::frontend
