// Byte-identity gate for generated CUDA source.
//
// For a corpus of small programs (one per kernel emitter and attribute
// branch), the CUDA text produced by dif::compiler::emit_cuda must match the
// committed snapshot under perf/regress/fixtures/kernel-sources/ byte for
// byte. The snapshots were taken before kernel bodies moved out of the C++
// emitters into src/compiler/kernels/*.cu templates, so the gate proves the
// move changed nothing the GPU sees (the runtime keys its PTX cache by a hash
// of this text).
//
//   dif_kernel_source_tests FIXTURE_DIR            compare
//   dif_kernel_source_tests FIXTURE_DIR --update   rewrite the snapshots
#include "dif/compiler/compiler.hpp"
#include "dif/ir/ir.hpp"
#include "dif/support/error.hpp"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

using namespace dif::ir;
using dif::fail;

struct Case {
  std::string name;
  std::function<Program()> build;
};

Program elementwise(Opcode opcode, DType dtype,
                    std::vector<Attribute> attributes = {}) {
  Program program;
  program.tensors = {{1, dtype, TensorRole::Input, {4, 8}},
                     {2, dtype, TensorRole::Output, {4, 8}}};
  program.operations = {{1, opcode, {1}, {2}, std::move(attributes)}};
  return program;
}

Program dequantize_int4(bool outliers) {
  Program program;
  program.tensors = {{1, DType::I8, TensorRole::Constant, {4, 64}},
                     {2, DType::BF16, TensorRole::Constant, {4, 2}},
                     {3, DType::BF16, TensorRole::Output, {4, 128}}};
  std::vector<std::uint32_t> inputs{1, 2};
  if (outliers) {
    program.tensors.push_back({4, DType::I8, TensorRole::Constant, {4, 2}});
    program.tensors.push_back({5, DType::BF16, TensorRole::Constant, {4, 2}});
    inputs = {1, 2, 4, 5};
  }
  program.operations = {{1, Opcode::DequantizeInt4, inputs, {3},
                         {Attribute::u64(AttrKey::GroupSize, 64U)}}};
  return program;
}

Program dequantize_int5(bool column_scales) {
  Program program;
  program.tensors = {{1, DType::I8, TensorRole::Constant, {4, 80}},
                     {2, DType::BF16, TensorRole::Constant, {4, 2}},
                     {3, DType::BF16, TensorRole::Output, {4, 128}}};
  std::vector<std::uint32_t> inputs{1, 2};
  if (column_scales) {
    program.tensors.push_back({4, DType::BF16, TensorRole::Constant, {128}});
    inputs = {1, 2, 4};
  }
  program.operations = {{1, Opcode::DequantizeInt5, inputs, {3},
                         {Attribute::u64(AttrKey::GroupSize, 64U)}}};
  return program;
}

Program rms_norm(std::uint64_t implementation, std::uint64_t columns,
                 std::uint64_t block = 0U, std::uint64_t reduction_tile = 0U,
                 double weight_offset = 0.0) {
  Program program;
  program.tensors = {{1, DType::BF16, TensorRole::Input, {4, columns}},
                     {2, DType::BF16, TensorRole::Constant, {columns}},
                     {3, DType::BF16, TensorRole::Output, {4, columns}}};
  std::vector<Attribute> attributes{Attribute::f64(AttrKey::Epsilon, 1.0e-6)};
  if (implementation != 1U)
    attributes.push_back(
        Attribute::u64(AttrKey::Implementation, implementation));
  if (block != 0U)
    attributes.push_back(Attribute::u64(AttrKey::BlockSize, block));
  if (reduction_tile != 0U)
    attributes.push_back(
        Attribute::u64(AttrKey::ReductionTileSize, reduction_tile));
  if (weight_offset != 0.0)
    attributes.push_back(Attribute::f64(AttrKey::WeightOffset, weight_offset));
  program.operations = {{1, Opcode::RmsNorm, {1, 2}, {3}, attributes}};
  return program;
}

Program quantize_int8_rows(Int8RowQuantization implementation,
                           std::uint64_t width, bool residual2,
                           bool dynamic_clip) {
  Program program;
  program.tensors = {{1, DType::BF16, TensorRole::Input, {2, width}},
                     {2, DType::I8, TensorRole::Output, {2, width}},
                     {3, DType::F32, TensorRole::Output, {2}}};
  std::vector<std::uint32_t> inputs{1};
  std::vector<std::uint32_t> outputs{2, 3};
  if (dynamic_clip) {
    program.tensors.push_back({4, DType::F32, TensorRole::Input, {1}});
    inputs.push_back(4);
  }
  if (residual2) {
    program.tensors.push_back({5, DType::I8, TensorRole::Output, {2, width}});
    program.tensors.push_back({6, DType::F32, TensorRole::Output, {2}});
    outputs = {2, 3, 5, 6};
  }
  program.operations = {
      {1, Opcode::QuantizeInt8Rows, inputs, outputs,
       {Attribute::u64(AttrKey::BlockSize, 256U),
        Attribute::u64(AttrKey::Implementation,
                       static_cast<std::uint64_t>(implementation))}}};
  return program;
}


Program unary(Opcode opcode, DType dtype, std::vector<Attribute> attributes = {}) {
  return elementwise(opcode, dtype, std::move(attributes));
}

Program binary(Opcode opcode, DType dtype) {
  Program program;
  program.tensors = {{1, dtype, TensorRole::Input, {4, 8}},
                     {2, dtype, TensorRole::Input, {4, 8}},
                     {3, dtype, TensorRole::Output, {4, 8}}};
  program.operations = {{1, opcode, {1, 2}, {3}, {}}};
  return program;
}

Attribute accumulate_f32() {
  return Attribute::u64(AttrKey::AccumulatorDType,
                        static_cast<std::uint64_t>(DType::F32));
}

Program with_vector(Opcode opcode, std::uint64_t vectors, bool bias_vector) {
  // x [4,8] plus `vectors` [8] inputs (scale/bias-style), output [4,8].
  Program program;
  program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}}};
  std::vector<std::uint32_t> inputs{1};
  for (std::uint64_t v = 0U; v < vectors; ++v) {
    const auto id = static_cast<std::uint32_t>(2U + v);
    program.tensors.push_back({id, DType::BF16, TensorRole::Constant, {8}});
    inputs.push_back(id);
  }
  (void)bias_vector;
  const auto out = static_cast<std::uint32_t>(2U + vectors);
  program.tensors.push_back({out, DType::BF16, TensorRole::Output, {4, 8}});
  program.operations = {{1, opcode, inputs, {out}, {}}};
  return program;
}

Program batch3_program(std::string_view which) {
  Program program;
  if (which == "silu_backward") {
    program = binary(Opcode::SiLUBackward, DType::BF16);
    program.operations[0].attributes = {accumulate_f32()};
  } else if (which == "residual_gate_backward") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Input, {4, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::ResidualGateBackward, {1, 2, 3}, {4, 5},
                           {accumulate_f32()}}};
  } else if (which == "bias_backward") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Output, {8}}};
    program.operations = {{1, Opcode::BiasBackward, {1}, {2}, {accumulate_f32()}}};
  } else if (which == "euler_velocity_step") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::F32, TensorRole::Input, {1}},
                       {4, DType::F32, TensorRole::Input, {1}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::EulerVelocityStep, {1, 2, 3, 4}, {5}, {}}};
  } else if (which == "linear_blend") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::F32, TensorRole::Input, {1}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::LinearBlend, {1, 2, 3}, {4}, {}}};
  } else if (which == "mse_loss") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::F32, TensorRole::Output, {1}}};
    program.operations = {{1, Opcode::MseLoss, {1, 2}, {3}, {accumulate_f32()}}};
  } else if (which == "swiglu" || which == "swiglu_gate_first") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 16}},
                       {2, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::SwiGlu, {1}, {2},
                           {Attribute::boolean(AttrKey::GateFirst,
                                               which == "swiglu_gate_first")}}};
  } else if (which == "residual_gate" || which == "residual_gate_broadcast") {
    const std::uint64_t gate_rows = which == "residual_gate" ? 4U : 1U;
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Input, {gate_rows, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::ResidualGate, {1, 2, 3}, {4}, {}}};
  } else if (which == "flow_euler_step") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::F32, TensorRole::Input, {3}},
                       {4, DType::F32, TensorRole::Input, {4}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::FlowEulerStep, {1, 2, 3, 4}, {5},
                           {Attribute::u64(AttrKey::StepIndex, 1U)}}};
  } else if (which == "snake_beta") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {2, 3, 8}},
                       {2, DType::BF16, TensorRole::Constant, {3}},
                       {3, DType::BF16, TensorRole::Constant, {3}},
                       {4, DType::BF16, TensorRole::Output, {2, 3, 8}}};
    program.operations = {{1, Opcode::SnakeBeta, {1, 2, 3}, {4}, {}}};
  } else if (which == "cast_bf16_f32" || which == "cast_f32_bf16") {
    const auto from = which == "cast_bf16_f32" ? DType::BF16 : DType::F32;
    const auto to = which == "cast_bf16_f32" ? DType::F32 : DType::BF16;
    program.tensors = {{1, from, TensorRole::Input, {4, 8}},
                       {2, to, TensorRole::Output, {4, 8}}};
    program.operations = {{1, Opcode::Cast, {1}, {2}, {}}};
  } else if (which == "reshape") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Output, {8, 4}}};
    program.operations = {{1, Opcode::Reshape, {1}, {2}, {}}};
  } else if (which == "fill" || which == "fill_default") {
    program.tensors = {{1, DType::BF16, TensorRole::Output, {4, 8}}};
    std::vector<Attribute> attributes;
    if (which == "fill")
      attributes.push_back(Attribute::f64(AttrKey::Value, 0.5));
    program.operations = {{1, Opcode::Fill, {}, {1}, attributes}};
  } else {
    fail("unknown batch3 corpus program " + std::string(which));
  }
  return program;
}


Program batch4_program(std::string_view which) {
  Program program;
  auto op = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                std::vector<std::uint32_t> outputs,
                std::vector<Attribute> attributes = {}) {
    program.operations = {{1, opcode, std::move(inputs), std::move(outputs),
                           std::move(attributes)}};
  };
  if (which == "upsample_nearest_2d") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 2, 3, 4}},
                       {2, DType::BF16, TensorRole::Output, {1, 2, 6, 8}}};
    op(Opcode::UpsampleNearest2d, {1}, {2},
       {Attribute::u64(AttrKey::ScaleH, 2U), Attribute::u64(AttrKey::ScaleW, 2U)});
  } else if (which == "broadcast_to") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 8}},
                       {2, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::BroadcastTo, {1}, {2});
  } else if (which == "broadcast_to_rank") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {8}},
                       {2, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::BroadcastTo, {1}, {2});
  } else if (which == "slice") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Output, {4, 3}}};
    op(Opcode::Slice, {1}, {2},
       {Attribute::u64(AttrKey::Axis, 1U), Attribute::u64(AttrKey::Start, 2U)});
  } else if (which == "rotary_frequency" || which == "rotary_frequency_theta") {
    program.tensors = {{1, DType::F32, TensorRole::Input, {1, 4, 2}},
                       {2, DType::I32, TensorRole::Constant, {3}},
                       {3, DType::I32, TensorRole::Constant, {3}},
                       {4, DType::I32, TensorRole::Constant, {2}},
                       {5, DType::F32, TensorRole::Output, {1, 4, 3}},
                       {6, DType::F32, TensorRole::Output, {1, 4, 3}}};
    std::vector<Attribute> attributes;
    if (which == "rotary_frequency_theta")
      attributes = {Attribute::f64(AttrKey::Theta, 1.0e9),
                    Attribute::f64(AttrKey::Ntk, 1.5)};
    op(Opcode::RotaryFrequency, {1, 2, 3, 4}, {5, 6}, attributes);
  } else if (which == "boolean_mask_to_bias" || which == "boolean_mask_to_bias_keys_only") {
    program.tensors = {{1, DType::Bool, TensorRole::Input, {2, 4}},
                       {2, DType::BF16, TensorRole::Output, {2, 1, 4, 4}}};
    std::vector<Attribute> attributes;
    if (which == "boolean_mask_to_bias_keys_only")
      attributes = {Attribute::boolean(AttrKey::MaskQueries, false)};
    op(Opcode::BooleanMaskToBias, {1}, {2}, attributes);
  } else if (which == "boolean_mask_to_bias_matrix") {
    program.tensors = {{1, DType::Bool, TensorRole::Input, {2, 4, 4}},
                       {2, DType::F32, TensorRole::Output, {2, 1, 4, 4}}};
    op(Opcode::BooleanMaskToBias, {1}, {2});
  } else if (which == "gather_rows") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {5, 8}},
                       {2, DType::I32, TensorRole::Input, {3}},
                       {3, DType::BF16, TensorRole::Output, {3, 8}}};
    op(Opcode::GatherRows, {1, 2}, {3});
  } else if (which == "indexed_update_rows") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {5, 8}},
                       {2, DType::BF16, TensorRole::Input, {2, 8}},
                       {3, DType::I32, TensorRole::Input, {5}},
                       {4, DType::BF16, TensorRole::Output, {5, 8}}};
    op(Opcode::IndexedUpdateRows, {1, 2, 3}, {4});
  } else if (which == "select_row_chunks") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {6, 16}},
                       {2, DType::I32, TensorRole::Input, {3}},
                       {3, DType::BF16, TensorRole::Output, {3, 8}},
                       {4, DType::BF16, TensorRole::Output, {3, 8}}};
    op(Opcode::SelectRowChunks, {1, 2}, {3, 4});
  } else if (which == "sinusoidal_timestep" || which == "sinusoidal_timestep_flip") {
    program.tensors = {{1, DType::F32, TensorRole::Input, {4}},
                       {2, DType::F32, TensorRole::Output, {4, 16}}};
    std::vector<Attribute> attributes{Attribute::f64(AttrKey::Scale, 1000.0)};
    if (which == "sinusoidal_timestep_flip")
      attributes.push_back(Attribute::boolean(AttrKey::FlipSinToCos, true));
    op(Opcode::SinusoidalTimestep, {1}, {2}, attributes);
  } else if (which == "rotary_position") {
    program.tensors = {{1, DType::F32, TensorRole::Input, {4, 2}},
                       {2, DType::F32, TensorRole::Constant, {3}},
                       {3, DType::BF16, TensorRole::Output, {4, 12}},
                       {4, DType::BF16, TensorRole::Output, {4, 12}}};
    op(Opcode::RotaryPosition, {1, 2}, {3, 4});
  } else if (which == "permute") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {2, 3, 4}},
                       {2, DType::BF16, TensorRole::Output, {4, 2, 3}}};
    op(Opcode::Permute, {1}, {2},
       {Attribute::u64(AttrKey::Permutation0, 2U),
        Attribute::u64(AttrKey::Permutation1, 0U),
        Attribute::u64(AttrKey::Permutation2, 1U)});
  } else if (which == "concat") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 3}},
                       {2, DType::BF16, TensorRole::Input, {4, 5}},
                       {3, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::Concat, {1, 2}, {3}, {Attribute::u64(AttrKey::Axis, 1U)});
  } else if (which == "concat_axis0_three") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 8}},
                       {2, DType::BF16, TensorRole::Input, {2, 8}},
                       {3, DType::BF16, TensorRole::Input, {1, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::Concat, {1, 2, 3}, {4}, {Attribute::u64(AttrKey::Axis, 0U)});
  } else {
    fail("unknown batch4 corpus program " + std::string(which));
  }
  return program;
}


Program batch5_program(std::string_view which) {
  Program program;
  auto op = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                std::vector<std::uint32_t> outputs,
                std::vector<Attribute> attributes = {}) {
    program.operations = {{1, opcode, std::move(inputs), std::move(outputs),
                           std::move(attributes)}};
  };
  if (which == "dequantize_int8_blocks") {
    program.tensors = {{1, DType::I8, TensorRole::Constant, {4, 64}},
                       {2, DType::F32, TensorRole::Constant, {4, 2}},
                       {3, DType::BF16, TensorRole::Output, {4, 64}}};
    op(Opcode::DequantizeInt8Blocks, {1, 2}, {3},
       {Attribute::u64(AttrKey::BlockSize, 32U)});
  } else if (which == "quantize_fp8_rows") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 64}},
                       {2, DType::FP8E4M3, TensorRole::Output, {4, 64}},
                       {3, DType::F32, TensorRole::Output, {4}}};
    op(Opcode::QuantizeFp8Rows, {1}, {2, 3});
  } else if (which == "linear_fp8_scaled") {
    program.tensors = {{1, DType::FP8E4M3, TensorRole::Input, {4, 64}},
                       {2, DType::FP8E4M3, TensorRole::Constant, {8, 64}},
                       {3, DType::F32, TensorRole::Input, {4}},
                       {4, DType::F32, TensorRole::Constant, {8}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::LinearFp8Scaled, {1, 2, 3, 4}, {5});
  } else if (which == "mse_loss_backward") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::F32, TensorRole::Input, {1}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::MseLossBackward, {1, 2, 3}, {4}, {accumulate_f32()});
  } else if (which == "linear_backward_input") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Constant, {8, 16}},
                       {3, DType::BF16, TensorRole::Output, {4, 16}}};
    op(Opcode::LinearBackwardInput, {1, 2}, {3}, {accumulate_f32()});
  } else if (which == "linear_backward_weight") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 16}},
                       {3, DType::BF16, TensorRole::Output, {8, 16}}};
    op(Opcode::LinearBackwardWeight, {1, 2}, {3}, {accumulate_f32()});
  } else if (which == "linear_addmm_prefill") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 16}},
                       {2, DType::BF16, TensorRole::Constant, {8, 16}},
                       {3, DType::BF16, TensorRole::Constant, {8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::Linear, {1, 2, 3}, {4},
       {Attribute::u64(AttrKey::LinearBiasMode,
                       static_cast<std::uint64_t>(LinearBiasMode::Addmm))});
  } else if (which == "h3_adaln_select") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {3, 144}},
                       {2, DType::I32, TensorRole::Input, {4}}};
    std::vector<std::uint32_t> outputs;
    for (std::uint32_t o = 0U; o < 6U; ++o) {
      program.tensors.push_back({3U + o, DType::BF16, TensorRole::Output, {4, 8}});
      outputs.push_back(3U + o);
    }
    op(Opcode::H3AdaLNSelect, {1, 2}, outputs);
  } else if (which == "h3_deinterleave_qkv") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 48}},
                       {2, DType::BF16, TensorRole::Output, {4, 2, 8}},
                       {3, DType::BF16, TensorRole::Output, {4, 2, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 2, 8}}};
    op(Opcode::H3DeinterleaveQkv, {1}, {2, 3, 4});
  } else if (which == "h3_deinterleave_qkv_weight") {
    program.tensors = {{1, DType::BF16, TensorRole::Constant, {48, 32}},
                       {2, DType::BF16, TensorRole::Output, {16, 32}},
                       {3, DType::BF16, TensorRole::Output, {16, 32}},
                       {4, DType::BF16, TensorRole::Output, {16, 32}}};
    op(Opcode::H3DeinterleaveQkvWeight, {1}, {2, 3, 4},
       {Attribute::u64(AttrKey::Heads, 2U), Attribute::u64(AttrKey::HeadDim, 8U)});
  } else if (which == "rms_norm_backward" || which == "rms_norm_backward_weight") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Constant, {8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}}};
    std::vector<std::uint32_t> outputs{4};
    if (which == "rms_norm_backward_weight") {
      program.tensors.push_back({5, DType::BF16, TensorRole::Output, {8}});
      outputs.push_back(5);
    }
    op(Opcode::RmsNormBackward, {1, 2, 3}, outputs,
       {accumulate_f32(), Attribute::f64(AttrKey::Epsilon, 1.0e-6)});
  } else if (which == "swiglu_backward" || which == "swiglu_backward_gate_first") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 16}},
                       {3, DType::BF16, TensorRole::Output, {4, 16}}};
    op(Opcode::SwiGluBackward, {1, 2}, {3},
       {accumulate_f32(),
        Attribute::boolean(AttrKey::GateFirst, which == "swiglu_backward_gate_first")});
  } else {
    fail("unknown batch5 corpus program " + std::string(which));
  }
  return program;
}


Program batch6_program(std::string_view which) {
  Program program;
  auto op = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                std::vector<std::uint32_t> outputs,
                std::vector<Attribute> attributes = {}) {
    program.operations = {{1, opcode, std::move(inputs), std::move(outputs),
                           std::move(attributes)}};
  };
  if (which == "channel_rms_norm") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {2, 8, 3}},
                       {2, DType::BF16, TensorRole::Constant, {8}},
                       {3, DType::BF16, TensorRole::Output, {2, 8, 3}}};
    op(Opcode::ChannelRmsNorm, {1, 2}, {3});
  } else if (which == "group_norm") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 4, 2, 2}},
                       {2, DType::BF16, TensorRole::Constant, {4}},
                       {3, DType::BF16, TensorRole::Constant, {4}},
                       {4, DType::BF16, TensorRole::Output, {1, 4, 2, 2}}};
    op(Opcode::GroupNorm, {1, 2, 3}, {4}, {Attribute::u64(AttrKey::Groups, 2U)});
  } else if (which == "pad_constant_4d") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 2, 3, 3}},
                       {2, DType::BF16, TensorRole::Output, {1, 2, 5, 6}}};
    op(Opcode::PadConstant, {1}, {2},
       {Attribute::u64(AttrKey::PadTop, 1U), Attribute::u64(AttrKey::PadBottom, 1U),
        Attribute::u64(AttrKey::PadWest, 2U), Attribute::u64(AttrKey::PadEast, 1U),
        Attribute::f64(AttrKey::Value, 0.5)});
  } else if (which == "pad_constant_5d") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 1, 2, 3, 3}},
                       {2, DType::BF16, TensorRole::Output, {1, 1, 4, 5, 6}}};
    op(Opcode::PadConstant, {1}, {2},
       {Attribute::u64(AttrKey::PadFront, 1U), Attribute::u64(AttrKey::PadBack, 1U),
        Attribute::u64(AttrKey::PadTop, 1U), Attribute::u64(AttrKey::PadBottom, 1U),
        Attribute::u64(AttrKey::PadWest, 2U), Attribute::u64(AttrKey::PadEast, 1U)});
  } else if (which == "pad_reflect_4d") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 2, 4, 4}},
                       {2, DType::BF16, TensorRole::Output, {1, 2, 6, 6}}};
    op(Opcode::PadReflect, {1}, {2},
       {Attribute::u64(AttrKey::PadTop, 1U), Attribute::u64(AttrKey::PadBottom, 1U),
        Attribute::u64(AttrKey::PadWest, 1U), Attribute::u64(AttrKey::PadEast, 1U)});
  } else if (which == "pad_reflect_5d") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 1, 3, 4, 4}},
                       {2, DType::BF16, TensorRole::Output, {1, 1, 5, 6, 6}}};
    op(Opcode::PadReflect, {1}, {2},
       {Attribute::u64(AttrKey::PadFront, 1U), Attribute::u64(AttrKey::PadBack, 1U),
        Attribute::u64(AttrKey::PadTop, 1U), Attribute::u64(AttrKey::PadBottom, 1U),
        Attribute::u64(AttrKey::PadWest, 1U), Attribute::u64(AttrKey::PadEast, 1U)});
  } else if (which == "adamw_update" || which == "adamw_update_bf16") {
    const auto parameter = which == "adamw_update" ? DType::F32 : DType::BF16;
    const auto gradient = which == "adamw_update" ? DType::BF16 : DType::F32;
    program.tensors = {{1, parameter, TensorRole::Input, {8}},
                       {2, gradient, TensorRole::Input, {8}},
                       {3, DType::F32, TensorRole::Input, {8}},
                       {4, DType::F32, TensorRole::Input, {8}},
                       {5, DType::I32, TensorRole::Input, {1}},
                       {6, parameter, TensorRole::Output, {8}},
                       {7, DType::F32, TensorRole::Output, {8}},
                       {8, DType::F32, TensorRole::Output, {8}}};
    op(Opcode::AdamWUpdate, {1, 2, 3, 4, 5}, {6, 7, 8},
       {accumulate_f32(), Attribute::f64(AttrKey::LearningRate, 1.0e-4),
        Attribute::f64(AttrKey::Beta1, 0.9), Attribute::f64(AttrKey::Beta2, 0.999),
        Attribute::f64(AttrKey::Epsilon, 1.0e-8), Attribute::f64(AttrKey::WeightDecay, 0.01),
        Attribute::f64(AttrKey::ClipScale, 0.5)});
  } else if (which == "patchify_3d" || which == "unpatchify_3d") {
    const bool forward = which == "patchify_3d";
    program.tensors = {{1, DType::BF16, TensorRole::Input,
                        forward ? std::vector<std::uint64_t>{1, 2, 4, 4, 4}
                                : std::vector<std::uint64_t>{8, 16}},
                       {2, DType::BF16, TensorRole::Output,
                        forward ? std::vector<std::uint64_t>{8, 16}
                                : std::vector<std::uint64_t>{1, 2, 4, 4, 4}}};
    op(forward ? Opcode::Patchify3D : Opcode::Unpatchify3D, {1}, {2},
       {Attribute::u64(AttrKey::PatchT, 2U), Attribute::u64(AttrKey::PatchH, 2U),
        Attribute::u64(AttrKey::PatchW, 2U)});
  } else if (which == "layer_norm" || which == "layer_norm_welford128") {
    const std::uint64_t columns = which == "layer_norm" ? 8U : 128U;
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, columns}},
                       {2, DType::BF16, TensorRole::Constant, {columns}},
                       {3, DType::BF16, TensorRole::Constant, {columns}},
                       {4, DType::BF16, TensorRole::Output, {4, columns}}};
    std::vector<Attribute> attributes{Attribute::f64(AttrKey::Epsilon, 1.0e-5)};
    if (which == "layer_norm_welford128")
      attributes.push_back(Attribute::u64(AttrKey::BlockSize, 128U));
    op(Opcode::LayerNorm, {1, 2, 3}, {4}, attributes);
  } else if (which == "layer_norm_backward") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Constant, {8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}},
                       {5, DType::BF16, TensorRole::Output, {8}},
                       {6, DType::BF16, TensorRole::Output, {8}}};
    op(Opcode::LayerNormBackward, {1, 2, 3}, {4, 5, 6},
       {accumulate_f32(), Attribute::f64(AttrKey::Epsilon, 1.0e-5)});
  } else if (which == "rms_norm_modulate_backward") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Input, {4, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 8}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}},
                       {6, DType::BF16, TensorRole::Output, {4, 8}}};
    op(Opcode::RmsNormModulateBackward, {1, 2, 3}, {4, 5, 6},
       {accumulate_f32(), Attribute::f64(AttrKey::Epsilon, 1.0e-5)});
  } else if (which == "rms_norm_modulate_backward_weighted") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 8}},
                       {3, DType::BF16, TensorRole::Constant, {8}},
                       {4, DType::BF16, TensorRole::Input, {4, 8}},
                       {5, DType::BF16, TensorRole::Output, {4, 8}},
                       {6, DType::BF16, TensorRole::Output, {4, 8}},
                       {7, DType::BF16, TensorRole::Output, {4, 8}},
                       {8, DType::BF16, TensorRole::Output, {8}}};
    op(Opcode::RmsNormModulateBackward, {1, 2, 3, 4}, {5, 6, 7, 8},
       {accumulate_f32(), Attribute::f64(AttrKey::Epsilon, 1.0e-5)});
  } else {
    fail("unknown batch6 corpus program " + std::string(which));
  }
  return program;
}


Program batch7_program(std::string_view which) {
  Program program;
  auto op = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                std::vector<std::uint32_t> outputs,
                std::vector<Attribute> attributes = {}) {
    program.operations = {{1, opcode, std::move(inputs), std::move(outputs),
                           std::move(attributes)}};
  };
  if (which == "conv1d" || which == "conv1d_bias" || which == "conv1d_replicate") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 2, 8}},
                       {2, DType::BF16, TensorRole::Constant, {4, 2, 3}},
                       {3, DType::BF16, TensorRole::Output, {1, 4, 8}}};
    std::vector<std::uint32_t> inputs{1, 2};
    if (which == "conv1d_bias") {
      program.tensors.push_back({4, DType::BF16, TensorRole::Constant, {4}});
      inputs.push_back(4);
    }
    std::vector<Attribute> attributes{Attribute::u64(AttrKey::PadLeft, 1U),
                                      Attribute::u64(AttrKey::PadRight, 1U)};
    if (which == "conv1d_replicate")
      attributes.push_back(Attribute::u64(AttrKey::PadMode, 1U));
    op(Opcode::Conv1d, inputs, {3}, attributes);
  } else if (which == "conv1d_transposed") {
    // stride 2 transposed conv: L_out = (L-1)*stride + K - pad_left - pad_right
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 2, 4}},
                       {2, DType::BF16, TensorRole::Constant, {2, 4, 4}},
                       {3, DType::BF16, TensorRole::Output, {1, 4, 10}}};
    op(Opcode::Conv1d, {1, 2}, {3},
       {Attribute::boolean(AttrKey::Transposed, true),
        Attribute::u64(AttrKey::Stride, 2U)});
  } else if (which == "quantize_fp8_blocks32") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 64}},
                       {2, DType::FP8E4M3, TensorRole::Output, {4, 64}},
                       {3, DType::FP8E8M0, TensorRole::Output, {128, 4}}};
    op(Opcode::QuantizeFp8Blocks32, {1}, {2, 3});
  } else if (which == "rotary_apply_interleaved" || which == "rotary_apply_half_split") {
    program.tensors = {{1, DType::BF16, TensorRole::Input, {1, 4, 2, 8}},
                       {2, DType::F32, TensorRole::Input, {1, 4, 4}},
                       {3, DType::F32, TensorRole::Input, {1, 4, 4}},
                       {4, DType::BF16, TensorRole::Output, {1, 4, 2, 8}}};
    op(Opcode::RotaryApply, {1, 2, 3}, {4},
       {Attribute::u64(AttrKey::RotaryLayout,
                       static_cast<std::uint64_t>(
                           which == "rotary_apply_half_split"
                               ? RotaryLayout::HalfSplit
                               : RotaryLayout::Interleaved))});
  } else if (which == "attention_exact" || which == "attention_exact_causal" ||
             which == "attention_exact_gqa") {
    const std::uint64_t kv = which == "attention_exact_gqa" ? 1U : 2U;
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 2, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, kv, 8}},
                       {3, DType::BF16, TensorRole::Input, {4, kv, 8}},
                       {4, DType::BF16, TensorRole::Output, {4, 2, 8}}};
    std::vector<Attribute> attributes;
    if (which == "attention_exact_causal")
      attributes.push_back(Attribute::boolean(AttrKey::Causal, true));
    if (which == "attention_exact_gqa")
      attributes.push_back(Attribute::u64(AttrKey::KvHeads, 1U));
    op(Opcode::Attention, {1, 2, 3}, {4}, attributes);
  } else if (which == "attention_lse" || which == "attention_lse_causal_gqa") {
    const bool gqa = which == "attention_lse_causal_gqa";
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 2, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, gqa ? 1U : 2U, 8}},
                       {3, DType::F32, TensorRole::Output, {4, 2}}};
    std::vector<Attribute> attributes{accumulate_f32()};
    if (gqa) {
      attributes.push_back(Attribute::boolean(AttrKey::Causal, true));
      attributes.push_back(Attribute::u64(AttrKey::KvHeads, 1U));
    }
    op(Opcode::AttentionLse, {1, 2}, {3}, attributes);
  } else if (which == "qk_norm_rope_backward" || which == "qk_norm_rope_backward_half_table" ||
             which == "qk_norm_rope_backward_weight") {
    const std::uint64_t table = which == "qk_norm_rope_backward_half_table" ? 4U : 8U;
    program.tensors = {{1, DType::BF16, TensorRole::Input, {4, 2, 8}},
                       {2, DType::BF16, TensorRole::Input, {4, 2, 8}},
                       {3, DType::BF16, TensorRole::Constant, {8}},
                       {4, DType::BF16, TensorRole::Input, {4, table}},
                       {5, DType::BF16, TensorRole::Input, {4, table}},
                       {6, DType::BF16, TensorRole::Output, {4, 2, 8}}};
    std::vector<std::uint32_t> outputs{6};
    if (which == "qk_norm_rope_backward_weight") {
      program.tensors.push_back({7, DType::BF16, TensorRole::Output, {8}});
      outputs.push_back(7);
    }
    op(Opcode::QkNormPartialRopeBackward, {1, 2, 3, 4, 5}, outputs,
       {accumulate_f32(), Attribute::u64(AttrKey::RotaryDim, 8U),
        Attribute::f64(AttrKey::Epsilon, 1.0e-6)});
  } else {
    fail("unknown batch7 corpus program " + std::string(which));
  }
  return program;
}

std::vector<Case> corpus() {
  using Q = Int8RowQuantization;
  return {
      {"silu_f32", [] { return elementwise(Opcode::SiLU, DType::F32); }},
      {"silu_bf16", [] { return elementwise(Opcode::SiLU, DType::BF16); }},
      {"sigmoid_bf16", [] { return elementwise(Opcode::Sigmoid, DType::BF16); }},
      {"gelu_tanh",
       [] {
         return elementwise(Opcode::Gelu, DType::BF16,
                            {Attribute::u64(AttrKey::Approximation,
                                            static_cast<std::uint64_t>(
                                                GeluApproximation::Tanh))});
       }},
      {"gelu_exacterf",
       [] {
         return elementwise(Opcode::Gelu, DType::BF16,
                            {Attribute::u64(AttrKey::Approximation,
                                            static_cast<std::uint64_t>(
                                                GeluApproximation::ExactErf))});
       }},
      {"dequantize_int4_g64", [] { return dequantize_int4(false); }},
      {"dequantize_int4_outliers", [] { return dequantize_int4(true); }},
      {"dequantize_int5_g64", [] { return dequantize_int5(false); }},
      {"dequantize_int5_column_scales", [] { return dequantize_int5(true); }},
      // rms_norm reduction strategies: packed4 (default block 256 on rows
      // divisible by 4), generic (odd width), per-row (block 128 on width
      // 128), blocked (block 512, width 6144, tile 8192), chunked (tile 2048),
      // plus the weight-offset literal and the vectorized Implementation 2.
      {"rms_norm_default_128", [] { return rms_norm(1U, 128U); }},
      {"rms_norm_default_768", [] { return rms_norm(1U, 768U); }},
      {"rms_norm_generic_100", [] { return rms_norm(1U, 100U); }},
      {"rms_norm_generic_block64_768", [] { return rms_norm(1U, 768U, 64U); }},
      {"rms_norm_per_row_128", [] { return rms_norm(1U, 128U, 128U); }},
      {"rms_norm_blocked_6144",
       [] { return rms_norm(1U, 6144U, 512U, 8192U); }},
      {"rms_norm_chunked_6144",
       [] { return rms_norm(1U, 6144U, 512U, 2048U); }},
      {"rms_norm_weight_offset_768",
       [] { return rms_norm(1U, 768U, 0U, 0U, 1.0); }},
      {"rms_norm_implementation2_128", [] { return rms_norm(2U, 128U, 128U); }},
      {"quantize_int8_rows_direct",
       [] { return quantize_int8_rows(Q::Direct, 512U, false, false); }},
      {"quantize_int8_rows_direct_dynamic_clip",
       [] { return quantize_int8_rows(Q::Direct, 512U, false, true); }},
      {"quantize_int8_rows_direct_residual2",
       [] { return quantize_int8_rows(Q::Direct, 512U, true, false); }},
      {"quantize_int8_rows_h256_convrot",
       [] { return quantize_int8_rows(Q::H256ConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h256_signed_convrot",
       [] { return quantize_int8_rows(Q::H256SignedConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h4096_signed_convrot",
       [] { return quantize_int8_rows(Q::H4096SignedConvRot, 4096U, false, false); }},
      {"quantize_int8_rows_h256_f32_convrot",
       [] { return quantize_int8_rows(Q::H256F32ConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h256_f32_signed_convrot",
       [] { return quantize_int8_rows(Q::H256F32SignedConvRot, 512U, false, false); }},
      {"quantize_int8_rows_h4096_f32_signed_convrot",
       [] { return quantize_int8_rows(Q::H4096F32SignedConvRot, 4096U, false, false); }},
      {"quantize_int8_rows_h4096_f32_signed_convrot_residual2",
       [] { return quantize_int8_rows(Q::H4096F32SignedConvRot, 4096U, true, false); }},
      {"quantize_int8_rows_h256_f32_sylvester_convrot",
       [] { return quantize_int8_rows(Q::H256F32SylvesterConvRot, 512U, false, false); }},
      // batch 3: the small elementwise-style kernels
      {"add_bf16", [] { return binary(Opcode::Add, DType::BF16); }},
      {"multiply_f32", [] { return binary(Opcode::Multiply, DType::F32); }},
      {"bias_add", [] { return with_vector(Opcode::BiasAdd, 1U, false); }},
      {"affine_last_dim", [] { return with_vector(Opcode::AffineLastDim, 1U, false); }},
      {"affine_last_dim_bias", [] { return with_vector(Opcode::AffineLastDim, 2U, true); }},
      {"clamp",
       [] {
         return unary(Opcode::Clamp, DType::BF16,
                      {Attribute::f64(AttrKey::Lower, -1.0),
                       Attribute::f64(AttrKey::Upper, 1.0)});
       }},
      {"reshape", [] { return batch3_program("reshape"); }},
      {"fill", [] { return batch3_program("fill"); }},
      {"fill_default", [] { return batch3_program("fill_default"); }},
      {"silu_backward", [] { return batch3_program("silu_backward"); }},
      {"residual_gate_backward", [] { return batch3_program("residual_gate_backward"); }},
      {"bias_backward", [] { return batch3_program("bias_backward"); }},
      {"euler_velocity_step", [] { return batch3_program("euler_velocity_step"); }},
      {"linear_blend", [] { return batch3_program("linear_blend"); }},
      {"mse_loss", [] { return batch3_program("mse_loss"); }},
      {"swiglu", [] { return batch3_program("swiglu"); }},
      {"swiglu_gate_first", [] { return batch3_program("swiglu_gate_first"); }},
      {"residual_gate", [] { return batch3_program("residual_gate"); }},
      {"residual_gate_broadcast", [] { return batch3_program("residual_gate_broadcast"); }},
      {"flow_euler_step", [] { return batch3_program("flow_euler_step"); }},
      {"snake_beta", [] { return batch3_program("snake_beta"); }},
      {"cast_bf16_f32", [] { return batch3_program("cast_bf16_f32"); }},
      {"cast_f32_bf16", [] { return batch3_program("cast_f32_bf16"); }},
      // batch 4: data movement, embeddings, masks
      {"upsample_nearest_2d", [] { return batch4_program("upsample_nearest_2d"); }},
      {"broadcast_to", [] { return batch4_program("broadcast_to"); }},
      {"broadcast_to_rank", [] { return batch4_program("broadcast_to_rank"); }},
      {"slice", [] { return batch4_program("slice"); }},
      {"rotary_frequency", [] { return batch4_program("rotary_frequency"); }},
      {"rotary_frequency_theta", [] { return batch4_program("rotary_frequency_theta"); }},
      {"boolean_mask_to_bias", [] { return batch4_program("boolean_mask_to_bias"); }},
      {"boolean_mask_to_bias_keys_only", [] { return batch4_program("boolean_mask_to_bias_keys_only"); }},
      {"boolean_mask_to_bias_matrix", [] { return batch4_program("boolean_mask_to_bias_matrix"); }},
      {"gather_rows", [] { return batch4_program("gather_rows"); }},
      {"indexed_update_rows", [] { return batch4_program("indexed_update_rows"); }},
      {"select_row_chunks", [] { return batch4_program("select_row_chunks"); }},
      {"sinusoidal_timestep", [] { return batch4_program("sinusoidal_timestep"); }},
      {"sinusoidal_timestep_flip", [] { return batch4_program("sinusoidal_timestep_flip"); }},
      {"rotary_position", [] { return batch4_program("rotary_position"); }},
      {"permute", [] { return batch4_program("permute"); }},
      {"concat", [] { return batch4_program("concat"); }},
      {"concat_axis0_three", [] { return batch4_program("concat_axis0_three"); }},
      // batch 5: low-bit codecs, training backward ops, H3 layout ops
      {"dequantize_int8_blocks", [] { return batch5_program("dequantize_int8_blocks"); }},
      {"quantize_fp8_rows", [] { return batch5_program("quantize_fp8_rows"); }},
      {"linear_fp8_scaled", [] { return batch5_program("linear_fp8_scaled"); }},
      {"mse_loss_backward", [] { return batch5_program("mse_loss_backward"); }},
      {"linear_backward_input", [] { return batch5_program("linear_backward_input"); }},
      {"linear_backward_weight", [] { return batch5_program("linear_backward_weight"); }},
      {"linear_addmm_prefill", [] { return batch5_program("linear_addmm_prefill"); }},
      {"h3_adaln_select", [] { return batch5_program("h3_adaln_select"); }},
      {"h3_deinterleave_qkv", [] { return batch5_program("h3_deinterleave_qkv"); }},
      {"h3_deinterleave_qkv_weight", [] { return batch5_program("h3_deinterleave_qkv_weight"); }},
      {"rms_norm_backward", [] { return batch5_program("rms_norm_backward"); }},
      {"rms_norm_backward_weight", [] { return batch5_program("rms_norm_backward_weight"); }},
      {"swiglu_backward", [] { return batch5_program("swiglu_backward"); }},
      {"swiglu_backward_gate_first", [] { return batch5_program("swiglu_backward_gate_first"); }},
      // batch 6: norms, padding, AdamW, patchify, layer norm forward/backward
      {"channel_rms_norm", [] { return batch6_program("channel_rms_norm"); }},
      {"group_norm", [] { return batch6_program("group_norm"); }},
      {"pad_constant_4d", [] { return batch6_program("pad_constant_4d"); }},
      {"pad_constant_5d", [] { return batch6_program("pad_constant_5d"); }},
      {"pad_reflect_4d", [] { return batch6_program("pad_reflect_4d"); }},
      {"pad_reflect_5d", [] { return batch6_program("pad_reflect_5d"); }},
      {"adamw_update", [] { return batch6_program("adamw_update"); }},
      {"adamw_update_bf16", [] { return batch6_program("adamw_update_bf16"); }},
      {"patchify_3d", [] { return batch6_program("patchify_3d"); }},
      {"unpatchify_3d", [] { return batch6_program("unpatchify_3d"); }},
      {"layer_norm", [] { return batch6_program("layer_norm"); }},
      {"layer_norm_welford128", [] { return batch6_program("layer_norm_welford128"); }},
      {"layer_norm_backward", [] { return batch6_program("layer_norm_backward"); }},
      {"rms_norm_modulate_backward", [] { return batch6_program("rms_norm_modulate_backward"); }},
      {"rms_norm_modulate_backward_weighted", [] { return batch6_program("rms_norm_modulate_backward_weighted"); }},
      // batch 7: conv1d, MXFP8, rotary apply, exact attention, LSE, qk-norm-rope backward
      {"conv1d", [] { return batch7_program("conv1d"); }},
      {"conv1d_bias", [] { return batch7_program("conv1d_bias"); }},
      {"conv1d_replicate", [] { return batch7_program("conv1d_replicate"); }},
      {"conv1d_transposed", [] { return batch7_program("conv1d_transposed"); }},
      {"quantize_fp8_blocks32", [] { return batch7_program("quantize_fp8_blocks32"); }},
      {"rotary_apply_interleaved", [] { return batch7_program("rotary_apply_interleaved"); }},
      {"rotary_apply_half_split", [] { return batch7_program("rotary_apply_half_split"); }},
      {"attention_exact", [] { return batch7_program("attention_exact"); }},
      {"attention_exact_causal", [] { return batch7_program("attention_exact_causal"); }},
      {"attention_exact_gqa", [] { return batch7_program("attention_exact_gqa"); }},
      {"attention_lse", [] { return batch7_program("attention_lse"); }},
      {"attention_lse_causal_gqa", [] { return batch7_program("attention_lse_causal_gqa"); }},
      {"qk_norm_rope_backward", [] { return batch7_program("qk_norm_rope_backward"); }},
      {"qk_norm_rope_backward_half_table", [] { return batch7_program("qk_norm_rope_backward_half_table"); }},
      {"qk_norm_rope_backward_weight", [] { return batch7_program("qk_norm_rope_backward_weight"); }},
  };
}

std::string read_file(const std::filesystem::path &path) {
  std::ifstream in(path, std::ios::binary);
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

void report_first_difference(const std::string &expected,
                             const std::string &actual) {
  std::size_t line = 1, column = 1, index = 0;
  const auto limit = std::min(expected.size(), actual.size());
  while (index < limit && expected[index] == actual[index]) {
    if (expected[index] == '\n') {
      ++line;
      column = 1;
    } else {
      ++column;
    }
    ++index;
  }
  std::cerr << "  first difference at line " << line << " column " << column
            << " (expected " << expected.size() << " bytes, got "
            << actual.size() << ")\n  expected: "
            << expected.substr(index, 80) << "\n  actual:   "
            << actual.substr(index, 80) << '\n';
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "usage: dif_kernel_source_tests FIXTURE_DIR [--update]\n";
    return 2;
  }
  const std::filesystem::path directory = argv[1];
  const bool update = argc > 2 && std::string(argv[2]) == "--update";
  if (update)
    std::filesystem::create_directories(directory);
  int failures = 0;
  std::size_t checked = 0;
  for (const auto &entry : corpus()) {
    std::string source;
    try {
      source = dif::compiler::emit_cuda(entry.build()).source;
    } catch (const dif::Error &error) {
      std::cerr << "FAIL: " << entry.name << ": emit_cuda threw: "
                << error.what() << '\n';
      ++failures;
      continue;
    }
    const auto path = directory / (entry.name + ".cu");
    if (update) {
      std::ofstream out(path, std::ios::binary | std::ios::trunc);
      out << source;
      std::cout << "wrote " << path.string() << " (" << source.size()
                << " bytes)\n";
      continue;
    }
    if (!std::filesystem::exists(path)) {
      std::cerr << "FAIL: " << entry.name << ": no snapshot at " << path
                << " (run with --update to create it)\n";
      ++failures;
      continue;
    }
    const auto expected = read_file(path);
    if (expected != source) {
      std::cerr << "FAIL: " << entry.name
                << ": generated CUDA source differs from the snapshot\n";
      report_first_difference(expected, source);
      ++failures;
      continue;
    }
    ++checked;
  }
  if (update) {
    std::cout << "updated " << corpus().size() << " kernel source snapshots\n";
    return 0;
  }
  if (failures != 0) {
    std::cerr << failures << " kernel source snapshot(s) failed\n";
    return 1;
  }
  std::cout << "PASS: " << checked
            << " generated CUDA sources are byte-identical to their snapshots\n";
  return 0;
}
