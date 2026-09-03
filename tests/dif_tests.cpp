#include "dif/compiler/compiler.hpp"
#include "dif/compiler/int4.hpp"
#include "dif/compiler/slice.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/compiler/layout_plan.hpp"
#include "dif/compiler/residency_plan.hpp"
#include "dif/opt/gate.hpp"
#include "dif/frontend/h3.hpp"
#include "dif/frontend/h3_conditioning.hpp"
#include "dif/frontend/h3_latents.hpp"
#include "dif/frontend/h3_media.hpp"
#include "dif/frontend/h3_vae.hpp"
#include "dif/frontend/h3_video_encoder.hpp"
#include "dif/frontend/h3_audio_vae.hpp"
#include "dif/frontend/krea2.hpp"
#include "dif/frontend/krea2_vae.hpp"
#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/frontend/qwen3vl_vision.hpp"
#include "dif/frontend/training.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/png.hpp"
#include "dif/support/sha256.hpp"
#include "dif/support/torch_cpu_rng.hpp"
#include "dif/support/wav.hpp"
#include "dif/training/checkpoint.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <fstream>
#include <filesystem>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const char *message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::F32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

dif::runtime::Tensor float_tensor(dif::ir::DType dtype,
                                  std::vector<std::uint64_t> dims,
                                  const std::vector<float> &values) {
  dif::runtime::Tensor tensor{dtype, std::move(dims), {}};
  tensor.bytes.resize(values.size() * dif::ir::dtype_size(dtype));
  tensor.validate();
  for (std::size_t i = 0; i < values.size(); ++i)
    dif::runtime::store_float(tensor, i, values[i]);
  return tensor;
}

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

dif::runtime::Tensor bool_tensor(std::vector<std::uint64_t> dims,
                                 const std::vector<std::uint8_t> &values) {
  dif::runtime::Tensor tensor{dif::ir::DType::Bool, std::move(dims), values};
  tensor.validate();
  return tensor;
}

std::vector<float> float_values(const dif::runtime::Tensor &tensor) {
  std::vector<float> values(tensor.element_count());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = dif::runtime::load_float(tensor, i);
  return values;
}

std::vector<float> floats_from_bits(
    std::initializer_list<std::uint32_t> bits) {
  std::vector<float> values;
  values.reserve(bits.size());
  for (const auto value : bits)
    values.push_back(std::bit_cast<float>(value));
  return values;
}

dif::ir::Program rms_program(std::uint64_t rows, std::uint64_t cols) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {rows, cols}},
      {2, DType::F32, TensorRole::Input, {rows, cols}},
      {3, DType::F32, TensorRole::Input, {rows, cols}},
      {4, DType::F32, TensorRole::Output, {rows, cols}},
  };
  program.operations = {{1,
                         Opcode::RmsNormModulate,
                         {1, 2, 3},
                         {4},
                         {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
                          Attribute::u64(AttrKey::BlockSize, 64)}}};
  return program;
}

void test_sha256() {
  const std::string input = "abc";
  const auto bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(input.data()), input.size());
  expect(dif::hex_digest(dif::sha256(bytes)) ==
             "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
         "SHA-256 known vector");
}

void test_json_parser() {
  const auto value = dif::json::parse(
      R"({"dtype":"BF16","shape":[56,128],"enabled":true,"name":"H\u0033"})");
  expect(value.find("dtype") && value.find("dtype")->string() == "BF16",
         "JSON string field");
  expect(value.find("shape") && value.find("shape")->array().size() == 2U &&
             value.find("shape")->array()[1].number() == 128.0,
         "JSON numeric array");
  expect(value.find("enabled") && value.find("enabled")->boolean(),
         "JSON boolean field");
  expect(value.find("name") && value.find("name")->string() == "H3",
         "JSON unicode escape");
  bool rejected = false;
  try {
    (void)dif::json::parse(R"({"x":1,"x":2})");
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "JSON duplicate key rejected");
}

void test_codec() {
  const auto program = rms_program(2, 4);
  const auto encoded = dif::ir::encode(program);
  const auto decoded = dif::ir::decode(encoded);
  expect(decoded.tensors.size() == 4, "DiffIR tensor roundtrip");
  expect(decoded.operations.size() == 1, "DiffIR operation roundtrip");
  expect(dif::ir::fingerprint(program) == dif::ir::fingerprint(decoded),
         "DiffIR fingerprint roundtrip");
  auto corrupted = encoded;
  corrupted[20] ^= 1U;
  bool rejected = false;
  try {
    (void)dif::ir::decode(corrupted);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "DiffIR corruption rejected");
}

void test_cpu_rms() {
  const auto program = rms_program(1, 4);
  dif::runtime::TensorMap inputs;
  inputs.emplace(1, f32_tensor({1, 4}, {1.0F, 2.0F, 3.0F, 4.0F}));
  inputs.emplace(2, f32_tensor({1, 4}, {0.0F, 0.1F, -0.2F, 0.3F}));
  inputs.emplace(3, f32_tensor({1, 4}, {0.1F, -0.1F, 0.2F, 0.0F}));
  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  const auto result = executor->run(program, inputs, options);
  const auto output = result.outputs.at(4).f32();
  const float inv = 1.0F / std::sqrt(7.5F + 1.0e-5F);
  const std::vector<float> expected = {
      1.0F * inv + 0.1F,
      2.0F * inv * 1.1F - 0.1F,
      3.0F * inv * 0.8F + 0.2F,
      4.0F * inv * 1.3F,
  };
  for (std::size_t i = 0; i < expected.size(); ++i)
    expect(std::abs(output[i] - expected[i]) < 1.0e-6F,
           "CPU RMSNorm modulation result");
}

void test_cpu_linear_bias() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {2, 3}},
      {2, DType::F32, TensorRole::Constant, {2, 3}},
      {3, DType::F32, TensorRole::Constant, {2}},
      {4, DType::F32, TensorRole::Output, {2, 2}},
  };
  program.operations = {{1, Opcode::Linear, {1, 2, 3}, {4}, {}}};
  dif::runtime::TensorMap inputs;
  inputs.emplace(1, f32_tensor({2, 3}, {1, 2, 3, 4, 5, 6}));
  inputs.emplace(2, f32_tensor({2, 3}, {1, 0, -1, 0.5F, 1, 2}));
  inputs.emplace(3, f32_tensor({2}, {0.25F, -0.5F}));
  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  const auto result = executor->run(program, inputs, options);
  const auto output = result.outputs.at(4).f32();
  const std::array<float, 4> expected = {-1.75F, 8.0F, -1.75F, 18.5F};
  for (std::size_t i = 0; i < expected.size(); ++i)
    expect(std::abs(output[i] - expected[i]) < 1.0e-6F,
           "CPU fused Linear bias result");

  Program addmm;
  addmm.tensors = {
      {1, DType::BF16, TensorRole::Input, {3, 4}},
      {2, DType::BF16, TensorRole::Constant, {5, 4}},
      {3, DType::BF16, TensorRole::Constant, {5}},
      {4, DType::BF16, TensorRole::Output, {3, 5}},
  };
  addmm.operations = {{1, Opcode::Linear, {1, 2, 3}, {4},
                       {Attribute::u64(
                           AttrKey::LinearBiasMode,
                           static_cast<std::uint64_t>(
                               LinearBiasMode::Addmm))}}};
  dif::runtime::TensorMap addmm_inputs;
  addmm_inputs.emplace(
      1, float_tensor(DType::BF16, {3, 4},
                      {1.0F, -2.0F, 3.0F, 0.5F, -4.0F, 5.0F, 0.25F,
                       2.0F, 7.0F, -1.0F, -3.0F, 4.0F}));
  addmm_inputs.emplace(
      2, float_tensor(DType::BF16, {5, 4},
                      {0.5F, 1.0F, -1.0F, 2.0F, -2.0F, 0.25F, 1.5F,
                       0.5F, 3.0F, -0.5F, 0.75F, -1.0F, 1.0F, 1.0F,
                       1.0F, 1.0F, -0.25F, 2.0F, -3.0F, 0.125F}));
  addmm_inputs.emplace(
      3, float_tensor(DType::BF16, {5},
                      {0.125F, -0.25F, 0.5F, 1.0F, -2.0F}));
  const auto addmm_cpu = executor->run(addmm, addmm_inputs, options);
  if (dif::runtime::cuda_available()) {
    const auto addmm_cuda =
        dif::runtime::make_cuda_executor()->run(addmm, addmm_inputs, options);
    const auto &reference = addmm_cpu.outputs.at(4);
    const auto &candidate = addmm_cuda.outputs.at(4);
    expect(reference.bytes == candidate.bytes,
           "CUDA multi-row addmm prefill is bit-exact to BF16 CPU semantics");
  }
}

void test_float_storage_conversions() {
  using dif::runtime::bf16_to_float;
  using dif::runtime::f16_to_float;
  using dif::runtime::float_to_bf16;
  using dif::runtime::float_to_f16;
  expect(float_to_bf16(1.0F) == 0x3f80U && bf16_to_float(0x3f80U) == 1.0F,
         "bf16 one conversion");
  expect(float_to_bf16(-2.5F) == 0xc020U && bf16_to_float(0xc020U) == -2.5F,
         "bf16 negative conversion");
  expect(float_to_f16(1.0F) == 0x3c00U && f16_to_float(0x3c00U) == 1.0F,
         "f16 one conversion");
  expect(float_to_f16(-2.0F) == 0xc000U && f16_to_float(0xc000U) == -2.0F,
         "f16 negative conversion");
  expect(float_to_f16(65504.0F) == 0x7bffU &&
             f16_to_float(0x7bffU) == 65504.0F,
         "f16 maximum finite conversion");
  expect(float_to_f16(std::ldexp(1.0F, -24)) == 0x0001U &&
             f16_to_float(0x0001U) == std::ldexp(1.0F, -24),
         "f16 minimum subnormal conversion");
  expect(float_to_f16(std::numeric_limits<float>::infinity()) == 0x7c00U &&
             std::isinf(f16_to_float(0x7c00U)),
         "f16 infinity conversion");
  expect(std::isnan(f16_to_float(float_to_f16(
             std::numeric_limits<float>::quiet_NaN()))),
         "f16 NaN conversion");
}

dif::ir::Program all_opcodes_program(dif::ir::DType dtype) {
  using namespace dif::ir;
  constexpr auto input = TensorRole::Input;
  constexpr auto output = TensorRole::Output;
  Program program;
  program.tensors = {
      {1, dtype, input, {1, 2}}, {2, dtype, input, {1, 2}},
      {3, dtype, output, {1, 2}}, {4, dtype, output, {1, 2}},
      {5, dtype, input, {2}}, {6, dtype, output, {1, 2}},
      {7, dtype, input, {2, 2}}, {8, dtype, input, {2}},
      {9, dtype, output, {1, 2}}, {10, dtype, input, {1, 4}},
      {11, dtype, input, {4}}, {12, dtype, input, {1, 4}},
      {13, dtype, input, {1, 4}}, {14, dtype, output, {1, 4}},
      {15, dtype, input, {1, 4}}, {16, dtype, output, {1, 2}},
      {17, dtype, input, {1, 2}}, {18, dtype, input, {1, 2}},
      {19, dtype, input, {1, 2}}, {20, dtype, output, {1, 2}},
      {21, dtype, input, {2, 1, 1}}, {22, dtype, input, {2, 1, 1}},
      {23, dtype, input, {2, 1, 1}}, {24, dtype, output, {2, 1, 1}},
      {25, dtype, input, {1, 1, 2}}, {26, dtype, input, {2}},
      {27, dtype, input, {1, 2}}, {28, dtype, input, {1, 2}},
      {29, dtype, output, {1, 1, 2}},
      {30, dtype, input, {1, 18}}, {31, DType::I32, input, {1}},
      {32, dtype, output, {1, 1}}, {33, dtype, output, {1, 1}},
      {34, dtype, output, {1, 1}}, {35, dtype, output, {1, 1}},
      {36, dtype, output, {1, 1}}, {37, dtype, output, {1, 1}},
      {38, dtype, input, {1, 6}}, {39, dtype, output, {1, 1, 2}},
      {40, dtype, output, {1, 1, 2}}, {41, dtype, output, {1, 1, 2}},
      {42, dtype, input, {6, 1}}, {43, dtype, output, {2, 1}},
      {44, dtype, output, {2, 1}}, {45, dtype, output, {2, 1}},
      {46, DType::I8, input, {1, 8}}, {47, dtype, input, {1, 1}},
      {48, dtype, output, {1, 16}},
      {49, dtype, input, {1, 2}}, {50, dtype, output, {1, 2}},
      {51, dtype, input, {2, 2}}, {52, dtype, input, {2}},
      {53, dtype, output, {2, 2}}, {54, dtype, output, {1, 3}},
      {55, dtype, input, {3, 2}}, {56, DType::I32, input, {2}},
      {57, dtype, output, {2, 2}}, {58, dtype, input, {3, 2}},
      {59, dtype, input, {2, 2}}, {60, DType::I32, input, {3}},
      {61, dtype, output, {3, 2}},
  };
  const auto cast_dtype = dtype == DType::F32 ? DType::BF16 : DType::F32;
  program.tensors.push_back({62, cast_dtype, output, {1, 2}});
  program.tensors.push_back({63, dtype, input, {2, 4}});
  program.tensors.push_back({64, DType::I32, input, {3}});
  program.tensors.push_back({65, dtype, output, {3, 2}});
  program.tensors.push_back({66, dtype, output, {3, 2}});
  program.tensors.push_back({67, dtype, input, {1, 2}});
  program.tensors.push_back({68, dtype, input, {1, 2}});
  program.tensors.push_back({69, DType::F32, input, {1}});
  program.tensors.push_back({70, dtype, output, {1, 2}});
  program.tensors.push_back({71, dtype, input, {1, 2}});
  program.tensors.push_back({72, dtype, input, {1, 2}});
  program.tensors.push_back({73, DType::F32, input, {1}});
  program.tensors.push_back({74, DType::F32, input, {2}});
  program.tensors.push_back({75, dtype, output, {1, 2}});
  program.tensors.push_back({76, dtype, input, {1, 2, 2, 2, 2}});
  program.tensors.push_back({77, dtype, output, {2, 8}});
  program.tensors.push_back({78, dtype, output, {1, 2, 2, 2, 2}});
  program.tensors.push_back({79, dtype, input, {1, 2}});
  program.tensors.push_back({80, dtype, output, {1, 2}});
  program.tensors.push_back({81, dtype, output, {1, 2}});
  program.operations = {
      {1, Opcode::Add, {1, 2}, {3}, {}},
      {2, Opcode::Multiply, {1, 2}, {4}, {}},
      {3, Opcode::BiasAdd, {1, 5}, {6}, {}},
      {4, Opcode::Linear, {1, 7, 8}, {9}, {}},
      {5, Opcode::RmsNormModulate, {10, 11, 12, 13}, {14},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5)}},
      {6, Opcode::SwiGlu, {15}, {16}, {}},
      {7, Opcode::ResidualGate, {17, 18, 19}, {20}, {}},
      {8, Opcode::Attention, {21, 22, 23}, {24},
       {Attribute::f64(AttrKey::AttentionScale, 1.0),
        Attribute::u64(AttrKey::BlockSize, 32)}},
      {9, Opcode::QkNormPartialRope, {25, 26, 27, 28}, {29},
       {Attribute::u64(AttrKey::Heads, 1),
        Attribute::u64(AttrKey::HeadDim, 2),
        Attribute::u64(AttrKey::RotaryDim, 2),
        Attribute::u64(AttrKey::BlockSize, 32),
        Attribute::f64(AttrKey::Epsilon, 1.0e-5)}},
      {10, Opcode::H3AdaLNSelect, {30, 31}, {32, 33, 34, 35, 36, 37}, {}},
      {11, Opcode::H3DeinterleaveQkv, {38}, {39, 40, 41},
       {Attribute::u64(AttrKey::Heads, 1),
        Attribute::u64(AttrKey::HeadDim, 2)}},
      {12, Opcode::H3DeinterleaveQkvWeight, {42}, {43, 44, 45},
       {Attribute::u64(AttrKey::Heads, 1),
        Attribute::u64(AttrKey::HeadDim, 2)}},
      {13, Opcode::Barrier, {}, {}, {}},
      {14, Opcode::DequantizeInt4, {46, 47}, {48},
       {Attribute::u64(AttrKey::GroupSize, 16)}},
      {15, Opcode::SiLU, {49}, {50}, {}},
      {16, Opcode::RmsNorm, {51, 52}, {53},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32)}},
      {17, Opcode::Fill, {}, {54},
       {Attribute::f64(AttrKey::Value, 2.5)}},
      {18, Opcode::GatherRows, {55, 56}, {57}, {}},
      {19, Opcode::IndexedUpdateRows, {58, 59, 60}, {61}, {}},
      {20, Opcode::Cast, {49}, {62}, {}},
      {21, Opcode::SelectRowChunks, {63, 64}, {65, 66}, {}},
      {22, Opcode::LinearBlend, {67, 68, 69}, {70}, {}},
      {23, Opcode::FlowEulerStep, {71, 72, 73, 74}, {75},
       {Attribute::u64(AttrKey::StepIndex, 0U)}},
      {24, Opcode::Patchify3D, {76}, {77},
       {Attribute::u64(AttrKey::PatchT, 1U),
        Attribute::u64(AttrKey::PatchH, 2U),
        Attribute::u64(AttrKey::PatchW, 2U)}},
      {25, Opcode::Unpatchify3D, {77}, {78},
       {Attribute::u64(AttrKey::PatchT, 1U),
        Attribute::u64(AttrKey::PatchH, 2U),
        Attribute::u64(AttrKey::PatchW, 2U)}},
      {26, Opcode::Gelu, {79}, {80},
       {Attribute::u64(
           AttrKey::Approximation,
           static_cast<std::uint64_t>(GeluApproximation::Tanh))}},
      {27, Opcode::Gelu, {79}, {81},
       {Attribute::u64(
           AttrKey::Approximation,
           static_cast<std::uint64_t>(GeluApproximation::ExactErf))}},
  };
  return program;
}

void test_cpu_all_opcodes_and_float_dtypes() {
  using namespace dif::ir;
  for (const auto dtype : {DType::F32, DType::BF16, DType::F16}) {
    const auto program = all_opcodes_program(dtype);
    dif::runtime::TensorMap inputs;
    const auto bind = [&](std::uint32_t id, std::vector<std::uint64_t> dims,
                          std::vector<float> values) {
      inputs.emplace(id, float_tensor(dtype, std::move(dims), values));
    };
    const auto bind_i32 = [&](std::uint32_t id,
                              std::vector<std::uint64_t> dims,
                              const std::vector<std::int32_t> &values) {
      dif::runtime::Tensor tensor{DType::I32, std::move(dims), {}};
      tensor.bytes.resize(values.size() * sizeof(std::int32_t));
      std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
      tensor.validate();
      inputs.emplace(id, std::move(tensor));
    };
    bind(1, {1, 2}, {1, 2});
    bind(2, {1, 2}, {3, 4});
    bind(5, {2}, {10, 20});
    bind(7, {2, 2}, {1, 0, 0, 1});
    bind(8, {2}, {0.5F, -0.5F});
    bind(10, {1, 4}, {1, 1, 1, 1});
    bind(11, {4}, {1, 1, 1, 1});
    bind(12, {1, 4}, {0, 0, 0, 0});
    bind(13, {1, 4}, {0, 0, 0, 0});
    bind(15, {1, 4}, {2, 3, 0, 0});
    bind(17, {1, 2}, {1, 2});
    bind(18, {1, 2}, {3, 4});
    bind(19, {1, 2}, {0, 1});
    bind(21, {2, 1, 1}, {0, 0});
    bind(22, {2, 1, 1}, {0, 0});
    bind(23, {2, 1, 1}, {2, 4});
    bind(25, {1, 1, 2}, {1, 0});
    bind(26, {2}, {1, 1});
    bind(27, {1, 2}, {1, 1});
    bind(28, {1, 2}, {0, 0});
    std::vector<float> projected(18);
    for (std::size_t i = 0; i < projected.size(); ++i)
      projected[i] = static_cast<float>(i);
    bind(30, {1, 18}, projected);
    dif::runtime::Tensor indices{DType::I32, {1}, {}};
    indices.bytes.resize(sizeof(std::int32_t));
    const std::int32_t zero = 0;
    std::memcpy(indices.bytes.data(), &zero, sizeof(zero));
    inputs.emplace(31, std::move(indices));
    bind(38, {1, 6}, {1, 2, 3, 4, 5, 6});
    bind(42, {6, 1}, {1, 2, 3, 4, 5, 6});
    dif::runtime::Tensor packed{DType::I8, {1, 8}, {}};
    packed.bytes = {0x98U, 0xbaU, 0xdcU, 0xfeU,
                    0x10U, 0x32U, 0x54U, 0x76U};
    inputs.emplace(46, std::move(packed));
    bind(47, {1, 1}, {0.5F});
    bind(49, {1, 2}, {-1.0F, 2.0F});
    bind(51, {2, 2}, {3.0F, 4.0F, 0.0F, 0.0F});
    bind(52, {2}, {1.0F, 2.0F});
    bind(55, {3, 2}, {1, 2, 3, 4, 5, 6});
    bind_i32(56, {2}, {2, 0});
    bind(58, {3, 2}, {1, 2, 3, 4, 5, 6});
    bind(59, {2, 2}, {10, 11, 20, 21});
    bind_i32(60, {3}, {-1, 1, 0});
    bind(63, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
    bind_i32(64, {3}, {1, 0, 1});
    bind(67, {1, 2}, {2, 6});
    bind(68, {1, 2}, {10, 14});
    inputs.emplace(69, f32_tensor({1}, {0.25F}));
    bind(71, {1, 2}, {1, 2});
    bind(72, {1, 2}, {0.5F, -1.0F});
    inputs.emplace(73, f32_tensor({1}, {0.25F}));
    inputs.emplace(74, f32_tensor({2}, {0.75F, 0.5F}));
    bind(76, {1, 2, 2, 2, 2},
         {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    bind(79, {1, 2}, {-1.0F, 1.0F});

    auto executor = dif::runtime::make_cpu_executor();
    dif::runtime::RunOptions options;
    options.warmups = 0;
    options.iterations = 1;
    const auto result = executor->run(program, inputs, options);
    const auto tolerance = dtype == DType::F32 ? 1.0e-6F : 1.0e-2F;
    const auto check = [&](std::uint32_t id, const std::vector<float> &expected,
                           const char *message) {
      const auto values = float_values(result.outputs.at(id));
      expect(values.size() == expected.size(), message);
      for (std::size_t i = 0; i < std::min(values.size(), expected.size()); ++i)
        expect(std::abs(values[i] - expected[i]) <= tolerance, message);
    };
    check(3, {4, 6}, "CPU typed Add");
    check(4, {3, 8}, "CPU typed Multiply");
    check(6, {11, 22}, "CPU typed BiasAdd");
    check(9, {1.5F, 1.5F}, "CPU typed Linear");
    const auto rms_value = 1.0F / std::sqrt(1.0F + 1.0e-5F);
    check(14, {rms_value, rms_value, rms_value, rms_value},
          "CPU typed RmsNormModulate");
    check(16, {0, 0}, "CPU typed SwiGlu");
    check(20, {1, 6}, "CPU typed ResidualGate");
    check(24, {3, 3}, "CPU typed Attention");
    const auto normalized = 1.0F / std::sqrt(0.5F + 1.0e-5F);
    check(29, {normalized, 0}, "CPU typed QkNormPartialRope");
    check(32, {0}, "CPU typed H3AdaLNSelect chunk 0");
    check(37, {5}, "CPU typed H3AdaLNSelect chunk 5");
    check(39, {1, 2}, "CPU typed H3DeinterleaveQkv q");
    check(40, {3, 4}, "CPU typed H3DeinterleaveQkv k");
    check(41, {5, 6}, "CPU typed H3DeinterleaveQkv v");
    check(43, {1, 2}, "CPU typed H3DeinterleaveQkvWeight q");
    check(44, {3, 4}, "CPU typed H3DeinterleaveQkvWeight k");
    check(45, {5, 6}, "CPU typed H3DeinterleaveQkvWeight v");
    check(48, {-4.0F, -3.5F, -3.0F, -2.5F, -2.0F, -1.5F, -1.0F,
               -0.5F, 0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F, 3.5F},
          "CPU typed DequantizeInt4");
    check(50, {-0.26894143F, 1.7615942F}, "CPU typed SiLU");
    const auto rms_inverse = 1.0F / std::sqrt(12.5F + 1.0e-5F);
    check(53, {3.0F * rms_inverse, 8.0F * rms_inverse, 0.0F, 0.0F},
          "CPU typed RmsNorm");
    check(54, {2.5F, 2.5F, 2.5F}, "CPU typed Fill");
    check(57, {5, 6, 1, 2}, "CPU typed GatherRows");
    check(61, {1, 2, 20, 21, 10, 11}, "CPU typed IndexedUpdateRows");
    check(62, {-1, 2}, "CPU typed Cast");
    check(65, {5, 6, 1, 2, 5, 6}, "CPU typed SelectRowChunks chunk 0");
    check(66, {7, 8, 3, 4, 7, 8}, "CPU typed SelectRowChunks chunk 1");
    check(70, {8, 12}, "CPU typed LinearBlend");
    check(75, {1.125F, 1.75F}, "CPU typed FlowEulerStep");
    check(77, {0, 1, 2, 3, 8, 9, 10, 11,
               4, 5, 6, 7, 12, 13, 14, 15},
          "CPU typed Patchify3D");
    check(78, {0, 1, 2, 3, 4, 5, 6, 7,
               8, 9, 10, 11, 12, 13, 14, 15},
          "CPU typed Unpatchify3D");
    check(80, {-0.1588079929F, 0.8411920071F}, "CPU typed tanh Gelu");
    check(81, {-0.1586552539F, 0.8413447461F},
          "CPU typed exact-erf Gelu");
  }
}

void test_krea2_gelu_creator_parity() {
  using namespace dif::ir;
  const std::vector<float> inputs{-8.0F, -5.0F, -3.0F, -1.0F, -0.5F,
                                  -0.1F, 0.0F,  0.1F,  0.5F,  1.0F,
                                  3.0F,  5.0F,  8.0F};
  const std::vector<float> f32_expected = floats_from_bits(
      {0x80000000U, 0xb4a00000U, 0xbb6e6200U, 0xbe229e90U, 0xbe1dfd26U,
       0xbd3c7c95U, 0x00000000U, 0x3d5d1d05U, 0x3eb1016dU, 0x3f57585cU,
       0x403fc468U, 0x409fffffU, 0x41000000U});
  const std::vector<float> bf16_expected{
      -0.0F,          -2.980232238769531e-7F, -0.003631591796875F,
      -0.1591796875F, -0.154296875F,           -0.046142578125F,
      0.0F,           0.053955078125F,          0.345703125F,
      0.83984375F,    3.0F,                     5.0F,
      8.0F};
  const std::vector<float> f16_expected{
      -0.0F,             -2.980232238769531e-7F,
      -0.0036373138427734375F, -0.1588134765625F,
      -0.154296875F,     -0.0460205078125F,
      0.0F,              0.053955078125F,
      0.345703125F,      0.84130859375F,
      2.99609375F,       5.0F,
      8.0F};

  const auto run_gate = [&](DType dtype, const std::vector<float> &expected,
                            const char *dtype_label) {
    Program program;
    program.tensors = {{1, dtype, TensorRole::Input, {inputs.size()}},
                       {2, dtype, TensorRole::Output, {inputs.size()}}};
    program.operations = {
        {1, Opcode::Gelu, {1}, {2},
         {Attribute::u64(
             AttrKey::Approximation,
             static_cast<std::uint64_t>(GeluApproximation::Tanh))}}};
    verify(program);
    dif::runtime::TensorMap bindings;
    bindings.emplace(1, float_tensor(dtype, {inputs.size()}, inputs));
    dif::runtime::TensorMap creator;
    creator.emplace(2, float_tensor(dtype, {expected.size()}, expected));
    dif::runtime::RunOptions options;
    options.warmups = 0;
    options.iterations = 1;
    options.minimum_free_bytes = 0;
    const auto cpu =
        dif::runtime::make_cpu_executor()->run(program, bindings, options);
    const dif::opt::AcceptanceBars bars{
        dtype == DType::F32
            ? 1.0e-6
            : (dtype == DType::BF16 ? 7.8125e-3 : 9.765625e-4),
        0.999999, 0.999, 1.001, 1.0e-3, ~std::uint64_t{0}};
    const dif::opt::AcceptanceGate gate(bars);
    const auto cpu_metrics = gate.measure(creator, cpu.outputs);
    expect(gate.judge(cpu_metrics, 0U) == dif::opt::Verdict::Accepted,
           "CPU tanh GELU matches the official Krea 2 creator fixture");
    std::cout << "GATE krea2_gelu creator=krea-2@db3984f dtype="
              << dtype_label << " backend=cpu"
              << " cosine=" << cpu_metrics.cosine_similarity
              << " rel_l2=" << cpu_metrics.relative_l2
              << " max_abs=" << cpu_metrics.max_absolute_error
              << " norm_ratio=" << cpu_metrics.norm_ratio
              << " nonfinite=" << cpu_metrics.nonfinite_count
              << " bit_mismatch=" << cpu_metrics.exact_mismatch_count
              << "\n";
    if (!dif::runtime::cuda_available())
      return;
    const auto cuda =
        dif::runtime::make_cuda_executor()->run(program, bindings, options);
    const auto cuda_metrics = gate.measure(creator, cuda.outputs);
    expect(gate.judge(cuda_metrics, 0U) == dif::opt::Verdict::Accepted,
           "CUDA tanh GELU matches the official Krea 2 creator fixture");
    std::cout << "GATE krea2_gelu creator=krea-2@db3984f dtype="
              << dtype_label << " backend=" << cuda.backend_name
              << " device=" << cuda.device_name
              << " cosine=" << cuda_metrics.cosine_similarity
              << " rel_l2=" << cuda_metrics.relative_l2
              << " max_abs=" << cuda_metrics.max_absolute_error
              << " norm_ratio=" << cuda_metrics.norm_ratio
              << " nonfinite=" << cuda_metrics.nonfinite_count
              << " bit_mismatch=" << cuda_metrics.exact_mismatch_count
              << "\n";
  };

  run_gate(DType::F32, f32_expected, "f32");
  run_gate(DType::BF16, bf16_expected, "bf16");
  run_gate(DType::F16, f16_expected, "f16");

  Program rejected;
  rejected.tensors = {{1, DType::F32, TensorRole::Input, {1}},
                      {2, DType::F32, TensorRole::Output, {1}}};
  rejected.operations = {{1, Opcode::Gelu, {1}, {2}, {}}};
  bool failed_closed = false;
  try {
    verify(rejected);
  } catch (const dif::Error &) {
    failed_closed = true;
  }
  expect(failed_closed,
         "Gelu without an explicit source approximation fails closed");
}

void test_krea2_euler_velocity_creator_parity() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {16}},
      {2, DType::BF16, TensorRole::Input, {16}},
      {3, DType::F32, TensorRole::Input, {1}},
      {4, DType::F32, TensorRole::Input, {1}},
      {5, DType::BF16, TensorRole::Output, {16}},
  };
  program.operations = {
      {1, Opcode::EulerVelocityStep, {1, 2, 3, 4}, {5}, {}},
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(
      1, float_tensor(DType::BF16, {16},
                      {-12.625F, -81.0F, 85.0F, 10.1875F, 55.75F,
                       0.765625F, 40.25F, -92.0F, -130.0F, 1.8046875F,
                       -60.5F, 67.0F, -20.125F, 42.5F, 86.5F, -127.5F}));
  bindings.emplace(
      2, float_tensor(DType::BF16, {16},
                      {96.0F, 152.0F, -28.125F, 1.296875F, -32.0F, 55.5F,
                       -124.0F, 10.5F, -38.0F, -1.890625F, -91.0F, 41.75F,
                       -83.0F, -71.0F, -236.0F, -50.75F}));
  // First two entries of the official Krea 2 Raw 1024/28 schedule. The
  // creator converts these F32 schedule values to Python scalars, then runs
  // eager BF16 multiply and add operations.
  bindings.emplace(3, f32_tensor({1}, {1.0F}));
  bindings.emplace(4, f32_tensor({1}, {0.9852563738822937F}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const std::vector<std::uint16_t> creator_bits{
      49505U, 49830U, 17067U, 16675U, 16993U, 48464U, 16936U, 49848U,
      49921U, 16363U, 49773U, 17029U, 49559U, 16942U, 17076U, 49918U};
  expect(cpu.outputs.at(5).byte_size() ==
                 creator_bits.size() * sizeof(std::uint16_t) &&
             std::memcmp(cpu.outputs.at(5).data(), creator_bits.data(),
                         cpu.outputs.at(5).byte_size()) == 0,
         "Krea 2 Euler velocity update preserves creator BF16 eager boundaries");
  if (!dif::runtime::cuda_available())
    return;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(5).bytes == cpu.outputs.at(5).bytes,
         "CUDA Krea 2 Euler velocity update is bit-exact to creator oracle");
  std::cout << "GATE krea2_euler bf16_bit_mismatch=0\n";
}

void test_krea2_schedule_and_cfg_creator_parity() {
  using namespace dif::frontend;
  const auto schedule = make_krea2_schedule();
  const std::vector<std::uint32_t> creator_schedule_bits{
      1065353216U, 1065221345U, 1065086385U, 1064948218U, 1064806731U,
      1064661807U, 1064513312U, 1064361120U, 1064205086U, 1064045062U,
      1063880897U, 1063712426U, 1063539479U, 1063361871U, 1063179416U,
      1062991909U, 1062799141U, 1062600883U, 1062396901U, 1062186943U,
      1061970738U, 1061748010U, 1061518452U, 1061281750U, 1061037564U,
      1060785532U, 1060525274U, 1060256375U, 1059978400U, 1059690881U,
      1059393319U, 1059085177U, 1058765881U, 1058434817U, 1058091316U,
      1057734670U, 1057364106U, 1056978794U, 1056191066U, 1055355914U,
      1054485415U, 1053577274U, 1052628999U, 1051637867U, 1050600906U,
      1049514862U, 1048176335U, 1045785792U, 1043273448U, 1040629755U,
      1035500870U, 1027445764U, 0U};
  expect(schedule.mu == 0.90625 &&
             schedule.timesteps.size() == creator_schedule_bits.size(),
         "Krea 2 Raw 1024 schedule has official mu and 52-step geometry");
  bool schedule_exact =
      schedule.timesteps.size() == creator_schedule_bits.size();
  if (schedule_exact) {
    for (std::size_t index = 0U; index < schedule.timesteps.size(); ++index) {
      const auto actual_bits =
          std::bit_cast<std::uint32_t>(schedule.timesteps[index]);
      if (actual_bits != creator_schedule_bits[index]) {
        std::cerr << "Krea schedule first mismatch index=" << index
                  << " expected_bits=" << creator_schedule_bits[index]
                  << " actual_bits=" << actual_bits
                  << " expected="
                  << std::bit_cast<float>(creator_schedule_bits[index])
                  << " actual=" << schedule.timesteps[index] << "\n";
        schedule_exact = false;
        break;
      }
    }
  }
  expect(schedule_exact,
         "Krea 2 Raw 1024/52 schedule is bit-exact to creator PyTorch");

  Krea2ScheduleConfig turbo_config;
  turbo_config.steps = 8U;
  turbo_config.fixed_mu = 1.15;
  const auto turbo_schedule = make_krea2_schedule({}, turbo_config);
  const std::vector<std::uint32_t> creator_turbo_schedule_bits{
      0x3f800000U, 0x3f74ebd8U, 0x3f678f54U,
      0x3f572119U, 0x3f426f4fU, 0x3f2791b1U,
      0x3f0349c0U, 0x3e9f2e6dU, 0x00000000U};
  bool turbo_schedule_exact =
      turbo_schedule.mu == 1.15 &&
      turbo_schedule.timesteps.size() == creator_turbo_schedule_bits.size();
  if (turbo_schedule_exact) {
    for (std::size_t index = 0U; index < turbo_schedule.timesteps.size();
         ++index) {
      if (std::bit_cast<std::uint32_t>(turbo_schedule.timesteps[index]) !=
          creator_turbo_schedule_bits[index]) {
        turbo_schedule_exact = false;
        break;
      }
    }
  }
  expect(turbo_schedule_exact,
         "Krea 2 Turbo 1024/8 fixed-mu schedule is bit-exact to creator PyTorch");

  const auto turbo_euler = make_krea2_euler_step({16U});
  expect(turbo_euler.program.operations.size() == 1U &&
             turbo_euler.program.operations.front().opcode ==
                 dif::ir::Opcode::EulerVelocityStep,
         "Krea 2 Turbo omits CFG and lowers directly to one Euler update");

  const auto build = make_krea2_cfg_euler_step({16U});
  dif::runtime::TensorMap bindings;
  bindings.emplace(
      build.sample_input,
      float_tensor(dif::ir::DType::BF16, {16U},
                   {-12.625F, -81.0F, 85.0F, 10.1875F, 55.75F, 0.765625F,
                    40.25F, -92.0F, -130.0F, 1.8046875F, -60.5F, 67.0F,
                    -20.125F, 42.5F, 86.5F, -127.5F}));
  bindings.emplace(
      build.conditional_velocity_input,
      float_tensor(dif::ir::DType::BF16, {16U},
                   {96.0F, 152.0F, -28.125F, 1.296875F, -32.0F, 55.5F,
                    -124.0F, 10.5F, -38.0F, -1.890625F, -91.0F, 41.75F,
                    -83.0F, -71.0F, -236.0F, -50.75F}));
  bindings.emplace(
      build.unconditional_velocity_input,
      float_tensor(dif::ir::DType::BF16, {16U},
                   {25.25F, -81.0F, 13.75F, -5.5F, 9.125F, 22.0F, -31.5F,
                    7.25F, -10.75F, 4.5F, 8.0F, -15.25F, 20.5F, -19.0F,
                    40.0F, 11.0F}));
  bindings.emplace(build.guidance_input,
                   float_tensor(dif::ir::DType::BF16, {1U}, {3.5F}));
  bindings.emplace(build.current_timestep_input, f32_tensor({1U}, {1.0F}));
  bindings.emplace(build.next_timestep_input,
                   f32_tensor({1U}, {0.9921398758888245F}));
  bindings.emplace(build.negative_one_constant,
                   float_tensor(dif::ir::DType::BF16, {1U}, {-1.0F}));

  const std::vector<std::pair<std::uint32_t, std::vector<std::uint16_t>>>
      creator_bits{
          {build.difference_output,
           {17038U, 17257U, 49704U, 16602U, 49700U, 16902U, 49849U, 16464U,
            49626U, 49356U, 49862U, 16996U, 49871U, 49744U, 50058U, 49783U}},
          {build.guided_delta_output,
           {17272U, 17484U, 49939U, 16831U, 49936U, 17130U, 50082U, 16694U,
            49855U, 49586U, 50093U, 17224U, 50101U, 49974U, 50290U, 50008U}},
          {build.velocity_output,
           {17324U, 17522U, 49967U, 16841U, 49968U, 17196U, 50144U, 16815U,
            49926U, 49601U, 50138U, 17266U, 50142U, 50045U, 50326U, 50053U}},
          {build.sample_output,
           {49525U, 49841U, 17069U, 16672U, 16997U, 48918U, 16943U, 49848U,
            49921U, 16383U, 49764U, 17026U, 49541U, 16946U, 17088U, 49915U}},
      };
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(build.program, bindings, options);
  for (const auto &[id, bits] : creator_bits) {
    const auto &actual = cpu.outputs.at(id);
    expect(actual.byte_size() == bits.size() * sizeof(std::uint16_t) &&
               std::memcmp(actual.data(), bits.data(), actual.byte_size()) == 0,
           "Krea 2 CFG/Euler CPU boundary is bit-exact to creator CUDA BF16");
  }
  if (dif::runtime::cuda_available()) {
    const auto cuda =
        dif::runtime::make_cuda_executor()->run(build.program, bindings, options);
    for (const auto &[id, bits] : creator_bits) {
      const auto &actual = cuda.outputs.at(id);
      expect(actual.byte_size() == bits.size() * sizeof(std::uint16_t) &&
                 std::memcmp(actual.data(), bits.data(), actual.byte_size()) == 0,
             "Krea 2 CFG/Euler CUDA boundary is bit-exact to creator CUDA BF16");
    }
  }
  std::cout << "GATE krea2_schedule points=53 f32_bit_mismatch="
            << (schedule_exact ? 0 : 1)
            << " turbo_points=9 turbo_f32_bit_mismatch="
            << (turbo_schedule_exact ? 0 : 1)
            << " cfg_euler_bf16_bit_mismatch=0 raw_mu=" << schedule.mu
            << " turbo_mu=" << turbo_schedule.mu << "\n";
}

void test_krea2_real_dimension_frontend_scaffold() {
  using namespace dif::frontend;
  using namespace dif::ir;
  const Krea2Config config;
  const auto architecture = inspect_krea2_architecture(config);
  expect(architecture.latent_height == 128U &&
             architecture.latent_width == 128U &&
             architecture.image_grid_height == 64U &&
             architecture.image_grid_width == 64U &&
             architecture.image_tokens == 4096U &&
             architecture.combined_tokens == 4608U &&
             architecture.padded_tokens == 4608U &&
             architecture.patch_input_dim == 64U &&
             architecture.patch_output_dim == 64U,
         "Krea 2 default geometry matches the official 1024px architecture");
  expect(Krea2Config::kFeatures == 6144U && Krea2Config::kHeads == 48U &&
             Krea2Config::kKvHeads == 12U &&
             Krea2Config::kHeadDim == 128U &&
             Krea2Config::kMlpDim == 16384U &&
             Krea2Config::kLayers == 28U &&
             Krea2Config::kTextDim == 2560U &&
             Krea2Config::kTextLayers == 12U,
         "Krea 2 frontend pins released checkpoint dimensions");

  const auto build = make_krea2_time_conditioning(config);
  verify(build.program);
  expect(build.program.tensors.size() == 15U &&
             build.program.operations.size() == 8U &&
             build.checkpoint_tensors.size() == 6U &&
             build.checkpoint_names ==
                 std::vector<std::string>{"tmlp.0.weight", "tmlp.0.bias",
                                          "tmlp.2.weight", "tmlp.2.bias",
                                          "tproj.1.weight", "tproj.1.bias"},
         "Krea 2 time scaffold has the exact creator tensor inventory");
  expect(build.program.tensor(build.timestep_input)->dtype == DType::BF16 &&
             build.program.tensor(build.timestep_input)->dims ==
                 std::vector<std::uint64_t>{1U} &&
             build.program.tensor(build.timestep_embedding)->dtype ==
                 DType::F32 &&
             build.program.tensor(build.timestep_embedding)->dims ==
                 std::vector<std::uint64_t>({1U, 256U}) &&
             build.program.tensor(build.timestep_output)->dims ==
                 std::vector<std::uint64_t>({1U, 6144U}) &&
             build.program.tensor(build.modulation_output)->dims ==
                 std::vector<std::uint64_t>({1U, 36864U}),
         "Krea 2 time scaffold preserves source BF16/F32 boundaries and real dims");
  std::uint64_t gelu_count = 0U;
  std::uint64_t linear_count = 0U;
  std::uint64_t addmm_linear_count = 0U;
  for (const auto &operation : build.program.operations) {
    gelu_count += operation.opcode == Opcode::Gelu ? 1U : 0U;
    linear_count += operation.opcode == Opcode::Linear ? 1U : 0U;
    addmm_linear_count +=
        operation.opcode == Opcode::Linear &&
                operation.u64(
                    AttrKey::LinearBiasMode,
                    static_cast<std::uint64_t>(LinearBiasMode::Epilogue)) ==
                    static_cast<std::uint64_t>(LinearBiasMode::Addmm)
            ? 1U
            : 0U;
    expect(operation.opcode != Opcode::H3AdaLNSelect &&
               operation.opcode != Opcode::H3DeinterleaveQkv &&
               operation.opcode != Opcode::H3DeinterleaveQkvWeight,
           "Krea 2 scaffold does not reuse H3-only frontend semantics");
  }
  expect(gelu_count == 2U && linear_count == 3U &&
             addmm_linear_count == 3U,
         "Krea 2 time scaffold matches creator operation and addmm ordering");
  const auto fingerprint = dif::hex_digest(dif::ir::fingerprint(build.program));
  const auto memory = dif::compiler::plan_memory(build.program, 256U);
  expect(fingerprint ==
             "dacf2f6296ceb27a0dfde2de6e3aa15a4e5fd0d1d2a3c0c1abbbc7aca281eb6c",
         "Krea 2 real-dimension scaffold has a stable DiffIR fingerprint");
  expect(memory.total_bytes == 531740672U,
         "Krea 2 streamed time scaffold memory plan is stable");
  std::cout << "GATE krea2_frontend source=krea-2@db3984f"
            << " fingerprint=" << fingerprint
            << " tensors=" << build.program.tensors.size()
            << " operations=" << build.program.operations.size()
            << " planned_bytes=" << memory.total_bytes << "\n";

  const auto conditioner_config = make_krea2_conditioner_config();
  const auto conditioner =
      build_qwen3vl_conditioner_program(546U, conditioner_config);
  verify(conditioner.program);
  expect(conditioner.attention_mask_input_id != 0U &&
             conditioner.position_ids_input_id != 0U &&
             conditioner.conditioning_output_ids.size() == 12U &&
             conditioner.bindings.size() == 397U &&
             conditioner.attention_operations == 36U &&
             conditioner.linear_operations == 252U,
         "Krea 2 conditioner exposes the exact masked 36-layer 12-tap Qwen3-VL contract");
  for (const auto output_id : conditioner.conditioning_output_ids) {
    const auto *output = conditioner.program.tensor(output_id);
    expect(output && output->dtype == DType::BF16 &&
               output->dims ==
                   std::vector<std::uint64_t>({512U, 2560U}),
           "each Krea 2 conditioner tap is BF16 [512,2560]");
  }

  const auto text_fusion = make_krea2_text_fusion();
  verify(text_fusion.program);
  expect(text_fusion.checkpoint_tensors.size() == 54U &&
             text_fusion.block_outputs.size() == 4U &&
             text_fusion.program.operations.size() == 122U &&
             text_fusion.program.tensor(text_fusion.conditioning_output)->dims ==
                 std::vector<std::uint64_t>({1U, 512U, 6144U}),
         "Krea 2 text fusion is a real-dimension shared DiffIR program");
  for (const auto &operation : text_fusion.program.operations)
    expect(operation.opcode != Opcode::H3AdaLNSelect &&
               operation.opcode != Opcode::H3DeinterleaveQkv &&
               operation.opcode != Opcode::H3DeinterleaveQkvWeight,
           "Krea 2 text fusion contains no H3-only runtime operation");

  const auto denoiser = make_krea2_denoiser();
  verify(denoiser.program);
  const auto *velocity = denoiser.program.tensor(denoiser.velocity_output);
  expect(denoiser.checkpoint_tensors.size() == 376U &&
             denoiser.block_outputs.size() == 28U &&
             denoiser.program.operations.size() == 1585U && velocity &&
             velocity->dtype == DType::BF16 &&
             velocity->dims == std::vector<std::uint64_t>({1U,4096U,64U}),
         "Krea 2 full denoiser is one real-dimension shared DiffIR program");
  expect(std::count_if(denoiser.program.operations.begin(),
                       denoiser.program.operations.end(),
                       [](const auto &operation) {
                         return operation.opcode == Opcode::Concat;
                       }) == 1,
         "Krea 2 full denoiser uses the generic text/image concat operation");
  expect(std::count_if(denoiser.program.operations.begin(),
                       denoiser.program.operations.end(),
                       [](const auto &operation) {
                         return operation.opcode == Opcode::AffineLastDim;
                       }) == 56,
         "Krea 2 full denoiser represents per-feature modulation with the "
         "shared affine operation");
  for (const auto &operation : denoiser.program.operations)
    expect(operation.opcode != Opcode::H3AdaLNSelect &&
               operation.opcode != Opcode::H3DeinterleaveQkv &&
               operation.opcode != Opcode::H3DeinterleaveQkvWeight,
           "Krea 2 full denoiser contains no H3-only runtime operation");
}

void test_backend_neutral_flow_scheduler() {
  using namespace dif::ir;
  const auto video_schedule =
      dif::sampling::make_exponential_shifted_schedule(5U, 12.0F);
  const auto audio_schedule =
      dif::sampling::make_exponential_shifted_schedule(5U, 3.0F);
  expect(video_schedule.sigmas.size() == 5U &&
             video_schedule.timesteps.size() == 4U &&
             video_schedule.sigmas.front() == 1.0F &&
             video_schedule.sigmas.back() == 0.0F &&
             video_schedule.timesteps.front() == 0.0F,
         "shifted video schedule includes terminal zero and four evaluations");
  expect(audio_schedule.sigmas.size() == 5U &&
             audio_schedule.timesteps.size() == 4U &&
             audio_schedule.sigmas[1] < video_schedule.sigmas[1],
         "modality shifts remain explicit schedule inputs");
  expect(video_schedule.sigmas == floats_from_bits(
             {0x3f800000U, 0x3f7914c2U, 0x3f6c4ec5U, 0x3f4ccccdU,
              0x00000000U}) &&
             video_schedule.timesteps == floats_from_bits(
                 {0x00000000U, 0x3cdd67c0U, 0x3d9d89d8U, 0x3e4cccccU}) &&
             audio_schedule.sigmas == floats_from_bits(
                 {0x3f800000U, 0x3f666666U, 0x3f400000U, 0x3f000000U,
                  0x00000000U}) &&
             audio_schedule.timesteps == floats_from_bits(
                 {0x00000000U, 0x3dccccd0U, 0x3e800000U, 0x3f000000U}),
         "shifted schedules are byte-exact to the pinned H3 source fixture");

  const auto simple_av = dif::sampling::make_h3_simple_av_schedule(7U);
  expect(simple_av.video_sigmas == floats_from_bits(
             {0x3f800000U, 0x3f7c8470U, 0x3f77c517U, 0x3f70f966U,
              0x3f6670b6U, 0x3f53e9c6U, 0x3f2abba6U, 0x00000000U}) &&
             simple_av.audio_sigmas == floats_from_bits(
                 {0x3f800000U, 0x3f729d96U, 0x3f61f9adU, 0x3f4ce541U,
                  0x3f31537dU, 0x3f0bb9a8U, 0x3eaacca4U,
                  0x00000000U}),
         "H3 simple AV schedule is byte-exact to ComfyUI's 1000-point table and mapped audio shift");

  const auto flux2_schedule =
      dif::sampling::make_flux2_klein_schedule(4U, 4096U);
  expect(flux2_schedule ==
             floats_from_bits({0x3f800000U, 0x3f77a67bU, 0x3f687c1eU,
                               0x3f446737U, 0x00000000U}),
         "FLUX.2 generalized-time schedule is byte-exact to creator F32");
  expect(std::abs(dif::sampling::flux2_empirical_mu(4096U, 50U) -
                  2.0233511571292637) < 1.0e-15,
         "FLUX.2 empirical mu matches creator double arithmetic");

  auto trajectory = floats_from_bits(
      {0x00000000U, 0x3de3166fU, 0x3e61aff2U, 0x3ea78611U,
       0x3edc233eU, 0x3f0704b2U, 0x3f1e4d7cU, 0x3f33a279U});
  const std::array<std::vector<float>, 4> velocities = {
      floats_from_bits({0x3e000000U, 0x3dfd6467U, 0x3df59f35U,
                        0x3de8d8f2U, 0x3dd7543eU, 0x3dc16c78U,
                        0x3da793dfU, 0x3d8a5141U}),
      floats_from_bits({0x3e780aa5U, 0x3e6c7f68U, 0x3e5c22afU,
                        0x3e4749d2U, 0x3e2e618dU, 0x3e11ebc6U,
                        0x3de4f9dcU, 0x3da171efU}),
      floats_from_bits({0x3ea87ef0U, 0x3e99ac95U, 0x3e87b8baU,
                        0x3e6601fcU, 0x3e37e2e6U, 0x3e0604b9U,
                        0x3da2d71eU, 0x3cd94dffU}),
      floats_from_bits({0x3ebb4ff6U, 0x3ea08f7aU, 0x3e828992U,
                        0x3e43b5a7U, 0x3dfcb6dfU, 0x3d59b8b4U,
                        0xbc94d7c8U, 0xbdb68622U})};
  const std::array<std::vector<float>, 4> expected_steps = {
      floats_from_bits({0x3b5d67c0U, 0x3de9efa2U, 0x3e6501aaU,
                        0x3ea918d4U, 0x3edd97b4U, 0x3f07abfbU,
                        0x3f1ede6bU, 0x3f341a1aU}),
      floats_from_bits({0x3c7d5f3dU, 0x3e00c4b2U, 0x3e6ffd8aU,
                        0x3eae119fU, 0x3ee1f16cU, 0x3f097df4U,
                        0x3f204c04U, 0x3f351be1U}),
      floats_from_bits({0x3d653f22U, 0x3e269888U, 0x3e88b30bU,
                        0x3ebc3920U, 0x3eed4257U, 0x3f0d9d9bU,
                        0x3f22cd5bU, 0x3f35f1d7U}),
      floats_from_bits({0x3eb28176U, 0x3ed3bf0cU, 0x3ef1211aU,
                        0x3f0540e5U, 0x3f0fe6a8U, 0x3f188071U,
                        0x3f1f14c3U, 0x3f23b13aU})};
  for (std::size_t step = 0U; step < velocities.size(); ++step) {
    dif::sampling::h3_euler_step_in_place(
        trajectory, velocities[step], 4U, 0U,
        video_schedule.timesteps[step], video_schedule.sigmas[step],
        video_schedule.sigmas[step + 1U]);
    expect(trajectory == expected_steps[step],
           "host H3 Euler trajectory is byte-exact to pinned source");
  }

  // PyTorch 2.12.1 CPU oracle for ComfyUI sample_res_multistep, eta=0,
  // cfg_pp=false.  This gates the interior second-order update separately
  // from the Euler-only creator trajectory above.
  auto res_trajectory =
      floats_from_bits({0x00000000U, 0x3dcccccdU, 0x3e4ccccdU,
                        0x3e99999aU, 0x3ecccccdU, 0x3f000000U,
                        0x3f19999aU, 0x3f333333U});
  const std::array<std::vector<float>, 4> res_velocities = {
      std::vector<float>{-0.109375F, -0.078125F, -0.046875F, -0.015625F,
                         0.015625F, 0.046875F, 0.078125F, 0.109375F},
      std::vector<float>{-0.21875F, -0.15625F, -0.09375F, -0.03125F,
                         0.03125F, 0.09375F, 0.15625F, 0.21875F},
      std::vector<float>{-0.328125F, -0.234375F, -0.140625F, -0.046875F,
                         0.046875F, 0.140625F, 0.234375F, 0.328125F},
      std::vector<float>{-0.4375F, -0.3125F, -0.1875F, -0.0625F,
                         0.0625F, 0.1875F, 0.3125F, 0.4375F}};
  const auto res_schedule =
      dif::sampling::make_exponential_shifted_schedule(5U, 12.0F);
  dif::sampling::H3ResMultistepState res_state;
  for (std::size_t step = 0U; step < res_velocities.size(); ++step)
    dif::sampling::h3_res_multistep_step_in_place(
        res_trajectory, res_velocities[step], 4U, 0U,
        res_schedule.timesteps[step], res_schedule.sigmas[step],
        res_schedule.sigmas[step + 1U], res_state);
  const auto res_expected =
      floats_from_bits({0xbedbc7b2U, 0xbe539250U, 0x3c835628U,
                        0x3e7467ddU, 0x3eec327aU, 0x3f2f1882U,
                        0x3f6817c7U, 0x3f908b87U});
  float res_max_abs = 0.0F;
  for (std::size_t index = 0U; index < res_trajectory.size(); ++index)
    res_max_abs = std::max(
        res_max_abs, std::abs(res_trajectory[index] - res_expected[index]));
  expect(res_max_abs <= 2.0e-7F,
         "H3 RES multistep trajectory matches PyTorch 2.12.1 scalar math");

  // ComfyUI keeps H3 audio inside the same packed RES state as video.  The
  // sampler carry runs on video sigma, while the network consumes physical
  // audio on its mapped sigma.  This PyTorch fixture covers every conversion,
  // the model-output wrapper, and the terminal process_latent_out scale.
  auto audio_carry =
      std::vector<float>{-0.75F, -0.25F, 0.25F, 0.75F};
  const std::array<std::vector<float>, 7> physical_audio_velocities = {
      std::vector<float>{-0.25F, -0.125F, 0.125F, 0.25F},
      std::vector<float>{-0.28125F, -0.140625F, 0.140625F, 0.28125F},
      std::vector<float>{-0.3125F, -0.15625F, 0.15625F, 0.3125F},
      std::vector<float>{-0.34375F, -0.171875F, 0.171875F, 0.34375F},
      std::vector<float>{-0.375F, -0.1875F, 0.1875F, 0.375F},
      std::vector<float>{-0.40625F, -0.203125F, 0.203125F, 0.40625F},
      std::vector<float>{-0.4375F, -0.21875F, 0.21875F, 0.4375F}};
  const std::array<std::vector<float>, 7> expected_audio_inputs = {
      floats_from_bits({0xbf400000U, 0xbe800000U, 0x3e800000U, 0x3f400000U}),
      floats_from_bits({0xbf435899U, 0xbe83589aU, 0x3e83589aU, 0x3f435899U}),
      floats_from_bits({0xbf486337U, 0xbe886338U, 0x3e886338U, 0x3f486337U}),
      floats_from_bits({0xbf4f7c16U, 0xbe8f7c16U, 0x3e8f7c16U, 0x3f4f7c16U}),
      floats_from_bits({0xbf59b6f7U, 0xbe99b6f8U, 0x3e99b6f8U, 0x3f59b6f7U}),
      floats_from_bits({0xbf69157cU, 0xbea9157eU, 0x3ea9157eU, 0x3f69157cU}),
      floats_from_bits({0xbf810a2cU, 0xbec2145dU, 0x3ec2145dU, 0x3f810a2cU})};
  dif::sampling::H3ResMultistepState audio_res_state;
  for (std::size_t step = 0U; step < physical_audio_velocities.size(); ++step) {
    auto model_input = audio_carry;
    dif::sampling::h3_av_audio_carry_to_model_input(
        model_input, audio_carry, 4U, 0U, simple_av.video_sigmas[step],
        simple_av.audio_sigmas[step]);
    float input_max_abs = 0.0F;
    for (std::size_t index = 0U; index < model_input.size(); ++index)
      input_max_abs = std::max(
          input_max_abs,
          std::abs(model_input[index] - expected_audio_inputs[step][index]));
    expect(input_max_abs <= 5.0e-7F,
           "H3 packed audio carry maps to the physical model-input fixture");
    dif::sampling::h3_res_multistep_av_audio_step_in_place(
        audio_carry, model_input, physical_audio_velocities[step], 4U, 0U,
        simple_av.video_sigmas[step], simple_av.audio_sigmas[step],
        simple_av.video_sigmas[step + 1U], 4.0F, audio_res_state);
  }
  dif::sampling::h3_av_audio_carry_to_physical_in_place(
      audio_carry, 4U, 0U, simple_av.video_sigmas.back(),
      simple_av.audio_sigmas.back(), 4.0F);
  const auto expected_physical_audio = floats_from_bits(
      {0xbf93b88eU, 0xbee77122U, 0x3ee77122U, 0x3f93b88eU});
  float audio_res_max_abs = 0.0F;
  for (std::size_t index = 0U; index < audio_carry.size(); ++index)
    audio_res_max_abs = std::max(
        audio_res_max_abs,
        std::abs(audio_carry[index] - expected_physical_audio[index]));
  expect(audio_res_max_abs <= 3.0e-7F,
         "H3 packed AV RES audio trajectory matches the ComfyUI wrapper fixture");

  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {2, 4}},
      {2, DType::F32, TensorRole::Input, {2, 4}},
      {3, DType::F32, TensorRole::Input, {1}},
      {4, DType::F32, TensorRole::Output, {2, 4}},
      {5, DType::F32, TensorRole::Input, {2, 4}},
      {6, DType::F32, TensorRole::Input, {2}},
      {7, DType::F32, TensorRole::Input, {3}},
      {8, DType::F32, TensorRole::Output, {2, 4}},
      {9, DType::F32, TensorRole::Input, {2, 4}},
      {10, DType::F32, TensorRole::Output, {2, 4}},
  };
  program.operations = {
      {1, Opcode::LinearBlend, {1, 2, 3}, {4}, {}},
      {2, Opcode::FlowEulerStep, {4, 5, 6, 7}, {8},
       {Attribute::u64(AttrKey::StepIndex, 0U)}},
      {3, Opcode::FlowEulerStep, {8, 9, 6, 7}, {10},
       {Attribute::u64(AttrKey::StepIndex, 1U)}},
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, f32_tensor({2, 4}, {0, 1, 2, 3, 4, 5, 6, 7}));
  bindings.emplace(2, f32_tensor({2, 4}, {7, 6, 5, 4, 3, 2, 1, 0}));
  bindings.emplace(3, f32_tensor({1}, {0.75F}));
  bindings.emplace(5, f32_tensor({2, 4}, {0.1F, 0.2F, 0.3F, 0.4F,
                                                   0.5F, 0.6F, 0.7F, 0.8F}));
  bindings.emplace(6, f32_tensor({2}, {0.0F, 0.25F}));
  bindings.emplace(7, f32_tensor({3}, {1.0F, 0.75F, 0.5F}));
  bindings.emplace(9, f32_tensor({2, 4}, {-0.2F, -0.1F, 0.0F, 0.1F,
                                                   0.2F, 0.3F, 0.4F, 0.5F}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto blended = float_values(reference.outputs.at(4));
  expect(blended == std::vector<float>({1.75F, 2.25F, 2.75F, 3.25F,
                                        3.75F, 4.25F, 4.75F, 5.25F}),
         "linear blend preserves source operand order");

  auto source_bindings = bindings;
  source_bindings.at(1) = f32_tensor(
      {2, 4}, floats_from_bits({0x00000000U, 0x3e4b6ff9U, 0x3ec761d7U,
                                0x3f108c69U, 0x3f37a4a7U, 0x3f576aa5U,
                                0x3f6e9a1dU, 0x3f7c466fU}));
  source_bindings.at(2) = f32_tensor(
      {2, 4}, floats_from_bits({0x3f800000U, 0x3f71e8b3U, 0x3f492fe8U,
                                0x3f0a5140U, 0x3e70e21cU, 0xbdc40ac0U,
                                0xbed51132U, 0xbf30d589U}));
  source_bindings.at(3) = f32_tensor({1}, {0.75F});
  const auto source_expected = floats_from_bits(
      {0x3e800000U, 0x3ec53e57U, 0x3efa2155U, 0x3f0efd9fU,
       0x3f18c99fU, 0x3f1b6fa6U, 0x3f185170U, 0x3f10ff71U});
  const auto source_cpu =
      dif::runtime::make_cpu_executor()->run(program, source_bindings, options);
  expect(float_values(source_cpu.outputs.at(4)) == source_expected,
         "CPU linear blend is byte-exact to pinned CUDA H3 scale_noise");

  auto bad_step = program;
  bad_step.operations[1].attributes[0] =
      Attribute::u64(AttrKey::StepIndex, 2U);
  bool rejected = false;
  try {
    dif::ir::verify(bad_step);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "flow Euler verifier rejects an out-of-range step index");

  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const auto source_cuda =
      dif::runtime::make_cuda_executor()->run(program, source_bindings, options);
  expect(float_values(source_cuda.outputs.at(4)) == source_expected,
         "CUDA linear blend is byte-exact to pinned CUDA H3 scale_noise");
  float maximum_absolute_error = 0.0F;
  for (const auto tensor_id : {4U, 8U, 10U}) {
    const auto expected = float_values(reference.outputs.at(tensor_id));
    const auto actual = float_values(candidate.outputs.at(tensor_id));
    for (std::size_t index = 0; index < expected.size(); ++index)
      maximum_absolute_error = std::max(
          maximum_absolute_error, std::abs(expected[index] - actual[index]));
  }
  expect(maximum_absolute_error <= 1.0e-6F,
         "CUDA flow scheduler primitives match CPU semantics");
  std::cout << "GATE flow_scheduler backend=" << candidate.backend_name
            << " device=" << candidate.device_name
            << " max_abs=" << maximum_absolute_error << "\n";
}

void test_h3_creator_noise_and_packing() {
  dif::TorchCpuMt19937 generator(42U);
  constexpr std::size_t video_value_count = 24U * 37U * 30U * 52U;
  const auto video_values =
      dif::torch_cpu_normal(generator, video_value_count);
  dif::runtime::Tensor video_raw{dif::ir::DType::F32,
                                 {1U, 24U, 37U, 30U, 52U}, {}};
  video_raw.bytes.resize(video_values.size() * sizeof(float));
  std::memcpy(video_raw.bytes.data(), video_values.data(),
              video_raw.bytes.size());
  const auto video_rows = dif::frontend::pack_h3_video_latent(video_raw);
  expect(video_rows.dims ==
             std::vector<std::uint64_t>({14430U, 96U}),
         "H3 124-frame video noise packs to the creator's t=37 row geometry");

  constexpr std::size_t audio_value_count = 32U * 2U * 207U;
  const auto audio_values =
      dif::torch_cpu_normal(generator, audio_value_count);
  dif::runtime::Tensor audio_raw{dif::ir::DType::F32,
                                 {1U, 32U, 2U, 207U}, {}};
  audio_raw.bytes.resize(audio_values.size() * sizeof(float));
  std::memcpy(audio_raw.bytes.data(), audio_values.data(),
              audio_raw.bytes.size());
  const auto audio_rows = dif::frontend::pack_h3_audio_latent(audio_raw);
  expect(audio_rows.dims == std::vector<std::uint64_t>({414U, 32U}),
         "H3 124-frame audio noise packs to the creator's 207-latent geometry");

  if (dif::torch_cpu_normal_uses_avx2()) {
    const auto video_hash = dif::hex_digest(dif::sha256(
        {video_rows.data(), static_cast<std::size_t>(video_rows.byte_size())}));
    const auto audio_hash = dif::hex_digest(dif::sha256(
        {audio_rows.data(), static_cast<std::size_t>(audio_rows.byte_size())}));
    expect(video_hash ==
               "6207522013a69ed8fbc0ed924c58aaae985760dac4690908a8dc914d1406e0fe",
           "H3 video noise is byte-exact to PyTorch 2.12.1 AVX2 seed 42");
    expect(audio_hash ==
               "31afc4bc194b6ab29f9162c01d0a7dac9176a5c6fb39c345bea4d1c41cb344e2",
           "H3 audio noise continues the same PyTorch generator byte-exactly");
    std::cout << "GATE h3_creator_noise rng=torch-cpu-avx2 seed=42"
                 " video_rows=14430 audio_rows=414 video_sha256="
              << video_hash << " audio_sha256=" << audio_hash << "\n";
  }
}

void test_h3_latent_handoff() {
  std::vector<float> video_values(2U * 96U, -1.0F);
  for (std::size_t index = 0U; index < 96U; ++index)
    video_values[96U + index] = static_cast<float>(index);
  const auto video_rows = f32_tensor({2U, 96U}, video_values);
  const auto video = dif::frontend::unpack_h3_video_rows(
      video_rows, 1U, 1U, 2U, 2U);
  expect(video.dims == std::vector<std::uint64_t>({1U, 24U, 1U, 2U, 2U}) &&
             std::equal(video.f32().begin(), video.f32().end(),
                        video_values.begin() + 96),
         "H3 video row handoff preserves channel-slowest source order");

  std::vector<float> audio_values(4U * 32U);
  for (std::size_t row = 0U; row < 4U; ++row) {
    for (std::size_t channel = 0U; channel < 32U; ++channel)
      audio_values[row * 32U + channel] =
          static_cast<float>(row * 100U + channel);
  }
  const auto audio_rows = f32_tensor({4U, 32U}, audio_values);
  const auto audio =
      dif::frontend::unpack_h3_audio_rows(audio_rows, 0U, 2U, false);
  expect(audio.dims == std::vector<std::uint64_t>({2U, 32U, 2U}) &&
             audio.f32()[0] == 0.0F && audio.f32()[1] == 100.0F &&
             audio.f32()[2] == 1.0F && audio.f32()[63] == 131.0F &&
             audio.f32()[64] == 200.0F && audio.f32()[127] == 331.0F,
         "H3 audio row handoff preserves stereo-major source order");
  const auto denormalized =
      dif::frontend::unpack_h3_audio_rows(audio_rows, 0U, 2U, true);
  expect(denormalized.f32()[0] == -0.020211687488382354F &&
             denormalized.f32()[1] ==
                 100.0F * 1.6895524230479284F - 0.020211687488382354F,
         "H3 audio handoff applies released per-channel latent normalization");

  const auto path = std::filesystem::temp_directory_path() /
                    "dif-h3-latent-handoff.safetensors";
  std::filesystem::remove(path);
  dif::frontend::write_h3_latent_handoff(path, video_rows, audio_rows);
  const auto file = dif::weights::read_safetensors(path);
  const auto mapped_video = dif::weights::map_safetensor(file,
                                                         "video_state_rows");
  const auto mapped_audio = dif::weights::map_safetensor(file,
                                                         "audio_state_rows");
  expect(mapped_video.dims == video_rows.dims &&
             mapped_video.byte_size() == video_rows.byte_size() &&
             std::equal(mapped_video.data(),
                        mapped_video.data() + mapped_video.byte_size(),
                        video_rows.data()) &&
             mapped_audio.dims == audio_rows.dims &&
             mapped_audio.byte_size() == audio_rows.byte_size() &&
             std::equal(mapped_audio.data(),
                        mapped_audio.data() + mapped_audio.byte_size(),
                        audio_rows.data()),
         "H3 SafeTensors handoff is byte-exact and Serenity-compatible");
  std::filesystem::remove(path);
}

void test_h3_media_handoff() {
  const auto decoded = f32_tensor(
      {1U, 3U, 1U, 1U, 2U},
      {0.0F, 1.0F, 0.5F, 0.25F, 1.0F, 0.0F});
  const auto rgb = dif::frontend::make_h3_rgb24_video(decoded);
  expect(rgb.frames == 1U && rgb.height == 1U && rgb.width == 2U &&
             rgb.minimum == 0.0F && rgb.maximum == 1.0F &&
             rgb.bytes == std::vector<std::uint8_t>(
                              {0U, 128U, 255U, 255U, 64U, 0U}),
         "H3 media handoff interleaves and quantizes source unit RGB");
}

void test_h3_conditioning_layout() {
  const std::vector<std::int32_t> text_tags = {1, 1, 0, 0, 1};
  const std::vector<dif::frontend::H3KeyframeAnchor> anchors = {
      dif::frontend::H3KeyframeAnchor::First,
      dif::frontend::H3KeyframeAnchor::Last};
  const auto layout = dif::frontend::make_h3_t2va_layout(
      text_tags, 3U, 4U, 6U, 4U, 1U, 2U, 2U, anchors);
  const auto plan = dif::frontend::make_h3_row_timestep_plan(
      layout, 0.25F, 0.5F, 0.999F, 1.0F);
  expect(layout.sequence_length == 43U && layout.text_indices.size() == 5U &&
             layout.video_indices.size() == 30U &&
             layout.audio_indices.size() == 8U &&
             layout.num_condition_video_rows == 12U &&
             layout.num_condition_audio_rows == 0U,
         "H3 conditioning layout preserves source row counts and padless document");

  const auto digest = [](const auto &values) {
    const auto bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t *>(values.data()),
        values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type));
    return dif::hex_digest(dif::sha256(bytes));
  };
  expect(digest(layout.position_ids) ==
             "1d1a17e46ee80b1cbfcadae76812877cc01ebdd2bdb300653adca6b7bf1a68c7",
         "H3 position grid is byte-exact to pinned packing.py source");
  expect(digest(layout.token_tags) ==
             "07840a7f949d6331f494f5616e1b273536e43495db5a602ae7e46776b8d536fa" &&
             digest(layout.text_indices) ==
                 "e528f4309e1413e6bc35aea5d8db8519384d2fcc33f9dd5d1126d73f104cf92a" &&
             digest(layout.video_indices) ==
                 "a1c53faa6d1b4955ca197908ccb671a04e58629df36f1adae2d321d42f1a12cf" &&
             digest(layout.audio_indices) ==
                 "7eb18d1ba2d7e7d8a7d1537de7e198525e9c9a315ca46a543ff62cf0f0cef16f",
         "H3 modality tags and row indices are byte-exact to source");
  expect(digest(layout.text_map) ==
             "96a7934a7f4846c901773035ba723b6e2f0a415cc5ceb1879aae488a2ef8c778" &&
             digest(layout.video_map) ==
                 "0665dd56aa4e884c33231c164ec22ed809a9274b3c9136fcaa5e8444a9f50995" &&
             digest(layout.audio_map) ==
                 "fd4c21c68231709e8759950bdba9c1e4e3da19721b871425e49b5db07e79f74c",
         "H3 inverse modality maps are byte-exact to source");
  expect(digest(plan.timesteps) ==
             "d0ad26e0325152a376cd3d05a2df2da2468a9d167f0140c7322222d6f8bcd9a4" &&
             digest(plan.timestep_indices) ==
                 "6b9b8dedb42abdd92fef06c90c22c4fda8af52defb42c38b5043720d18fcb331" &&
             digest(plan.adaln_indices) ==
                 "c1dbd818693e1d201de0bd3ec8e87deaa2fb28cc8f7a1a0b2b2a297a9d96e019",
         "H3 row timestep and AdaLN plans are byte-exact to source");

  const std::vector<float> video_sigmas = {1.0F, 0.25F, 0.0F};
  const std::vector<float> audio_sigmas = {1.0F, 0.5F, 0.0F};
  const auto fl_schedule = dif::frontend::make_h3_schedule_timestep_table(
      video_sigmas, audio_sigmas, true, false);
  expect(fl_schedule.evaluations == 2U && fl_schedule.tables == 3U &&
             fl_schedule.timesteps ==
                 std::vector<float>({0.0F, 0.999F, 0.999F,
                                     0.5F, 0.75F, 0.999F}),
         "H3 FL2VA prepared timestep slices preserve creator sorted-unique indices");
  const auto ref_schedule = dif::frontend::make_h3_schedule_timestep_table(
      video_sigmas, audio_sigmas, true, true);
  expect(ref_schedule.evaluations == 2U && ref_schedule.tables == 4U &&
             ref_schedule.timesteps ==
                 std::vector<float>({0.0F, 0.999F, 1.0F, 1.0F,
                                     0.5F, 0.75F, 0.999F, 1.0F}),
         "H3 Ref2VA prepared timestep slices preserve creator sorted-unique indices");

  const std::vector<dif::frontend::H3ReferenceGeometry> references = {
      {dif::frontend::H3ReferenceKind::Image, 1U, 4U, 6U, 0U},
      {dif::frontend::H3ReferenceKind::Video, 3U, 6U, 4U, 2U},
      {dif::frontend::H3ReferenceKind::Audio, 0U, 0U, 0U, 3U},
  };
  const std::vector<std::int32_t> ref_text_tags = {1, 0, 1, 0, 1};
  const auto ref_layout = dif::frontend::make_h3_ref2va_layout(
      ref_text_tags, references, 3U, 4U, 6U, 4U, 1U, 2U, 2U);
  expect(ref_layout.sequence_length == 65U &&
             ref_layout.text_indices.size() == 5U &&
             ref_layout.video_indices.size() == 42U &&
             ref_layout.audio_indices.size() == 18U &&
             ref_layout.num_condition_video_rows == 24U &&
             ref_layout.num_condition_audio_rows == 10U,
         "H3 Ref2VA layout preserves ordered mixed-reference row counts");
  expect(digest(ref_layout.position_ids) ==
             "9ae67e9914dc784fc136710f2c1d01324604d26ceaeda92ca9939f55954fef38" &&
             digest(ref_layout.token_tags) ==
                 "f68263bad5e8bebd13ba6ae653dbe87e3395dae46e6a4eb70bf072570fff7282" &&
             digest(ref_layout.text_indices) ==
                 "e528f4309e1413e6bc35aea5d8db8519384d2fcc33f9dd5d1126d73f104cf92a" &&
             digest(ref_layout.video_indices) ==
                 "3770b65430e6d89f17194ee974ae5d62f1032bff1a00eece38a6ce9673408098" &&
             digest(ref_layout.audio_indices) ==
                 "2e7b932c2f86d861770a277c779dc63c50dc7f46378229664f6203ca050502e2",
         "H3 Ref2VA positions, tags, and row indices are byte-exact to pinned creator packing_ref2va.py");
  expect(digest(ref_layout.text_map) ==
             "a4ae6210aca2932ba47c663f871be66f2ca989ee0f664109a7b963bb3335c791" &&
             digest(ref_layout.video_map) ==
                 "27542ead508d1e38b85708e5caba6cc74677d3e2e8db58da28b230d6213b13a1" &&
             digest(ref_layout.audio_map) ==
                 "37be623b2544f4effd2822deaf62d8043bc9667effbc5cc5a61c2becf855d266",
         "H3 Ref2VA inverse modality maps are byte-exact to creator indices");
}

void test_krea2_rotary_layout_mask_and_broadcast_oracle() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4, 3}},
      {2, DType::I32, TensorRole::Constant, {4}},
      {3, DType::I32, TensorRole::Constant, {4}},
      {4, DType::I32, TensorRole::Constant, {3}},
      {5, DType::F32, TensorRole::Output, {1, 4, 4}},
      {6, DType::F32, TensorRole::Output, {1, 4, 4}},
      {7, DType::BF16, TensorRole::Input, {1, 4, 2, 8}},
      {8, DType::BF16, TensorRole::Output, {1, 4, 2, 8}},
      {9, DType::Bool, TensorRole::Input, {1, 4}},
      {10, DType::BF16, TensorRole::Output, {1, 1, 4, 4}},
      {11, DType::BF16, TensorRole::Input, {1, 12}},
      {12, DType::BF16, TensorRole::Output, {1, 4}},
      {13, DType::BF16, TensorRole::Output, {1, 1, 4}},
      {14, DType::BF16, TensorRole::Output, {1, 4, 4}},
      {15, DType::BF16, TensorRole::Output, {1, 4, 4}},
      {16, DType::BF16, TensorRole::Input, {2, 3, 4}},
      {17, DType::BF16, TensorRole::Output, {2, 4, 3}},
      {18, DType::BF16, TensorRole::Input, {2, 1, 3}},
      {19, DType::BF16, TensorRole::Input, {2, 2, 3}},
      {20, DType::BF16, TensorRole::Output, {2, 3, 3}},
      {21, DType::BF16, TensorRole::Output, {1, 1, 4, 4}},
  };
  program.operations = {
      {1, Opcode::RotaryFrequency, {1, 2, 3, 4}, {5, 6},
       {Attribute::f64(AttrKey::Theta, 1000.0),
        Attribute::f64(AttrKey::Ntk, 1.0)}},
      {2, Opcode::RotaryApply, {7, 5, 6}, {8},
       {Attribute::u64(
           AttrKey::RotaryLayout,
           static_cast<std::uint64_t>(RotaryLayout::Interleaved))}},
      {3, Opcode::BooleanMaskToBias, {9}, {10}, {}},
      {4, Opcode::Slice, {11}, {12},
       {Attribute::u64(AttrKey::Axis, 1U),
        Attribute::u64(AttrKey::Start, 4U)}},
      {5, Opcode::Reshape, {12}, {13}, {}},
      {6, Opcode::BroadcastTo, {13}, {14}, {}},
      {7, Opcode::Sigmoid, {14}, {15}, {}},
      {8, Opcode::Permute, {16}, {17},
       {Attribute::u64(AttrKey::Permutation0, 0U),
        Attribute::u64(AttrKey::Permutation1, 2U),
        Attribute::u64(AttrKey::Permutation2, 1U)}},
      {9, Opcode::Concat, {18, 19}, {20},
       {Attribute::u64(AttrKey::Axis, 1U)}},
      {10, Opcode::BooleanMaskToBias, {9}, {21},
       {Attribute::boolean(AttrKey::MaskQueries, false)}},
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, f32_tensor(
                          {1, 4, 3},
                          {0, 0, 0, 0, 1, 2, 0, 2, 3, 0, 3, 4}));
  bindings.emplace(2, i32_tensor({4}, {0, 1, 2, 2}));
  bindings.emplace(3, i32_tensor({4}, {0, 0, 0, 1}));
  bindings.emplace(4, i32_tensor({3}, {2, 2, 4}));
  bindings.emplace(
      7, float_tensor(
             DType::BF16, {1, 4, 2, 8},
             {.25F, -.5F, .75F, -1.F, 1.25F, -1.5F, 1.75F, -2.F,
              .1F, .2F, .3F, .4F, .5F, .6F, .7F, .8F,
              .2F, -.3F, .4F, -.5F, .6F, -.7F, .8F, -.9F,
              .9F, -.8F, .7F, -.6F, .5F, -.4F, .3F, -.2F,
              -.1F, .2F, -.3F, .4F, -.5F, .6F, -.7F, .8F,
              .8F, .7F, .6F, .5F, .4F, .3F, .2F, .1F,
              1.F, .5F, 0.F, -.5F, -1.F, -1.5F, -2.F, -2.5F,
              -.2F, -.1F, 0.F, .1F, .2F, .3F, .4F, .5F}));
  bindings.emplace(9, bool_tensor({1, 4}, {1, 1, 0, 1}));
  bindings.emplace(11, float_tensor(DType::BF16, {1, 12},
                                    {-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6,
                                     7}));
  bindings.emplace(16, float_tensor(DType::BF16, {2, 3, 4},
                                    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                     12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                                     22, 23}));
  bindings.emplace(18, float_tensor(DType::BF16, {2, 1, 3},
                                    {1, 2, 3, 10, 11, 12}));
  bindings.emplace(19, float_tensor(DType::BF16, {2, 2, 3},
                                    {4, 5, 6, 7, 8, 9,
                                     13, 14, 15, 16, 17, 18}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);

  const std::vector<float> creator_cosine{
      1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 0.5403022766113281F,
      -0.416146844625473F, 0.9980006814002991F, 1.0F,
      -0.416146844625473F, -0.9899924993515015F, 0.9955033659934998F,
      1.0F, -0.9899924993515015F, -0.6536436080932617F,
      0.9920106530189514F};
  expect(float_values(cpu.outputs.at(5)) == creator_cosine,
         "Krea 2 rotary frequencies are bit-exact to creator F64 construction");
  const std::vector<std::uint16_t> creator_rotary_bits{
      16000,48896,16192,49024,16288,49088,16352,49152,15821,15949,16026,
      16077,16128,16154,16179,16205,15949,48794,16163,15753,16069,16215,
      16219,48985,16230,48973,16226,16007,15904,16159,16032,48697,48589,
      15949,48757,48865,16082,48939,48966,16187,16205,16179,48948,16046,
      48865,48759,15938,15859,16256,16128,15761,16125,48887,16350,49110,
      49199,48717,48589,48231,48587,15814,48818,16043,16140};
  expect(cpu.outputs.at(8).byte_size() ==
             creator_rotary_bits.size() * sizeof(std::uint16_t) &&
             std::memcmp(cpu.outputs.at(8).data(), creator_rotary_bits.data(),
                         cpu.outputs.at(8).byte_size()) == 0,
         "Krea 2 interleaved rotary apply is BF16 bit-exact to creator");
  const auto bias = float_values(cpu.outputs.at(10));
  expect(bias[0] == 0.0F && std::isinf(bias[2]) && bias[2] < 0.0F &&
             bias[3] == 0.0F && std::isinf(bias[8]) && bias[15] == 0.0F,
         "Krea 2 vector validity mask expands across a padding gap");
  const auto key_only_bias = float_values(cpu.outputs.at(21));
  expect(key_only_bias[8] == 0.0F && key_only_bias[9] == 0.0F &&
             std::isinf(key_only_bias[10]) && key_only_bias[10] < 0.0F &&
             key_only_bias[11] == 0.0F,
         "key-only padding leaves invalid query rows observable");
  expect(float_values(cpu.outputs.at(12)) ==
             std::vector<float>({0, 1, 2, 3}) &&
             float_values(cpu.outputs.at(14)) ==
                 std::vector<float>({0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3}),
         "slice, reshape, and right-aligned broadcast preserve creator layout");
  expect(float_values(cpu.outputs.at(17)) ==
             std::vector<float>({0,4,8,1,5,9,2,6,10,3,7,11,
                                 12,16,20,13,17,21,14,18,22,15,19,23}),
         "generic permute performs a bit-exact physical layout transform");
  expect(float_values(cpu.outputs.at(20)) ==
             std::vector<float>({1,2,3,4,5,6,7,8,9,
                                 10,11,12,13,14,15,16,17,18}),
         "generic concat joins row-major tensors along the selected axis");

  if (!dif::runtime::cuda_available())
    return;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(8).bytes == cpu.outputs.at(8).bytes,
         "CUDA interleaved rotary is BF16 bit-exact to CPU creator gate");
  expect(cuda.outputs.at(10).bytes == cpu.outputs.at(10).bytes &&
             cuda.outputs.at(12).bytes == cpu.outputs.at(12).bytes &&
             cuda.outputs.at(13).bytes == cpu.outputs.at(13).bytes &&
             cuda.outputs.at(14).bytes == cpu.outputs.at(14).bytes &&
             cuda.outputs.at(17).bytes == cpu.outputs.at(17).bytes &&
             cuda.outputs.at(20).bytes == cpu.outputs.at(20).bytes &&
             cuda.outputs.at(21).bytes == cpu.outputs.at(21).bytes,
         "CUDA mask and layout operations are bit-exact to CPU semantics");
}

void test_krea2_cudnn_masked_gqa_creator_oracle() {
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {1, 4, 2, 8}},
      {2, DType::BF16, TensorRole::Input, {1, 4, 1, 8}},
      {3, DType::BF16, TensorRole::Input, {1, 4, 1, 8}},
      {4, DType::Bool, TensorRole::Input, {1, 4}},
      {5, DType::BF16, TensorRole::Internal, {1, 1, 4, 4}},
      {6, DType::BF16, TensorRole::Output, {1, 4, 2, 8}},
  };
  program.operations = {
      {1, Opcode::BooleanMaskToBias, {4}, {5}, {}},
      {2, Opcode::Attention, {1, 2, 3, 5}, {6},
       {Attribute::u64(AttrKey::KvHeads, 1U),
        Attribute::u64(AttrKey::Implementation, 2U)}},
  };
  std::vector<float> q(64), k(32), v(32);
  for (std::size_t index = 0; index < q.size(); ++index)
    q[index] = (static_cast<float>(index % 17U) - 8.0F) / 8.0F;
  for (std::size_t index = 0; index < k.size(); ++index)
    k[index] = (static_cast<float>(index % 11U) - 5.0F) / 7.0F;
  for (std::size_t index = 0; index < v.size(); ++index)
    v[index] = (static_cast<float>(index % 13U) - 6.0F) / 9.0F;
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, float_tensor(DType::BF16, {1, 4, 2, 8}, q));
  bindings.emplace(2, float_tensor(DType::BF16, {1, 4, 1, 8}, k));
  bindings.emplace(3, float_tensor(DType::BF16, {1, 4, 1, 8}, v));
  bindings.emplace(4, bool_tensor({1, 4}, {1, 1, 0, 1}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const std::vector<std::uint16_t> creator_bits{
      48591,15424,48803,48723,48578,48794,48706,48545,15799,15951,48815,
      48747,48627,48814,48745,48624,48254,15815,48686,48495,15706,48833,
      48776,48671,15752,15928,48812,48742,48618,48812,48741,48616,48048,
      15834,48200,15818,15960,48814,48744,48623,48414,15767,48785,48688,
      48506,48728,48588,15416,15885,16000,15924,16018,16076,48908,48862,
      48805,15522,15880,48808,48734,48601,48807,48733,48598};
  const auto *actual = reinterpret_cast<const std::uint16_t *>(
      cuda.outputs.at(6).data());
  std::size_t mismatches = 0U;
  bool only_masked_query_row = true;
  for (std::size_t index = 0; index < creator_bits.size(); ++index) {
    if (actual[index] == creator_bits[index])
      continue;
    ++mismatches;
    only_masked_query_row =
        only_masked_query_row && index >= 32U && index < 48U;
  }
  // PyTorch's cuDNN boolean-mask route emits nonzero values for an all-false
  // query row, while direct cuDNN additive -inf bias emits zeros. Those rows
  // are padding: they are masked as keys in every block and never selected as
  // image output. Valid text/image query rows must remain bit-identical.
  expect(mismatches == 16U && only_masked_query_row,
         "native cuDNN masked GQA is bit-exact on every observable creator row and differs only on the all-false padding query");
  std::cout << "GATE krea2_masked_gqa valid_row_bit_mismatch=0"
            << " padded_query_bit_mismatch=" << mismatches << "\n";
}

void test_new_primitives_cuda_parity() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {3, 2}},
      {2, DType::F32, TensorRole::Constant, {2}},
      {3, DType::I32, TensorRole::Input, {2}},
      {4, DType::F32, TensorRole::Output, {2, 2}},
      {5, DType::F32, TensorRole::Output, {2, 2}},
      {6, DType::F32, TensorRole::Output, {2, 2}},
      {7, DType::F32, TensorRole::Output, {3, 2}},
      {8, DType::I32, TensorRole::Input, {3}},
      {9, DType::F32, TensorRole::Output, {3, 2}},
      {10, DType::BF16, TensorRole::Output, {3, 2}},
      {11, DType::BF16, TensorRole::Output, {3, 2}},
      {12, DType::F32, TensorRole::Output, {3, 2}},
      {13, DType::F32, TensorRole::Input, {2, 4}},
      {14, DType::I32, TensorRole::Input, {3}},
      {15, DType::F32, TensorRole::Output, {3, 2}},
      {16, DType::F32, TensorRole::Output, {3, 2}},
  };
  program.operations = {
      {1, Opcode::GatherRows, {1, 3}, {4}, {}},
      {2, Opcode::RmsNorm, {4, 2}, {5},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32)}},
      {3, Opcode::SiLU, {5}, {6}, {}},
      {4, Opcode::Fill, {}, {7}, {Attribute::f64(AttrKey::Value, 0.5)}},
      {5, Opcode::IndexedUpdateRows, {7, 6, 8}, {9}, {}},
      {6, Opcode::Cast, {9}, {10}, {}},
      {7, Opcode::SiLU, {10}, {11}, {}},
      {8, Opcode::Cast, {11}, {12}, {}},
      {9, Opcode::SelectRowChunks, {13, 14}, {15, 16}, {}},
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, f32_tensor({3, 2}, {1, 2, 3, 4, 5, 6}));
  bindings.emplace(2, f32_tensor({2}, {1, 2}));
  bindings.emplace(3, i32_tensor({2}, {2, 0}));
  bindings.emplace(8, i32_tensor({3}, {-1, 1, 0}));
  bindings.emplace(13, f32_tensor({2, 4}, {1, 2, 3, 4, 5, 6, 7, 8}));
  bindings.emplace(14, i32_tensor({3}, {1, 0, 1}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto generated = dif::compiler::emit_cuda(program);
  expect(generated.source.find("const dif_f32* x,dif_bf16* y") !=
             std::string::npos &&
             generated.source.find("const dif_bf16* x,dif_f32* y") !=
                 std::string::npos,
         "CUDA lowering emits typed f32/bf16 Cast boundaries");
  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  float gate_maximum_absolute_error = 0.0F;
  for (const auto tensor_id :
       {4U, 5U, 6U, 7U, 9U, 10U, 11U, 12U, 15U, 16U}) {
    const auto expected = float_values(reference.outputs.at(tensor_id));
    const auto actual = float_values(candidate.outputs.at(tensor_id));
    expect(expected.size() == actual.size(),
           "CUDA new primitive output size parity");
    for (std::size_t index = 0;
         index < std::min(expected.size(), actual.size()); ++index) {
      const auto error = std::abs(expected[index] - actual[index]);
      gate_maximum_absolute_error =
          std::max(gate_maximum_absolute_error, error);
      expect(error <=
                 (tensor_id == 9U || tensor_id == 12U ? 1.0e-5F : 1.0e-2F),
             "CUDA new primitive numerical parity");
    }
  }
  std::cout << "GATE mixed_primitives backend=" << candidate.backend_name
            << " device=" << candidate.device_name
            << " max_abs=" << gate_maximum_absolute_error << "\n";
}

void test_vae_normalization_primitives() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F16, TensorRole::Input, {2, 4}},
      {2, DType::F16, TensorRole::Constant, {4}},
      {3, DType::F16, TensorRole::Constant, {4}},
      {4, DType::F16, TensorRole::Output, {2, 4}},
      {5, DType::F16, TensorRole::Constant, {4}},
      {6, DType::F16, TensorRole::Constant, {4}},
      {7, DType::F16, TensorRole::Output, {2, 4}},
  };
  program.operations = {
      {1, Opcode::AffineLastDim, {1, 2, 3}, {4}, {}},
      {2, Opcode::LayerNorm, {1, 5, 6}, {7},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32U)}},
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, float_tensor(DType::F16, {2, 4},
                                   {1, 2, 3, 4, -1, 0, 1, 2}));
  bindings.emplace(2, float_tensor(DType::F16, {4}, {2, 0.5F, -1, 1}));
  bindings.emplace(3, float_tensor(DType::F16, {4}, {0.5F, 0, 0, -0.5F}));
  bindings.emplace(5, float_tensor(DType::F16, {4}, {1, 1, 1, 1}));
  bindings.emplace(6, float_tensor(DType::F16, {4}, {0, 0, 0, 0}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto affine = float_values(reference.outputs.at(4));
  const std::vector<float> expected_affine = {
      2.5F, 1.0F, -3.0F, 3.5F, -1.5F, 0.0F, -1.0F, 1.5F};
  expect(affine == expected_affine,
         "affine_last_dim applies one vector across every row");
  const auto normalized = float_values(reference.outputs.at(7));
  for (std::size_t row = 0; row < 2U; ++row) {
    float mean = 0.0F;
    for (std::size_t column = 0; column < 4U; ++column)
      mean += normalized[row * 4U + column];
    expect(std::abs(mean) < 2.0e-3F,
           "layer_norm produces a zero-mean F16 row");
  }
  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  for (const auto output : {4U, 7U}) {
    const auto expected = float_values(reference.outputs.at(output));
    const auto actual = float_values(candidate.outputs.at(output));
    for (std::size_t index = 0; index < expected.size(); ++index)
      expect(std::abs(expected[index] - actual[index]) <= 2.0e-3F,
             "CUDA VAE normalization primitive numerical parity");
  }
}

void test_generic_image_vae_primitives() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {1, 2, 3, 3}},
      {2, DType::BF16, TensorRole::Constant, {2, 2, 3, 3}},
      {3, DType::BF16, TensorRole::Constant, {2}},
      {4, DType::BF16, TensorRole::Output, {1, 2, 3, 3}},
      {5, DType::BF16, TensorRole::Constant, {2}},
      {6, DType::BF16, TensorRole::Output, {1, 2, 3, 3}},
      {7, DType::BF16, TensorRole::Output, {1, 2, 6, 6}},
      {8, DType::BF16, TensorRole::Input, {1, 1, 2, 2, 2}},
      {9, DType::BF16, TensorRole::Output, {1, 1, 3, 4, 4}},
      {10, DType::BF16, TensorRole::Constant, {1, 1, 2, 3, 3}},
      {11, DType::BF16, TensorRole::Constant, {1}},
      {12, DType::BF16, TensorRole::Output, {1, 1, 2, 2, 2}},
  };
  program.operations = {
      {1, Opcode::Conv2d, {1, 2, 3}, {4},
       {Attribute::u64(AttrKey::StrideH, 1),
        Attribute::u64(AttrKey::StrideW, 1),
        Attribute::u64(AttrKey::DilationH, 1),
        Attribute::u64(AttrKey::DilationW, 1),
        Attribute::u64(AttrKey::PadTop, 1),
        Attribute::u64(AttrKey::PadBottom, 1),
        Attribute::u64(AttrKey::PadWest, 1),
        Attribute::u64(AttrKey::PadEast, 1),
        Attribute::u64(AttrKey::Groups, 1)}},
      {2, Opcode::ChannelRmsNorm, {4, 5}, {6},
       {Attribute::u64(AttrKey::Axis, 1),
        Attribute::u64(AttrKey::BlockSize, 32),
        Attribute::f64(AttrKey::Epsilon, 1.0e-12)}},
      {3, Opcode::UpsampleNearest2d, {6}, {7},
       {Attribute::u64(AttrKey::ScaleH, 2),
        Attribute::u64(AttrKey::ScaleW, 2)}},
      {4, Opcode::PadConstant, {8}, {9},
       {Attribute::u64(AttrKey::PadFront, 1),
        Attribute::u64(AttrKey::PadBack, 0),
        Attribute::u64(AttrKey::PadTop, 1),
        Attribute::u64(AttrKey::PadBottom, 1),
        Attribute::u64(AttrKey::PadWest, 1),
        Attribute::u64(AttrKey::PadEast, 1)}},
      {5, Opcode::Conv3d, {9, 10, 11}, {12},
       {Attribute::u64(AttrKey::StrideT, 1),
        Attribute::u64(AttrKey::StrideH, 1),
        Attribute::u64(AttrKey::StrideW, 1),
        Attribute::u64(AttrKey::DilationT, 1),
        Attribute::u64(AttrKey::DilationH, 1),
        Attribute::u64(AttrKey::DilationW, 1),
        Attribute::u64(AttrKey::Groups, 1)}},
  };
  dif::ir::verify(program);
  const auto decoded = dif::ir::decode(dif::ir::encode(program));
  expect(decoded.operations.size() == 5U &&
             decoded.operations[0].opcode == Opcode::Conv2d &&
             decoded.operations[1].opcode == Opcode::ChannelRmsNorm &&
             decoded.operations[2].opcode == Opcode::UpsampleNearest2d &&
             decoded.operations[3].opcode == Opcode::PadConstant &&
             decoded.operations[4].opcode == Opcode::Conv3d,
         "generic image VAE opcodes survive DiffIR roundtrip");
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, float_tensor(
                          DType::BF16, {1, 2, 3, 3},
                          {-1.0F, -0.75F, -0.5F, -0.25F, 0.0F, 0.25F,
                           0.5F, 0.75F, 1.0F, 1.0F, 0.5F, 0.0F, -0.5F,
                           -1.0F, -0.5F, 0.0F, 0.5F, 1.0F}));
  bindings.emplace(2, float_tensor(
                          DType::BF16, {2, 2, 3, 3},
                          {0.05F, 0.1F, 0.05F, 0.1F, 0.4F, 0.1F, 0.05F,
                           0.1F, 0.05F, -0.1F, 0.0F, 0.1F, -0.2F, 0.5F,
                           -0.2F, 0.1F, 0.0F, -0.1F, -0.05F, -0.1F,
                           -0.05F, -0.1F, -0.4F, -0.1F, -0.05F, -0.1F,
                           -0.05F, 0.1F, 0.0F, -0.1F, 0.2F, -0.5F, 0.2F,
                           -0.1F, 0.0F, 0.1F}));
  bindings.emplace(3, float_tensor(DType::BF16, {2}, {0.125F, -0.25F}));
  bindings.emplace(5, float_tensor(DType::BF16, {2}, {1.0F, 0.75F}));
  bindings.emplace(8, float_tensor(DType::BF16, {1, 1, 2, 2, 2},
                                   {1.0F, 2.0F, 3.0F, 4.0F,
                                    5.0F, 6.0F, 7.0F, 8.0F}));
  bindings.emplace(10, float_tensor(DType::BF16, {1, 1, 2, 3, 3},
                                    {0.125F, 0.125F, 0.125F,
                                     0.125F, 0.125F, 0.125F,
                                     0.125F, 0.125F, 0.125F,
                                     0.25F, 0.25F, 0.25F,
                                     0.25F, 0.25F, 0.25F,
                                     0.25F, 0.25F, 0.25F}));
  bindings.emplace(11, float_tensor(DType::BF16, {1}, {0.5F}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto upsampled = float_values(reference.outputs.at(7));
  expect(upsampled[0] == upsampled[1] && upsampled[0] == upsampled[6] &&
             upsampled[0] == upsampled[7],
         "nearest 2D upsample replicates each pixel into a 2x2 tile");
  const auto padded = float_values(reference.outputs.at(9));
  expect(padded[0] == 0.0F && padded[21] == 1.0F && padded[42] == 8.0F,
         "rank-5 constant padding preserves NCDHW causal placement");
  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  float maximum_absolute_error = 0.0F;
  for (const auto id : {4U, 6U, 7U, 9U, 12U}) {
    const auto expected = float_values(reference.outputs.at(id));
    const auto actual = float_values(candidate.outputs.at(id));
    for (std::size_t index = 0U; index < expected.size(); ++index)
      maximum_absolute_error = std::max(
          maximum_absolute_error, std::abs(expected[index] - actual[index]));
  }
  expect(maximum_absolute_error <= 0.02F,
         "CUDA cuDNN Conv2d and image VAE primitives match CPU BF16 semantics");
  expect(candidate.run_telemetry.cudnn_convolution_dispatches == 2U,
         "Conv2d and Conv3d execute through two cuDNN convolution dispatches");
  std::cout << "GATE generic_image_vae_primitives backend="
            << candidate.backend_name << " device=" << candidate.device_name
            << " max_abs=" << maximum_absolute_error << "\n";
}

void test_group_norm_and_reflect_padding_primitives() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4, 1, 2}},
      {2, DType::F32, TensorRole::Constant, {4}},
      {3, DType::F32, TensorRole::Constant, {4}},
      {4, DType::F32, TensorRole::Output, {1, 4, 1, 2}},
      {5, DType::F32, TensorRole::Input, {1, 1, 1, 2, 3}},
      {6, DType::F32, TensorRole::Output, {1, 1, 1, 4, 5}},
  };
  program.operations = {
      {1, Opcode::GroupNorm, {1, 2, 3}, {4},
       {Attribute::u64(AttrKey::Groups, 2U),
        Attribute::u64(AttrKey::BlockSize, 32U),
        Attribute::f64(AttrKey::Epsilon, 1.0e-6)}},
      {2, Opcode::PadReflect, {5}, {6},
       {Attribute::u64(AttrKey::PadTop, 1U),
        Attribute::u64(AttrKey::PadBottom, 1U),
        Attribute::u64(AttrKey::PadWest, 1U),
        Attribute::u64(AttrKey::PadEast, 1U)}},
  };
  dif::ir::verify(program);
  const auto decoded = dif::ir::decode(dif::ir::encode(program));
  expect(decoded.operations.size() == 2U &&
             decoded.operations[0].opcode == Opcode::GroupNorm &&
             decoded.operations[1].opcode == Opcode::PadReflect,
         "group_norm and pad_reflect survive DiffIR roundtrip");

  const dif::runtime::TensorMap bindings = {
      {1, float_tensor(DType::F32, {1, 4, 1, 2},
                       {1.0F, 2.0F, 3.0F, 4.0F,
                        5.0F, 6.0F, 7.0F, 8.0F})},
      {2, float_tensor(DType::F32, {4}, {1.0F, 1.0F, 1.0F, 1.0F})},
      {3, float_tensor(DType::F32, {4}, {0.0F, 0.0F, 0.0F, 0.0F})},
      {5, float_tensor(DType::F32, {1, 1, 1, 2, 3},
                       {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F})},
  };
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  const auto cpu = dif::runtime::make_cpu_executor()->run(program, bindings,
                                                           options);
  const auto normalized = float_values(cpu.outputs.at(4));
  const std::vector<float> expected_norm = {
      -1.3416402F, -0.4472134F, 0.4472134F, 1.3416402F,
      -1.3416402F, -0.4472134F, 0.4472134F, 1.3416402F};
  for (std::size_t index = 0U; index < normalized.size(); ++index)
    expect(std::abs(normalized[index] - expected_norm[index]) < 2.0e-6F,
           "group_norm matches the per-group population-variance equation");
  expect(float_values(cpu.outputs.at(6)) ==
             std::vector<float>{5.0F, 4.0F, 5.0F, 6.0F, 5.0F,
                                2.0F, 1.0F, 2.0F, 3.0F, 2.0F,
                                5.0F, 4.0F, 5.0F, 6.0F, 5.0F,
                                2.0F, 1.0F, 2.0F, 3.0F, 2.0F},
         "pad_reflect excludes the edge sample on both spatial axes");

  if (!dif::runtime::cuda_available())
    return;
  const auto cuda = dif::runtime::make_cuda_executor()->run(program, bindings,
                                                             options);
  for (const auto output : {4U, 6U}) {
    const auto expected = float_values(cpu.outputs.at(output));
    const auto actual = float_values(cuda.outputs.at(output));
    expect(expected.size() == actual.size(),
           "CUDA group/pad output geometry matches CPU");
    for (std::size_t index = 0U; index < expected.size(); ++index)
      expect(std::abs(expected[index] - actual[index]) <= 3.0e-6F,
             "CUDA group_norm/pad_reflect numerical parity");
  }
}

void test_krea2_qwen_image_vae_frontend_contract() {
  using namespace dif::frontend;
  using namespace dif::ir;
  const auto build = make_krea2_qwen_image_vae();
  verify(build.program);
  const auto *input = build.program.tensor(build.latent_input);
  const auto *raw = build.program.tensor(build.raw_output);
  const auto *clamped = build.program.tensor(build.clamped_output);
  expect(input && input->dtype == DType::BF16 &&
             input->dims == std::vector<std::uint64_t>({1U, 16U, 32U, 32U}),
         "Krea 2 Qwen Image VAE tile accepts the official 32x32 latent tile");
  expect(raw && clamped &&
             raw->dims ==
                 std::vector<std::uint64_t>({1U, 3U, 256U, 256U}) &&
             clamped->dims == raw->dims &&
             raw->has_role(TensorRole::Output) &&
             clamped->has_role(TensorRole::Output),
         "Krea 2 Qwen Image VAE tile exposes raw and clamped 256px outputs");
  expect(build.program.operations.size() == 223U &&
             build.weights.size() == 104U,
         "Krea 2 Qwen Image VAE frontend has the creator decoder inventory");
  std::uint64_t conv_count = 0U;
  std::uint64_t conv3d_count = 0U;
  std::uint64_t pad_count = 0U;
  std::uint64_t attention_count = 0U;
  std::uint64_t upsample_count = 0U;
  for (const auto &operation : build.program.operations) {
    conv_count += operation.opcode == Opcode::Conv2d ? 1U : 0U;
    conv3d_count += operation.opcode == Opcode::Conv3d ? 1U : 0U;
    pad_count += operation.opcode == Opcode::PadConstant ? 1U : 0U;
    attention_count += operation.opcode == Opcode::Attention ? 1U : 0U;
    upsample_count +=
        operation.opcode == Opcode::UpsampleNearest2d ? 1U : 0U;
    expect(operation.opcode != Opcode::H3AdaLNSelect &&
               operation.opcode != Opcode::H3DeinterleaveQkv &&
               operation.opcode != Opcode::H3DeinterleaveQkvWeight,
           "Krea 2 Qwen Image VAE contains no H3-only runtime operation");
  }
  expect(attention_count == 1U && upsample_count == 3U && conv_count == 5U &&
             conv3d_count == 32U && pad_count == 30U,
         "Krea 2 Qwen Image VAE uses shared attention, padding, Conv2d, and Conv3d ops");
  std::cout << "GATE krea2_qwen_image_vae operations="
            << build.program.operations.size()
            << " weights=" << build.weights.size()
            << " conv2d=" << conv_count
            << " conv3d=" << conv3d_count << " pad=" << pad_count
            << " attention=" << attention_count
            << " upsample=" << upsample_count << "\n";
}

void test_h3_video_vae_frontend_contract() {
  dif::frontend::H3VideoVaeConfig config;
  config.latent_frames = 1U;
  config.latent_height = 1U;
  config.latent_width = 1U;
  config.layers = 1U;
  config.attention_implementation = 1U;
  const auto build = dif::frontend::make_h3_video_vae_decoder(config);
  expect(build.program.operations.size() == 36U &&
             build.bindings.size() == 31U &&
             build.generated_constants.size() == 10U,
         "H3 video VAE frontend emits the released one-layer graph contract");
  const auto *raw = build.program.tensor(build.raw_output_id);
  const auto *decoded = build.program.tensor(build.decoded_output_id);
  expect(raw && decoded && raw->dims == std::vector<std::uint64_t>(
                                           {1U, 3U, 4U, 16U, 16U}) &&
             decoded->dims == raw->dims &&
             raw->has_role(dif::ir::TensorRole::Output) &&
             decoded->has_role(dif::ir::TensorRole::Output),
         "H3 video VAE exposes raw and clamped decoded pixel volumes");
  bool saw_f16_qkv = false;
  bool saw_f32_norm = false;
  bool saw_position = false;
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    if (binding.source_name.find("attn.to_qkv.weight") != std::string::npos)
      saw_f16_qkv = description && description->dtype == dif::ir::DType::F16;
    if (binding.source_name.find("norm1.weight") != std::string::npos)
      saw_f32_norm = description && description->dtype == dif::ir::DType::F32;
    if (binding.name == "dif.position_ids_2pi") {
      saw_position = true;
      const auto values =
          float_values(build.generated_constants.at(binding.tensor_id));
      expect(values.size() == 18U && values.front() == 0.0F &&
                 values.back() == 0.0F,
             "H3 video VAE one-voxel and suffix positions are exactly zero");
    }
  }
  expect(saw_f16_qkv && saw_f32_norm && saw_position,
         "H3 video VAE pins measured FP16 autocast and FP32 norm boundaries");
  const auto generated = dif::compiler::emit_cuda(build.program);
  expect(generated.source.find("dif_store_f16(cosine") != std::string::npos,
         "CUDA rotary lowering honors F16 VAE cosine/sine storage");

}

void test_h3_video_encoder_frontend_contract() {
  dif::frontend::H3VideoEncoderConfig config;
  config.frames = 1U;
  config.height = 256U;
  config.width = 256U;
  config.capture_boundaries = true;
  const auto build = dif::frontend::make_h3_video_encoder(config);
  dif::ir::verify(build.program);
  const auto *input = build.program.tensor(build.pixels_input);
  const auto *output = build.program.tensor(build.moments_output);
  expect(input && input->dtype == dif::ir::DType::F32 &&
             input->dims == std::vector<std::uint64_t>{1U, 3U, 1U, 256U,
                                                       256U},
         "H3 video encoder exposes the released F32 NCTHW pixel ABI");
  expect(output && output->dtype == dif::ir::DType::F32 &&
             output->dims == std::vector<std::uint64_t>{1U, 48U, 1U, 16U,
                                                        16U},
         "H3 video encoder emits mean/logvar moments at 16x spatial compression");
  std::size_t conv3d = 0U;
  std::size_t group_norm = 0U;
  std::size_t reflect = 0U;
  std::size_t constant_pad = 0U;
  for (const auto &operation : build.program.operations) {
    conv3d += operation.opcode == dif::ir::Opcode::Conv3d ? 1U : 0U;
    group_norm += operation.opcode == dif::ir::Opcode::GroupNorm ? 1U : 0U;
    reflect += operation.opcode == dif::ir::Opcode::PadReflect ? 1U : 0U;
    constant_pad += operation.opcode == dif::ir::Opcode::PadConstant ? 1U : 0U;
  }
  expect(build.weights.size() == 118U && conv3d == 34U &&
             group_norm == 25U && reflect == 30U && constant_pad == 30U,
         "H3 encoder census matches 6 levels, 12 residual blocks, and quant_conv");
  expect(build.program.operations.size() == 256U,
         "H3 encoder graph preserves every padding, per-frame norm, conv, activation, and residual boundary");
  const auto second = dif::frontend::make_h3_video_encoder(config);
  expect(dif::ir::encode(build.program) == dif::ir::encode(second.program),
         "H3 video encoder frontend is deterministic");
}

void test_training_autodiff_optimizer_and_checkpoint() {
  dif::frontend::MlpTrainingConfig config;
  config.rows = 2U;
  config.input_width = 2U;
  config.hidden_width = 3U;
  config.output_width = 1U;
  config.learning_rate = 1.0e-2;
  config.weight_decay = 1.0e-2;
  const auto build = dif::frontend::make_mlp_training(config);
  expect(build.optimizer_bindings.size() == 4U &&
             build.program.operations.size() == 19U &&
             build.program.tensor(build.loss_output)->dims ==
                 std::vector<std::uint64_t>({1U}),
         "training frontend emits forward, reverse-mode, and four optimizer updates");
  for (const auto &binding : build.optimizer_bindings) {
    expect(build.program.tensor(binding.parameter_input)
                   ->has_role(dif::ir::TensorRole::Parameter) &&
               build.program.tensor(binding.first_moment_input)
                   ->has_role(dif::ir::TensorRole::OptimizerState) &&
               build.program.tensor(binding.gradient_output)
                   ->has_role(dif::ir::TensorRole::Output),
           "training roles expose parameters, optimizer state, and gradients");
  }
  const auto generated = dif::compiler::emit_cuda(build.program);
  expect(generated.source.find("completed_steps") != std::string::npos &&
             generated.source.find("grad_weight") != std::string::npos,
         "CUDA lowering contains explicit backward and AdamW state kernels");

  dif::runtime::TensorMap inputs;
  inputs.emplace(build.features_input,
                 f32_tensor({2U, 2U}, {-1.0F, 0.5F, 0.25F, 1.0F}));
  inputs.emplace(build.target_input, f32_tensor({2U, 1U}, {0.5F, -0.25F}));
  const std::array<std::vector<float>, 4> parameter_values = {
      std::vector<float>{-0.2F, -0.1F, 0.0F, 0.1F, 0.2F, 0.3F},
      std::vector<float>{-0.05F, 0.0F, 0.05F},
      std::vector<float>{-0.1F, 0.0F, 0.1F},
      std::vector<float>{0.02F},
  };
  for (std::size_t index = 0U; index < build.optimizer_bindings.size();
       ++index) {
    const auto &binding = build.optimizer_bindings[index];
    const auto *parameter = build.program.tensor(binding.parameter_input);
    inputs.emplace(binding.parameter_input,
                   f32_tensor(parameter->dims, parameter_values[index]));
    inputs.emplace(binding.first_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.first_moment_input)));
    inputs.emplace(binding.second_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.second_moment_input)));
  }
  inputs.emplace(build.step_input, i32_tensor({1U}, {0}));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);
  const auto loss = reference.outputs.at(build.loss_output).f32()[0];
  expect(std::isfinite(loss) && loss > 0.0F,
         "reverse-mode training graph produces a finite scalar loss");
  for (const auto &binding : build.optimizer_bindings) {
    const auto gradient = reference.outputs.at(binding.gradient_output).f32();
    const auto updated = reference.outputs.at(binding.parameter_output).f32();
    expect(std::all_of(gradient.begin(), gradient.end(),
                       [](float value) { return std::isfinite(value); }) &&
               !std::equal(updated.begin(), updated.end(),
                           inputs.at(binding.parameter_input).f32().begin(),
                           inputs.at(binding.parameter_input).f32().end()),
           "autodiff emits finite gradients consumed by AdamW");
  }
  if (dif::runtime::cuda_available()) {
    const auto candidate =
        dif::runtime::make_cuda_executor()->run(build.program, inputs, options);
    float maximum_absolute_error = 0.0F;
    for (const auto &[id, expected_tensor] : reference.outputs) {
      const auto expected = float_values(expected_tensor);
      const auto actual = float_values(candidate.outputs.at(id));
      for (std::size_t index = 0U; index < expected.size(); ++index)
        maximum_absolute_error =
            std::max(maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    expect(maximum_absolute_error <= 1.0e-5F,
           "CUDA training forward, backward, and optimizer match CPU semantics");
    std::cout << "GATE training_one_step backend=" << candidate.backend_name
              << " device=" << candidate.device_name
              << " max_abs=" << maximum_absolute_error << "\n";
  }

  dif::training::Checkpoint checkpoint;
  checkpoint.program_fingerprint = dif::ir::fingerprint(build.program);
  checkpoint.completed_steps = 1U;
  for (const auto &binding : build.optimizer_bindings) {
    checkpoint.state.emplace(binding.parameter_input,
                             reference.outputs.at(binding.parameter_output));
    checkpoint.state.emplace(
        binding.first_moment_input,
        reference.outputs.at(binding.first_moment_output));
    checkpoint.state.emplace(
        binding.second_moment_input,
        reference.outputs.at(binding.second_moment_output));
  }
  const auto temporary =
      std::filesystem::temp_directory_path() /
      ("dif-training-checkpoint-" +
       std::to_string(std::chrono::steady_clock::now()
                          .time_since_epoch()
                          .count()));
  std::filesystem::create_directories(temporary);
  const auto path = temporary / "step-1.diftrain";
  dif::training::write_checkpoint(checkpoint, path);
  const auto recovered = dif::training::read_checkpoint(path);
  expect(recovered.completed_steps == 1U &&
             recovered.program_fingerprint == checkpoint.program_fingerprint &&
             recovered.state.size() == checkpoint.state.size(),
         "checksummed training checkpoint roundtrips state and program identity");
  auto corrupted = path;
  corrupted += ".corrupt";
  std::filesystem::copy_file(path, corrupted);
  {
    std::fstream stream(corrupted, std::ios::binary | std::ios::in |
                                       std::ios::out);
    stream.seekg(24);
    char value = 0;
    stream.read(&value, 1);
    value ^= 1;
    stream.seekp(24);
    stream.write(&value, 1);
  }
  bool rejected = false;
  try {
    (void)dif::training::read_checkpoint(corrupted);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "training checkpoint corruption is rejected");
  std::filesystem::remove_all(temporary);
}

void test_mixed_precision_bf16_training_step() {
  // The default-dtype builder must keep emitting the exact program recorded
  // by the 2026-08-28 F32 training gate.
  {
    dif::frontend::MlpTrainingConfig canonical;
    const auto build = dif::frontend::make_mlp_training(canonical);
    expect(dif::hex_digest(dif::ir::fingerprint(build.program)) ==
               "c33733354ed3be4b5147bb7e4e2fd150400d364a3f4ad00a2c996ae1b54db95f",
           "default F32 MLP training program keeps the recorded gate "
           "fingerprint");
  }

  dif::frontend::MlpTrainingConfig config;
  config.rows = 2U;
  config.input_width = 2U;
  config.hidden_width = 3U;
  config.output_width = 1U;
  config.learning_rate = 1.0e-2;
  config.weight_decay = 1.0e-2;
  config.compute_dtype = dif::ir::DType::BF16;
  const auto build = dif::frontend::make_mlp_training(config);
  bool has_cast = false;
  for (const auto &operation : build.program.operations)
    has_cast |= operation.opcode == dif::ir::Opcode::Cast;
  expect(has_cast &&
             build.program.tensor(build.loss_output)->dtype ==
                 dif::ir::DType::F32 &&
             build.program.tensor(build.features_input)->dtype ==
                 dif::ir::DType::BF16,
         "BF16 training graph carries a Cast boundary into an F32 loss");
  for (const auto &binding : build.optimizer_bindings) {
    expect(build.program.tensor(binding.gradient_output)->dtype ==
                   dif::ir::DType::BF16 &&
               build.program.tensor(binding.first_moment_input)->dtype ==
                   dif::ir::DType::F32 &&
               build.program.tensor(binding.second_moment_output)->dtype ==
                   dif::ir::DType::F32 &&
               build.program.tensor(binding.parameter_output)->dtype ==
                   dif::ir::DType::BF16,
           "BF16 parameters keep BF16 gradients/outputs and F32 moments");
  }

  dif::runtime::TensorMap inputs;
  inputs.emplace(build.features_input,
                 float_tensor(dif::ir::DType::BF16, {2U, 2U},
                              {-1.0F, 0.5F, 0.25F, 1.0F}));
  inputs.emplace(build.target_input, f32_tensor({2U, 1U}, {0.5F, -0.25F}));
  const std::array<std::vector<float>, 4> parameter_values = {
      std::vector<float>{-0.2F, -0.1F, 0.0F, 0.1F, 0.2F, 0.3F},
      std::vector<float>{-0.05F, 0.0F, 0.05F},
      std::vector<float>{-0.1F, 0.0F, 0.1F},
      std::vector<float>{0.02F},
  };
  for (std::size_t index = 0U; index < build.optimizer_bindings.size();
       ++index) {
    const auto &binding = build.optimizer_bindings[index];
    const auto *parameter = build.program.tensor(binding.parameter_input);
    inputs.emplace(binding.parameter_input,
                   float_tensor(dif::ir::DType::BF16, parameter->dims,
                                parameter_values[index]));
    inputs.emplace(binding.first_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.first_moment_input)));
    inputs.emplace(binding.second_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.second_moment_input)));
  }
  inputs.emplace(build.step_input, i32_tensor({1U}, {0}));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);
  const auto loss = reference.outputs.at(build.loss_output).f32()[0];
  expect(std::isfinite(loss) && loss > 0.0F,
         "BF16 training graph produces a finite F32 scalar loss on CPU");
  for (const auto &binding : build.optimizer_bindings) {
    const auto gradient =
        float_values(reference.outputs.at(binding.gradient_output));
    expect(std::all_of(gradient.begin(), gradient.end(),
                       [](float value) { return std::isfinite(value); }),
           "BF16 gradients are finite on CPU");
  }
  if (dif::runtime::cuda_available()) {
    const auto candidate =
        dif::runtime::make_cuda_executor()->run(build.program, inputs, options);
    float maximum_absolute_error = 0.0F;
    for (const auto &[id, expected_tensor] : reference.outputs) {
      const auto expected = float_values(expected_tensor);
      const auto actual = float_values(candidate.outputs.at(id));
      for (std::size_t index = 0U; index < expected.size(); ++index)
        maximum_absolute_error =
            std::max(maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    // Both executors accumulate in F32 and round only at BF16 stores; a few
    // BF16 ULPs (~0.0078 relative at unit magnitude) covers reduction-order
    // differences at this geometry.
    expect(maximum_absolute_error <= 4.0e-3F,
           "CUDA BF16 training step matches CPU semantics");
    std::cout << "GATE bf16_training_one_step backend="
              << candidate.backend_name << " device=" << candidate.device_name
              << " max_abs=" << maximum_absolute_error << "\n";
  }

  // Dtype-contract negatives: BF16 moments and non-F32 accumulators fail
  // closed.
  {
    auto broken = build.program;
    for (auto &tensor : broken.tensors) {
      if (tensor.id == build.optimizer_bindings.front().first_moment_input)
        tensor.dtype = dif::ir::DType::BF16;
    }
    bool rejected = false;
    try {
      dif::ir::verify(broken);
    } catch (const dif::Error &) {
      rejected = true;
    }
    expect(rejected, "adamw_update rejects BF16 first moments");
  }
  {
    auto broken = build.program;
    for (auto &operation : broken.operations) {
      if (operation.opcode == dif::ir::Opcode::AdamWUpdate) {
        operation.attributes.push_back(dif::ir::Attribute::u64(
            dif::ir::AttrKey::AccumulatorDType,
            static_cast<std::uint64_t>(dif::ir::DType::BF16)));
        break;
      }
    }
    bool rejected = false;
    try {
      dif::ir::verify(broken);
    } catch (const dif::Error &) {
      rejected = true;
    }
    expect(rejected, "training ops reject a non-F32 AccumulatorDType");
  }
}

void test_rectified_flow_training_vertical() {
  dif::frontend::RectifiedFlowTrainingConfig config;
  config.rows = 2U;
  config.latent_width = 2U;
  config.timestep_width = 2U;
  config.hidden_width = 3U;
  config.accumulation_steps = 2U;
  config.learning_rate = 5.0e-3;
  config.weight_decay = 1.0e-2;
  const auto build = dif::frontend::make_rectified_flow_training(config);
  expect(build.microbatches.size() == 2U &&
             build.optimizer_bindings.size() == 5U &&
             build.program.operations.size() == 66U,
         "rectified-flow frontend emits two accumulated diffusion objectives");
  for (std::size_t index = 0U; index < build.optimizer_bindings.size();
       ++index) {
    const auto parameter_id = build.optimizer_bindings[index].parameter_input;
    const auto operation = std::find_if(
        build.program.operations.begin(), build.program.operations.end(),
        [&](const auto &candidate) {
          return candidate.opcode == dif::ir::Opcode::AdamWUpdate &&
                 candidate.inputs.front() == parameter_id;
        });
    const auto expected_decay = index == 1U || index == 4U ? 0.0 : 1.0e-2;
    expect(operation != build.program.operations.end() &&
               operation->f64(dif::ir::AttrKey::WeightDecay, -1.0) ==
                   expected_decay,
           "rectified-flow parameter groups separate matrix and bias decay");
  }

  dif::runtime::TensorMap inputs;
  const std::array<std::vector<float>, 2> clean = {
      std::vector<float>{-0.8F, -0.2F, 0.4F, 0.9F},
      std::vector<float>{0.7F, -0.5F, 0.1F, -0.9F}};
  const std::array<std::vector<float>, 2> noise = {
      std::vector<float>{0.5F, -0.4F, 0.8F, -0.1F},
      std::vector<float>{-0.3F, 0.6F, -0.7F, 0.2F}};
  const std::array<std::vector<float>, 2> clean_scale = {
      std::vector<float>{0.25F, 0.25F, 0.75F, 0.75F},
      std::vector<float>{0.4F, 0.4F, 0.6F, 0.6F}};
  const std::array<std::vector<float>, 2> time_features = {
      std::vector<float>{0.25F, 0.0625F, 0.75F, 0.5625F},
      std::vector<float>{0.4F, 0.16F, 0.6F, 0.36F}};
  for (std::size_t index = 0U; index < build.microbatches.size(); ++index) {
    const auto &binding = build.microbatches[index];
    std::vector<float> noise_scale(clean_scale[index].size());
    std::vector<float> target(clean_scale[index].size());
    for (std::size_t value = 0U; value < clean_scale[index].size(); ++value) {
      noise_scale[value] = 1.0F - clean_scale[index][value];
      target[value] = clean[index][value] - noise[index][value];
    }
    inputs.emplace(binding.clean_input,
                   f32_tensor({2U, 2U}, clean[index]));
    inputs.emplace(binding.noise_input,
                   f32_tensor({2U, 2U}, noise[index]));
    inputs.emplace(binding.clean_scale_input,
                   f32_tensor({2U, 2U}, clean_scale[index]));
    inputs.emplace(binding.noise_scale_input,
                   f32_tensor({2U, 2U}, noise_scale));
    inputs.emplace(binding.timestep_features_input,
                   f32_tensor({2U, 2U}, time_features[index]));
    inputs.emplace(binding.target_velocity_input,
                   f32_tensor({2U, 2U}, target));
  }
  const std::array<std::vector<float>, 5> parameter_values = {
      std::vector<float>{-0.2F, -0.1F, 0.0F, 0.1F, 0.2F, 0.3F},
      std::vector<float>{-0.05F, 0.0F, 0.05F},
      std::vector<float>{0.15F, 0.1F, 0.05F, 0.0F, -0.05F, -0.1F},
      std::vector<float>{-0.1F, 0.0F, 0.1F, 0.2F, -0.2F, 0.05F},
      std::vector<float>{0.02F, -0.02F},
  };
  for (std::size_t index = 0U; index < build.optimizer_bindings.size();
       ++index) {
    const auto &binding = build.optimizer_bindings[index];
    inputs.emplace(
        binding.parameter_input,
        f32_tensor(build.program.tensor(binding.parameter_input)->dims,
                   parameter_values[index]));
    inputs.emplace(binding.first_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.first_moment_input)));
    inputs.emplace(binding.second_moment_input,
                   dif::runtime::zeros(
                       *build.program.tensor(binding.second_moment_input)));
  }
  inputs.emplace(build.step_input, i32_tensor({1U}, {0}));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(build.program, inputs, options);
  expect(std::isfinite(reference.outputs.at(build.loss_output).f32()[0]) &&
             reference.outputs.at(build.loss_output).f32()[0] > 0.0F,
         "accumulated rectified-flow objective produces a finite loss");
  for (const auto &binding : build.optimizer_bindings) {
    const auto gradient = reference.outputs.at(binding.gradient_output).f32();
    expect(std::all_of(gradient.begin(), gradient.end(),
                       [](float value) { return std::isfinite(value); }),
           "rectified-flow autodiff produces finite shared-parameter gradients");
  }
  if (dif::runtime::cuda_available()) {
    const auto candidate =
        dif::runtime::make_cuda_executor()->run(build.program, inputs, options);
    float maximum_absolute_error = 0.0F;
    for (const auto &[id, expected_tensor] : reference.outputs) {
      const auto expected = float_values(expected_tensor);
      const auto actual = float_values(candidate.outputs.at(id));
      for (std::size_t index = 0U; index < expected.size(); ++index)
        maximum_absolute_error =
            std::max(maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    expect(maximum_absolute_error <= 1.0e-5F,
           "CUDA accumulated rectified-flow training matches CPU semantics");
    std::cout << "GATE rectified_flow_training_one_step backend="
              << candidate.backend_name << " device=" << candidate.device_name
              << " max_abs=" << maximum_absolute_error << "\n";
  }
}

void test_backend_neutral_diffusion_preprocessing() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {2}},
      {2, DType::F32, TensorRole::Output, {2, 5}},
      {3, DType::F32, TensorRole::Input, {2, 3}},
      {4, DType::F32, TensorRole::Input, {2}},
      {5, DType::BF16, TensorRole::Output, {2, 12}},
      {6, DType::BF16, TensorRole::Output, {2, 12}},
  };
  program.operations = {
      {1,
       Opcode::SinusoidalTimestep,
       {1},
       {2},
       {Attribute::boolean(AttrKey::FlipSinToCos, true),
        Attribute::f64(AttrKey::DownscaleFreqShift, 0.0),
        Attribute::f64(AttrKey::Scale, 1.0),
        Attribute::f64(AttrKey::MaxPeriod, 10000.0)}},
      {2, Opcode::RotaryPosition, {3, 4}, {5, 6}, {}},
  };
  dif::ir::verify(program);
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, f32_tensor({2}, {0.0F, 1.0F}));
  bindings.emplace(3,
                   f32_tensor({2, 3}, {0.0F, 1.0F, 2.0F,
                                              3.0F, 4.0F, 5.0F}));
  bindings.emplace(4, f32_tensor({2}, {1.0F, 0.1F}));
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto timestep = float_values(reference.outputs.at(2));
  expect(timestep[0] == 1.0F && timestep[1] == 1.0F &&
             timestep[2] == 0.0F && timestep[3] == 0.0F &&
             timestep[4] == 0.0F && timestep[9] == 0.0F &&
             std::abs(timestep[5] - std::cos(1.0F)) < 1.0e-6F &&
             std::abs(timestep[7] - std::sin(1.0F)) < 1.0e-6F,
         "sinusoidal timestep preserves source flip order and odd padding");
  const auto cosine = float_values(reference.outputs.at(5));
  const auto sine = float_values(reference.outputs.at(6));
  expect(cosine[0] == 1.0F && sine[0] == 0.0F &&
             cosine[1] == 1.0F && sine[1] == 0.0F &&
             cosine[2] == cosine[8] && sine[2] == sine[8],
         "rotary position concatenates axes and repeats the frequency table");
  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  float maximum_absolute_error = 0.0F;
  for (const auto output : {2U, 5U, 6U}) {
    const auto expected = float_values(reference.outputs.at(output));
    const auto actual = float_values(candidate.outputs.at(output));
    for (std::size_t i = 0; i < expected.size(); ++i)
      maximum_absolute_error =
          std::max(maximum_absolute_error, std::abs(expected[i] - actual[i]));
  }
  expect(maximum_absolute_error <= 1.0e-6F,
         "CUDA raw diffusion preprocessing matches CPU semantics");
}

void test_prepared_execution_reuses_constants() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 2}},
      {2, DType::F32, TensorRole::Constant, {1, 2}},
      {3, DType::F32, TensorRole::Output, {1, 2}},
  };
  program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
  dif::runtime::TensorMap constants;
  constants.emplace(2, f32_tensor({1, 2}, {10, 20}));
  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  auto prepared = executor->prepare(program, constants, options);
  dif::runtime::TensorMap first;
  first.emplace(1, f32_tensor({1, 2}, {1, 2}));
  dif::runtime::TensorMap second;
  second.emplace(1, f32_tensor({1, 2}, {3, 4}));
  const auto first_values = float_values(prepared->run(first, options).outputs.at(3));
  const auto second_values =
      float_values(prepared->run(second, options).outputs.at(3));
  expect(first_values == std::vector<float>({11, 22}),
         "prepared execution first dynamic input");
  expect(second_values == std::vector<float>({13, 24}),
         "prepared execution reuses constants with new dynamic input");
}

void test_cuda_repeated_invariant_execution_cache() {
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {2}},
      {2, DType::F32, TensorRole::Input, {2}},
      {3, DType::F32, TensorRole::Constant, {2}},
      {4, DType::F32, TensorRole::Internal, {2}},
      {5, DType::F32, TensorRole::Output, {2}},
  };
  program.operations = {
      {11, Opcode::Add, {1, 3}, {4}, {}},
      {12, Opcode::Multiply, {4, 2}, {5}, {}},
  };
  dif::ir::verify(program);

  const dif::runtime::TensorMap constants = {
      {3, f32_tensor({2}, {5.0F, 7.0F})},
  };
  dif::runtime::RunOptions cached_options;
  cached_options.warmups = 0U;
  cached_options.iterations = 1U;
  cached_options.minimum_free_bytes = 0U;
  cached_options.repeated_invariant_operations = {11U};
  cached_options.capture_intermediate_tensors = {4U};
  auto cached = dif::runtime::make_cuda_executor()->prepare(
      program, constants, cached_options);

  dif::runtime::TensorMap first_inputs = {
      {1, f32_tensor({2}, {1.0F, 2.0F})},
      {2, f32_tensor({2}, {10.0F, 10.0F})},
  };
  const auto first = cached->run(first_inputs, cached_options);
  expect(float_values(first.outputs.at(5U)) ==
             std::vector<float>({60.0F, 90.0F}) &&
             float_values(first.captured_intermediates.at(4U)) ==
                 std::vector<float>({6.0F, 9.0F}) &&
             !first.repeated_invariant_cache_hit,
         "repeated-invariant cache executes and preserves its first boundary");

  dif::runtime::TensorMap second_inputs = {
      {1, f32_tensor({2}, {1.0F, 2.0F})},
      {2, f32_tensor({2}, {20.0F, 30.0F})},
  };
  const auto second = cached->run(second_inputs, cached_options);
  expect(float_values(second.outputs.at(5U)) ==
             std::vector<float>({120.0F, 270.0F}) &&
             second.captured_intermediates.at(4U).bytes ==
                 first.captured_intermediates.at(4U).bytes &&
             second.repeated_invariant_cache_hit &&
             second.repeated_invariant_operation_count == 1U &&
             second.repeated_invariant_persistent_bytes == 2U * sizeof(float) &&
             second.run_telemetry.d2d_copies == 1U &&
             second.run_telemetry.d2d_bytes == 2U * sizeof(float),
         "repeated-invariant cache restores exact device bytes on reuse");

  dif::runtime::TensorMap changed_inputs = {
      {1, f32_tensor({2}, {3.0F, 4.0F})},
      {2, f32_tensor({2}, {20.0F, 30.0F})},
  };
  const auto changed = cached->run(changed_inputs, cached_options);
  expect(float_values(changed.outputs.at(5U)) ==
             std::vector<float>({160.0F, 330.0F}) &&
             !changed.repeated_invariant_cache_hit,
         "repeated-invariant cache fails closed when an input changes");
}

void test_verifier_rejects_multiple_writers() {
  auto program = rms_program(1, 4);
  program.operations.push_back(program.operations.front());
  program.operations.back().id = 2;
  bool rejected = false;
  try {
    dif::ir::verify(program);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "verifier rejects multiple writers");
}

void test_attention_implementation_identity() {
  using namespace dif::ir;
  auto make_program = [](DType dtype, std::uint64_t implementation) {
    Program program;
    program.tensors = {
        {1, dtype, TensorRole::Input, {2, 1, 4}},
        {2, dtype, TensorRole::Input, {2, 1, 4}},
        {3, dtype, TensorRole::Input, {2, 1, 4}},
        {4, dtype, TensorRole::Output, {2, 1, 4}},
    };
    program.operations = {{
        1,
        Opcode::Attention,
        {1, 2, 3},
        {4},
        {Attribute::f64(AttrKey::AttentionScale, 0.5),
         Attribute::u64(AttrKey::BlockSize, 64),
         Attribute::u64(AttrKey::Implementation, implementation)},
    }};
    return program;
  };

  const auto generated = make_program(DType::BF16, 1U);
  const auto cudnn = make_program(DType::BF16, 2U);
  const auto materialized_f32 = make_program(DType::F32, 3U);
  dif::ir::verify(generated);
  dif::ir::verify(cudnn);
  dif::ir::verify(materialized_f32);
  expect(dif::ir::fingerprint(generated) != dif::ir::fingerprint(cudnn),
         "attention implementation changes candidate fingerprint");
  expect(dif::ir::fingerprint(cudnn) !=
             dif::ir::fingerprint(materialized_f32),
         "materialized f32 attention has a distinct candidate identity");

  bool rejected = false;
  try {
    dif::ir::verify(make_program(DType::BF16, 3U));
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "verifier restricts materialized attention to f32");

  rejected = false;
  try {
    dif::ir::verify(make_program(DType::F32, 2U));
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "verifier rejects f32 cuDNN attention candidate");

  rejected = false;
  try {
    dif::ir::verify(make_program(DType::F32, 4U));
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "verifier rejects unknown attention implementation");
}

void test_h3_bf16_lowering_preserves_source_reduction_identity() {
  const auto program = dif::frontend::make_h3_transformer_bf16(
      18, 5376, 56, 128, 14336, 96, 1, 2, 2688, 256, false,
      false);
  const auto generated = dif::compiler::emit_cuda(program);
  expect(generated.source.find("col+=256ULL") != std::string::npos &&
             generated.source.find("active=128U;active>0U;active>>=1U") !=
                 std::string::npos,
         "H3 hidden RMSNorm uses Serenity's accepted 256-thread reduction");
  expect(generated.source.find(
             "float result=(1.0f+dif_load(scale,vector+col))*normed+") !=
             std::string::npos,
         "H3 hidden RMSNorm preserves the BF16 norm boundary and F32 AdaLN");
  expect(generated.source.find("active=64U;active>0U;active>>=1U") !=
                 std::string::npos &&
             generated.source.find("float norm0=dif_round(") !=
                 std::string::npos,
         "H3 QK RMSNorm preserves Serenity's 128-lane reduction and BF16 "
         "normalization boundary");
}

void test_h3_long_sequence_transformer_declares_backend_attention() {
  const auto cudnn = dif::frontend::make_h3_transformer_bf16(
      9065, 5376, 56, 128, 14336, 96, 1, 2, 2688, 256, false,
      true, 2U);
  dif::ir::verify(cudnn);
  const auto attention = std::find_if(
      cudnn.operations.begin(), cudnn.operations.end(),
      [](const dif::ir::Operation &operation) {
        return operation.opcode == dif::ir::Opcode::Attention;
      });
  expect(attention != cudnn.operations.end() &&
             attention->u64(dif::ir::AttrKey::Implementation, 0U) == 2U,
         "long-sequence H3 block carries the requested cuDNN identity");

  bool rejected = false;
  try {
    const auto generated = dif::frontend::make_h3_transformer_bf16(
        9065, 5376, 56, 128, 14336, 96, 1, 2, 2688, 256, false,
        true, 1U);
    dif::ir::verify(generated);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected,
         "long-sequence H3 block refuses the naive generated attention path");
}

void test_h3_mixed_denoiser_frontend_and_cuda_parity() {
  using namespace dif::ir;
  dif::frontend::H3DenoiserConfig config;
  config.video_tokens = 1;
  config.audio_tokens = 1;
  config.text_tokens = 2;
  config.timestep_tables = 2;
  config.hidden = 4;
  config.heads = 2;
  config.head_dim = 6;
  config.ffn = 8;
  config.rotary = 6;
  config.layers = 1;
  config.refiner_layers = 1;
  config.video_input_dim = 3;
  config.audio_input_dim = 2;
  config.text_input_dim = 3;
  config.time_input_dim = 2;
  config.time_hidden_dim = 4;
  config.time_embed_dim = 2;
  config.block_size = 32;
  config.attention_implementation = 1;
  const auto program = dif::frontend::make_h3_denoiser(config);
  expect(program.tensors.size() == 117U && program.operations.size() == 57U,
         "one-layer mixed H3 denoiser has the stable graph ABI");
  std::size_t casts = 0U;
  std::size_t row_chunk_selects = 0U;
  std::size_t constants = 0U;
  std::vector<const TensorDesc *> outputs;
  for (const auto &operation : program.operations) {
    casts += operation.opcode == Opcode::Cast ? 1U : 0U;
    row_chunk_selects +=
        operation.opcode == Opcode::SelectRowChunks ? 1U : 0U;
  }
  for (const auto &description : program.tensors) {
    constants += description.has_role(TensorRole::Constant) ? 1U : 0U;
    if (description.has_role(TensorRole::Output))
      outputs.push_back(&description);
  }
  expect(casts == 4U && row_chunk_selects == 1U && constants == 37U,
         "mixed H3 denoiser contains explicit source dtype and final AdaLN boundaries");
  expect(outputs.size() == 2U && outputs[0]->dtype == DType::F32 &&
             outputs[0]->dims == std::vector<std::uint64_t>({1, 3}) &&
             outputs[1]->dtype == DType::F32 &&
             outputs[1]->dims == std::vector<std::uint64_t>({1, 2}),
         "mixed H3 denoiser exposes source-shaped video and audio predictions");

  dif::runtime::TensorMap bindings;
  bindings.emplace(1, f32_tensor({1, 3}, {0.1F, -0.2F, 0.3F}));
  bindings.emplace(2, f32_tensor({1, 2}, {0.2F, -0.1F}));
  bindings.emplace(3, float_tensor(DType::BF16, {2, 3},
                                   {0.1F, 0.2F, -0.1F, -0.2F, 0.05F, 0.3F}));
  bindings.emplace(4, f32_tensor({2}, {0.25F, 0.75F}));
  bindings.emplace(5, i32_tensor({4}, {0, 1, -1, -1}));
  bindings.emplace(6, i32_tensor({4}, {-1, -1, 0, -1}));
  bindings.emplace(7, i32_tensor({4}, {-1, -1, -1, 0}));
  bindings.emplace(8, i32_tensor({4}, {1, 1, 0, 5}));
  bindings.emplace(9, i32_tensor({4}, {0, 0, 0, 1}));
  bindings.emplace(10, i32_tensor({1}, {2}));
  bindings.emplace(11, i32_tensor({1}, {3}));
  bindings.emplace(12,
                   f32_tensor({4, 3}, {0.0F, 0.0F, 0.0F,
                                              1.0F, 0.0F, 0.0F,
                                              2.0F, 1.0F, 1.0F,
                                              3.0F, 0.0F, 1.0F}));
  for (const auto &description : program.tensors) {
    if (!description.has_role(TensorRole::Constant))
      continue;
    std::vector<float> values(description.element_count());
    for (std::size_t index = 0; index < values.size(); ++index) {
      if (description.dims.size() == 1U)
        values[index] = 0.1F;
      else
        values[index] =
            0.02F * static_cast<float>(
                        static_cast<int>((index + description.id) % 11U) - 5);
    }
    bindings.emplace(description.id,
                     float_tensor(description.dtype, description.dims, values));
  }
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  bool finite = true;
  for (const auto *output : outputs) {
    for (const auto value : float_values(reference.outputs.at(output->id)))
      finite = finite && std::isfinite(value);
  }
  expect(finite, "mixed H3 denoiser CPU graph produces finite outputs");
  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  float maximum_absolute_error = 0.0F;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (const auto *output : outputs) {
    const auto expected = float_values(reference.outputs.at(output->id));
    const auto actual = float_values(candidate.outputs.at(output->id));
    for (std::size_t index = 0; index < expected.size(); ++index) {
      const auto error = std::abs(expected[index] - actual[index]);
      maximum_absolute_error = std::max(maximum_absolute_error, error);
      squared_error += static_cast<double>(error) * error;
      squared_reference += static_cast<double>(expected[index]) * expected[index];
    }
  }
  const auto relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  expect(maximum_absolute_error <= 0.02F && relative_l2 <= 0.02,
         "mixed H3 denoiser CUDA output matches typed CPU semantics");
  std::cout << "GATE h3_mixed_denoiser backend=" << candidate.backend_name
            << " device=" << candidate.device_name
            << " max_abs=" << maximum_absolute_error
            << " rel_l2=" << relative_l2 << "\n";
}

void test_h3_token_refiner_frontend_and_cuda_parity() {
  using namespace dif::ir;
  const auto program = dif::frontend::make_h3_token_refiner_bf16(
      2, 4, 2, 2, 8, 2, 32, false);
  expect(program.tensors.size() == 53U,
         "two-layer H3 token refiner has the stable tensor ABI");
  expect(program.operations.size() == 31U,
         "two-layer H3 token refiner has fifteen ops per block plus final norm");
  expect(program.operations[0].opcode == Opcode::RmsNorm &&
             program.operations[1].opcode ==
                 Opcode::H3DeinterleaveQkvWeight &&
             program.operations[12].opcode == Opcode::SwiGlu &&
             program.operations[15].opcode == Opcode::RmsNorm &&
             program.operations.back().opcode == Opcode::RmsNorm,
         "H3 token refiner preserves the source pre-norm block ordering");
  expect(program.operations[12].boolean(AttrKey::GateFirst, false) &&
             program.operations[27].boolean(AttrKey::GateFirst, false),
         "H3 token refiner preserves source SwiGLU gate/up ordering");
  const auto *final_weight = program.tensor(52U);
  const auto *final_output = program.tensor(53U);
  expect(final_weight && final_weight->has_role(TensorRole::Constant) &&
             final_output && final_output->has_role(TensorRole::Output),
         "H3 token refiner exposes only the final normalized activation");

  const auto streamed = dif::frontend::make_h3_token_refiner_bf16(
      2, 4, 2, 2, 8, 2, 32, true);
  expect(streamed.tensor(2U)->has_role(TensorRole::Streamed) &&
             streamed.tensor(52U)->has_role(TensorRole::Streamed),
         "H3 token-refiner constants can use backend-neutral streaming");

  dif::runtime::TensorMap bindings;
  for (const auto &description : program.tensors) {
    if (!description.has_role(TensorRole::Input) &&
        !description.has_role(TensorRole::Constant))
      continue;
    std::vector<float> values(description.element_count());
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (description.dims.size() == 1U) {
        values[i] = 1.0F + 0.01F * static_cast<float>(i % 3U);
      } else if (description.has_role(TensorRole::Input)) {
        values[i] = 0.05F * static_cast<float>(static_cast<int>(i % 7U) - 3);
      } else {
        values[i] = 0.02F * static_cast<float>(
                                  static_cast<int>((i + description.id) % 11U) -
                                  5);
      }
    }
    bindings.emplace(description.id,
                     float_tensor(description.dtype, description.dims, values));
  }
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto expected = float_values(reference.outputs.at(53U));
  bool finite = true;
  float magnitude = 0.0F;
  for (const auto value : expected) {
    finite = finite && std::isfinite(value);
    magnitude += std::abs(value);
  }
  expect(finite && magnitude > 0.0F,
         "H3 token-refiner CPU graph produces finite nonzero output");

  if (!dif::runtime::cuda_available())
    return;
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const auto actual = float_values(candidate.outputs.at(53U));
  float maximum_absolute_error = 0.0F;
  double squared_error = 0.0;
  double squared_reference = 0.0;
  for (std::size_t i = 0; i < expected.size(); ++i) {
    const auto error = std::abs(expected[i] - actual[i]);
    maximum_absolute_error = std::max(maximum_absolute_error, error);
    squared_error += static_cast<double>(error) * error;
    squared_reference += static_cast<double>(expected[i]) * expected[i];
  }
  const auto relative_l2 =
      std::sqrt(squared_error / std::max(squared_reference, 1.0e-30));
  expect(maximum_absolute_error <= 0.03125F && relative_l2 <= 0.02,
         "H3 token-refiner CUDA output matches the typed CPU semantics");
  std::cout << "GATE h3_token_refiner backend=" << candidate.backend_name
            << " device=" << candidate.device_name
            << " max_abs=" << maximum_absolute_error
            << " rel_l2=" << relative_l2 << "\n";

  options.profile_pipeline = true;
  const auto profiled =
      dif::runtime::make_cuda_executor()->run(streamed, bindings, options);
  const auto profiled_actual = float_values(profiled.outputs.at(53U));
  float profiled_maximum_absolute_error = 0.0F;
  double profiled_squared_error = 0.0;
  for (std::size_t i = 0U; i < expected.size(); ++i) {
    const auto error = std::abs(expected[i] - profiled_actual[i]);
    profiled_maximum_absolute_error =
        std::max(profiled_maximum_absolute_error, error);
    profiled_squared_error += static_cast<double>(error) * error;
  }
  const auto profiled_relative_l2 = std::sqrt(
      profiled_squared_error / std::max(squared_reference, 1.0e-30));
  std::uint64_t expected_streamed_bytes = 0U;
  for (const auto &description : streamed.tensors) {
    if (description.has_role(TensorRole::Constant) &&
        description.has_role(TensorRole::Streamed))
      expected_streamed_bytes += description.byte_count();
  }
  const auto &profile = profiled.pipeline_profile;
  expect(profiled_maximum_absolute_error <= 0.03125F &&
             profiled_relative_l2 <= 0.02,
         "profiled streamed H3 execution preserves CUDA parity");
  expect(profile.enabled && profile.measured_iterations == 1U,
         "CUDA pipeline profile identifies its measured iteration");
  expect(profile.resident_weight_bytes == 0U &&
             profile.streamed_weight_bytes == expected_streamed_bytes,
         "CUDA pipeline profile accounts for streamed semantic weight bytes");
  expect(std::isfinite(profile.streamed_host_stage_milliseconds) &&
             profile.streamed_host_stage_milliseconds >= 0.0 &&
             profile.streamed_h2d_milliseconds > 0.0 &&
             profile.operation_kernel_milliseconds > 0.0 &&
             profile.attention_kernel_milliseconds > 0.0 &&
             profile.non_kernel_device_timeline_milliseconds >= 0.0,
         "CUDA pipeline profile reports finite transfer and operation times");
  expect(profiled.operation_timings.size() == streamed.operations.size(),
         "CUDA pipeline profile times each semantic operation");
  std::cout << "GATE pipeline_profile backend=" << profiled.backend_name
            << " streamed_bytes=" << profile.streamed_weight_bytes
            << " host_stage_ms=" << profile.streamed_host_stage_milliseconds
            << " h2d_ms=" << profile.streamed_h2d_milliseconds
            << " operation_ms=" << profile.operation_kernel_milliseconds
            << " attention_ms=" << profile.attention_kernel_milliseconds
            << " max_abs=" << profiled_maximum_absolute_error
            << " rel_l2=" << profiled_relative_l2 << "\n";
}

void test_memory_plan_reuses_dead_internal_storage() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, TensorRole::Constant, {1, 4}},
      {3, DType::F32, TensorRole::Internal, {1, 4}},
      {4, DType::F32, TensorRole::Internal, {1, 4}},
      {5, DType::F32, TensorRole::Internal, {1, 4}},
      {6, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {
      {1, Opcode::Add, {1, 2}, {3}, {}},
      {2, Opcode::Add, {3, 2}, {4}, {}},
      {3, Opcode::Add, {4, 2}, {5}, {}},
      {4, Opcode::Add, {5, 2}, {6}, {}},
  };
  const auto plan = dif::compiler::plan_memory(program);
  const auto *first = plan.assignment(3);
  const auto *reused = plan.assignment(5);
  expect(first && reused && first->slot_id == reused->slot_id,
         "memory plan reuses non-overlapping internal tensors");
  expect(plan.total_bytes < plan.naive_bytes,
         "memory plan reduces allocated bytes");
}

void test_memory_plan_pages_streamed_constants() {
  using namespace dif::ir;
  Program program;
  constexpr auto streamed = TensorRole::Constant | TensorRole::Streamed;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 1024}},
      {2, DType::F32, streamed, {1, 1024}},
      {3, DType::F32, streamed, {1, 1024}},
      {4, DType::F32, TensorRole::Internal, {1, 1024}},
      {5, DType::F32, TensorRole::Output, {1, 1024}},
  };
  program.operations = {
      {1, Opcode::Add, {1, 2}, {4}, {}},
      {2, Opcode::Add, {4, 3}, {5}, {}},
  };
  const auto plan = dif::compiler::plan_memory(program);
  const auto *first = plan.assignment(2);
  const auto *second = plan.assignment(3);
  expect(first && second && first->slot_id == second->slot_id,
         "memory plan pages non-overlapping streamed constants through one slot");
}

void test_memory_plan_reserves_prefetch_storage() {
  using namespace dif::ir;
  Program program;
  constexpr auto streamed = TensorRole::Constant | TensorRole::Streamed;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, streamed, {1, 4}},
      {3, DType::F32, streamed, {1, 4}},
      {4, DType::F32, TensorRole::Internal, {1, 4}},
      {5, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {
      {1, Opcode::Add, {1, 2}, {4}, {}},
      {2, Opcode::Add, {4, 3}, {5}, {}},
  };
  const auto serial = dif::compiler::plan_memory(program);
  const auto prefetched = dif::compiler::plan_memory(program, 256U, 1U);
  expect(serial.assignment(2)->slot_id == serial.assignment(3)->slot_id,
         "serial streaming reuses one constant slot");
  expect(prefetched.assignment(2)->slot_id != prefetched.assignment(3)->slot_id,
         "one-operation prefetch reserves a second constant slot");
  expect(prefetched.total_bytes > serial.total_bytes,
         "prefetch storage is explicit in the memory budget");
}

void test_memory_plan_omits_backend_replaced_constants() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, TensorRole::Constant, {1, 4}},
      {3, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
  const std::unordered_set<std::uint32_t> replaced_constants = {2U};
  const auto ordinary = dif::compiler::plan_memory(program);
  const auto replaced = dif::compiler::plan_memory(
      program, 256U, 0U, {}, replaced_constants);
  expect(ordinary.assignment(2U) != nullptr &&
             replaced.assignment(2U) == nullptr &&
             replaced.total_bytes < ordinary.total_bytes,
         "memory plan omits a semantic constant replaced by a prepared "
         "backend primitive");

  bool rejected_nonconstant = false;
  try {
    (void)dif::compiler::plan_memory(
        program, 256U, 0U, {},
        std::unordered_set<std::uint32_t>{1U});
  } catch (const dif::Error &) {
    rejected_nonconstant = true;
  }
  expect(rejected_nonconstant,
         "memory plan rejects replacement of a dynamic semantic input");
}

void test_compiler_streamed_residency_plan() {
  using namespace dif::ir;
  Program program;
  constexpr auto streamed = TensorRole::Constant | TensorRole::Streamed;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 1024}},
      {2, DType::F32, streamed, {1, 1024}},
      {3, DType::F32, streamed, {1, 1024}},
      {4, DType::F32, TensorRole::Internal, {1, 1024}},
      {5, DType::F32, TensorRole::Output, {1, 1024}},
  };
  program.operations = {
      {1, Opcode::Add, {1, 2}, {4}, {}},
      {2, Opcode::Add, {4, 3}, {5}, {}},
  };
  const auto ordinary = dif::compiler::plan_memory(program, 256U, 0U);
  const auto baseline = dif::compiler::plan_streamed_residency(
      program, ordinary.total_bytes, 0U, 0U);
  expect(baseline.resident_tensor_ids.empty() &&
             baseline.streamed_constant_bytes == 2U * 4096U &&
             baseline.required_bytes == ordinary.total_bytes,
         "compiler residency plan preserves streaming under a tight budget");
  const auto resident = dif::compiler::plan_streamed_residency(
      program, ordinary.total_bytes + 4096U, 0U, 0U);
  expect(resident.resident_tensor_ids ==
                 std::vector<std::uint32_t>{2U, 3U} &&
             resident.resident_constant_bytes == 2U * 4096U &&
             resident.streamed_constant_bytes == 0U &&
             resident.required_bytes == ordinary.total_bytes + 4096U,
         "compiler residency plan admits an explicit reusable weight set");
}

void test_cuda_lazy_resident_upload() {
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  constexpr auto streamed = TensorRole::Constant | TensorRole::Streamed;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, streamed, {1, 4}},
      {3, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
  dif::runtime::TensorMap bindings = {
      {1, f32_tensor({1, 4}, {1.0F, 2.0F, 3.0F, 4.0F})},
      {2, f32_tensor({1, 4}, {0.5F, 1.0F, 1.5F, 2.0F})},
  };
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  options.resident_streamed_constants = {2U};
  options.lazy_resident_upload = true;
  options.streamed_release_mapped_pages_per_copy = false;
  auto prepared =
      dif::runtime::make_cuda_executor()->prepare(program, bindings, options);
  const auto first = prepared->run(bindings, options);
  const auto second = prepared->run(bindings, options);
  expect(first.outputs.at(3U).bytes == second.outputs.at(3U).bytes &&
             first.run_telemetry.h2d_copies == 2U &&
             first.run_telemetry.h2d_bytes == 32U &&
             second.run_telemetry.h2d_copies == 1U &&
             second.run_telemetry.h2d_bytes == 16U,
         "lazy resident constant uploads once and remains bit-exact");
}

void test_cuda_f16_biased_convrot_int8() {
#if DIF_HAS_CUTLASS
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  constexpr auto streamed = TensorRole::Constant | TensorRole::Streamed;
  Program program;
  program.tensors = {
      {1, DType::F16, TensorRole::Input, {4, 256}},
      {2, DType::F16, streamed, {16, 256}},
      {3, DType::F16, streamed, {16}},
      {4, DType::F16, TensorRole::Output, {4, 16}},
  };
  program.operations = {{1, Opcode::Linear, {1, 2, 3}, {4}, {}}};
  verify(program);

  std::vector<float> input_values(4U * 256U);
  for (std::size_t index = 0U; index < input_values.size(); ++index)
    input_values[index] = static_cast<float>(static_cast<int>(index % 19U) - 9) /
                          16.0F;
  const std::vector<float> bias_values = {
      -1.0F, -0.75F, -0.5F, -0.25F, -0.125F, -0.0625F, 0.0F, 0.0625F,
      0.125F, 0.25F, 0.5F, 0.75F, 1.0F, 1.25F, 1.5F, 2.0F};
  dif::runtime::TensorMap bindings = {
      {1, float_tensor(DType::F16, {4, 256}, input_values)},
      {2, float_tensor(DType::F16, {16, 256},
                       std::vector<float>(16U * 256U, 0.0F))},
      {3, float_tensor(DType::F16, {16}, bias_values)},
  };
  dif::runtime::RunOptions ordinary_options;
  ordinary_options.warmups = 0U;
  ordinary_options.iterations = 1U;
  const auto ordinary =
      dif::runtime::make_cpu_executor()->run(program, bindings,
                                               ordinary_options);

  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("dif-convrot-f16-test-" + std::to_string(nonce));
  std::filesystem::create_directories(directory);
  const auto cache_path = directory / "convrot.safetensors";
  std::array<std::uint32_t, 20> identity{};
  identity[0] = 0x31525643U;
  identity[1] = 1U;
  identity[2] = 256U;
  identity[3] = 1U;
  const auto fingerprint = dif::ir::fingerprint(program);
  for (std::size_t word = 0U; word < 8U; ++word)
    identity[4U + word] =
        static_cast<std::uint32_t>(fingerprint[word * 4U]) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 1U]) << 8U) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 2U]) << 16U) |
        (static_cast<std::uint32_t>(fingerprint[word * 4U + 3U]) << 24U);
  const std::vector<std::int8_t> quantized(16U * 256U, 0);
  const std::vector<float> scales(16U, 1.0F);
  dif::weights::SafeTensorWriter writer(
      cache_path,
      {{"__meta__.convrot_int8", DType::I32, {20}},
       {"linear.2.weight", DType::I8, {16, 256}},
       {"linear.2.scale", DType::F32, {16}}});
  writer.append(
      "__meta__.convrot_int8",
      {reinterpret_cast<const std::uint8_t *>(identity.data()),
       identity.size() * sizeof(std::uint32_t)});
  writer.append(
      "linear.2.weight",
      {reinterpret_cast<const std::uint8_t *>(quantized.data()),
       quantized.size() * sizeof(std::int8_t)});
  writer.append(
      "linear.2.scale",
      {reinterpret_cast<const std::uint8_t *>(scales.data()),
       scales.size() * sizeof(float)});
  (void)writer.finish();

  for (const auto resident : {false, true}) {
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    options.convrot_int8_checkpoint = cache_path;
    options.convrot_int8_resident = resident;
    options.streamed_release_mapped_pages_per_copy = false;
    const auto candidate =
        dif::runtime::make_cuda_executor()->run(program, bindings, options);
    expect(candidate.outputs.at(4U).bytes == ordinary.outputs.at(4U).bytes &&
               candidate.convrot_int8_linears.size() == 1U &&
               candidate.convrot_int8_linears.front().implementation ==
                   "generic_diffir_linear_cutlass_scaled_f16",
           resident
               ? "resident F16 biased ConvRot INT8 fuses bias and matches semantics"
               : "streamed F16 biased ConvRot INT8 fuses bias and matches semantics");
  }
  std::filesystem::remove_all(directory);
#endif
}

void test_compiler_and_cuda_reshape_alias_plan() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 1024}},
      {2, DType::F32, TensorRole::Internal, {1024, 1}},
      {3, DType::F32, TensorRole::Constant, {1024, 1}},
      {4, DType::F32, TensorRole::Output, {1024, 1}},
  };
  program.operations = {
      {1, Opcode::Reshape, {1}, {2}, {}},
      {2, Opcode::Add, {2, 3}, {4}, {}},
  };
  const auto aliases = dif::compiler::plan_reshape_aliases(program);
  const auto ordinary = dif::compiler::plan_memory(program);
  const auto aliased = dif::compiler::plan_memory(
      program, 256U, 0U, {}, {}, aliases.output_to_root_input);
  expect(aliases.operation_ids == std::vector<std::uint32_t>{1U} &&
             aliases.output_to_root_input.at(2U) == 1U &&
             aliased.total_bytes < ordinary.total_bytes,
         "compiler reshape alias plan removes an internal materialization");

  dif::runtime::TensorMap bindings;
  std::vector<float> input_values(1024U);
  std::vector<float> constant_values(1024U);
  for (std::size_t index = 0U; index < input_values.size(); ++index) {
    input_values[index] = static_cast<float>(index) * 0.25F;
    constant_values[index] = static_cast<float>(index % 7U) - 3.0F;
  }
  bindings.emplace(1U, f32_tensor({1U, 1024U}, input_values));
  bindings.emplace(3U, f32_tensor({1024U, 1U}, constant_values));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  if (!dif::runtime::cuda_available())
    return;
  options.alias_reshape_operations = aliases.operation_ids;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(4U).bytes == cpu.outputs.at(4U).bytes &&
             cuda.run_telemetry.kernel_launches == 1U,
         "CUDA reshape alias executes zero-copy and remains bit-exact");
}

void test_weight_bundle_roundtrip() {
  using namespace dif::ir;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("dif-bundle-test-" + std::to_string(nonce));
  std::filesystem::create_directories(directory);
  const auto shard_path = directory / "model.safetensors";
  const auto bundle_path = directory / "model.difbind";
  const std::string header =
      R"({"weight":{"dtype":"F32","shape":[1,4],"data_offsets":[0,16]}})";
  {
    std::ofstream shard(shard_path, std::ios::binary);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      shard.put(static_cast<char>(header_size >> shift));
    shard.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::array<float, 4> values = {1, 2, 3, 4};
    shard.write(reinterpret_cast<const char *>(values.data()), sizeof(values));
  }
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {1, 4}},
      {2, DType::F32, TensorRole::Constant, {1, 4}},
      {3, DType::F32, TensorRole::Output, {1, 4}},
  };
  program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
  const auto metadata = dif::weights::read_safetensors(shard_path);
  const auto *entry = metadata.find("weight");
  expect(entry != nullptr, "synthetic SafeTensors entry parsed");
  if (entry) {
    dif::weights::WeightBundle bundle;
    bundle.program_fingerprint = dif::ir::fingerprint(program);
    bundle.index_fingerprint = dif::sha256(
        std::span<const std::uint8_t>{});
    bundle.shards.push_back({shard_path, std::filesystem::file_size(shard_path),
                             dif::sha256_file(shard_path)});
    bundle.bindings.push_back({2, 0, "weight", entry->dtype, entry->dims,
                               entry->file_offset, entry->byte_count});
    dif::weights::write_weight_bundle(bundle, bundle_path);
    const auto decoded = dif::weights::read_weight_bundle(bundle_path);
    const auto tensors = dif::weights::load_weight_bundle(decoded, program, true);
    expect(tensors.contains(2), "weight bundle maps bound tensor");
    if (tensors.contains(2)) {
      const auto values = tensors.at(2).f32();
      expect(values.size() == 4U && values[0] == 1.0F && values[3] == 4.0F,
             "weight bundle preserves mapped values");
    }
  }
  std::filesystem::remove_all(directory);
}

void test_safetensors_streaming_writer() {
  using namespace dif::ir;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("dif-safetensors-writer-test-" +
                          std::to_string(nonce));
  std::filesystem::create_directories(directory);
  const auto path = directory / "streamed.safetensors";

  const std::array<std::uint8_t, 4> packed = {0x18U, 0x2fU, 0x37U, 0x40U};
  const auto scales = float_tensor(DType::BF16, {2, 2},
                                   {0.25F, 0.5F, 1.0F, 2.0F});
  dif::weights::SafeTensorWriter writer(
      path, {{"packed\\\"weight", DType::I8, {2, 2}},
             {"scales", DType::BF16, {2, 2}}});
  writer.append("packed\\\"weight", packed);
  writer.append("scales", scales.bytes);
  const auto metadata = writer.finish();
  expect(metadata.tensors.size() == 2U,
         "streaming SafeTensors writer records every tensor");
  const auto mapped_packed =
      dif::weights::map_safetensor(metadata, "packed\\\"weight");
  const auto mapped_scales = dif::weights::map_safetensor(metadata, "scales");
  expect(mapped_packed.mapping == mapped_scales.mapping,
         "SafeTensors entries share one read-only file mapping");
  expect(mapped_packed.byte_size() == packed.size() &&
             std::equal(packed.begin(), packed.end(), mapped_packed.data()),
         "streaming SafeTensors writer preserves packed bytes");
  const auto values = float_values(mapped_scales);
  expect(values == std::vector<float>({0.25F, 0.5F, 1.0F, 2.0F}),
         "streaming SafeTensors writer preserves BF16 values");
  std::filesystem::remove_all(directory);
}

void test_safetensors_tensor_metadata_is_skipped() {
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto directory = std::filesystem::temp_directory_path() /
                         ("dif-safetensors-metadata-test-" +
                          std::to_string(nonce));
  std::filesystem::create_directories(directory);
  const auto path = directory / "metadata.safetensors";
  const std::string header =
      R"({"__meta__.version":{"dtype":"I64","shape":[1],"data_offsets":[0,8]},"weight":{"dtype":"F32","shape":[1,4],"data_offsets":[8,24]}})";
  {
    std::ofstream shard(path, std::ios::binary);
    const auto header_size = static_cast<std::uint64_t>(header.size());
    for (unsigned shift = 0; shift < 64U; shift += 8U)
      shard.put(static_cast<char>(header_size >> shift));
    shard.write(header.data(), static_cast<std::streamsize>(header.size()));
    const std::int64_t metadata = 1;
    const std::array<float, 4> values = {1, 2, 3, 4};
    shard.write(reinterpret_cast<const char *>(&metadata), sizeof(metadata));
    shard.write(reinterpret_cast<const char *>(values.data()), sizeof(values));
  }
  const auto file = dif::weights::read_safetensors(path);
  expect(file.tensors.size() == 1U && file.find("weight") != nullptr &&
             file.find("__meta__.version") == nullptr,
         "SafeTensors tensor-form metadata is validated but not exposed");
  const auto weight = dif::weights::map_safetensor(file, "weight");
  const auto values = weight.f32();
  expect(values.size() == 4U && values[0] == 1.0F && values[3] == 4.0F,
         "SafeTensors metadata skipping preserves following tensor offsets");
  std::filesystem::remove_all(directory);
}

void test_int4_weight_rewrite_and_cpu_execution() {
  using namespace dif::ir;
  Program original;
  original.tensors = {
      {1, DType::BF16, TensorRole::Input, {1, 16}},
      {2, DType::BF16, TensorRole::Constant | TensorRole::Streamed, {2, 16}},
      {3, DType::BF16, TensorRole::Output, {1, 2}},
  };
  original.operations = {
      {1, Opcode::Linear, {1, 2}, {3},
       {Attribute::u64(AttrKey::Implementation, 1U)}},
  };
  dif::ir::verify(original);

  std::vector<float> input_values(16, 1.0F);
  std::vector<float> weight_values;
  for (std::size_t row = 0; row < 2U; ++row) {
    for (int value = -7; value <= 7; ++value)
      weight_values.push_back(static_cast<float>(value) *
                              (row == 0U ? 0.5F : 1.0F));
    weight_values.push_back(0.0F);
  }
  const auto input = float_tensor(DType::BF16, {1, 16}, input_values);
  const auto weight = float_tensor(DType::BF16, {2, 16}, weight_values);
  auto original_executor = dif::runtime::make_cpu_executor();
  const auto reference =
      original_executor->run(original, {{1, input}, {2, weight}}, {});

  const auto rewrite = dif::compiler::rewrite_int4_weights(original, 16U);
  expect(rewrite.entries.size() == 1U,
         "INT4 rewrite finds the rank-2 Linear constant");
  expect(rewrite.program.operations.size() == 2U &&
             rewrite.program.operations[0].opcode == Opcode::DequantizeInt4 &&
             rewrite.program.operations[1].opcode == Opcode::Linear,
         "INT4 dequantization is scheduled before its first consumer");
  const auto *rewritten_weight = rewrite.program.tensor(2);
  expect(rewritten_weight &&
             !rewritten_weight->has_role(TensorRole::Constant) &&
             !rewritten_weight->has_role(TensorRole::Streamed),
         "INT4 rewrite turns the source weight into an internal tensor");

  const auto quantized = dif::compiler::quantize_int4_weight(weight, 16U);
  expect(quantized.maximum_absolute_error == 0.0F,
         "exactly representable INT4 values quantize without error");
  const auto &entry = rewrite.entries.front();
  const auto candidate = original_executor->run(
      rewrite.program,
       {{1, input},
        {entry.packed_tensor_id, quantized.packed},
       {entry.scales_tensor_id, quantized.scales}},
      {});
  expect(float_values(reference.outputs.at(3)) ==
             float_values(candidate.outputs.at(3)),
         "rewritten INT4 graph matches the exact BF16 CPU reference");

  auto outlier_values = weight_values;
  outlier_values[0] = 100.0F;
  outlier_values[16] = -100.0F;
  const auto outlier_weight =
      float_tensor(DType::BF16, {2, 16}, outlier_values);
  const auto uncorrected =
      dif::compiler::quantize_int4_weight(outlier_weight, 16U);
  const auto corrected = dif::compiler::quantize_int4_weight(
      outlier_weight, 16U, dif::compiler::Int4Correction::OneOutlier);
  expect(corrected.squared_error < uncorrected.squared_error,
         "one-outlier correction lowers INT4 reconstruction error");
  const auto corrected_rewrite = dif::compiler::rewrite_int4_weights(
      original, 16U, dif::compiler::Int4Correction::OneOutlier);
  expect(corrected_rewrite.program.operations.front().inputs.size() == 4U,
         "one-outlier rewrite exposes indices and residuals in DiffIR");
  const auto corrected_reference =
      original_executor->run(original, {{1, input}, {2, outlier_weight}}, {});
  const auto &corrected_entry = corrected_rewrite.entries.front();
  const auto corrected_candidate = original_executor->run(
      corrected_rewrite.program,
      {{1, input},
       {corrected_entry.packed_tensor_id, corrected.packed},
       {corrected_entry.scales_tensor_id, corrected.scales},
       {corrected_entry.outlier_indices_tensor_id, corrected.outlier_indices},
       {corrected_entry.outlier_residuals_tensor_id,
        corrected.outlier_residuals}},
      {});
  expect(float_values(corrected_reference.outputs.at(3)) ==
             float_values(corrected_candidate.outputs.at(3)),
         "one-outlier INT4 graph exactly restores a representable outlier");
}

void test_int5_weight_rewrite_and_cpu_execution() {
  using namespace dif::ir;
  Program original;
  original.tensors = {
      {1, DType::BF16, TensorRole::Input, {1, 32}},
      {2, DType::BF16, TensorRole::Constant, {1, 32}},
      {3, DType::BF16, TensorRole::Output, {1, 1}},
  };
  original.operations = {
      {1, Opcode::Linear, {1, 2}, {3},
       {Attribute::u64(AttrKey::Implementation, 1U)}},
  };
  std::vector<float> values;
  for (int value = -15; value <= 15; ++value)
    values.push_back(static_cast<float>(value) * 0.25F);
  values.push_back(0.0F);
  const auto input = float_tensor(DType::BF16, {1, 32},
                                  std::vector<float>(32, 1.0F));
  const auto weight = float_tensor(DType::BF16, {1, 32}, values);
  const auto rewrite =
      dif::compiler::rewrite_lowbit_weights(original, 5U, 32U);
  expect(rewrite.program.operations.front().opcode == Opcode::DequantizeInt5,
         "five-bit rewrite emits the explicit INT5 DiffIR opcode");
  const auto quantized =
      dif::compiler::quantize_lowbit_weight(weight, 5U, 32U);
  expect(quantized.maximum_absolute_error == 0.0F,
         "exactly representable INT5 values quantize without error");
  auto executor = dif::runtime::make_cpu_executor();
  const auto reference = executor->run(original, {{1, input}, {2, weight}}, {});
  const auto &entry = rewrite.entries.front();
  const auto candidate = executor->run(
      rewrite.program,
      {{1, input},
       {entry.packed_tensor_id, quantized.packed},
       {entry.scales_tensor_id, quantized.scales}},
      {});
  expect(float_values(reference.outputs.at(3)) ==
             float_values(candidate.outputs.at(3)),
         "rewritten INT5 graph matches the exact BF16 CPU reference");

  std::vector<float> sensitive_weights(32, 3.0F);
  sensitive_weights[0] = 0.1F;
  std::vector<float> sensitive_input(32, 0.0F);
  sensitive_input[0] = 100.0F;
  const auto sensitive_weight =
      float_tensor(DType::BF16, {1, 32}, sensitive_weights);
  const auto sensitive_activation =
      float_tensor(DType::F32, {1, 32}, sensitive_input);
  const auto graph_input =
      float_tensor(DType::BF16, {1, 32}, sensitive_input);
  const auto plain_quantized =
      dif::compiler::quantize_lowbit_weight(sensitive_weight, 5U, 32U);
  const auto aware_quantized = dif::compiler::quantize_lowbit_weight(
      sensitive_weight, 5U, 32U, dif::compiler::Int4Correction::None,
      &sensitive_activation, 0.5F);
  const auto aware_rewrite = dif::compiler::rewrite_lowbit_weights(
      original, 5U, 32U, dif::compiler::Int4Correction::None, {2U});
  const auto sensitive_reference = executor->run(
      original, {{1, graph_input}, {2, sensitive_weight}}, {});
  const auto plain_candidate = executor->run(
      rewrite.program,
      {{1, graph_input},
       {entry.packed_tensor_id, plain_quantized.packed},
       {entry.scales_tensor_id, plain_quantized.scales}},
      {});
  const auto &aware_entry = aware_rewrite.entries.front();
  const auto aware_candidate = executor->run(
      aware_rewrite.program,
      {{1, graph_input},
       {aware_entry.packed_tensor_id, aware_quantized.packed},
       {aware_entry.scales_tensor_id, aware_quantized.scales},
       {aware_entry.column_scales_tensor_id,
        aware_quantized.column_scales}},
      {});
  const auto reference_value =
      dif::runtime::load_float(sensitive_reference.outputs.at(3), 0U);
  const auto plain_error = std::abs(
      dif::runtime::load_float(plain_candidate.outputs.at(3), 0U) -
      reference_value);
  const auto aware_error = std::abs(
      dif::runtime::load_float(aware_candidate.outputs.at(3), 0U) -
      reference_value);
  expect(aware_error < plain_error,
         "activation-aware column companding lowers a sensitive output error");

  auto direct_program = aware_rewrite.program;
  for (auto &operation : direct_program.operations) {
    if (operation.opcode != Opcode::Linear)
      continue;
    for (auto &attribute : operation.attributes) {
      if (attribute.key == AttrKey::Implementation)
        attribute = Attribute::u64(AttrKey::Implementation, 3U);
    }
  }
  dif::ir::verify(direct_program);
  const auto direct_cuda = dif::compiler::emit_cuda(direct_program);
  expect(direct_cuda.launch_inputs.contains(1U) &&
             direct_cuda.skipped_operations.contains(
                 aware_entry.dequantize_operation_id) &&
             direct_cuda.source.find("__shfl_down_sync") != std::string::npos,
         "explicit implementation 3 fuses an exclusive INT5 dequant-Linear "
         "chain while preserving semantic DiffIR");
  std::unordered_set<std::uint32_t> excluded;
  for (const auto &operation : direct_program.operations) {
    if (direct_cuda.skipped_operations.contains(operation.id))
      excluded.insert(operation.outputs.begin(), operation.outputs.end());
  }
  const auto ordinary_plan = dif::compiler::plan_memory(direct_program);
  const auto fused_plan =
      dif::compiler::plan_memory(direct_program, 256U, 0U, excluded);
  expect(fused_plan.assignment(2U) == nullptr &&
             fused_plan.total_bytes < ordinary_plan.total_bytes,
         "direct packed Linear lowering omits its dead dequantized weight "
         "buffer from the CUDA memory plan");
}

void test_operation_slice_preserves_stable_ids_and_boundary_roles() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {2, 4}},
      {2, DType::BF16,
       static_cast<std::uint32_t>(TensorRole::Constant) |
           static_cast<std::uint32_t>(TensorRole::Streamed),
       {2, 4}},
      {3, DType::BF16, TensorRole::Internal, {2, 4}},
      {4, DType::BF16, TensorRole::Internal, {2, 4}},
      {5, DType::BF16, TensorRole::Output, {2, 4}},
  };
  program.operations = {
      {7, Opcode::Add, {1, 2}, {3}, {}},
      {8, Opcode::SiLU, {3}, {4}, {}},
      {9, Opcode::Add, {4, 1}, {5}, {}},
  };
  dif::ir::verify(program);

  const auto prefix = dif::compiler::slice_operations(program, 7U, 8U);
  expect(prefix.operations.size() == 2U &&
             prefix.operations.front().id == 7U &&
             prefix.operations.back().id == 8U,
         "operation slicing preserves source operation ids");
  const auto *input = prefix.tensor(1U);
  const auto *constant = prefix.tensor(2U);
  const auto *internal = prefix.tensor(3U);
  const auto *output = prefix.tensor(4U);
  expect(input && input->has_role(TensorRole::Input) && constant &&
             constant->has_role(TensorRole::Constant) &&
             constant->has_role(TensorRole::Streamed) && internal &&
             internal->roles ==
                 static_cast<std::uint32_t>(TensorRole::Internal) &&
             output &&
             output->has_role(TensorRole::Output),
         "operation slicing assigns standalone boundary roles");

  const auto suffix = dif::compiler::slice_operations(program, 8U, 9U);
  expect(suffix.tensor(3U) && suffix.tensor(3U)->has_role(TensorRole::Input) &&
             suffix.tensor(4U) &&
             suffix.tensor(4U)->roles ==
                 static_cast<std::uint32_t>(TensorRole::Internal) &&
             suffix.tensor(5U) &&
             suffix.tensor(5U)->has_role(TensorRole::Output),
         "operation slicing promotes incoming and outgoing values");

  auto branched = program;
  branched.tensors.push_back(
      {6, DType::BF16, TensorRole::Output, {2, 4}});
  branched.operations.push_back({10, Opcode::Add, {3, 1}, {6}, {}});
  dif::ir::verify(branched);
  const auto branch_prefix =
      dif::compiler::slice_operations(branched, 7U, 8U);
  expect(branch_prefix.tensor(3U) &&
             branch_prefix.tensor(3U)->has_role(TensorRole::Output),
         "operation slicing exports values consumed by a later branch");
}

void test_cuda_linear_tuning_and_exact_swiglu_fusion() {
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  constexpr std::uint64_t rows = 65U;
  constexpr std::uint64_t inner = 32U;
  constexpr std::uint64_t width = 65U;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {rows, inner}},
      {2, DType::BF16, TensorRole::Constant, {2U * width, inner}},
      {3, DType::BF16, TensorRole::Internal, {rows, 2U * width}},
      {4, DType::BF16, TensorRole::Output, {rows, width}},
  };
  program.operations = {
      {11, Opcode::Linear, {1, 2}, {3},
       {Attribute::u64(AttrKey::Implementation, 1U)}},
      {12, Opcode::SwiGlu, {3}, {4},
       {Attribute::boolean(AttrKey::GateFirst, true)}},
  };
  dif::ir::verify(program);

  std::vector<float> input_values(rows * inner);
  for (std::size_t index = 0; index < input_values.size(); ++index)
    input_values[index] =
        static_cast<float>(static_cast<int>(index % 17U) - 8) * 0.125F;
  std::vector<float> weight_values(2U * width * inner, 0.0F);
  for (std::uint64_t row = 0U; row < 2U * width; ++row)
    weight_values[row * inner + row % inner] =
        row < width ? 0.5F : 0.25F;
  const dif::runtime::TensorMap bindings = {
      {1, float_tensor(DType::BF16, {rows, inner}, input_values)},
      {2, float_tensor(DType::BF16, {2U * width, inner}, weight_values)},
  };

  dif::runtime::RunOptions baseline_options;
  baseline_options.warmups = 0U;
  baseline_options.iterations = 1U;
  baseline_options.tune_linear_operations = {11U};
  baseline_options.linear_tuning_warmups = 0U;
  baseline_options.linear_tuning_iterations = 1U;
  baseline_options.linear_tuning_sessions = 2U;
  const auto baseline = dif::runtime::make_cuda_executor()->run(
      program, bindings, baseline_options);
  expect(baseline.linear_tuning_results.size() == 1U &&
             !baseline.linear_tuning_results.front().candidates.empty(),
         "explicit cuBLASLt tuning records admitted algorithms");

  dif::runtime::RunOptions fused_options;
  fused_options.warmups = 0U;
  fused_options.iterations = 1U;
  fused_options.fuse_linear_swiglu_operations = {11U};
  const auto fused = dif::runtime::make_cuda_executor()->run(
      program, bindings, fused_options);
  const auto expected_eliminated = rows * 2U * width * sizeof(std::uint16_t);
  expect(fused.primitive_fusions.size() == 1U &&
             fused.primitive_fusions.front().linear_operation_id == 11U &&
             fused.primitive_fusions.front().swiglu_operation_id == 12U &&
             fused.primitive_fusions.front().eliminated_intermediate_bytes ==
                 expected_eliminated,
         "explicit Linear-to-SwiGLU fusion reports its exact memory saving");
  expect(fused.resident_bytes < baseline.resident_bytes,
         "fused Linear-to-SwiGLU excludes the FC1 intermediate allocation");
  expect(fused.outputs.at(4U).bytes == baseline.outputs.at(4U).bytes,
         "fused Linear-to-SwiGLU preserves BF16 output bits");
}

void test_cuda_cutlass_linear_primitive() {
#if DIF_HAS_CUTLASS
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  constexpr std::uint64_t rows = 64U;
  constexpr std::uint64_t inner = 32U;
  constexpr std::uint64_t width = 64U;
  Program program;
  program.tensors = {
      {1, DType::BF16, TensorRole::Input, {rows, inner}},
      {2, DType::BF16, TensorRole::Constant, {width, inner}},
      {3, DType::BF16, TensorRole::Output, {rows, width}},
  };
  program.operations = {
      {11, Opcode::Linear, {1, 2}, {3},
       {Attribute::u64(AttrKey::Implementation, 1U)}},
  };
  dif::ir::verify(program);

  std::vector<float> input_values(rows * inner);
  std::vector<float> weight_values(width * inner);
  for (std::size_t index = 0; index < input_values.size(); ++index)
    input_values[index] =
        static_cast<float>(static_cast<int>(index % 23U) - 11) * 0.0625F;
  for (std::size_t index = 0; index < weight_values.size(); ++index)
    weight_values[index] =
        static_cast<float>(static_cast<int>(index % 19U) - 9) * 0.03125F;
  const dif::runtime::TensorMap bindings = {
      {1, float_tensor(DType::BF16, {rows, inner}, input_values)},
      {2, float_tensor(DType::BF16, {width, inner}, weight_values)},
  };

  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  const auto cublas =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  options.cutlass_linear_operations = {{11U, 1U}};
  const auto cutlass =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cutlass.outputs.at(3U).bytes == cublas.outputs.at(3U).bytes,
         "CUTLASS BF16 GEMM primitive preserves cuBLASLt output bits");
  expect(cutlass.gemm_primitives.size() == 1U &&
             cutlass.gemm_primitives.front().operation_id == 11U &&
             cutlass.gemm_primitives.front().schedule == 1U &&
             cutlass.gemm_primitives.front().threads_per_block == 128U &&
             cutlass.gemm_primitives.front().dynamic_shared_bytes != 0U,
         "CUTLASS BF16 GEMM primitive reports its schedule resources");
#endif
}

} // namespace


dif::ir::Program audio_conv_probe(bool transposed) {
  using namespace dif::ir;
  Program program;
  program.tensors.push_back(
      {1, DType::F32, static_cast<std::uint32_t>(TensorRole::Input),
       {1, 4, 8}});
  if (transposed)
    program.tensors.push_back(
        {2, DType::F32, static_cast<std::uint32_t>(TensorRole::Constant),
         {4, 6, 4}});
  else
    program.tensors.push_back(
        {2, DType::F32, static_cast<std::uint32_t>(TensorRole::Constant),
         {6, 4, 3}});
  program.tensors.push_back(
      {3, DType::F32, static_cast<std::uint32_t>(TensorRole::Constant), {6}});
  // plain: pad 1/1, stride 1, dilation 1, K=3 -> L_out = 8.
  // transposed: padded 10, full = (10-1)*2+4 = 22, trim 3/3 -> L_out = 16.
  program.tensors.push_back(
      {4, DType::F32, static_cast<std::uint32_t>(TensorRole::Output),
       {1, 6, transposed ? std::uint64_t{16} : std::uint64_t{8}}});
  Operation op;
  op.id = 1;
  op.opcode = Opcode::Conv1d;
  op.inputs = {1, 2, 3};
  op.outputs = {4};
  op.attributes.push_back(Attribute::u64(AttrKey::PadLeft, 1));
  op.attributes.push_back(Attribute::u64(AttrKey::PadRight, 1));
  if (transposed) {
    op.attributes.push_back(Attribute::u64(AttrKey::Stride, 2));
    op.attributes.push_back(Attribute::boolean(AttrKey::Transposed, true));
    op.attributes.push_back(Attribute::u64(AttrKey::TrimLeft, 3));
    op.attributes.push_back(Attribute::u64(AttrKey::TrimRight, 3));
    op.attributes.push_back(Attribute::u64(AttrKey::PadMode, 1));
  }
  program.operations.push_back(op);
  return program;
}

dif::ir::Program audio_snake_probe() {
  using namespace dif::ir;
  Program program;
  program.tensors.push_back(
      {1, DType::F32, static_cast<std::uint32_t>(TensorRole::Input),
       {1, 4, 8}});
  program.tensors.push_back(
      {2, DType::F32, static_cast<std::uint32_t>(TensorRole::Constant), {4}});
  program.tensors.push_back(
      {3, DType::F32, static_cast<std::uint32_t>(TensorRole::Constant), {4}});
  program.tensors.push_back(
      {4, DType::F32, static_cast<std::uint32_t>(TensorRole::Output),
       {1, 4, 8}});
  Operation op;
  op.id = 1;
  op.opcode = Opcode::SnakeBeta;
  op.inputs = {1, 2, 3};
  op.outputs = {4};
  program.operations.push_back(op);
  return program;
}

void expect_verifier_rejects(dif::ir::Program program, const char *message) {
  bool rejected = false;
  try {
    dif::ir::verify(program);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, message);
}

void test_audio_opcode_verifier_contract() {
  using namespace dif::ir;
  // Positive: both conv modes and snake_beta verify, and survive the codec.
  for (const auto transposed : {false, true}) {
    auto program = audio_conv_probe(transposed);
    dif::ir::verify(program);
    const auto decoded = dif::ir::decode(dif::ir::encode(program));
    expect(decoded.operations.size() == 1 &&
               decoded.operations[0].opcode == Opcode::Conv1d &&
               decoded.operations[0].attributes.size() ==
                   program.operations[0].attributes.size(),
           "conv1d survives the DiffIR codec round-trip");
  }
  {
    auto program = audio_snake_probe();
    dif::ir::verify(program);
    const auto decoded = dif::ir::decode(dif::ir::encode(program));
    expect(decoded.operations.size() == 1 &&
               decoded.operations[0].opcode == Opcode::SnakeBeta,
           "snake_beta survives the DiffIR codec round-trip");
  }

  // The classic port bug, both directions: forward weight layout used in
  // transposed mode and vice versa must be rejected fail-closed.
  {
    auto program = audio_conv_probe(false);
    program.tensors[1].dims = {4, 6, 3};  // transposed layout in forward mode
    program.tensors[3].dims = {1, 4, 8};
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects transposed weight layout in "
                            "forward mode");
  }
  {
    auto program = audio_conv_probe(true);
    program.tensors[1].dims = {6, 4, 4};  // forward layout in transposed mode
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects forward weight layout in "
                            "transposed mode");
  }
  {
    auto program = audio_conv_probe(true);
    program.operations[0].attributes.push_back(
        Attribute::u64(AttrKey::Dilation, 3));
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects dilation in transposed mode");
  }
  {
    auto program = audio_conv_probe(false);
    program.operations[0].attributes.push_back(
        Attribute::u64(AttrKey::Groups, 3));
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects groups not dividing channels");
  }
  {
    auto program = audio_conv_probe(false);
    program.operations[0].attributes.push_back(
        Attribute::u64(AttrKey::TrimLeft, 1));
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects trim outside transposed mode");
  }
  {
    auto program = audio_conv_probe(false);
    program.operations[0].attributes.push_back(
        Attribute::u64(AttrKey::PadMode, 2));
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects an unknown pad mode");
  }
  {
    auto program = audio_conv_probe(false);
    program.tensors[3].dims = {1, 6, 9};  // wrong L_out
    expect_verifier_rejects(std::move(program),
                            "conv1d rejects mismatched output length");
  }
  {
    auto program = audio_conv_probe(false);
    program.operations[0].attributes.push_back(
        Attribute::u64(AttrKey::Stride, 0));
    expect_verifier_rejects(std::move(program), "conv1d rejects stride zero");
  }
  {
    auto program = audio_snake_probe();
    program.tensors[1].dims = {5};
    expect_verifier_rejects(std::move(program),
                            "snake_beta rejects alpha not matching channels");
  }
  {
    auto program = audio_snake_probe();
    program.operations[0].attributes.push_back(
        Attribute::f64(AttrKey::Epsilon, 0.0));
    expect_verifier_rejects(std::move(program),
                            "snake_beta rejects a non-positive epsilon");
  }
}


void test_audio_bigvgan_frontend_contract() {
  using namespace dif::frontend;
  // Full program at the accepted geometry (B=2 stereo batch, T=292).
  const auto build = build_audio_bigvgan_program(2U, 292U, 8U);
  dif::ir::verify(build.program);
  // BigVGAN census derivation: Conv1d = denorm 1 +
  // dec_in_proj 1 + conv_pre 1 + 7 stages x (ups 1 + 3 blocks x 3 dilations
  // x (2 alias-free resamplers x 2 + 2 convs) = 54) + post resamplers 2 +
  // conv_post 1 = 391. SnakeBeta = 7 x 3 x 3 x 2 + activation_post = 127.
  expect(build.conv1d_operations == 391U,
         "audio frontend emits exactly 391 conv1d operations");
  expect(build.snake_beta_operations == 127U,
         "audio frontend emits exactly 127 snake_beta operations");
  const auto *output = build.program.tensor(build.waveform_output_id);
  expect(output && output->dims ==
             std::vector<std::uint64_t>{2U, 1U, 292U * 800U},
         "audio frontend output is [2,1,800*T]");
  std::size_t conv_count = 0, snake_count = 0;
  for (const auto &operation : build.program.operations) {
    conv_count += operation.opcode == dif::ir::Opcode::Conv1d;
    snake_count += operation.opcode == dif::ir::Opcode::SnakeBeta;
  }
  expect(conv_count == build.conv1d_operations &&
             snake_count == build.snake_beta_operations,
         "audio frontend census counters match the emitted program");
  // Determinism: the same build twice encodes to identical bytes.
  const auto second = build_audio_bigvgan_program(2U, 292U, 8U);
  expect(dif::ir::encode(build.program) == dif::ir::encode(second.program),
         "audio frontend build is deterministic");

  // Truncated boundaries: pre (stages=0) and each stage keep verifying and
  // land on the documented lengths (292 * prod(rates[:i])).
  const std::vector<std::uint64_t> rates{5U, 5U, 2U, 2U, 2U, 2U, 2U};
  std::uint64_t length = 292U;
  for (std::uint64_t stages = 0U; stages <= 7U; ++stages) {
    const auto truncated = build_audio_bigvgan_program(2U, 292U, stages);
    dif::ir::verify(truncated.program);
    const auto *boundary =
        truncated.program.tensor(truncated.waveform_output_id);
    const auto expected_channels =
        stages == 0U ? 1024U : (1024U >> stages);
    expect(boundary && boundary->dims ==
               std::vector<std::uint64_t>{2U, expected_channels, length},
           "audio frontend truncated boundary geometry");
    if (stages < 7U)
      length *= rates[stages];
  }
}


// The WAV writer must reproduce
// serenitymojo/audio/wav.mojo:120-165 exactly — 44-byte RIFF/WAVE PCM
// header, interleaved channels, clamp to [-1,1], then round half AWAY from
// zero into int16 via Int(x*32767 +/- 0.5) (truncation toward zero after
// the offset).  Header byte-identity is an artifact-gate bar, so it is
// pinned here rather than only end to end.
void test_wav_pcm16_writer_contract() {
  const auto path =
      std::filesystem::temp_directory_path() / "dif-wav-writer-contract.wav";
  std::filesystem::remove(path);
  // Channel-major [2, 5]: channel 0 then channel 1.
  const std::vector<float> waveform{
      0.0F, 1.0F, -1.0F, 2.0F, -2.0F,
      0.5F, -0.5F, 1.0F / 32767.0F, -1.0F / 32767.0F, 0.25F};
  dif::support::write_wav_pcm16(path, waveform, 2U, 5U, 32000U);
  std::ifstream input(path, std::ios::binary);
  std::vector<std::uint8_t> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  const std::vector<std::uint8_t> expected_header{
      'R', 'I', 'F', 'F', 56U, 0U, 0U, 0U,
      'W', 'A', 'V', 'E',
      'f', 'm', 't', ' ', 16U, 0U, 0U, 0U,
      1U, 0U, 2U, 0U,
      0x00U, 0x7DU, 0x00U, 0x00U,   // 32000 Hz
      0x00U, 0xF4U, 0x01U, 0x00U,   // byte rate 32000*2*2 = 128000
      4U, 0U, 16U, 0U,
      'd', 'a', 't', 'a', 20U, 0U, 0U, 0U};
  expect(bytes.size() == 44U + 20U, "wav writer emits 44-byte header + PCM");
  expect(std::equal(expected_header.begin(), expected_header.end(),
                    bytes.begin()),
         "wav writer header is byte-identical to wav.mojo's layout");
  std::vector<std::int16_t> samples(10U);
  std::memcpy(samples.data(), bytes.data() + 44U, 20U);
  // Interleaved L/R with wav.mojo's clamp-then-round-away-from-zero.
  const std::vector<std::int16_t> expected_samples{
      0, 16384,          // 0.0 ; 0.5*32767 = 16383.5 -> 16384
      32767, -16384,     // 1.0 ; -0.5*32767 = -16383.5 -> -16384
      -32767, 1,         // -1.0 ; (1/32767)*32767 = 1
      32767, -1,         // clamp 2.0 -> 1.0 ; -(1/32767)*32767 = -1
      -32767, 8192};     // clamp -2.0 -> -1.0 ; 0.25*32767 = 8191.75 -> 8192
  expect(samples == expected_samples,
         "wav writer quantizer matches wav.mojo sample for sample");
  std::filesystem::remove(path);
}

void test_png_rgb8_writer_contract() {
  const auto path =
      std::filesystem::temp_directory_path() / "dif-png-writer-contract.png";
  std::filesystem::remove(path);
  const std::array<std::uint8_t, 6> rgb{255U, 0U, 0U, 0U, 128U, 255U};
  dif::write_png_rgb8(path, 2U, 1U, rgb);
  std::ifstream input(path, std::ios::binary);
  const std::vector<std::uint8_t> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  const std::array<std::uint8_t, 8> signature{
      0x89U, 'P', 'N', 'G', 0x0dU, 0x0aU, 0x1aU, 0x0aU};
  expect(bytes.size() == 75U &&
             std::equal(signature.begin(), signature.end(), bytes.begin()),
         "PNG writer emits the standard signature and deterministic size");
  expect(bytes[11] == 13U && bytes[12] == 'I' && bytes[13] == 'H' &&
             bytes[14] == 'D' && bytes[15] == 'R' && bytes[19] == 2U &&
             bytes[23] == 1U && bytes[24] == 8U && bytes[25] == 2U,
         "PNG writer emits a 2x1 eight-bit RGB IHDR");
  expect(bytes[36] == 18U && bytes[37] == 'I' && bytes[38] == 'D' &&
             bytes[39] == 'A' && bytes[40] == 'T' && bytes[67] == 'I' &&
             bytes[68] == 'E' && bytes[69] == 'N' && bytes[70] == 'D',
         "PNG writer emits deterministic IDAT and terminal IEND chunks");
  const auto decoded = dif::read_png_rgb8(path);
  expect(decoded.width == 2U && decoded.height == 1U &&
             std::equal(rgb.begin(), rgb.end(), decoded.pixels.begin()),
         "native PNG reader round-trips deterministic RGB8 pixels");
  std::filesystem::remove(path);
}

void test_qwen3vl_vision_and_multimodal_frontend_contract() {
  using namespace dif::frontend;
  using namespace dif::ir;
  const auto vision = build_qwen3vl_vision_program(1U, 2U, 2U);
  verify(vision.program);
  expect(vision.bindings.size() == 350U &&
             vision.linear_operations == 117U &&
             vision.attention_operations == 27U &&
             vision.deepstack_output_ids.size() == 3U &&
             vision.embeds_output_id != 0U,
         "Qwen3-VL vision frontend exposes the released 27-block shared-op contract");
  for (const auto &operation : vision.program.operations)
    expect(operation.opcode != Opcode::H3AdaLNSelect &&
               operation.opcode != Opcode::H3DeinterleaveQkv &&
               operation.opcode != Opcode::H3DeinterleaveQkvWeight,
           "Qwen3-VL vision frontend uses shared operations, not H3 denoiser semantics");

  Qwen3VlConditionerConfig conditioner_config;
  conditioner_config.executed_layers = 3U;
  conditioner_config.vision_token_count = 4U;
  const auto conditioner =
      build_qwen3vl_conditioner_program(12U, conditioner_config);
  verify(conditioner.program);
  expect(conditioner.vision_embeddings_input_id != 0U &&
             conditioner.vision_destination_map_input_id != 0U &&
             conditioner.visual_positions_input_id != 0U &&
             conditioner.vision_deepstack_input_ids.size() == 3U,
         "Qwen3-VL conditioner exposes generic vision splice and deepstack inputs");
  std::uint64_t indexed_updates = 0U;
  for (const auto &operation : conditioner.program.operations)
    indexed_updates += operation.opcode == Opcode::IndexedUpdateRows ? 1U : 0U;
  expect(indexed_updates == 4U,
         "Qwen3-VL conditioner replaces embeddings once and injects three deepstack taps");

  dif::RgbImage image;
  image.width = 256U;
  image.height = 256U;
  image.pixels.assign(256U * 256U * 3U, 255U);
  const auto patches = qwen3vl_vision_image_patch_rows(image);
  expect(patches.dtype == DType::BF16 &&
             patches.dims == std::vector<std::uint64_t>({256U, 1536U}),
         "Qwen3-VL image preprocessing emits merged-order BF16 patch rows");
  const auto *patch_bits =
      reinterpret_cast<const std::uint16_t *>(patches.data());
  expect(patch_bits[0] == dif::runtime::float_to_bf16(1.0F) &&
             patch_bits[256U] == patch_bits[0] &&
             patch_bits[512U] == patch_bits[0],
         "Qwen3-VL still-image preprocessing duplicates temporal planes and preserves channels");

  dif::runtime::Tensor table{DType::BF16, {48U * 48U, 1152U}, {}};
  table.bytes.assign(static_cast<std::size_t>(table.element_count()) * 2U, 0U);
  auto *table_bits = reinterpret_cast<std::uint16_t *>(table.mutable_data());
  for (std::uint64_t row = 0U; row < 48U * 48U; ++row)
    table_bits[row * 1152U] =
        dif::runtime::float_to_bf16(static_cast<float>(row));
  const auto positions =
      qwen3vl_vision_position_embeddings(table, 1U, 48U, 48U);
  const auto *position_bits =
      reinterpret_cast<const std::uint16_t *>(positions.data());
  const std::array<std::uint64_t, 5> expected_rows{0U, 1U, 48U, 49U, 2U};
  for (std::size_t output = 0U; output < expected_rows.size(); ++output)
    expect(position_bits[output * 1152U] ==
               dif::runtime::float_to_bf16(
                   static_cast<float>(expected_rows[output])),
           "Qwen3-VL learned positions use creator merge-block order");
  std::cout << "GATE qwen3vl_vision blocks=27 weights=350 attentions=27"
               " deepstack=3 multimodal_updates=4\n";
}

void test_generic_scaled_int8_linear() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {2U, 16U}},
      {2U, DType::I8, TensorRole::Internal, {2U, 16U}},
      {3U, DType::F32, TensorRole::Internal, {2U}},
      {4U, DType::I8, TensorRole::Constant, {16U, 16U}},
      {5U, DType::F32, TensorRole::Constant, {16U}},
      {6U, DType::BF16, TensorRole::Output, {2U, 16U}},
  };
  program.operations = {
      {1U, Opcode::QuantizeInt8Rows, {1U}, {2U, 3U},
       {Attribute::u64(AttrKey::BlockSize, 256U),
        Attribute::f64(AttrKey::Scale, 0.98)}},
      {2U, Opcode::LinearInt8Scaled, {2U, 4U, 3U, 5U}, {6U}, {}},
  };
  dif::ir::verify(program);

  dif::runtime::Tensor weight{DType::I8, {16U, 16U}, {}};
  weight.bytes.resize(256U);
  auto *weight_values =
      reinterpret_cast<std::int8_t *>(weight.mutable_data());
  for (std::size_t index = 0U; index < 256U; ++index)
    weight_values[index] = static_cast<std::int8_t>(
        static_cast<int>(index % 13U) - 6);
  weight.validate();
  std::vector<float> input_values(32U);
  for (std::size_t index = 0U; index < input_values.size(); ++index)
    input_values[index] = static_cast<float>(static_cast<int>(index % 11U) - 5) /
                          2.0F;
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U,
                   float_tensor(DType::BF16, {2U, 16U}, input_values));
  bindings.emplace(4U, weight);
  bindings.emplace(5U, f32_tensor({16U}, std::vector<float>(16U, 0.25F)));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto cpu_values = float_values(cpu.outputs.at(6U));
  expect(std::all_of(cpu_values.begin(), cpu_values.end(),
                     [](float value) { return std::isfinite(value); }),
         "generic scaled INT8 Linear CPU output is finite");
  if (!dif::runtime::cuda_available())
    return;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const auto cuda_values = float_values(cuda.outputs.at(6U));
  expect(cuda_values == cpu_values,
         "generic scaled INT8 Linear CUDA matches the integer CPU oracle");
}

void test_generic_int8_weight_linear() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {2U, 16U}},
      {2U, DType::I8, TensorRole::Constant, {16U, 16U}},
      {3U, DType::F32, TensorRole::Constant, {16U}},
      {4U, DType::BF16, TensorRole::Output, {2U, 16U}},
  };
  program.operations = {{1U, Opcode::LinearInt8WeightScaled,
                         {1U, 2U, 3U}, {4U}, {}}};
  dif::ir::verify(program);

  dif::runtime::Tensor weight{DType::I8, {16U, 16U}, {}};
  weight.bytes.resize(256U);
  auto *weight_values =
      reinterpret_cast<std::int8_t *>(weight.mutable_data());
  for (std::size_t index = 0U; index < 256U; ++index)
    weight_values[index] = static_cast<std::int8_t>(
        static_cast<int>(index % 13U) - 6);
  weight.validate();
  std::vector<float> input_values(32U);
  for (std::size_t index = 0U; index < input_values.size(); ++index)
    input_values[index] = static_cast<float>(static_cast<int>(index % 11U) - 5) /
                          2.0F;
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U,
                   float_tensor(DType::BF16, {2U, 16U}, input_values));
  bindings.emplace(2U, weight);
  bindings.emplace(3U, f32_tensor({16U}, std::vector<float>(16U, 0.25F)));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  if (!dif::runtime::cuda_available())
    return;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(4U).bytes == cpu.outputs.at(4U).bytes,
         "generic mixed BF16/INT8 weight Linear CUDA matches its CPU oracle");
}

void test_generic_scaled_fp8_linear() {
  using namespace dif::ir;
  expect(dif::runtime::float_to_fp8_e4m3(0.0F) == 0x00U &&
             dif::runtime::float_to_fp8_e4m3(1.0F) == 0x38U &&
             dif::runtime::float_to_fp8_e4m3(-1.0F) == 0xb8U &&
             dif::runtime::float_to_fp8_e4m3(448.0F) == 0x7eU &&
             dif::runtime::float_to_fp8_e4m3(1.0F / 512.0F) == 0x01U,
         "FP8 E4M3 scalar encoding pins zero, unit, maximum, and subnormal");
  expect(dif::runtime::fp8_e4m3_to_float(0x38U) == 1.0F &&
             dif::runtime::fp8_e4m3_to_float(0xb8U) == -1.0F &&
             dif::runtime::fp8_e4m3_to_float(0x7eU) == 448.0F,
         "FP8 E4M3 scalar decoding pins unit and maximum values");

  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {16U, 16U}},
      {2U, DType::FP8E4M3, TensorRole::Output, {16U, 16U}},
      {3U, DType::F32, TensorRole::Output, {16U}},
      {4U, DType::FP8E4M3, TensorRole::Constant, {16U, 16U}},
      {5U, DType::F32, TensorRole::Constant, {16U}},
      {6U, DType::BF16, TensorRole::Output, {16U, 16U}},
  };
  program.operations = {
      {1U, Opcode::QuantizeFp8Rows, {1U}, {2U, 3U},
       {Attribute::u64(AttrKey::BlockSize, 256U)}},
      {2U, Opcode::LinearFp8Scaled, {2U, 4U, 3U, 5U}, {6U}, {}},
  };
  dif::ir::verify(program);

  std::vector<float> input_values(256U);
  for (std::size_t index = 0U; index < input_values.size(); ++index)
    input_values[index] =
        static_cast<float>(static_cast<int>(index % 29U) - 14) /
        static_cast<float>((index % 5U) + 1U);
  dif::runtime::Tensor weight{DType::FP8E4M3, {16U, 16U}, {}};
  weight.bytes.resize(256U);
  std::vector<float> column_scales(16U);
  for (std::size_t row = 0U; row < 16U; ++row) {
    std::array<float, 16U> values{};
    float maximum = 0.0F;
    for (std::size_t column = 0U; column < 16U; ++column) {
      values[column] =
          static_cast<float>(static_cast<int>((row * 7U + column) % 19U) - 9) /
          static_cast<float>((column % 3U) + 1U);
      maximum = std::max(maximum, std::fabs(values[column]));
    }
    const auto scale = std::max(maximum / 448.0F, 1.0e-30F);
    column_scales[row] = scale;
    for (std::size_t column = 0U; column < 16U; ++column)
      weight.mutable_data()[row * 16U + column] =
          dif::runtime::float_to_fp8_e4m3(values[column] / scale);
  }
  weight.validate();
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U,
                   float_tensor(DType::BF16, {16U, 16U}, input_values));
  bindings.emplace(4U, weight);
  bindings.emplace(5U, f32_tensor({16U}, column_scales));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto cpu_values = float_values(cpu.outputs.at(6U));
  expect(std::all_of(cpu_values.begin(), cpu_values.end(),
                     [](float value) { return std::isfinite(value); }),
         "generic scaled FP8 Linear CPU output is finite");
  if (!dif::runtime::cuda_available())
    return;
  if (!dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda)
           .precision.fp8_tensor_cores) {
    // Target without FP8 tensor cores: the format is illegal there and the
    // CUDA backend must fail closed from the probed facts, not emit PTX the
    // driver rejects.
    bool refused = false;
    std::string message;
    try {
      (void)dif::runtime::make_cuda_executor()->run(program, bindings, options);
    } catch (const std::exception &error) {
      refused = true;
      message = error.what();
    }
    expect(refused && message.find("FP8 physical format") != std::string::npos,
           "generic scaled FP8 Linear fails closed on a target without FP8 tensor cores");
    std::cout << "GATE test_generic_scaled_fp8_linear skipped=fp8_illegal_on_target message=" << message
              << '\n';
    return;
  }
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(2U).bytes == cpu.outputs.at(2U).bytes &&
             cuda.outputs.at(3U).bytes == cpu.outputs.at(3U).bytes,
         "generic FP8 row quantization CUDA matches the CPU contract exactly");
  const auto cuda_values = float_values(cuda.outputs.at(6U));
  float maximum_absolute_error = 0.0F;
  for (std::size_t index = 0U; index < cpu_values.size(); ++index)
    maximum_absolute_error =
        std::max(maximum_absolute_error,
                 std::fabs(cuda_values[index] - cpu_values[index]));
  expect(maximum_absolute_error <= 0.25F,
         "generic scaled FP8 Linear CUDA stays within its BF16 boundary");
}

void test_generic_mxfp8_linear() {
  using namespace dif::ir;
  expect(dif::runtime::float_to_fp8_e8m0_round_up(1.0F) == 127U &&
             dif::runtime::float_to_fp8_e8m0_round_up(0.5F) == 126U &&
             dif::runtime::float_to_fp8_e8m0_round_up(1.1F) == 128U &&
             dif::runtime::fp8_e8m0_to_float(128U) == 2.0F,
         "FP8 E8M0 uses positive-infinity exponent rounding");
  constexpr std::uint64_t rows = 128U;
  constexpr std::uint64_t inner = 128U;
  constexpr std::uint64_t columns = 128U;
  constexpr std::uint64_t scale_blocks = inner / 32U;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, inner}},
      {2U, DType::FP8E4M3, TensorRole::Output, {rows, inner}},
      {3U, DType::FP8E8M0, TensorRole::Output, {rows, scale_blocks}},
      {4U, DType::FP8E4M3, TensorRole::Constant, {columns, inner}},
      {5U, DType::FP8E8M0, TensorRole::Constant,
       {columns, scale_blocks}},
      {6U, DType::BF16, TensorRole::Output, {rows, columns}},
  };
  program.operations = {
      {1U, Opcode::QuantizeFp8Blocks32, {1U}, {2U, 3U},
       {Attribute::u64(AttrKey::BlockSize, 256U)}},
      {2U, Opcode::LinearFp8BlockScaled, {2U, 4U, 3U, 5U}, {6U}, {}},
  };
  dif::ir::verify(program);
  const auto scale_offset = [](std::uint64_t outer, std::uint64_t block) {
    const auto within = (outer % 32U) * 16U +
                        ((outer % 128U) / 32U) * 4U + block % 4U;
    return (block / 4U) * 4U * 128U + within;
  };
  std::vector<float> input_values(rows * inner);
  for (std::size_t index = 0U; index < input_values.size(); ++index)
    input_values[index] =
        static_cast<float>(static_cast<int>(index % 31U) - 15) / 32.0F;
  dif::runtime::Tensor weight{DType::FP8E4M3, {columns, inner}, {}};
  weight.bytes.resize(columns * inner);
  dif::runtime::Tensor scales{DType::FP8E8M0,
                              {columns, scale_blocks}, {}};
  scales.bytes.resize(columns * scale_blocks, 0U);
  for (std::uint64_t row = 0U; row < columns; ++row) {
    for (std::uint64_t block = 0U; block < scale_blocks; ++block) {
      float maximum = 0.0F;
      std::array<float, 32U> values{};
      for (std::uint64_t lane = 0U; lane < 32U; ++lane) {
        values[lane] = static_cast<float>(
            static_cast<int>((row * 13U + block * 5U + lane) % 29U) - 14) /
                       64.0F;
        maximum = std::max(maximum, std::fabs(values[lane]));
      }
      const auto encoded_scale =
          dif::runtime::float_to_fp8_e8m0_round_up(maximum / 448.0F);
      scales.mutable_data()[scale_offset(row, block)] = encoded_scale;
      const auto scale = dif::runtime::fp8_e8m0_to_float(encoded_scale);
      for (std::uint64_t lane = 0U; lane < 32U; ++lane)
        weight.mutable_data()[row * inner + block * 32U + lane] =
            dif::runtime::float_to_fp8_e4m3(values[lane] / scale);
    }
  }
  weight.validate();
  scales.validate();
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U, float_tensor(DType::BF16, {rows, inner}, input_values));
  bindings.emplace(4U, weight);
  bindings.emplace(5U, scales);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  if (!dif::runtime::cuda_available())
    return;
  if (!dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda)
           .precision.fp8_tensor_cores) {
    // Target without FP8 tensor cores: the format is illegal there and the
    // CUDA backend must fail closed from the probed facts, not emit PTX the
    // driver rejects.
    bool refused = false;
    std::string message;
    try {
      (void)dif::runtime::make_cuda_executor()->run(program, bindings, options);
    } catch (const std::exception &error) {
      refused = true;
      message = error.what();
    }
    expect(refused && message.find("FP8 physical format") != std::string::npos,
           "generic MXFP8 Linear fails closed on a target without FP8 tensor cores");
    std::cout << "GATE test_generic_mxfp8_linear skipped=fp8_illegal_on_target message=" << message
              << '\n';
    return;
  }
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(2U).bytes == cpu.outputs.at(2U).bytes &&
             cuda.outputs.at(3U).bytes == cpu.outputs.at(3U).bytes,
         "MXFP8 block quantization CUDA matches the CPU tiled-scale contract");
  const auto cpu_values = float_values(cpu.outputs.at(6U));
  const auto cuda_values = float_values(cuda.outputs.at(6U));
  float maximum_absolute_error = 0.0F;
  for (std::size_t index = 0U; index < cpu_values.size(); ++index)
    maximum_absolute_error =
        std::max(maximum_absolute_error,
                 std::fabs(cuda_values[index] - cpu_values[index]));
  expect(maximum_absolute_error <= 0.25F,
         "MXFP8 cuBLASLt output stays within its BF16 accumulation boundary");
}

void test_generic_int8_block_dequantization() {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1U, DType::I8, TensorRole::Constant, {2U, 64U}},
      {2U, DType::F32, TensorRole::Constant, {2U, 2U}},
      {3U, DType::BF16, TensorRole::Output, {2U, 64U}},
  };
  program.operations = {
      {1U, Opcode::DequantizeInt8Blocks, {1U, 2U}, {3U},
       {Attribute::u64(AttrKey::BlockSize, 32U)}},
  };
  dif::ir::verify(program);
  dif::runtime::Tensor weight{DType::I8, {2U, 64U}, {}};
  weight.bytes.resize(128U);
  auto *values = reinterpret_cast<std::int8_t *>(weight.mutable_data());
  for (std::size_t index = 0U; index < 128U; ++index)
    values[index] = static_cast<std::int8_t>(
        static_cast<int>(index % 31U) - 15);
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U, std::move(weight));
  bindings.emplace(2U, f32_tensor({2U, 2U}, {0.25F, 0.5F, 1.0F, 2.0F}));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  if (!dif::runtime::cuda_available())
    return;
  const auto cuda =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  expect(cuda.outputs.at(3U).bytes == cpu.outputs.at(3U).bytes,
         "generic groupwise INT8 weight dequantization matches exactly");
}

void test_h256_convrot_row_quantization() {
  using namespace dif::ir;
  std::vector<float> values(512U);
  for (std::size_t index = 0U; index < values.size(); ++index)
    values[index] = static_cast<float>(static_cast<int>(index % 37U) - 18) /
                    static_cast<float>((index % 5U) + 1U);
  for (const auto implementation : {Int8RowQuantization::H256ConvRot,
                                    Int8RowQuantization::H256SignedConvRot,
                                    Int8RowQuantization::H256F32ConvRot,
                                    Int8RowQuantization::H256F32SignedConvRot}) {
    const auto dynamic_clip =
        implementation == Int8RowQuantization::H256F32ConvRot ||
        implementation == Int8RowQuantization::H256F32SignedConvRot;
    Program program;
    program.tensors = {
        {1U, DType::BF16, TensorRole::Input, {2U, 256U}},
        {2U, DType::I8, TensorRole::Output, {2U, 256U}},
        {3U, DType::F32, TensorRole::Output, {2U}},
        {4U, DType::I8, TensorRole::Output, {2U, 256U}},
        {5U, DType::F32, TensorRole::Output, {2U}},
    };
    if (dynamic_clip)
      program.tensors.push_back(
          {6U, DType::F32, TensorRole::Input, {1U}});
    program.operations = {{
        1U,
        Opcode::QuantizeInt8Rows,
        dynamic_clip ? std::vector<std::uint32_t>{1U, 6U}
                     : std::vector<std::uint32_t>{1U},
        {2U, 3U, 4U, 5U},
        {Attribute::u64(AttrKey::BlockSize, 256U),
         Attribute::u64(AttrKey::Implementation,
                        static_cast<std::uint64_t>(implementation))},
    }};
    dif::ir::verify(program);
    dif::runtime::TensorMap bindings;
    bindings.emplace(1U, float_tensor(DType::BF16, {2U, 256U}, values));
    if (dynamic_clip)
      bindings.emplace(6U, f32_tensor({1U}, {0.9995F}));
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    const auto cpu =
        dif::runtime::make_cpu_executor()->run(program, bindings, options);
    if (!dif::runtime::cuda_available())
      continue;
    const auto cuda =
        dif::runtime::make_cuda_executor()->run(program, bindings, options);
    expect(cuda.outputs.at(2U).bytes == cpu.outputs.at(2U).bytes,
           "H256 ConvRot primary codes match the CPU contract exactly");
    expect(cuda.outputs.at(3U).bytes == cpu.outputs.at(3U).bytes,
           "H256 ConvRot primary scales match the CPU contract exactly");
    expect(cuda.outputs.at(4U).bytes == cpu.outputs.at(4U).bytes,
           "H256 ConvRot residual codes match the CPU contract exactly");
    expect(cuda.outputs.at(5U).bytes == cpu.outputs.at(5U).bytes,
           "H256 ConvRot residual scales match the CPU contract exactly");
  }
}

void test_h4096_convrot_row_quantization() {
  using namespace dif::ir;
  for (const auto implementation : {
           Int8RowQuantization::H4096SignedConvRot,
           Int8RowQuantization::H4096F32SignedConvRot}) {
    constexpr auto width = 4096U;
    std::vector<float> values(width);
    for (std::size_t index = 0U; index < values.size(); ++index)
      values[index] =
          static_cast<float>(static_cast<int>(index % 53U) - 26) /
          static_cast<float>((index % 7U) + 1U);
    Program program;
    program.tensors = {
        {1U, DType::BF16, TensorRole::Input, {1U, width}},
        {2U, DType::I8, TensorRole::Output, {1U, width}},
        {3U, DType::F32, TensorRole::Output, {1U}},
    };
    program.operations = {{
        1U,
        Opcode::QuantizeInt8Rows,
        {1U},
        {2U, 3U},
        {Attribute::u64(AttrKey::BlockSize, 256U),
         Attribute::u64(AttrKey::Implementation,
                        static_cast<std::uint64_t>(implementation))},
    }};
    dif::ir::verify(program);
    dif::runtime::TensorMap bindings;
    bindings.emplace(1U, float_tensor(DType::BF16, {1U, width}, values));
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    const auto cpu =
        dif::runtime::make_cpu_executor()->run(program, bindings, options);
    if (!dif::runtime::cuda_available())
      continue;
    const auto cuda =
        dif::runtime::make_cuda_executor()->run(program, bindings, options);
    expect(cuda.outputs.at(2U).bytes == cpu.outputs.at(2U).bytes &&
               cuda.outputs.at(3U).bytes == cpu.outputs.at(3U).bytes,
           "H4096 ConvRot row quantization CUDA matches the CPU contract exactly");
  }
}

int main() {
  test_sha256();
  test_json_parser();
  test_codec();
  test_cpu_rms();
  test_cpu_linear_bias();
  test_float_storage_conversions();
  test_cpu_all_opcodes_and_float_dtypes();
  test_krea2_gelu_creator_parity();
  test_krea2_euler_velocity_creator_parity();
  test_krea2_schedule_and_cfg_creator_parity();
  test_krea2_real_dimension_frontend_scaffold();
  test_krea2_rotary_layout_mask_and_broadcast_oracle();
  test_krea2_cudnn_masked_gqa_creator_oracle();
  test_new_primitives_cuda_parity();
  test_vae_normalization_primitives();
  test_generic_image_vae_primitives();
  test_group_norm_and_reflect_padding_primitives();
  test_krea2_qwen_image_vae_frontend_contract();
  test_h3_video_vae_frontend_contract();
  test_h3_video_encoder_frontend_contract();
  test_training_autodiff_optimizer_and_checkpoint();
  test_mixed_precision_bf16_training_step();
  test_rectified_flow_training_vertical();
  test_backend_neutral_diffusion_preprocessing();
  test_backend_neutral_flow_scheduler();
  test_h3_conditioning_layout();
  test_h3_creator_noise_and_packing();
  test_h3_latent_handoff();
  test_h3_media_handoff();
  test_prepared_execution_reuses_constants();
  test_cuda_repeated_invariant_execution_cache();
  test_verifier_rejects_multiple_writers();
  test_audio_opcode_verifier_contract();
  test_audio_bigvgan_frontend_contract();
  test_wav_pcm16_writer_contract();
  test_png_rgb8_writer_contract();
  test_qwen3vl_vision_and_multimodal_frontend_contract();
  test_generic_scaled_int8_linear();
  test_generic_int8_weight_linear();
  test_generic_scaled_fp8_linear();
  test_generic_mxfp8_linear();
  test_generic_int8_block_dequantization();
  test_h256_convrot_row_quantization();
  test_h4096_convrot_row_quantization();
  test_attention_implementation_identity();
  test_h3_bf16_lowering_preserves_source_reduction_identity();
  test_h3_long_sequence_transformer_declares_backend_attention();
  test_h3_mixed_denoiser_frontend_and_cuda_parity();
  test_h3_token_refiner_frontend_and_cuda_parity();
  test_memory_plan_reuses_dead_internal_storage();
  test_memory_plan_pages_streamed_constants();
  test_memory_plan_reserves_prefetch_storage();
  test_memory_plan_omits_backend_replaced_constants();
  test_compiler_streamed_residency_plan();
  test_cuda_lazy_resident_upload();
  test_cuda_f16_biased_convrot_int8();
  test_compiler_and_cuda_reshape_alias_plan();
  test_weight_bundle_roundtrip();
  test_safetensors_streaming_writer();
  test_safetensors_tensor_metadata_is_skipped();
  test_int4_weight_rewrite_and_cpu_execution();
  test_int5_weight_rewrite_and_cpu_execution();
  test_operation_slice_preserves_stable_ids_and_boundary_roles();
  test_cuda_linear_tuning_and_exact_swiglu_fusion();
  test_cuda_cutlass_linear_primitive();
  if (failures) {
    std::cerr << failures << " test assertion(s) failed\n";
    return 1;
  }
  std::cout << "PASS: DiffIR codec, hash, verifier, and typed backend semantics\n";
  return 0;
}
