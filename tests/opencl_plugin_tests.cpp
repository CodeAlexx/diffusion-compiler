#include "dif/backend/plugin.hpp"
#include "dif/compiler/int4.hpp"
#include "dif/frontend/training.hpp"
#include "dif/ir/ir.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

dif::runtime::Tensor tensor(const std::vector<float> &values) {
  dif::runtime::Tensor result{dif::ir::DType::F32,
                              {1U, static_cast<std::uint64_t>(values.size())},
                              {}};
  result.bytes.resize(values.size() * sizeof(float));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  return result;
}

dif::runtime::Tensor i32_tensor(const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor result{dif::ir::DType::I32,
                              {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  return result;
}

} // namespace

int main() {
  try {
    using namespace dif::ir;
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {1, 4}},
        {2, DType::F32, TensorRole::Constant, {1, 4}},
        {3, DType::F32, TensorRole::Output, {1, 4}},
    };
    program.operations = {{1, Opcode::Add, {1, 2}, {3}, {}}};
    dif::runtime::TensorMap bindings;
    bindings.emplace(1, tensor({1.0F, 2.0F, 3.0F, 4.0F}));
    bindings.emplace(2, tensor({0.5F, 1.0F, 1.5F, 2.0F}));
    auto executor =
        dif::backend::make_plugin_executor(DIF_OPENCL_BACKEND_PATH);
    dif::runtime::RunOptions options;
    options.warmups = 0U;
    options.iterations = 1U;
    options.minimum_free_bytes = 0U;
    const auto result = executor->run(program, bindings, options);
    const auto output = result.outputs.at(3).f32();
    const std::vector<float> expected = {1.5F, 3.0F, 4.5F, 6.0F};
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (std::abs(output[index] - expected[index]) > 1.0e-7F) {
        std::cerr << "OpenCL plugin output mismatch\n";
        return 1;
      }
    }
    if (result.backend_name != "opencl-reference-v2" ||
        result.device_name.empty() || result.resident_bytes == 0U) {
      std::cerr << "OpenCL plugin telemetry mismatch\n";
      return 1;
    }

    Program lowbit_program;
    lowbit_program.tensors = {
        {1, DType::I8, TensorRole::Constant, {1, 20}},
        {2, DType::BF16, TensorRole::Constant, {1, 1}},
        {3, DType::BF16, TensorRole::Output, {1, 32}},
    };
    lowbit_program.operations = {
        {1, Opcode::DequantizeInt5, {1, 2}, {3},
         {Attribute::u64(AttrKey::GroupSize, 32U)}},
    };
    std::vector<float> weight_values;
    for (int value = -15; value <= 15; ++value)
      weight_values.push_back(static_cast<float>(value) * 0.25F);
    weight_values.push_back(0.0F);
    dif::runtime::Tensor weight{DType::BF16, {1, 32}, {}};
    weight.bytes.resize(32U * sizeof(std::uint16_t));
    for (std::size_t index = 0; index < weight_values.size(); ++index)
      dif::runtime::store_float(weight, index, weight_values[index]);
    const auto quantized =
        dif::compiler::quantize_lowbit_weight(weight, 5U, 32U);
    dif::runtime::TensorMap lowbit_bindings;
    lowbit_bindings.emplace(1, quantized.packed);
    lowbit_bindings.emplace(2, quantized.scales);
    const auto lowbit_result =
        executor->run(lowbit_program, lowbit_bindings, options);
    const auto &decoded = lowbit_result.outputs.at(3);
    for (std::size_t index = 0; index < weight_values.size(); ++index) {
      if (dif::runtime::load_float(decoded, index) != weight_values[index]) {
        std::cerr << "OpenCL INT5/BF16 output mismatch\n";
        return 1;
      }
    }

    Program primitive_program;
    primitive_program.tensors = {
        {1, DType::F32, TensorRole::Input, {3, 2}},
        {2, DType::F32, TensorRole::Constant, {2}},
        {3, DType::I32, TensorRole::Input, {2}},
        {4, DType::F32, TensorRole::Internal, {2, 2}},
        {5, DType::F32, TensorRole::Internal, {2, 2}},
        {6, DType::F32, TensorRole::Internal, {2, 2}},
        {7, DType::F32, TensorRole::Internal, {3, 2}},
        {8, DType::I32, TensorRole::Input, {3}},
        {9, DType::F32, TensorRole::Output, {3, 2}},
        {10, DType::BF16, TensorRole::Internal, {3, 2}},
        {11, DType::BF16, TensorRole::Internal, {3, 2}},
        {12, DType::F32, TensorRole::Output, {3, 2}},
        {13, DType::F32, TensorRole::Input, {2, 4}},
        {14, DType::I32, TensorRole::Input, {3}},
        {15, DType::F32, TensorRole::Output, {3, 2}},
        {16, DType::F32, TensorRole::Output, {3, 2}},
        {17, DType::F32, TensorRole::Constant, {2}},
        {18, DType::F32, TensorRole::Constant, {2}},
        {19, DType::F32, TensorRole::Output, {3, 2}},
        {20, DType::F32, TensorRole::Constant, {2}},
        {21, DType::F32, TensorRole::Constant, {2}},
        {22, DType::F32, TensorRole::Output, {3, 2}},
    };
    primitive_program.operations = {
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
        {10, Opcode::AffineLastDim, {9, 17, 18}, {19}, {}},
        {11, Opcode::LayerNorm, {19, 20, 21}, {22},
         {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
          Attribute::u64(AttrKey::BlockSize, 32U)}},
    };
    dif::runtime::TensorMap primitive_bindings;
    primitive_bindings.emplace(1, tensor({1, 2, 3, 4, 5, 6}));
    primitive_bindings.at(1).dims = {3, 2};
    primitive_bindings.emplace(2, tensor({1, 2}));
    primitive_bindings.at(2).dims = {2};
    primitive_bindings.emplace(3, i32_tensor({2, 0}));
    primitive_bindings.emplace(8, i32_tensor({-1, 1, 0}));
    primitive_bindings.emplace(13, tensor({1, 2, 3, 4, 5, 6, 7, 8}));
    primitive_bindings.at(13).dims = {2, 4};
    primitive_bindings.emplace(14, i32_tensor({1, 0, 1}));
    primitive_bindings.emplace(17, tensor({2.0F, 0.5F}));
    primitive_bindings.at(17).dims = {2};
    primitive_bindings.emplace(18, tensor({0.25F, -0.5F}));
    primitive_bindings.at(18).dims = {2};
    primitive_bindings.emplace(20, tensor({1.0F, 1.0F}));
    primitive_bindings.at(20).dims = {2};
    primitive_bindings.emplace(21, tensor({0.0F, 0.0F}));
    primitive_bindings.at(21).dims = {2};
    const auto primitive_result =
        executor->run(primitive_program, primitive_bindings, options);
    const auto primitive_output = primitive_result.outputs.at(9).f32();
    const auto inv_a = 1.0F / std::sqrt(30.5F + 1.0e-5F);
    const auto inv_b = 1.0F / std::sqrt(2.5F + 1.0e-5F);
    const std::vector<float> normalized = {
        5.0F * inv_a, 12.0F * inv_a, 1.0F * inv_b, 4.0F * inv_b};
    const std::vector<float> primitive_expected = {
        0.5F,
        0.5F,
        normalized[2] / (1.0F + std::exp(-normalized[2])),
        normalized[3] / (1.0F + std::exp(-normalized[3])),
        normalized[0] / (1.0F + std::exp(-normalized[0])),
        normalized[1] / (1.0F + std::exp(-normalized[1])),
    };
    for (std::size_t index = 0; index < primitive_expected.size(); ++index) {
      if (std::abs(primitive_output[index] - primitive_expected[index]) >
          2.0e-5F) {
        std::cerr << "OpenCL new primitive output mismatch\n";
        return 1;
      }
    }
    const std::vector<float> selected_first = {5, 6, 1, 2, 5, 6};
    const std::vector<float> selected_second = {7, 8, 3, 4, 7, 8};
    const auto first_output = primitive_result.outputs.at(15).f32();
    const auto second_output = primitive_result.outputs.at(16).f32();
    for (std::size_t index = 0; index < selected_first.size(); ++index) {
      if (first_output[index] != selected_first[index] ||
          second_output[index] != selected_second[index]) {
        std::cerr << "OpenCL SelectRowChunks output mismatch\n";
        return 1;
      }
    }
    const auto mixed_output = primitive_result.outputs.at(12).f32();
    for (std::size_t index = 0; index < primitive_expected.size(); ++index) {
      const auto expected_value = primitive_expected[index] /
                                  (1.0F + std::exp(-primitive_expected[index]));
      if (std::abs(mixed_output[index] - expected_value) > 1.0e-2F) {
        std::cerr << "OpenCL mixed f32/bf16 Cast output mismatch\n";
        return 1;
      }
    }
    const auto affine_output = primitive_result.outputs.at(19).f32();
    for (std::size_t row = 0; row < 3U; ++row) {
      if (std::abs(affine_output[row * 2U] -
                       (primitive_expected[row * 2U] * 2.0F + 0.25F)) >
              2.0e-5F ||
          std::abs(affine_output[row * 2U + 1U] -
                       (primitive_expected[row * 2U + 1U] * 0.5F - 0.5F)) >
              2.0e-5F) {
        std::cerr << "OpenCL affine-last-dimension output mismatch\n";
        return 1;
      }
    }
    const auto layer_norm_output = primitive_result.outputs.at(22).f32();
    for (std::size_t row = 0; row < 3U; ++row) {
      if (std::abs(layer_norm_output[row * 2U] +
                   layer_norm_output[row * 2U + 1U]) > 2.0e-5F) {
        std::cerr << "OpenCL layer-norm output mismatch\n";
        return 1;
      }
    }

    Program preprocessing_program;
    preprocessing_program.tensors = {
        {1, DType::F32, TensorRole::Input, {2}},
        {2, DType::F32, TensorRole::Output, {2, 4}},
        {3, DType::F32, TensorRole::Input, {2, 3}},
        {4, DType::F32, TensorRole::Constant, {2}},
        {5, DType::BF16, TensorRole::Output, {2, 12}},
        {6, DType::BF16, TensorRole::Output, {2, 12}},
    };
    preprocessing_program.operations = {
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
    dif::runtime::TensorMap preprocessing_bindings;
    preprocessing_bindings.emplace(1, tensor({0.0F, 1.0F}));
    preprocessing_bindings.at(1).dims = {2};
    preprocessing_bindings.emplace(
        3, tensor({0.0F, 1.0F, 2.0F, 3.0F, 4.0F, 5.0F}));
    preprocessing_bindings.at(3).dims = {2, 3};
    preprocessing_bindings.emplace(4, tensor({1.0F, 0.1F}));
    preprocessing_bindings.at(4).dims = {2};
    const auto preprocessing_result =
        executor->run(preprocessing_program, preprocessing_bindings, options);
    const auto timestep_output = preprocessing_result.outputs.at(2).f32();
    if (timestep_output[0] != 1.0F || timestep_output[1] != 1.0F ||
        timestep_output[2] != 0.0F || timestep_output[3] != 0.0F ||
        std::abs(timestep_output[4] - std::cos(1.0F)) > 2.0e-6F ||
        std::abs(timestep_output[6] - std::sin(1.0F)) > 2.0e-6F) {
      std::cerr << "OpenCL sinusoidal timestep mismatch\n";
      return 1;
    }
    const auto &rotary_cos = preprocessing_result.outputs.at(5);
    const auto &rotary_sin = preprocessing_result.outputs.at(6);
    if (dif::runtime::load_float(rotary_cos, 0) != 1.0F ||
        dif::runtime::load_float(rotary_sin, 0) != 0.0F ||
        dif::runtime::load_float(rotary_cos, 2) !=
            dif::runtime::load_float(rotary_cos, 8) ||
        dif::runtime::load_float(rotary_sin, 2) !=
            dif::runtime::load_float(rotary_sin, 8)) {
      std::cerr << "OpenCL rotary position mismatch\n";
      return 1;
    }

    Program flow_program;
    flow_program.tensors = {
        {1, DType::F32, TensorRole::Input, {1, 2}},
        {2, DType::F32, TensorRole::Input, {1, 2}},
        {3, DType::F32, TensorRole::Input, {1}},
        {4, DType::F32, TensorRole::Output, {1, 2}},
        {5, DType::F32, TensorRole::Input, {1, 2}},
        {6, DType::F32, TensorRole::Input, {1}},
        {7, DType::F32, TensorRole::Input, {2}},
        {8, DType::F32, TensorRole::Output, {1, 2}},
    };
    flow_program.operations = {
        {1, Opcode::LinearBlend, {1, 2, 3}, {4}, {}},
        {2, Opcode::FlowEulerStep, {4, 5, 6, 7}, {8},
         {Attribute::u64(AttrKey::StepIndex, 0U)}},
    };
    dif::runtime::TensorMap flow_bindings;
    flow_bindings.emplace(1, tensor({2.0F, 6.0F}));
    flow_bindings.emplace(2, tensor({10.0F, 14.0F}));
    flow_bindings.emplace(3, tensor({0.25F}));
    flow_bindings.at(3).dims = {1};
    flow_bindings.emplace(5, tensor({0.5F, -1.0F}));
    flow_bindings.emplace(6, tensor({0.25F}));
    flow_bindings.at(6).dims = {1};
    flow_bindings.emplace(7, tensor({0.75F, 0.5F}));
    flow_bindings.at(7).dims = {2};
    const auto flow_result =
        executor->run(flow_program, flow_bindings, options);
    const auto flow_output = flow_result.outputs.at(8).f32();
    if (std::abs(flow_output[0] - 8.125F) > 1.0e-6F ||
        std::abs(flow_output[1] - 11.75F) > 1.0e-6F) {
      std::cerr << "OpenCL flow scheduler primitive mismatch\n";
      return 1;
    }

    Program patch_program;
    patch_program.tensors = {
        {1, DType::F32, TensorRole::Input, {1, 2, 2, 2, 2}},
        {2, DType::F32, TensorRole::Output, {2, 8}},
        {3, DType::F32, TensorRole::Output, {1, 2, 2, 2, 2}},
    };
    const auto patch_attributes = std::vector<Attribute>{
        Attribute::u64(AttrKey::PatchT, 1U),
        Attribute::u64(AttrKey::PatchH, 2U),
        Attribute::u64(AttrKey::PatchW, 2U)};
    patch_program.operations = {
        {1, Opcode::Patchify3D, {1}, {2}, patch_attributes},
        {2, Opcode::Unpatchify3D, {2}, {3}, patch_attributes},
    };
    auto patch_input = tensor(
        {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    patch_input.dims = {1, 2, 2, 2, 2};
    const auto patch_result =
        executor->run(patch_program, {{1, patch_input}}, options);
    const auto patch_rows = patch_result.outputs.at(2).f32();
    const auto patch_roundtrip = patch_result.outputs.at(3).f32();
    const std::vector<float> expected_rows = {
        0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15};
    if (!std::equal(patch_rows.begin(), patch_rows.end(),
                    expected_rows.begin()) ||
        !std::equal(patch_roundtrip.begin(), patch_roundtrip.end(),
                    patch_input.f32().begin())) {
      std::cerr << "OpenCL patchify/unpatchify primitive mismatch\n";
      return 1;
    }

    dif::frontend::MlpTrainingConfig training_config;
    training_config.rows = 2U;
    training_config.input_width = 2U;
    training_config.hidden_width = 3U;
    training_config.output_width = 1U;
    const auto training =
        dif::frontend::make_mlp_training(training_config);
    dif::runtime::TensorMap training_bindings;
    auto features = tensor({-1.0F, 0.5F, 0.25F, 1.0F});
    features.dims = {2U, 2U};
    auto targets = tensor({0.5F, -0.25F});
    targets.dims = {2U, 1U};
    training_bindings.emplace(training.features_input, std::move(features));
    training_bindings.emplace(training.target_input, std::move(targets));
    const std::array<std::vector<float>, 4> parameter_values = {
        std::vector<float>{-0.2F, -0.1F, 0.0F, 0.1F, 0.2F, 0.3F},
        std::vector<float>{-0.05F, 0.0F, 0.05F},
        std::vector<float>{-0.1F, 0.0F, 0.1F},
        std::vector<float>{0.02F},
    };
    for (std::size_t index = 0U; index < training.optimizer_bindings.size();
         ++index) {
      const auto &binding = training.optimizer_bindings[index];
      auto parameter = tensor(parameter_values[index]);
      parameter.dims = training.program.tensor(binding.parameter_input)->dims;
      training_bindings.emplace(binding.parameter_input, std::move(parameter));
      training_bindings.emplace(
          binding.first_moment_input,
          dif::runtime::zeros(
              *training.program.tensor(binding.first_moment_input)));
      training_bindings.emplace(
          binding.second_moment_input,
          dif::runtime::zeros(
              *training.program.tensor(binding.second_moment_input)));
    }
    training_bindings.emplace(training.step_input, i32_tensor({0}));
    const auto training_reference = dif::runtime::make_cpu_executor()->run(
        training.program, training_bindings, options);
    const auto training_candidate =
        executor->run(training.program, training_bindings, options);
    float training_maximum_absolute_error = 0.0F;
    for (const auto &[id, expected_tensor] : training_reference.outputs) {
      const auto expected = expected_tensor.f32();
      const auto actual = training_candidate.outputs.at(id).f32();
      for (std::size_t index = 0U; index < expected.size(); ++index)
        training_maximum_absolute_error =
            std::max(training_maximum_absolute_error,
                     std::abs(expected[index] - actual[index]));
    }
    if (training_maximum_absolute_error > 2.0e-5F) {
      std::cerr << "OpenCL training/autodiff output mismatch\n";
      return 1;
    }
    std::cout << "GATE OpenCL training_one_step device="
              << training_candidate.device_name
              << " max_abs=" << training_maximum_absolute_error << "\n";
    std::cout << "PASS: real OpenCL device through backend ABI v2\n";
    return 0;
  } catch (const std::exception &exception) {
    const std::string message = exception.what();
    if (message.find("unavailable") != std::string::npos ||
        message.find("status 2") != std::string::npos) {
      std::cout << "SKIP: no OpenCL GPU device\n";
      return 77;
    }
    std::cerr << "OpenCL plugin test failed: " << message << "\n";
    return 1;
  }
}
