// DiT backward opcode tests: finite-difference gradient checks against the
// CPU reference forward, CPU-vs-CUDA cross-checks, and verifier fail-closed
// negatives.  The finite-difference harness perturbs every element of every
// differentiation target on the FORWARD program and compares the central
// difference of the F32 loss against the autodiff gradient (flame lesson:
// measurement beats assertion; the fixture-based torch gates are the
// reference authority, these are the always-on structural gates).

#include "dif/compiler/compiler.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/training/autodiff.hpp"

#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

// Deterministic SplitMix64-derived filler in roughly [-1, 1].
float mixed_unit(std::uint64_t &state) {
  state += 0x9e3779b97f4a7c15ULL;
  auto z = state;
  z = (z ^ (z >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27U)) * 0x94d049bb133111ebULL;
  z ^= z >> 31U;
  return static_cast<float>(static_cast<double>(z >> 11U) /
                            static_cast<double>(1ULL << 53U)) *
             2.0F -
         1.0F;
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

dif::runtime::Tensor random_tensor(dif::ir::DType dtype,
                                   std::vector<std::uint64_t> dims,
                                   std::uint64_t seed, float amplitude = 1.0F,
                                   float offset = 0.0F) {
  std::uint64_t count = 1U;
  for (const auto dim : dims)
    count *= dim;
  std::vector<float> values(count);
  auto state = seed;
  for (auto &value : values)
    value = mixed_unit(state) * amplitude + offset;
  return float_tensor(dtype, std::move(dims), values);
}

std::vector<float> float_values(const dif::runtime::Tensor &tensor) {
  std::vector<float> values(tensor.element_count());
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = dif::runtime::load_float(tensor, i);
  return values;
}

dif::runtime::RunOptions single_run_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return options;
}

// A forward graph ending in an F32[1] loss plus the bindings to run it.
struct GradientCase {
  std::string name;
  dif::ir::Program forward;
  std::uint32_t loss{};
  std::vector<std::uint32_t> targets;
  dif::runtime::TensorMap bindings;
};

float run_loss(const GradientCase &grad_case,
               const dif::runtime::TensorMap &bindings) {
  const auto result = dif::runtime::make_cpu_executor()->run(
      grad_case.forward, bindings, single_run_options());
  return result.outputs.at(grad_case.loss).f32()[0];
}

// Central finite differences on the CPU forward against the autodiff
// gradients on the CPU backward.  F32 forward at these tiny dims puts the
// FD noise floor orders of magnitude under the admitted bars.
void finite_difference_check(const GradientCase &grad_case,
                             double minimum_cosine = 0.9995,
                             double maximum_relative_l2 = 0.02) {
  std::cout << "CASE fd " << grad_case.name << "\n" << std::flush;
  const auto differentiated = dif::training::differentiate(
      grad_case.forward, grad_case.loss, grad_case.targets);
  const auto gradients = dif::runtime::make_cpu_executor()->run(
      differentiated.program, grad_case.bindings, single_run_options());

  for (const auto target : grad_case.targets) {
    const auto analytic = float_values(
        gradients.outputs.at(differentiated.gradients.at(target)));
    const auto &bound = grad_case.bindings.at(target);
    std::vector<float> numeric(analytic.size());
    for (std::size_t index = 0; index < numeric.size(); ++index) {
      const auto original = dif::runtime::load_float(bound, index);
      const auto step =
          1.0e-3F * std::max(1.0F, std::abs(original));
      auto bindings = grad_case.bindings;
      auto perturbed = bound;
      dif::runtime::store_float(perturbed, index, original + step);
      bindings.insert_or_assign(target, perturbed);
      const auto upper = run_loss(grad_case, bindings);
      dif::runtime::store_float(perturbed, index, original - step);
      bindings.insert_or_assign(target, perturbed);
      const auto lower = run_loss(grad_case, bindings);
      numeric[index] = (upper - lower) / (2.0F * step);
    }
    double dot = 0.0;
    double analytic_norm = 0.0;
    double numeric_norm = 0.0;
    double difference_norm = 0.0;
    for (std::size_t index = 0; index < numeric.size(); ++index) {
      dot += static_cast<double>(analytic[index]) * numeric[index];
      analytic_norm +=
          static_cast<double>(analytic[index]) * analytic[index];
      numeric_norm += static_cast<double>(numeric[index]) * numeric[index];
      const auto difference =
          static_cast<double>(analytic[index]) - numeric[index];
      difference_norm += difference * difference;
    }
    const auto cosine =
        dot / std::max(1.0e-30, std::sqrt(analytic_norm * numeric_norm));
    const auto relative_l2 = std::sqrt(difference_norm) /
                             std::max(1.0e-30, std::sqrt(numeric_norm));
    expect(numeric_norm > 0.0,
           grad_case.name + " target " + std::to_string(target) +
               " finite-difference gradient is nonzero");
    expect(cosine >= minimum_cosine,
           grad_case.name + " target " + std::to_string(target) +
               " gradient cosine vs finite differences (" +
               std::to_string(cosine) + ")");
    expect(relative_l2 <= maximum_relative_l2,
           grad_case.name + " target " + std::to_string(target) +
               " gradient rel-L2 vs finite differences (" +
               std::to_string(relative_l2) + ")");
    std::cout << "FD " << grad_case.name << " target=" << target
              << " cos=" << cosine << " rel_l2=" << relative_l2 << "\n";
  }
}

// CPU-vs-CUDA cross-check on the differentiated program.  Bars were set
// AFTER measurement (2026-08-31 run log): F32 worst max_abs measured
// 8.9e-8, admitted 1e-5; BF16 worst measured 3.9e-3 (one BF16 ulp at
// gradient magnitude), admitted 1.6e-2.
void backend_cross_check(const GradientCase &grad_case, float bar) {
  if (!dif::runtime::cuda_available())
    return;
  std::cout << "CASE cross " << grad_case.name << "\n" << std::flush;
  const auto differentiated = dif::training::differentiate(
      grad_case.forward, grad_case.loss, grad_case.targets);
  const auto reference = dif::runtime::make_cpu_executor()->run(
      differentiated.program, grad_case.bindings, single_run_options());
  const auto candidate = dif::runtime::make_cuda_executor()->run(
      differentiated.program, grad_case.bindings, single_run_options());
  float maximum_absolute_error = 0.0F;
  for (const auto target : grad_case.targets) {
    const auto gradient_id = differentiated.gradients.at(target);
    const auto expected = float_values(reference.outputs.at(gradient_id));
    const auto actual = float_values(candidate.outputs.at(gradient_id));
    expect(expected.size() == actual.size(),
           grad_case.name + " CUDA gradient size parity");
    for (std::size_t index = 0; index < expected.size(); ++index)
      maximum_absolute_error =
          std::max(maximum_absolute_error,
                   std::abs(expected[index] - actual[index]));
  }
  expect(maximum_absolute_error <= bar,
         grad_case.name + " CPU-vs-CUDA gradient parity (max_abs=" +
             std::to_string(maximum_absolute_error) + ")");
  std::cout << "GATE dit_backward_cross " << grad_case.name
            << " backend=" << candidate.backend_name
            << " max_abs=" << maximum_absolute_error << "\n";
}

// ---- graph builders -------------------------------------------------------

// Shared suffix: mse_loss(y, target) so the upstream gradient into the op
// under test is non-uniform.
void append_loss(GradientCase &grad_case, std::uint32_t prediction,
                 std::uint64_t seed) {
  auto &program = grad_case.forward;
  // Copy by value: push_back below reallocates program.tensors, which would
  // dangle a held TensorDesc pointer.
  const auto prediction_dims = program.tensor(prediction)->dims;
  const auto prediction_dtype = program.tensor(prediction)->dtype;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);
  const auto target = next_tensor++;
  const auto loss = next_tensor++;
  if (prediction_dtype == dif::ir::DType::F32) {
    program.tensors.push_back(
        {target, dif::ir::DType::F32, dif::ir::TensorRole::Input,
         prediction_dims});
    program.tensors.push_back({loss, dif::ir::DType::F32,
                               dif::ir::TensorRole::Output, {1U}});
    program.operations.push_back(
        {next_operation++, dif::ir::Opcode::MseLoss, {prediction, target},
         {loss}, {}});
  } else {
    // BF16 prediction crosses a Cast boundary into the F32 loss (the
    // mixed-precision graph shape proven by the Wave-1 gate).
    const auto cast = next_tensor++;
    program.tensors.push_back(
        {target, dif::ir::DType::F32, dif::ir::TensorRole::Input,
         prediction_dims});
    program.tensors.push_back({loss, dif::ir::DType::F32,
                               dif::ir::TensorRole::Output, {1U}});
    program.tensors.push_back({cast, dif::ir::DType::F32,
                               dif::ir::TensorRole::Internal,
                               prediction_dims});
    program.operations.push_back(
        {next_operation++, dif::ir::Opcode::Cast, {prediction}, {cast}, {}});
    program.operations.push_back(
        {next_operation++, dif::ir::Opcode::MseLoss, {cast, target}, {loss},
         {}});
  }
  grad_case.loss = loss;
  grad_case.bindings.emplace(
      target, random_tensor(dif::ir::DType::F32, prediction_dims, seed));
}

GradientCase rms_norm_case(dif::ir::DType dtype) {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name = std::string("rms_norm_") + std::string(dtype_name(dtype));
  const std::uint64_t rows = 5U;
  const std::uint64_t cols = 7U;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, cols}},
      {2U, dtype, TensorRole::Input, {cols}},
      {3U, dtype, TensorRole::Internal, {rows, cols}},
  };
  grad_case.forward.operations = {
      {1U, Opcode::RmsNorm, {1U, 2U}, {3U},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32U)}},
  };
  grad_case.targets = {1U, 2U};
  grad_case.bindings.emplace(1U, random_tensor(dtype, {rows, cols}, 11U));
  grad_case.bindings.emplace(
      2U, random_tensor(dtype, {cols}, 12U, 0.5F, 1.0F));
  append_loss(grad_case, 3U, 13U);
  return grad_case;
}

GradientCase rms_norm_modulate_case(dif::ir::DType dtype, bool weighted) {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name = std::string("rms_norm_modulate_") +
                   (weighted ? "weighted_" : "plain_") +
                   std::string(dtype_name(dtype));
  const std::uint64_t rows = 4U;
  const std::uint64_t cols = 6U;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, cols}},
      {2U, dtype, TensorRole::Input, {rows, cols}},
      {3U, dtype, TensorRole::Input, {rows, cols}},
      {4U, dtype, TensorRole::Internal, {rows, cols}},
  };
  std::vector<std::uint32_t> inputs{1U};
  if (weighted) {
    grad_case.forward.tensors.push_back(
        {5U, dtype, TensorRole::Input, {cols}});
    inputs.push_back(5U);
  }
  inputs.push_back(2U);
  inputs.push_back(3U);
  grad_case.forward.operations = {
      {1U, Opcode::RmsNormModulate, inputs, {4U},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32U)}},
  };
  grad_case.targets = {1U, 2U, 3U};
  if (weighted)
    grad_case.targets.push_back(5U);
  grad_case.bindings.emplace(1U, random_tensor(dtype, {rows, cols}, 21U));
  grad_case.bindings.emplace(
      2U, random_tensor(dtype, {rows, cols}, 22U, 0.3F));
  grad_case.bindings.emplace(
      3U, random_tensor(dtype, {rows, cols}, 23U, 0.3F));
  if (weighted)
    grad_case.bindings.emplace(
        5U, random_tensor(dtype, {cols}, 24U, 0.5F, 1.0F));
  append_loss(grad_case, 4U, 25U);
  return grad_case;
}

GradientCase swiglu_case(dif::ir::DType dtype, bool gate_first) {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name = std::string("swiglu_") +
                   (gate_first ? "gatefirst_" : "valuefirst_") +
                   std::string(dtype_name(dtype));
  const std::uint64_t rows = 5U;
  const std::uint64_t width = 4U;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, width * 2U}},
      {2U, dtype, TensorRole::Internal, {rows, width}},
  };
  grad_case.forward.operations = {
      {1U, Opcode::SwiGlu, {1U}, {2U},
       {Attribute::boolean(AttrKey::GateFirst, gate_first)}},
  };
  grad_case.targets = {1U};
  grad_case.bindings.emplace(
      1U, random_tensor(dtype, {rows, width * 2U}, 31U));
  append_loss(grad_case, 2U, 32U);
  return grad_case;
}

GradientCase layer_norm_case(dif::ir::DType dtype) {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name =
      std::string("layer_norm_") + std::string(dtype_name(dtype));
  const std::uint64_t rows = 5U;
  const std::uint64_t cols = 6U;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, cols}},
      {2U, dtype, TensorRole::Input, {cols}},
      {3U, dtype, TensorRole::Input, {cols}},
      {4U, dtype, TensorRole::Internal, {rows, cols}},
  };
  grad_case.forward.operations = {
      {1U, Opcode::LayerNorm, {1U, 2U, 3U}, {4U},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32U)}},
  };
  grad_case.targets = {1U, 2U, 3U};
  grad_case.bindings.emplace(1U, random_tensor(dtype, {rows, cols}, 61U));
  grad_case.bindings.emplace(
      2U, random_tensor(dtype, {cols}, 62U, 0.5F, 1.0F));
  grad_case.bindings.emplace(3U, random_tensor(dtype, {cols}, 63U, 0.3F));
  append_loss(grad_case, 4U, 64U);
  return grad_case;
}

GradientCase residual_gate_case(dif::ir::DType dtype) {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name =
      std::string("residual_gate_") + std::string(dtype_name(dtype));
  const std::uint64_t rows = 4U;
  const std::uint64_t cols = 5U;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, cols}},
      {2U, dtype, TensorRole::Input, {rows, cols}},
      {3U, dtype, TensorRole::Input, {rows, cols}},
      {4U, dtype, TensorRole::Internal, {rows, cols}},
  };
  grad_case.forward.operations = {
      {1U, Opcode::ResidualGate, {1U, 2U, 3U}, {4U}, {}},
  };
  grad_case.targets = {1U, 2U, 3U};
  grad_case.bindings.emplace(1U, random_tensor(dtype, {rows, cols}, 41U));
  grad_case.bindings.emplace(2U, random_tensor(dtype, {rows, cols}, 42U));
  grad_case.bindings.emplace(3U, random_tensor(dtype, {rows, cols}, 43U));
  append_loss(grad_case, 4U, 44U);
  return grad_case;
}

// A small composed chain: RmsNormModulate -> SwiGlu-shaped MLP piece ->
// ResidualGate, probing rule composition through shared tensors.
GradientCase composed_group1_case() {
  using namespace dif::ir;
  GradientCase grad_case;
  grad_case.name = "composed_group1_f32";
  const std::uint64_t rows = 4U;
  const std::uint64_t cols = 8U;
  const auto dtype = DType::F32;
  grad_case.forward.tensors = {
      {1U, dtype, TensorRole::Input, {rows, cols}},          // x
      {2U, dtype, TensorRole::Input, {cols}},                // norm weight
      {3U, dtype, TensorRole::Input, {rows, cols}},          // scale
      {4U, dtype, TensorRole::Input, {rows, cols}},          // shift
      {5U, dtype, TensorRole::Internal, {rows, cols}},       // modulated
      {6U, dtype, TensorRole::Internal, {rows, cols / 2U}},  // swiglu out
      {7U, dtype, TensorRole::Input, {rows, cols / 2U}},     // residual
      {8U, dtype, TensorRole::Input, {rows, cols / 2U}},     // gate
      {9U, dtype, TensorRole::Internal, {rows, cols / 2U}},  // gated out
  };
  grad_case.forward.operations = {
      {1U, Opcode::RmsNormModulate, {1U, 2U, 3U, 4U}, {5U},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::u64(AttrKey::BlockSize, 32U)}},
      {2U, Opcode::SwiGlu, {5U}, {6U},
       {Attribute::boolean(AttrKey::GateFirst, true)}},
      {3U, Opcode::ResidualGate, {7U, 6U, 8U}, {9U}, {}},
  };
  grad_case.targets = {1U, 2U, 3U, 4U, 7U, 8U};
  grad_case.bindings.emplace(1U, random_tensor(dtype, {rows, cols}, 51U));
  grad_case.bindings.emplace(
      2U, random_tensor(dtype, {cols}, 52U, 0.5F, 1.0F));
  grad_case.bindings.emplace(
      3U, random_tensor(dtype, {rows, cols}, 53U, 0.3F));
  grad_case.bindings.emplace(
      4U, random_tensor(dtype, {rows, cols}, 54U, 0.3F));
  grad_case.bindings.emplace(
      7U, random_tensor(dtype, {rows, cols / 2U}, 55U));
  grad_case.bindings.emplace(
      8U, random_tensor(dtype, {rows, cols / 2U}, 56U, 0.5F));
  append_loss(grad_case, 9U, 57U);
  return grad_case;
}

void test_group1_finite_differences() {
  finite_difference_check(rms_norm_case(dif::ir::DType::F32));
  finite_difference_check(
      rms_norm_modulate_case(dif::ir::DType::F32, true));
  finite_difference_check(
      rms_norm_modulate_case(dif::ir::DType::F32, false));
  finite_difference_check(swiglu_case(dif::ir::DType::F32, true));
  finite_difference_check(swiglu_case(dif::ir::DType::F32, false));
  finite_difference_check(residual_gate_case(dif::ir::DType::F32));
  finite_difference_check(layer_norm_case(dif::ir::DType::F32));
  finite_difference_check(composed_group1_case());
}

void test_group1_backend_parity() {
  constexpr float f32_bar = 1.0e-5F;
  constexpr float bf16_bar = 1.6e-2F;
  backend_cross_check(rms_norm_case(dif::ir::DType::F32), f32_bar);
  backend_cross_check(rms_norm_modulate_case(dif::ir::DType::F32, true),
                      f32_bar);
  backend_cross_check(rms_norm_modulate_case(dif::ir::DType::F32, false),
                      f32_bar);
  backend_cross_check(swiglu_case(dif::ir::DType::F32, true), f32_bar);
  backend_cross_check(swiglu_case(dif::ir::DType::F32, false), f32_bar);
  backend_cross_check(residual_gate_case(dif::ir::DType::F32), f32_bar);
  backend_cross_check(layer_norm_case(dif::ir::DType::F32), f32_bar);
  backend_cross_check(composed_group1_case(), f32_bar);
  backend_cross_check(rms_norm_case(dif::ir::DType::BF16), bf16_bar);
  backend_cross_check(rms_norm_modulate_case(dif::ir::DType::BF16, true),
                      bf16_bar);
  backend_cross_check(swiglu_case(dif::ir::DType::BF16, true), bf16_bar);
  backend_cross_check(residual_gate_case(dif::ir::DType::BF16), bf16_bar);
  backend_cross_check(layer_norm_case(dif::ir::DType::BF16), bf16_bar);
}

void test_group1_frozen_weight_economy() {
  // Differentiating an RmsNorm graph wrt the input only must emit the
  // one-output RmsNormBackward arity (no dead weight-gradient work).
  auto grad_case = rms_norm_case(dif::ir::DType::F32);
  const std::vector<std::uint32_t> input_only{1U};
  const auto differentiated = dif::training::differentiate(
      grad_case.forward, grad_case.loss, input_only);
  bool found = false;
  for (const auto &operation : differentiated.program.operations) {
    if (operation.opcode != dif::ir::Opcode::RmsNormBackward)
      continue;
    found = true;
    expect(operation.outputs.size() == 1U,
           "input-only RmsNorm differentiation emits the dx-only arity");
  }
  expect(found, "input-only RmsNorm differentiation emits RmsNormBackward");

  const auto full = dif::training::differentiate(
      grad_case.forward, grad_case.loss, grad_case.targets);
  for (const auto &operation : full.program.operations)
    if (operation.opcode == dif::ir::Opcode::RmsNormBackward)
      expect(operation.outputs.size() == 2U,
             "weight-target RmsNorm differentiation emits the weight "
             "gradient");
}

void expect_rejected(dif::ir::Program program, const char *message) {
  bool rejected = false;
  try {
    dif::ir::verify(program);
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, message);
}

void test_group1_verifier_negatives() {
  using namespace dif::ir;
  // Well-formed base program: SwiGluBackward.
  Program program;
  program.tensors = {
      {1U, DType::F32, TensorRole::Input, {2U, 3U}},
      {2U, DType::F32, TensorRole::Input, {2U, 6U}},
      {3U, DType::F32, TensorRole::Output, {2U, 6U}},
  };
  program.operations = {
      {1U, Opcode::SwiGluBackward, {1U, 2U}, {3U}, {}},
  };
  dif::ir::verify(program);

  {
    auto broken = program;
    broken.tensors[0].dims = {2U, 4U};
    for (auto &tensor : broken.tensors)
      if (tensor.id == 1U)
        tensor.dims = {2U, 4U};
    expect_rejected(broken,
                    "swiglu_backward rejects a non-halved grad_output");
  }
  {
    auto broken = program;
    for (auto &tensor : broken.tensors)
      if (tensor.id == 3U)
        tensor.dtype = DType::BF16;
    expect_rejected(broken, "swiglu_backward rejects mixed dtypes");
  }
  {
    auto broken = program;
    broken.operations[0].attributes.push_back(Attribute::u64(
        AttrKey::AccumulatorDType,
        static_cast<std::uint64_t>(DType::BF16)));
    expect_rejected(broken,
                    "swiglu_backward rejects a non-F32 accumulator");
  }
  {
    // RmsNormModulateBackward arity mismatch: weighted inputs with
    // unweighted outputs.
    Program mismatch;
    mismatch.tensors = {
        {1U, DType::F32, TensorRole::Input, {2U, 4U}},
        {2U, DType::F32, TensorRole::Input, {2U, 4U}},
        {3U, DType::F32, TensorRole::Input, {4U}},
        {4U, DType::F32, TensorRole::Input, {2U, 4U}},
        {5U, DType::F32, TensorRole::Output, {2U, 4U}},
        {6U, DType::F32, TensorRole::Output, {2U, 4U}},
        {7U, DType::F32, TensorRole::Output, {2U, 4U}},
    };
    mismatch.operations = {
        {1U, Opcode::RmsNormModulateBackward, {1U, 2U, 3U, 4U},
         {5U, 6U, 7U}, {}},
    };
    expect_rejected(mismatch,
                    "rms_norm_modulate_backward rejects an input/output "
                    "arity mismatch");
  }
  {
    // ResidualGateBackward requires two outputs.
    Program mismatch;
    mismatch.tensors = {
        {1U, DType::F32, TensorRole::Input, {2U, 4U}},
        {2U, DType::F32, TensorRole::Input, {2U, 4U}},
        {3U, DType::F32, TensorRole::Input, {2U, 4U}},
        {4U, DType::F32, TensorRole::Output, {2U, 4U}},
    };
    mismatch.operations = {
        {1U, Opcode::ResidualGateBackward, {1U, 2U, 3U}, {4U}, {}},
    };
    expect_rejected(mismatch,
                    "residual_gate_backward rejects a single output");
  }
  {
    // LayerNormBackward affine gradients must match the affine vectors.
    Program mismatch;
    mismatch.tensors = {
        {1U, DType::F32, TensorRole::Input, {2U, 4U}},
        {2U, DType::F32, TensorRole::Input, {2U, 4U}},
        {3U, DType::F32, TensorRole::Input, {4U}},
        {4U, DType::F32, TensorRole::Output, {2U, 4U}},
        {5U, DType::F32, TensorRole::Output, {4U}},
        {6U, DType::F32, TensorRole::Output, {2U}},
    };
    mismatch.operations = {
        {1U, Opcode::LayerNormBackward, {1U, 2U, 3U}, {4U, 5U, 6U}, {}},
    };
    expect_rejected(mismatch,
                    "layer_norm_backward rejects a mismatched bias gradient");
  }
  {
    // RmsNormBackward weight-gradient shape must match the weight.
    Program mismatch;
    mismatch.tensors = {
        {1U, DType::F32, TensorRole::Input, {2U, 4U}},
        {2U, DType::F32, TensorRole::Input, {2U, 4U}},
        {3U, DType::F32, TensorRole::Input, {4U}},
        {4U, DType::F32, TensorRole::Output, {2U, 4U}},
        {5U, DType::F32, TensorRole::Output, {2U}},
    };
    mismatch.operations = {
        {1U, Opcode::RmsNormBackward, {1U, 2U, 3U}, {4U, 5U}, {}},
    };
    expect_rejected(mismatch,
                    "rms_norm_backward rejects a mismatched weight gradient");
  }
}

} // namespace

int main() {
  try {
    test_group1_finite_differences();
    test_group1_backend_parity();
    test_group1_frozen_weight_economy();
    test_group1_verifier_negatives();
  } catch (const std::exception &error) {
    std::cerr << "UNCAUGHT: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " failures\n";
    return 1;
  }
  std::cout << "dit_backward_tests passed\n";
  return 0;
}
