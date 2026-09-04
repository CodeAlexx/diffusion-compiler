// The flow-matching objective, checked against arithmetic rather than
// against itself.
//
// The interesting failure here is not a crash. It is a graph that runs, and
// trains, and produces a model that predicts the velocity at a timestep it
// was never shown -- because the interpolation used one timestep tensor and
// the model was told about another. No shape check catches that, so the
// structural test below does.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/training/flow_matching.hpp"
#include "dif/training/session.hpp"

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

void expect_near(double actual, double expected, double tolerance,
                 const std::string &message) {
  if (!(std::abs(actual - expected) <= tolerance)) {
    ++failures;
    std::cerr << "FAIL: " << message << " -- expected " << expected
              << ", got " << actual << "\n";
  }
}

template <typename Body>
void expect_refused(Body &&body, const std::string &fragment,
                    const std::string &message) {
  try {
    body();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(fragment) != std::string::npos)
      return;
    ++failures;
    std::cerr << "FAIL: " << message << " -- wrong reason: " << error.what()
              << "\n";
    return;
  }
  ++failures;
  std::cerr << "FAIL: " << message << " -- was accepted\n";
}

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

constexpr std::uint64_t kBatch = 2U;
constexpr std::uint64_t kTokens = 3U;
constexpr std::uint64_t kChannels = 4U;

// A stand-in for a denoiser: it takes a noisy sample and a timestep, and
// produces something the same shape. What it computes does not matter; that
// it consumes the timestep does.
struct Forward {
  Program program;
  std::uint32_t sample{};
  std::uint32_t timestep{};
  std::uint32_t weight{};
  std::uint32_t prediction{};
};

Forward toy_forward() {
  Forward forward;
  const std::vector<std::uint64_t> shape{kBatch, kTokens, kChannels};
  forward.sample = 1U;
  forward.timestep = 2U;
  forward.weight = 3U;
  forward.prediction = 5U;
  forward.program.tensors = {
      {1U, DType::F32, TensorRole::Input, shape},
      {2U, DType::F32, TensorRole::Input, {kBatch}},
      {3U, DType::F32, TensorRole::Input | TensorRole::Parameter,
       {kTokens * kChannels, kTokens * kChannels}},
      {4U, DType::F32, TensorRole::Internal, {kBatch, 1U, 1U}},
      {5U, DType::F32, TensorRole::Output, shape}};
  // The timestep is consumed, so a builder that quietly adds a SECOND one
  // leaves the first still connected and the disagreement is invisible.
  forward.program.operations = {
      {1U, Opcode::Reshape, {2U}, {4U}, {}},
      {2U, Opcode::Linear, {1U, 3U}, {5U}, {}}};
  dif::ir::verify(forward.program);
  return forward;
}

std::vector<float> ramp(std::size_t count, float start, float step) {
  std::vector<float> values(count);
  for (std::size_t index = 0U; index < count; ++index)
    values[index] = start + step * static_cast<float>(index);
  return values;
}

dif::runtime::Tensor f32(std::vector<std::uint64_t> dims,
                         std::vector<float> values) {
  dif::runtime::Tensor tensor;
  tensor.dtype = DType::F32;
  tensor.dims = std::move(dims);
  tensor.bytes.resize(values.size() * sizeof(float));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  return tensor;
}

std::vector<float> floats(const dif::runtime::Tensor &tensor) {
  std::vector<float> values(tensor.bytes.size() / sizeof(float));
  std::memcpy(values.data(), tensor.bytes.data(), tensor.bytes.size());
  return values;
}

void the_interpolation_is_the_arithmetic_it_claims() {
  const auto forward = toy_forward();
  const auto build = dif::training::add_flow_matching_loss(
      forward.program, forward.sample, forward.prediction, forward.timestep);

  const std::size_t count = kBatch * kTokens * kChannels;
  const auto clean = ramp(count, 0.5F, 0.25F);
  const auto noise = ramp(count, -1.0F, 0.125F);
  const std::vector<float> timesteps{0.25F, 0.75F};

  dif::runtime::TensorMap inputs;
  inputs.emplace(build.clean_input,
                 f32({kBatch, kTokens, kChannels}, clean));
  inputs.emplace(build.noise_input,
                 f32({kBatch, kTokens, kChannels}, noise));
  inputs.emplace(build.timestep_input, f32({kBatch}, timesteps));
  inputs.emplace(forward.weight,
                 f32({kTokens * kChannels, kTokens * kChannels},
                     std::vector<float>(kTokens * kChannels * kTokens *
                                            kChannels,
                                        0.0F)));

  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  const auto prepared = executor->prepare(build.program, inputs, options);
  const auto result = prepared->run(inputs, options);

  const auto target = floats(result.outputs.at(build.target_output));
  const auto noised = floats(result.outputs.at(build.noised_output));
  expect(target.size() == count && noised.size() == count,
         "the objective produces one target and one noised sample per value");
  const std::size_t per_row = kTokens * kChannels;
  for (std::size_t index = 0U; index < count; ++index) {
    const float t = timesteps[index / per_row];
    expect_near(target[index], noise[index] - clean[index], 1e-6,
                "the target is the straight-line velocity");
    expect_near(noised[index],
                clean[index] + t * (noise[index] - clean[index]), 1e-6,
                "the noised sample is the interpolation at t");
  }
  // t=0 must give the clean latent back and t=1 the noise exactly. These are
  // the two ends the whole objective is defined by.
  expect_near(noised[0], clean[0] + 0.25F * (noise[0] - clean[0]), 1e-6,
              "and the first row used the first row's timestep");
  expect_near(noised[per_row],
              clean[per_row] + 0.75F * (noise[per_row] - clean[per_row]),
              1e-6, "and the second row used its own, not the first's");

  // The loss is the mean squared error against that target, and with a zero
  // weight the prediction is zero, so it is the mean square of the target.
  double sum = 0.0;
  for (const auto value : target)
    sum += static_cast<double>(value) * value;
  expect_near(floats(result.outputs.at(build.loss_output)).front(),
              sum / static_cast<double>(count), 1e-5,
              "the loss is the mean squared error against the target");
}

// The failure no numeric check would find.
void there_is_exactly_one_timestep() {
  const auto forward = toy_forward();
  const auto build = dif::training::add_flow_matching_loss(
      forward.program, forward.sample, forward.prediction, forward.timestep);
  expect(build.timestep_input == forward.timestep,
         "the objective uses the model's own timestep, not a second one");
  std::size_t timestep_inputs = 0U;
  for (const auto &tensor : build.program.tensors)
    if (tensor.has_role(TensorRole::Input) && tensor.dtype == DType::F32 &&
        tensor.element_count() == kBatch)
      ++timestep_inputs;
  expect(timestep_inputs == 1U,
         "and the program has exactly one tensor shaped like a timestep");

  // The sample stops being an input: it is produced now, so nothing can
  // supply it from the host and silently bypass the noising.
  const auto *sample = build.program.tensor(forward.sample);
  expect(sample != nullptr && !sample->has_role(TensorRole::Input),
         "the model's sample input is now produced by the interpolation");
  expect(build.noised_output == forward.sample,
         "and it keeps its identity, so nothing downstream is rewritten");
}

void it_refuses_what_it_cannot_do() {
  const auto forward = toy_forward();
  expect_refused(
      [&] {
        dif::training::add_flow_matching_loss(forward.program, 99U,
                                              forward.prediction);
      },
      "names no tensor", "a sample that does not exist is refused");
  expect_refused(
      [&] {
        // The loss output is not an input at all: noising it would be
        // noising a result.
        dif::training::add_flow_matching_loss(forward.program,
                                              forward.prediction,
                                              forward.prediction);
      },
      "must be a program input", "a sample that is not an input is refused");
  expect_refused(
      [&] {
        dif::training::add_flow_matching_loss(
            forward.program, forward.sample, forward.prediction, 99U);
      },
      "names no tensor", "a timestep that does not exist is refused");
  expect_refused(
      [&] {
        // A timestep with the wrong number of rows would broadcast the wrong
        // sample's noise level across the batch. The sample itself is an
        // input of the wrong size, which is exactly the shape of that
        // mistake.
        dif::training::add_flow_matching_loss(
            forward.program, forward.sample, forward.prediction,
            forward.sample);
      },
      "one value per batch row",
      "a timestep that is not one value per sample is refused");
  {
    // A prediction that is not the sample's shape is a model predicting
    // something other than a velocity.
    auto mismatched = forward;
    mismatched.program.tensors.push_back(
        {6U, DType::F32, TensorRole::Output, {kBatch, kTokens}});
    mismatched.program.operations.push_back(
        {3U, Opcode::Slice, {5U}, {6U}, {}});
    expect_refused(
        [&] {
          dif::training::add_flow_matching_loss(mismatched.program,
                                                mismatched.sample, 6U);
        },
        "one value per value it was given",
        "a prediction shaped unlike the sample is refused");
  }
}

// The objective has to differentiate, or it trains nothing.
void the_objective_trains() {
  const auto forward = toy_forward();
  const auto build = dif::training::add_flow_matching_loss(
      forward.program, forward.sample, forward.prediction, forward.timestep);
  const auto plan = dif::training::compile(
      build.program, build.loss_output,
      std::vector<std::uint32_t>{forward.weight}, {});
  expect(plan.forward_operations > 0U && plan.optimizer_operations >
                                             plan.forward_operations,
         "the composed step has a forward, a backward and an optimizer");
  expect(!plan.bindings.empty(),
         "and the weight the loss depends on is trainable");

  const std::size_t count = kBatch * kTokens * kChannels;
  dif::runtime::TensorMap initial;
  initial.emplace(build.clean_input,
                  f32({kBatch, kTokens, kChannels}, ramp(count, 0.5F, 0.25F)));
  initial.emplace(build.noise_input,
                  f32({kBatch, kTokens, kChannels}, ramp(count, -1.0F, 0.1F)));
  initial.emplace(build.timestep_input,
                  f32({kBatch}, std::vector<float>{0.25F, 0.75F}));
  for (const auto &binding : plan.bindings) {
    const auto *parameter = plan.program.tensor(binding.parameter_input);
    const std::vector<float> zeros(parameter->element_count(), 0.0F);
    initial.emplace(binding.parameter_input, f32(parameter->dims, zeros));
    const auto *first = plan.program.tensor(binding.first_moment_input);
    initial.emplace(binding.first_moment_input,
                    f32(first->dims,
                        std::vector<float>(first->element_count(), 0.0F)));
    const auto *second = plan.program.tensor(binding.second_moment_input);
    initial.emplace(binding.second_moment_input,
                    f32(second->dims,
                        std::vector<float>(second->element_count(), 0.0F)));
  }
  for (const auto &tensor : plan.program.tensors)
    if (tensor.has_role(TensorRole::Constant) && !initial.contains(tensor.id))
      initial.emplace(tensor.id,
                      f32(tensor.dims,
                          std::vector<float>(tensor.element_count(), 0.0F)));

  auto executor = dif::runtime::make_cpu_executor();
  dif::training::TrainingSession session(plan, *executor, initial, {});
  dif::runtime::TensorMap batch;
  batch.emplace(build.clean_input, initial.at(build.clean_input));
  batch.emplace(build.noise_input, initial.at(build.noise_input));
  batch.emplace(build.timestep_input, initial.at(build.timestep_input));
  const auto first = session.step(batch);
  const auto second = session.step(batch);
  expect(std::isfinite(first.loss) && first.loss > 0.0F,
         "the first step has a finite, non-zero loss");
  expect(second.loss < first.loss,
         "and the second step's loss is lower: the objective trains");
}

} // namespace

int main() {
  the_interpolation_is_the_arithmetic_it_claims();
  there_is_exactly_one_timestep();
  it_refuses_what_it_cannot_do();
  the_objective_trains();
  if (failures != 0) {
    std::cerr << failures << " flow matching failure(s)\n";
    return 1;
  }
  std::cout << "flow matching tests passed\n";
  return 0;
}
