// Regression gate for cuBLASLt bias-epilogue absorption and Linear
// heuristic persistence (both difrun-explicit candidate knobs).
//
// Frozen from measured evidence (2026-08-31, RTX 3090 Ti):
// - F32 absorption was BYTE-IDENTICAL to the unfused Linear+BiasAdd pair on
//   the mlp and rectified-flow training programs; that is asserted here.
// - BF16 absorption differed by 1-2 BF16 ulp at the prediction (measured
//   max-abs 0.125 at value magnitude ~8-16); the frozen bar is 0.25 and the
//   launch/fusion accounting is asserted exactly.
// - Persistence restored every plan on a second prepare (rejected=0) with
//   outputs byte-identical to the persist-off run.
// GPU-only: every claim needs the real cuBLASLt path.

#include "dif/frontend/training.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &label) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << label << "\n";
}

std::uint16_t bf16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(
      (bits + 0x7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

float bf16_value(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

dif::runtime::Tensor make_tensor(const dif::ir::TensorDesc &description,
                                 std::uint64_t seed) {
  dif::runtime::Tensor tensor{description.dtype, description.dims, {}};
  std::uint64_t count = 1;
  for (const auto dim : description.dims)
    count *= dim;
  const bool zeroed =
      description.has_role(dif::ir::TensorRole::OptimizerState);
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  const auto next = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(
        (static_cast<double>((state >> 33U) & 0x7fffffffU) /
         static_cast<double>(0x7fffffffU)) *
            2.0 -
        1.0);
  };
  if (description.dtype == dif::ir::DType::I32) {
    tensor.bytes.assign(count * 4U, 0);
  } else if (description.dtype == dif::ir::DType::F32) {
    tensor.bytes.resize(count * 4U);
    for (std::uint64_t index = 0; index < count; ++index) {
      const float value = zeroed ? 0.0F : next();
      std::memcpy(tensor.bytes.data() + index * 4U, &value, 4U);
    }
  } else {
    tensor.bytes.resize(count * 2U);
    for (std::uint64_t index = 0; index < count; ++index) {
      const std::uint16_t value = zeroed ? 0U : bf16_bits(next());
      std::memcpy(tensor.bytes.data() + index * 2U, &value, 2U);
    }
  }
  return tensor;
}

dif::runtime::TensorMap bindings_for(const dif::ir::Program &program) {
  dif::runtime::TensorMap bindings;
  for (const auto &tensor : program.tensors)
    if (tensor.has_role(dif::ir::TensorRole::Input) ||
        tensor.has_role(dif::ir::TensorRole::Constant))
      bindings.emplace(tensor.id, make_tensor(tensor, 500U + tensor.id));
  return bindings;
}

std::vector<std::uint32_t> absorbable_ids(const dif::ir::Program &program) {
  std::vector<std::uint32_t> ids;
  for (std::size_t index = 0; index + 1U < program.operations.size();
       ++index) {
    const auto &linear = program.operations[index];
    const auto &bias = program.operations[index + 1U];
    if (linear.opcode != dif::ir::Opcode::Linear ||
        linear.inputs.size() != 2U ||
        bias.opcode != dif::ir::Opcode::BiasAdd ||
        bias.inputs.at(0) != linear.outputs.at(0))
      continue;
    std::size_t uses = 0;
    for (const auto &operation : program.operations)
      for (const auto input : operation.inputs)
        if (input == linear.outputs.at(0))
          ++uses;
    if (uses == 1U && program.tensor(linear.outputs.at(0))->roles == 0U)
      ids.push_back(linear.id);
  }
  return ids;
}

bool byte_identical(const dif::runtime::TensorMap &left,
                    const dif::runtime::TensorMap &right) {
  if (left.size() != right.size())
    return false;
  for (const auto &[id, tensor] : left) {
    const auto found = right.find(id);
    if (found == right.end() ||
        tensor.bytes.size() != found->second.bytes.size() ||
        std::memcmp(tensor.bytes.data(), found->second.bytes.data(),
                    tensor.bytes.size()) != 0)
      return false;
  }
  return true;
}

double max_abs_difference(const dif::runtime::Tensor &left,
                          const dif::runtime::Tensor &right) {
  double result = 0.0;
  if (left.dtype == dif::ir::DType::F32) {
    for (std::size_t index = 0; index * 4U < left.bytes.size(); ++index) {
      float a, b;
      std::memcpy(&a, left.bytes.data() + index * 4U, 4U);
      std::memcpy(&b, right.bytes.data() + index * 4U, 4U);
      result = std::max(result, std::abs(static_cast<double>(a) - b));
    }
  } else {
    for (std::size_t index = 0; index * 2U < left.bytes.size(); ++index) {
      std::uint16_t a, b;
      std::memcpy(&a, left.bytes.data() + index * 2U, 2U);
      std::memcpy(&b, right.bytes.data() + index * 2U, 2U);
      result = std::max(result, std::abs(static_cast<double>(bf16_value(a)) -
                                         bf16_value(b)));
    }
  }
  return result;
}

dif::runtime::RunOptions single_run_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  return options;
}

void test_absorption(const std::string &label,
                     const dif::ir::Program &program,
                     std::uint32_t prediction_output, bool require_bytes) {
  const auto ids = absorbable_ids(program);
  expect(!ids.empty(), label + ": program has absorbable sites");
  const auto bindings = bindings_for(program);
  const auto options = single_run_options();
  const auto plain =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  auto absorbed_options = options;
  absorbed_options.absorb_linear_bias_operations = ids;
  const auto absorbed = dif::runtime::make_cuda_executor()->run(
      program, bindings, absorbed_options);
  expect(absorbed.linear_bias_fusions.size() == ids.size(),
         label + ": every requested site reports a fusion");
  expect(plain.run_telemetry.kernel_launches ==
             absorbed.run_telemetry.kernel_launches + ids.size(),
         label + ": kernel launches drop by the site count");
  expect(plain.run_telemetry.cublaslt_matmuls ==
             absorbed.run_telemetry.cublaslt_matmuls,
         label + ": cuBLASLt dispatch count is unchanged");
  if (require_bytes) {
    expect(byte_identical(plain.outputs, absorbed.outputs),
           label + ": F32 absorption is byte-identical (measured bar)");
  } else {
    const auto delta = max_abs_difference(plain.outputs.at(prediction_output),
                                          absorbed.outputs.at(prediction_output));
    expect(delta <= 0.25,
           label + ": BF16 prediction stays within the frozen 0.25 bar");
    std::cout << label << " prediction max_abs=" << delta << "\n";
  }
}

void test_fail_closed(const dif::ir::Program &program) {
  const auto bindings = bindings_for(program);
  auto options = single_run_options();
  // A BiasAdd id is not a Linear.
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::BiasAdd) {
      options.absorb_linear_bias_operations = {operation.id};
      break;
    }
  bool threw = false;
  try {
    (void)dif::runtime::make_cuda_executor()->run(program, bindings, options);
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, "absorbing a non-Linear id fails closed");
  // A Linear NOT followed by its BiasAdd (the timestep projection feeds an
  // Add) must be refused.
  const auto eligible = absorbable_ids(program);
  options.absorb_linear_bias_operations.clear();
  for (const auto &operation : program.operations) {
    if (operation.opcode != dif::ir::Opcode::Linear)
      continue;
    if (std::find(eligible.begin(), eligible.end(), operation.id) ==
        eligible.end()) {
      options.absorb_linear_bias_operations = {operation.id};
      break;
    }
  }
  expect(!options.absorb_linear_bias_operations.empty(),
         "program offers a non-adjacent Linear for the negative");
  threw = false;
  try {
    (void)dif::runtime::make_cuda_executor()->run(program, bindings, options);
  } catch (const std::exception &) {
    threw = true;
  }
  expect(threw, "absorbing a Linear without an adjacent BiasAdd fails closed");
}

void test_persistence(const dif::ir::Program &program) {
  const auto bindings = bindings_for(program);
  const auto cache = std::filesystem::temp_directory_path() /
                     "dif-epilogue-test-algo-cache";
  std::filesystem::remove_all(cache);
  std::filesystem::create_directories(cache);
  auto base = single_run_options();
  base.cache_directory = cache;
  const auto off =
      dif::runtime::make_cuda_executor()->run(program, bindings, base);
  expect(off.linear_heuristic_cache.restored == 0U &&
             off.linear_heuristic_cache.saved_passive == 0U,
         "persistence stays inert while the flag is off");
  auto persist = base;
  persist.persist_linear_heuristics = true;
  const auto first =
      dif::runtime::make_cuda_executor()->run(program, bindings, persist);
  const auto second =
      dif::runtime::make_cuda_executor()->run(program, bindings, persist);
  std::size_t linear_count = 0;
  for (const auto &operation : program.operations)
    if (operation.opcode == dif::ir::Opcode::Linear)
      ++linear_count;
  expect(first.linear_heuristic_cache.saved_passive +
                 first.linear_heuristic_cache.restored ==
             linear_count,
         "first persisted prepare saves or restores every Linear plan");
  expect(second.linear_heuristic_cache.restored == linear_count &&
             second.linear_heuristic_cache.rejected == 0U,
         "second persisted prepare restores every Linear plan");
  expect(byte_identical(off.outputs, first.outputs) &&
             byte_identical(off.outputs, second.outputs),
         "persisted selections are byte-identical to fresh heuristics");
  std::cout << "persist prepare_ms first=" << first.preparation_milliseconds
            << " second=" << second.preparation_milliseconds << "\n";
  std::filesystem::remove_all(cache);
}

} // namespace

int main() {
  if (!dif::runtime::cuda_available()) {
    std::cout << "CUDA unavailable; epilogue gates skipped\n";
    return 0;
  }
  {
    dif::frontend::MlpTrainingConfig config;
    config.rows = 8;
    config.input_width = 32;
    config.hidden_width = 64;
    config.output_width = 16;
    const auto f32 = dif::frontend::make_mlp_training(config);
    test_absorption("mlp-f32", f32.program, f32.prediction_output, true);
    config.compute_dtype = dif::ir::DType::BF16;
    const auto bf16 = dif::frontend::make_mlp_training(config);
    test_absorption("mlp-bf16", bf16.program, bf16.prediction_output, false);
  }
  dif::frontend::RectifiedFlowTrainingConfig rf;
  rf.rows = 8;
  rf.latent_width = 64;
  rf.timestep_width = 32;
  rf.hidden_width = 128;
  rf.accumulation_steps = 2;
  const auto build = dif::frontend::make_rectified_flow_training(rf);
  test_absorption("rf-f32", build.program, 0U, true);
  test_fail_closed(build.program);
  test_persistence(build.program);
  if (failures != 0) {
    std::cerr << failures << " epilogue test failure(s)\n";
    return 1;
  }
  std::cout << "epilogue tests passed\n";
  return 0;
}
