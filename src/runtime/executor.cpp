#include "dif/runtime/executor.hpp"

#include "dif/runtime/device_probe.hpp"
#include "dif/support/error.hpp"

#include <unordered_set>

namespace dif::runtime {

RunResult Executor::run(const ir::Program &program, const TensorMap &inputs,
                        const RunOptions &options) {
  auto prepared = prepare(program, inputs, options);
  auto result = prepared->run(inputs, options);
  if (result.target_profile.backend.empty()) {
    const auto probe = probe_device(ProbeBackend::Host);
    result.target_profile = probe.target;
    result.runtime_budget = probe.budget;
  }
  result.preparation_milliseconds = prepared->preparation_milliseconds();
  result.resident_bytes = prepared->resident_bytes();
  return result;
}

TensorMap PreparedExecution::capture_persistent_state() const {
  fail(name() + " does not implement persistent state capture");
}

void PreparedExecution::restore_persistent_state(const TensorMap &) {
  fail(name() + " does not implement persistent state restore");
}

void validate_persistent_state(
    const ir::Program &program,
    const std::vector<PersistentStateBinding> &state) {
  std::unordered_set<std::uint32_t> named;
  std::unordered_set<std::uint32_t> produced;
  for (const auto &operation : program.operations)
    for (const auto output : operation.outputs)
      produced.insert(output);
  for (const auto &binding : state) {
    const auto *input = program.tensor(binding.input);
    const auto *output = program.tensor(binding.output);
    if (!input || !output)
      fail("persistent state names a tensor the program does not have");
    if (!input->has_role(ir::TensorRole::Input))
      fail("persistent state source " + std::to_string(binding.input) +
           " is not a program input");
    if (!output->has_role(ir::TensorRole::Output))
      fail("persistent state destination " + std::to_string(binding.output) +
           " is not a program output");
    if (input->dtype != output->dtype || input->dims != output->dims)
      fail("persistent state pair " + std::to_string(binding.input) + " -> " +
           std::to_string(binding.output) +
           " disagrees on shape or dtype; the value has to keep its identity "
           "across steps");
    // The source must be a true input: if an operation also wrote it, the
    // carried value and the computed one would race for the same tensor.
    if (produced.contains(binding.input))
      fail("persistent state source " + std::to_string(binding.input) +
           " is also written by an operation");
    if (!produced.contains(binding.output))
      fail("persistent state destination " + std::to_string(binding.output) +
           " is not written by any operation");
    if (binding.input == binding.output)
      fail("persistent state pair must name two distinct tensors");
    if (!named.insert(binding.input).second ||
        !named.insert(binding.output).second)
      fail("persistent state names a tensor more than once");
  }
}

} // namespace dif::runtime
