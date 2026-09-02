#include "dif/runtime/executor.hpp"
#include "dif/runtime/device_probe.hpp"

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

} // namespace dif::runtime
