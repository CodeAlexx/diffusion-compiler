#include "dif/runtime/executor.hpp"

namespace dif::runtime {

RunResult Executor::run(const ir::Program &program, const TensorMap &inputs,
                        const RunOptions &options) {
  auto prepared = prepare(program, inputs, options);
  auto result = prepared->run(inputs, options);
  result.preparation_milliseconds = prepared->preparation_milliseconds();
  result.resident_bytes = prepared->resident_bytes();
  return result;
}

} // namespace dif::runtime
