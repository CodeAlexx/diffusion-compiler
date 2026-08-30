#include "dif/compiler/slice.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <unordered_set>

namespace dif::compiler {

ir::Program slice_operations(const ir::Program &program,
                             std::uint32_t first_operation,
                             std::uint32_t last_operation) {
  ir::verify(program);
  if (first_operation == 0U || last_operation < first_operation)
    fail("operation slice requires a nonzero ordered id range");

  ir::Program result;
  result.version = program.version;
  std::unordered_set<std::uint32_t> referenced;
  std::unordered_set<std::uint32_t> produced;
  std::unordered_set<std::uint32_t> consumed;
  std::unordered_set<std::uint32_t> consumed_outside;
  for (const auto &operation : program.operations) {
    const bool selected = operation.id >= first_operation &&
                          operation.id <= last_operation;
    if (!selected) {
      consumed_outside.insert(operation.inputs.begin(), operation.inputs.end());
      continue;
    }
    result.operations.push_back(operation);
    for (const auto input : operation.inputs) {
      referenced.insert(input);
      consumed.insert(input);
    }
    for (const auto output : operation.outputs) {
      referenced.insert(output);
      produced.insert(output);
    }
  }
  if (result.operations.empty())
    fail("operation slice selected no operations");

  for (const auto &description : program.tensors) {
    if (!referenced.contains(description.id))
      continue;
    auto sliced = description;
    if (description.has_role(ir::TensorRole::Constant)) {
      sliced.roles = static_cast<std::uint32_t>(ir::TensorRole::Constant);
      if (description.has_role(ir::TensorRole::Streamed))
        sliced.roles |= static_cast<std::uint32_t>(ir::TensorRole::Streamed);
    } else if (!produced.contains(description.id)) {
      sliced.roles = static_cast<std::uint32_t>(ir::TensorRole::Input);
      if (description.has_role(ir::TensorRole::Parameter))
        sliced.roles |= static_cast<std::uint32_t>(ir::TensorRole::Parameter);
      if (description.has_role(ir::TensorRole::OptimizerState))
        sliced.roles |=
            static_cast<std::uint32_t>(ir::TensorRole::OptimizerState);
    } else if (!consumed.contains(description.id) ||
               consumed_outside.contains(description.id) ||
               description.has_role(ir::TensorRole::Output)) {
      sliced.roles = static_cast<std::uint32_t>(ir::TensorRole::Output);
      if (description.has_role(ir::TensorRole::Parameter))
        sliced.roles |= static_cast<std::uint32_t>(ir::TensorRole::Parameter);
      if (description.has_role(ir::TensorRole::OptimizerState))
        sliced.roles |=
            static_cast<std::uint32_t>(ir::TensorRole::OptimizerState);
    } else {
      sliced.roles = static_cast<std::uint32_t>(ir::TensorRole::Internal);
    }
    result.tensors.push_back(std::move(sliced));
  }
  ir::verify(result);
  return result;
}

} // namespace dif::compiler
