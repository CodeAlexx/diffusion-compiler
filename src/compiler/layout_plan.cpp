#include "dif/compiler/layout_plan.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <limits>

namespace dif::compiler {

ReshapeAliasPlan plan_reshape_aliases(const ir::Program &program) {
  ir::verify(program);
  ReshapeAliasPlan result;
  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::Reshape ||
        operation.inputs.size() != 1U || operation.outputs.size() != 1U)
      continue;
    const auto *input = program.tensor(operation.inputs.front());
    const auto *output = program.tensor(operation.outputs.front());
    if (!input || !output || input->dtype != output->dtype ||
        input->byte_count() != output->byte_count())
      fail("reshape alias candidate found an invalid reshape contract");
    // Dedicated/public tensors retain independent storage. Internal SSA
    // values are immutable, so all their consumers may safely share the
    // source bytes once liveness is extended to the last alias consumer.
    if (output->roles != ir::TensorRole::Internal)
      continue;
    auto root = input->id;
    if (const auto found = result.output_to_root_input.find(root);
        found != result.output_to_root_input.end())
      root = found->second;
    result.operation_ids.push_back(operation.id);
    result.output_to_root_input.emplace(output->id, root);
    if (result.eliminated_materialization_bytes >
        std::numeric_limits<std::uint64_t>::max() - output->byte_count())
      fail("reshape alias eliminated-byte total overflow");
    result.eliminated_materialization_bytes += output->byte_count();
  }
  return result;
}

} // namespace dif::compiler
