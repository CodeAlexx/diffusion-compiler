#include "dif/training/autodiff.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <unordered_map>

namespace dif::training {

AutodiffResult differentiate(const ir::Program &forward,
                             std::uint32_t loss_tensor,
                             std::span<const std::uint32_t> with_respect_to) {
  ir::verify(forward);
  const auto *loss = forward.tensor(loss_tensor);
  if (!loss || loss->dtype != ir::DType::F32 ||
      loss->dims != std::vector<std::uint64_t>{1U})
    fail("autodiff loss must be an available F32[1] tensor");
  if (with_respect_to.empty())
    fail("autodiff requires at least one differentiation target");
  for (const auto tensor : with_respect_to) {
    const auto *description = forward.tensor(tensor);
    if (!description || description->dtype != ir::DType::F32)
      fail("autodiff target must be an F32 tensor");
  }

  AutodiffResult result;
  result.program = forward;
  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : result.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : result.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  auto add_tensor = [&](const ir::TensorDesc &primal) {
    const auto id = next_tensor++;
    result.program.tensors.push_back(
        {id, primal.dtype, ir::TensorRole::Internal, primal.dims});
    return id;
  };
  auto add_operation = [&](ir::Opcode opcode,
                           std::vector<std::uint32_t> inputs,
                           std::vector<std::uint32_t> outputs,
                           std::vector<ir::Attribute> attributes = {}) {
    result.program.operations.push_back(
        {next_operation++, opcode, std::move(inputs), std::move(outputs),
         std::move(attributes)});
  };

  std::unordered_map<std::uint32_t, std::uint32_t> gradients;
  const auto seed = add_tensor(*loss);
  add_operation(ir::Opcode::Fill, {}, {seed},
                {ir::Attribute::f64(ir::AttrKey::Value, 1.0)});
  gradients.emplace(loss_tensor, seed);

  auto accumulate = [&](std::uint32_t primal, std::uint32_t contribution) {
    const auto found = gradients.find(primal);
    if (found == gradients.end()) {
      gradients.emplace(primal, contribution);
      return;
    }
    const auto *description = result.program.tensor(primal);
    const auto sum = add_tensor(*description);
    add_operation(ir::Opcode::Add, {found->second, contribution}, {sum});
    found->second = sum;
  };

  for (auto iterator = forward.operations.rbegin();
       iterator != forward.operations.rend(); ++iterator) {
    const auto &operation = *iterator;
    std::uint32_t grad_output = 0U;
    for (const auto output : operation.outputs) {
      const auto found = gradients.find(output);
      if (found == gradients.end())
        continue;
      if (grad_output != 0U)
        fail("autodiff does not yet support active multi-output operations");
      grad_output = found->second;
    }
    if (grad_output == 0U)
      continue;

    switch (operation.opcode) {
    case ir::Opcode::MseLoss: {
      const auto grad_prediction =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::MseLossBackward,
                    {operation.inputs[0], operation.inputs[1], grad_output},
                    {grad_prediction});
      accumulate(operation.inputs[0], grad_prediction);
      break;
    }
    case ir::Opcode::BiasAdd: {
      accumulate(operation.inputs[0], grad_output);
      const auto grad_bias =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      add_operation(ir::Opcode::BiasBackward, {grad_output}, {grad_bias});
      accumulate(operation.inputs[1], grad_bias);
      break;
    }
    case ir::Opcode::Linear: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      const auto grad_weight =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      add_operation(ir::Opcode::LinearBackwardInput,
                    {grad_output, operation.inputs[1]}, {grad_input});
      add_operation(ir::Opcode::LinearBackwardWeight,
                    {grad_output, operation.inputs[0]}, {grad_weight});
      accumulate(operation.inputs[0], grad_input);
      accumulate(operation.inputs[1], grad_weight);
      if (operation.inputs.size() == 3U) {
        const auto grad_bias =
            add_tensor(*result.program.tensor(operation.inputs[2]));
        add_operation(ir::Opcode::BiasBackward, {grad_output}, {grad_bias});
        accumulate(operation.inputs[2], grad_bias);
      }
      break;
    }
    case ir::Opcode::SiLU: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::SiLUBackward,
                    {operation.inputs[0], grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::Add:
      accumulate(operation.inputs[0], grad_output);
      accumulate(operation.inputs[1], grad_output);
      break;
    case ir::Opcode::Multiply: {
      const auto grad_a =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      const auto grad_b =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      add_operation(ir::Opcode::Multiply,
                    {grad_output, operation.inputs[1]}, {grad_a});
      add_operation(ir::Opcode::Multiply,
                    {grad_output, operation.inputs[0]}, {grad_b});
      accumulate(operation.inputs[0], grad_a);
      accumulate(operation.inputs[1], grad_b);
      break;
    }
    case ir::Opcode::Fill:
      // A Fill has no primal inputs.  Its active output is a graph constant,
      // so reverse mode terminates at this leaf.
      break;
    default:
      fail("autodiff encountered unsupported active opcode " +
           std::string(ir::opcode_name(operation.opcode)));
    }
  }

  for (const auto primal : with_respect_to) {
    const auto found = gradients.find(primal);
    if (found == gradients.end())
      fail("autodiff target is disconnected from the loss");
    auto tensor = std::find_if(result.program.tensors.begin(),
                               result.program.tensors.end(), [&](const auto &v) {
                                 return v.id == found->second;
                               });
    tensor->roles |= ir::TensorRole::Output;
    result.gradients.emplace(primal, found->second);
  }
  ir::verify(result.program);
  return result;
}

} // namespace dif::training
