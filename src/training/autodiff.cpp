#include "dif/training/autodiff.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

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
    // Targets may live in any supported float storage dtype (flame
    // BF16_GRAD_DECISION Option A: a gradient tensor carries the dtype of
    // its forward tensor; kernels accumulate in F32 internally).  The loss
    // itself stays F32[1].
    if (!description || (description->dtype != ir::DType::F32 &&
                         description->dtype != ir::DType::BF16 &&
                         description->dtype != ir::DType::F16))
      fail("autodiff target must be a floating tensor");
  }

  const std::unordered_set<std::uint32_t> requested(with_respect_to.begin(),
                                                    with_respect_to.end());
  std::unordered_set<std::uint32_t> produced;
  for (const auto &operation : forward.operations)
    for (const auto output : operation.outputs)
      produced.insert(output);

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
      add_operation(ir::Opcode::LinearBackwardInput,
                    {grad_output, operation.inputs[1]}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
      // Frozen-weight economy (flame lesson): the gradient of a leaf weight
      // is a reverse-mode sink — nothing else consumes it.  Emit
      // LinearBackwardWeight only when the weight is a differentiation
      // target, or when the weight is produced by another operation (then
      // its gradient is the path to earlier primals).  Graphs that request
      // every parameter gradient are emitted unchanged.
      const auto weight = operation.inputs[1];
      if (requested.contains(weight) || produced.contains(weight)) {
        const auto grad_weight =
            add_tensor(*result.program.tensor(weight));
        add_operation(ir::Opcode::LinearBackwardWeight,
                      {grad_output, operation.inputs[0]}, {grad_weight});
        accumulate(weight, grad_weight);
      }
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
    case ir::Opcode::RmsNorm: {
      const auto weight = operation.inputs[1];
      // Frozen-weight economy (mirrors Linear): the weight gradient is
      // emitted only when the weight is a differentiation target or is
      // produced by another operation.
      const bool needs_weight =
          requested.contains(weight) || produced.contains(weight);
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      std::vector<std::uint32_t> outputs{grad_input};
      std::uint32_t grad_weight = 0U;
      if (needs_weight) {
        grad_weight = add_tensor(*result.program.tensor(weight));
        outputs.push_back(grad_weight);
      }
      add_operation(ir::Opcode::RmsNormBackward,
                    {grad_output, operation.inputs[0], weight},
                    std::move(outputs),
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(operation.inputs[0], grad_input);
      if (needs_weight)
        accumulate(weight, grad_weight);
      break;
    }
    case ir::Opcode::RmsNormModulate: {
      const bool weighted = operation.inputs.size() == 4U;
      const auto x = operation.inputs[0];
      const auto scale = operation.inputs[weighted ? 2U : 1U];
      const auto shift = operation.inputs[weighted ? 3U : 2U];
      const auto grad_input = add_tensor(*result.program.tensor(x));
      const auto grad_scale = add_tensor(*result.program.tensor(scale));
      const auto grad_shift = add_tensor(*result.program.tensor(shift));
      std::vector<std::uint32_t> inputs{grad_output, x};
      std::vector<std::uint32_t> outputs{grad_input, grad_scale, grad_shift};
      std::uint32_t grad_weight = 0U;
      if (weighted) {
        inputs.push_back(operation.inputs[1]);
        grad_weight =
            add_tensor(*result.program.tensor(operation.inputs[1]));
      }
      inputs.push_back(scale);
      if (weighted)
        outputs.push_back(grad_weight);
      add_operation(ir::Opcode::RmsNormModulateBackward, std::move(inputs),
                    std::move(outputs),
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(x, grad_input);
      accumulate(scale, grad_scale);
      accumulate(shift, grad_shift);
      if (weighted)
        accumulate(operation.inputs[1], grad_weight);
      break;
    }
    case ir::Opcode::QkNormPartialRope: {
      const auto x = operation.inputs[0];
      const auto weight = operation.inputs[1];
      // cos/sin are precomputed non-differentiable tables; the rotation
      // layout travels on the op as an explicit RotaryDim (stamped with the
      // executor's default so forward and backward can never disagree).
      const bool needs_weight =
          requested.contains(weight) || produced.contains(weight);
      const auto grad_input = add_tensor(*result.program.tensor(x));
      std::vector<std::uint32_t> outputs{grad_input};
      std::uint32_t grad_weight = 0U;
      if (needs_weight) {
        grad_weight = add_tensor(*result.program.tensor(weight));
        outputs.push_back(grad_weight);
      }
      const auto head_dim = result.program.tensor(x)->dims[2];
      add_operation(
          ir::Opcode::QkNormPartialRopeBackward,
          {grad_output, x, weight, operation.inputs[2],
           operation.inputs[3]},
          std::move(outputs),
          {ir::Attribute::f64(ir::AttrKey::Epsilon,
                              operation.f64(ir::AttrKey::Epsilon, 1.0e-5)),
           ir::Attribute::u64(ir::AttrKey::RotaryDim,
                              operation.u64(ir::AttrKey::RotaryDim,
                                            head_dim))});
      accumulate(x, grad_input);
      if (needs_weight)
        accumulate(weight, grad_weight);
      break;
    }
    case ir::Opcode::LayerNorm: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      const auto grad_weight =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      const auto grad_bias =
          add_tensor(*result.program.tensor(operation.inputs[2]));
      add_operation(ir::Opcode::LayerNormBackward,
                    {grad_output, operation.inputs[0], operation.inputs[1]},
                    {grad_input, grad_weight, grad_bias},
                    {ir::Attribute::f64(
                        ir::AttrKey::Epsilon,
                        operation.f64(ir::AttrKey::Epsilon, 1.0e-5))});
      accumulate(operation.inputs[0], grad_input);
      accumulate(operation.inputs[1], grad_weight);
      accumulate(operation.inputs[2], grad_bias);
      break;
    }
    case ir::Opcode::SwiGlu: {
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::SwiGluBackward,
                    {grad_output, operation.inputs[0]}, {grad_input},
                    {ir::Attribute::boolean(
                        ir::AttrKey::GateFirst,
                        operation.boolean(ir::AttrKey::GateFirst, false))});
      accumulate(operation.inputs[0], grad_input);
      break;
    }
    case ir::Opcode::ResidualGate: {
      // d_residual = g (direct accumulation); one kernel produces the branch
      // and gate gradients.  DiffIR's ResidualGate is fully elementwise
      // (gate has the residual's shape), so d_gate = g*branch elementwise —
      // flame's sum-over-sequence applies only to its broadcast [B,1,C]
      // gate, which DiffIR expresses with explicitly expanded tensors.
      accumulate(operation.inputs[0], grad_output);
      const auto grad_branch =
          add_tensor(*result.program.tensor(operation.inputs[1]));
      const auto grad_gate =
          add_tensor(*result.program.tensor(operation.inputs[2]));
      add_operation(ir::Opcode::ResidualGateBackward,
                    {grad_output, operation.inputs[1], operation.inputs[2]},
                    {grad_branch, grad_gate});
      accumulate(operation.inputs[1], grad_branch);
      accumulate(operation.inputs[2], grad_gate);
      break;
    }
    case ir::Opcode::Cast: {
      // Cast is the mixed-precision boundary op.  The gradient of
      // Cast(x, dt) with upstream gradient g is Cast(g, dtype(x)):
      // add_tensor copies the primal description, so the contribution lands
      // in the source storage dtype.
      const auto grad_input =
          add_tensor(*result.program.tensor(operation.inputs[0]));
      add_operation(ir::Opcode::Cast, {grad_output}, {grad_input});
      accumulate(operation.inputs[0], grad_input);
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
