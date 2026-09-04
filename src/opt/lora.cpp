#include "dif/opt/lora.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace dif::opt {

LoraResult insert_lora(const ir::Program &program, const LoraSpec &spec) {
  ir::verify(program);
  if (spec.rank == 0U)
    fail("LoRA rank must be positive");
  if (!(spec.alpha > 0.0))
    fail("LoRA alpha must be positive");
  if (spec.operations.empty())
    fail("LoRA needs at least one site to adapt");

  std::unordered_set<std::uint32_t> wanted;
  for (const auto id : spec.operations)
    if (!wanted.insert(id).second)
      fail("LoRA names operation " + std::to_string(id) + " more than once");

  LoraResult result;
  result.program.tensors = program.tensors;

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  const auto add_tensor = [&](ir::DType dtype, std::uint32_t roles,
                              std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    result.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  };
  const auto add_operation = [&](ir::Opcode opcode,
                                 std::vector<std::uint32_t> inputs,
                                 std::vector<std::uint32_t> outputs,
                                 std::vector<ir::Attribute> attributes = {}) {
    result.program.operations.push_back({next_operation++, opcode,
                                         std::move(inputs), std::move(outputs),
                                         std::move(attributes)});
  };

  const auto scale = spec.alpha / static_cast<double>(spec.rank);
  std::size_t adapted = 0U;

  for (const auto &operation : program.operations) {
    if (!wanted.contains(operation.id)) {
      result.program.operations.push_back(operation);
      continue;
    }
    if (operation.opcode != ir::Opcode::Linear)
      fail("LoRA site " + std::to_string(operation.id) + " is not a Linear");
    ++adapted;

    const auto activation = operation.inputs[0];
    const auto base_weight = operation.inputs[1];
    const auto *weight = program.tensor(base_weight);
    const auto *base_output = program.tensor(operation.outputs[0]);
    const auto *activation_tensor = program.tensor(activation);
    if (!weight || !base_output || !activation_tensor)
      fail("LoRA site references a tensor the program does not have");
    // Copied BY VALUE: result.program.tensor() returns a pointer into a
    // vector that every add_tensor below grows, and holding one across an
    // insertion reads freed memory.
    const auto compute_dtype = activation_tensor->dtype;
    const auto output_dtype = base_output->dtype;
    const auto output_dims = base_output->dims;
    const auto out_features = weight->dims[0];
    const auto in_features = weight->dims[1];

    LoraSite site;
    site.operation = operation.id;
    site.base_weight = base_weight;
    // The ADAPTED value keeps the tensor id the Linear used to produce, so
    // every existing reader sees it with no rewiring and the program's
    // interface -- output roles, tensor ids a caller already holds -- is
    // exactly what it was. The frozen path moves to a fresh internal tensor
    // instead.
    site.output = operation.outputs[0];
    site.base_output =
        add_tensor(output_dtype, ir::TensorRole::Internal, output_dims);

    auto frozen = operation;
    frozen.outputs[0] = site.base_output;
    result.program.operations.push_back(std::move(frozen));

    site.down = add_tensor(spec.parameter_dtype,
                           ir::TensorRole::Input | ir::TensorRole::Parameter,
                           {spec.rank, in_features});
    site.up = add_tensor(spec.parameter_dtype,
                         ir::TensorRole::Input | ir::TensorRole::Parameter,
                         {out_features, spec.rank});
    result.parameters.push_back(site.down);
    result.parameters.push_back(site.up);

    // Mixed precision: the adapters are stored in their own dtype and cross
    // an explicit Cast into the compute dtype, as the hand-built graphs did.
    auto down = site.down;
    auto up = site.up;
    if (spec.parameter_dtype != compute_dtype) {
      const auto cast_down = add_tensor(compute_dtype, ir::TensorRole::Internal,
                                        {spec.rank, in_features});
      add_operation(ir::Opcode::Cast, {site.down}, {cast_down});
      down = cast_down;
      const auto cast_up = add_tensor(compute_dtype, ir::TensorRole::Internal,
                                      {out_features, spec.rank});
      add_operation(ir::Opcode::Cast, {site.up}, {cast_up});
      up = cast_up;
    }

    // The low-rank path. The dense delta is only formed by the second
    // Linear, which is the point of the factorization.
    auto low_dims = output_dims;
    low_dims.back() = spec.rank;
    const auto low = add_tensor(compute_dtype, ir::TensorRole::Internal,
                                std::move(low_dims));
    add_operation(ir::Opcode::Linear, {activation, down}, {low});
    const auto delta =
        add_tensor(output_dtype, ir::TensorRole::Internal, output_dims);
    add_operation(ir::Opcode::Linear, {low, up}, {delta});
    const auto delta_scale =
        add_tensor(output_dtype, ir::TensorRole::Internal, output_dims);
    add_operation(ir::Opcode::Fill, {}, {delta_scale},
                  {ir::Attribute::f64(ir::AttrKey::Value, scale)});
    const auto scaled =
        add_tensor(output_dtype, ir::TensorRole::Internal, output_dims);
    add_operation(ir::Opcode::Multiply, {delta, delta_scale}, {scaled});
    add_operation(ir::Opcode::Add, {site.base_output, scaled}, {site.output});

    result.sites.push_back(site);
  }

  if (adapted != wanted.size())
    fail("LoRA named an operation the program does not have");
  ir::verify(result.program);
  return result;
}

} // namespace dif::opt
