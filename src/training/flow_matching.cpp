#include "dif/training/flow_matching.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>

namespace dif::training {

FlowMatchingBuild add_flow_matching_loss(ir::Program forward,
                                         std::uint32_t sample_input,
                                         std::uint32_t prediction,
                                         std::uint32_t model_timestep) {
  using ir::AttrKey;
  using ir::Attribute;
  using ir::DType;
  using ir::Opcode;
  using ir::TensorRole;

  FlowMatchingBuild build;
  build.program = std::move(forward);
  auto &program = build.program;

  auto *sample = const_cast<ir::TensorDesc *>(program.tensor(sample_input));
  const auto *predicted = program.tensor(prediction);
  if (sample == nullptr)
    fail("the flow-matching sample input names no tensor");
  if (predicted == nullptr)
    fail("the flow-matching prediction names no tensor");
  if (!sample->has_role(TensorRole::Input))
    fail("the flow-matching sample must be a program input");
  if (sample->dims != predicted->dims || sample->dtype != predicted->dtype)
    fail("the flow-matching prediction must have the sample's shape and "
         "dtype: a model that predicts a velocity predicts one value per "
         "value it was given");
  if (sample->dims.empty())
    fail("the flow-matching sample must have a batch dimension");

  const auto dims = sample->dims;
  const auto dtype = sample->dtype;
  const auto batch = dims.front();

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  const auto add_tensor = [&](DType type, std::uint32_t roles,
                              std::vector<std::uint64_t> shape) {
    const auto id = next_tensor++;
    program.tensors.push_back({id, type, roles, std::move(shape)});
    return id;
  };
  // The noising runs BEFORE everything the forward graph does, so its
  // operations go at the front. Ordering is what the verifier checks; ids
  // only have to be unique.
  std::vector<ir::Operation> prologue;
  const auto emit = [&](Opcode opcode, std::vector<std::uint32_t> inputs,
                        std::vector<std::uint32_t> outputs,
                        std::vector<Attribute> attributes = {}) {
    prologue.push_back({next_operation++, opcode, std::move(inputs),
                        std::move(outputs), std::move(attributes)});
  };

  build.clean_input = add_tensor(dtype, TensorRole::Input, dims);
  build.noise_input = add_tensor(dtype, TensorRole::Input, dims);
  // One timestep per sample, broadcast over everything else it owns.
  std::vector<std::uint64_t> timestep_dims(dims.size(), 1U);
  timestep_dims.front() = batch;
  // The forward graph's own timestep, reshaped rather than duplicated. Two
  // timestep tensors is one too many: nothing would stop a caller filling
  // them differently, and the model would learn to predict the velocity at a
  // time it was never shown.
  std::uint32_t timestep_rows = 0U;
  if (model_timestep != 0U) {
    const auto *existing = program.tensor(model_timestep);
    if (existing == nullptr)
      fail("the flow-matching timestep names no tensor");
    if (!existing->has_role(TensorRole::Input))
      fail("the flow-matching timestep must be a program input");
    if (existing->element_count() != batch)
      fail("the flow-matching timestep must carry one value per batch row");
    build.timestep_input = model_timestep;
    if (existing->dims == timestep_dims) {
      timestep_rows = model_timestep;
    } else {
      timestep_rows =
          add_tensor(existing->dtype, TensorRole::Internal, timestep_dims);
      emit(Opcode::Reshape, {model_timestep}, {timestep_rows});
    }
  } else {
    build.timestep_input =
        add_tensor(DType::F32, TensorRole::Input, timestep_dims);
    timestep_rows = build.timestep_input;
  }

  // difference = noise - clean. There is no subtract opcode, and adding one
  // for this would be a worse trade than a fill nobody has to maintain.
  const auto minus_one = add_tensor(dtype, TensorRole::Internal, dims);
  emit(Opcode::Fill, {}, {minus_one},
       {Attribute::f64(AttrKey::Value, -1.0)});
  const auto negated_clean = add_tensor(dtype, TensorRole::Internal, dims);
  emit(Opcode::Multiply, {sample_input, minus_one}, {negated_clean});
  build.target_output = add_tensor(dtype, TensorRole::Output, dims);
  emit(Opcode::Add, {build.noise_input, negated_clean},
       {build.target_output});

  // noised = clean + t * difference
  // The timestep is rounded to the sample's dtype before it multiplies
  // anything. That is a perturbation of a value that was sampled at random
  // to begin with, and the TARGET does not depend on t at all, so the
  // objective stays exactly consistent with the noise level actually used.
  auto broadcast_source = timestep_rows;
  const auto *rows = program.tensor(timestep_rows);
  if (rows->dtype != dtype) {
    const auto cast = add_tensor(dtype, TensorRole::Internal, timestep_dims);
    emit(Opcode::Cast, {timestep_rows}, {cast});
    broadcast_source = cast;
  }
  const auto spread = add_tensor(dtype, TensorRole::Internal, dims);
  emit(Opcode::BroadcastTo, {broadcast_source}, {spread});
  const auto scaled = add_tensor(dtype, TensorRole::Internal, dims);
  emit(Opcode::Multiply, {spread, build.target_output}, {scaled});
  build.noised_output = add_tensor(dtype, TensorRole::Output, dims);
  emit(Opcode::Add, {sample_input, scaled}, {build.noised_output});

  // The forward graph's sample input is now produced here. The tensor keeps
  // its identity so not one operation downstream has to be rewritten.
  const auto clean = build.clean_input;
  for (auto &operation : prologue) {
    for (auto &input : operation.inputs)
      if (input == sample_input)
        input = clean;
    for (auto &output : operation.outputs)
      if (output == sample_input)
        output = clean;
  }
  // Now sample_input is written by the interpolation instead of read from
  // the host.
  for (auto &operation : prologue)
    for (auto &output : operation.outputs)
      if (output == build.noised_output)
        output = sample_input;
  program.tensors.erase(
      std::remove_if(program.tensors.begin(), program.tensors.end(),
                     [&](const ir::TensorDesc &desc) {
                       return desc.id == build.noised_output;
                     }),
      program.tensors.end());
  build.noised_output = sample_input;
  sample = const_cast<ir::TensorDesc *>(program.tensor(sample_input));
  sample->roles &= ~static_cast<std::uint32_t>(TensorRole::Input);
  sample->roles |= static_cast<std::uint32_t>(TensorRole::Output);

  program.operations.insert(program.operations.begin(), prologue.begin(),
                            prologue.end());

  build.loss_output = add_tensor(DType::F32, TensorRole::Output, {1U});
  program.operations.push_back({next_operation++, Opcode::MseLoss,
                                {prediction, build.target_output},
                                {build.loss_output},
                                {}});
  ir::verify(program);
  return build;
}

} // namespace dif::training
