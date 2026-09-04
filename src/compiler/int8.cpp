#include "dif/compiler/int8.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <set>

namespace dif::compiler {
namespace {

float load_value(const runtime::Tensor &tensor, std::uint64_t index) {
  switch (tensor.dtype) {
  case ir::DType::F32: {
    float value = 0.0F;
    std::memcpy(&value, tensor.data() + index * sizeof(float), sizeof(float));
    return value;
  }
  case ir::DType::BF16: {
    std::uint16_t raw = 0U;
    std::memcpy(&raw, tensor.data() + index * sizeof(std::uint16_t),
                sizeof(raw));
    const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16U;
    float value = 0.0F;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
  }
  default:
    fail("an INT8 weight can only be made from an F32 or BF16 weight");
  }
}

} // namespace

Int8QuantizedWeight quantize_int8_weight(const runtime::Tensor &source) {
  if (source.dims.size() != 2U)
    fail("an INT8 weight is made from a rank-2 weight");
  const auto columns = source.dims[0];
  const auto inner = source.dims[1];

  Int8QuantizedWeight result;
  result.weight.dtype = ir::DType::I8;
  result.weight.dims = source.dims;
  result.weight.bytes.resize(static_cast<std::size_t>(columns * inner));
  result.scales.dtype = ir::DType::F32;
  result.scales.dims = {columns};
  result.scales.bytes.resize(static_cast<std::size_t>(columns) *
                             sizeof(float));

  auto *quantized =
      reinterpret_cast<std::int8_t *>(result.weight.mutable_data());
  auto *scales = reinterpret_cast<float *>(result.scales.mutable_data());
  for (std::uint64_t row = 0U; row < columns; ++row) {
    const auto base = row * inner;
    float maximum = 0.0F;
    for (std::uint64_t index = 0U; index < inner; ++index)
      maximum = std::max(maximum, std::fabs(load_value(source, base + index)));
    // The same expression QuantizeInt8Rows uses in its Direct mode, floor and
    // all. The floor matters: a row that is entirely zero would otherwise
    // divide by zero rather than quantize to zeros.
    const float scale = std::max(maximum / 127.0F, 1.0e-30F);
    scales[row] = scale;
    for (std::uint64_t index = 0U; index < inner; ++index) {
      const auto value = load_value(source, base + index);
      const auto rounded = static_cast<int>(std::nearbyint(value / scale));
      const auto code = static_cast<std::int8_t>(std::clamp(rounded, -127,
                                                            127));
      quantized[base + index] = code;
      const auto error =
          value - static_cast<float>(code) * scale;
      result.squared_error += static_cast<double>(error) * error;
      result.squared_reference += static_cast<double>(value) * value;
      result.maximum_absolute_error =
          std::max(result.maximum_absolute_error, std::fabs(error));
    }
  }
  return result;
}

Int8WeightOnlyRewrite
rewrite_int8_weight_only(const ir::Program &program,
                         const std::vector<std::uint32_t> &operations) {
  Int8WeightOnlyRewrite rewrite;
  rewrite.program = program;

  std::map<std::uint32_t, std::size_t> position;
  for (std::size_t index = 0U; index < rewrite.program.operations.size();
       ++index)
    position.emplace(rewrite.program.operations[index].id, index);

  // How many operations read each tensor. A weight two Linears share cannot
  // be converted for one of them alone.
  std::map<std::uint32_t, std::size_t> readers;
  for (const auto &operation : rewrite.program.operations)
    for (const auto input : operation.inputs)
      ++readers[input];

  std::uint32_t next_tensor = 1U;
  std::uint32_t next_operation = 1U;
  for (const auto &tensor : rewrite.program.tensors)
    next_tensor = std::max(next_tensor, tensor.id + 1U);
  for (const auto &operation : rewrite.program.operations)
    next_operation = std::max(next_operation, operation.id + 1U);

  std::set<std::uint32_t> seen;
  std::set<std::uint32_t> retired;
  for (const auto id : operations) {
    if (!seen.insert(id).second)
      fail("operation " + std::to_string(id) +
           " was named twice for INT8 residency");
    const auto found = position.find(id);
    if (found == position.end())
      fail("the program has no operation " + std::to_string(id));
    auto &operation = rewrite.program.operations[found->second];
    if (operation.opcode != ir::Opcode::Linear)
      fail("operation " + std::to_string(id) +
           " is not a linear, so it has no weight to make resident");
    if (operation.inputs.size() != 2U)
      fail("operation " + std::to_string(id) +
           " has a bias; the INT8 weight-only matmul takes none");
    const auto weight_id = operation.inputs[1];
    const auto *weight = rewrite.program.tensor(weight_id);
    if (weight == nullptr || weight->dims.size() != 2U)
      fail("operation " + std::to_string(id) + " has no rank-2 weight");
    if (!weight->has_role(ir::TensorRole::Constant))
      fail("the weight of operation " + std::to_string(id) +
           " is not a constant, so it is not frozen and must not be made "
           "resident in a format it cannot be trained in");
    if (readers[weight_id] != 1U)
      fail("the weight of operation " + std::to_string(id) +
           " is read by more than one operation");
    const auto *input = rewrite.program.tensor(operation.inputs[0]);
    const auto *output = rewrite.program.tensor(operation.outputs[0]);
    if (input->dtype != ir::DType::BF16 || output->dtype != ir::DType::BF16)
      fail("the INT8 weight-only matmul takes a BF16 input and produces a "
           "BF16 output; operation " + std::to_string(id) + " does not");

    Int8WeightOnlyEntry entry;
    entry.operation = id;
    entry.source_tensor_id = weight_id;
    entry.weight_tensor_id = next_tensor++;
    entry.scales_tensor_id = next_tensor++;
    rewrite.program.tensors.push_back(
        {entry.weight_tensor_id, ir::DType::I8, weight->roles, weight->dims});
    rewrite.program.tensors.push_back({entry.scales_tensor_id, ir::DType::F32,
                                       static_cast<std::uint32_t>(
                                           ir::TensorRole::Constant),
                                       {weight->dims[0]}});
    rewrite.bytes_before += weight->byte_count();
    rewrite.bytes_after +=
        weight->element_count() + weight->dims[0] * sizeof(float);

    operation.opcode = ir::Opcode::LinearInt8WeightScaled;
    operation.inputs = {operation.inputs[0], entry.weight_tensor_id,
                        entry.scales_tensor_id};
    retired.insert(weight_id);
    rewrite.entries.push_back(entry);
  }
  (void)next_operation;

  // The float weights are gone: leaving them in the program would keep them
  // in the memory plan, which is the whole cost this removes.
  rewrite.program.tensors.erase(
      std::remove_if(rewrite.program.tensors.begin(),
                     rewrite.program.tensors.end(),
                     [&](const ir::TensorDesc &desc) {
                       return retired.contains(desc.id);
                     }),
      rewrite.program.tensors.end());
  ir::verify(rewrite.program);
  return rewrite;
}

} // namespace dif::compiler
