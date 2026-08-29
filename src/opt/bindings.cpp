#include "dif/opt/bindings.hpp"

#include "dif/runtime/scalar.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <unordered_map>

namespace dif::opt {
namespace {

std::uint64_t mix(std::uint64_t value) {
  value ^= value >> 33U;
  value *= 0xff51afd7ed558ccdULL;
  value ^= value >> 33U;
  value *= 0xc4ceb9fe1a85ec53ULL;
  value ^= value >> 33U;
  return value;
}

float sample(std::uint64_t seed, std::uint32_t id, std::uint64_t index) {
  const auto bits = mix(seed * 0x9e3779b97f4a7c15ULL +
                        (static_cast<std::uint64_t>(id) << 32U) + index);
  const auto unit = static_cast<double>(bits >> 40U) / 16777216.0;
  return static_cast<float>((unit - 0.5) * 0.1);
}

// The largest index each integer tensor may hold, derived from the operations
// that consume it. Zero means "no consuming operation constrains it".
std::unordered_map<std::uint32_t, std::uint64_t>
index_ranges(const ir::Program &program) {
  std::unordered_map<std::uint32_t, std::uint64_t> ranges;
  const auto constrain = [&](std::uint32_t tensor, std::uint64_t limit) {
    if (limit == 0U)
      fail("index range for tensor " + std::to_string(tensor) + " is empty");
    auto &current = ranges[tensor];
    current = current == 0U ? limit : std::min(current, limit);
  };
  for (const auto &operation : program.operations) {
    switch (operation.opcode) {
    case ir::Opcode::GatherRows:
      constrain(operation.inputs[1],
                program.tensor(operation.inputs[0])->dims[0]);
      break;
    case ir::Opcode::SelectRowChunks:
      constrain(operation.inputs[1],
                program.tensor(operation.inputs[0])->dims[0]);
      break;
    case ir::Opcode::IndexedUpdateRows:
      constrain(operation.inputs[2],
                program.tensor(operation.inputs[1])->dims[0]);
      break;
    case ir::Opcode::H3AdaLNSelect:
      constrain(operation.inputs[1],
                program.tensor(operation.inputs[0])->dims[0] * 3U);
      break;
    default:
      break;
    }
  }
  return ranges;
}

} // namespace

runtime::TensorMap synthesize_bindings(const ir::Program &program,
                                       std::uint64_t seed) {
  const auto ranges = index_ranges(program);
  runtime::TensorMap bindings;
  for (const auto &description : program.tensors) {
    if (!description.has_role(ir::TensorRole::Input) &&
        !description.has_role(ir::TensorRole::Constant))
      continue;
    runtime::Tensor tensor{description.dtype, description.dims, {}};
    tensor.bytes.resize(description.byte_count());
    tensor.validate();
    const auto elements = description.element_count();
    if (description.dtype == ir::DType::I32) {
      const auto found = ranges.find(description.id);
      const auto limit = found == ranges.end() ? 0U : found->second;
      auto *values = reinterpret_cast<std::int32_t *>(tensor.bytes.data());
      for (std::uint64_t index = 0; index < elements; ++index) {
        // An unconstrained integer tensor is a scalar counter such as an
        // optimizer step, where one is the first meaningful value.
        values[index] = limit == 0U
                            ? 1
                            : static_cast<std::int32_t>(index % limit);
      }
    } else if (description.dtype == ir::DType::I8) {
      std::memset(tensor.bytes.data(), 0, tensor.bytes.size());
    } else {
      // Rank-one tensors are normalization weights and biases in every current
      // frontend; centring them on one keeps normalized activations in range.
      const auto vector = description.dims.size() == 1U;
      for (std::uint64_t index = 0; index < elements; ++index) {
        const auto value = sample(seed, description.id, index);
        runtime::store_float(tensor, index, vector ? 1.0F + value : value);
      }
    }
    bindings.emplace(description.id, std::move(tensor));
  }
  return bindings;
}

} // namespace dif::opt
