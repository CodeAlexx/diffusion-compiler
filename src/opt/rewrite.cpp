#include "dif/opt/rewrite.hpp"

#include "dif/compiler/int4.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/semantics.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dif::opt {
namespace {

using OperationIds = std::vector<std::uint32_t>;

const ir::Operation *find_operation(const ir::Program &program,
                                    std::uint32_t id) {
  for (const auto &operation : program.operations) {
    if (operation.id == id)
      return &operation;
  }
  return nullptr;
}

std::size_t operation_index(const ir::Program &program, std::uint32_t id) {
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    if (program.operations[index].id == id)
      return index;
  }
  fail("transform names operation " + std::to_string(id) +
       " which is not in the program");
}

ir::TensorDesc &mutable_tensor(ir::Program &program, std::uint32_t id) {
  for (auto &tensor : program.tensors) {
    if (tensor.id == id)
      return tensor;
  }
  fail("transform names tensor " + std::to_string(id) +
       " which is not in the program");
}

std::uint32_t fresh_tensor_id(const ir::Program &program) {
  std::uint32_t maximum = 0U;
  for (const auto &tensor : program.tensors)
    maximum = std::max(maximum, tensor.id);
  if (maximum == std::numeric_limits<std::uint32_t>::max())
    fail("DiffIR tensor id space is exhausted");
  return maximum + 1U;
}

std::uint32_t fresh_operation_id(const ir::Program &program) {
  std::uint32_t maximum = 0U;
  for (const auto &operation : program.operations)
    maximum = std::max(maximum, operation.id);
  if (maximum == std::numeric_limits<std::uint32_t>::max())
    fail("DiffIR operation id space is exhausted");
  return maximum + 1U;
}

bool internal_only(const ir::TensorDesc &tensor) {
  return tensor.roles == static_cast<std::uint32_t>(ir::TensorRole::Internal);
}

std::unordered_map<std::uint32_t, std::uint32_t>
producer_map(const ir::Program &program) {
  std::unordered_map<std::uint32_t, std::uint32_t> producers;
  for (const auto &operation : program.operations) {
    for (const auto output : operation.outputs)
      producers.emplace(output, operation.id);
  }
  return producers;
}

std::unordered_map<std::uint32_t, OperationIds>
consumer_map(const ir::Program &program) {
  std::unordered_map<std::uint32_t, OperationIds> consumers;
  for (const auto &operation : program.operations) {
    std::unordered_set<std::uint32_t> seen;
    for (const auto input : operation.inputs) {
      if (seen.insert(input).second)
        consumers[input].push_back(operation.id);
    }
  }
  return consumers;
}

void set_attribute(ir::Operation &operation, ir::AttrKey key,
                   const ir::Attribute &value) {
  for (auto &attribute : operation.attributes) {
    if (attribute.key == key) {
      attribute = value;
      return;
    }
  }
  operation.attributes.push_back(value);
}

bool equal_attributes(const std::vector<ir::Attribute> &left,
                      const std::vector<ir::Attribute> &right) {
  return left.size() == right.size() &&
         std::equal(left.begin(), left.end(), right.begin(),
                    [](const ir::Attribute &a, const ir::Attribute &b) {
                      return a.key == b.key && a.kind == b.kind &&
                             a.bits == b.bits;
                    });
}

void erase_operations(ir::Program &program,
                      const std::unordered_set<std::uint32_t> &ids) {
  std::erase_if(program.operations, [&](const ir::Operation &operation) {
    return ids.contains(operation.id);
  });
}

// Replaces every operation in `removed` with `replacement`, emitted at the
// position of the earliest removed operation. Rebuilding the vector keeps
// program order exact, which erase-then-insert by stale index would not.
void splice_operations(ir::Program &program,
                       const std::unordered_set<std::uint32_t> &removed,
                       const std::vector<ir::Operation> &replacement) {
  std::vector<ir::Operation> result;
  result.reserve(program.operations.size() + replacement.size());
  bool inserted = false;
  for (const auto &operation : program.operations) {
    if (removed.contains(operation.id)) {
      if (!inserted) {
        result.insert(result.end(), replacement.begin(), replacement.end());
        inserted = true;
      }
      continue;
    }
    result.push_back(operation);
  }
  if (!inserted)
    fail("operation splice found no operation to replace");
  program.operations = std::move(result);
}

// Removes tensor declarations that no operation produces or consumes and that
// carry no interface role, together with any binding they held. Constants that
// a rewrite made redundant disappear here, which is where the memory win of
// constant folding actually lands.
void prune_unreferenced(RewriteContext &context) {
  std::unordered_set<std::uint32_t> referenced;
  for (const auto &operation : context.program.operations) {
    referenced.insert(operation.inputs.begin(), operation.inputs.end());
    referenced.insert(operation.outputs.begin(), operation.outputs.end());
  }
  std::unordered_set<std::uint32_t> removed;
  std::erase_if(context.program.tensors, [&](const ir::TensorDesc &tensor) {
    const auto interface_role =
        tensor.has_role(ir::TensorRole::Input) ||
        tensor.has_role(ir::TensorRole::Output) ||
        tensor.has_role(ir::TensorRole::Parameter) ||
        tensor.has_role(ir::TensorRole::OptimizerState);
    if (interface_role || referenced.contains(tensor.id))
      return false;
    removed.insert(tensor.id);
    return true;
  });
  if (removed.empty())
    return;
  std::erase_if(context.bindings, [&](const auto &entry) {
    return removed.contains(entry.first);
  });
}

void substitute_input(ir::Program &program, std::uint32_t from,
                      std::uint32_t to) {
  for (auto &operation : program.operations) {
    for (auto &input : operation.inputs) {
      if (input == from)
        input = to;
    }
  }
}

bool widening_cast(ir::DType from, ir::DType to) {
  return to == ir::DType::F32 &&
         (from == ir::DType::BF16 || from == ir::DType::F16);
}

ir::DType dtype_from_code(std::uint64_t code) {
  switch (code) {
  case 1U:
    return ir::DType::F32;
  case 2U:
    return ir::DType::BF16;
  case 3U:
    return ir::DType::F16;
  default:
    fail("precision transform names an unsupported dtype code");
  }
}

bool float_dtype(ir::DType dtype) {
  return dtype == ir::DType::F32 || dtype == ir::DType::BF16 ||
         dtype == ir::DType::F16;
}

void expect_parameters(const Transform &transform, std::size_t count) {
  if (transform.parameters.size() != count)
    fail(std::string("transform ") +
         std::string(transform_kind_name(transform.kind)) +
         " expects " + std::to_string(count) + " parameters");
}

std::pair<std::uint64_t, std::uint64_t>
block_size_range(const ir::Operation &operation) {
  if (operation.opcode == ir::Opcode::Attention)
    return {32U, 256U};
  return {32U, 1024U};
}

bool power_of_two(std::uint64_t value) {
  return value != 0U && (value & (value - 1U)) == 0U;
}

} // namespace

std::string program_fingerprint(const ir::Program &program) {
  return hex_digest(ir::fingerprint(program));
}

std::string candidate_fingerprint(const RewriteContext &context) {
  std::vector<std::uint8_t> payload = ir::encode(context.program);
  for (unsigned shift = 0; shift < 64U; shift += 8U)
    payload.push_back(static_cast<std::uint8_t>(context.prefetch_distance >> shift));
  // Constant values are part of a candidate's identity: folding and
  // quantization both leave graph shapes that only differ in what they bind.
  std::map<std::uint32_t, const runtime::Tensor *> constants;
  for (const auto &tensor : context.program.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant))
      continue;
    const auto found = context.bindings.find(tensor.id);
    if (found == context.bindings.end())
      continue;
    constants.emplace(tensor.id, &found->second);
  }
  for (const auto &[id, tensor] : constants) {
    for (unsigned shift = 0; shift < 32U; shift += 8U)
      payload.push_back(static_cast<std::uint8_t>(id >> shift));
    const auto digest = sha256(std::span<const std::uint8_t>(
        tensor->data(), tensor->byte_size()));
    payload.insert(payload.end(), digest.begin(), digest.end());
  }
  return hex_digest(sha256(payload));
}

MemoryFootprint measure_memory(const RewriteContext &context) {
  MemoryFootprint footprint;
  const auto plan = compiler::plan_memory(context.program, 256U,
                                          context.prefetch_distance);
  footprint.planned_bytes = plan.total_bytes;
  footprint.naive_bytes = plan.naive_bytes;
  for (const auto &tensor : context.program.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant))
      continue;
    if (tensor.has_role(ir::TensorRole::Streamed))
      footprint.streamed_constant_bytes += tensor.byte_count();
    else
      footprint.resident_constant_bytes += tensor.byte_count();
  }
  return footprint;
}

runtime::TensorMap
evaluate_constant_operation(const ir::Program &program,
                            const ir::Operation &operation,
                            const runtime::TensorMap &bindings) {
  ir::Program isolated;
  runtime::TensorMap isolated_bindings;
  for (const auto input : operation.inputs) {
    const auto *description = program.tensor(input);
    if (!description)
      fail("constant folding lost an input tensor declaration");
    auto copy = *description;
    copy.roles = ir::TensorRole::Input;
    isolated.tensors.push_back(copy);
    const auto found = bindings.find(input);
    if (found == bindings.end())
      fail("constant folding requires a bound value for tensor " +
           std::to_string(input));
    isolated_bindings.emplace(input, found->second);
  }
  for (const auto output : operation.outputs) {
    const auto *description = program.tensor(output);
    if (!description)
      fail("constant folding lost an output tensor declaration");
    auto copy = *description;
    copy.roles = ir::TensorRole::Output;
    isolated.tensors.push_back(copy);
  }
  auto folded = operation;
  folded.id = 1U;
  isolated.operations = {folded};
  ir::verify(isolated);
  runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  // The portable typed CPU reference defines DiffIR semantics, so a folded
  // constant is the value the program is specified to compute.
  return runtime::make_cpu_executor()
      ->run(isolated, isolated_bindings, options)
      .outputs;
}


namespace {

// ---------------------------------------------------------------------------
// Site discovery. Each helper returns, in program order, the operation ids where
// one structural rewrite is legal. `apply` reuses the same helpers so a
// transform recorded with an empty scope replays over exactly the sites the
// search saw.
// ---------------------------------------------------------------------------

bool foldable_site(const ir::Program &program, const ir::Operation &operation,
                   const runtime::TensorMap &bindings, bool allow_arithmetic) {
  if (operation.opcode == ir::Opcode::Barrier || operation.outputs.empty())
    return false;
  if (!bit_exact_data_movement(operation.opcode) && !allow_arithmetic)
    return false;
  if (operation.opcode == ir::Opcode::Cast) {
    const auto *input = program.tensor(operation.inputs.at(0));
    const auto *output = program.tensor(operation.outputs.at(0));
    // A narrowing cast rounds, and rounding is the backend's to define.
    if (!input || !output || !widening_cast(input->dtype, output->dtype))
      return allow_arithmetic;
  }
  for (const auto input : operation.inputs) {
    const auto *description = program.tensor(input);
    if (!description || !description->has_role(ir::TensorRole::Constant))
      return false;
    if (!bindings.contains(input))
      return false;
  }
  for (const auto output : operation.outputs) {
    const auto *description = program.tensor(output);
    if (!description || !internal_only(*description))
      return false;
  }
  return true;
}

OperationIds foldable_sites(const RewriteContext &context,
                            bool allow_arithmetic) {
  OperationIds sites;
  for (const auto &operation : context.program.operations) {
    if (foldable_site(context.program, operation, context.bindings,
                      allow_arithmetic))
      sites.push_back(operation.id);
  }
  return sites;
}

OperationIds dead_operations(const ir::Program &program) {
  const auto consumers = consumer_map(program);
  OperationIds dead;
  for (const auto &operation : program.operations) {
    if (!pure_operation(operation.opcode) || operation.outputs.empty())
      continue;
    bool unused = true;
    for (const auto output : operation.outputs) {
      const auto *description = program.tensor(output);
      if (!description || !internal_only(*description) ||
          consumers.contains(output)) {
        unused = false;
        break;
      }
    }
    if (unused)
      dead.push_back(operation.id);
  }
  return dead;
}

std::string subexpression_key(const ir::Program &program,
                              const ir::Operation &operation) {
  std::string key(ir::opcode_name(operation.opcode));
  for (const auto input : operation.inputs)
    key += "/i" + std::to_string(input);
  for (const auto output : operation.outputs) {
    const auto *description = program.tensor(output);
    key += "/o" + std::to_string(static_cast<std::uint32_t>(description->dtype));
    for (const auto dimension : description->dims)
      key += ":" + std::to_string(dimension);
  }
  std::map<std::uint32_t, std::uint64_t> attributes;
  for (const auto &attribute : operation.attributes)
    attributes.emplace(static_cast<std::uint32_t>(attribute.key),
                       attribute.bits);
  for (const auto &[key_id, bits] : attributes)
    key += "/a" + std::to_string(key_id) + ":" + std::to_string(bits);
  return key;
}

// Returns pairs of (duplicate operation id, original operation id).
std::vector<std::pair<std::uint32_t, std::uint32_t>>
common_subexpressions(const ir::Program &program) {
  std::unordered_map<std::string, std::uint32_t> seen;
  std::vector<std::pair<std::uint32_t, std::uint32_t>> duplicates;
  for (const auto &operation : program.operations) {
    if (!pure_operation(operation.opcode) || operation.outputs.empty())
      continue;
    bool replaceable = true;
    for (const auto output : operation.outputs) {
      const auto *description = program.tensor(output);
      replaceable = replaceable && description && internal_only(*description);
    }
    const auto key = subexpression_key(program, operation);
    const auto found = seen.find(key);
    if (found == seen.end()) {
      seen.emplace(key, operation.id);
      continue;
    }
    if (replaceable)
      duplicates.emplace_back(operation.id, found->second);
  }
  return duplicates;
}

OperationIds linear_bias_sites(const ir::Program &program) {
  const auto producers = producer_map(program);
  const auto consumers = consumer_map(program);
  OperationIds sites;
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    const auto &operation = program.operations[index];
    if (operation.opcode != ir::Opcode::BiasAdd ||
        pinned_numeric_semantics(operation))
      continue;
    const auto intermediate = operation.inputs.at(0);
    const auto *description = program.tensor(intermediate);
    if (!description || !internal_only(*description))
      continue;
    const auto consumed = consumers.find(intermediate);
    if (consumed == consumers.end() || consumed->second.size() != 1U)
      continue;
    const auto produced = producers.find(intermediate);
    if (produced == producers.end())
      continue;
    const auto *linear = find_operation(program, produced->second);
    if (!linear || linear->opcode != ir::Opcode::Linear ||
        linear->inputs.size() != 2U || pinned_numeric_semantics(*linear))
      continue;
    const auto linear_index = operation_index(program, linear->id);
    // The bias has to be available where the Linear runs, not merely where the
    // BiasAdd ran.
    const auto bias = operation.inputs.at(1);
    const auto bias_producer = producers.find(bias);
    if (bias_producer != producers.end() &&
        operation_index(program, bias_producer->second) >= linear_index)
      continue;
    sites.push_back(operation.id);
  }
  return sites;
}

struct ParallelLinearSwiGluSite {
  std::uint32_t multiply{};
  std::uint32_t silu{};
  std::uint32_t gate_linear{};
  std::uint32_t value_linear{};
  std::uint32_t activation{};
  std::uint32_t gate_weight{};
  std::uint32_t value_weight{};
  std::uint32_t output{};
};

bool parallel_linear_swiglu_site(const ir::Program &program,
                                 const ir::Operation &multiply,
                                 ParallelLinearSwiGluSite &site) {
  if (multiply.opcode != ir::Opcode::Multiply ||
      multiply.inputs.size() != 2U || multiply.outputs.size() != 1U ||
      pinned_numeric_semantics(multiply))
    return false;
  const auto producers = producer_map(program);
  const auto consumers = consumer_map(program);
  const ir::Operation *silu = nullptr;
  const ir::Operation *value_linear = nullptr;
  for (std::size_t input = 0U; input < 2U; ++input) {
    const auto produced = producers.find(multiply.inputs[input]);
    if (produced == producers.end())
      return false;
    const auto *operation = find_operation(program, produced->second);
    if (!operation)
      return false;
    if (operation->opcode == ir::Opcode::SiLU)
      silu = operation;
    else if (operation->opcode == ir::Opcode::Linear)
      value_linear = operation;
    else
      return false;
  }
  if (!silu || !value_linear || silu->inputs.size() != 1U ||
      silu->outputs.size() != 1U || value_linear->inputs.size() != 2U ||
      value_linear->outputs.size() != 1U ||
      pinned_numeric_semantics(*silu) ||
      pinned_numeric_semantics(*value_linear))
    return false;
  const auto gate_produced = producers.find(silu->inputs[0]);
  if (gate_produced == producers.end())
    return false;
  const auto *gate_linear = find_operation(program, gate_produced->second);
  if (!gate_linear || gate_linear->opcode != ir::Opcode::Linear ||
      gate_linear->inputs.size() != 2U ||
      gate_linear->outputs.size() != 1U ||
      pinned_numeric_semantics(*gate_linear) ||
      gate_linear->inputs[0] != value_linear->inputs[0] ||
      gate_linear->attributes.size() != value_linear->attributes.size())
    return false;
  for (std::size_t index = 0U; index < gate_linear->attributes.size();
       ++index)
    if (gate_linear->attributes[index].key !=
            value_linear->attributes[index].key ||
        gate_linear->attributes[index].kind !=
            value_linear->attributes[index].kind ||
        gate_linear->attributes[index].bits !=
            value_linear->attributes[index].bits)
      return false;
  const auto exclusive = [&](std::uint32_t tensor, std::uint32_t consumer) {
    const auto found = consumers.find(tensor);
    return found != consumers.end() && found->second.size() == 1U &&
           found->second.front() == consumer;
  };
  if (!exclusive(gate_linear->outputs[0], silu->id) ||
      !exclusive(silu->outputs[0], multiply.id) ||
      !exclusive(value_linear->outputs[0], multiply.id))
    return false;
  const auto *activation = program.tensor(gate_linear->inputs[0]);
  const auto *gate_weight = program.tensor(gate_linear->inputs[1]);
  const auto *value_weight = program.tensor(value_linear->inputs[1]);
  const auto *gate = program.tensor(gate_linear->outputs[0]);
  const auto *value = program.tensor(value_linear->outputs[0]);
  const auto *output = program.tensor(multiply.outputs[0]);
  if (!activation || !gate_weight || !value_weight || !gate || !value ||
      !output || activation->dtype != ir::DType::BF16 ||
      gate_weight->dtype != ir::DType::BF16 ||
      value_weight->dtype != ir::DType::BF16 ||
      gate->dtype != ir::DType::BF16 || value->dtype != ir::DType::BF16 ||
      output->dtype != ir::DType::BF16 || gate_weight->dims.size() != 2U ||
      gate_weight->dims != value_weight->dims || gate->dims != value->dims ||
      gate->dims != output->dims || !internal_only(*gate) ||
      !internal_only(*value) || !internal_only(*program.tensor(silu->outputs[0])) ||
      gate_weight->roles != value_weight->roles ||
      !gate_weight->has_role(ir::TensorRole::Constant))
    return false;
  site = {multiply.id,
          silu->id,
          gate_linear->id,
          value_linear->id,
          gate_linear->inputs[0],
          gate_linear->inputs[1],
          value_linear->inputs[1],
          multiply.outputs[0]};
  return true;
}

OperationIds parallel_linear_swiglu_sites(const ir::Program &program) {
  OperationIds result;
  for (const auto &operation : program.operations) {
    ParallelLinearSwiGluSite site;
    if (parallel_linear_swiglu_site(program, operation, site))
      result.push_back(operation.id);
  }
  return result;
}

struct QkvFusionSite {
  std::uint32_t weight_split_operation{};
  std::uint32_t packed_weight{};
  std::uint32_t activation{};
  std::array<std::uint32_t, 3> projections{};
  std::array<std::uint32_t, 3> results{};
  std::size_t insertion_index{};
};

bool qkv_fusion_site(const ir::Program &program, const ir::Operation &split,
                     QkvFusionSite &site) {
  if (split.opcode != ir::Opcode::H3DeinterleaveQkvWeight ||
      split.outputs.size() != 3U)
    return false;
  const auto consumers = consumer_map(program);
  site.weight_split_operation = split.id;
  site.packed_weight = split.inputs.at(0);
  site.insertion_index = program.operations.size();
  const ir::Operation *first = nullptr;
  for (std::size_t component = 0; component < 3U; ++component) {
    const auto weight = split.outputs[component];
    const auto *description = program.tensor(weight);
    if (!description || !internal_only(*description))
      return false;
    const auto consumed = consumers.find(weight);
    if (consumed == consumers.end() || consumed->second.size() != 1U)
      return false;
    const auto *projection = find_operation(program, consumed->second.front());
    if (!projection || projection->opcode != ir::Opcode::Linear ||
        projection->inputs.size() != 2U || projection->inputs[1] != weight)
      return false;
    const auto *result = program.tensor(projection->outputs.at(0));
    if (!result || !internal_only(*result) || result->dims.size() != 3U)
      return false;
    if (!first) {
      first = projection;
    } else if (projection->inputs[0] != first->inputs[0] ||
               projection->attributes.size() != first->attributes.size()) {
      return false;
    } else {
      for (std::size_t i = 0; i < projection->attributes.size(); ++i) {
        if (projection->attributes[i].key != first->attributes[i].key ||
            projection->attributes[i].bits != first->attributes[i].bits)
          return false;
      }
    }
    site.projections[component] = projection->id;
    site.results[component] = projection->outputs[0];
    site.insertion_index =
        std::min(site.insertion_index, operation_index(program, projection->id));
  }
  site.activation = first->inputs[0];
  site.insertion_index =
      std::min(site.insertion_index, operation_index(program, split.id));
  // The fused projection runs where the earliest replaced operation ran, so the
  // shared activation has to be available there.
  const auto producers = producer_map(program);
  const auto activation_producer = producers.find(site.activation);
  if (activation_producer != producers.end() &&
      operation_index(program, activation_producer->second) >=
          site.insertion_index)
    return false;
  const auto *packed = program.tensor(site.packed_weight);
  const auto *activation = program.tensor(site.activation);
  const auto *result = program.tensor(site.results[0]);
  if (!packed || !activation || !result || packed->dtype != activation->dtype ||
      packed->dims.size() != 2U || activation->dims.size() != 2U)
    return false;
  const auto heads = split.u64(ir::AttrKey::Heads, 0U);
  const auto head_dim = split.u64(ir::AttrKey::HeadDim, 0U);
  if (heads == 0U || head_dim == 0U || result->dims[1] != heads ||
      result->dims[2] != head_dim || result->dims[0] != activation->dims[0])
    return false;
  return true;
}

OperationIds qkv_fusion_sites(const ir::Program &program) {
  OperationIds sites;
  for (const auto &operation : program.operations) {
    QkvFusionSite site;
    if (qkv_fusion_site(program, operation, site))
      sites.push_back(operation.id);
  }
  return sites;
}

struct QkvSplitSite {
  std::uint32_t deinterleave_operation{};
  std::uint32_t projection_operation{};
  std::uint32_t packed_activation{};
  std::uint32_t activation{};
  std::uint32_t packed_weight{};
  std::uint64_t heads{};
  std::uint64_t head_dim{};
};

bool qkv_split_site(const ir::Program &program,
                    const ir::Operation &deinterleave, QkvSplitSite &site) {
  if (deinterleave.opcode != ir::Opcode::H3DeinterleaveQkv ||
      deinterleave.outputs.size() != 3U)
    return false;
  const auto producers = producer_map(program);
  const auto consumers = consumer_map(program);
  site.deinterleave_operation = deinterleave.id;
  site.packed_activation = deinterleave.inputs.at(0);
  const auto *packed = program.tensor(site.packed_activation);
  if (!packed || !internal_only(*packed))
    return false;
  const auto consumed = consumers.find(site.packed_activation);
  if (consumed == consumers.end() || consumed->second.size() != 1U)
    return false;
  const auto produced = producers.find(site.packed_activation);
  if (produced == producers.end())
    return false;
  const auto *projection = find_operation(program, produced->second);
  if (!projection || projection->opcode != ir::Opcode::Linear ||
      projection->inputs.size() != 2U)
    return false;
  site.projection_operation = projection->id;
  site.activation = projection->inputs[0];
  site.packed_weight = projection->inputs[1];
  const auto *weight = program.tensor(site.packed_weight);
  const auto *result = program.tensor(deinterleave.outputs[0]);
  if (!weight || !result || weight->dims.size() != 2U ||
      result->dims.size() != 3U)
    return false;
  site.heads = deinterleave.u64(ir::AttrKey::Heads, result->dims[1]);
  site.head_dim = deinterleave.u64(ir::AttrKey::HeadDim, result->dims[2]);
  if (site.heads != result->dims[1] || site.head_dim != result->dims[2] ||
      weight->dims[0] != 3U * site.heads * site.head_dim)
    return false;
  for (const auto output : deinterleave.outputs) {
    const auto *description = program.tensor(output);
    if (!description || description->dims.size() != 3U)
      return false;
  }
  return true;
}

OperationIds qkv_split_sites(const ir::Program &program) {
  OperationIds sites;
  for (const auto &operation : program.operations) {
    QkvSplitSite site;
    if (qkv_split_site(program, operation, site))
      sites.push_back(operation.id);
  }
  return sites;
}

OperationIds cast_round_trip_sites(const ir::Program &program) {
  const auto producers = producer_map(program);
  const auto consumers = consumer_map(program);
  OperationIds sites;
  for (const auto &operation : program.operations) {
    if (operation.opcode != ir::Opcode::Cast)
      continue;
    const auto intermediate = operation.inputs.at(0);
    const auto *middle = program.tensor(intermediate);
    if (!middle || !internal_only(*middle))
      continue;
    const auto consumed = consumers.find(intermediate);
    if (consumed == consumers.end() || consumed->second.size() != 1U)
      continue;
    const auto produced = producers.find(intermediate);
    if (produced == producers.end())
      continue;
    const auto *widen = find_operation(program, produced->second);
    if (!widen || widen->opcode != ir::Opcode::Cast)
      continue;
    const auto *source = program.tensor(widen->inputs.at(0));
    const auto *result = program.tensor(operation.outputs.at(0));
    if (!source || !result || !internal_only(*result))
      continue;
    // Only a widen-then-narrow pair is exact; the reverse loses mantissa bits.
    if (!widening_cast(source->dtype, middle->dtype) ||
        result->dtype != source->dtype)
      continue;
    sites.push_back(operation.id);
  }
  return sites;
}

struct RematerializationSite {
  std::uint32_t producer{};
  std::uint32_t consumer{};
};

std::vector<RematerializationSite>
rematerialization_sites(const ir::Program &program) {
  const auto consumers = consumer_map(program);
  std::unordered_map<std::uint32_t, std::size_t> last_use;
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    for (const auto input : program.operations[index].inputs)
      last_use[input] = index;
  }
  std::vector<RematerializationSite> sites;
  for (std::size_t index = 0; index < program.operations.size(); ++index) {
    const auto &operation = program.operations[index];
    if (!pure_operation(operation.opcode) || operation.outputs.size() != 1U)
      continue;
    const auto value = operation.outputs[0];
    const auto *description = program.tensor(value);
    if (!description || !internal_only(*description))
      continue;
    const auto consumed = consumers.find(value);
    if (consumed == consumers.end() || consumed->second.size() < 2U)
      continue;
    const auto last = last_use.at(value);
    if (last <= index + 1U)
      continue;
    bool inputs_live = true;
    for (const auto input : operation.inputs) {
      const auto *input_description = program.tensor(input);
      if (!input_description)
        return {};
      const auto persistent =
          input_description->has_role(ir::TensorRole::Input) ||
          input_description->has_role(ir::TensorRole::Constant);
      const auto used = last_use.find(input);
      inputs_live = inputs_live &&
                    (persistent || (used != last_use.end() && used->second >= last));
    }
    if (!inputs_live)
      continue;
    sites.push_back({operation.id, program.operations[last].id});
  }
  return sites;
}

} // namespace

namespace {

// ---------------------------------------------------------------------------
// Transform application. A transform with an empty scope applies to every legal
// site in program order; a transform that names sites requires each of them to
// be legal, so a replayed plan reports drift instead of silently doing less.
// ---------------------------------------------------------------------------

OperationIds resolve_scope(const Transform &transform,
                           const OperationIds &available) {
  if (transform.operations.empty()) {
    if (available.empty())
      fail(std::string("transform ") +
           std::string(transform_kind_name(transform.kind)) +
           " has no legal site in this program");
    return available;
  }
  const std::unordered_set<std::uint32_t> legal(available.begin(),
                                                available.end());
  for (const auto id : transform.operations) {
    if (!legal.contains(id))
      fail(std::string("transform ") +
           std::string(transform_kind_name(transform.kind)) +
           " is not legal at operation " + std::to_string(id));
  }
  return transform.operations;
}

void apply_fold_constant_subgraph(const Transform &transform,
                                  RewriteContext &context) {
  expect_parameters(transform, 1U);
  const auto allow_arithmetic = transform.parameters[0] != 0U;
  if (transform.operations.empty()) {
    // Fold to a fixed point: folding one operation can make its consumer's
    // inputs constant in turn.
    bool changed = false;
    for (;;) {
      const auto sites = foldable_sites(context, allow_arithmetic);
      if (sites.empty())
        break;
      Transform single = transform;
      single.operations = {sites.front()};
      apply_fold_constant_subgraph(single, context);
      changed = true;
    }
    if (!changed)
      fail("fold_constant_subgraph has no legal site in this program");
    return;
  }
  for (const auto id : transform.operations) {
    const auto *operation = find_operation(context.program, id);
    if (!operation ||
        !foldable_site(context.program, *operation, context.bindings,
                       allow_arithmetic))
      fail("fold_constant_subgraph is not legal at operation " +
           std::to_string(id));
    const auto folded = *operation;
    auto values = evaluate_constant_operation(context.program, folded,
                                              context.bindings);
    // A folded value inherits the residency policy of everything it was
    // computed from, so folding does not quietly move weights on-device.
    bool streamed = !folded.inputs.empty();
    for (const auto input : folded.inputs)
      streamed = streamed &&
                 context.program.tensor(input)->has_role(ir::TensorRole::Streamed);
    for (const auto output : folded.outputs) {
      auto &description = mutable_tensor(context.program, output);
      description.roles = static_cast<std::uint32_t>(ir::TensorRole::Constant) |
                          (streamed
                               ? static_cast<std::uint32_t>(ir::TensorRole::Streamed)
                               : 0U);
      const auto value = values.find(output);
      if (value == values.end())
        fail("constant folding did not produce tensor " +
             std::to_string(output));
      context.bindings.insert_or_assign(output, value->second);
    }
    erase_operations(context.program, {id});
    prune_unreferenced(context);
  }
}

void apply_eliminate_dead_operations(const Transform &transform,
                                     RewriteContext &context) {
  expect_parameters(transform, 0U);
  auto dead = dead_operations(context.program);
  if (!transform.operations.empty())
    dead = resolve_scope(transform, dead);
  if (dead.empty())
    fail("eliminate_dead_operations has no legal site in this program");
  for (;;) {
    erase_operations(context.program,
                     std::unordered_set<std::uint32_t>(dead.begin(), dead.end()));
    prune_unreferenced(context);
    if (!transform.operations.empty())
      break;
    dead = dead_operations(context.program);
    if (dead.empty())
      break;
  }
}

void apply_common_subexpression(const Transform &transform,
                                RewriteContext &context) {
  expect_parameters(transform, 0U);
  auto duplicates = common_subexpressions(context.program);
  if (duplicates.empty())
    fail("common_subexpression has no legal site in this program");
  if (!transform.operations.empty()) {
    OperationIds available;
    for (const auto &[duplicate, original] : duplicates) {
      (void)original;
      available.push_back(duplicate);
    }
    const auto scope = resolve_scope(transform, available);
    const std::unordered_set<std::uint32_t> selected(scope.begin(), scope.end());
    std::erase_if(duplicates, [&](const auto &entry) {
      return !selected.contains(entry.first);
    });
  }
  for (const auto &[duplicate_id, original_id] : duplicates) {
    const auto *duplicate = find_operation(context.program, duplicate_id);
    const auto *original = find_operation(context.program, original_id);
    if (!duplicate || !original ||
        duplicate->outputs.size() != original->outputs.size())
      continue;
    const auto outputs = duplicate->outputs;
    const auto replacements = original->outputs;
    erase_operations(context.program, {duplicate_id});
    for (std::size_t index = 0; index < outputs.size(); ++index)
      substitute_input(context.program, outputs[index], replacements[index]);
    prune_unreferenced(context);
  }
}

void apply_fuse_linear_bias(const Transform &transform,
                            RewriteContext &context) {
  expect_parameters(transform, 0U);
  const auto scope =
      resolve_scope(transform, linear_bias_sites(context.program));
  for (const auto id : scope) {
    const auto sites = linear_bias_sites(context.program);
    if (std::find(sites.begin(), sites.end(), id) == sites.end())
      fail("fuse_linear_bias is not legal at operation " + std::to_string(id));
    const auto bias_add = *find_operation(context.program, id);
    const auto intermediate = bias_add.inputs[0];
    const auto producers = producer_map(context.program);
    auto &linear = context.program.operations[operation_index(
        context.program, producers.at(intermediate))];
    linear.inputs.push_back(bias_add.inputs[1]);
    linear.outputs[0] = bias_add.outputs[0];
    erase_operations(context.program, {id});
    prune_unreferenced(context);
  }
}

void apply_fuse_parallel_linear_swiglu(const Transform &transform,
                                       RewriteContext &context) {
  expect_parameters(transform, 0U);
  const auto scope =
      resolve_scope(transform, parallel_linear_swiglu_sites(context.program));
  for (const auto id : scope) {
    const auto *multiply = find_operation(context.program, id);
    ParallelLinearSwiGluSite site;
    if (!multiply ||
        !parallel_linear_swiglu_site(context.program, *multiply, site))
      fail("fuse_parallel_linear_swiglu is not legal at operation " +
           std::to_string(id));
    const auto gate_linear = *find_operation(context.program, site.gate_linear);
    const auto *gate_weight_desc = context.program.tensor(site.gate_weight);
    const auto *value_weight_desc = context.program.tensor(site.value_weight);
    const auto gate_binding = context.bindings.find(site.gate_weight);
    const auto value_binding = context.bindings.find(site.value_weight);
    if (gate_binding == context.bindings.end() ||
        value_binding == context.bindings.end())
      fail("fuse_parallel_linear_swiglu requires both bound weights");

    const auto packed_weight_id = fresh_tensor_id(context.program);
    auto packed_weight_dims = gate_weight_desc->dims;
    packed_weight_dims[0] += value_weight_desc->dims[0];
    context.program.tensors.push_back(
        {packed_weight_id, gate_weight_desc->dtype, gate_weight_desc->roles,
         packed_weight_dims});
    runtime::Tensor packed_weight{gate_weight_desc->dtype,
                                  packed_weight_dims, {}};
    packed_weight.bytes.resize(static_cast<std::size_t>(
        gate_binding->second.byte_size() + value_binding->second.byte_size()));
    std::memcpy(packed_weight.mutable_data(), gate_binding->second.data(),
                gate_binding->second.byte_size());
    std::memcpy(packed_weight.mutable_data() + gate_binding->second.byte_size(),
                value_binding->second.data(),
                value_binding->second.byte_size());
    packed_weight.validate();
    context.bindings.emplace(packed_weight_id, std::move(packed_weight));

    const auto *output_desc = context.program.tensor(site.output);
    auto packed_output_dims = output_desc->dims;
    packed_output_dims.back() *= 2U;
    const auto packed_output_id = fresh_tensor_id(context.program);
    context.program.tensors.push_back(
        {packed_output_id, output_desc->dtype, ir::TensorRole::Internal,
         packed_output_dims});
    const auto packed_linear_id = fresh_operation_id(context.program);
    ir::Operation packed_linear{
        packed_linear_id, ir::Opcode::Linear,
        {site.activation, packed_weight_id}, {packed_output_id},
        gate_linear.attributes};
    ir::Operation swiglu{
        site.multiply, ir::Opcode::SwiGlu, {packed_output_id}, {site.output},
        {ir::Attribute::boolean(ir::AttrKey::GateFirst, true)}};
    splice_operations(context.program,
                      {site.gate_linear, site.value_linear, site.silu,
                       site.multiply},
                      {packed_linear, swiglu});
    prune_unreferenced(context);
  }
}

void apply_fuse_parallel_linears(const Transform &transform,
                                 RewriteContext &context) {
  expect_parameters(transform, 0U);
  if (!transform.tensors.empty() || transform.operations.size() < 2U)
    fail("fuse_parallel_linears requires at least two operation ids and no "
         "tensor ids");

  std::vector<ir::Operation> linears;
  linears.reserve(transform.operations.size());
  const ir::TensorDesc *activation = nullptr;
  const ir::TensorDesc *first_weight = nullptr;
  const ir::TensorDesc *first_output = nullptr;
  std::uint64_t packed_width = 0U;
  std::size_t packed_weight_bytes = 0U;
  for (const auto operation_id : transform.operations) {
    const auto *operation = find_operation(context.program, operation_id);
    if (!operation || operation->opcode != ir::Opcode::Linear ||
        operation->inputs.size() != 2U || operation->outputs.size() != 1U)
      fail("fuse_parallel_linears requires unbiased Linear operations");
    const auto *current_activation =
        context.program.tensor(operation->inputs[0]);
    const auto *weight = context.program.tensor(operation->inputs[1]);
    const auto *output = context.program.tensor(operation->outputs[0]);
    const auto binding = context.bindings.find(operation->inputs[1]);
    if (!current_activation || !weight || !output ||
        weight->dims.size() != 2U || output->dims.empty() ||
        binding == context.bindings.end())
      fail("fuse_parallel_linears requires bound rank-2 weights");
    if (!activation) {
      activation = current_activation;
      first_weight = weight;
      first_output = output;
    } else if (operation->inputs[0] != linears.front().inputs[0] ||
               current_activation->dtype != activation->dtype ||
               current_activation->dims != activation->dims ||
               weight->dtype != first_weight->dtype ||
               weight->dims[1] != first_weight->dims[1] ||
               output->dtype != first_output->dtype ||
               output->dims.size() != first_output->dims.size() ||
               !std::equal(output->dims.begin(), output->dims.end() - 1,
                           first_output->dims.begin()) ||
               !equal_attributes(operation->attributes,
                                 linears.front().attributes))
      fail("fuse_parallel_linears operations do not share one compatible "
           "projection contract");
    if (packed_width > std::numeric_limits<std::uint64_t>::max() -
                           weight->dims[0] ||
        packed_weight_bytes > std::numeric_limits<std::size_t>::max() -
                                  binding->second.byte_size())
      fail("fuse_parallel_linears packed tensor size overflow");
    packed_width += weight->dims[0];
    packed_weight_bytes += binding->second.byte_size();
    linears.push_back(*operation);
  }

  const auto packed_weight_dtype = first_weight->dtype;
  const auto packed_weight_roles = first_weight->roles;
  const auto packed_weight_inner = first_weight->dims[1];
  const auto packed_output_dtype = first_output->dtype;
  auto packed_output_dims = first_output->dims;
  packed_output_dims.back() = packed_width;
  const auto packed_weight_id = fresh_tensor_id(context.program);
  context.program.tensors.push_back(
      {packed_weight_id, packed_weight_dtype, packed_weight_roles,
       {packed_width, packed_weight_inner}});
  runtime::Tensor packed_weight{packed_weight_dtype,
                                {packed_width, packed_weight_inner}, {}};
  packed_weight.bytes.resize(packed_weight_bytes);
  auto weight_offset = std::size_t{0U};
  for (const auto &operation : linears) {
    const auto &weight = context.bindings.at(operation.inputs[1]);
    std::memcpy(packed_weight.mutable_data() + weight_offset, weight.data(),
                weight.byte_size());
    weight_offset += weight.byte_size();
  }
  packed_weight.validate();
  context.bindings.emplace(packed_weight_id, std::move(packed_weight));

  const auto packed_output_id = fresh_tensor_id(context.program);
  context.program.tensors.push_back(
      {packed_output_id, packed_output_dtype, ir::TensorRole::Internal,
       packed_output_dims});
  std::vector<ir::Operation> replacement;
  replacement.reserve(linears.size() + 1U);
  replacement.push_back(
      {fresh_operation_id(context.program), ir::Opcode::Linear,
       {linears.front().inputs[0], packed_weight_id}, {packed_output_id},
       linears.front().attributes});
  std::uint64_t start = 0U;
  for (const auto &operation : linears) {
    replacement.push_back(
        {operation.id, ir::Opcode::Slice, {packed_output_id}, operation.outputs,
         {ir::Attribute::u64(ir::AttrKey::Axis,
                             packed_output_dims.size() - 1U),
          ir::Attribute::u64(ir::AttrKey::Start, start)}});
    start += context.program.tensor(operation.outputs[0])->dims.back();
  }
  splice_operations(
      context.program,
      std::unordered_set<std::uint32_t>(transform.operations.begin(),
                                        transform.operations.end()),
      replacement);
  prune_unreferenced(context);
}

void apply_fuse_qkv_projection(const Transform &transform,
                               RewriteContext &context) {
  expect_parameters(transform, 0U);
  const auto scope =
      resolve_scope(transform, qkv_fusion_sites(context.program));
  for (const auto id : scope) {
    const auto *split = find_operation(context.program, id);
    QkvFusionSite site;
    if (!split || !qkv_fusion_site(context.program, *split, site))
      fail("fuse_qkv_projection is not legal at operation " +
           std::to_string(id));
    const auto split_operation = *split;
    const auto *projection =
        find_operation(context.program, site.projections[0]);
    const auto projection_attributes = projection->attributes;
    // Preserve descriptions by value before appending to program.tensors;
    // vector growth invalidates pointers returned by Program::tensor().
    const auto activation = *context.program.tensor(site.activation);
    const auto result = *context.program.tensor(site.results[0]);
    const auto packed_id = fresh_tensor_id(context.program);
    context.program.tensors.push_back(
        {packed_id, activation.dtype, ir::TensorRole::Internal,
         {activation.dims[0], 3U * result.dims[1] * result.dims[2]}});
    auto operation_id = fresh_operation_id(context.program);
    ir::Operation packed_projection{operation_id++, ir::Opcode::Linear,
                                    {site.activation, site.packed_weight},
                                    {packed_id},
                                    projection_attributes};
    std::vector<ir::Attribute> deinterleave_attributes = {
        ir::Attribute::u64(ir::AttrKey::Heads, result.dims[1]),
        ir::Attribute::u64(ir::AttrKey::HeadDim, result.dims[2])};
    if (const auto *block = split_operation.find(ir::AttrKey::BlockSize))
      deinterleave_attributes.push_back(*block);
    ir::Operation deinterleave{
        operation_id, ir::Opcode::H3DeinterleaveQkv, {packed_id},
        {site.results[0], site.results[1], site.results[2]},
        std::move(deinterleave_attributes)};
    splice_operations(context.program,
                      {split_operation.id, site.projections[0],
                       site.projections[1], site.projections[2]},
                      {packed_projection, deinterleave});
    prune_unreferenced(context);
  }
}

void apply_split_qkv_projection(const Transform &transform,
                                RewriteContext &context) {
  expect_parameters(transform, 0U);
  const auto scope = resolve_scope(transform, qkv_split_sites(context.program));
  for (const auto id : scope) {
    const auto *deinterleave = find_operation(context.program, id);
    QkvSplitSite site;
    if (!deinterleave || !qkv_split_site(context.program, *deinterleave, site))
      fail("split_qkv_projection is not legal at operation " +
           std::to_string(id));
    const auto results = deinterleave->outputs;
    // Preserve the description by value: appending component tensors may
    // reallocate program.tensors and invalidate pointers into that vector.
    const auto weight = *context.program.tensor(site.packed_weight);
    const auto *projection =
        find_operation(context.program, site.projection_operation);
    const auto projection_attributes = projection->attributes;
    const auto inner = site.heads * site.head_dim;
    std::array<std::uint32_t, 3> component_weights{};
    for (std::size_t component = 0; component < 3U; ++component) {
      component_weights[component] = fresh_tensor_id(context.program);
      context.program.tensors.push_back({component_weights[component],
                                         weight.dtype,
                                         ir::TensorRole::Internal,
                                         {inner, weight.dims[1]}});
    }
    auto operation_id = fresh_operation_id(context.program);
    std::vector<ir::Operation> replacement;
    replacement.push_back(
        {operation_id++, ir::Opcode::H3DeinterleaveQkvWeight,
         {site.packed_weight},
         {component_weights[0], component_weights[1], component_weights[2]},
         {ir::Attribute::u64(ir::AttrKey::Heads, site.heads),
          ir::Attribute::u64(ir::AttrKey::HeadDim, site.head_dim)}});
    for (std::size_t component = 0; component < 3U; ++component)
      replacement.push_back({operation_id++, ir::Opcode::Linear,
                             {site.activation, component_weights[component]},
                             {results[component]},
                             projection_attributes});
    splice_operations(context.program, {site.projection_operation, id},
                      replacement);
    prune_unreferenced(context);
  }
}

void apply_elide_cast_round_trip(const Transform &transform,
                                 RewriteContext &context) {
  expect_parameters(transform, 0U);
  const auto scope =
      resolve_scope(transform, cast_round_trip_sites(context.program));
  for (const auto id : scope) {
    const auto sites = cast_round_trip_sites(context.program);
    if (std::find(sites.begin(), sites.end(), id) == sites.end())
      fail("elide_cast_round_trip is not legal at operation " +
           std::to_string(id));
    const auto narrow = *find_operation(context.program, id);
    const auto producers = producer_map(context.program);
    const auto widen =
        *find_operation(context.program, producers.at(narrow.inputs[0]));
    erase_operations(context.program, {narrow.id, widen.id});
    substitute_input(context.program, narrow.outputs[0], widen.inputs[0]);
    prune_unreferenced(context);
  }
}

void apply_rematerialize_producer(const Transform &transform,
                                  RewriteContext &context) {
  expect_parameters(transform, 0U);
  if (transform.operations.size() != 2U)
    fail("rematerialize_producer names a producer and a consumer operation");
  const auto producer_id = transform.operations[0];
  const auto consumer_id = transform.operations[1];
  const auto sites = rematerialization_sites(context.program);
  const auto found = std::find_if(
      sites.begin(), sites.end(), [&](const RematerializationSite &site) {
        return site.producer == producer_id && site.consumer == consumer_id;
      });
  if (found == sites.end())
    fail("rematerialize_producer is not legal for operations " +
         std::to_string(producer_id) + " and " + std::to_string(consumer_id));
  const auto producer = *find_operation(context.program, producer_id);
  const auto *description = context.program.tensor(producer.outputs[0]);
  const auto clone_value = fresh_tensor_id(context.program);
  context.program.tensors.push_back({clone_value, description->dtype,
                                     ir::TensorRole::Internal,
                                     description->dims});
  ir::Operation clone = producer;
  clone.id = fresh_operation_id(context.program);
  clone.outputs = {clone_value};
  const auto consumer_index = operation_index(context.program, consumer_id);
  context.program.operations.insert(
      context.program.operations.begin() +
          static_cast<std::ptrdiff_t>(consumer_index),
      clone);
  auto &consumer =
      context.program.operations[operation_index(context.program, consumer_id)];
  for (auto &input : consumer.inputs) {
    if (input == producer.outputs[0])
      input = clone_value;
  }
}

void apply_set_block_size(const Transform &transform, RewriteContext &context) {
  expect_parameters(transform, 1U);
  const auto block = transform.parameters[0];
  if (!power_of_two(block))
    fail("set_block_size requires a power-of-two block size");
  OperationIds scope = transform.operations;
  if (scope.empty()) {
    for (const auto &operation : context.program.operations) {
      if (operation.find(ir::AttrKey::BlockSize) == nullptr)
        continue;
      const auto [low, high] = block_size_range(operation);
      if (block >= low && block <= high)
        scope.push_back(operation.id);
    }
    if (scope.empty())
      fail("set_block_size has no legal site in this program");
  }
  for (const auto id : scope) {
    auto &operation = context.program.operations[operation_index(context.program, id)];
    if (operation.find(ir::AttrKey::BlockSize) == nullptr)
      fail("set_block_size names operation " + std::to_string(id) +
           " which has no block-size attribute");
    const auto [low, high] = block_size_range(operation);
    if (block < low || block > high)
      fail("set_block_size value is outside the legal range of operation " +
           std::to_string(id));
    set_attribute(operation, ir::AttrKey::BlockSize,
                  ir::Attribute::u64(ir::AttrKey::BlockSize, block));
  }
}

void apply_set_tile_shape(const Transform &transform, RewriteContext &context) {
  expect_parameters(transform, 3U);
  for (const auto value : transform.parameters) {
    if (value == 0U)
      fail("set_tile_shape requires nonzero tile extents");
  }
  OperationIds scope = transform.operations;
  if (scope.empty()) {
    for (const auto &operation : context.program.operations) {
      if (operation.opcode == ir::Opcode::Linear)
        scope.push_back(operation.id);
    }
    if (scope.empty())
      fail("set_tile_shape has no legal site in this program");
  }
  for (const auto id : scope) {
    auto &operation =
        context.program.operations[operation_index(context.program, id)];
    if (operation.opcode != ir::Opcode::Linear)
      fail("set_tile_shape applies to linear operations");
    set_attribute(operation, ir::AttrKey::TileM,
                  ir::Attribute::u64(ir::AttrKey::TileM, transform.parameters[0]));
    set_attribute(operation, ir::AttrKey::TileN,
                  ir::Attribute::u64(ir::AttrKey::TileN, transform.parameters[1]));
    set_attribute(operation, ir::AttrKey::TileK,
                  ir::Attribute::u64(ir::AttrKey::TileK, transform.parameters[2]));
  }
}

void apply_set_implementation(const Transform &transform,
                              RewriteContext &context, ir::Opcode opcode) {
  expect_parameters(transform, 1U);
  const auto implementation = transform.parameters[0];
  OperationIds scope = transform.operations;
  if (scope.empty()) {
    for (const auto &operation : context.program.operations) {
      if (operation.opcode == opcode && !pinned_numeric_semantics(operation))
        scope.push_back(operation.id);
    }
    if (scope.empty())
      fail("implementation selection has no legal site in this program");
  }
  for (const auto id : scope) {
    auto &operation =
        context.program.operations[operation_index(context.program, id)];
    if (operation.opcode != opcode)
      fail("implementation selection names an operation of the wrong opcode");
    if (pinned_numeric_semantics(operation))
      fail("operation " + std::to_string(id) +
           " pins its numeric semantics and cannot change implementation");
    set_attribute(
        operation, ir::AttrKey::Implementation,
        ir::Attribute::u64(ir::AttrKey::Implementation, implementation));
  }
}

void apply_set_operation_precision(const Transform &transform,
                                   RewriteContext &context) {
  expect_parameters(transform, 1U);
  if (transform.operations.size() != 1U)
    fail("set_operation_precision applies to exactly one operation");
  const auto target = dtype_from_code(transform.parameters[0]);
  const auto index = operation_index(context.program, transform.operations[0]);
  auto operation = context.program.operations[index];
  if (!dtype_uniform(operation.opcode))
    fail("set_operation_precision requires a dtype-uniform operation");
  if (pinned_numeric_semantics(operation))
    fail("operation " + std::to_string(operation.id) +
         " pins its numeric semantics and cannot change precision");
  ir::DType current{};
  bool first = true;
  const auto check = [&](std::uint32_t id) {
    const auto *description = context.program.tensor(id);
    if (!description || !float_dtype(description->dtype))
      fail("set_operation_precision requires floating-point operands");
    if (first) {
      current = description->dtype;
      first = false;
    } else if (description->dtype != current) {
      fail("set_operation_precision requires one operand dtype");
    }
  };
  for (const auto input : operation.inputs)
    check(input);
  for (const auto output : operation.outputs)
    check(output);
  if (current == target)
    fail("operation " + std::to_string(operation.id) +
         " already runs at the requested precision");
  std::vector<ir::Operation> prologue;
  std::vector<ir::Operation> epilogue;
  auto operation_id = fresh_operation_id(context.program);
  for (auto &input : operation.inputs) {
    const auto *description = context.program.tensor(input);
    const auto converted = fresh_tensor_id(context.program);
    context.program.tensors.push_back(
        {converted, target, ir::TensorRole::Internal, description->dims});
    prologue.push_back(
        {operation_id++, ir::Opcode::Cast, {input}, {converted}, {}});
    input = converted;
  }
  for (auto &output : operation.outputs) {
    const auto *description = context.program.tensor(output);
    const auto converted = fresh_tensor_id(context.program);
    context.program.tensors.push_back(
        {converted, target, ir::TensorRole::Internal, description->dims});
    epilogue.push_back(
        {operation_id++, ir::Opcode::Cast, {converted}, {output}, {}});
    output = converted;
  }
  // Implementation selections that depend on storage dtype are reset rather
  // than silently carried into a precision they are not defined for.
  if (operation.opcode == ir::Opcode::Linear &&
      operation.u64(ir::AttrKey::Implementation, 1U) == 2U &&
      target != ir::DType::F32)
    set_attribute(operation, ir::AttrKey::Implementation,
                  ir::Attribute::u64(ir::AttrKey::Implementation, 1U));
  if (operation.opcode == ir::Opcode::Attention &&
      operation.u64(ir::AttrKey::Implementation, 1U) == 2U &&
      target == ir::DType::F32)
    set_attribute(operation, ir::AttrKey::Implementation,
                  ir::Attribute::u64(ir::AttrKey::Implementation, 1U));
  std::vector<ir::Operation> replacement = std::move(prologue);
  replacement.push_back(operation);
  replacement.insert(replacement.end(), epilogue.begin(), epilogue.end());
  context.program.operations.erase(context.program.operations.begin() +
                                   static_cast<std::ptrdiff_t>(index));
  context.program.operations.insert(
      context.program.operations.begin() + static_cast<std::ptrdiff_t>(index),
      replacement.begin(), replacement.end());
}

void apply_quantize_constant_weights(const Transform &transform,
                                     RewriteContext &context) {
  expect_parameters(transform, 3U);
  const auto bit_width = static_cast<std::uint32_t>(transform.parameters[0]);
  const auto group_size = transform.parameters[1];
  const auto correction = transform.parameters[2] == 0U
                              ? compiler::Int4Correction::None
                              : compiler::Int4Correction::OneOutlier;
  auto rewrite = compiler::rewrite_lowbit_weights(context.program, bit_width,
                                                  group_size, correction);
  if (rewrite.entries.empty())
    fail("quantize_constant_weights found no admissible constant weight");
  auto bindings = context.bindings;
  for (const auto &entry : rewrite.entries) {
    const auto source = bindings.find(entry.source_tensor_id);
    if (source == bindings.end())
      fail("quantize_constant_weights requires a bound weight for tensor " +
           std::to_string(entry.source_tensor_id));
    const auto quantized = compiler::quantize_lowbit_weight(
        source->second, bit_width, group_size, correction);
    bindings.erase(entry.source_tensor_id);
    bindings.insert_or_assign(entry.packed_tensor_id, quantized.packed);
    bindings.insert_or_assign(entry.scales_tensor_id, quantized.scales);
    if (entry.column_scales_tensor_id != 0U)
      bindings.insert_or_assign(entry.column_scales_tensor_id,
                                quantized.column_scales);
    if (entry.outlier_indices_tensor_id != 0U) {
      bindings.insert_or_assign(entry.outlier_indices_tensor_id,
                                quantized.outlier_indices);
      bindings.insert_or_assign(entry.outlier_residuals_tensor_id,
                                quantized.outlier_residuals);
    }
  }
  context.program = std::move(rewrite.program);
  context.bindings = std::move(bindings);
}

void apply_set_constant_residency(const Transform &transform,
                                  RewriteContext &context) {
  expect_parameters(transform, 1U);
  const auto streamed = transform.parameters[0] != 0U;
  auto tensors = transform.tensors;
  if (tensors.empty()) {
    for (const auto &tensor : context.program.tensors) {
      if (tensor.has_role(ir::TensorRole::Constant) &&
          tensor.has_role(ir::TensorRole::Streamed) != streamed)
        tensors.push_back(tensor.id);
    }
    if (tensors.empty())
      fail("set_constant_residency would not change any constant");
  }
  for (const auto id : tensors) {
    auto &tensor = mutable_tensor(context.program, id);
    if (!tensor.has_role(ir::TensorRole::Constant))
      fail("set_constant_residency names tensor " + std::to_string(id) +
           " which is not a constant");
    if (streamed)
      tensor.roles |= static_cast<std::uint32_t>(ir::TensorRole::Streamed);
    else
      tensor.roles &= ~static_cast<std::uint32_t>(ir::TensorRole::Streamed);
  }
}

void apply_set_prefetch_distance(const Transform &transform,
                                 RewriteContext &context) {
  expect_parameters(transform, 1U);
  if (transform.parameters[0] == context.prefetch_distance)
    fail("set_prefetch_distance would not change the streaming policy");
  context.prefetch_distance = transform.parameters[0];
}

} // namespace

void apply(const Transform &transform, RewriteContext &context) {
  switch (transform.kind) {
  case TransformKind::FoldConstantSubgraph:
    apply_fold_constant_subgraph(transform, context);
    return;
  case TransformKind::EliminateDeadOperations:
    apply_eliminate_dead_operations(transform, context);
    return;
  case TransformKind::CommonSubexpression:
    apply_common_subexpression(transform, context);
    return;
  case TransformKind::FuseLinearBias:
    apply_fuse_linear_bias(transform, context);
    return;
  case TransformKind::FuseQkvProjection:
    apply_fuse_qkv_projection(transform, context);
    return;
  case TransformKind::SplitQkvProjection:
    apply_split_qkv_projection(transform, context);
    return;
  case TransformKind::ElideCastRoundTrip:
    apply_elide_cast_round_trip(transform, context);
    return;
  case TransformKind::RematerializeProducer:
    apply_rematerialize_producer(transform, context);
    return;
  case TransformKind::SetBlockSize:
    apply_set_block_size(transform, context);
    return;
  case TransformKind::SetTileShape:
    apply_set_tile_shape(transform, context);
    return;
  case TransformKind::SetLinearImplementation:
    apply_set_implementation(transform, context, ir::Opcode::Linear);
    return;
  case TransformKind::SetAttentionImplementation:
    apply_set_implementation(transform, context, ir::Opcode::Attention);
    return;
  case TransformKind::SetOperationPrecision:
    apply_set_operation_precision(transform, context);
    return;
  case TransformKind::QuantizeConstantWeights:
    apply_quantize_constant_weights(transform, context);
    return;
  case TransformKind::SetConstantResidency:
    apply_set_constant_residency(transform, context);
    return;
  case TransformKind::SetPrefetchDistance:
    apply_set_prefetch_distance(transform, context);
    return;
  case TransformKind::FuseParallelLinearSwiGlu:
    apply_fuse_parallel_linear_swiglu(transform, context);
    return;
  case TransformKind::FuseParallelLinears:
    apply_fuse_parallel_linears(transform, context);
    return;
  }
  fail("unknown transform kind");
}


namespace {

Transform whole_program(TransformKind kind,
                        std::vector<std::uint64_t> parameters = {}) {
  Transform transform;
  transform.kind = kind;
  transform.parameters = std::move(parameters);
  return transform;
}

// True when rewrite_lowbit_weights would accept every constant it selects. The
// rewrite fails hard on an incompatible K dimension, so discovery must not
// propose a combination it cannot legally apply.
bool quantization_admissible(const RewriteContext &context,
                             std::uint32_t bit_width,
                             std::uint64_t group_size) {
  if (group_size < 16U || group_size > 256U || !power_of_two(group_size))
    return false;
  std::size_t candidates = 0U;
  for (const auto &operation : context.program.operations) {
    std::uint32_t weight = 0U;
    if (operation.opcode == ir::Opcode::Linear && operation.inputs.size() >= 2U)
      weight = operation.inputs[1];
    else if (operation.opcode == ir::Opcode::H3DeinterleaveQkvWeight &&
             !operation.inputs.empty())
      weight = operation.inputs[0];
    if (weight == 0U)
      continue;
    const auto *description = context.program.tensor(weight);
    if (!description || !description->has_role(ir::TensorRole::Constant) ||
        !float_dtype(description->dtype) || description->dims.size() != 2U)
      continue;
    if (description->dims[1] % group_size != 0U ||
        (description->dims[1] * bit_width) % 8U != 0U)
      return false;
    if (!context.bindings.contains(weight))
      return false;
    ++candidates;
  }
  return candidates != 0U;
}

std::uint64_t operand_elements(const ir::Program &program,
                               const ir::Operation &operation) {
  std::uint64_t total = 0U;
  for (const auto id : operation.inputs) {
    if (const auto *description = program.tensor(id))
      total += description->element_count();
  }
  for (const auto id : operation.outputs) {
    if (const auto *description = program.tensor(id))
      total += description->element_count();
  }
  return total;
}

bool uniform_float_operation(const ir::Program &program,
                             const ir::Operation &operation, ir::DType &dtype) {
  if (!dtype_uniform(operation.opcode) || pinned_numeric_semantics(operation))
    return false;
  bool first = true;
  const auto inspect = [&](std::uint32_t id) {
    const auto *description = program.tensor(id);
    if (!description || !float_dtype(description->dtype))
      return false;
    if (first) {
      dtype = description->dtype;
      first = false;
      return true;
    }
    return description->dtype == dtype;
  };
  for (const auto id : operation.inputs) {
    if (!inspect(id))
      return false;
  }
  for (const auto id : operation.outputs) {
    if (!inspect(id))
      return false;
  }
  return !first;
}

} // namespace

std::vector<Transform> discover(const RewriteContext &context,
                                const DiscoveryOptions &options) {
  std::vector<Transform> transforms;
  const auto &program = context.program;

  if (options.structural) {
    const auto folds = foldable_sites(context, false);
    if (!folds.empty()) {
      transforms.push_back(
          whole_program(TransformKind::FoldConstantSubgraph, {0U}));
      // With only a handful of sites the search can still afford to explore
      // them one at a time; a fifty-block denoiser gets the whole-program form
      // only, so discovery cannot flood the beam.
      if (folds.size() <= 4U) {
        for (const auto id : folds) {
          Transform single =
              whole_program(TransformKind::FoldConstantSubgraph, {0U});
          single.operations = {id};
          transforms.push_back(single);
        }
      }
    }
    if (options.arithmetic_constant_folding &&
        foldable_sites(context, true).size() > folds.size())
      transforms.push_back(
          whole_program(TransformKind::FoldConstantSubgraph, {1U}));
    if (!dead_operations(program).empty())
      transforms.push_back(whole_program(TransformKind::EliminateDeadOperations));
    if (!common_subexpressions(program).empty())
      transforms.push_back(whole_program(TransformKind::CommonSubexpression));
    if (!linear_bias_sites(program).empty())
      transforms.push_back(whole_program(TransformKind::FuseLinearBias));
    if (!qkv_fusion_sites(program).empty())
      transforms.push_back(whole_program(TransformKind::FuseQkvProjection));
    if (!qkv_split_sites(program).empty())
      transforms.push_back(whole_program(TransformKind::SplitQkvProjection));
    if (!cast_round_trip_sites(program).empty())
      transforms.push_back(whole_program(TransformKind::ElideCastRoundTrip));
    auto remat = rematerialization_sites(program);
    std::stable_sort(remat.begin(), remat.end(),
                     [&](const RematerializationSite &left,
                         const RematerializationSite &right) {
                       const auto *a = find_operation(program, left.producer);
                       const auto *b = find_operation(program, right.producer);
                       const auto left_bytes =
                           program.tensor(a->outputs[0])->byte_count();
                       const auto right_bytes =
                           program.tensor(b->outputs[0])->byte_count();
                       if (left_bytes != right_bytes)
                         return left_bytes > right_bytes;
                       return left.producer < right.producer;
                     });
    for (std::size_t index = 0; index < remat.size() && index < 2U; ++index) {
      Transform transform;
      transform.kind = TransformKind::RematerializeProducer;
      transform.operations = {remat[index].producer, remat[index].consumer};
      transforms.push_back(transform);
    }
  }

  if (options.schedule) {
    for (const auto block : options.block_sizes) {
      if (!power_of_two(block))
        continue;
      OperationIds scope;
      for (const auto &operation : program.operations) {
        const auto *current = operation.find(ir::AttrKey::BlockSize);
        if (!current || current->as_u64() == block)
          continue;
        const auto [low, high] = block_size_range(operation);
        if (block >= low && block <= high)
          scope.push_back(operation.id);
      }
      if (scope.empty())
        continue;
      Transform transform = whole_program(TransformKind::SetBlockSize, {block});
      transform.operations = std::move(scope);
      transforms.push_back(transform);
    }
    for (const auto tile : options.tile_shapes) {
      if (tile == 0U)
        continue;
      OperationIds scope;
      for (const auto &operation : program.operations) {
        if (operation.opcode != ir::Opcode::Linear)
          continue;
        if (operation.u64(ir::AttrKey::TileM, 0U) != tile)
          scope.push_back(operation.id);
      }
      if (scope.empty())
        continue;
      Transform transform =
          whole_program(TransformKind::SetTileShape, {tile, tile, tile});
      transform.operations = std::move(scope);
      transforms.push_back(transform);
    }
  }

  if (options.numeric) {
    for (const std::uint64_t implementation : {1U, 2U}) {
      OperationIds linear_scope;
      OperationIds attention_scope;
      for (const auto &operation : program.operations) {
        if (pinned_numeric_semantics(operation))
          continue;
        if (operation.opcode == ir::Opcode::Linear &&
            operation.u64(ir::AttrKey::Implementation, 1U) != implementation) {
          const auto *input = program.tensor(operation.inputs.at(0));
          if (implementation != 2U ||
              (input && input->dtype == ir::DType::F32))
            linear_scope.push_back(operation.id);
        }
        if (operation.opcode == ir::Opcode::Attention &&
            operation.u64(ir::AttrKey::Implementation, 1U) != implementation) {
          const auto *query = program.tensor(operation.inputs.at(0));
          if (implementation != 2U ||
              (query && (query->dtype == ir::DType::BF16 ||
                         query->dtype == ir::DType::F16)))
            attention_scope.push_back(operation.id);
        }
      }
      if (!linear_scope.empty()) {
        Transform transform = whole_program(
            TransformKind::SetLinearImplementation, {implementation});
        transform.operations = std::move(linear_scope);
        transforms.push_back(transform);
      }
      if (!attention_scope.empty()) {
        Transform transform = whole_program(
            TransformKind::SetAttentionImplementation, {implementation});
        transform.operations = std::move(attention_scope);
        transforms.push_back(transform);
      }
    }

    // Bounded physical-format competition: derive the precision targets and
    // quantization widths from the formats that are legal on the target and
    // implemented as search candidates; everything else is reported through
    // format_statuses() and never proposed.
    std::vector<ir::DType> precisions = options.precisions;
    std::vector<std::uint64_t> quantization_bits = options.quantization_bits;
    if (!options.physical_formats.empty()) {
      precisions.clear();
      quantization_bits.clear();
      for (const auto &status : format_statuses(options)) {
        if (!status.competes)
          continue;
        if (const auto dtype = format_precision(status.format))
          precisions.push_back(*dtype);
        if (const auto bits = format_quantization_bits(status.format))
          quantization_bits.push_back(*bits);
      }
    }

    std::vector<std::pair<std::uint64_t, std::uint32_t>> precision_candidates;
    std::unordered_map<std::uint32_t, ir::DType> current_precision;
    for (const auto &operation : program.operations) {
      ir::DType dtype{};
      if (!uniform_float_operation(program, operation, dtype))
        continue;
      precision_candidates.emplace_back(operand_elements(program, operation),
                                        operation.id);
      current_precision.emplace(operation.id, dtype);
    }
    std::sort(precision_candidates.begin(), precision_candidates.end(),
              [](const auto &left, const auto &right) {
                if (left.first != right.first)
                  return left.first > right.first;
                return left.second < right.second;
              });
    std::size_t emitted = 0U;
    for (const auto &[elements, id] : precision_candidates) {
      (void)elements;
      if (emitted >= options.max_precision_candidates)
        break;
      for (const auto target : precisions) {
        if (target == current_precision.at(id))
          continue;
        if (emitted >= options.max_precision_candidates)
          break;
        Transform transform =
            whole_program(TransformKind::SetOperationPrecision,
                          {static_cast<std::uint64_t>(target)});
        transform.operations = {id};
        transforms.push_back(transform);
        ++emitted;
      }
    }

    for (const auto bits : quantization_bits) {
      if (bits != 4U && bits != 5U)
        continue;
      for (const auto group : options.quantization_group_sizes) {
        if (!quantization_admissible(context, static_cast<std::uint32_t>(bits),
                                     group))
          continue;
        transforms.push_back(whole_program(
            TransformKind::QuantizeConstantWeights, {bits, group, 0U}));
      }
    }
  }

  if (options.memory) {
    for (const std::uint64_t streamed : {0U, 1U}) {
      const auto wants_streaming = streamed != 0U;
      bool changes = false;
      for (const auto &tensor : program.tensors) {
        if (tensor.has_role(ir::TensorRole::Constant) &&
            tensor.has_role(ir::TensorRole::Streamed) != wants_streaming) {
          changes = true;
          break;
        }
      }
      if (changes)
        transforms.push_back(
            whole_program(TransformKind::SetConstantResidency, {streamed}));
    }
    for (const auto distance : options.prefetch_distances) {
      if (distance == context.prefetch_distance)
        continue;
      transforms.push_back(
          whole_program(TransformKind::SetPrefetchDistance, {distance}));
    }
  }

  return transforms;
}

std::vector<FormatStatus> format_statuses(const DiscoveryOptions &options) {
  std::vector<FormatStatus> statuses;
  for (const auto format : options.physical_formats)
    statuses.push_back(physical_format_status(
        format, options.target ? &*options.target : nullptr));
  return statuses;
}

} // namespace dif::opt
