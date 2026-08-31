#include "dif/opt/optimizer.hpp"

#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <set>
#include <span>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace dif::opt {
namespace {

bool same_target(const Transformation &left, const Transformation &right) {
  if (left.kind == TransformationKind::SetStreamPrefetchDistance &&
      right.kind == TransformationKind::SetStreamPrefetchDistance)
    return true;
  return left.kind == right.kind && left.target_id == right.target_id &&
         (left.kind == TransformationKind::SetTensorResidency ||
          left.attribute_key == right.attribute_key);
}

bool transformation_less(const Transformation &left,
                         const Transformation &right) {
  return std::tie(left.kind, left.target_id, left.attribute_key, left.value) <
         std::tie(right.kind, right.target_id, right.attribute_key,
                  right.value);
}

std::vector<Transformation>
normalized_transformations(const Recipe &recipe) {
  auto result = recipe.transformations;
  std::sort(result.begin(), result.end(), transformation_less);
  for (std::size_t index = 1U; index < result.size(); ++index) {
    if (same_target(result[index - 1U], result[index]))
      fail("optimization recipe sets one target more than once");
  }
  return result;
}

void verify_pass_name(const std::string &name) {
  if (name.empty())
    fail("optimization recipe pass name is empty");
  for (const auto character : name) {
    const auto value = static_cast<unsigned char>(character);
    if (!std::isalnum(value) && character != '-' && character != '_' &&
        character != '.' && character != '+')
      fail("optimization recipe pass name contains an invalid character");
  }
}

Recipe combine_recipes(const Recipe &left, const Recipe &right) {
  Recipe combined = left;
  combined.passes.insert(combined.passes.end(), right.passes.begin(),
                         right.passes.end());
  for (const auto &transformation : right.transformations) {
    const auto found = std::find_if(
        combined.transformations.begin(), combined.transformations.end(),
        [&](const Transformation &candidate) {
          return same_target(candidate, transformation);
        });
    if (found == combined.transformations.end())
      combined.transformations.push_back(transformation);
    else
      *found = transformation;
  }
  return combined;
}

ir::TensorDesc *find_tensor(ir::Program &program, std::uint32_t id) {
  const auto found = std::find_if(
      program.tensors.begin(), program.tensors.end(),
      [id](const ir::TensorDesc &tensor) { return tensor.id == id; });
  return found == program.tensors.end() ? nullptr : &*found;
}

ir::Operation *find_operation(ir::Program &program, std::uint32_t id) {
  const auto found = std::find_if(
      program.operations.begin(), program.operations.end(),
      [id](const ir::Operation &operation) { return operation.id == id; });
  return found == program.operations.end() ? nullptr : &*found;
}

std::uint32_t next_tensor_id(const ir::Program &program) {
  std::uint32_t maximum = 0U;
  for (const auto &tensor : program.tensors)
    maximum = std::max(maximum, tensor.id);
  if (maximum == std::numeric_limits<std::uint32_t>::max())
    fail("optimization rewrite tensor id space is exhausted");
  return maximum + 1U;
}

std::uint32_t next_operation_id(const ir::Program &program) {
  std::uint32_t maximum = 0U;
  for (const auto &operation : program.operations)
    maximum = std::max(maximum, operation.id);
  if (maximum == std::numeric_limits<std::uint32_t>::max())
    fail("optimization rewrite operation id space is exhausted");
  return maximum + 1U;
}

void split_residual_gate(ir::Program &program, std::uint32_t operation_id) {
  const auto operation_it = std::find_if(
      program.operations.begin(), program.operations.end(),
      [operation_id](const ir::Operation &operation) {
        return operation.id == operation_id;
      });
  if (operation_it == program.operations.end() ||
      operation_it->opcode != ir::Opcode::ResidualGate)
    fail("split rewrite requires a residual_gate target");
  const auto source = *operation_it;
  if (source.inputs.size() != 3U || source.outputs.size() != 1U)
    fail("split residual_gate target has invalid arity");
  const auto *output = program.tensor(source.outputs.front());
  if (!output)
    fail("split residual_gate output tensor is missing");

  const auto intermediate_id = next_tensor_id(program);
  const auto add_operation_id = next_operation_id(program);
  program.tensors.push_back(
      {intermediate_id, output->dtype, ir::TensorRole::Internal, output->dims});

  ir::Operation multiply{source.id,
                         ir::Opcode::Multiply,
                         {source.inputs[1], source.inputs[2]},
                         {intermediate_id},
                         source.attributes};
  ir::Operation add{add_operation_id,
                    ir::Opcode::Add,
                    {source.inputs[0], intermediate_id},
                    source.outputs,
                    source.attributes};
  const auto index = static_cast<std::size_t>(
      std::distance(program.operations.begin(), operation_it));
  program.operations[index] = std::move(multiply);
  program.operations.insert(program.operations.begin() +
                                static_cast<std::ptrdiff_t>(index + 1U),
                            std::move(add));
}

bool can_fuse_multiply_add(const ir::Program &program,
                           std::uint32_t add_operation_id) {
  const auto add_it = std::find_if(
      program.operations.begin(), program.operations.end(),
      [add_operation_id](const ir::Operation &operation) {
        return operation.id == add_operation_id;
      });
  if (add_it == program.operations.end() || add_it->opcode != ir::Opcode::Add ||
      add_it->inputs.size() != 2U || add_it->outputs.size() != 1U)
    return false;
  for (const auto candidate_tensor : add_it->inputs) {
    const auto producer = std::find_if(
        program.operations.begin(), add_it,
        [candidate_tensor](const ir::Operation &operation) {
          return operation.opcode == ir::Opcode::Multiply &&
                 operation.outputs.size() == 1U &&
                 operation.outputs.front() == candidate_tensor;
        });
    if (producer == add_it)
      continue;
    const auto *intermediate = program.tensor(candidate_tensor);
    if (!intermediate || intermediate->roles != ir::TensorRole::Internal)
      continue;
    std::uint64_t uses = 0U;
    for (const auto &operation : program.operations)
      uses += static_cast<std::uint64_t>(std::count(
          operation.inputs.begin(), operation.inputs.end(), candidate_tensor));
    if (uses == 1U)
      return true;
  }
  return false;
}

void fuse_multiply_add(ir::Program &program, std::uint32_t add_operation_id) {
  auto add_it = std::find_if(
      program.operations.begin(), program.operations.end(),
      [add_operation_id](const ir::Operation &operation) {
        return operation.id == add_operation_id;
      });
  if (add_it == program.operations.end() || !can_fuse_multiply_add(
                                                program, add_operation_id))
    fail("fuse rewrite requires an exclusive multiply-to-add region");

  for (std::size_t input_index = 0U; input_index < 2U; ++input_index) {
    const auto intermediate_id = add_it->inputs[input_index];
    auto multiply_it = std::find_if(
        program.operations.begin(), add_it,
        [intermediate_id](const ir::Operation &operation) {
          return operation.opcode == ir::Opcode::Multiply &&
                 operation.outputs.size() == 1U &&
                 operation.outputs.front() == intermediate_id;
        });
    if (multiply_it == add_it)
      continue;
    const auto *intermediate = program.tensor(intermediate_id);
    if (!intermediate || intermediate->roles != ir::TensorRole::Internal)
      continue;
    std::uint64_t uses = 0U;
    for (const auto &operation : program.operations)
      uses += static_cast<std::uint64_t>(std::count(
          operation.inputs.begin(), operation.inputs.end(), intermediate_id));
    if (uses != 1U)
      continue;

    const auto residual_id = add_it->inputs[1U - input_index];
    const auto multiply_inputs = multiply_it->inputs;
    add_it->opcode = ir::Opcode::ResidualGate;
    add_it->inputs = {residual_id, multiply_inputs[0], multiply_inputs[1]};
    program.operations.erase(multiply_it);
    program.tensors.erase(std::remove_if(
                              program.tensors.begin(), program.tensors.end(),
                              [intermediate_id](const ir::TensorDesc &tensor) {
                                return tensor.id == intermediate_id;
                              }),
                          program.tensors.end());
    return;
  }
  fail("fuse rewrite lost its multiply-to-add region");
}

void checked_add(std::uint64_t &target, std::uint64_t value,
                 const char *label) {
  if (value > std::numeric_limits<std::uint64_t>::max() - target)
    fail(std::string(label) + " overflows");
  target += value;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               const char *label) {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left)
    fail(std::string(label) + " overflows");
  return left * right;
}

struct RankedWeight {
  std::uint32_t tensor_id{};
  std::uint64_t uses{};
  std::uint64_t bytes{};
};

} // namespace

Transformation Transformation::make_resident(std::uint32_t tensor_id) {
  return {TransformationKind::SetTensorResidency, tensor_id, {}, 1U};
}

Transformation Transformation::make_streamed(std::uint32_t tensor_id) {
  return {TransformationKind::SetTensorResidency, tensor_id, {}, 0U};
}

Transformation Transformation::set_u64(std::uint32_t operation_id,
                                        ir::AttrKey key,
                                        std::uint64_t value) {
  return {TransformationKind::SetOperationU64Attribute, operation_id, key,
          value};
}

Transformation
Transformation::split_residual_gate(std::uint32_t operation_id) {
  return {TransformationKind::SplitResidualGate, operation_id, {}, 0U};
}

Transformation Transformation::fuse_multiply_add(
    std::uint32_t operation_id) {
  return {TransformationKind::FuseMultiplyAdd, operation_id, {}, 0U};
}

Transformation Transformation::set_stream_prefetch_distance(
    std::uint64_t distance) {
  return {TransformationKind::SetStreamPrefetchDistance, 0U, {}, distance};
}

Transformation Transformation::set_recompute_candidate(
    std::uint32_t tensor_id, bool enabled) {
  return {TransformationKind::SetTensorRecomputeCandidate, tensor_id, {},
          enabled ? 1U : 0U};
}

Transformation Transformation::set_tensor_dtype(std::uint32_t tensor_id,
                                                 ir::DType dtype) {
  return {TransformationKind::SetTensorDType, tensor_id, {},
          static_cast<std::uint64_t>(dtype)};
}

std::string Recipe::canonical_text() const {
  std::ostringstream output;
  output << "dif-optimization-recipe-v1\n";
  for (const auto &pass : passes) {
    verify_pass_name(pass);
    output << "pass=" << pass << "\n";
  }
  for (const auto &transformation : normalized_transformations(*this)) {
    switch (transformation.kind) {
    case TransformationKind::SetTensorResidency:
      if (transformation.target_id == 0U || transformation.value > 1U)
        fail("optimization recipe has invalid tensor residency transform");
      output << "residency tensor=" << transformation.target_id
             << " resident=" << transformation.value << "\n";
      break;
    case TransformationKind::SetOperationU64Attribute:
      if (transformation.target_id == 0U)
        fail("optimization recipe has invalid operation attribute target");
      output << "u64 operation=" << transformation.target_id
             << " key=" << static_cast<std::uint32_t>(
                                transformation.attribute_key)
             << " value=" << transformation.value << "\n";
      break;
    case TransformationKind::SplitResidualGate:
      if (transformation.target_id == 0U || transformation.value != 0U)
        fail("optimization recipe has invalid split target");
      output << "split residual_gate=" << transformation.target_id << "\n";
      break;
    case TransformationKind::FuseMultiplyAdd:
      if (transformation.target_id == 0U || transformation.value != 0U)
        fail("optimization recipe has invalid fusion target");
      output << "fuse multiply_add=" << transformation.target_id << "\n";
      break;
    case TransformationKind::SetStreamPrefetchDistance:
      if (transformation.target_id != 0U || transformation.value > 1U)
        fail("optimization recipe has invalid prefetch distance");
      output << "policy stream_prefetch_distance=" << transformation.value
             << "\n";
      break;
    case TransformationKind::SetTensorRecomputeCandidate:
      if (transformation.target_id == 0U || transformation.value > 1U)
        fail("optimization recipe has invalid recomputation target");
      output << "recompute tensor=" << transformation.target_id
             << " enabled=" << transformation.value << "\n";
      break;
    case TransformationKind::SetTensorDType:
      if (transformation.target_id == 0U ||
          (transformation.value !=
               static_cast<std::uint64_t>(ir::DType::F32) &&
           transformation.value !=
               static_cast<std::uint64_t>(ir::DType::BF16) &&
           transformation.value !=
               static_cast<std::uint64_t>(ir::DType::F16)))
        fail("optimization recipe has invalid tensor precision");
      output << "dtype tensor=" << transformation.target_id
             << " value=" << transformation.value << "\n";
      break;
    default:
      fail("optimization recipe has unknown transformation kind");
    }
  }
  return output.str();
}

Sha256Digest Recipe::fingerprint() const {
  const auto text = canonical_text();
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(text.data()), text.size()));
}

ir::Program apply_recipe(const ir::Program &program, const Recipe &recipe) {
  ir::verify(program);
  const auto transformations = normalized_transformations(recipe);
  for (const auto &pass : recipe.passes)
    verify_pass_name(pass);

  auto result = program;
  for (const auto &transformation : transformations) {
    switch (transformation.kind) {
    case TransformationKind::SetTensorResidency: {
      if (transformation.value > 1U)
        fail("tensor residency transform value must be zero or one");
      auto *tensor = find_tensor(result, transformation.target_id);
      if (!tensor)
        fail("tensor residency transform references a missing tensor");
      if (!tensor->has_role(ir::TensorRole::Constant))
        fail("tensor residency transform requires an immutable constant");
      if (transformation.value == 1U)
        tensor->roles &= ~static_cast<std::uint32_t>(ir::TensorRole::Streamed);
      else
        tensor->roles |= static_cast<std::uint32_t>(ir::TensorRole::Streamed);
      break;
    }
    case TransformationKind::SetOperationU64Attribute: {
      auto *operation = find_operation(result, transformation.target_id);
      if (!operation)
        fail("operation attribute transform references a missing operation");
      auto *attribute = const_cast<ir::Attribute *>(
          operation->find(transformation.attribute_key));
      if (attribute) {
        if (attribute->kind != ir::AttrKind::U64)
          fail("operation U64 transform cannot replace a non-U64 attribute");
        *attribute = ir::Attribute::u64(transformation.attribute_key,
                                        transformation.value);
      } else {
        operation->attributes.push_back(ir::Attribute::u64(
            transformation.attribute_key, transformation.value));
        std::sort(operation->attributes.begin(), operation->attributes.end(),
                  [](const ir::Attribute &left, const ir::Attribute &right) {
                    return static_cast<std::uint32_t>(left.key) <
                           static_cast<std::uint32_t>(right.key);
                  });
      }
      break;
    }
    case TransformationKind::SplitResidualGate:
      if (transformation.value != 0U)
        fail("split residual_gate transform value must be zero");
      split_residual_gate(result, transformation.target_id);
      break;
    case TransformationKind::FuseMultiplyAdd:
      if (transformation.value != 0U)
        fail("fuse multiply_add transform value must be zero");
      fuse_multiply_add(result, transformation.target_id);
      break;
    case TransformationKind::SetStreamPrefetchDistance:
      if (transformation.target_id != 0U || transformation.value > 1U)
        fail("stream prefetch distance must be zero or one");
      break;
    case TransformationKind::SetTensorRecomputeCandidate: {
      if (transformation.value > 1U)
        fail("recomputation candidate value must be zero or one");
      auto *tensor = find_tensor(result, transformation.target_id);
      if (!tensor)
        fail("recomputation transform references a missing tensor");
      if ((tensor->roles & ~static_cast<std::uint32_t>(
                               ir::TensorRole::RecomputeCandidate)) != 0U)
        fail("recomputation transform requires an internal tensor");
      if (transformation.value == 1U)
        tensor->roles |= static_cast<std::uint32_t>(
            ir::TensorRole::RecomputeCandidate);
      else
        tensor->roles &= ~static_cast<std::uint32_t>(
            ir::TensorRole::RecomputeCandidate);
      break;
    }
    case TransformationKind::SetTensorDType: {
      const auto dtype = static_cast<ir::DType>(transformation.value);
      if (dtype != ir::DType::F32 && dtype != ir::DType::BF16 &&
          dtype != ir::DType::F16)
        fail("tensor precision transform requires a float dtype");
      auto *tensor = find_tensor(result, transformation.target_id);
      if (!tensor)
        fail("tensor precision transform references a missing tensor");
      if ((tensor->roles & ~static_cast<std::uint32_t>(
                               ir::TensorRole::RecomputeCandidate)) != 0U)
        fail("tensor precision transform requires an internal tensor");
      const auto writer = std::find_if(
          result.operations.begin(), result.operations.end(),
          [&](const ir::Operation &operation) {
            return operation.opcode == ir::Opcode::Cast &&
                   std::find(operation.outputs.begin(),
                             operation.outputs.end(), tensor->id) !=
                       operation.outputs.end();
          });
      if (writer == result.operations.end())
        fail("tensor precision transform requires a Cast producer");
      bool consumed = false;
      for (const auto &operation : result.operations) {
        if (std::find(operation.inputs.begin(), operation.inputs.end(),
                      tensor->id) == operation.inputs.end())
          continue;
        consumed = true;
        if (operation.opcode != ir::Opcode::Cast)
          fail("tensor precision transform requires Cast-only consumers");
      }
      if (!consumed)
        fail("tensor precision transform requires a Cast consumer");
      tensor->dtype = dtype;
      break;
    }
    default:
      fail("optimization recipe has unknown transformation kind");
    }
  }
  ir::verify(result);
  return result;
}

ExecutionPolicy execution_policy(const Recipe &recipe) {
  ExecutionPolicy policy;
  for (const auto &transformation : normalized_transformations(recipe)) {
    if (transformation.kind ==
        TransformationKind::SetStreamPrefetchDistance) {
      if (transformation.target_id != 0U || transformation.value > 1U)
        fail("stream prefetch distance must be zero or one");
      policy.stream_prefetch_distance = transformation.value;
    }
  }
  return policy;
}

Sha256Digest candidate_identity(const Sha256Digest &program_fingerprint,
                                const ExecutionPolicy &policy) {
  if (policy.stream_prefetch_distance == 1U)
    return program_fingerprint;
  std::ostringstream text;
  text << "dif-optimization-candidate-v1\nprogram="
       << hex_digest(program_fingerprint)
       << "\nstream_prefetch_distance="
       << policy.stream_prefetch_distance << "\n";
  const auto value = text.str();
  return sha256(std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(value.data()), value.size()));
}

Candidate make_candidate(ir::Program program, Recipe recipe) {
  ir::verify(program);
  const auto policy = execution_policy(recipe);
  const auto program_fingerprint = ir::fingerprint(program);
  return {std::move(program), std::move(recipe), policy, program_fingerprint,
          candidate_identity(program_fingerprint, policy)};
}

std::vector<Candidate>
compose_candidates(const ir::Program &program,
                   const std::vector<const Pass *> &passes,
                   const ComposeOptions &options) {
  ir::verify(program);
  if (options.maximum_candidates == 0U)
    fail("optimization candidate limit must be nonzero");

  Candidate identity = make_candidate(program, {});
  std::vector<Candidate> current{std::move(identity)};
  for (const auto *pass : passes) {
    if (!pass)
      fail("optimization pass list contains a null pass");
    verify_pass_name(std::string(pass->name()));
    auto next = current;
    std::set<std::string> fingerprints;
    for (const auto &candidate : next)
      fingerprints.insert(hex_digest(candidate.candidate_fingerprint));

    for (const auto &candidate : current) {
      const auto proposals = pass->propose(candidate.program);
      for (auto proposal : proposals) {
        if (proposal.passes.empty())
          proposal.passes.push_back(std::string(pass->name()));
        const auto transformed = apply_recipe(candidate.program, proposal);
        auto combined = combine_recipes(candidate.recipe, proposal);
        auto next_candidate =
            make_candidate(std::move(transformed), std::move(combined));
        if (!fingerprints
                 .insert(hex_digest(next_candidate.candidate_fingerprint))
                 .second)
          continue;
        if (next.size() == options.maximum_candidates)
          fail("optimization candidate limit exceeded");
        next.push_back(std::move(next_candidate));
      }
    }
    current = std::move(next);
  }
  return current;
}

OperationU64AttributePass::OperationU64AttributePass(
    std::uint32_t operation_id, ir::AttrKey key,
    std::vector<std::uint64_t> values)
    : operation_id_(operation_id), key_(key), values_(std::move(values)) {
  if (operation_id_ == 0U || values_.empty())
    fail("operation attribute pass requires a target and candidate values");
}

std::string_view OperationU64AttributePass::name() const {
  return "operation-u64";
}

std::vector<Recipe>
OperationU64AttributePass::propose(const ir::Program &program) const {
  if (std::none_of(program.operations.begin(), program.operations.end(),
                   [&](const ir::Operation &operation) {
                     return operation.id == operation_id_;
                   }))
    fail("operation attribute pass target is missing");
  std::vector<Recipe> recipes;
  recipes.reserve(values_.size());
  for (const auto value : values_)
    recipes.push_back({{std::string(name())},
                       {Transformation::set_u64(operation_id_, key_, value)}});
  return recipes;
}

MatchingOperationU64AttributePass::MatchingOperationU64AttributePass(
    std::string pass_name, std::optional<ir::Opcode> opcode, ir::AttrKey key,
    std::vector<std::uint64_t> values, bool require_existing_attribute)
    : pass_name_(std::move(pass_name)), key_(key),
      values_(std::move(values)),
      require_existing_attribute_(require_existing_attribute) {
  if (opcode)
    opcodes_.push_back(*opcode);
  verify_pass_name(pass_name_);
  if (values_.empty())
    fail("matching operation attribute pass requires candidate values");
}

MatchingOperationU64AttributePass::MatchingOperationU64AttributePass(
    std::string pass_name, std::vector<ir::Opcode> opcodes, ir::AttrKey key,
    std::vector<std::uint64_t> values, bool require_existing_attribute)
    : pass_name_(std::move(pass_name)), opcodes_(std::move(opcodes)),
      key_(key), values_(std::move(values)),
      require_existing_attribute_(require_existing_attribute) {
  verify_pass_name(pass_name_);
  if (opcodes_.empty() || values_.empty())
    fail("matching operation-set pass requires opcodes and candidate values");
  std::sort(opcodes_.begin(), opcodes_.end());
  if (std::adjacent_find(opcodes_.begin(), opcodes_.end()) != opcodes_.end())
    fail("matching operation-set pass contains duplicate opcodes");
}

std::string_view MatchingOperationU64AttributePass::name() const {
  return pass_name_;
}

std::vector<Recipe>
MatchingOperationU64AttributePass::propose(
    const ir::Program &program) const {
  std::vector<std::uint32_t> targets;
  for (const auto &operation : program.operations) {
    if (!opcodes_.empty() &&
        std::find(opcodes_.begin(), opcodes_.end(), operation.opcode) ==
            opcodes_.end())
      continue;
    const auto *attribute = operation.find(key_);
    if (require_existing_attribute_ && !attribute)
      continue;
    if (attribute && attribute->kind != ir::AttrKind::U64)
      fail("matching operation U64 pass selected a non-U64 attribute");
    targets.push_back(operation.id);
  }
  if (targets.empty())
    fail("matching operation attribute pass selected no operations");

  std::vector<Recipe> recipes;
  recipes.reserve(values_.size());
  for (const auto value : values_) {
    Recipe recipe{{pass_name_}, {}};
    recipe.transformations.reserve(targets.size());
    for (const auto operation_id : targets)
      recipe.transformations.push_back(
          Transformation::set_u64(operation_id, key_, value));
    recipes.push_back(std::move(recipe));
  }
  return recipes;
}

WeightPlacementResult
place_weights(const ir::Program &program,
              const WeightPlacementOptions &options) {
  ir::verify(program);
  if (options.expected_evaluations == 0U)
    fail("weight placement expected evaluations must be nonzero");

  std::unordered_map<std::uint32_t, std::uint64_t> uses;
  for (const auto &operation : program.operations) {
    std::unordered_set<std::uint32_t> used_by_operation;
    for (const auto tensor_id : operation.inputs) {
      const auto *tensor = program.tensor(tensor_id);
      if (tensor && tensor->has_role(ir::TensorRole::Constant))
        used_by_operation.insert(tensor_id);
    }
    for (const auto tensor_id : used_by_operation)
      ++uses[tensor_id];
  }

  Recipe all_streamed_recipe{{"weight-placement.baseline"}, {}};
  std::vector<RankedWeight> ranked;
  std::uint64_t total_weight_bytes = 0U;
  for (const auto &tensor : program.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant))
      continue;
    checked_add(total_weight_bytes, tensor.byte_count(),
                "total immutable weight bytes");
    all_streamed_recipe.transformations.push_back(
        Transformation::make_streamed(tensor.id));
    const auto use_count = uses[tensor.id];
    if (use_count != 0U)
      ranked.push_back({tensor.id, use_count, tensor.byte_count()});
  }
  if (all_streamed_recipe.transformations.empty())
    fail("weight placement requires at least one immutable constant");

  auto selected = apply_recipe(program, all_streamed_recipe);
  const auto all_streamed_plan = compiler::plan_memory(
      selected, options.alignment, options.stream_prefetch_distance);
  if (all_streamed_plan.total_bytes > options.device_budget_bytes)
    fail("weight placement budget cannot fit the all-streamed memory plan");

  std::sort(ranked.begin(), ranked.end(),
            [](const RankedWeight &left, const RankedWeight &right) {
              if (left.uses != right.uses)
                return left.uses > right.uses;
              if (left.bytes != right.bytes)
                return left.bytes > right.bytes;
              return left.tensor_id < right.tensor_id;
            });
  for (const auto &weight : ranked) {
    const Recipe trial_recipe{{"weight-placement.trial"},
                              {Transformation::make_resident(
                                  weight.tensor_id)}};
    auto trial = apply_recipe(selected, trial_recipe);
    const auto plan = compiler::plan_memory(
        trial, options.alignment, options.stream_prefetch_distance);
    if (plan.total_bytes <= options.device_budget_bytes)
      selected = std::move(trial);
  }

  Recipe final_recipe{{"weight-placement.greedy-v1"}, {}};
  WeightPlacementStats stats;
  stats.device_budget_bytes = options.device_budget_bytes;
  stats.all_streamed_planned_bytes = all_streamed_plan.total_bytes;
  stats.total_weight_bytes = total_weight_bytes;
  stats.expected_evaluations = options.expected_evaluations;
  for (const auto &tensor : selected.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant))
      continue;
    const auto resident = !tensor.has_role(ir::TensorRole::Streamed);
    final_recipe.transformations.push_back(
        resident ? Transformation::make_resident(tensor.id)
                 : Transformation::make_streamed(tensor.id));
    if (resident) {
      checked_add(stats.resident_weight_bytes, tensor.byte_count(),
                  "resident weight bytes");
      ++stats.resident_weights;
    } else {
      checked_add(stats.streamed_weight_bytes, tensor.byte_count(),
                  "streamed weight bytes");
      ++stats.streamed_weights;
    }
  }
  final_recipe.transformations.push_back(
      Transformation::set_stream_prefetch_distance(
          options.stream_prefetch_distance));

  auto final_program = apply_recipe(program, final_recipe);
  stats.planned_device_bytes =
      compiler::plan_memory(final_program, options.alignment,
                            options.stream_prefetch_distance)
          .total_bytes;
  if (stats.planned_device_bytes > options.device_budget_bytes)
    fail("weight placement produced an over-budget candidate");
  if (stats.resident_weight_bytes + stats.streamed_weight_bytes !=
      stats.total_weight_bytes)
    fail("weight placement weight-byte accounting is inconsistent");
  stats.estimated_repeated_transfer_bytes_saved = checked_multiply(
      stats.resident_weight_bytes, options.expected_evaluations - 1U,
      "estimated repeated transfer savings");

  Candidate candidate =
      make_candidate(std::move(final_program), std::move(final_recipe));
  return {std::move(candidate), stats};
}

std::string_view SplitResidualGatePass::name() const {
  return "graph.split-residual-gate";
}

std::vector<Recipe>
SplitResidualGatePass::propose(const ir::Program &program) const {
  std::vector<Recipe> recipes;
  for (const auto &operation : program.operations) {
    if (operation.opcode == ir::Opcode::ResidualGate)
      recipes.push_back({{std::string(name())},
                         {Transformation::split_residual_gate(operation.id)}});
  }
  if (recipes.empty())
    fail("split residual_gate pass selected no operations");
  return recipes;
}

std::string_view FuseMultiplyAddPass::name() const {
  return "graph.fuse-multiply-add";
}

std::vector<Recipe>
FuseMultiplyAddPass::propose(const ir::Program &program) const {
  std::vector<Recipe> recipes;
  for (const auto &operation : program.operations) {
    if (can_fuse_multiply_add(program, operation.id))
      recipes.push_back({{std::string(name())},
                         {Transformation::fuse_multiply_add(operation.id)}});
  }
  if (recipes.empty())
    fail("fuse multiply_add pass selected no regions");
  return recipes;
}

StreamPrefetchPass::StreamPrefetchPass(
    std::vector<std::uint64_t> distances)
    : distances_(std::move(distances)) {
  if (distances_.empty() ||
      std::any_of(distances_.begin(), distances_.end(),
                  [](std::uint64_t value) { return value > 1U; }))
    fail("stream prefetch pass admits only distances zero and one");
}

std::string_view StreamPrefetchPass::name() const {
  return "memory.stream-prefetch-distance";
}

std::vector<Recipe>
StreamPrefetchPass::propose(const ir::Program &) const {
  std::vector<Recipe> recipes;
  recipes.reserve(distances_.size());
  for (const auto distance : distances_)
    recipes.push_back(
        {{std::string(name())},
         {Transformation::set_stream_prefetch_distance(distance)}});
  return recipes;
}

std::string_view RecomputeCandidatePass::name() const {
  return "memory.mark-recompute-candidate";
}

std::vector<Recipe>
RecomputeCandidatePass::propose(const ir::Program &program) const {
  std::vector<Recipe> recipes;
  for (const auto &tensor : program.tensors) {
    if (tensor.roles == ir::TensorRole::Internal)
      recipes.push_back(
          {{std::string(name())},
           {Transformation::set_recompute_candidate(tensor.id, true)}});
  }
  if (recipes.empty())
    fail("recompute candidate pass selected no internal tensors");
  return recipes;
}

CastStoragePrecisionPass::CastStoragePrecisionPass(
    std::vector<ir::DType> dtypes)
    : dtypes_(std::move(dtypes)) {
  if (dtypes_.empty() ||
      std::any_of(dtypes_.begin(), dtypes_.end(), [](ir::DType dtype) {
        return dtype != ir::DType::F32 && dtype != ir::DType::BF16 &&
               dtype != ir::DType::F16;
      }))
    fail("cast storage precision pass requires float dtypes");
}

std::string_view CastStoragePrecisionPass::name() const {
  return "numeric.cast-storage-precision";
}

std::vector<Recipe>
CastStoragePrecisionPass::propose(const ir::Program &program) const {
  std::vector<Recipe> recipes;
  for (const auto &tensor : program.tensors) {
    if ((tensor.roles & ~static_cast<std::uint32_t>(
                            ir::TensorRole::RecomputeCandidate)) != 0U)
      continue;
    const auto writer = std::find_if(
        program.operations.begin(), program.operations.end(),
        [&](const ir::Operation &operation) {
          return operation.opcode == ir::Opcode::Cast &&
                 std::find(operation.outputs.begin(), operation.outputs.end(),
                           tensor.id) != operation.outputs.end();
        });
    if (writer == program.operations.end())
      continue;
    bool consumed = false;
    bool cast_only = true;
    for (const auto &operation : program.operations) {
      if (std::find(operation.inputs.begin(), operation.inputs.end(),
                    tensor.id) == operation.inputs.end())
        continue;
      consumed = true;
      cast_only = cast_only && operation.opcode == ir::Opcode::Cast;
    }
    if (!consumed || !cast_only)
      continue;
    for (const auto dtype : dtypes_)
      recipes.push_back(
          {{std::string(name())},
           {Transformation::set_tensor_dtype(tensor.id, dtype)}});
  }
  if (recipes.empty())
    fail("cast storage precision pass selected no legal boundaries");
  return recipes;
}

WeightPlacementPass::WeightPlacementPass(
    std::vector<WeightPlacementOptions> options)
    : options_(std::move(options)) {
  if (options_.empty())
    fail("weight placement pass requires at least one budget");
}

std::string_view WeightPlacementPass::name() const {
  return "weight-placement";
}

std::vector<Recipe>
WeightPlacementPass::propose(const ir::Program &program) const {
  std::vector<Recipe> recipes;
  recipes.reserve(options_.size());
  for (const auto &options : options_)
    recipes.push_back(place_weights(program, options).candidate.recipe);
  return recipes;
}

} // namespace dif::opt
