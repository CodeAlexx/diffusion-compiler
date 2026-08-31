#pragma once

#include "dif/ir/ir.hpp"
#include "dif/support/sha256.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dif::opt {

enum class TransformationKind : std::uint32_t {
  SetTensorResidency = 1,
  SetOperationU64Attribute = 2,
  SplitResidualGate = 3,
  FuseMultiplyAdd = 4,
  SetStreamPrefetchDistance = 5,
  SetTensorRecomputeCandidate = 6,
  SetTensorDType = 7,
};

struct Transformation {
  TransformationKind kind{};
  std::uint32_t target_id{};
  ir::AttrKey attribute_key{};
  std::uint64_t value{};

  static Transformation make_resident(std::uint32_t tensor_id);
  static Transformation make_streamed(std::uint32_t tensor_id);
  static Transformation set_u64(std::uint32_t operation_id, ir::AttrKey key,
                                std::uint64_t value);
  static Transformation split_residual_gate(std::uint32_t operation_id);
  static Transformation fuse_multiply_add(std::uint32_t operation_id);
  static Transformation set_stream_prefetch_distance(std::uint64_t distance);
  static Transformation set_recompute_candidate(std::uint32_t tensor_id,
                                                bool enabled);
  static Transformation set_tensor_dtype(std::uint32_t tensor_id,
                                         ir::DType dtype);
};

struct Recipe {
  std::vector<std::string> passes;
  std::vector<Transformation> transformations;

  std::string canonical_text() const;
  Sha256Digest fingerprint() const;
};

ir::Program apply_recipe(const ir::Program &program, const Recipe &recipe);

struct ExecutionPolicy {
  // The current portable runtime contract admits serial streaming (0) or one
  // operation of look-ahead (1). Larger distances fail closed until every
  // backend ABI can honor them.
  std::uint64_t stream_prefetch_distance{1U};
};

ExecutionPolicy execution_policy(const Recipe &recipe);

struct Candidate {
  ir::Program program;
  Recipe recipe;
  ExecutionPolicy policy;
  Sha256Digest program_fingerprint{};
  // Identifies the complete executable candidate. It equals the DiffIR
  // fingerprint for the default policy and includes policy otherwise.
  Sha256Digest candidate_fingerprint{};
};

Candidate make_candidate(ir::Program program, Recipe recipe);

class Pass {
public:
  virtual ~Pass() = default;
  virtual std::string_view name() const = 0;
  virtual std::vector<Recipe> propose(const ir::Program &program) const = 0;
};

struct ComposeOptions {
  std::size_t maximum_candidates{1024U};
};

// The identity program is always candidate zero. Each pass is applied to the
// candidates produced so far; verified duplicates are removed by DiffIR
// fingerprint. A limit is a fail-closed search bound, not silent truncation.
std::vector<Candidate>
compose_candidates(const ir::Program &program,
                   const std::vector<const Pass *> &passes,
                   const ComposeOptions &options = {});

class OperationU64AttributePass final : public Pass {
public:
  OperationU64AttributePass(std::uint32_t operation_id, ir::AttrKey key,
                            std::vector<std::uint64_t> values);

  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;

private:
  std::uint32_t operation_id_{};
  ir::AttrKey key_{};
  std::vector<std::uint64_t> values_;
};

// Applies one candidate value to every operation selected by opcode and/or an
// already-present attribute. This is the general schedule/implementation pass
// used by search; it has no model-specific operation knowledge.
class MatchingOperationU64AttributePass final : public Pass {
public:
  MatchingOperationU64AttributePass(
      std::string pass_name, std::optional<ir::Opcode> opcode,
      ir::AttrKey key, std::vector<std::uint64_t> values,
      bool require_existing_attribute);
  MatchingOperationU64AttributePass(
      std::string pass_name, std::vector<ir::Opcode> opcodes,
      ir::AttrKey key, std::vector<std::uint64_t> values,
      bool require_existing_attribute);

  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;

private:
  std::string pass_name_;
  std::vector<ir::Opcode> opcodes_;
  ir::AttrKey key_{};
  std::vector<std::uint64_t> values_;
  bool require_existing_attribute_{};
};

struct WeightPlacementOptions {
  std::uint64_t device_budget_bytes{};
  std::uint64_t expected_evaluations{1U};
  std::uint64_t alignment{256U};
  std::uint64_t stream_prefetch_distance{1U};
};

struct WeightPlacementStats {
  std::uint64_t device_budget_bytes{};
  std::uint64_t all_streamed_planned_bytes{};
  std::uint64_t planned_device_bytes{};
  std::uint64_t total_weight_bytes{};
  std::uint64_t resident_weight_bytes{};
  std::uint64_t streamed_weight_bytes{};
  std::uint64_t estimated_repeated_transfer_bytes_saved{};
  std::uint64_t expected_evaluations{};
  std::uint64_t resident_weights{};
  std::uint64_t streamed_weights{};
};

struct WeightPlacementResult {
  Candidate candidate;
  WeightPlacementStats stats;
};

// This first policy is deterministic greedy placement, not a globally optimal
// knapsack solver. It ranks used immutable constants by use count, then bytes,
// then tensor id, and admits a resident tensor only if the complete DiffIR
// memory plan remains inside the requested budget.
WeightPlacementResult
place_weights(const ir::Program &program,
              const WeightPlacementOptions &options);

class WeightPlacementPass final : public Pass {
public:
  explicit WeightPlacementPass(std::vector<WeightPlacementOptions> options);

  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;

private:
  std::vector<WeightPlacementOptions> options_;
};

// General semantic graph rewrites. ResidualGate is a canonical diffusion
// primitive rather than a model-dialect opcode. Splitting it materializes the
// product as Multiply + Add; fusion recognizes that legal inverse pattern.
// Both preserve the explicit round-at-the-product boundary semantics.
class SplitResidualGatePass final : public Pass {
public:
  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;
};

class FuseMultiplyAddPass final : public Pass {
public:
  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;
};

class StreamPrefetchPass final : public Pass {
public:
  explicit StreamPrefetchPass(std::vector<std::uint64_t> distances);

  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;

private:
  std::vector<std::uint64_t> distances_;
};

class RecomputeCandidatePass final : public Pass {
public:
  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;
};

class CastStoragePrecisionPass final : public Pass {
public:
  explicit CastStoragePrecisionPass(std::vector<ir::DType> dtypes);

  std::string_view name() const override;
  std::vector<Recipe> propose(const ir::Program &program) const override;

private:
  std::vector<ir::DType> dtypes_;
};

} // namespace dif::opt
