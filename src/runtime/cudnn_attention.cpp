#include "dif/runtime/cudnn_attention.hpp"

#include "dif/runtime/cudnn_handle.hpp"
#include "dif/support/error.hpp"

#include <cudnn_frontend.h>
#include <cuda_runtime_api.h>

#include <array>
#include <iostream>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dif::runtime {
namespace {

namespace fe = cudnn_frontend;

constexpr std::int64_t kQueryUid = 1;
constexpr std::int64_t kKeyUid = 2;
constexpr std::int64_t kValueUid = 3;
constexpr std::int64_t kOutputUid = 4;
constexpr std::int64_t kScaleUid = 5;
constexpr std::int64_t kBiasUid = 6;
constexpr std::int64_t kGradOutputUid = 7;
constexpr std::int64_t kStatsUid = 8;
constexpr std::int64_t kGradQueryUid = 9;
constexpr std::int64_t kGradKeyUid = 10;
constexpr std::int64_t kGradValueUid = 11;

void check(cudnnStatus_t status, const char *action) {
  if (status == CUDNN_STATUS_SUCCESS)
    return;
  fail(std::string(action) + ": " + cudnnGetErrorString(status));
}

std::int64_t dimension(std::uint64_t value, const char *label) {
  if (value == 0U ||
      value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
    fail(std::string("cuDNN attention ") + label + " is outside int64 range");
  return static_cast<std::int64_t>(value);
}

} // namespace

struct CudnnAttentionPlan::Impl {
  cudnnHandle_t handle{};
  std::shared_ptr<fe::graph::Graph> graph;
  std::size_t workspace{};
  float scale{};
  bool additive_bias{};
  bool autotune{};
  bool deterministic{};
  bool tuned{};

  ~Impl() {
    graph.reset();
    // The handle is shared and outlives the plan.
  }
};

CudnnAttentionPlan::CudnnAttentionPlan(const ir::TensorDesc &query,
                                       std::uint64_t kv_heads, double scale,
                                       bool causal, bool additive_bias,
                                       std::uint32_t heuristic,
                                       std::uint64_t key_value_sequence)
    : impl_(std::make_unique<Impl>()) {
  if ((query.dtype != ir::DType::BF16 && query.dtype != ir::DType::F16) ||
      (query.dims.size() != 3U && query.dims.size() != 4U))
    fail("cuDNN attention requires BF16 or F16 [S,H,D] or [B,S,H,D]");
  const bool batched = query.dims.size() == 4U;
  const auto batch = dimension(batched ? query.dims[0] : 1U, "batch");
  const auto sequence =
      dimension(query.dims[batched ? 1U : 0U], "sequence");
  const auto heads = dimension(query.dims[batched ? 2U : 1U], "heads");
  const auto head_dim =
      dimension(query.dims[batched ? 3U : 2U], "head dimension");
  // GQA: cudnn_frontend SDPA supports grouped K/V natively via differing
  // head counts on the K/V tensor descriptors (H % KvH == 0).
  const auto key_value_heads = dimension(kv_heads, "kv heads");
  if (key_value_heads > heads || heads % key_value_heads != 0)
    fail("cuDNN attention kv heads must divide the query head count");
  const auto kv_sequence =
      key_value_sequence == 0U ? sequence
                               : dimension(key_value_sequence, "kv sequence");
  if (causal && kv_sequence != sequence)
    fail("cuDNN attention causal masks require equal query and K/V rows");
  if (!(scale > 0.0))
    fail("cuDNN attention scale must be positive");

  // Shared per-thread handle: see cudnn_handle.hpp.
  impl_->handle = shared_cudnn_handle();
  impl_->graph = std::make_shared<fe::graph::Graph>();
  impl_->graph
      ->set_io_data_type(query.dtype == ir::DType::BF16
                             ? fe::DataType_t::BFLOAT16
                             : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  impl_->scale = static_cast<float>(scale);
  impl_->additive_bias = additive_bias;
  const std::vector<std::int64_t> dims{batch, heads, sequence, head_dim};
  // DiffIR stores [B,S,H,D] (or the legacy [S,H,D]). Expose those bytes as
  // the same non-contiguous
  // [B,H,S,D] view that PyTorch creates by permuting a contiguous [B,S,H,D]
  // tensor before SDPA.  K/V use their own (possibly smaller) head count.
  const std::vector<std::int64_t> strides{
      heads * sequence * head_dim, head_dim, heads * head_dim, 1};
  const std::vector<std::int64_t> key_value_dims{batch, key_value_heads,
                                                 kv_sequence, head_dim};
  const std::vector<std::int64_t> key_value_strides{
      key_value_heads * kv_sequence * head_dim, head_dim,
      key_value_heads * head_dim, 1};
  auto tensor = [&](const char *name, std::int64_t uid,
                    const std::vector<std::int64_t> &tensor_dims,
                    const std::vector<std::int64_t> &tensor_strides) {
    return impl_->graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_uid(uid)
                                    .set_dim(tensor_dims)
                                    .set_stride(tensor_strides));
  };
  auto q = tensor("Q", kQueryUid, dims, strides);
  auto k = tensor("K", kKeyUid, key_value_dims, key_value_strides);
  auto v = tensor("V", kValueUid, key_value_dims, key_value_strides);
  auto attention_scale =
      impl_->graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Attn_scale")
                               .set_uid(kScaleUid)
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));
  auto attributes = fe::graph::SDPA_attributes()
                        .set_name("dif_cudnn_sdpa")
                        .set_generate_stats(false)
                        .set_attn_scale(attention_scale);
  if (causal)
    attributes.set_causal_mask(true);
  if (additive_bias) {
    const std::vector<std::int64_t> bias_dims{batch, 1, sequence, kv_sequence};
    const std::vector<std::int64_t> bias_strides{
        sequence * kv_sequence, sequence * kv_sequence, kv_sequence, 1};
    auto bias = tensor("Bias", kBiasUid, bias_dims, bias_strides);
    attributes.set_bias(bias);
  }
  auto [output, stats] = impl_->graph->sdpa(q, k, v, attributes);
  (void)stats;
  output->set_output(true)
      .set_uid(kOutputUid)
      .set_dim(dims)
      .set_stride(strides);

  fe::HeurMode_t mode = fe::HeurMode_t::A;
  switch (heuristic) {
  case 0U:
    mode = fe::HeurMode_t::A;
    break;
  case 1U:
    mode = fe::HeurMode_t::B;
    break;
  case 2U:
    mode = fe::HeurMode_t::FALLBACK;
    break;
  case 3U:
    mode = fe::HeurMode_t::A;
    impl_->autotune = true;
    break;
  case 4U:
    mode = fe::HeurMode_t::A;
    impl_->deterministic = true;
    break;
  default:
    fail("cuDNN attention heuristic must be A, B, FALLBACK, autotune, or deterministic");
  }
  auto status = fe::error_t{fe::error_code_t::OK, ""};
  if (impl_->deterministic) {
    status = impl_->graph->validate();
    if (status.is_good())
      status = impl_->graph->build_operation_graph(impl_->handle);
    if (status.is_good())
      status = impl_->graph->create_execution_plans({mode});
    if (status.is_good())
      impl_->graph->deselect_numeric_notes(
          {fe::NumericalNote_t::NONDETERMINISTIC});
    if (status.is_good())
      status = impl_->graph->check_support(impl_->handle);
    if (status.is_good())
      status = impl_->graph->build_plans(
          impl_->handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false);
  } else {
    status = impl_->graph->build(
        impl_->handle, {mode},
        impl_->autotune ? fe::BuildPlanPolicy_t::ALL
                        : fe::BuildPlanPolicy_t::HEURISTICS_CHOICE,
        false);
  }
  if (!status.is_good())
    fail("cuDNN attention graph build: " + status.get_message());
  std::int64_t workspace = 0;
  if (impl_->autotune) {
    workspace = impl_->graph->get_autotune_workspace_size();
  } else {
    status = impl_->graph->get_workspace_size(workspace);
    if (!status.is_good())
      fail("cuDNN attention workspace query: " + status.get_message());
  }
  if (workspace < 0 ||
      static_cast<std::uint64_t>(workspace) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    fail("cuDNN attention workspace is outside size_t range");
  impl_->workspace = static_cast<std::size_t>(workspace);
}

CudnnAttentionPlan::~CudnnAttentionPlan() = default;
CudnnAttentionPlan::CudnnAttentionPlan(CudnnAttentionPlan &&) noexcept = default;
CudnnAttentionPlan &
CudnnAttentionPlan::operator=(CudnnAttentionPlan &&) noexcept = default;

std::size_t CudnnAttentionPlan::workspace_bytes() const {
  return impl_->workspace;
}

void CudnnAttentionPlan::execute(std::uintptr_t query, std::uintptr_t key,
                                 std::uintptr_t value,
                                 std::uintptr_t additive_bias,
                                 std::uintptr_t output, std::uintptr_t workspace,
                                 std::uintptr_t stream) {
  if (!query || !key || !value || !output)
    fail("cuDNN attention received a null tensor pointer");
  if (impl_->workspace != 0U && !workspace)
    fail("cuDNN attention received a null workspace");
  if (impl_->additive_bias && !additive_bias)
    fail("cuDNN attention received a null additive bias");
  check(cudnnSetStream(impl_->handle, reinterpret_cast<cudaStream_t>(stream)),
        "cudnnSetStream");
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> bindings{
      {kQueryUid, reinterpret_cast<void *>(query)},
      {kKeyUid, reinterpret_cast<void *>(key)},
      {kValueUid, reinterpret_cast<void *>(value)},
      {kOutputUid, reinterpret_cast<void *>(output)},
      {kScaleUid, &impl_->scale},
  };
  if (impl_->additive_bias)
    bindings.emplace(kBiasUid, reinterpret_cast<void *>(additive_bias));
  if (impl_->autotune && !impl_->tuned) {
    auto status = impl_->graph->autotune(
        impl_->handle, bindings, reinterpret_cast<void *>(workspace));
    if (!status.is_good())
      fail("cuDNN attention autotune: " + status.get_message());
    impl_->tuned = true;
    std::string plan_name;
    status = impl_->graph->get_plan_name(plan_name);
    if (!status.is_good())
      fail("cuDNN attention selected-plan query: " + status.get_message());
    std::cerr << "CUDNN_ATTENTION_AUTOTUNE plans="
              << impl_->graph->get_execution_plan_count()
              << " selected=\"" << plan_name << "\""
              << " workspace_bytes=" << impl_->workspace << '\n';
  }
  auto status = impl_->graph->execute(
      impl_->handle, bindings, reinterpret_cast<void *>(workspace));
  if (!status.is_good())
    fail("cuDNN attention execute: " + status.get_message());
}

struct CudnnAttentionBackwardPlan::Impl {
  cudnnHandle_t handle{};
  std::shared_ptr<fe::graph::Graph> graph;
  std::size_t workspace{};
  float scale{};

  ~Impl() {
    graph.reset();
    // The handle is shared and outlives the plan.
  }
};

CudnnAttentionBackwardPlan::CudnnAttentionBackwardPlan(
    const ir::TensorDesc &query, std::uint64_t kv_heads, double scale,
    bool causal, std::uint32_t heuristic)
    : impl_(std::make_unique<Impl>()) {
  if ((query.dtype != ir::DType::BF16 && query.dtype != ir::DType::F16) ||
      query.dims.size() != 3U)
    fail("cuDNN attention backward requires BF16 or F16 [S,H,D]");
  const auto sequence = dimension(query.dims[0], "sequence");
  const auto heads = dimension(query.dims[1], "heads");
  const auto head_dim = dimension(query.dims[2], "head dimension");
  const auto key_value_heads = dimension(kv_heads, "kv heads");
  if (key_value_heads > heads || heads % key_value_heads != 0)
    fail("cuDNN attention backward kv heads must divide the query head count");
  if (!(scale > 0.0))
    fail("cuDNN attention backward scale must be positive");

  // Shared per-thread handle: see cudnn_handle.hpp.
  impl_->handle = shared_cudnn_handle();
  impl_->graph = std::make_shared<fe::graph::Graph>();
  impl_->graph
      ->set_io_data_type(query.dtype == ir::DType::BF16
                             ? fe::DataType_t::BFLOAT16
                             : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);
  impl_->scale = static_cast<float>(scale);
  constexpr std::int64_t batch = 1;
  // Same [B,H,S,D] view over DiffIR's [S,H,D] bytes as the forward plan.
  const std::vector<std::int64_t> dims{batch, heads, sequence, head_dim};
  const std::vector<std::int64_t> strides{
      heads * sequence * head_dim, head_dim, heads * head_dim, 1};
  const std::vector<std::int64_t> key_value_dims{batch, key_value_heads,
                                                 sequence, head_dim};
  const std::vector<std::int64_t> key_value_strides{
      key_value_heads * sequence * head_dim, head_dim,
      key_value_heads * head_dim, 1};
  // cuDNN reads its softmax stats as the packed [B,H,S,1] tensor its own
  // forward writes (strides {H*S, S, 1, 1}); the strides declared here are
  // not honoured for other layouts (measured: cosine 0.998 against the math
  // path with the program's [S,H] strides). The executor hands in a packed
  // transposed copy of the program's F32 [S,H] logsumexp.
  const std::vector<std::int64_t> stats_dims{batch, heads, sequence, 1};
  const std::vector<std::int64_t> stats_strides{heads * sequence, sequence, 1, 1};
  auto tensor = [&](const char *name, std::int64_t uid,
                    const std::vector<std::int64_t> &tensor_dims,
                    const std::vector<std::int64_t> &tensor_strides) {
    return impl_->graph->tensor(fe::graph::Tensor_attributes()
                                    .set_name(name)
                                    .set_uid(uid)
                                    .set_dim(tensor_dims)
                                    .set_stride(tensor_strides));
  };
  auto q = tensor("Q", kQueryUid, dims, strides);
  auto k = tensor("K", kKeyUid, key_value_dims, key_value_strides);
  auto v = tensor("V", kValueUid, key_value_dims, key_value_strides);
  auto o = tensor("O", kOutputUid, dims, strides);
  auto d_o = tensor("dO", kGradOutputUid, dims, strides);
  auto stats = impl_->graph->tensor(fe::graph::Tensor_attributes()
                                        .set_name("Stats")
                                        .set_uid(kStatsUid)
                                        .set_dim(stats_dims)
                                        .set_stride(stats_strides)
                                        .set_data_type(fe::DataType_t::FLOAT));
  auto attention_scale =
      impl_->graph->tensor(fe::graph::Tensor_attributes()
                               .set_name("Attn_scale")
                               .set_uid(kScaleUid)
                               .set_dim({1, 1, 1, 1})
                               .set_stride({1, 1, 1, 1})
                               .set_is_pass_by_value(true)
                               .set_data_type(fe::DataType_t::FLOAT));
  auto attributes = fe::graph::SDPA_backward_attributes()
                        .set_name("dif_cudnn_sdpa_backward")
                        .set_attn_scale(attention_scale);
  if (causal)
    attributes.set_causal_mask(true);
  const bool deterministic = heuristic == 4U;
  if (deterministic)
    attributes.set_deterministic_algorithm(true);
  auto [grad_q, grad_k, grad_v] =
      impl_->graph->sdpa_backward(q, k, v, o, d_o, stats, attributes);
  grad_q->set_output(true).set_uid(kGradQueryUid).set_dim(dims).set_stride(strides);
  grad_k->set_output(true)
      .set_uid(kGradKeyUid)
      .set_dim(key_value_dims)
      .set_stride(key_value_strides);
  grad_v->set_output(true)
      .set_uid(kGradValueUid)
      .set_dim(key_value_dims)
      .set_stride(key_value_strides);

  fe::HeurMode_t mode = fe::HeurMode_t::A;
  switch (heuristic) {
  case 0U:
  case 3U:
  case 4U:
    mode = fe::HeurMode_t::A;
    break;
  case 1U:
    mode = fe::HeurMode_t::B;
    break;
  case 2U:
    mode = fe::HeurMode_t::FALLBACK;
    break;
  default:
    fail("cuDNN attention backward heuristic must be A, B, FALLBACK, autotune, "
         "or deterministic");
  }
  auto status = fe::error_t{fe::error_code_t::OK, ""};
  if (deterministic) {
    status = impl_->graph->validate();
    if (status.is_good())
      status = impl_->graph->build_operation_graph(impl_->handle);
    if (status.is_good())
      status = impl_->graph->create_execution_plans({mode});
    if (status.is_good())
      impl_->graph->deselect_numeric_notes(
          {fe::NumericalNote_t::NONDETERMINISTIC});
    if (status.is_good())
      status = impl_->graph->check_support(impl_->handle);
    if (status.is_good())
      status = impl_->graph->build_plans(
          impl_->handle, fe::BuildPlanPolicy_t::HEURISTICS_CHOICE, false);
  } else {
    status = impl_->graph->build(impl_->handle, {mode},
                                 fe::BuildPlanPolicy_t::HEURISTICS_CHOICE,
                                 false);
  }
  if (!status.is_good())
    fail("cuDNN attention backward graph build: " + status.get_message());
  std::int64_t workspace = 0;
  status = impl_->graph->get_workspace_size(workspace);
  if (!status.is_good())
    fail("cuDNN attention backward workspace query: " + status.get_message());
  if (workspace < 0 ||
      static_cast<std::uint64_t>(workspace) >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
    fail("cuDNN attention backward workspace is outside size_t range");
  impl_->workspace = static_cast<std::size_t>(workspace);
}

CudnnAttentionBackwardPlan::~CudnnAttentionBackwardPlan() = default;

std::size_t CudnnAttentionBackwardPlan::workspace_bytes() const {
  return impl_->workspace;
}

void CudnnAttentionBackwardPlan::execute(
    std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
    std::uintptr_t output, std::uintptr_t grad_output, std::uintptr_t logsumexp,
    std::uintptr_t grad_query, std::uintptr_t grad_key,
    std::uintptr_t grad_value, std::uintptr_t workspace,
    std::uintptr_t stream) {
  if (!query || !key || !value || !output || !grad_output || !logsumexp ||
      !grad_query || !grad_key || !grad_value)
    fail("cuDNN attention backward received a null tensor pointer");
  if (impl_->workspace != 0U && !workspace)
    fail("cuDNN attention backward received a null workspace");
  check(cudnnSetStream(impl_->handle, reinterpret_cast<cudaStream_t>(stream)),
        "cudnnSetStream");
  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void *> bindings{
      {kQueryUid, reinterpret_cast<void *>(query)},
      {kKeyUid, reinterpret_cast<void *>(key)},
      {kValueUid, reinterpret_cast<void *>(value)},
      {kOutputUid, reinterpret_cast<void *>(output)},
      {kGradOutputUid, reinterpret_cast<void *>(grad_output)},
      {kStatsUid, reinterpret_cast<void *>(logsumexp)},
      {kGradQueryUid, reinterpret_cast<void *>(grad_query)},
      {kGradKeyUid, reinterpret_cast<void *>(grad_key)},
      {kGradValueUid, reinterpret_cast<void *>(grad_value)},
      {kScaleUid, &impl_->scale},
  };
  auto status = impl_->graph->execute(
      impl_->handle, bindings, reinterpret_cast<void *>(workspace));
  if (!status.is_good())
    fail("cuDNN attention backward execute: " + status.get_message());
}

} // namespace dif::runtime
