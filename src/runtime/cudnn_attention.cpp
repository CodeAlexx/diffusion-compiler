#include "dif/runtime/cudnn_attention.hpp"

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
  bool tuned{};

  ~Impl() {
    graph.reset();
    if (handle)
      (void)cudnnDestroy(handle);
  }
};

CudnnAttentionPlan::CudnnAttentionPlan(const ir::TensorDesc &query,
                                       std::uint64_t kv_heads, double scale,
                                       bool causal, bool additive_bias,
                                       std::uint32_t heuristic)
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
  if (!(scale > 0.0))
    fail("cuDNN attention scale must be positive");

  check(cudnnCreate(&impl_->handle), "cudnnCreate");
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
                                                 sequence, head_dim};
  const std::vector<std::int64_t> key_value_strides{
      key_value_heads * sequence * head_dim, head_dim,
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
    const std::vector<std::int64_t> bias_dims{batch, 1, sequence, sequence};
    const std::vector<std::int64_t> bias_strides{
        sequence * sequence, sequence * sequence, sequence, 1};
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
  default:
    fail("cuDNN attention heuristic must be A, B, FALLBACK, or autotune");
  }
  auto status = impl_->graph->build(
      impl_->handle, {mode},
      impl_->autotune ? fe::BuildPlanPolicy_t::ALL
                      : fe::BuildPlanPolicy_t::HEURISTICS_CHOICE,
      false);
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

} // namespace dif::runtime
