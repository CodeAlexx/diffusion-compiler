#pragma once

#include "dif/ir/ir.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace dif::runtime {

class CudnnAttentionPlan {
public:
  // kv_heads: number of K/V heads (GQA); must divide the query head count.
  // Pass query.dims[1] for dense attention.
  CudnnAttentionPlan(const ir::TensorDesc &query, std::uint64_t kv_heads,
                     double scale, bool causal, bool additive_bias = false,
                     std::uint32_t heuristic = 0U);
  ~CudnnAttentionPlan();

  CudnnAttentionPlan(const CudnnAttentionPlan &) = delete;
  CudnnAttentionPlan &operator=(const CudnnAttentionPlan &) = delete;
  CudnnAttentionPlan(CudnnAttentionPlan &&) noexcept;
  CudnnAttentionPlan &operator=(CudnnAttentionPlan &&) noexcept;

  std::size_t workspace_bytes() const;
  void execute(std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
               std::uintptr_t additive_bias, std::uintptr_t output,
               std::uintptr_t workspace, std::uintptr_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// cuDNN SDPA backward for AttentionBackward implementation 2. Consumes the
// program's saved logsumexp (F32 [S,H], natural log of the scaled scores,
// exactly cuDNN's "stats") and the forward output, and writes dQ, dK, dV.
// heuristic 4 requests cuDNN's deterministic algorithms and fails closed when
// none is supported for the geometry.
class CudnnAttentionBackwardPlan {
public:
  CudnnAttentionBackwardPlan(const ir::TensorDesc &query,
                             std::uint64_t kv_heads, double scale,
                             bool causal, std::uint32_t heuristic = 0U);
  ~CudnnAttentionBackwardPlan();
  CudnnAttentionBackwardPlan(const CudnnAttentionBackwardPlan &) = delete;
  CudnnAttentionBackwardPlan &
  operator=(const CudnnAttentionBackwardPlan &) = delete;

  std::size_t workspace_bytes() const;
  void execute(std::uintptr_t query, std::uintptr_t key, std::uintptr_t value,
               std::uintptr_t output, std::uintptr_t grad_output,
               std::uintptr_t logsumexp, std::uintptr_t grad_query,
               std::uintptr_t grad_key, std::uintptr_t grad_value,
               std::uintptr_t workspace, std::uintptr_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace dif::runtime
