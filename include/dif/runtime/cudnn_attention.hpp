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
                     double scale, bool causal, bool additive_bias = false);
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

} // namespace dif::runtime
