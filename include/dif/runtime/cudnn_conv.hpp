#pragma once

#include "dif/ir/ir.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace dif::runtime {

class CudnnConv2dPlan {
public:
  CudnnConv2dPlan(const ir::TensorDesc &input,
                  const ir::TensorDesc &weight,
                  const ir::TensorDesc &output, std::uint64_t stride_h,
                  std::uint64_t stride_w, std::uint64_t pad_h,
                  std::uint64_t pad_w, std::uint64_t dilation_h,
                  std::uint64_t dilation_w, std::uint64_t groups,
                  bool biased, std::size_t workspace_limit_bytes,
                  bool deterministic = false);
  ~CudnnConv2dPlan();

  CudnnConv2dPlan(const CudnnConv2dPlan &) = delete;
  CudnnConv2dPlan &operator=(const CudnnConv2dPlan &) = delete;
  CudnnConv2dPlan(CudnnConv2dPlan &&) noexcept;
  CudnnConv2dPlan &operator=(CudnnConv2dPlan &&) noexcept;

  std::size_t workspace_bytes() const;
  void execute(std::uintptr_t input, std::uintptr_t weight,
               std::uintptr_t bias, std::uintptr_t output,
               std::uintptr_t workspace, std::uintptr_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// The three convolution gradients. Each takes the same geometry the forward
// plan took, so a training program describes its convolution once and the
// gradients inherit it; the algorithm is chosen by the same v7 heuristic and
// the math type it returns is applied, as on the forward.
class CudnnConv2dBackwardPlan {
public:
  enum class Kind { Input, Weight, Bias };

  CudnnConv2dBackwardPlan(Kind kind, const ir::TensorDesc &input,
                          const ir::TensorDesc &weight,
                          const ir::TensorDesc &grad_output,
                          std::uint64_t stride_h, std::uint64_t stride_w,
                          std::uint64_t pad_h, std::uint64_t pad_w,
                          std::uint64_t dilation_h, std::uint64_t dilation_w,
                          std::uint64_t groups,
                          std::size_t workspace_limit_bytes,
                          bool deterministic = false);
  ~CudnnConv2dBackwardPlan();

  CudnnConv2dBackwardPlan(const CudnnConv2dBackwardPlan &) = delete;
  CudnnConv2dBackwardPlan &operator=(const CudnnConv2dBackwardPlan &) = delete;
  CudnnConv2dBackwardPlan(CudnnConv2dBackwardPlan &&) noexcept;
  CudnnConv2dBackwardPlan &operator=(CudnnConv2dBackwardPlan &&) noexcept;

  std::size_t workspace_bytes() const;
  // `operand` is the weight for an input gradient and the input for a weight
  // gradient; a bias gradient reads only the output gradient.
  void execute(std::uintptr_t grad_output, std::uintptr_t operand,
               std::uintptr_t gradient, std::uintptr_t workspace,
               std::uintptr_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

class CudnnConv3dPlan {
public:
  CudnnConv3dPlan(const ir::TensorDesc &input,
                  const ir::TensorDesc &weight,
                  const ir::TensorDesc &output, std::uint64_t stride_t,
                  std::uint64_t stride_h, std::uint64_t stride_w,
                  std::uint64_t pad_t, std::uint64_t pad_h,
                  std::uint64_t pad_w, std::uint64_t dilation_t,
                  std::uint64_t dilation_h, std::uint64_t dilation_w,
                  std::uint64_t groups, bool biased,
                  std::size_t workspace_limit_bytes,
                  bool deterministic = false);
  ~CudnnConv3dPlan();

  CudnnConv3dPlan(const CudnnConv3dPlan &) = delete;
  CudnnConv3dPlan &operator=(const CudnnConv3dPlan &) = delete;
  CudnnConv3dPlan(CudnnConv3dPlan &&) noexcept;
  CudnnConv3dPlan &operator=(CudnnConv3dPlan &&) noexcept;

  std::size_t workspace_bytes() const;
  void execute(std::uintptr_t input, std::uintptr_t weight,
               std::uintptr_t bias, std::uintptr_t output,
               std::uintptr_t workspace, std::uintptr_t stream);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace dif::runtime
