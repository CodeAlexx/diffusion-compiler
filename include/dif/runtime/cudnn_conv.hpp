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
                  bool biased, std::size_t workspace_limit_bytes);
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
                  std::size_t workspace_limit_bytes);
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
