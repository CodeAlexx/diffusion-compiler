#include "dif/runtime/cudnn_conv.hpp"

#include "dif/runtime/cudnn_handle.hpp"
#include "dif/support/error.hpp"

#include <cudnn.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <limits>
#include <string>
#include <utility>

namespace dif::runtime {
namespace {

void check(cudnnStatus_t status, const char *action) {
  if (status == CUDNN_STATUS_SUCCESS)
    return;
  fail(std::string(action) + ": " + cudnnGetErrorString(status));
}

int dimension(std::uint64_t value, const char *label) {
  if (value == 0U ||
      value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    fail(std::string("cuDNN Conv2d ") + label + " is outside int range");
  return static_cast<int>(value);
}

int parameter(std::uint64_t value, const char *label) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
    fail(std::string("cuDNN Conv2d ") + label + " is outside int range");
  return static_cast<int>(value);
}

cudnnDataType_t data_type(ir::DType dtype) {
  if (dtype == ir::DType::F32)
    return CUDNN_DATA_FLOAT;
  if (dtype == ir::DType::BF16)
    return CUDNN_DATA_BFLOAT16;
  if (dtype == ir::DType::F16)
    return CUDNN_DATA_HALF;
  fail("cuDNN Conv2d requires F32, BF16, or F16");
}

} // namespace

struct CudnnConv2dPlan::Impl {
  cudnnHandle_t handle{};
  cudnnTensorDescriptor_t input{};
  cudnnTensorDescriptor_t output{};
  cudnnTensorDescriptor_t bias{};
  cudnnFilterDescriptor_t weight{};
  cudnnConvolutionDescriptor_t convolution{};
  cudnnConvolutionFwdAlgo_t algorithm{};
  std::size_t workspace{};
  bool biased{};
  // Shapes and the chosen algorithm, so a library failure names the
  // convolution instead of only its status.
  std::string description;

  ~Impl() {
    if (convolution)
      (void)cudnnDestroyConvolutionDescriptor(convolution);
    if (weight)
      (void)cudnnDestroyFilterDescriptor(weight);
    if (bias)
      (void)cudnnDestroyTensorDescriptor(bias);
    if (output)
      (void)cudnnDestroyTensorDescriptor(output);
    if (input)
      (void)cudnnDestroyTensorDescriptor(input);
    // The handle is shared and outlives the plan.
  }
};

CudnnConv2dPlan::CudnnConv2dPlan(
    const ir::TensorDesc &input, const ir::TensorDesc &weight,
    const ir::TensorDesc &output, std::uint64_t stride_h,
    std::uint64_t stride_w, std::uint64_t pad_h, std::uint64_t pad_w,
    std::uint64_t dilation_h, std::uint64_t dilation_w,
    std::uint64_t groups, bool biased, std::size_t workspace_limit_bytes,
    bool deterministic)
    : impl_(std::make_unique<Impl>()) {
  if (input.dims.size() != 4U || weight.dims.size() != 4U ||
      output.dims.size() != 4U || input.dtype != weight.dtype ||
      input.dtype != output.dtype)
    fail("cuDNN Conv2d requires matching NCHW/OIHW rank-4 tensors");
  const auto dtype = data_type(input.dtype);
  // Shared per-thread handle: see cudnn_handle.hpp.
  impl_->handle = shared_cudnn_handle();
  check(cudnnCreateTensorDescriptor(&impl_->input),
        "cudnnCreateTensorDescriptor Conv2d input");
  check(cudnnCreateTensorDescriptor(&impl_->output),
        "cudnnCreateTensorDescriptor Conv2d output");
  check(cudnnCreateTensorDescriptor(&impl_->bias),
        "cudnnCreateTensorDescriptor Conv2d bias");
  check(cudnnCreateFilterDescriptor(&impl_->weight),
        "cudnnCreateFilterDescriptor Conv2d");
  check(cudnnCreateConvolutionDescriptor(&impl_->convolution),
        "cudnnCreateConvolutionDescriptor Conv2d");
  check(cudnnSetTensor4dDescriptor(
            impl_->input, CUDNN_TENSOR_NCHW, dtype,
            dimension(input.dims[0], "batch"),
            dimension(input.dims[1], "input channels"),
            dimension(input.dims[2], "input height"),
            dimension(input.dims[3], "input width")),
        "cudnnSetTensor4dDescriptor Conv2d input");
  check(cudnnSetTensor4dDescriptor(
            impl_->output, CUDNN_TENSOR_NCHW, dtype,
            dimension(output.dims[0], "output batch"),
            dimension(output.dims[1], "output channels"),
            dimension(output.dims[2], "output height"),
            dimension(output.dims[3], "output width")),
        "cudnnSetTensor4dDescriptor Conv2d output");
  check(cudnnSetTensor4dDescriptor(
            impl_->bias, CUDNN_TENSOR_NCHW, dtype, 1,
            dimension(output.dims[1], "bias channels"), 1, 1),
        "cudnnSetTensor4dDescriptor Conv2d bias");
  check(cudnnSetFilter4dDescriptor(
            impl_->weight, dtype, CUDNN_TENSOR_NCHW,
            dimension(weight.dims[0], "filter outputs"),
            dimension(weight.dims[1], "filter inputs"),
            dimension(weight.dims[2], "filter height"),
            dimension(weight.dims[3], "filter width")),
        "cudnnSetFilter4dDescriptor Conv2d");
  check(cudnnSetConvolution2dDescriptor(
            impl_->convolution, parameter(pad_h, "padding height"),
            parameter(pad_w, "padding width"),
            dimension(stride_h, "stride height"),
            dimension(stride_w, "stride width"),
            dimension(dilation_h, "dilation height"),
            dimension(dilation_w, "dilation width"),
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor Conv2d");
  check(cudnnSetConvolutionGroupCount(impl_->convolution,
                                      dimension(groups, "groups")),
        "cudnnSetConvolutionGroupCount Conv2d");
  if (input.dtype != ir::DType::F32)
    check(cudnnSetConvolutionMathType(impl_->convolution,
                                      CUDNN_TENSOR_OP_MATH),
          "cudnnSetConvolutionMathType Conv2d");

  std::array<cudnnConvolutionFwdAlgoPerf_t, 8> candidates{};
  int returned = 0;
  check(cudnnGetConvolutionForwardAlgorithm_v7(
            impl_->handle, impl_->input, impl_->weight, impl_->convolution,
            impl_->output, static_cast<int>(candidates.size()), &returned,
            candidates.data()),
        "cudnnGetConvolutionForwardAlgorithm_v7");
  const auto selected = std::find_if(
      candidates.begin(), candidates.begin() + returned,
      [&](const cudnnConvolutionFwdAlgoPerf_t &candidate) {
        return candidate.status == CUDNN_STATUS_SUCCESS &&
               candidate.memory <= workspace_limit_bytes &&
               (!deterministic ||
                candidate.determinism == CUDNN_DETERMINISTIC);
      });
  if (selected == candidates.begin() + returned)
    fail(deterministic
             ? "cuDNN Conv2d has no deterministic algorithm within its workspace limit"
             : "cuDNN Conv2d has no algorithm within its workspace limit");
  impl_->algorithm = selected->algo;
  // The v7 heuristic returns the math type that belongs with each
  // algorithm. Applying the algorithm while leaving the descriptor on the
  // math type used for the query is a parameter mismatch: cuDNN rejects
  // the launch (a 256-channel 3x3 convolution at 256x256 in BF16 chose an
  // algorithm wanting default math and failed with BAD_PARAM).
  check(cudnnSetConvolutionMathType(impl_->convolution, selected->mathType),
        "cudnnSetConvolutionMathType Conv2d (selected algorithm)");
  impl_->workspace = selected->memory;
  impl_->biased = biased;
  const auto shape = [](const std::vector<std::uint64_t> &dims) {
    std::string text = "[";
    for (std::size_t index = 0; index < dims.size(); ++index)
      text += (index ? "," : "") + std::to_string(dims[index]);
    return text + "]";
  };
  impl_->description =
      " input=" + shape(input.dims) + " weight=" + shape(weight.dims) +
      " output=" + shape(output.dims) + " stride=" + std::to_string(stride_h) +
      "x" + std::to_string(stride_w) + " pad=" + std::to_string(pad_h) + "x" +
      std::to_string(pad_w) + " groups=" + std::to_string(groups) +
      " algorithm=" + std::to_string(static_cast<int>(impl_->algorithm)) +
      " workspace=" + std::to_string(impl_->workspace);
}

CudnnConv2dPlan::~CudnnConv2dPlan() = default;
CudnnConv2dPlan::CudnnConv2dPlan(CudnnConv2dPlan &&) noexcept = default;
CudnnConv2dPlan &
CudnnConv2dPlan::operator=(CudnnConv2dPlan &&) noexcept = default;

std::size_t CudnnConv2dPlan::workspace_bytes() const {
  return impl_->workspace;
}

void CudnnConv2dPlan::execute(std::uintptr_t input, std::uintptr_t weight,
                              std::uintptr_t bias, std::uintptr_t output,
                              std::uintptr_t workspace,
                              std::uintptr_t stream) {
  check(cudnnSetStream(impl_->handle, reinterpret_cast<cudaStream_t>(stream)),
        "cudnnSetStream Conv2d");
  constexpr float one = 1.0F;
  constexpr float zero = 0.0F;
  const auto status = cudnnConvolutionForward(
      impl_->handle, &one, impl_->input,
      reinterpret_cast<const void *>(input), impl_->weight,
      reinterpret_cast<const void *>(weight), impl_->convolution,
      impl_->algorithm, reinterpret_cast<void *>(workspace), impl_->workspace,
      &zero, impl_->output, reinterpret_cast<void *>(output));
  if (status != CUDNN_STATUS_SUCCESS)
    fail("cudnnConvolutionForward Conv2d" + impl_->description + ": " +
         cudnnGetErrorString(status));
  if (impl_->biased)
    check(cudnnAddTensor(impl_->handle, &one, impl_->bias,
                         reinterpret_cast<const void *>(bias), &one,
                         impl_->output, reinterpret_cast<void *>(output)),
          "cudnnAddTensor Conv2d bias");
}

struct CudnnConv2dBackwardPlan::Impl {
  cudnnHandle_t handle{};
  cudnnTensorDescriptor_t input{};
  cudnnTensorDescriptor_t grad_output{};
  cudnnTensorDescriptor_t bias{};
  cudnnFilterDescriptor_t weight{};
  cudnnConvolutionDescriptor_t convolution{};
  CudnnConv2dBackwardPlan::Kind kind{};
  cudnnConvolutionBwdDataAlgo_t data_algorithm{};
  cudnnConvolutionBwdFilterAlgo_t filter_algorithm{};
  std::size_t workspace{};
  std::string description;

  ~Impl() {
    if (convolution)
      (void)cudnnDestroyConvolutionDescriptor(convolution);
    if (weight)
      (void)cudnnDestroyFilterDescriptor(weight);
    if (bias)
      (void)cudnnDestroyTensorDescriptor(bias);
    if (grad_output)
      (void)cudnnDestroyTensorDescriptor(grad_output);
    if (input)
      (void)cudnnDestroyTensorDescriptor(input);
    // The handle is shared and outlives the plan.
  }
};

CudnnConv2dBackwardPlan::CudnnConv2dBackwardPlan(
    Kind kind, const ir::TensorDesc &input, const ir::TensorDesc &weight,
    const ir::TensorDesc &grad_output, std::uint64_t stride_h,
    std::uint64_t stride_w, std::uint64_t pad_h, std::uint64_t pad_w,
    std::uint64_t dilation_h, std::uint64_t dilation_w, std::uint64_t groups,
    std::size_t workspace_limit_bytes, bool deterministic)
    : impl_(std::make_unique<Impl>()) {
  if (input.dims.size() != 4U || weight.dims.size() != 4U ||
      grad_output.dims.size() != 4U || input.dtype != weight.dtype ||
      input.dtype != grad_output.dtype)
    fail("cuDNN Conv2d gradient requires matching NCHW/OIHW rank-4 tensors");
  const auto dtype = data_type(input.dtype);
  impl_->kind = kind;
  impl_->handle = shared_cudnn_handle();
  check(cudnnCreateTensorDescriptor(&impl_->input),
        "cudnnCreateTensorDescriptor Conv2d gradient input");
  check(cudnnCreateTensorDescriptor(&impl_->grad_output),
        "cudnnCreateTensorDescriptor Conv2d gradient output");
  check(cudnnCreateTensorDescriptor(&impl_->bias),
        "cudnnCreateTensorDescriptor Conv2d gradient bias");
  check(cudnnCreateFilterDescriptor(&impl_->weight),
        "cudnnCreateFilterDescriptor Conv2d gradient");
  check(cudnnCreateConvolutionDescriptor(&impl_->convolution),
        "cudnnCreateConvolutionDescriptor Conv2d gradient");
  check(cudnnSetTensor4dDescriptor(
            impl_->input, CUDNN_TENSOR_NCHW, dtype,
            dimension(input.dims[0], "batch"),
            dimension(input.dims[1], "input channels"),
            dimension(input.dims[2], "input height"),
            dimension(input.dims[3], "input width")),
        "cudnnSetTensor4dDescriptor Conv2d gradient input");
  check(cudnnSetTensor4dDescriptor(
            impl_->grad_output, CUDNN_TENSOR_NCHW, dtype,
            dimension(grad_output.dims[0], "output batch"),
            dimension(grad_output.dims[1], "output channels"),
            dimension(grad_output.dims[2], "output height"),
            dimension(grad_output.dims[3], "output width")),
        "cudnnSetTensor4dDescriptor Conv2d gradient output");
  check(cudnnSetTensor4dDescriptor(
            impl_->bias, CUDNN_TENSOR_NCHW, dtype, 1,
            dimension(grad_output.dims[1], "bias channels"), 1, 1),
        "cudnnSetTensor4dDescriptor Conv2d gradient bias");
  check(cudnnSetFilter4dDescriptor(
            impl_->weight, dtype, CUDNN_TENSOR_NCHW,
            dimension(weight.dims[0], "filter outputs"),
            dimension(weight.dims[1], "filter inputs"),
            dimension(weight.dims[2], "filter height"),
            dimension(weight.dims[3], "filter width")),
        "cudnnSetFilter4dDescriptor Conv2d gradient");
  check(cudnnSetConvolution2dDescriptor(
            impl_->convolution, parameter(pad_h, "padding height"),
            parameter(pad_w, "padding width"),
            dimension(stride_h, "stride height"),
            dimension(stride_w, "stride width"),
            dimension(dilation_h, "dilation height"),
            dimension(dilation_w, "dilation width"),
            CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor Conv2d gradient");
  check(cudnnSetConvolutionGroupCount(impl_->convolution,
                                      dimension(groups, "groups")),
        "cudnnSetConvolutionGroupCount Conv2d gradient");
  if (input.dtype != ir::DType::F32)
    check(cudnnSetConvolutionMathType(impl_->convolution,
                                      CUDNN_TENSOR_OP_MATH),
          "cudnnSetConvolutionMathType Conv2d gradient");

  const auto shape = [](const std::vector<std::uint64_t> &dims) {
    std::string text = "[";
    for (std::size_t index = 0; index < dims.size(); ++index)
      text += (index ? "," : "") + std::to_string(dims[index]);
    return text + "]";
  };
  impl_->description = " input=" + shape(input.dims) + " weight=" +
                       shape(weight.dims) + " grad_output=" +
                       shape(grad_output.dims);
  if (kind == Kind::Bias) {
    // The bias gradient is a plain channel reduction: no algorithm, no
    // workspace.
    impl_->workspace = 0U;
    return;
  }
  if (kind == Kind::Input) {
    std::array<cudnnConvolutionBwdDataAlgoPerf_t, 8> candidates{};
    int returned = 0;
    check(cudnnGetConvolutionBackwardDataAlgorithm_v7(
              impl_->handle, impl_->weight, impl_->grad_output,
              impl_->convolution, impl_->input,
              static_cast<int>(candidates.size()), &returned,
              candidates.data()),
          "cudnnGetConvolutionBackwardDataAlgorithm_v7");
    const auto selected = std::find_if(
        candidates.begin(), candidates.begin() + returned,
        [&](const cudnnConvolutionBwdDataAlgoPerf_t &candidate) {
          return candidate.status == CUDNN_STATUS_SUCCESS &&
                 candidate.memory <= workspace_limit_bytes &&
                 (!deterministic ||
                  candidate.determinism == CUDNN_DETERMINISTIC);
        });
    if (selected == candidates.begin() + returned)
      fail(deterministic
               ? "cuDNN Conv2d input gradient has no deterministic algorithm "
                 "within its workspace limit"
               : "cuDNN Conv2d input gradient has no algorithm within its "
                 "workspace limit");
    impl_->data_algorithm = selected->algo;
    impl_->workspace = selected->memory;
    check(cudnnSetConvolutionMathType(impl_->convolution, selected->mathType),
          "cudnnSetConvolutionMathType Conv2d input gradient");
    return;
  }
  std::array<cudnnConvolutionBwdFilterAlgoPerf_t, 8> candidates{};
  int returned = 0;
  check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
            impl_->handle, impl_->input, impl_->grad_output,
            impl_->convolution, impl_->weight,
            static_cast<int>(candidates.size()), &returned, candidates.data()),
        "cudnnGetConvolutionBackwardFilterAlgorithm_v7");
  const auto selected = std::find_if(
      candidates.begin(), candidates.begin() + returned,
      [&](const cudnnConvolutionBwdFilterAlgoPerf_t &candidate) {
        return candidate.status == CUDNN_STATUS_SUCCESS &&
               candidate.memory <= workspace_limit_bytes &&
               (!deterministic ||
                candidate.determinism == CUDNN_DETERMINISTIC);
      });
  if (selected == candidates.begin() + returned)
    fail(deterministic
             ? "cuDNN Conv2d weight gradient has no deterministic algorithm "
               "within its workspace limit"
             : "cuDNN Conv2d weight gradient has no algorithm within its "
               "workspace limit");
  impl_->filter_algorithm = selected->algo;
  impl_->workspace = selected->memory;
  check(cudnnSetConvolutionMathType(impl_->convolution, selected->mathType),
        "cudnnSetConvolutionMathType Conv2d weight gradient");
}

CudnnConv2dBackwardPlan::~CudnnConv2dBackwardPlan() = default;
CudnnConv2dBackwardPlan::CudnnConv2dBackwardPlan(
    CudnnConv2dBackwardPlan &&) noexcept = default;
CudnnConv2dBackwardPlan &
CudnnConv2dBackwardPlan::operator=(CudnnConv2dBackwardPlan &&) noexcept =
    default;

std::size_t CudnnConv2dBackwardPlan::workspace_bytes() const {
  return impl_->workspace;
}

void CudnnConv2dBackwardPlan::execute(std::uintptr_t grad_output,
                                      std::uintptr_t operand,
                                      std::uintptr_t gradient,
                                      std::uintptr_t workspace,
                                      std::uintptr_t stream) {
  check(cudnnSetStream(impl_->handle, reinterpret_cast<cudaStream_t>(stream)),
        "cudnnSetStream Conv2d gradient");
  constexpr float one = 1.0F;
  constexpr float zero = 0.0F;
  cudnnStatus_t status = CUDNN_STATUS_SUCCESS;
  const char *action = "";
  switch (impl_->kind) {
  case Kind::Input:
    action = "cudnnConvolutionBackwardData";
    status = cudnnConvolutionBackwardData(
        impl_->handle, &one, impl_->weight,
        reinterpret_cast<const void *>(operand), impl_->grad_output,
        reinterpret_cast<const void *>(grad_output), impl_->convolution,
        impl_->data_algorithm, reinterpret_cast<void *>(workspace),
        impl_->workspace, &zero, impl_->input,
        reinterpret_cast<void *>(gradient));
    break;
  case Kind::Weight:
    action = "cudnnConvolutionBackwardFilter";
    status = cudnnConvolutionBackwardFilter(
        impl_->handle, &one, impl_->input,
        reinterpret_cast<const void *>(operand), impl_->grad_output,
        reinterpret_cast<const void *>(grad_output), impl_->convolution,
        impl_->filter_algorithm, reinterpret_cast<void *>(workspace),
        impl_->workspace, &zero, impl_->weight,
        reinterpret_cast<void *>(gradient));
    break;
  case Kind::Bias:
    action = "cudnnConvolutionBackwardBias";
    status = cudnnConvolutionBackwardBias(
        impl_->handle, &one, impl_->grad_output,
        reinterpret_cast<const void *>(grad_output), &zero, impl_->bias,
        reinterpret_cast<void *>(gradient));
    break;
  }
  if (status != CUDNN_STATUS_SUCCESS)
    fail(std::string(action) + " Conv2d" + impl_->description + ": " +
         cudnnGetErrorString(status));
}

struct CudnnConv3dPlan::Impl {
  cudnnHandle_t handle{};
  cudnnTensorDescriptor_t input{};
  cudnnTensorDescriptor_t output{};
  cudnnTensorDescriptor_t bias{};
  cudnnFilterDescriptor_t weight{};
  cudnnConvolutionDescriptor_t convolution{};
  cudnnConvolutionFwdAlgo_t algorithm{};
  std::size_t workspace{};
  bool biased{};

  ~Impl() {
    if (convolution)
      (void)cudnnDestroyConvolutionDescriptor(convolution);
    if (weight)
      (void)cudnnDestroyFilterDescriptor(weight);
    if (bias)
      (void)cudnnDestroyTensorDescriptor(bias);
    if (output)
      (void)cudnnDestroyTensorDescriptor(output);
    if (input)
      (void)cudnnDestroyTensorDescriptor(input);
    // The handle is shared and outlives the plan.
  }
};

CudnnConv3dPlan::CudnnConv3dPlan(
    const ir::TensorDesc &input, const ir::TensorDesc &weight,
    const ir::TensorDesc &output, std::uint64_t stride_t,
    std::uint64_t stride_h, std::uint64_t stride_w, std::uint64_t pad_t,
    std::uint64_t pad_h, std::uint64_t pad_w, std::uint64_t dilation_t,
    std::uint64_t dilation_h, std::uint64_t dilation_w,
    std::uint64_t groups, bool biased, std::size_t workspace_limit_bytes,
    bool deterministic)
    : impl_(std::make_unique<Impl>()) {
  if (input.dims.size() != 5U || weight.dims.size() != 5U ||
      output.dims.size() != 5U || input.dtype != weight.dtype ||
      input.dtype != output.dtype)
    fail("cuDNN Conv3d requires matching NCDHW/OIDHW rank-5 tensors");
  const auto dtype = data_type(input.dtype);
  // Shared per-thread handle: see cudnn_handle.hpp.
  impl_->handle = shared_cudnn_handle();
  check(cudnnCreateTensorDescriptor(&impl_->input),
        "cudnnCreateTensorDescriptor Conv3d input");
  check(cudnnCreateTensorDescriptor(&impl_->output),
        "cudnnCreateTensorDescriptor Conv3d output");
  check(cudnnCreateTensorDescriptor(&impl_->bias),
        "cudnnCreateTensorDescriptor Conv3d bias");
  check(cudnnCreateFilterDescriptor(&impl_->weight),
        "cudnnCreateFilterDescriptor Conv3d");
  check(cudnnCreateConvolutionDescriptor(&impl_->convolution),
        "cudnnCreateConvolutionDescriptor Conv3d");
  const auto tensor_dimensions = [](const std::vector<std::uint64_t> &dims,
                                    const char *label) {
    std::array<int, 5> result{};
    for (std::size_t index = 0U; index < result.size(); ++index)
      result[index] = dimension(dims[index], label);
    return result;
  };
  const auto contiguous_strides = [](const std::array<int, 5> &dims) {
    std::array<int, 5> result{};
    result[4] = 1;
    for (std::size_t index = 4U; index-- > 0U;)
      result[index] = result[index + 1U] * dims[index + 1U];
    return result;
  };
  const auto input_dims = tensor_dimensions(input.dims, "input dimension");
  const auto output_dims = tensor_dimensions(output.dims, "output dimension");
  const auto input_strides = contiguous_strides(input_dims);
  const auto output_strides = contiguous_strides(output_dims);
  check(cudnnSetTensorNdDescriptor(impl_->input, dtype, 5, input_dims.data(),
                                   input_strides.data()),
        "cudnnSetTensorNdDescriptor Conv3d input");
  check(cudnnSetTensorNdDescriptor(impl_->output, dtype, 5,
                                   output_dims.data(), output_strides.data()),
        "cudnnSetTensorNdDescriptor Conv3d output");
  const std::array<int, 5> bias_dims{
      1, dimension(output.dims[1], "bias channels"), 1, 1, 1};
  const auto bias_strides = contiguous_strides(bias_dims);
  check(cudnnSetTensorNdDescriptor(impl_->bias, dtype, 5, bias_dims.data(),
                                   bias_strides.data()),
        "cudnnSetTensorNdDescriptor Conv3d bias");
  const auto weight_dims = tensor_dimensions(weight.dims, "filter dimension");
  check(cudnnSetFilterNdDescriptor(impl_->weight, dtype, CUDNN_TENSOR_NCHW, 5,
                                   weight_dims.data()),
        "cudnnSetFilterNdDescriptor Conv3d");
  const std::array<int, 3> padding{
      parameter(pad_t, "padding time"), parameter(pad_h, "padding height"),
      parameter(pad_w, "padding width")};
  const std::array<int, 3> stride{
      dimension(stride_t, "stride time"),
      dimension(stride_h, "stride height"),
      dimension(stride_w, "stride width")};
  const std::array<int, 3> dilation{
      dimension(dilation_t, "dilation time"),
      dimension(dilation_h, "dilation height"),
      dimension(dilation_w, "dilation width")};
  check(cudnnSetConvolutionNdDescriptor(
            impl_->convolution, 3, padding.data(), stride.data(),
            dilation.data(), CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor Conv3d");
  check(cudnnSetConvolutionGroupCount(impl_->convolution,
                                      dimension(groups, "groups")),
        "cudnnSetConvolutionGroupCount Conv3d");
  if (input.dtype != ir::DType::F32)
    check(cudnnSetConvolutionMathType(impl_->convolution,
                                      CUDNN_TENSOR_OP_MATH),
          "cudnnSetConvolutionMathType Conv3d");
  std::array<cudnnConvolutionFwdAlgoPerf_t, 8> candidates{};
  int returned = 0;
  check(cudnnGetConvolutionForwardAlgorithm_v7(
            impl_->handle, impl_->input, impl_->weight, impl_->convolution,
            impl_->output, static_cast<int>(candidates.size()), &returned,
            candidates.data()),
        "cudnnGetConvolutionForwardAlgorithm_v7 Conv3d");
  const auto selected = std::find_if(
      candidates.begin(), candidates.begin() + returned,
      [&](const cudnnConvolutionFwdAlgoPerf_t &candidate) {
        return candidate.status == CUDNN_STATUS_SUCCESS &&
               candidate.memory <= workspace_limit_bytes &&
               (!deterministic ||
                candidate.determinism == CUDNN_DETERMINISTIC);
      });
  if (selected == candidates.begin() + returned)
    fail(deterministic
             ? "cuDNN Conv3d has no deterministic algorithm within its workspace limit"
             : "cuDNN Conv3d has no algorithm within its workspace limit");
  impl_->algorithm = selected->algo;
  // The v7 heuristic returns the math type that belongs with each
  // algorithm. Applying the algorithm while leaving the descriptor on the
  // math type used for the query is a parameter mismatch: cuDNN rejects
  // the launch (a 256-channel 3x3 convolution at 256x256 in BF16 chose an
  // algorithm wanting default math and failed with BAD_PARAM).
  check(cudnnSetConvolutionMathType(impl_->convolution, selected->mathType),
        "cudnnSetConvolutionMathType Conv3d (selected algorithm)");
  impl_->workspace = selected->memory;
  impl_->biased = biased;
}

CudnnConv3dPlan::~CudnnConv3dPlan() = default;
CudnnConv3dPlan::CudnnConv3dPlan(CudnnConv3dPlan &&) noexcept = default;
CudnnConv3dPlan &
CudnnConv3dPlan::operator=(CudnnConv3dPlan &&) noexcept = default;

std::size_t CudnnConv3dPlan::workspace_bytes() const {
  return impl_->workspace;
}

void CudnnConv3dPlan::execute(std::uintptr_t input, std::uintptr_t weight,
                              std::uintptr_t bias, std::uintptr_t output,
                              std::uintptr_t workspace,
                              std::uintptr_t stream) {
  check(cudnnSetStream(impl_->handle, reinterpret_cast<cudaStream_t>(stream)),
        "cudnnSetStream Conv3d");
  constexpr float one = 1.0F;
  constexpr float zero = 0.0F;
  check(cudnnConvolutionForward(
            impl_->handle, &one, impl_->input,
            reinterpret_cast<const void *>(input), impl_->weight,
            reinterpret_cast<const void *>(weight), impl_->convolution,
            impl_->algorithm, reinterpret_cast<void *>(workspace),
            impl_->workspace, &zero, impl_->output,
            reinterpret_cast<void *>(output)),
        "cudnnConvolutionForward Conv3d");
  if (impl_->biased)
    check(cudnnAddTensor(impl_->handle, &one, impl_->bias,
                         reinterpret_cast<const void *>(bias), &one,
                         impl_->output, reinterpret_cast<void *>(output)),
          "cudnnAddTensor Conv3d bias");
}

} // namespace dif::runtime
