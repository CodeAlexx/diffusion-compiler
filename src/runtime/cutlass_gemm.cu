#include "dif/runtime/cutlass_gemm.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

namespace dif::runtime {
namespace {

void set_error(char *destination, std::size_t capacity,
               const std::string &message) {
  if (!destination || capacity == 0U)
    return;
  std::snprintf(destination, capacity, "%s", message.c_str());
}

using Element = cutlass::bfloat16_t;
using Accumulator = float;
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;
using Epilogue = cutlass::epilogue::thread::LinearCombination<
    Element, 8, Accumulator, Accumulator>;

template <int ThreadblockM, int ThreadblockN, int ThreadblockK, int WarpM,
          int WarpN, int WarpK, int Stages, int Swizzle = 1>
using AmpereGemm = cutlass::gemm::device::Gemm<
    Element, cutlass::layout::RowMajor, Element, cutlass::layout::ColumnMajor,
    Element, cutlass::layout::RowMajor, Accumulator,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<ThreadblockM, ThreadblockN, ThreadblockK>,
    cutlass::gemm::GemmShape<WarpM, WarpN, WarpK>, InstructionShape, Epilogue,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Swizzle>,
    Stages, 8, 8, false, cutlass::arch::OpMultiplyAdd>;

class PlanBase {
public:
  virtual ~PlanBase() = default;
  virtual bool launch(cudaStream_t stream, std::string &error) = 0;
  virtual CutlassGemmResources resources() const = 0;
};

template <typename Gemm, int ThreadblockM, int ThreadblockN, int ThreadblockK,
          int WarpM, int WarpN, int WarpK, int Stages>
class Plan final : public PlanBase {
public:
  Plan(const char *name, std::uint32_t m, std::uint32_t n, std::uint32_t k,
       std::uintptr_t input, std::uintptr_t weight, std::uintptr_t output,
       cudaStream_t stream, std::string &error)
      : name_(name),
        arguments_({static_cast<int>(m), static_cast<int>(n),
                    static_cast<int>(k)},
                   {reinterpret_cast<const Element *>(input),
                    static_cast<int>(k)},
                   {reinterpret_cast<const Element *>(weight),
                    static_cast<int>(k)},
                   {reinterpret_cast<const Element *>(output),
                    static_cast<int>(n)},
                   {reinterpret_cast<Element *>(output), static_cast<int>(n)},
                   {1.0F, 0.0F}) {
    auto status = Gemm::can_implement(arguments_);
    if (status != cutlass::Status::kSuccess) {
      error = std::string("CUTLASS can_implement failed: ") +
              cutlassGetStatusString(status);
      return;
    }
    status = gemm_.initialize(arguments_, nullptr, stream);
    if (status != cutlass::Status::kSuccess) {
      error = std::string("CUTLASS initialize failed: ") +
              cutlassGetStatusString(status);
      return;
    }
    cudaFuncAttributes attributes{};
    const auto attribute_status = cudaFuncGetAttributes(
        &attributes, cutlass::Kernel<typename Gemm::GemmKernel>);
    if (attribute_status != cudaSuccess) {
      error = std::string("cudaFuncGetAttributes failed: ") +
              cudaGetErrorString(attribute_status);
      return;
    }
    attributes_ = attributes;
    ready_ = true;
  }

  bool ready() const { return ready_; }

  bool launch(cudaStream_t stream, std::string &error) override {
    const auto status = gemm_.run(stream);
    if (status == cutlass::Status::kSuccess)
      return true;
    error = std::string("CUTLASS run failed: ") + cutlassGetStatusString(status);
    return false;
  }

  CutlassGemmResources resources() const override {
    constexpr auto warps =
        (ThreadblockM / WarpM) * (ThreadblockN / WarpN) *
        (ThreadblockK / WarpK);
    return {name_,
            ThreadblockM,
            ThreadblockN,
            ThreadblockK,
            WarpM,
            WarpN,
            WarpK,
            Stages,
            warps * 32U,
            static_cast<std::uint32_t>(attributes_.numRegs),
            static_cast<std::uint64_t>(attributes_.sharedSizeBytes),
            sizeof(typename Gemm::GemmKernel::SharedStorage),
            static_cast<std::uint64_t>(attributes_.maxDynamicSharedSizeBytes)};
  }

private:
  const char *name_{};
  typename Gemm::Arguments arguments_;
  Gemm gemm_;
  cudaFuncAttributes attributes_{};
  bool ready_{};
};

template <typename Gemm, int ThreadblockM, int ThreadblockN, int ThreadblockK,
          int WarpM, int WarpN, int WarpK, int Stages>
std::unique_ptr<PlanBase> make_plan(
    const char *name, std::uint32_t m, std::uint32_t n, std::uint32_t k,
    std::uintptr_t input, std::uintptr_t weight, std::uintptr_t output,
    cudaStream_t stream, std::string &error) {
  auto result = std::make_unique<Plan<Gemm, ThreadblockM, ThreadblockN,
                                      ThreadblockK, WarpM, WarpN, WarpK,
                                      Stages>>(name, m, n, k, input, weight,
                                               output, stream, error);
  if (!result->ready())
    return nullptr;
  return result;
}

using Gemm128x128x32S3 =
    AmpereGemm<128, 128, 32, 64, 64, 32, 3, 1>;
using Gemm256x128x32S3 =
    AmpereGemm<256, 128, 32, 64, 64, 32, 3, 1>;
using Gemm128x256x32S3 =
    AmpereGemm<128, 256, 32, 64, 64, 32, 3, 1>;
using Gemm256x64x32S4 =
    AmpereGemm<256, 64, 32, 64, 64, 32, 4, 1>;
using Gemm64x256x32S4 =
    AmpereGemm<64, 256, 32, 64, 64, 32, 4, 1>;
using Gemm128x128x32S4 =
    AmpereGemm<128, 128, 32, 64, 64, 32, 4, 1>;
using Gemm256x128x32S3Swizzle8 =
    AmpereGemm<256, 128, 32, 64, 64, 32, 3, 8>;
using Gemm128x128x32Warp64x32S3 =
    AmpereGemm<128, 128, 32, 64, 32, 32, 3, 1>;
using Gemm128x128x32Warp64x32S3Swizzle8 =
    AmpereGemm<128, 128, 32, 64, 32, 32, 3, 8>;
using Gemm128x128x32Warp32x64S3 =
    AmpereGemm<128, 128, 32, 32, 64, 32, 3, 1>;
using Gemm128x64x32Warp64x32S5 =
    AmpereGemm<128, 64, 32, 64, 32, 32, 5, 1>;
using Gemm64x128x32Warp32x64S5 =
    AmpereGemm<64, 128, 32, 32, 64, 32, 5, 1>;
using Gemm64x64x32Warp32x32S6 =
    AmpereGemm<64, 64, 32, 32, 32, 32, 6, 1>;
using Gemm128x128x64S3 =
    AmpereGemm<128, 128, 64, 64, 64, 64, 3, 1>;
using Gemm256x128x64S2 =
    AmpereGemm<256, 128, 64, 64, 64, 64, 2, 1>;
using Gemm128x256x64S2 =
    AmpereGemm<128, 256, 64, 64, 64, 64, 2, 1>;

} // namespace

struct CutlassGemmHandle {
  std::unique_ptr<PlanBase> plan;
};

const char *cutlass_gemm_schedule_name(std::uint32_t schedule) {
  switch (schedule) {
  case 1:
    return "ampere_128x128x32_s3";
  case 2:
    return "ampere_256x128x32_s3";
  case 3:
    return "ampere_128x256x32_s3";
  case 4:
    return "ampere_256x64x32_s4";
  case 5:
    return "ampere_64x256x32_s4";
  case 6:
    return "ampere_128x128x32_s4";
  case 7:
    return "ampere_256x128x32_s3_swizzle8";
  case 8:
    return "ampere_128x128x32_warp64x32_s3";
  case 9:
    return "ampere_128x128x32_warp64x32_s3_swizzle8";
  case 10:
    return "ampere_128x128x32_warp32x64_s3";
  case 11:
    return "ampere_128x64x32_warp64x32_s5";
  case 12:
    return "ampere_64x128x32_warp32x64_s5";
  case 13:
    return "ampere_64x64x32_warp32x32_s6";
  case 14:
    return "ampere_128x128x64_s3";
  case 15:
    return "ampere_256x128x64_s2";
  case 16:
    return "ampere_128x256x64_s2";
  default:
    return nullptr;
  }
}

CutlassGemmHandle *create_cutlass_gemm(
    std::uint32_t schedule, std::uint32_t m, std::uint32_t n,
    std::uint32_t k, std::uintptr_t input, std::uintptr_t weight,
    std::uintptr_t output, std::uintptr_t stream, char *error,
    std::size_t error_capacity) {
  const auto *name = cutlass_gemm_schedule_name(schedule);
  if (!name) {
    set_error(error, error_capacity, "unknown CUTLASS GEMM schedule");
    return nullptr;
  }
  std::string message;
  std::unique_ptr<PlanBase> plan;
  const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  switch (schedule) {
  case 1:
    plan = make_plan<Gemm128x128x32S3, 128, 128, 32, 64, 64, 32, 3>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 2:
    plan = make_plan<Gemm256x128x32S3, 256, 128, 32, 64, 64, 32, 3>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 3:
    plan = make_plan<Gemm128x256x32S3, 128, 256, 32, 64, 64, 32, 3>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 4:
    plan = make_plan<Gemm256x64x32S4, 256, 64, 32, 64, 64, 32, 4>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 5:
    plan = make_plan<Gemm64x256x32S4, 64, 256, 32, 64, 64, 32, 4>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 6:
    plan = make_plan<Gemm128x128x32S4, 128, 128, 32, 64, 64, 32, 4>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 7:
    plan = make_plan<Gemm256x128x32S3Swizzle8, 256, 128, 32, 64, 64, 32,
                     3>(name, m, n, k, input, weight, output, cuda_stream,
                        message);
    break;
  case 8:
    plan = make_plan<Gemm128x128x32Warp64x32S3, 128, 128, 32, 64, 32, 32,
                     3>(name, m, n, k, input, weight, output, cuda_stream,
                        message);
    break;
  case 9:
    plan = make_plan<Gemm128x128x32Warp64x32S3Swizzle8, 128, 128, 32, 64,
                     32, 32, 3>(name, m, n, k, input, weight, output,
                                cuda_stream, message);
    break;
  case 10:
    plan = make_plan<Gemm128x128x32Warp32x64S3, 128, 128, 32, 32, 64, 32,
                     3>(name, m, n, k, input, weight, output, cuda_stream,
                        message);
    break;
  case 11:
    plan = make_plan<Gemm128x64x32Warp64x32S5, 128, 64, 32, 64, 32, 32, 5>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 12:
    plan = make_plan<Gemm64x128x32Warp32x64S5, 64, 128, 32, 32, 64, 32, 5>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 13:
    plan = make_plan<Gemm64x64x32Warp32x32S6, 64, 64, 32, 32, 32, 32, 6>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 14:
    plan = make_plan<Gemm128x128x64S3, 128, 128, 64, 64, 64, 64, 3>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 15:
    plan = make_plan<Gemm256x128x64S2, 256, 128, 64, 64, 64, 64, 2>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  case 16:
    plan = make_plan<Gemm128x256x64S2, 128, 256, 64, 64, 64, 64, 2>(
        name, m, n, k, input, weight, output, cuda_stream, message);
    break;
  default:
    break;
  }
  if (!plan) {
    set_error(error, error_capacity, message);
    return nullptr;
  }
  auto *handle = new CutlassGemmHandle;
  handle->plan = std::move(plan);
  return handle;
}

bool launch_cutlass_gemm(CutlassGemmHandle *handle, std::uintptr_t stream,
                         char *error, std::size_t error_capacity) {
  if (!handle || !handle->plan) {
    set_error(error, error_capacity, "CUTLASS GEMM handle is null");
    return false;
  }
  std::string message;
  if (handle->plan->launch(reinterpret_cast<cudaStream_t>(stream), message))
    return true;
  set_error(error, error_capacity, message);
  return false;
}

CutlassGemmResources cutlass_gemm_resources(CutlassGemmHandle *handle) {
  if (!handle || !handle->plan)
    return {};
  return handle->plan->resources();
}

void destroy_cutlass_gemm(CutlassGemmHandle *handle) { delete handle; }

} // namespace dif::runtime
