#include "dif/runtime/cutlass_gemm.hpp"

#include <cutlass/arch/arch.h>
#include <cutlass/arch/mma.h>
#include <cutlass/bfloat16.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/kernel/default_gemm_universal_with_visitor.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/epilogue/threadblock/fusion/visitors.hpp>

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

// H3's direct INT8 path observes a BF16 projection boundary after applying
// one dynamic scale per input row and one static scale per output channel.
// Express that boundary as an Ampere epilogue visitor tree so the I32
// accumulator is never materialized in global memory.
using Int8Element = std::int8_t;
using Int8Accumulator = std::int32_t;
using Int8Compute = float;
using Int8Output = cutlass::bfloat16_t;
// CUTLASS 2.x does not provide a BF16 <= I32 base-epilogue iterator
// specialization.  The visitor owns the real BF16 output store, while the
// wrapped base epilogue is only used to select fragment iteration.  FP16 has
// the same 16-bit fragment width and supplies the required I32 specialization.
using Int8BaseEpilogueElement = cutlass::half_t;
using Int8LayoutA = cutlass::layout::RowMajor;
using Int8LayoutB = cutlass::layout::ColumnMajor;
using Int8LayoutC = cutlass::layout::RowMajor;
using Int8ThreadblockShape = cutlass::gemm::GemmShape<256, 128, 64>;
using Int8WarpShape = cutlass::gemm::GemmShape<64, 64, 64>;
using Int8InstructionShape = cutlass::gemm::GemmShape<16, 8, 32>;
constexpr int kInt8Alignment = 16;
constexpr int kInt8OutputAlignment = 8;
constexpr int kInt8Stages = 3;
constexpr int kInt8EpilogueStages = 1;

using Int8OutputThreadMap =
    cutlass::epilogue::threadblock::OutputTileThreadLayout<
        Int8ThreadblockShape, Int8WarpShape, Int8Output,
        kInt8OutputAlignment, kInt8EpilogueStages>;
using Int8Accum = cutlass::epilogue::threadblock::VisitorAccFetch;
using Int8RowScale = cutlass::epilogue::threadblock::VisitorColBroadcast<
    Int8OutputThreadMap, float,
    cute::Stride<cute::_1, cute::_0, std::int64_t>>;
using Int8MulRow = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, float, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using Int8ScaledRows = cutlass::epilogue::threadblock::Sm80EVT<
    Int8MulRow, Int8Accum, Int8RowScale>;
using Int8ColumnScale = cutlass::epilogue::threadblock::VisitorRowBroadcast<
    Int8OutputThreadMap, float,
    cute::Stride<cute::_0, cute::_1, std::int64_t>>;
using Int8MulColumn = cutlass::epilogue::threadblock::VisitorCompute<
    cutlass::multiplies, Int8Output, float,
    cutlass::FloatRoundStyle::round_to_nearest>;
using Int8ScaledOutput = cutlass::epilogue::threadblock::Sm80EVT<
    Int8MulColumn, Int8ScaledRows, Int8ColumnScale>;
using Int8Store = cutlass::epilogue::threadblock::VisitorAuxStore<
    Int8OutputThreadMap, Int8Output,
    cutlass::FloatRoundStyle::round_to_nearest,
    cute::Stride<std::int64_t, cute::_1, std::int64_t>>;
using Int8Epilogue = cutlass::epilogue::threadblock::Sm80EVT<
    Int8Store, Int8ScaledOutput>;
using Int8ScaledDefinition =
    cutlass::gemm::kernel::DefaultGemmWithVisitor<
        Int8Element, Int8LayoutA, cutlass::ComplexTransform::kNone,
        kInt8Alignment, Int8Element, Int8LayoutB,
        cutlass::ComplexTransform::kNone, kInt8Alignment,
        Int8BaseEpilogueElement,
        Int8LayoutC, kInt8OutputAlignment, Int8Accumulator, Int8Compute,
        cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
        Int8ThreadblockShape, Int8WarpShape, Int8InstructionShape,
        Int8Epilogue,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        kInt8Stages, cutlass::arch::OpMultiplyAddSaturate,
        kInt8EpilogueStages>;
using Int8ScaledKernelBase = typename Int8ScaledDefinition::GemmKernel;
// CUTLASS f7b19de's identity-swizzle visitor kernel inherits its universal
// base privately.  Re-expose the metadata required by GemmUniversalAdapter;
// execution and argument semantics remain CUTLASS's unchanged kernel.
struct Int8ScaledKernel : Int8ScaledKernelBase {
  using Mma = typename Int8ScaledDefinition::GemmBase::Mma;
  using Epilogue = typename Int8ScaledDefinition::Epilogue;
  using EpilogueOutputOp = typename Epilogue::OutputOp;
  using Operator = typename Mma::Operator;
  using WarpShape = typename Mma::Operator::Shape;
  using InstructionShape = typename Mma::Policy::Operator::InstructionShape;
  static constexpr auto kTransformA = cutlass::ComplexTransform::kNone;
  static constexpr auto kTransformB = cutlass::ComplexTransform::kNone;
  static constexpr int kAlignmentA = kInt8Alignment;
  static constexpr int kAlignmentB = kInt8Alignment;
  static constexpr int kAlignmentC = kInt8OutputAlignment;
  static constexpr int kThreadCount = 32 * Mma::WarpCount::kCount;

  static cutlass::Status can_implement(Arguments const &arguments) {
    const auto &problem = arguments.problem_size;
    if ((problem.k() % kAlignmentA) != 0 ||
        (problem.k() % kAlignmentB) != 0 ||
        (problem.n() % kAlignmentC) != 0)
      return cutlass::Status::kErrorMisalignedOperand;
    return cutlass::Status::kSuccess;
  }
};
using Int8ScaledDeviceGemm =
    cutlass::gemm::device::GemmUniversalAdapter<Int8ScaledKernel>;

typename Int8ScaledDeviceGemm::Arguments int8_scaled_arguments(
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    std::uintptr_t input, std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t output) {
  using namespace cute;
  typename Int8Epilogue::Arguments callbacks{
      {{{},
        {reinterpret_cast<const float *>(row_scale), 0.0F,
         {_1{}, _0{}, static_cast<std::int64_t>(m)}},
        {}},
       {reinterpret_cast<const float *>(column_scale), 0.0F,
        {_0{}, _1{}, static_cast<std::int64_t>(n)}},
       {}},
      {reinterpret_cast<Int8Output *>(output),
       {static_cast<std::int64_t>(n), _1{},
        static_cast<std::int64_t>(m) * n}}};
  return typename Int8ScaledDeviceGemm::Arguments(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {static_cast<int>(m), static_cast<int>(n), static_cast<int>(k)}, 1,
      callbacks, reinterpret_cast<const Int8Element *>(input),
      reinterpret_cast<const Int8Element *>(weight), nullptr, nullptr,
      static_cast<std::int64_t>(m) * k,
      static_cast<std::int64_t>(n) * k, 0, 0, static_cast<int>(k),
      static_cast<int>(k), 0, 0);
}

} // namespace

struct CutlassGemmHandle {
  std::unique_ptr<PlanBase> plan;
};

struct CutlassInt8ScaledGemmHandle {
  std::uint32_t m{};
  std::uint32_t n{};
  std::uint32_t k{};
  std::uintptr_t input{};
  std::uintptr_t weight{};
  std::uintptr_t row_scale{};
  std::uintptr_t column_scale{};
  std::uintptr_t output{};
  Int8ScaledDeviceGemm gemm;
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

CutlassInt8ScaledGemmHandle *create_cutlass_int8_scaled_gemm(
    std::uint32_t m, std::uint32_t n, std::uint32_t k,
    std::uintptr_t input, std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity) {
  auto handle = std::make_unique<CutlassInt8ScaledGemmHandle>();
  handle->m = m;
  handle->n = n;
  handle->k = k;
  handle->input = input;
  handle->weight = weight;
  handle->row_scale = row_scale;
  handle->column_scale = column_scale;
  handle->output = output;
  auto arguments = int8_scaled_arguments(
      m, n, k, input, weight, row_scale, column_scale, output);
  auto status = Int8ScaledDeviceGemm::can_implement(arguments);
  if (status != cutlass::Status::kSuccess) {
    set_error(error, error_capacity,
              std::string("CUTLASS INT8 scaled can_implement failed: ") +
                  cutlassGetStatusString(status));
    return nullptr;
  }
  if (Int8ScaledDeviceGemm::get_workspace_size(arguments) != 0U) {
    set_error(error, error_capacity,
              "CUTLASS INT8 scaled GEMM unexpectedly requires workspace");
    return nullptr;
  }
  status = handle->gemm.initialize(
      arguments, nullptr, reinterpret_cast<cudaStream_t>(stream));
  if (status != cutlass::Status::kSuccess) {
    set_error(error, error_capacity,
              std::string("CUTLASS INT8 scaled initialize failed: ") +
                  cutlassGetStatusString(status));
    return nullptr;
  }
  return handle.release();
}

bool launch_cutlass_int8_scaled_gemm(
    CutlassInt8ScaledGemmHandle *handle, std::uintptr_t input,
    std::uintptr_t weight, std::uintptr_t row_scale,
    std::uintptr_t column_scale, std::uintptr_t output,
    std::uintptr_t stream, char *error, std::size_t error_capacity) {
  if (!handle) {
    set_error(error, error_capacity, "null CUTLASS INT8 scaled GEMM handle");
    return false;
  }
  auto status = cutlass::Status::kSuccess;
  if (input != handle->input || weight != handle->weight ||
      row_scale != handle->row_scale ||
      column_scale != handle->column_scale || output != handle->output) {
    auto arguments = int8_scaled_arguments(
        handle->m, handle->n, handle->k, input, weight, row_scale,
        column_scale, output);
    status = handle->gemm.update(arguments);
    if (status == cutlass::Status::kSuccess) {
      handle->input = input;
      handle->weight = weight;
      handle->row_scale = row_scale;
      handle->column_scale = column_scale;
      handle->output = output;
    }
  }
  if (status == cutlass::Status::kSuccess)
    status = handle->gemm.run(reinterpret_cast<cudaStream_t>(stream));
  if (status == cutlass::Status::kSuccess)
    return true;
  set_error(error, error_capacity,
            std::string("CUTLASS INT8 scaled launch failed: ") +
                cutlassGetStatusString(status));
  return false;
}

void destroy_cutlass_int8_scaled_gemm(
    CutlassInt8ScaledGemmHandle *handle) {
  delete handle;
}

} // namespace dif::runtime
