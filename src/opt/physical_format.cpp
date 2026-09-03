#include "dif/opt/physical_format.hpp"

namespace dif::opt {

std::string_view physical_format_name(PhysicalFormat format) {
  switch (format) {
  case PhysicalFormat::Fp32:
    return "fp32";
  case PhysicalFormat::Bf16:
    return "bf16";
  case PhysicalFormat::Fp16:
    return "fp16";
  case PhysicalFormat::Fp8E4M3:
    return "fp8-e4m3";
  case PhysicalFormat::Int8ConvRot:
    return "int8-convrot";
  case PhysicalFormat::Int4Group:
    return "int4-group";
  case PhysicalFormat::Int5Group:
    return "int5-group";
  case PhysicalFormat::SquareQW8:
    return "squareq-w8";
  case PhysicalFormat::SquareQW4:
    return "squareq-w4";
  case PhysicalFormat::SquareQNvfp4:
    return "squareq-nvfp4";
  case PhysicalFormat::Fp8BlockScaled:
    return "mxfp8-block-scaled";
  }
  return "unknown";
}

bool physical_format_from_name(std::string_view name, PhysicalFormat &format) {
  for (const auto candidate : all_physical_formats()) {
    if (physical_format_name(candidate) == name) {
      format = candidate;
      return true;
    }
  }
  return false;
}

std::vector<PhysicalFormat> all_physical_formats() {
  return {PhysicalFormat::Fp32,        PhysicalFormat::Bf16,
          PhysicalFormat::Fp16,        PhysicalFormat::Fp8E4M3,
          PhysicalFormat::Int8ConvRot, PhysicalFormat::Int4Group,
          PhysicalFormat::Int5Group,   PhysicalFormat::SquareQW8,
          PhysicalFormat::SquareQW4,   PhysicalFormat::SquareQNvfp4,
          PhysicalFormat::Fp8BlockScaled};
}

std::string_view format_availability_name(FormatAvailability availability) {
  switch (availability) {
  case FormatAvailability::SearchCandidate:
    return "search-candidate";
  case FormatAvailability::ExecutionPolicy:
    return "execution-policy";
  case FormatAvailability::HookOnly:
    return "hook-only";
  }
  return "unknown";
}

namespace {

struct Requirement {
  bool needs_bf16{};
  bool needs_fp16{};
  bool needs_fp8{};
  bool needs_int8{};
  bool needs_nvfp4{};
  bool needs_device{};
  // Minimum linked cuBLASLt version (cublasLtGetVersion() form,
  // major*10000 + minor*100 + patch); 0 = no library requirement.
  std::uint64_t min_cublaslt_version{};
};

constexpr std::uint64_t kCublasLtBlockScaledMatmulVersion = 120800U;

Requirement requirement(PhysicalFormat format) {
  switch (format) {
  case PhysicalFormat::Fp32:
    return {};
  case PhysicalFormat::Bf16:
    return {};
  case PhysicalFormat::Fp16:
    return {};
  case PhysicalFormat::Fp8E4M3:
    return {.needs_fp8 = true, .needs_device = true};
  case PhysicalFormat::Int8ConvRot:
    return {.needs_int8 = true, .needs_device = true};
  case PhysicalFormat::Int4Group:
    return {};
  case PhysicalFormat::Int5Group:
    return {};
  case PhysicalFormat::SquareQW8:
    return {.needs_int8 = true, .needs_device = true};
  case PhysicalFormat::SquareQW4:
    return {.needs_bf16 = true, .needs_device = true};
  case PhysicalFormat::SquareQNvfp4:
    return {.needs_nvfp4 = true, .needs_device = true};
  case PhysicalFormat::Fp8BlockScaled:
    return {.needs_fp8 = true,
            .needs_device = true,
            .min_cublaslt_version = kCublasLtBlockScaledMatmulVersion};
  }
  return {};
}

} // namespace

FormatStatus physical_format_status(PhysicalFormat format,
                                    const target::TargetProfile *target) {
  FormatStatus status;
  status.format = format;
  switch (format) {
  case PhysicalFormat::Fp32:
  case PhysicalFormat::Bf16:
  case PhysicalFormat::Fp16:
    status.availability = FormatAvailability::SearchCandidate;
    status.availability_reason =
        "SetOperationPrecision transform; measured by the optimizer search";
    break;
  case PhysicalFormat::Int4Group:
  case PhysicalFormat::Int5Group:
    status.availability = FormatAvailability::SearchCandidate;
    status.availability_reason =
        "QuantizeConstantWeights transform with grouped dequantization; "
        "measured by the optimizer search";
    break;
  case PhysicalFormat::Int8ConvRot:
    status.availability = FormatAvailability::ExecutionPolicy;
    status.availability_reason =
        "generic ConvRot INT8 lowering is explicit RunOptions execution "
        "policy over a prepared rotated cache (convrot_int8_checkpoint); it "
        "is not a DiffIR transform and does not enter the search";
    break;
  case PhysicalFormat::Fp8E4M3:
    status.availability = FormatAvailability::HookOnly;
    status.availability_reason =
        "no FP8 Linear implementation in this build; identity and target "
        "requirements are registered so a later backend can compete";
    break;
  case PhysicalFormat::SquareQW4:
    status.availability = FormatAvailability::ExecutionPolicy;
    status.availability_reason =
        "SquareQ W4 slab (squareq_w4_v1) consumed by the frontend rewrite "
        "dif::frontend::rewrite_linear_weights_squareq_w4: dequantize_int4 + "
        "Hadamard-256 + low-rank on existing DiffIR semantics; selected by "
        "the tool (difflux2sample --squareq-w4-slab), not by the search";
    break;
  case PhysicalFormat::SquareQW8:
  case PhysicalFormat::SquareQNvfp4:
    status.availability = FormatAvailability::HookOnly;
    status.availability_reason =
        "SquareQ physical format hook: identity and target requirements are "
        "registered; no backend implementation in this build";
    break;
  case PhysicalFormat::Fp8BlockScaled:
    status.availability = FormatAvailability::ExecutionPolicy;
    status.availability_reason =
        "MXFP8 lowering is an explicit per-block program rewrite selected by "
        "the FLUX.2 sampler flags (QuantizeFp8BlockScaled + "
        "LinearFp8BlockScaled) executed through cuBLASLt block-scaled matmul; "
        "it is not a DiffIR search transform in this build";
    break;
  }
  if (!target) {
    status.legal_on_target = false;
    status.legality_reason = "no target profile supplied; probe the target "
                             "before selecting a physical format";
    status.competes = false;
    return status;
  }
  const auto need = requirement(format);
  const bool device = target->vendor == target::Vendor::Nvidia;
  std::string missing;
  if (need.needs_device && !device)
    missing += (missing.empty() ? "" : ", ") + std::string("an NVIDIA device");
  if (need.needs_bf16 && !target->precision.bf16_tensor_cores)
    missing += (missing.empty() ? "" : ", ") + std::string("BF16 tensor cores");
  if (need.needs_fp16 && !target->precision.fp16_tensor_cores)
    missing += (missing.empty() ? "" : ", ") + std::string("FP16 tensor cores");
  if (need.needs_fp8 && !target->precision.fp8_tensor_cores)
    missing += (missing.empty() ? "" : ", ") + std::string("FP8 tensor cores");
  if (need.needs_int8 && !target->precision.int8_tensor_cores)
    missing += (missing.empty() ? "" : ", ") + std::string("INT8 tensor cores");
  if (need.needs_nvfp4 && !target->precision.nvfp4_tensor_cores)
    missing += (missing.empty() ? "" : ", ") + std::string("NVFP4 tensor cores");
  if (need.min_cublaslt_version != 0U &&
      target->cublaslt_version < need.min_cublaslt_version)
    missing += (missing.empty() ? "" : ", ") +
               std::string("cuBLASLt >= ") +
               std::to_string(need.min_cublaslt_version) +
               " (block-scaled matmul; linked " +
               std::to_string(target->cublaslt_version) + ")";
  if (missing.empty()) {
    status.legal_on_target = true;
    status.legality_reason =
        std::string("target ") +
        std::string(target::architecture_name(target->architecture)) +
        " provides every capability this format requires";
  } else {
    status.legal_on_target = false;
    status.legality_reason =
        std::string("target ") +
        std::string(target::architecture_name(target->architecture)) +
        " lacks " + missing;
  }
  status.competes = status.legal_on_target &&
                    status.availability == FormatAvailability::SearchCandidate;
  return status;
}

std::optional<ir::DType> format_precision(PhysicalFormat format) {
  switch (format) {
  case PhysicalFormat::Fp32:
    return ir::DType::F32;
  case PhysicalFormat::Bf16:
    return ir::DType::BF16;
  case PhysicalFormat::Fp16:
    return ir::DType::F16;
  default:
    return std::nullopt;
  }
}

std::optional<std::uint64_t> format_quantization_bits(PhysicalFormat format) {
  switch (format) {
  case PhysicalFormat::Int4Group:
    return 4U;
  case PhysicalFormat::Int5Group:
    return 5U;
  default:
    return std::nullopt;
  }
}

} // namespace dif::opt
