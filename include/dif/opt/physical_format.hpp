#pragma once

// Compiler-wide physical weight/execution formats a Linear (or other
// uniform-float operation) may be lowered to. DiffIR semantics stay generic:
// a format is a candidate physical implementation the compiler may select
// when the target supports it and this build implements it. SquareQ W8/W4/
// NVFP4 and FP8 are registered here as hooks so their legality and
// availability are reported honestly; a format without a backend
// implementation never competes and is never silently skipped.

#include "dif/ir/ir.hpp"
#include "dif/target/profile.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dif::opt {

enum class PhysicalFormat : std::uint32_t {
  Fp32 = 1,
  Bf16 = 2,
  Fp16 = 3,
  Fp8E4M3 = 4,
  Int8ConvRot = 5,
  Int4Group = 6,
  Int5Group = 7,
  SquareQW8 = 8,
  SquareQW4 = 9,
  SquareQNvfp4 = 10,
  // MXFP8: E4M3 values with one UE8M0 scale per 32-element block on both
  // operands, F32 accumulation, BF16 output (DiffIR LinearFp8BlockScaled).
  // Legal only where the target has FP8 tensor cores and the linked cuBLASLt
  // provides block-scaled matmul (12.8 or newer); the CUDA backend compiles
  // the plan only against such a toolkit and fails closed otherwise.
  Fp8BlockScaled = 11,
};

std::string_view physical_format_name(PhysicalFormat format);
bool physical_format_from_name(std::string_view name, PhysicalFormat &format);
std::vector<PhysicalFormat> all_physical_formats();

// How a format can participate in this build.
enum class FormatAvailability : std::uint32_t {
  // Expressible as a DiffIR transform the optimizer search can measure.
  SearchCandidate = 1,
  // Implemented as explicit execution policy (RunOptions) that needs
  // prepared state outside the search; reported, not searched.
  ExecutionPolicy = 2,
  // Registered identity and requirements only; no backend implementation.
  HookOnly = 3,
};

std::string_view format_availability_name(FormatAvailability availability);

struct FormatStatus {
  PhysicalFormat format{};
  bool legal_on_target{};
  std::string legality_reason;
  FormatAvailability availability{FormatAvailability::HookOnly};
  std::string availability_reason;
  // True only when the format is legal on the target and is a search
  // candidate in this build.
  bool competes{};
};

// Legality is decided from architecture-defined capabilities in the
// TargetProfile, never from product names. A null target makes every format
// illegal with the reason that no target was probed.
FormatStatus physical_format_status(PhysicalFormat format,
                                    const target::TargetProfile *target);

// Discovery mapping for search-candidate formats.
std::optional<ir::DType> format_precision(PhysicalFormat format);
std::optional<std::uint64_t> format_quantization_bits(PhysicalFormat format);

} // namespace dif::opt
