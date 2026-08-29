#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <span>
#include <unordered_map>

namespace dif::training {

struct AutodiffResult {
  ir::Program program;
  std::unordered_map<std::uint32_t, std::uint32_t> gradients;
};

AutodiffResult differentiate(const ir::Program &forward,
                             std::uint32_t loss_tensor,
                             std::span<const std::uint32_t> with_respect_to);

} // namespace dif::training
