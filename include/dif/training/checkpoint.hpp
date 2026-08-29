#pragma once

#include "dif/runtime/executor.hpp"
#include "dif/support/sha256.hpp"

#include <cstdint>
#include <filesystem>

namespace dif::training {

struct Checkpoint {
  Sha256Digest program_fingerprint{};
  std::uint64_t completed_steps{};
  runtime::TensorMap state;
};

void write_checkpoint(const Checkpoint &checkpoint,
                      const std::filesystem::path &path);
Checkpoint read_checkpoint(const std::filesystem::path &path);

} // namespace dif::training
