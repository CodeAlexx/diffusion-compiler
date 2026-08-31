#pragma once

#include "dif/opt/optimizer.hpp"

#include <filesystem>

namespace dif::opt {

struct Plan {
  Sha256Digest base_program_fingerprint{};
  Sha256Digest candidate_program_fingerprint{};
  Sha256Digest candidate_fingerprint{};
  ExecutionPolicy policy;
  Recipe recipe;
};

Plan make_plan(const ir::Program &base, const Candidate &candidate);
void write_plan(const Plan &plan, const std::filesystem::path &path);
Plan read_plan(const std::filesystem::path &path);

// Replay refuses a different base program and refuses any recipe whose result
// no longer matches the recorded candidate fingerprint.
ir::Program replay_plan(const ir::Program &base, const Plan &plan);
Candidate replay_candidate(const ir::Program &base, const Plan &plan);

} // namespace dif::opt
