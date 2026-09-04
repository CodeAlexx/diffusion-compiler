#pragma once

#include "dif/weights/safetensors.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace dif::training {

// Checking that a config describes the checkpoint it names.
//
// A trainer reads its architecture from a file, which is the only way a run
// is reproducible from something a person can read. But a file can say
// anything, and a wrong dimension does not fail loudly: it builds a graph
// whose weights load into the wrong shapes, or worse, into the right shapes
// with the wrong meaning. So every dimension a config claims is checked
// against the checkpoint that has to supply it, and against what the frontend
// will actually build, before a single step runs.
//
// Generic on purpose. difcore does not know what "joint_attention_dim" means;
// a frontend says which tensor proves it and which of that tensor's
// dimensions is the answer.
struct ArchitectureClaim {
  // The config key that made the claim, so an error names the line to fix.
  std::string key;
  // What the config said.
  std::uint64_t claimed{};
  // The checkpoint tensor that settles it, and which of its dimensions.
  // An empty tensor name means the checkpoint cannot settle this claim --
  // it is still checked against the frontend.
  std::string tensor;
  std::size_t dimension{};
  // What must be true independently of the checkpoint header -- normally
  // what the frontend will build with. Zero means nothing constrains it.
  std::uint64_t built{};
  // Who says so, for the error message. A claim checked against something
  // other than the frontend says what that something was.
  std::string built_by{"the frontend builds"};
  // A claim the checkpoint states as a multiple, e.g. a fused QKV projection
  // whose row count is three times the feature width.
  std::uint64_t multiple{1U};
};

struct ArchitectureDisagreement {
  std::string key;
  std::uint64_t claimed{};
  std::uint64_t checkpoint{};
  std::uint64_t built{};
  std::string detail;
};

// Every disagreement, not the first one. A config with three wrong dimensions
// should be fixed in one pass, not three runs.
std::vector<ArchitectureDisagreement>
check_architecture(const weights::SafeTensorFile &checkpoint,
                   const std::vector<ArchitectureClaim> &claims);

// The same, and fails with all of them named if any disagree.
void verify_architecture(const weights::SafeTensorFile &checkpoint,
                         const std::vector<ArchitectureClaim> &claims,
                         const std::filesystem::path &source);

} // namespace dif::training
