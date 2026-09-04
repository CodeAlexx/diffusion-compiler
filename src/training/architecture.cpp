#include "dif/training/architecture.hpp"

#include "dif/support/error.hpp"

#include <sstream>

namespace dif::training {

std::vector<ArchitectureDisagreement>
check_architecture(const weights::SafeTensorFile &checkpoint,
                   const std::vector<ArchitectureClaim> &claims) {
  std::vector<ArchitectureDisagreement> disagreements;
  for (const auto &claim : claims) {
    ArchitectureDisagreement problem;
    problem.key = claim.key;
    problem.claimed = claim.claimed;
    problem.built = claim.built;

    if (claim.built != 0U && claim.built != claim.claimed) {
      problem.detail = claim.built_by + " " + std::to_string(claim.built);
      disagreements.push_back(problem);
      continue;
    }
    if (claim.tensor.empty())
      continue;

    const auto *entry = checkpoint.find(claim.tensor);
    if (entry == nullptr) {
      problem.detail = "the checkpoint has no tensor '" + claim.tensor + "'";
      disagreements.push_back(problem);
      continue;
    }
    if (claim.dimension >= entry->dims.size()) {
      problem.detail = "'" + claim.tensor + "' has " +
                       std::to_string(entry->dims.size()) +
                       " dimensions, so dimension " +
                       std::to_string(claim.dimension) + " does not exist";
      disagreements.push_back(problem);
      continue;
    }
    const auto found = entry->dims[claim.dimension];
    problem.checkpoint = found;
    const auto expected = claim.claimed * claim.multiple;
    if (found != expected) {
      std::ostringstream detail;
      detail << "'" << claim.tensor << "' dimension " << claim.dimension
             << " is " << found << ", not " << expected;
      if (claim.multiple != 1U)
        detail << " (" << claim.claimed << " x " << claim.multiple << ")";
      problem.detail = detail.str();
      disagreements.push_back(problem);
    }
  }
  return disagreements;
}

void verify_architecture(const weights::SafeTensorFile &checkpoint,
                         const std::vector<ArchitectureClaim> &claims,
                         const std::filesystem::path &source) {
  const auto disagreements = check_architecture(checkpoint, claims);
  if (disagreements.empty())
    return;
  std::ostringstream message;
  message << source.string() << " does not describe "
          << checkpoint.path.string() << ":";
  for (const auto &problem : disagreements)
    message << "\n  '" << problem.key << "' says " << problem.claimed
            << " but " << problem.detail;
  fail(message.str());
}

} // namespace dif::training
