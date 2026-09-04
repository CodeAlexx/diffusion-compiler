#pragma once

#include "dif/frontend/krea2.hpp"
#include "dif/training/architecture.hpp"
#include "dif/training/config.hpp"

#include <vector>

namespace dif::frontend {

// Krea 2 as a consumer of the generic training machinery.
//
// This file is the whole model-specific part of training Krea: it reads the
// architecture the config claims, says which checkpoint tensor settles each
// claim, and names the tensors a LoRA adapts. Everything else -- the plan, the
// session, the optimizer, the record -- is difcore's and is not mentioned
// here. That is the test of whether the foundation is real: adding the next
// model should mean writing a file this size, not another trainer.
struct Krea2TrainingArchitecture {
  Krea2Config config;
  std::vector<training::ArchitectureClaim> claims;
};

// Reads the architecture from the config. Every dimension comes from the
// file; none is assumed. The claims are what makes that safe to do.
Krea2TrainingArchitecture
krea2_training_architecture(const training::TrainingConfig &config);

// The eight tensors a LoRA adapts in each of the 28 blocks: the four
// attention projections, the attention gate, and the three MLP projections.
// Taken from the checkpoint's own per-block 2-D tensors rather than from a
// list somebody kept in step by hand.
std::vector<std::string> krea2_lora_sites();

} // namespace dif::frontend
