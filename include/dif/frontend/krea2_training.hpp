#pragma once

#include "dif/frontend/krea2.hpp"
#include "dif/opt/lora.hpp"
#include "dif/training/architecture.hpp"
#include "dif/training/config.hpp"
#include "dif/training/flow_matching.hpp"
#include "dif/training/session.hpp"

#include <string>
#include <utility>
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

// One Krea 2 training step, composed entirely from generic parts.
//
// Nothing below this comment is a Krea decision except which sites a LoRA
// adapts and which tensors the checkpoint fills. The denoiser is the same
// forward graph that samples; the objective is difcore's; the low-rank
// adaptation is difcore's; the optimizer, the plan, the persistent state and
// the record are difcore's. If a second model needs anything here that is
// not in that list, the foundation is not finished.
struct Krea2TrainingBuild {
  training::TrainingPlan plan;
  Krea2Config config;

  // What a step supplies. The noisy sample is NOT here: it is produced
  // inside the program by the interpolation, which is the point.
  std::uint32_t clean_latents{};
  std::uint32_t noise{};
  std::uint32_t timestep{};
  std::uint32_t context{};
  std::uint32_t positions{};
  std::uint32_t validity_mask{};
  std::uint32_t loss{};
  std::uint32_t velocity{};
  std::uint32_t target{};

  // The adapters, with the checkpoint tensor each one adapts, so an exported
  // adapter can be named the way every other tool expects to read it.
  std::vector<opt::LoraSite> sites;
  std::vector<std::string> site_names;
  // Every weight the checkpoint has to fill, by name. A trainer that guesses
  // this list is a trainer that silently trains against zeros.
  std::vector<std::pair<std::uint32_t, std::string>> frozen;
};

// Reads everything from the config. When the config names a checkpoint that
// exists, its architecture claims are verified against that checkpoint's
// header first -- so a wrong dimension is caught before a graph is built for
// it, not after.
Krea2TrainingBuild
build_krea2_training(const training::TrainingConfig &config);

} // namespace dif::frontend
