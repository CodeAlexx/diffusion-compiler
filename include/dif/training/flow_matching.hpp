#pragma once

#include "dif/ir/ir.hpp"

#include <cstdint>
#include <vector>

namespace dif::training {

// The flow-matching objective, added to a forward graph that does not know
// it is being trained.
//
// This is generic because the objective is. Krea 2, FLUX and SD3 all train
// the same way: interpolate between a clean latent and noise, ask the model
// for the velocity, and score it against the straight-line velocity that
// actually connects them.
//
//   difference = noise - clean          the target velocity
//   noised     = clean + t * difference the model's input
//   loss       = mean((prediction - difference)^2)
//
// The timestep is per sample, not per batch: t has one value per batch row
// and is broadcast over the tokens. A single t for a whole batch would make
// every sample in it see the same noise level, which is a different -- and
// worse -- objective, not merely a simpler one.
//
// The model's sample input STOPS being an input. It is now produced by the
// interpolation, and the caller supplies the clean latent and the noise
// instead. That is the whole trick: the forward graph is untouched, and the
// tensor it read its sample from is now fed from inside the program, so
// noising costs no host round trip and no second program.
struct FlowMatchingBuild {
  ir::Program program;
  // What a step supplies.
  std::uint32_t clean_input{};
  std::uint32_t noise_input{};
  // The timestep the step supplies. When the forward graph already had one,
  // this IS that tensor rather than a second one beside it.
  std::uint32_t timestep_input{};
  // What it produces. The loss is what the optimizer differentiates; the
  // target and the noised sample are outputs because a gate has to be able
  // to look at them, and a value nobody can inspect is a value nobody can
  // check.
  std::uint32_t loss_output{};
  std::uint32_t target_output{};
  std::uint32_t noised_output{};
};

// `sample_input` is the tensor the forward graph reads its noisy sample
// from, and `prediction` is the velocity it produces. Both are named by the
// frontend, which is the only thing that knows which is which.
//
// `model_timestep` is the timestep the forward graph ALREADY takes. Naming
// it means the program has exactly one timestep tensor, so the noise level
// the interpolation uses and the noise level the model is told about cannot
// drift apart -- a disagreement that trains a model to predict the velocity
// at a time it was never shown, and that no shape check would ever catch.
// Zero adds a separate timestep input, which is right only for a forward
// graph that takes none.
FlowMatchingBuild add_flow_matching_loss(ir::Program forward,
                                         std::uint32_t sample_input,
                                         std::uint32_t prediction,
                                         std::uint32_t model_timestep = 0U);

} // namespace dif::training
