#include "dif/frontend/flux2.hpp"
#include "dif/frontend/flux2_vae.hpp"
#include "dif/frontend/qwen3vl_conditioner.hpp"
#include "dif/ir/verify.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

} // namespace

int main() {
  try {
  {
      const auto config =
          dif::frontend::make_flux2_klein_9b_conditioner_config(1U);
      const auto build = dif::frontend::build_qwen3vl_conditioner_program(
          512U, config);
      dif::ir::verify(build.program);
      expect(build.attention_operations == 1U,
             "depth-one FLUX.2 conditioner has one attention");
      expect(build.linear_operations == 7U,
             "depth-one FLUX.2 conditioner has seven linears");
      expect(build.conditioning_output_ids.size() == 1U,
             "depth-one FLUX.2 conditioner captures one raw state");
      expect(build.bindings.front().name == "model.embed_tokens.weight",
             "ordinary Qwen3 checkpoint embedding prefix");
      expect(std::any_of(build.bindings.begin(), build.bindings.end(),
                         [](const auto &binding) {
                           return binding.name ==
                                  "model.layers.0.self_attn.q_proj.weight";
                         }),
             "ordinary Qwen3 layer prefix");
    }

    {
      const auto config =
          dif::frontend::make_flux2_klein_9b_conditioner_config();
      const auto build = dif::frontend::build_qwen3vl_conditioner_program(
          512U, config);
      dif::ir::verify(build.program);
      expect(build.attention_operations == 27U,
             "observable FLUX.2 conditioner stops after tap 27");
      expect(build.conditioning_output_ids.size() == 3U,
             "FLUX.2 conditioner exposes taps 9, 18, and 27");
      const auto *output = build.program.tensor(build.conditioning_output_id);
      expect(output != nullptr &&
                 output->dims == std::vector<std::uint64_t>{512U, 12288U},
             "FLUX.2 conditioner concatenates to [512,12288]");
      const auto concat = std::find_if(
          build.program.operations.begin(), build.program.operations.end(),
          [](const auto &operation) {
            return operation.opcode == dif::ir::Opcode::Concat;
          });
      expect(concat != build.program.operations.end() &&
                 concat->inputs == build.conditioning_output_ids,
             "FLUX.2 hidden-state concatenation stays in DiffIR");
    }

    {
      dif::frontend::Flux2KleinDoubleBlockConfig config;
      config.image_tokens = 8U;
      config.text_tokens = 4U;
      const auto build =
          dif::frontend::make_flux2_klein_9b_double_block(config);
      dif::ir::verify(build.program);
      expect(build.checkpoint_tensors.size() == 12U,
             "FLUX.2 double block owns the creator twelve-tensor inventory");
      expect(build.generated_constants.size() == 3U,
             "FLUX.2 double block derives generic four-axis RoPE maps");
      const auto *image = build.program.tensor(build.image_output);
      const auto *text = build.program.tensor(build.text_output);
      expect(image != nullptr &&
                 image->dims == std::vector<std::uint64_t>{8U, 4096U},
             "FLUX.2 image output preserves real hidden width");
      expect(text != nullptr &&
                 text->dims == std::vector<std::uint64_t>{4U, 4096U},
             "FLUX.2 text output preserves real hidden width");
      expect(std::all_of(build.program.operations.begin(),
                         build.program.operations.end(), [](const auto &op) {
                           return op.opcode != dif::ir::Opcode::H3AdaLNSelect &&
                                  op.opcode !=
                                      dif::ir::Opcode::H3DeinterleaveQkv &&
                                  op.opcode !=
                                      dif::ir::Opcode::H3DeinterleaveQkvWeight;
                         }),
             "FLUX.2 block uses only shared DiffIR semantics");
    }

    {
      dif::frontend::Flux2KleinSingleBlockConfig config;
      config.tokens = 13U;
      config.block_index = 0U;
      const auto build =
          dif::frontend::make_flux2_klein_9b_single_block(config);
      expect(build.checkpoint_tensors.size() == 4U,
             "FLUX.2 single block checkpoint tensor count");
      expect(build.generated_constants.size() == 3U,
             "FLUX.2 single block generated constant count");
      expect(build.program.tensor(build.sequence_output) != nullptr,
             "FLUX.2 single block output exists");
    }

    {
      dif::frontend::Flux2KleinTransformerConfig config;
      config.image_tokens = 8U;
      config.text_tokens = 5U;
      config.streamed_constants = true;
      const auto build =
          dif::frontend::make_flux2_klein_9b_transformer(config);
      expect(build.checkpoint_tensors.size() == 201U,
             "FLUX.2 complete transformer binds all 201 checkpoint tensors");
      expect(build.checkpoint_names.front() == "img_in.weight" &&
                 build.checkpoint_names.back() == "final_layer.linear.weight",
             "FLUX.2 transformer checkpoint inventory preserves creator names");
      const auto *prediction = build.program.tensor(build.prediction_output);
      expect(prediction != nullptr &&
                 prediction->dims == std::vector<std::uint64_t>{8U, 128U},
             "FLUX.2 full transformer returns latent velocity rows");
      expect(std::all_of(build.checkpoint_tensors.begin(),
                         build.checkpoint_tensors.end(),
                         [&](std::uint32_t id) {
                           const auto *tensor = build.program.tensor(id);
                           return tensor != nullptr &&
                                  tensor->has_role(dif::ir::TensorRole::Streamed);
                         }),
             "FLUX.2 5080 plan marks checkpoint weights streamed by policy");
    }

    {
      dif::frontend::Flux2KleinTransformerConfig config;
      config.batch_size = 2U;
      config.image_tokens = 8U;
      config.text_tokens = 5U;
      const auto build =
          dif::frontend::make_flux2_klein_9b_transformer(config);
      dif::ir::verify(build.program);
      expect(build.program.tensor(build.latent_input)->dims ==
                 std::vector<std::uint64_t>{2U, 8U, 128U} &&
                 build.program.tensor(build.conditioning_input)->dims ==
                     std::vector<std::uint64_t>{2U, 5U, 12288U} &&
                 build.program.tensor(build.prediction_output)->dims ==
                     std::vector<std::uint64_t>{2U, 8U, 128U},
             "FLUX.2 creator CFG batch is one shared transformer program");
    }

    {
      dif::frontend::Flux2VaeConfig config;
      config.latent_height = 2U;
      config.latent_width = 2U;
      const auto build = dif::frontend::make_flux2_vae_decoder(config);
      expect(build.weights.size() == 142U,
             "FLUX.2 VAE decoder binds its complete creator tensor inventory");
      const auto *output = build.program.tensor(build.clamped_output);
      expect(output != nullptr &&
                 output->dims == std::vector<std::uint64_t>{1U, 3U, 32U, 32U},
             "FLUX.2 VAE maps 2x2 token latents to 32x32 pixels");
    }

    {
      const auto build =
          dif::frontend::make_flux2_klein_base_cfg_euler_step({4U, 8U});
      expect(build.program.operations.size() == 7U,
             "FLUX.2 CFG/Euler policy uses shared elementwise and Euler ops");
      const auto guided = std::find_if(
          build.program.operations.begin(), build.program.operations.end(),
          [&](const auto &operation) {
            return !operation.outputs.empty() &&
                   operation.outputs.front() ==
                       build.guided_velocity_output;
          });
      expect(guided != build.program.operations.end() &&
                 guided->opcode == dif::ir::Opcode::Add &&
                 guided->inputs.front() ==
                     build.unconditional_velocity_input,
             "FLUX.2 CFG is uncond + guidance * (cond - uncond)");
      expect(build.program.operations.back().opcode ==
                 dif::ir::Opcode::EulerVelocityStep,
             "FLUX.2 CFG policy ends at the generic eager Euler semantic");
    }

    {
      // Geometry: the Klein default stays what it was, and the [dev]
      // geometry builds a verified 8 + 48 block program with the guidance
      // embedding and Dev-width checkpoint tensors.
      dif::frontend::Flux2KleinTransformerConfig klein;
      klein.image_tokens = 64U;
      klein.text_tokens = 16U;
      const auto klein_build =
          dif::frontend::make_flux2_klein_9b_transformer(klein);
      expect(klein_build.guidance_input == 0U,
             "klein geometry has no guidance embedding input");
      expect(std::none_of(klein_build.checkpoint_names.begin(),
                          klein_build.checkpoint_names.end(),
                          [](const std::string &name) {
                            return name.rfind("guidance_in", 0) == 0;
                          }),
             "klein geometry binds no guidance_in weights");
      dif::frontend::Flux2KleinTransformerConfig dev;
      dev.geometry = dif::frontend::flux2_dev_geometry();
      dev.image_tokens = 64U;
      dev.text_tokens = 16U;
      dev.double_depth = 8U;
      dev.single_depth = 48U;
      const auto dev_build =
          dif::frontend::make_flux2_klein_9b_transformer(dev);
      expect(dev_build.guidance_input != 0U,
             "dev geometry exposes the guidance input");
      const auto has = [&](const char *name) {
        return std::find(dev_build.checkpoint_names.begin(),
                         dev_build.checkpoint_names.end(),
                         std::string(name)) != dev_build.checkpoint_names.end();
      };
      expect(has("guidance_in.in_layer.weight") &&
                 has("guidance_in.out_layer.weight") &&
                 has("single_blocks.47.linear2.weight") &&
                 has("double_blocks.7.txt_mlp.2.weight"),
             "dev geometry binds guidance_in and all 8 + 48 blocks");
      const auto qkv = std::find(dev_build.checkpoint_names.begin(),
                                 dev_build.checkpoint_names.end(),
                                 std::string("double_blocks.0.img_attn.qkv.weight"));
      expect(qkv != dev_build.checkpoint_names.end(),
             "dev geometry binds the first double-block qkv");
      if (qkv != dev_build.checkpoint_names.end()) {
        const auto id = dev_build.checkpoint_tensors.at(static_cast<std::size_t>(
            std::distance(dev_build.checkpoint_names.begin(), qkv)));
        const auto *desc = dev_build.program.tensor(id);
        expect(desc && desc->dims == std::vector<std::uint64_t>{18432U, 6144U},
               "dev qkv weight is [18432, 6144]");
      }
      const auto *conditioning = dev_build.program.tensor(dev_build.conditioning_input);
      expect(conditioning && conditioning->dims.back() == 15360U,
             "dev conditioning input is 15360 wide");
      dif::ir::verify(dev_build.program);
    }
    {
      // [dev] conditioner: Mistral tower, no QK-norm, language_model.model
      // keys, three hidden states concatenated to 15360.
      const auto config = dif::frontend::make_flux2_dev_conditioner_config();
      expect(config.executed_layers == 30U && !config.qk_norm &&
                 config.selected_hidden_states ==
                     std::vector<std::uint64_t>{10U, 20U, 30U} &&
                 config.concatenate_selected_hidden_states,
             "dev conditioner config selects states 10/20/30 without QK-norm");
      const auto build = dif::frontend::build_qwen3vl_conditioner_program(512U, config);
      expect(std::none_of(build.bindings.begin(), build.bindings.end(),
                          [](const auto &b) {
                            return b.name.find("q_norm") != std::string::npos ||
                                   b.name.find("k_norm") != std::string::npos;
                          }),
             "dev conditioner binds no QK-norm weights");
      expect(std::any_of(build.bindings.begin(), build.bindings.end(),
                         [](const auto &b) {
                           return b.name ==
                                  "language_model.model.layers.29.mlp.down_proj.weight";
                         }),
             "dev conditioner binds layer 29 under language_model.model");
      expect(build.attention_operations == 30U,
             "dev conditioner executes 30 attention layers");
      const auto concat = std::find_if(
          build.program.operations.begin(), build.program.operations.end(),
          [](const dif::ir::Operation &op) {
            return op.opcode == dif::ir::Opcode::Concat;
          });
      const auto *output = concat == build.program.operations.end()
                               ? nullptr
                               : build.program.tensor(concat->outputs.front());
      expect(output != nullptr && output->dims.back() == 15360U,
             "dev conditioning output is 15360 wide");
      const auto rotary = std::count_if(
          build.program.operations.begin(), build.program.operations.end(),
          [](const dif::ir::Operation &op) {
            return op.opcode == dif::ir::Opcode::RotaryApply;
          });
      expect(rotary == 60U, "dev conditioner applies plain RoPE to Q and K of every layer");
      dif::ir::verify(build.program);
    }
    if (failures != 0) {
      std::cerr << failures << " failure(s)\n";
      return 1;
    }
    std::cout << "FLUX.2 conditioner frontend tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FLUX.2 conditioner frontend test: " << error.what()
              << "\n";
    return 1;
  }
}
