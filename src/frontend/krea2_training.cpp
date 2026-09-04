#include "dif/frontend/krea2_training.hpp"

#include "dif/compiler/int8.hpp"
#include "dif/support/error.hpp"
#include "dif/weights/safetensors.hpp"

#include <filesystem>
#include <map>
#include <string>

namespace dif::frontend {
namespace {

using training::ArchitectureClaim;

ArchitectureClaim claim(std::string key, std::uint64_t claimed,
                        std::string tensor, std::size_t dimension,
                        std::uint64_t built, std::uint64_t multiple = 1U) {
  ArchitectureClaim value;
  value.key = std::move(key);
  value.claimed = claimed;
  value.tensor = std::move(tensor);
  value.dimension = dimension;
  value.built = built;
  value.multiple = multiple;
  return value;
}

} // namespace

std::vector<std::string> krea2_lora_sites() {
  // Ordered as the checkpoint orders them, so an adapter's tensors come out
  // in an order a reader can predict.
  return {"attn.wq.weight",   "attn.wk.weight",   "attn.wv.weight",
          "attn.wo.weight",   "attn.gate.weight", "mlp.gate.weight",
          "mlp.up.weight",    "mlp.down.weight"};
}

Krea2TrainingArchitecture
krea2_training_architecture(const training::TrainingConfig &config) {
  Krea2TrainingArchitecture architecture;
  auto &model = architecture.config;

  // Every one of these is READ, not assumed. A config that omits one is an
  // error naming the key, because a training run whose architecture came
  // half from a file and half from a header is not reproducible from the
  // file.
  const auto inner_dim = config.u64("inner_dim");
  const auto text_dim = config.u64("joint_attention_dim");
  const auto layers = config.u64("num_single");
  const auto heads = config.u64("num_heads");
  const auto head_dim = config.u64("head_dim");
  const auto mlp_hidden = config.u64("mlp_hidden");
  const auto timestep_dim = config.u64("timestep_dim");
  const auto in_channels = config.u64("in_channels");
  const auto out_channels = config.u64("out_channels");
  const auto double_blocks = config.u64_or("num_double", 0U);

  // Runtime geometry, which genuinely varies per run. A square resolution is
  // the common case and the only one a single "resolution" field can state;
  // a run that wants a rectangle says so with width and height.
  model.batch = config.u64_or("batch_size", 1U);
  const auto run = training::read_run(config);
  const auto square = run.resolutions.empty() ? 1024U : run.resolutions.front();
  model.width = config.u64_or("width", square);
  model.height = config.u64_or("height", square);
  model.text_tokens = config.u64_or("text_tokens", model.text_tokens);

  auto &claims = architecture.claims;
  claims.push_back(claim("inner_dim", inner_dim, "blocks.0.attn.wq.weight",
                         0U, Krea2Config::kFeatures));
  claims.push_back(claim("joint_attention_dim", text_dim,
                         "txtfusion.layerwise_blocks.0.attn.wq.weight", 0U,
                         Krea2Config::kTextDim));
  claims.push_back(claim("num_heads", heads, "blocks.0.attn.wq.weight", 0U,
                         Krea2Config::kHeads, head_dim));
  claims.push_back(claim("head_dim", head_dim,
                         "blocks.0.attn.qknorm.qnorm.scale", 0U,
                         Krea2Config::kHeadDim));
  claims.push_back(claim("mlp_hidden", mlp_hidden, "blocks.0.mlp.gate.weight",
                         0U, Krea2Config::kMlpDim));
  claims.push_back(claim("timestep_dim", timestep_dim, "tmlp.0.weight", 1U,
                         Krea2Config::kTimestepDim));
  claims.push_back(claim("in_channels", in_channels, "first.weight", 1U,
                         Krea2Config::kPatch * Krea2Config::kPatch *
                             Krea2Config::kLatentChannels));
  claims.push_back(claim("out_channels", out_channels,
                         "last.linear.weight", 0U,
                         Krea2Config::kPatch * Krea2Config::kPatch *
                             Krea2Config::kLatentChannels));
  // Grouped-query attention: the config does not name the KV head count, but
  // the checkpoint states it and the frontend depends on it, so it is checked
  // anyway. A claim nobody wrote is still a claim somebody relies on.
  claims.push_back(claim("kv_heads (implied)", Krea2Config::kKvHeads,
                         "blocks.0.attn.wk.weight", 0U,
                         Krea2Config::kKvHeads, head_dim));
  // Krea 2 is single-stream. A config asking for double-stream blocks is
  // describing a different model, and building 28 single blocks for it would
  // be answering a question nobody asked.
  if (double_blocks != 0U)
    fail("'num_double' in " + config.source().string() + " asks for " +
         std::to_string(double_blocks) +
         " double-stream blocks; Krea 2 is single-stream");
  claims.push_back(claim("num_single", layers, {}, 0U, Krea2Config::kLayers));
  // ...and the checkpoint has to actually contain the last block the config
  // asks for, which is what catches a layer count that is merely too large.
  claims.push_back(claim("num_single (last block)", inner_dim,
                         "blocks." + std::to_string(layers - 1U) +
                             ".prenorm.scale",
                         0U, 0U));
  return architecture;
}

} // namespace dif::frontend

namespace dif::frontend {

Krea2TrainingBuild
build_krea2_training(const training::TrainingConfig &config) {
  using ir::Opcode;

  Krea2TrainingBuild build;
  const auto architecture = krea2_training_architecture(config);
  build.config = architecture.config;

  const auto run = training::read_run(config);
  // The checkpoint is the authority on the architecture. Check the config
  // against it BEFORE building a graph, so a wrong dimension costs a message
  // rather than a 12-billion-parameter graph nobody can use.
  if (std::filesystem::exists(run.checkpoint))
    training::verify_architecture(weights::read_safetensors(run.checkpoint),
                                  architecture.claims, config.source());

  // Training keeps its weights resident. Streaming them would re-stage the
  // whole base through the host on every step, which is the cost the
  // resident format and the persistent state both exist to remove.
  build.config.streamed_constants = false;
  auto denoiser = make_krea2_denoiser(build.config);
  build.context = denoiser.context_input;
  build.positions = denoiser.positions_input;
  build.validity_mask = denoiser.validity_mask_input;
  build.rotary_pair_axes = denoiser.rotary_pair_axes;
  build.rotary_pair_indices = denoiser.rotary_pair_indices;
  build.rotary_axis_dims = denoiser.rotary_axis_dims;

  // Which Linears a LoRA adapts: the eight per block the checkpoint names,
  // found through the frontend's own provenance rather than by matching
  // shapes. A predicate over shapes adapts the wrong sites the first time an
  // architecture changes.
  std::map<std::string, std::uint32_t> weight_of;
  for (std::size_t index = 0U; index < denoiser.checkpoint_tensors.size();
       ++index)
    weight_of.emplace(denoiser.checkpoint_names[index],
                      denoiser.checkpoint_tensors[index]);
  std::map<std::uint32_t, std::uint32_t> linear_for_weight;
  for (const auto &operation : denoiser.program.operations)
    if (operation.opcode == Opcode::Linear && operation.inputs.size() >= 2U)
      linear_for_weight.emplace(operation.inputs[1], operation.id);

  opt::LoraSpec spec;
  spec.rank = run.lora_rank;
  spec.alpha = static_cast<double>(run.lora_alpha);
  if (spec.rank == 0U)
    fail("'lora_rank' in " + config.source().string() + " adapts nothing");
  std::vector<std::string> names;
  for (std::uint64_t block = 0U; block < build.config.kLayers; ++block)
    for (const auto &site : krea2_lora_sites()) {
      const auto name = "blocks." + std::to_string(block) + "." + site;
      const auto weight = weight_of.find(name);
      if (weight == weight_of.end())
        fail("the Krea 2 denoiser has no weight named '" + name +
             "', so a LoRA cannot adapt it");
      const auto linear = linear_for_weight.find(weight->second);
      if (linear == linear_for_weight.end())
        fail("'" + name + "' is not consumed by a linear operation");
      spec.operations.push_back(linear->second);
      names.push_back(name);
    }

  auto adapted = opt::insert_lora(denoiser.program, spec);
  build.sites = std::move(adapted.sites);
  build.site_names = std::move(names);

  // The resident format the config asked for, actually applied. Naming a
  // format and then building the graph in BF16 anyway is how a request like
  // this quietly does nothing.
  std::map<std::uint32_t, std::string> name_of_weight;
  for (std::size_t index = 0U; index < denoiser.checkpoint_tensors.size();
       ++index)
    name_of_weight.emplace(denoiser.checkpoint_tensors[index],
                           denoiser.checkpoint_names[index]);
  build.resident_format = run.resident_format;
  if (!run.resident_format.empty()) {
    if (run.resident_format != "int8" && run.resident_format != "int_w8a8")
      fail("'quantized_resident' asks for '" + run.resident_format +
           "', which this compiler does not consume directly. The formats "
           "it can feed to a matmul without converting first are 'int8' and "
           "'int_w8a8'; anything else would be dequantized once per linear "
           "per step, which is the cost a resident format exists to avoid.");
    // The 224 adapted base linears ARE the base: they are 12.16 of its 12.42
    // billion parameters.
    std::vector<std::uint32_t> frozen_linears;
    for (const auto &site : build.sites)
      frozen_linears.push_back(site.operation);
    auto resident =
        compiler::rewrite_int8_weight_only(adapted.program, frozen_linears);
    for (const auto &entry : resident.entries) {
      const auto found = name_of_weight.find(entry.source_tensor_id);
      if (found == name_of_weight.end())
        fail("a weight made resident has no checkpoint name");
      build.resident.push_back(
          {found->second, entry.weight_tensor_id, entry.scales_tensor_id});
      name_of_weight.erase(found);
    }
    build.resident_bytes_before = resident.bytes_before;
    build.resident_bytes_after = resident.bytes_after;
    adapted.program = std::move(resident.program);
  }

  // The objective. The denoiser's own timestep is reused rather than
  // duplicated, so the noise level it is told about is the one it was given.
  auto objective = training::add_flow_matching_loss(
      std::move(adapted.program), denoiser.image_tokens_input,
      denoiser.velocity_output, denoiser.timestep_input);
  build.clean_latents = objective.clean_input;
  build.noise = objective.noise_input;
  build.timestep = objective.timestep_input;
  build.loss = objective.loss_output;
  build.target = objective.target_output;
  build.velocity = denoiser.velocity_output;

  training::OptimizerHyperparameters hyperparameters = run.optimizer;
  // Adapters are F32, so no master copy is needed; the frozen base never
  // moves at all.
  hyperparameters.master_weights = false;
  build.plan = training::compile(objective.program, build.loss,
                                 adapted.parameters, hyperparameters);

  // Whatever was not made resident is bound as it comes off disk.
  for (const auto &[id, name] : name_of_weight)
    build.frozen.emplace_back(id, name);
  return build;
}

} // namespace dif::frontend
