#include "dif/frontend/krea2_training.hpp"

#include "dif/support/error.hpp"

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
