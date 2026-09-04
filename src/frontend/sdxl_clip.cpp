#include "dif/frontend/sdxl_clip.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dif::frontend {
namespace {

using ir::Attribute;
using ir::AttrKey;
using ir::DType;
using ir::Opcode;
using ir::TensorRole;

struct LayerNames {
  std::string norm1, norm2, q, k, v, out, fc1, fc2;
  bool fused_qkv{};
};

class Builder {
public:
  explicit Builder(ClipTextTowerConfig value_config)
      : config(std::move(value_config)) {
    if (config.hidden_size == 0U || config.heads == 0U ||
        config.hidden_size % config.heads != 0U || config.layers == 0U ||
        config.positions == 0U || config.vocabulary == 0U)
      fail("CLIP text tower geometry is invalid");
    if (config.executed_layers == 0U || config.executed_layers > config.layers ||
        config.hidden_layers == 0U ||
        config.hidden_layers > config.executed_layers)
      fail("CLIP text tower layer selection is invalid");
    if (config.pooled_output && config.executed_layers != config.layers)
      fail("CLIP pooled output needs the full stack executed");
    if (config.attention_implementation == 1U && config.dtype != DType::F32)
      fail("generated exact attention is the F32 parity form; use cuDNN "
           "(2) for bf16/f16");
    build.config = config;
  }

  ClipTextTowerBuild finish() {
    const auto hidden = config.hidden_size;
    const auto positions = config.positions;
    build.token_ids_input =
        tensor(DType::I32, TensorRole::Input, {positions});
    const auto token_table = checkpoint(
        config.layout == ClipCheckpointLayout::OpenClip
            ? "token_embedding.weight"
            : "embeddings.token_embedding.weight",
        {config.vocabulary, hidden});
    const auto position_table = checkpoint(
        config.layout == ClipCheckpointLayout::OpenClip
            ? "positional_embedding"
            : "embeddings.position_embedding.weight",
        {positions, hidden});
    auto tokens = tensor(config.dtype, TensorRole::Internal, {positions, hidden});
    operation(Opcode::GatherRows, {token_table, build.token_ids_input},
              {tokens});
    auto x = same(tokens);
    operation(Opcode::Add, {tokens, position_table}, {x});
    capture("embeddings", x);

    for (std::uint64_t layer = 0U; layer < config.executed_layers; ++layer) {
      x = encoder_layer(x, layer);
      if (layer == 0U)
        capture("layer_0", x);
      if (layer + 1U == config.hidden_layers) {
        build.hidden_output = x;
        mark_output(x);
        build.boundaries.emplace_back("hidden", x);
      }
    }

    if (config.pooled_output) {
      const auto normalized = layer_norm(
          x, config.layout == ClipCheckpointLayout::OpenClip
                 ? "ln_final"
                 : "final_layer_norm");
      capture("final_norm", normalized);
      build.pooled_row_input = tensor(DType::I32, TensorRole::Input, {1U});
      const auto row = tensor(config.dtype, TensorRole::Internal, {1U, hidden});
      operation(Opcode::GatherRows, {normalized, build.pooled_row_input}, {row});
      const auto projection =
          config.layout == ClipCheckpointLayout::OpenClip
              ? checkpoint("text_projection", {hidden, hidden},
                           ClipWeightTransform::Transpose)
              : checkpoint("text_projection.weight", {hidden, hidden});
      build.pooled_output =
          tensor(config.dtype, TensorRole::Output, {1U, hidden});
      operation(Opcode::Linear, {row, projection}, {build.pooled_output});
      build.boundaries.emplace_back("pooled", build.pooled_output);
    }
    ir::verify(build.program);
    return std::move(build);
  }

private:
  ClipTextTowerConfig config;
  ClipTextTowerBuild build;
  std::uint32_t next_tensor{1U};
  std::uint32_t next_operation{1U};

  std::uint32_t tensor(DType dtype, std::uint32_t roles,
                       std::vector<std::uint64_t> dims) {
    const auto id = next_tensor++;
    build.program.tensors.push_back({id, dtype, roles, std::move(dims)});
    return id;
  }

  const ir::TensorDesc &description(std::uint32_t id) const {
    // Ids are handed out sequentially and pushed in the same order, so the
    // descriptor is at id - 1. Program::tensor is a linear scan and the
    // builders look up thousands of times.
    if (id == 0U || id > build.program.tensors.size() ||
        build.program.tensors[id - 1U].id != id)
      fail("CLIP text tower builder lost a tensor descriptor");
    return build.program.tensors[id - 1U];
  }

  std::uint32_t same(std::uint32_t source) {
    const auto dims = description(source).dims;
    return tensor(config.dtype, TensorRole::Internal, dims);
  }

  void operation(Opcode opcode, std::vector<std::uint32_t> inputs,
                 std::vector<std::uint32_t> outputs,
                 std::vector<Attribute> attributes = {}) {
    build.program.operations.push_back({next_operation++, opcode,
                                        std::move(inputs), std::move(outputs),
                                        std::move(attributes)});
  }

  std::uint32_t checkpoint(
      const std::string &name, std::vector<std::uint64_t> dims,
      ClipWeightTransform transform = ClipWeightTransform::Direct) {
    const auto id = tensor(config.dtype, TensorRole::Constant, std::move(dims));
    build.weights.push_back({id, config.checkpoint_prefix + name, transform});
    return id;
  }

  LayerNames names(std::uint64_t layer) const {
    LayerNames result;
    if (config.layout == ClipCheckpointLayout::OpenClip) {
      const auto prefix = "transformer.resblocks." + std::to_string(layer) + ".";
      result.norm1 = prefix + "ln_1";
      result.norm2 = prefix + "ln_2";
      result.q = result.k = result.v = prefix + "attn.in_proj";
      result.out = prefix + "attn.out_proj";
      result.fc1 = prefix + "mlp.c_fc";
      result.fc2 = prefix + "mlp.c_proj";
      result.fused_qkv = true;
    } else {
      const auto prefix = "encoder.layers." + std::to_string(layer) + ".";
      result.norm1 = prefix + "layer_norm1";
      result.norm2 = prefix + "layer_norm2";
      result.q = prefix + "self_attn.q_proj";
      result.k = prefix + "self_attn.k_proj";
      result.v = prefix + "self_attn.v_proj";
      result.out = prefix + "self_attn.out_proj";
      result.fc1 = prefix + "mlp.fc1";
      result.fc2 = prefix + "mlp.fc2";
    }
    return result;
  }

  std::uint32_t layer_norm(std::uint32_t input, const std::string &prefix) {
    const auto hidden = description(input).dims.back();
    const auto weight = checkpoint(prefix + ".weight", {hidden});
    const auto bias = checkpoint(prefix + ".bias", {hidden});
    const auto output = same(input);
    operation(Opcode::LayerNorm, {input, weight, bias}, {output},
              {Attribute::f64(AttrKey::Epsilon, config.layer_norm_epsilon)});
    return output;
  }

  std::uint32_t linear(std::uint32_t input, const std::string &prefix,
                       std::uint64_t out_features,
                       ClipWeightTransform transform) {
    const auto rows = description(input).dims[0];
    const auto in_features = description(input).dims[1];
    // Fused in_proj rows: the checkpoint holds [3C, C] / [3C]; the binder
    // slices the requested third.
    const auto weight = checkpoint(prefix + (transform == ClipWeightTransform::Direct
                                                 ? ".weight"
                                                 : "_weight"),
                                   {out_features, in_features}, transform);
    const auto bias = checkpoint(prefix + (transform == ClipWeightTransform::Direct
                                               ? ".bias"
                                               : "_bias"),
                                 {out_features}, transform);
    const auto output =
        tensor(config.dtype, TensorRole::Internal, {rows, out_features});
    operation(Opcode::Linear, {input, weight, bias}, {output});
    return output;
  }

  std::uint32_t encoder_layer(std::uint32_t x, std::uint64_t layer) {
    const auto hidden = config.hidden_size;
    const auto positions = config.positions;
    const auto heads = config.heads;
    const auto head_dim = hidden / heads;
    const auto layer_names = names(layer);
    const auto normalized = layer_norm(x, layer_names.norm1);
    const auto q = linear(normalized, layer_names.q, hidden,
                          layer_names.fused_qkv ? ClipWeightTransform::FusedRowsQ
                                                : ClipWeightTransform::Direct);
    const auto k = linear(normalized, layer_names.k, hidden,
                          layer_names.fused_qkv ? ClipWeightTransform::FusedRowsK
                                                : ClipWeightTransform::Direct);
    const auto v = linear(normalized, layer_names.v, hidden,
                          layer_names.fused_qkv ? ClipWeightTransform::FusedRowsV
                                                : ClipWeightTransform::Direct);
    auto head_view = [&](std::uint32_t source) {
      const auto view = tensor(config.dtype, TensorRole::Internal,
                               {positions, heads, head_dim});
      operation(Opcode::Reshape, {source}, {view});
      return view;
    };
    const auto attended = tensor(config.dtype, TensorRole::Internal,
                                 {positions, heads, head_dim});
    operation(Opcode::Attention, {head_view(q), head_view(k), head_view(v)},
              {attended},
              {Attribute::u64(AttrKey::Heads, heads),
               Attribute::u64(AttrKey::KvHeads, heads),
               Attribute::u64(AttrKey::HeadDim, head_dim),
               Attribute::f64(AttrKey::AttentionScale,
                              1.0 / std::sqrt(static_cast<double>(head_dim))),
               Attribute::boolean(AttrKey::Causal, true),
               Attribute::u64(AttrKey::Implementation,
                              config.attention_implementation)});
    const auto flat = tensor(config.dtype, TensorRole::Internal,
                             {positions, hidden});
    operation(Opcode::Reshape, {attended}, {flat});
    const auto projected =
        linear(flat, layer_names.out, hidden, ClipWeightTransform::Direct);
    auto residual = same(x);
    operation(Opcode::Add, {x, projected}, {residual});
    const auto normalized2 = layer_norm(residual, layer_names.norm2);
    const auto expanded = linear(normalized2, layer_names.fc1,
                                 config.intermediate_size,
                                 ClipWeightTransform::Direct);
    const auto activated = same(expanded);
    operation(Opcode::Gelu, {expanded}, {activated},
              {Attribute::u64(AttrKey::Approximation,
                              static_cast<std::uint64_t>(config.activation))});
    const auto contracted =
        linear(activated, layer_names.fc2, hidden, ClipWeightTransform::Direct);
    auto output = same(residual);
    operation(Opcode::Add, {residual, contracted}, {output});
    return output;
  }

  void capture(const std::string &name, std::uint32_t value) {
    if (!config.capture_boundaries)
      return;
    mark_output(value);
    build.boundaries.emplace_back(name, value);
  }

  void mark_output(std::uint32_t value) {
    if (value == 0U || value > build.program.tensors.size() ||
        build.program.tensors[value - 1U].id != value)
      fail("CLIP text tower output lost its tensor");
    build.program.tensors[value - 1U].roles |= ir::TensorRole::Output;
  }
};

} // namespace

ClipTextTowerBuild make_clip_text_tower(const ClipTextTowerConfig &config) {
  return Builder(config).finish();
}

ClipTextTowerConfig sdxl_clip_l_config() {
  ClipTextTowerConfig config;
  config.hidden_size = 768U;
  config.layers = 12U;
  config.heads = 12U;
  config.intermediate_size = 3072U;
  config.activation = ir::GeluApproximation::QuickSigmoid;
  // hidden_states[-2]: the residual after 11 of 12 layers; the pooled
  // vector of CLIP-L is unused by SDXL, so the last layer never runs.
  config.executed_layers = 11U;
  config.hidden_layers = 11U;
  config.pooled_output = false;
  config.layout = ClipCheckpointLayout::HuggingFace;
  config.checkpoint_prefix =
      "conditioner.embedders.0.transformer.text_model.";
  return config;
}

ClipTextTowerConfig sdxl_clip_g_config() {
  ClipTextTowerConfig config;
  config.hidden_size = 1280U;
  config.layers = 32U;
  config.heads = 20U;
  config.intermediate_size = 5120U;
  config.activation = ir::GeluApproximation::ExactErf;
  // hidden_states[-2] after 31 layers; the pooled vector needs all 32.
  config.executed_layers = 32U;
  config.hidden_layers = 31U;
  config.pooled_output = true;
  config.layout = ClipCheckpointLayout::OpenClip;
  config.checkpoint_prefix = "conditioner.embedders.1.model.";
  return config;
}

} // namespace dif::frontend
