#include "dif/frontend/h3_audio_encoder.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>

namespace dif::frontend {
namespace {
using ir::AttrKey;
using ir::Attribute;
using ir::DType;
using ir::Opcode;
using ir::TensorRole;
using Shape = std::vector<std::uint64_t>;

std::uint64_t mul(std::uint64_t a, std::uint64_t b) {
  if (b && a > std::numeric_limits<std::uint64_t>::max() / b)
    fail("audio encoder dimension overflow");
  return a * b;
}
std::uint64_t integer(const json::Value &value) {
  const auto n = value.number();
  if (!std::isfinite(n) || n < 1 || n > 1000000000 || std::floor(n) != n)
    fail("audio encoder configuration requires positive integral dimensions");
  return static_cast<std::uint64_t>(n);
}
const json::Value &required(const json::Value &doc, const char *key) {
  const auto *value = doc.find(key);
  if (!value) fail(std::string("audio encoder config missing ") + key);
  return *value;
}

struct Builder {
  H3AudioEncoderBuild result;
  std::uint32_t tensor(Shape dims, DType dtype = DType::F32,
                       std::uint32_t role = TensorRole::Internal) {
    const auto id = static_cast<std::uint32_t>(result.program.tensors.size() + 1);
    result.program.tensors.push_back({id, dtype, role, std::move(dims)});
    return id;
  }
  Shape shape(std::uint32_t id) const { return result.program.tensor(id)->dims; }
  void op(Opcode code, std::vector<std::uint32_t> in, std::uint32_t out,
          std::vector<Attribute> attrs = {}) {
    result.program.operations.push_back({
        static_cast<std::uint32_t>(result.program.operations.size() + 1),
        code, std::move(in), {out}, std::move(attrs)});
  }
  std::uint32_t weight(std::string name, Shape dims, bool wn = false) {
    const auto id = tensor(std::move(dims), DType::F32, TensorRole::Constant);
    result.bindings.push_back({id, std::move(name), wn});
    return id;
  }
  std::uint32_t zero(Shape dims) {
    const auto id = tensor(std::move(dims), DType::F32, TensorRole::Constant);
    const auto *desc = result.program.tensor(id);
    runtime::Tensor value{DType::F32, desc->dims, {}};
    value.bytes.resize(mul(desc->element_count(), sizeof(float)), 0);
    result.generated_constants.emplace(id, std::move(value));
    return id;
  }
  std::uint32_t reshape(std::uint32_t x, Shape dims) {
    const auto y = tensor(std::move(dims), result.program.tensor(x)->dtype);
    op(Opcode::Reshape, {x}, y);
    return y;
  }
  std::uint32_t permute(std::uint32_t x, std::initializer_list<unsigned> axes) {
    const auto source = shape(x);
    Shape dims;
    std::vector<Attribute> attrs;
    unsigned index = 0;
    for (const auto axis : axes) {
      dims.push_back(source.at(axis));
      attrs.push_back(Attribute::u64(static_cast<AttrKey>(
          static_cast<unsigned>(AttrKey::Permutation0) + index++), axis));
    }
    const auto y = tensor(std::move(dims), result.program.tensor(x)->dtype);
    op(Opcode::Permute, {x}, y, std::move(attrs));
    return y;
  }
  std::uint32_t binary(Opcode code, std::uint32_t x, std::uint32_t other) {
    const auto y = tensor(shape(x));
    op(code, {x, other}, y);
    return y;
  }
  std::uint32_t norm(std::uint32_t x, const std::string &prefix, double eps) {
    const auto dims = shape(x);
    const auto gamma = weight(prefix + ".weight", {dims.back()});
    const auto beta = weight(prefix + ".bias", {dims.back()});
    const auto y = tensor(dims);
    op(Opcode::LayerNorm, {x, gamma, beta}, y,
       {Attribute::f64(AttrKey::Epsilon, eps), Attribute::u64(AttrKey::BlockSize, 256)});
    return y;
  }
  std::uint32_t linear(std::uint32_t x, const std::string &prefix,
                       std::uint64_t width, std::uint32_t bias = 0) {
    auto dims = shape(x);
    const auto inner = dims.back();
    const auto rows = result.program.tensor(x)->element_count() / inner;
    const auto flat = reshape(x, {rows, inner});
    const auto w = weight(prefix + ".weight", {width, inner});
    if (!bias) bias = weight(prefix + ".bias", {width});
    const auto y = tensor({rows, width});
    op(Opcode::Linear, {flat, w, bias}, y,
       {Attribute::u64(AttrKey::LinearBiasMode,
                       static_cast<std::uint64_t>(ir::LinearBiasMode::Addmm))});
    dims.back() = width;
    return reshape(y, std::move(dims));
  }
  std::uint32_t conv(std::uint32_t x, const std::string &prefix,
                     std::uint64_t out_channels, std::uint64_t kernel,
                     std::uint64_t stride, std::uint64_t pad,
                     std::uint64_t dilation = 1, bool wn = true) {
    const auto dims = shape(x);
    const auto extent = mul(dilation, kernel - 1) + 1;
    if (dims[2] > std::numeric_limits<std::uint64_t>::max() - 2 * pad ||
        dims[2] + 2 * pad < extent)
      fail("audio encoder convolution geometry is invalid");
    const auto length = (dims[2] + 2 * pad - extent) / stride + 1;
    const auto w = weight(prefix + ".weight", {out_channels, dims[1], kernel}, wn);
    const auto bias = weight(prefix + ".bias", {out_channels});
    const auto y = tensor({dims[0], out_channels, length});
    op(Opcode::Conv1d, {x, w, bias}, y,
       {Attribute::u64(AttrKey::Stride, stride),
        Attribute::u64(AttrKey::Dilation, dilation), Attribute::u64(AttrKey::Groups, 1),
        Attribute::u64(AttrKey::PadLeft, pad), Attribute::u64(AttrKey::PadRight, pad),
        Attribute::u64(AttrKey::PadMode, 0), Attribute::boolean(AttrKey::Transposed, false),
        Attribute::u64(AttrKey::TrimLeft, 0), Attribute::u64(AttrKey::TrimRight, 0)});
    return y;
  }
  std::uint32_t snake(std::uint32_t x, const std::string &prefix) {
    const auto dims = shape(x);
    const auto a = weight(prefix + ".alpha", {1, dims[1], 1});
    const auto y = tensor(dims);
    op(Opcode::SnakeAlpha, {x, a}, y, {Attribute::f64(AttrKey::Epsilon, 1e-9)});
    return y;
  }
  std::uint32_t cast(std::uint32_t x, DType dtype) {
    const auto y = tensor(shape(x), dtype);
    op(Opcode::Cast, {x}, y);
    return y;
  }
  std::uint32_t mean(std::uint32_t x, unsigned axis) {
    auto dims = shape(x);
    dims.erase(dims.begin() + axis);
    const auto y = tensor(std::move(dims));
    op(Opcode::ReduceMean, {x}, y, {Attribute::u64(AttrKey::Axis, axis)});
    return y;
  }
  void output(std::uint32_t id) {
    for (auto &t : result.program.tensors)
      if (t.id == id) t.roles = TensorRole::Output;
  }
};
} // namespace

H3AudioEncoderConfig h3_audio_encoder_config(const json::Value &doc) {
  H3AudioEncoderConfig config;
  config.encoder_dim = integer(required(doc, "encoder_dim"));
  for (const auto &v : required(doc, "encoder_rates").array())
    config.encoder_rates.push_back(integer(v));
  config.latent_dim = integer(required(doc, "latent_dim"));
  config.latent_channels = integer(required(doc, "latent_channels"));
  config.num_attention_heads = integer(required(doc, "num_attention_heads"));
  config.eps = required(doc, "eps").number();
  return config;
}

H3AudioEncoderBuild build_h3_audio_encoder_program(
    std::uint64_t batch, std::uint64_t samples, const H3AudioEncoderConfig &c) {
  if (!batch || !samples || !c.encoder_dim || !c.latent_dim || !c.latent_channels ||
      !c.num_attention_heads || c.encoder_rates.empty() ||
      c.encoder_rates.size() > 16 || !std::isfinite(c.eps) || c.eps <= 0 ||
      c.latent_dim % c.num_attention_heads ||
      (c.latent_dim / c.num_attention_heads) % c.latent_channels)
    fail("audio encoder configuration is incompatible with the native port");
  std::uint64_t hop = 1;
  for (const auto rate : c.encoder_rates) {
    if (!rate || rate > 1000000) fail("invalid audio encoder stride");
    hop = mul(hop, rate);
  }
  const auto padded = mul(samples / hop + (samples % hop != 0), hop);
  Builder b;
  auto x = b.tensor({batch, 1, samples}, DType::F32, TensorRole::Input);
  b.result.waveform_input_id = x;
  if (padded != samples) {
    const auto zeros = b.zero({batch, 1, padded - samples});
    const auto joined = b.tensor({batch, 1, padded});
    b.op(Opcode::Concat, {x, zeros}, joined, {Attribute::u64(AttrKey::Axis, 2)});
    x = joined;
  }
  auto channels = c.encoder_dim;
  x = b.conv(x, "encoder.block.0", channels, 7, 1, 3);
  for (std::size_t stage = 0; stage < c.encoder_rates.size(); ++stage) {
    const auto prefix = "encoder.block." + std::to_string(stage + 1);
    constexpr std::array<std::uint64_t, 3> dilations{1, 3, 9};
    for (std::size_t unit = 0; unit < dilations.size(); ++unit) {
      const auto path = prefix + ".block." + std::to_string(unit) + ".block.";
      auto residual = b.snake(x, path + "0");
      residual = b.conv(residual, path + "1", channels, 7, 1, 3 * dilations[unit], dilations[unit]);
      residual = b.snake(residual, path + "2");
      residual = b.conv(residual, path + "3", channels, 1, 1, 0);
      x = b.binary(Opcode::Add, x, residual);
    }
    x = b.snake(x, prefix + ".block.3");
    channels = mul(channels, 2);
    const auto rate = c.encoder_rates[stage];
    x = b.conv(x, prefix + ".block.4", channels, 2 * rate, rate, (rate + 1) / 2);
  }
  const auto tail = c.encoder_rates.size() + 1;
  x = b.snake(x, "encoder.block." + std::to_string(tail));
  x = b.conv(x, "encoder.block." + std::to_string(tail + 1), c.latent_dim, 3, 1, 1);
  b.result.trunk_output_id = x;
  b.output(x);

  // audio_encoder_device.mojo:396-469, unchanged sequence of operations.
  const auto seq = b.shape(x)[2];
  if (seq != padded / hop) fail("audio encoder trunk did not preserve hop geometry");
  x = b.permute(x, {0, 2, 1});
  const auto shortcut = b.linear(b.norm(x, "pre_block.norm3", c.eps), "pre_block.proj", c.latent_channels);
  const auto n1 = b.norm(x, "pre_block.norm1", c.eps);
  const auto qb = b.weight("pre_block.attn.q_bias", {c.latent_dim});
  const auto kb = b.zero({c.latent_dim});
  const auto vb = b.weight("pre_block.attn.v_bias", {c.latent_dim});
  const auto bias = b.tensor({mul(3, c.latent_dim)});
  b.op(Opcode::Concat, {qb, kb, vb}, bias, {Attribute::u64(AttrKey::Axis, 0)});
  const auto qkv = b.linear(n1, "pre_block.attn.qkv", mul(3, c.latent_dim), bias);
  const auto head_dim = c.latent_dim / c.num_attention_heads;
  const Shape attention_shape{batch, seq, c.num_attention_heads, head_dim};
  std::array<std::uint32_t, 3> parts{};
  for (unsigned i = 0; i < 3; ++i) {
    const auto part = b.tensor({batch, seq, c.latent_dim});
    b.op(Opcode::Slice, {qkv}, part,
         {Attribute::u64(AttrKey::Axis, 2), Attribute::u64(AttrKey::Start, i * c.latent_dim)});
    parts[i] = b.cast(b.reshape(part, attention_shape), DType::BF16);
  }
  const auto attended16 = b.tensor(attention_shape, DType::BF16);
  b.op(Opcode::Attention, {parts[0], parts[1], parts[2]}, attended16,
       {Attribute::boolean(AttrKey::Causal, true),
        Attribute::u64(AttrKey::Implementation, 2),
        Attribute::u64(AttrKey::KvHeads, c.num_attention_heads),
        Attribute::f64(AttrKey::AttentionScale,
                       1.0F / std::sqrt(static_cast<float>(head_dim)))});
  auto attended = b.mean(b.cast(attended16, DType::F32), 2);
  attended = b.reshape(attended, {batch, seq, c.latent_channels, head_dim / c.latent_channels});
  attended = b.mean(attended, 3);
  attended = b.linear(attended, "pre_block.attn.proj", c.latent_channels);
  auto hidden = b.binary(Opcode::Add, shortcut, attended);
  const auto inner = b.norm(b.norm(hidden, "pre_block.norm2", c.eps), "pre_block.mlp.norm", c.eps);
  const auto gate = b.linear(inner, "pre_block.mlp.w0", mul(2, c.latent_channels));
  const auto value = b.linear(inner, "pre_block.mlp.w1", mul(2, c.latent_channels));
  const auto activated = b.tensor(b.shape(gate));
  b.op(Opcode::Gelu, {gate}, activated,
       {Attribute::u64(AttrKey::Approximation, static_cast<std::uint64_t>(ir::GeluApproximation::Tanh))});
  const auto mlp = b.linear(b.binary(Opcode::Multiply, activated, value), "pre_block.mlp.w2", c.latent_channels);
  hidden = b.permute(b.binary(Opcode::Add, hidden, mlp), {0, 2, 1});
  b.result.preblock_output_id = hidden;
  b.output(hidden);
  const auto mean = b.conv(hidden, "mean_proj", c.latent_channels, 1, 1, 0, 1, false);
  b.result.mean_output_id = mean;
  b.output(mean);
  ir::verify(b.result.program);
  return std::move(b.result);
}

runtime::TensorMap bind_h3_audio_encoder_weights(
    const H3AudioEncoderBuild &build, const weights::SafeTensorFile &checkpoint) {
  auto bindings = build.generated_constants;
  // The creator stores a frozen zero key bias. Never silently treat a nonzero
  // checkpoint buffer as the zero required by the proven native port.
  if (checkpoint.find("pre_block.attn.zero_k_bias")) {
    const auto k = weights::map_safetensor(checkpoint, "pre_block.attn.zero_k_bias");
    const auto *q = checkpoint.find("pre_block.attn.q_bias");
    if (!q || k.dtype != DType::F32 || k.dims != q->dims)
      fail("audio encoder zero key bias has invalid shape or dtype");
    const auto *values = reinterpret_cast<const float *>(k.data());
    for (std::uint64_t i = 0; i < k.element_count(); ++i)
      if (values[i] != 0) fail("audio encoder key bias must be zero");
  }
  for (const auto &binding : build.bindings) {
    const auto *description = build.program.tensor(binding.tensor_id);
    runtime::Tensor value;
    if (!binding.weight_normalized) {
      value = weights::map_safetensor(checkpoint, binding.name);
    } else {
      const auto prefix = binding.name.substr(0, binding.name.size() - 7);
      const auto v = weights::map_safetensor(checkpoint, prefix + ".weight_v");
      const auto g = weights::map_safetensor(checkpoint, prefix + ".weight_g");
      if (v.dtype != DType::F32 || v.dims != description->dims || g.dtype != DType::F32 ||
          g.dims != Shape{v.dims[0], 1, 1})
        fail("audio encoder weight norm shape/dtype mismatch: " + binding.name);
      value = runtime::Tensor{DType::F32, v.dims, {}};
      value.bytes.resize(v.byte_size());
      const auto channels = v.dims[0], width = v.element_count() / channels;
      const auto *vv = reinterpret_cast<const float *>(v.data());
      const auto *gg = reinterpret_cast<const float *>(g.data());
      auto *out = reinterpret_cast<float *>(value.bytes.data());
      // audio_decoder.mojo:54 fold_weight_norm, shared by its native encoder.
      for (std::uint64_t c = 0; c < channels; ++c) {
        double squares = 0;
        for (std::uint64_t i = 0; i < width; ++i) {
          const double v64 = vv[c * width + i];
          squares += v64 * v64;
        }
        const float scale = gg[c] / static_cast<float>(std::sqrt(squares));
        if (!std::isfinite(scale)) fail("invalid audio encoder weight norm: " + binding.name);
        for (std::uint64_t i = 0; i < width; ++i) out[c * width + i] = vv[c * width + i] * scale;
      }
    }
    if (value.dtype != description->dtype || value.dims != description->dims)
      fail("audio encoder binding shape/dtype mismatch: " + binding.name);
    bindings.emplace(binding.tensor_id, std::move(value));
  }
  return bindings;
}
} // namespace dif::frontend
