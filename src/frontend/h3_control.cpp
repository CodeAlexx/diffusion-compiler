#include "dif/frontend/h3_control.hpp"

#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <set>

namespace dif::frontend {

float h3_control_strength(const H3ControlSchedule &s, float progress) {
  if (!std::isfinite(s.strength) || !std::isfinite(s.start) ||
      !std::isfinite(s.end) || s.start < 0 || s.end > 1 || s.start > s.end ||
      !std::isfinite(progress) || progress < 0 || progress > 1)
    fail("H3 ControlNet requires finite strength and 0 <= start <= end <= 1");
  return s.start <= progress && progress <= s.end ? s.strength : 0.0F;
}

H3DenoiserConfig h3_control_denoiser_config(const ir::Program &base) {
  using namespace ir;
  verify(base);
  H3DenoiserConfig d;
  auto input = [&](std::uint32_t id, DType dtype, std::size_t rank) -> const TensorDesc & {
    const auto *t=base.tensor(id);
    if (!t || !t->has_role(TensorRole::Input) || t->dtype!=dtype || t->dims.size()!=rank)
      fail("H3 ControlNet requires the canonical denoiser input contract");
    return *t;
  };
  const auto &video=input(1,DType::F32,2), &audio=input(2,DType::F32,2);
  const auto &text=input(3,DType::BF16,2), &times=input(4,DType::F32,1);
  d.video_tokens=video.dims[0];d.video_input_dim=video.dims[1];
  d.audio_tokens=audio.dims[0];d.audio_input_dim=audio.dims[1];
  d.text_tokens=text.dims[0];d.text_input_dim=text.dims[1];d.timestep_tables=times.dims[0];
  auto find_op=[&](Opcode code)->const Operation& {
    const auto it=std::find_if(base.operations.begin(),base.operations.end(),
                             [&](const auto &op){return op.opcode==code;});
    if (it==base.operations.end()) fail("H3 ControlNet missing canonical operation");
    return *it;
  };
  auto producer=[&](std::uint32_t id)->const Operation& {
    const auto it=std::find_if(base.operations.begin(),base.operations.end(),
        [&](const auto &op){return std::find(op.outputs.begin(),op.outputs.end(),id)!=op.outputs.end();});
    if (it==base.operations.end()) fail("H3 ControlNet missing canonical producer");
    return *it;
  };
  const auto &mod=producer(find_op(Opcode::H3AdaLNSelect).inputs[0]);
  if (mod.opcode!=Opcode::Linear || mod.inputs.size()!=3)
    fail("H3 ControlNet unsupported timestep projection");
  const auto *mw=base.tensor(mod.inputs[1]);
  if (!mw || mw->dims.size()!=2 || mw->dims[0]%18)
    fail("H3 ControlNet unsupported modulation dimensions");
  d.hidden=mw->dims[0]/18;d.time_embed_dim=mw->dims[1];
  d.block_size=mod.u64(AttrKey::BlockSize,256);
  d.streamed_constants=mw->has_role(TensorRole::Streamed);
  const auto &qkv=find_op(Opcode::H3DeinterleaveQkvWeight);
  d.heads=qkv.u64(AttrKey::Heads,0);d.head_dim=qkv.u64(AttrKey::HeadDim,0);
  d.rotary=base.tensor(find_op(Opcode::RotaryPosition).outputs[0])->dims.back();
  d.ffn=base.tensor(find_op(Opcode::SwiGlu).outputs[0])->dims.back();
  d.layers=static_cast<std::uint64_t>(std::count_if(base.operations.begin(),base.operations.end(),
      [](const auto &op){return op.opcode==Opcode::H3AdaLNSelect;}));
  const auto attention_count=static_cast<std::uint64_t>(std::count_if(base.operations.begin(),base.operations.end(),
      [](const auto &op){return op.opcode==Opcode::Attention;}));
  if (attention_count<=d.layers) fail("H3 ControlNet missing canonical text refiner");
  d.refiner_layers=attention_count-d.layers;
  d.attention_implementation=find_op(Opcode::Attention).u64(AttrKey::Implementation,1);
  const auto feature=find_op(Opcode::SinusoidalTimestep).outputs[0];
  d.time_input_dim=base.tensor(feature)->dims.back();
  const auto time_linear=std::find_if(base.operations.begin(),base.operations.end(),
      [&](const auto &op){return op.opcode==Opcode::Linear && op.inputs[0]==feature;});
  if (time_linear==base.operations.end()) fail("H3 ControlNet missing canonical timestep MLP");
  d.time_hidden_dim=base.tensor(time_linear->outputs[0])->dims.back();
  return d;
}

H3ControlGraph add_h3_control(const ir::Program &base,
                             const H3DenoiserConfig &d,
                             const H3ControlConfig &c) {
  using namespace ir;
  if (c.controls == 0 || c.controls > 4 || c.patch_features == 0 ||
      c.injection_layers.empty() || c.injection_layers.front() != 0 ||
      c.injection_layers.back() >= d.layers ||
      !std::is_sorted(c.injection_layers.begin(), c.injection_layers.end()) ||
      std::adjacent_find(c.injection_layers.begin(), c.injection_layers.end()) !=
          c.injection_layers.end())
    fail("invalid H3 ControlNet stream count or injection layers");
  H3ControlGraph result;
  result.program = base;
  result.program.operations.clear();
  auto &p = result.program;
  std::uint32_t tid = 1, oid = 1;
  for (const auto &t : base.tensors) tid = std::max(tid, t.id + 1);
  for (const auto &o : base.operations) oid = std::max(oid, o.id + 1);
  const auto s = d.video_tokens + d.audio_tokens + d.text_tokens;
  const auto h = d.hidden, inner = d.heads * d.head_dim;
  const auto roles = static_cast<std::uint32_t>(TensorRole::Constant |
      (c.streamed_constants ? TensorRole::Streamed : TensorRole::Internal));
  auto tensor = [&](DType dtype, std::vector<std::uint64_t> dims,
                    std::uint32_t role = TensorRole::Internal) {
    const auto id = tid++;
    p.tensors.push_back({id, dtype, role, std::move(dims)});
    return id;
  };
  auto hidden = [&] { return tensor(DType::BF16, {s, h}); };
  auto op = [&](Opcode code, std::vector<std::uint32_t> in,
                std::vector<std::uint32_t> out,
                std::vector<Attribute> attrs = {}) {
    p.operations.push_back({oid++, code, std::move(in), std::move(out), std::move(attrs)});
  };
  std::map<std::string, std::uint32_t> shared_weights;
  auto weight = [&](const std::string &name, DType type,
                    std::vector<std::uint64_t> dims) {
    if (auto it = shared_weights.find(name); it != shared_weights.end()) return it->second;
    const auto id = tensor(type, std::move(dims), roles);
    result.weights.push_back({id, name});
    shared_weights.emplace(name, id);
    result.mapped_weight_bytes += p.tensor(id)->byte_count();
    return id;
  };
  const std::vector<Attribute> linear_attrs{
      Attribute::u64(AttrKey::BlockSize, d.block_size),
      Attribute::u64(AttrKey::Implementation, 1)};
  const std::vector<Attribute> norm_attrs{
      Attribute::f64(AttrKey::Epsilon, 1e-5),
      Attribute::u64(AttrKey::BlockSize, d.block_size)};
  const std::vector<Attribute> rope_attrs{
      Attribute::f64(AttrKey::Epsilon, 1e-5),
      Attribute::u64(AttrKey::Heads, d.heads),
      Attribute::u64(AttrKey::HeadDim, d.head_dim),
      Attribute::u64(AttrKey::RotaryDim, d.rotary),
      Attribute::u64(AttrKey::BlockSize, d.block_size)};
  const std::vector<Attribute> attention_attrs{
      Attribute::f64(AttrKey::AttentionScale, 1.0 / std::sqrt(double(d.head_dim))),
      Attribute::boolean(AttrKey::Causal, false),
      Attribute::u64(AttrKey::Implementation, d.attention_implementation)};

  // Canonical graph inputs 6/7/8 are video/audio destination maps and the
  // per-token timestep/modality index. Locate all remaining splice points
  // through producer relationships, never through a fixed block tensor ID.
  std::uint32_t initial = 0, temb = 0, cos = 0, sin = 0;
  std::vector<std::uint32_t> base_outputs;
  bool in_block = false;
  unsigned gates = 0;
  for (const auto &o : base.operations) {
    if (o.opcode == Opcode::IndexedUpdateRows && o.inputs[2] == 7) initial = o.outputs[0];
    if (o.opcode == Opcode::RotaryPosition) { cos = o.outputs[0]; sin = o.outputs[1]; }
    if (o.opcode == Opcode::H3AdaLNSelect) { in_block = true; gates = 0; }
    if (in_block && o.opcode == Opcode::ResidualGate && ++gates == 2) {
      base_outputs.push_back(o.outputs[0]); in_block = false;
    }
    if (!temb && o.opcode == Opcode::H3AdaLNSelect) {
      const auto producer = std::find_if(base.operations.begin(), base.operations.end(),
          [&](const auto &x) { return x.outputs == std::vector<std::uint32_t>{o.inputs[0]}; });
      if (producer == base.operations.end() || producer->opcode != Opcode::Linear)
        fail("H3 ControlNet cannot locate the shared timestep embedding");
      temb = producer->inputs[0];
    }
  }
  if (!initial || !temb || !cos || !sin || base_outputs.size() != d.layers)
    fail("H3 ControlNet could not resolve canonical trunk boundaries");

  std::vector<std::uint32_t> side(c.controls);
  auto initialize = [&] {
    const auto pw = weight("control_proj_in.weight", DType::F32, {h, c.patch_features});
    const auto pb = weight("control_proj_in.bias", DType::F32, {h});
    const auto bw = weight("control_blocks.0.before_proj.weight", DType::BF16, {h, h});
    const auto bb = weight("control_blocks.0.before_proj.bias", DType::BF16, {h});
    for (std::uint32_t i = 0; i < c.controls; ++i) {
      const auto rows = tensor(DType::F32, {d.video_tokens, c.patch_features}, TensorRole::Input);
      result.rows_inputs.push_back(rows);
      result.strength_inputs.push_back(tensor(DType::F32, {1}, TensorRole::Input));
      const auto projected = tensor(DType::F32, {d.video_tokens, h});
      const auto projected_bf16 = tensor(DType::BF16, {d.video_tokens, h});
      const auto replaced = hidden(), before = hidden(); side[i] = hidden();
      op(Opcode::Linear, {rows, pw, pb}, {projected}, linear_attrs);
      op(Opcode::Cast, {projected}, {projected_bf16});
      op(Opcode::IndexedUpdateRows, {initial, projected_bf16, 6}, {replaced});
      op(Opcode::Linear, {replaced, bw, bb}, {before}, linear_attrs);
      op(Opcode::Add, {before, initial}, {side[i]});
    }
  };

  auto advance = [&](std::uint32_t input, std::size_t index) {
    const auto prefix = "control_blocks." + std::to_string(index) + ".";
    auto w = [&](const std::string &name, std::vector<std::uint64_t> dims) {
      return weight(prefix + name, DType::BF16, std::move(dims));
    };
    const auto aw = w("adaln_proj.linear.weight", {18*h, d.time_embed_dim});
    const auto ab = w("adaln_proj.linear.bias", {18*h});
    const auto mod = tensor(DType::BF16, {d.timestep_tables, 18*h});
    const auto table = tensor(DType::BF16, {d.timestep_tables*3, 6*h});
    std::vector<std::uint32_t> m;
    for (unsigned j = 0; j != 6; ++j) m.push_back(hidden());
    op(Opcode::Linear, {temb, aw, ab}, {mod}, linear_attrs);
    op(Opcode::Reshape, {mod}, {table});
    op(Opcode::SelectRowChunks, {table, 8}, m);
    const auto norm1 = w("norm1.weight", {h}), norm2 = w("norm2.weight", {h});
    const auto a = hidden();
    op(Opcode::RmsNormModulate, {input, norm1, m[1], m[0]}, {a}, norm_attrs);
    const auto qw = w("attn.to_q.weight", {inner, h});
    const auto kw = w("attn.to_k.weight", {inner, h});
    const auto vw = w("attn.to_v.weight", {inner, h});
    const auto qkv_weight = tensor(DType::BF16, {3 * inner, h});
    const auto packed_qkv = tensor(DType::BF16, {s, 3 * inner});
    const auto q_flat = tensor(DType::BF16, {s, inner});
    const auto k_flat = tensor(DType::BF16, {s, inner});
    const auto v_flat = tensor(DType::BF16, {s, inner});
    const auto q_linear = tensor(DType::BF16, {s, d.heads, d.head_dim});
    const auto k_linear = tensor(DType::BF16, {s, d.heads, d.head_dim});
    const auto v_linear = tensor(DType::BF16, {s, d.heads, d.head_dim});
    op(Opcode::Concat, {qw, kw, vw}, {qkv_weight},
       {Attribute::u64(AttrKey::Axis, 0)});
    op(Opcode::Linear, {a, qkv_weight}, {packed_qkv}, linear_attrs);
    op(Opcode::Slice, {packed_qkv}, {q_flat},
       {Attribute::u64(AttrKey::Axis, 1), Attribute::u64(AttrKey::Start, 0)});
    op(Opcode::Slice, {packed_qkv}, {k_flat},
       {Attribute::u64(AttrKey::Axis, 1), Attribute::u64(AttrKey::Start, inner)});
    op(Opcode::Slice, {packed_qkv}, {v_flat},
       {Attribute::u64(AttrKey::Axis, 1), Attribute::u64(AttrKey::Start, 2 * inner)});
    op(Opcode::Reshape, {q_flat}, {q_linear});
    op(Opcode::Reshape, {k_flat}, {k_linear});
    op(Opcode::Reshape, {v_flat}, {v_linear});
    const auto q = tensor(DType::BF16, {s,d.heads,d.head_dim});
    const auto k = tensor(DType::BF16, {s,d.heads,d.head_dim});
    const auto att = tensor(DType::BF16, {s,d.heads,d.head_dim});
    op(Opcode::QkNormPartialRope, {q_linear,w("attn.norm_q.weight",{d.head_dim}),cos,sin}, {q}, rope_attrs);
    op(Opcode::QkNormPartialRope, {k_linear,w("attn.norm_k.weight",{d.head_dim}),cos,sin}, {k}, rope_attrs);
    op(Opcode::Attention, {q,k,v_linear}, {att}, attention_attrs);
    const auto projection = hidden(), after = hidden(), mlp_in = hidden();
    op(Opcode::Linear, {att,w("attn.to_out.0.weight",{h,inner})}, {projection}, linear_attrs);
    op(Opcode::ResidualGate, {input,projection,m[2]}, {after}, linear_attrs);
    op(Opcode::RmsNormModulate, {after,norm2,m[4],m[3]}, {mlp_in}, norm_attrs);
    // `ff.net.0.proj` needs NO reorder, and reordering it swaps the SwiGLU
    // gate and value halves.  The released Union checkpoint is in the
    // diffusers order, [value; gate] -- the OPPOSITE of the base checkpoint's
    // [gate; value] `mlp.fc1`, which h3.cpp feeds to SwiGlu with
    // GateFirst=true.  Measured on the real checkpoints,
    // control_blocks.0.ff.net.0.proj matches blocks.0.mlp.fc1 only ACROSS
    // halves (cos 0.9995 crossed, -0.006 uncrossed).  GateFirst=false already
    // reads first-half-as-value, so the raw tensor is the correct operand; the
    // former slice/concat pair produced [gate; value] and then read the gate
    // half as the value half.
    const auto fc1_weight = w("ff.net.0.proj.weight",{2*d.ffn,h});
    const auto fc = tensor(DType::BF16,{s,2*d.ffn}), sw = tensor(DType::BF16,{s,d.ffn});
    const auto mlp = hidden(), output = hidden(), skip = hidden();
    op(Opcode::Linear, {mlp_in,fc1_weight}, {fc}, linear_attrs);
    op(Opcode::SwiGlu, {fc}, {sw}, {Attribute::boolean(AttrKey::GateFirst,false)});
    op(Opcode::Linear, {sw,w("ff.net.2.weight",{h,d.ffn})}, {mlp}, linear_attrs);
    op(Opcode::ResidualGate, {after,mlp,m[5]}, {output}, linear_attrs);
    op(Opcode::Linear, {output,w("after_proj.weight",{h,h}),w("after_proj.bias",{h})}, {skip}, linear_attrs);
    result.side_outputs.push_back(output);
    return std::pair{output, skip};
  };

  std::map<std::uint32_t,std::uint32_t> rewrites;
  bool initialized = false;
  for (auto operation : base.operations) {
    for (auto &id : operation.inputs)
      if (auto found = rewrites.find(id); found != rewrites.end()) id = found->second;
    p.operations.push_back(operation);
    // The initial packed hidden is complete here; side blocks consume shared
    // temb and rope later, when their base injection layer has finished.
    if (!initialized && operation.outputs == std::vector<std::uint32_t>{initial}) {
      initialize(); initialized = true;
    }
    for (std::size_t j = 0; j < c.injection_layers.size(); ++j) {
      const auto original = base_outputs[c.injection_layers[j]];
      if (operation.outputs != std::vector<std::uint32_t>{original}) continue;
      auto residual = original;
      for (std::uint32_t i = 0; i < c.controls; ++i) {
        auto [next, skip] = advance(side[i], j); side[i] = next;
        if (!c.apply_audio) {
          const auto zero = tensor(DType::BF16, {d.audio_tokens,h}), masked = hidden();
          op(Opcode::Fill, {}, {zero}, {Attribute::f64(AttrKey::Value,0)});
          op(Opcode::IndexedUpdateRows, {skip,zero,7}, {masked}); skip = masked;
        }
        // Mojo minimax_h3_control_inject_active stores mul_scalar's result
        // in BF16 before add. The strength itself remains F32. Folding this
        // into torch.add(alpha=...) loses the intermediate BF16 rounding.
        const auto sf = tensor(DType::F32,{s,h}), alpha = tensor(DType::F32,{s,h});
        const auto product = tensor(DType::F32,{s,h});
        const auto scaled = hidden(), output = hidden();
        op(Opcode::Cast, {skip}, {sf});
        op(Opcode::BroadcastTo, {result.strength_inputs[i]}, {alpha});
        op(Opcode::Multiply, {sf,alpha}, {product});
        op(Opcode::Cast, {product}, {scaled});
        op(Opcode::Add, {residual,scaled}, {output}); residual = output;
        result.injected_outputs.push_back(output);
      }
      rewrites.emplace(original,residual);
    }
  }
  verify(p);
  return result;
}

void validate_h3_control_checkpoint(const weights::SafeTensorFile &file,
                                    const H3ControlGraph &graph) {
  std::set<std::string> expected;
  for (const auto &w : graph.weights) {
    expected.insert(w.name);
    const auto *entry = file.find(w.name);
    const auto *desc = graph.program.tensor(w.tensor_id);
    if (!entry || entry->dtype != desc->dtype || entry->dims != desc->dims)
      fail("H3 ControlNet missing or incompatible dense weight: " + w.name);
  }
  for (const auto &[name, entry] : file.tensors) {
    (void)entry;
    if (!expected.contains(name))
      fail("H3 ControlNet has unconsumed/unsupported checkpoint tensor: " + name);
  }
  if (!file.metadata_tensors.empty())
    fail("H3 ControlNet has unsupported packed/quantized metadata tensors");
}

runtime::TensorMap map_h3_control_weights(const weights::SafeTensorFile &file,
                                          const H3ControlGraph &graph) {
  validate_h3_control_checkpoint(file,graph);
  runtime::TensorMap result;
  for (const auto &w : graph.weights)
    result.emplace(w.tensor_id, weights::map_safetensor(file,w.name));
  return result;
}

runtime::Tensor h3_control_rows(const runtime::Tensor &latent,
                               std::uint64_t ph, std::uint64_t pw,
                               std::uint64_t features) {
  latent.validate();
  if (latent.dtype != ir::DType::F32 || latent.dims.size() != 5 ||
      latent.dims[0] != 1 || !ph || !pw || !features ||
      ph > std::numeric_limits<std::uint64_t>::max()/pw ||
      latent.dims[1] > features/(ph*pw))
    fail("H3 ControlNet guide must be F32 [1,C,T,H,W] fitting the patch width");
  const auto channels=latent.dims[1], frames=latent.dims[2], height=latent.dims[3], width=latent.dims[4];
  if (height%ph || width%pw) fail("H3 ControlNet guide canvas must be patch aligned");
  const auto rows=frames*(height/ph)*(width/pw);
  auto result=runtime::zeros({0,ir::DType::F32,ir::TensorRole::Input,{rows,features}});
  const auto source=latent.f32(); auto out=result.f32();
  for (std::uint64_t t=0; t<frames; ++t)
    for (std::uint64_t y=0; y<height/ph; ++y)
      for (std::uint64_t x=0; x<width/pw; ++x)
        for (std::uint64_t ch=0; ch<channels; ++ch)
          for (std::uint64_t dy=0; dy<ph; ++dy)
            for (std::uint64_t dx=0; dx<pw; ++dx) {
              const auto value=source[((ch*frames+t)*height+y*ph+dy)*width+x*pw+dx];
              if (!std::isfinite(value)) fail("H3 ControlNet guide contains nonfinite values");
              out[((t*(height/ph)+y)*(width/pw)+x)*features+(ch*ph+dy)*pw+dx]=value;
            }
  return result;
}

} // namespace dif::frontend
