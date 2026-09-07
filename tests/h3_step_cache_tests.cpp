#include "dif/frontend/h3.hpp"
#include "dif/frontend/h3_step_cache.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
using namespace dif;
using namespace dif::frontend;
int failures{};
void expect(bool ok, const char *label) {
  if (!ok) { ++failures; std::cerr << "FAIL " << label << '\n'; }
}
template<class F> void rejects(F function, const char *label) {
  bool rejected = false;
  try { function(); } catch (const std::exception &) { rejected = true; }
  expect(rejected, label);
}
H3MiddleCacheDecision decide(H3MiddleCachePolicy &policy, std::uint64_t i,
                              float main, float audio = 100.0F) {
  const std::array before{0.0F, 0.0F};
  const std::array after{main, -main};
  const std::array audio_after{audio, -audio};
  return policy.evaluate(i, before, after, before, audio_after);
}
void warm(H3MiddleCachePolicy &policy) {
  for (std::uint64_t i = 0; i < 4; ++i) {
    auto d = decide(policy, i, 100.0F);
    expect(!d.reuse_middle && d.refresh_residual && d.discard_residual &&
               d.reason == H3MiddleCacheReason::Warmup, "four exact warmup evaluations");
    policy.mark_residual_ready();
  }
}
void decision_gate() {
  rejects([] { H3MiddleCachePolicy p(true, 0, "request"); }, "bounded evaluation count required");
  rejects([] { H3MiddleCachePolicy p(true, 12, ""); }, "execution identity required");
  H3MiddleCachePolicy p(true, 12, "frozen-model-plan-request-A");
  warm(p);
  auto d = decide(p, 4, 101.0F, 101.0F);
  expect(d.reuse_middle && !d.refresh_residual && !d.discard_residual &&
             d.residual_diff == 0.01F && d.audio_residual_diff == 0.01F,
         "first reuse compares front-band residual against full-refresh residual");
  d = decide(p, 5, 102.0F, 102.0F);
  expect(d.reuse_middle && d.residual_diff == 0.02F &&
             p.accumulated_diff() == 0.01F + 0.02F &&
             p.accumulated_audio_diff() == 0.01F + 0.02F,
         "reuse does not change baseline and budgets accumulate independently in F32");
  d = decide(p, 6, 102.0F, 102.0F);
  expect(!d.reuse_middle && d.reason == H3MiddleCacheReason::ConsecutiveLimit &&
             !p.residual_ready() && p.accumulated_diff() == 0.0F &&
             p.accumulated_audio_diff() == 0.0F,
         "third consecutive reuse prohibited and stale device cache discarded");
  p.mark_residual_ready();
  expect(decide(p, 7, 102.0F, 102.0F).reuse_middle &&
             decide(p, 8, 102.0F, 102.0F).reuse_middle,
         "fresh middle establishes a new baseline");
  d = decide(p, 9, 102.0F, 102.0F);
  expect(d.reason == H3MiddleCacheReason::ExactTail && !d.refresh_residual &&
             d.discard_residual && !p.enabled(), "last-three tail releases cache without requantization");
  expect(!p.evaluate(10, {}, {}).reuse_middle && !p.evaluate(11, {}, {}).reuse_middle &&
             p.full_evaluations() == 8 && p.cached_evaluations() == 4,
         "all final evaluations exact and complete counters retained");
  rejects([&] { p.mark_residual_ready(); }, "cannot resurrect disabled tail cache");
  rejects([&] { p.evaluate(12, {}, {}); }, "cannot run beyond sealed evaluation schedule");
  p.reset(true, 12, "different-model-plan-request-B");
  expect(!p.residual_ready() && p.full_evaluations() == 0 &&
             p.execution_identity() == "different-model-plan-request-B", "identity reset clears all numerical state");
  warm(p);
  d = decide(p, 4, 101.0F, 106.0F);
  expect(!d.reuse_middle && d.reason == H3MiddleCacheReason::AudioProbeVeto &&
             d.audio_residual_diff == H3MiddleCacheContract::audio_threshold,
         "audio veto is independent and threshold comparison strictly less-than");
  p.mark_residual_ready();
  d = decide(p, 5, 113.12F, 106.0F);
  expect(!d.reuse_middle && d.reason == H3MiddleCacheReason::MainProbeVeto,
         "main-band drift veto independent of stable audio");

  H3MiddleCachePolicy strict(true, 12, "strict-boundary");
  warm(strict);
  d = decide(strict, 4, 112.0F);
  expect(d.residual_diff == H3MiddleCacheContract::residual_threshold && !d.reuse_middle,
         "main threshold exactly equal refuses reuse");
  H3MiddleCachePolicy missing(true, 12, "missing-device-residual");
  for (std::uint64_t i = 0; i < 4; ++i) decide(missing, i, 100.0F);
  expect(decide(missing, 4, 100.0F).reason == H3MiddleCacheReason::MissingResidual,
         "policy never treats a requested GPU refresh as completed");
  const std::array finite{0.0F};
  const std::array nan{std::numeric_limits<float>::quiet_NaN()};
  rejects([&] { missing.evaluate(5, finite, nan); }, "nonfinite probes rejected before state mutation");
  expect(decide(missing, 5, 100.0F).refresh_residual, "rejected probe did not advance schedule");
  rejects([&] { decide(missing, 7, 100.0F); }, "out-of-order evaluation refused");

  H3MiddleCachePolicy short_run(true, 3, "short-exact-run");
  expect(decide(short_run, 0, 100.0F).reason == H3MiddleCacheReason::ExactTail,
         "schedule shorter than warmup stays fully exact");
  const auto main_rows = h3_middle_cache_probe_rows(100);
  expect(main_rows.size() == 16 && main_rows.front() == 0 && main_rows.back() == 99 &&
             main_rows[1] == 6, "sixteen evenly spaced whole-sequence probes");
  const std::array<std::int32_t, 3> audio_rows{80, 2, 99};
  expect(h3_middle_cache_audio_probe_rows(audio_rows, 100) ==
             std::vector<std::int32_t>({80, 2, 99}), "audio probes preserve physical row-index order");
  expect(h3_middle_cache_probe_rows(1) == std::vector<std::int32_t>({0}) &&
             h3_middle_cache_audio_probe_rows({}, 100).empty(), "single/absent modality probe boundaries");
  rejects([] { h3_middle_cache_probe_rows(0); }, "empty sequence refused");
  rejects([&] { h3_middle_cache_audio_probe_rows(audio_rows, 99); }, "out-of-bounds audio row refused");
}

H3DenoiserConfig config() {
  H3DenoiserConfig c;
  c.video_tokens = 1; c.audio_tokens = 1; c.text_tokens = 2; c.timestep_tables = 2;
  c.hidden = 32; c.heads = 2; c.head_dim = 6; c.ffn = 32; c.rotary = 6;
  c.layers = 50; c.refiner_layers = 1; c.video_input_dim = 3; c.audio_input_dim = 2;
  c.text_input_dim = 3; c.time_input_dim = 2; c.time_hidden_dim = 32; c.time_embed_dim = 8;
  c.block_size = 32; c.attention_implementation = 1;
  return c;
}
runtime::Tensor float_tensor(ir::DType dtype, const std::vector<std::uint64_t> &dims,
                              std::vector<float> values) {
  auto f = runtime::zeros({0, ir::DType::F32, 0, dims});
  expect(f.f32().size() == values.size(), "fixture float shape");
  std::copy(values.begin(), values.end(), f.f32().begin());
  return dtype == ir::DType::F32 ? std::move(f) : runtime::convert_float_tensor(f, dtype);
}
runtime::Tensor i32_tensor(std::vector<std::int32_t> values) {
  auto result = runtime::zeros({0, ir::DType::I32, 0, {values.size()}});
  std::memcpy(result.mutable_data(), values.data(), values.size() * 4);
  return result;
}
void partition_gate(bool execute) {
  const auto source = make_h3_denoiser(config());
  const auto partition = partition_h3_middle_cache(source);
  expect(partition.layers == 50 && partition.sequence == 4 && partition.hidden == 32,
         "complete fifty-block stack partitioned without substituting a prefix");
  const auto blocks = [](const auto &program) {
    return std::count_if(program.operations.begin(), program.operations.end(),
        [](const auto &o) { return o.opcode == ir::Opcode::H3AdaLNSelect; });
  };
  expect(blocks(partition.prelude) == 0 && blocks(partition.front) == 8 &&
             blocks(partition.middle) == 34 && blocks(partition.back) == 8,
         "front-eight and back-eight always exact; only middle34 can be reused");
  auto reassembled = source;
  reassembled.operations.clear();
  for (const auto *part : {&partition.prelude, &partition.front, &partition.middle, &partition.back}) {
    ir::verify(*part);
    reassembled.operations.insert(reassembled.operations.end(), part->operations.begin(), part->operations.end());
    for (const auto &tensor : part->tensors) {
      if (tensor.has_role(ir::TensorRole::Constant)) {
        const auto *original = source.tensor(tensor.id);
        expect(original && original->roles == tensor.roles && original->dtype == tensor.dtype &&
                   original->dims == tensor.dims, "weight IDs/storage policies survive slicing");
      }
    }
  }
  expect(ir::encode(reassembled) == ir::encode(source), "all operation IDs, ordering, attributes byte-exact on reassembly");
  expect(partition.prelude.tensor(partition.before_front_tensor)->has_role(ir::TensorRole::Output) &&
             partition.front.tensor(partition.after_front_tensor)->has_role(ir::TensorRole::Output) &&
             partition.middle.tensor(partition.after_middle_tensor)->has_role(ir::TensorRole::Output) &&
             partition.back.tensor(partition.after_middle_tensor)->has_role(ir::TensorRole::Input),
         "probe and reconstructed residual boundaries exposed as real graph values");
  rejects([&] { partition_h3_middle_cache(source, 49); }, "wrong layer count refused");
  auto small = config(); small.layers = 16;
  rejects([&] { partition_h3_middle_cache(make_h3_denoiser(small), 16); }, "no skipped middle band refused");
  small = config(); small.hidden = 12;
  rejects([&] { partition_h3_middle_cache(make_h3_denoiser(small)); }, "non-group32 hidden refused");
  if (!execute) return;

  runtime::TensorMap bindings;
  bindings.emplace(1, float_tensor(ir::DType::F32, {1, 3}, {0.1F,-0.2F,0.3F}));
  bindings.emplace(2, float_tensor(ir::DType::F32, {1, 2}, {0.2F,-0.1F}));
  bindings.emplace(3, float_tensor(ir::DType::BF16, {2, 3}, {0.1F,0.2F,-0.1F,-0.2F,0.05F,0.3F}));
  bindings.emplace(4, float_tensor(ir::DType::F32, {2}, {0.25F,0.75F}));
  bindings.emplace(5, i32_tensor({0,1,-1,-1}));
  bindings.emplace(6, i32_tensor({-1,-1,0,-1}));
  bindings.emplace(7, i32_tensor({-1,-1,-1,0}));
  bindings.emplace(8, i32_tensor({1,1,0,5}));
  bindings.emplace(9, i32_tensor({0,0,0,1}));
  bindings.emplace(10, i32_tensor({2})); bindings.emplace(11, i32_tensor({3}));
  bindings.emplace(12, float_tensor(ir::DType::F32, {4,3}, {0,0,0,1,0,0,2,1,1,3,0,1}));
  for (const auto &tensor : source.tensors) {
    if (!tensor.has_role(ir::TensorRole::Constant)) continue;
    std::vector<float> values(tensor.element_count());
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = tensor.dims.size() == 1 ? 0.1F :
          0.002F * static_cast<float>(static_cast<int>((i + tensor.id) % 11) - 5);
    bindings.emplace(tensor.id, float_tensor(tensor.dtype, tensor.dims, std::move(values)));
  }
  runtime::RunOptions options;
  options.warmups = 0; options.iterations = 1; options.minimum_free_bytes = 0;
  const auto executor = runtime::make_cpu_executor();
  const auto unsplit = executor->run(source, bindings, options);
  for (const auto *part : {&partition.prelude, &partition.front, &partition.middle, &partition.back}) {
    const auto run = executor->run(*part, bindings, options);
    for (const auto &[id, value] : run.outputs) bindings.insert_or_assign(id, value);
  }
  for (const auto &[id, expected] : unsplit.outputs) {
    const auto &actual = bindings.at(id);
    expect(actual.bytes == expected.bytes, "split full-refresh produces byte-exact unsplit video/audio prediction");
    for (const auto value : actual.f32()) expect(std::isfinite(value), "partition CPU output finite");
  }
}
} // namespace
int main(int argc, char **argv) {
  try {
    const bool execute = !(argc == 2 && std::string(argv[1]) == "--policy-only");
    decision_gate(); partition_gate(execute);
    if (failures) return 1;
    std::cout << "H3_MIDDLE_CACHE_CPU PASS source=Mojo-policy front=8 middle=34 back=8"
              << " warmup=4 exact_tail=3 audio_veto=independent full_refresh="
              << (execute ? "byte-exact-unsplit" : "not-run")
              << " GPU_residual_gate=not-run decoded_gate=not-run\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL exception " << error.what() << '\n'; return 1;
  }
}
