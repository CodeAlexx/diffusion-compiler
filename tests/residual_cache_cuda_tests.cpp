#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <tuple>

namespace {
using namespace dif;
int failures{};
void expect(bool value, const char *message) {
  if (!value) { ++failures; std::cerr << "FAIL " << message << '\n'; }
}
ir::Program fixture() {
  ir::Program p;
  p.tensors.push_back({1, ir::DType::BF16, ir::TensorRole::Input, {65,64}});
  p.tensors.push_back({2, ir::DType::BF16, ir::TensorRole::Constant, {65,64}});
  for (std::uint32_t op = 1; op <= 9; ++op) {
    const auto out = 9 + op;
    const auto role = (out == 10 || out == 12 || out == 16 || out == 18)
        ? ir::TensorRole::Output : ir::TensorRole::Internal;
    p.tensors.push_back({out, ir::DType::BF16, static_cast<std::uint32_t>(role), {65,64}});
    p.operations.push_back({op, ir::Opcode::Add, {op == 1 ? 1 : out - 1, 2}, {out}, {}});
  }
  ir::verify(p);
  return p;
}
struct Quantized { std::vector<std::int8_t> codes; std::vector<float> scales; };
Quantized quantize(std::span<const float> values) {
  Quantized q;
  q.codes.resize(values.size()); q.scales.resize(values.size() / 32);
  for (std::size_t g = 0; g < q.scales.size(); ++g) {
    float maximum = 0;
    for (std::size_t j = 0; j < 32; ++j) maximum = std::max(maximum, std::abs(values[g * 32 + j]));
    q.scales[g] = std::max(maximum / 127.0F, 1e-30F);
    for (std::size_t j = 0; j < 32; ++j)
      q.codes[g * 32 + j] = static_cast<std::int8_t>(std::clamp(
          std::nearbyint(values[g * 32 + j] / q.scales[g]), -127.0F, 127.0F));
  }
  return q;
}
std::vector<float> floats(const runtime::Tensor &tensor) {
  std::vector<float> result(tensor.element_count());
  for (std::size_t i = 0; i < result.size(); ++i) result[i] = runtime::load_float(tensor, i);
  return result;
}
void native_gate(bool overlap) {
  const auto p = fixture();
  runtime::TensorMap inputs;
  inputs.emplace(1, runtime::zeros(*p.tensor(1)));
  inputs.emplace(2, runtime::zeros(*p.tensor(2)));
  for (std::uint64_t i = 0; i < inputs.at(2).element_count(); ++i)
    runtime::store_float(inputs.at(2), i, static_cast<float>(static_cast<int>(i % 32) - 16) / 64.0F);
  runtime::RunOptions cpu_options;
  cpu_options.warmups = 0; cpu_options.iterations = 1; cpu_options.minimum_free_bytes = 0;
  auto cpu = runtime::make_cpu_executor();
  auto gpu = runtime::make_cuda_executor();
  auto options = cpu_options;
  options.overlap_streaming = overlap;
  options.requested_outputs = {18};
  options.residual_cache_region = runtime::ResidualCacheRegion{
      2,4,8,10,12,16,65,64,32,{0,32,64},{9,17}};
  auto callbacks = std::make_shared<runtime::ResidualCacheCallbacks>();
  options.residual_cache_callbacks = callbacks;
  std::size_t evaluation = 0, decisions = 0, refreshes = 0;
  const std::array actions{runtime::ResidualCacheAction::Refresh, runtime::ResidualCacheAction::Reuse,
                          runtime::ResidualCacheAction::Reuse, runtime::ResidualCacheAction::Exact};
  runtime::TensorMap expected_states;
  callbacks->needs_probes = [&] { return evaluation != 3; };
  callbacks->decide = [&](std::span<const float> before, std::span<const float> after,
                          std::span<const float> audio_before, std::span<const float> audio_after) {
    ++decisions;
    if (evaluation != 3) {
      for (const auto &[row_ids, actual_before, actual_after] :
          {std::tuple{std::vector<std::int32_t>{0,32,64}, before, after},
           std::tuple{std::vector<std::int32_t>{9,17}, audio_before, audio_after}}) {
        expect(actual_before.size() == row_ids.size() * 64 && actual_after.size() == row_ids.size() * 64,
               "native loop transfers only declared probes");
        for (std::size_t i = 0; i < actual_before.size(); ++i) {
          const auto index = row_ids[i / 64] * 64 + i % 64;
          expect(actual_before[i] == runtime::load_float(expected_states.at(10), index) &&
                     actual_after[i] == runtime::load_float(expected_states.at(12), index),
                 "native hook probes actual front boundaries");
        }
      }
    } else expect(before.empty() && after.empty() && audio_before.empty() && audio_after.empty(),
                  "native exact-tail callback has no probe copies");
    return actions[evaluation];
  };
  callbacks->refreshed = [&] { ++refreshes; };
  auto prepared = gpu->prepare(p, inputs, options);
  Quantized residual;
  for (; evaluation < actions.size(); ++evaluation) {
    for (std::uint64_t i = 0; i < inputs.at(1).element_count(); ++i)
      runtime::store_float(inputs.at(1), i,
          static_cast<float>(i % 17) / 32.0F + static_cast<float>(evaluation) / 512.0F);
    expected_states = cpu->run(p, inputs, cpu_options).outputs;
    auto expected = expected_states.at(18);
    if (actions[evaluation] == runtime::ResidualCacheAction::Refresh) {
      const auto front = quantize(floats(expected_states.at(12)));
      auto middle = floats(expected_states.at(16));
      for (std::size_t i = 0; i < middle.size(); ++i) {
        const volatile float dequant = static_cast<float>(front.codes[i]) * front.scales[i / 32];
        middle[i] = runtime::bf16_to_float(runtime::float_to_bf16(middle[i] - dequant));
      }
      residual = quantize(middle);
    } else if (actions[evaluation] == runtime::ResidualCacheAction::Reuse) {
      auto middle = floats(expected_states.at(12));
      for (std::size_t i = 0; i < middle.size(); ++i) {
        const volatile float delta = static_cast<float>(residual.codes[i]) * residual.scales[i / 32];
        float value = runtime::bf16_to_float(runtime::float_to_bf16(middle[i] + delta));
        for (unsigned back = 0; back < 2; ++back)
          value = runtime::bf16_to_float(runtime::float_to_bf16(value + runtime::load_float(inputs.at(2), i)));
        runtime::store_float(expected, i, value);
      }
    }
    const auto actual = prepared->run(inputs, options).outputs.at(18);
    expect(actual.bytes == expected.bytes, "native refresh/reuse/exact result matches typed CPU residual oracle byte-for-byte");
  }
  expect(decisions == 4 && refreshes == 1, "native region evaluates policy exactly once and refreshes once");
}
} // namespace
int main(int argc, char **argv) {
  try {
    fixture();
    if (argc != 2 || std::string(argv[1]) != "--run-cuda") {
      std::cout << "RESIDUAL_CACHE_NATIVE_GATE SKIP GPU requires explicit --run-cuda and external GPU lock\n";
      return 0;
    }
    native_gate(false); native_gate(true);
    if (failures) return 1;
    std::cout << "RESIDUAL_CACHE_NATIVE_GATE PASS graph=generic-nine-op-BF16"
              << " serial+overlap=byte-exact-to-CPU-oracle decisions=refresh,reuse,reuse,exact"
              << " H3_decoded_gate=not-run\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAIL " << error.what() << '\n'; return 1;
  }
}
