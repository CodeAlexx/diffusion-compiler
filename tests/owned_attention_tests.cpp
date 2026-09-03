// Gate for the in-tree owned H3 dense INT8 attention (RunOptions::
// h3_owned_attention). Approximate route: the bar is the reuse audit's per-op
// gate, cosine >= 0.9999 against the exact cuDNN Attention at S=1536 and at
// the H3 fixture length S=16880, finite output, repeat-bit-exact, the run
// receipt naming the in-tree implementation, and fail-closed when two routes
// are requested at once. Skips (exit 0) without CUDA or off sm_86.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &label) {
  if (condition)
    return;
  ++failures;
  std::cerr << "FAIL: " << label << "\n";
}

std::uint16_t bf16_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t rounding = 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

float bf16_value(std::uint16_t bits) {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

// Adds a fixed per-channel offset (bias_amplitude * pattern) so a K tensor has
// the nonzero per-head means real attention keys carry.
dif::runtime::Tensor random_bf16(const std::vector<std::uint64_t> &dims,
                                 std::uint64_t seed, float amplitude,
                                 float bias_amplitude = 0.0f) {
  std::uint64_t count = 1;
  for (const auto dim : dims)
    count *= dim;
  std::vector<std::uint8_t> bytes(count * 2U);
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (std::uint64_t i = 0; i < count; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto unit = static_cast<double>((state >> 33U) & 0x7fffffffU) /
                      static_cast<double>(0x7fffffffU);
    const auto channel = static_cast<float>(i % 128U);
    const float bias = bias_amplitude * std::sin(channel * 0.37f);
    const auto value =
        bf16_bits(static_cast<float>((unit * 2.0 - 1.0) * amplitude) + bias);
    std::memcpy(bytes.data() + i * 2U, &value, 2U);
  }
  return dif::runtime::Tensor{dif::ir::DType::BF16, dims, std::move(bytes)};
}

struct Comparison {
  double cosine{};
  double max_abs{};
  bool finite{true};
};

Comparison compare(const dif::runtime::Tensor &a, const dif::runtime::Tensor &b) {
  Comparison result;
  double dot = 0.0, na = 0.0, nb = 0.0;
  const auto count = a.bytes.size() / 2U;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t ba = 0, bb = 0;
    std::memcpy(&ba, a.bytes.data() + i * 2U, 2U);
    std::memcpy(&bb, b.bytes.data() + i * 2U, 2U);
    const double x = bf16_value(ba), y = bf16_value(bb);
    if (!std::isfinite(y))
      result.finite = false;
    dot += x * y;
    na += x * x;
    nb += y * y;
    result.max_abs = std::max(result.max_abs, std::fabs(x - y));
  }
  result.cosine = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
  return result;
}

dif::ir::Program attention_program(std::uint64_t sequence, std::uint64_t heads) {
  using namespace dif::ir;
  Program program;
  const std::vector<std::uint64_t> shape{sequence, heads, 128U};
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, shape},
      {2U, DType::BF16, TensorRole::Input, shape},
      {3U, DType::BF16, TensorRole::Input, shape},
      {4U, DType::BF16, TensorRole::Output, shape},
  };
  program.operations = {
      {1U, Opcode::Attention, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::Implementation, 2U)}},
  };
  dif::ir::verify(program);
  return program;
}

dif::runtime::RunOptions base_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  return options;
}

void gate(std::uint64_t sequence, std::uint64_t heads, double cosine_bar) {
  const auto label = "S=" + std::to_string(sequence) + " H=" + std::to_string(heads);
  const auto program = attention_program(sequence, heads);
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, random_bf16({sequence, heads, 128U}, 11U + sequence, 1.0f));
  inputs.emplace(2U, random_bf16({sequence, heads, 128U}, 23U + sequence, 1.0f));
  inputs.emplace(3U, random_bf16({sequence, heads, 128U}, 37U + sequence, 1.0f));
  const auto exact =
      dif::runtime::make_cuda_executor()->run(program, inputs, base_options());
  auto owned_options = base_options();
  owned_options.h3_owned_attention = true;
  const auto owned =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  const auto repeat =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  expect(owned.h3_ck_attentions.size() == 1U,
         label + ": receipt lists one owned attention operation");
  if (!owned.h3_ck_attentions.empty()) {
    const auto &receipt = owned.h3_ck_attentions.front();
    expect(receipt.implementation == "owned_h3_dense_int8_v4_in_tree",
           label + ": receipt implementation is the in-tree kernel, got " +
               receipt.implementation);
    expect(receipt.classification == "approximate_owned_h3_dense_int8_gate",
           label + ": receipt classification stays approximate");
    expect(receipt.dso_path == "in-tree", label + ": receipt path is in-tree");
    expect(receipt.target_sm == 86U, label + ": receipt target_sm is 86");
  }
  const auto comparison = compare(exact.outputs.at(4U), owned.outputs.at(4U));
  std::cout << "OWNED_ATTENTION " << label << " cosine=" << comparison.cosine
            << " max_abs=" << comparison.max_abs
            << " finite=" << (comparison.finite ? 1 : 0)
            << " exact_ms=" << exact.mean_milliseconds
            << " owned_ms=" << owned.mean_milliseconds << "\n";
  expect(comparison.finite, label + ": owned output is finite");
  expect(comparison.cosine >= cosine_bar,
         label + ": cosine vs exact cuDNN >= " + std::to_string(cosine_bar));
  expect(owned.outputs.at(4U).bytes == repeat.outputs.at(4U).bytes,
         label + ": owned route is repeat-bit-exact");
}

// K mean-centering: on keys with a per-channel offset the centered route must
// hold or improve the cosine against exact cuDNN, stay finite and
// repeat-bit-exact, and carry its own receipt identity.
void gate_center_k(std::uint64_t sequence, std::uint64_t heads) {
  const auto label = "center-k S=" + std::to_string(sequence);
  const auto program = attention_program(sequence, heads);
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, random_bf16({sequence, heads, 128U}, 41U, 1.0f));
  inputs.emplace(2U, random_bf16({sequence, heads, 128U}, 43U, 1.0f, 2.5f));
  inputs.emplace(3U, random_bf16({sequence, heads, 128U}, 47U, 1.0f));
  const auto exact =
      dif::runtime::make_cuda_executor()->run(program, inputs, base_options());
  auto owned_options = base_options();
  owned_options.h3_owned_attention = true;
  const auto plain =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  owned_options.h3_owned_attention_center_k = true;
  const auto centered =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  const auto repeat =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  const auto plain_cmp = compare(exact.outputs.at(4U), plain.outputs.at(4U));
  const auto centered_cmp = compare(exact.outputs.at(4U), centered.outputs.at(4U));
  std::cout << "OWNED_ATTENTION_CENTER_K " << label << " plain_cosine="
            << plain_cmp.cosine << " centered_cosine=" << centered_cmp.cosine
            << " plain_ms=" << plain.mean_milliseconds
            << " centered_ms=" << centered.mean_milliseconds << "\n";
  expect(centered_cmp.finite, label + ": centered output is finite");
  expect(centered_cmp.cosine >= plain_cmp.cosine - 1.0e-6,
         label + ": centering holds or improves the cosine vs exact cuDNN");
  expect(centered_cmp.cosine >= 0.9999, label + ": centered cosine >= 0.9999");
  expect(centered.outputs.at(4U).bytes == repeat.outputs.at(4U).bytes,
         label + ": centered route is repeat-bit-exact");
  expect(!centered.h3_ck_attentions.empty() &&
             centered.h3_ck_attentions.front().implementation ==
                 "owned_h3_dense_int8_v4_in_tree_center_k",
         label + ": receipt names the centered identity");
  auto orphan = base_options();
  orphan.h3_owned_attention_center_k = true;
  bool refused = false;
  try {
    (void)dif::runtime::make_cuda_executor()->run(program, inputs, orphan);
  } catch (const dif::Error &error) {
    refused = std::string(error.what()).find("requires h3_owned_attention") !=
              std::string::npos;
  }
  expect(refused, label + ": center_k without the owned route is refused");
}

void test_two_routes_fail_closed() {
  const auto program = attention_program(256U, 2U);
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, random_bf16({256U, 2U, 128U}, 1U, 1.0f));
  inputs.emplace(2U, random_bf16({256U, 2U, 128U}, 2U, 1.0f));
  inputs.emplace(3U, random_bf16({256U, 2U, 128U}, 3U, 1.0f));
  auto options = base_options();
  options.h3_owned_attention = true;
  options.h3_ck_attention_dso = "/nonexistent/route.so";
  bool refused = false;
  try {
    (void)dif::runtime::make_cuda_executor()->run(program, inputs, options);
  } catch (const dif::Error &error) {
    refused = std::string(error.what()).find("choose one") != std::string::npos;
  }
  expect(refused, "two H3 attention routes at once are refused at prepare");
}

} // namespace

int main() {
  if (!dif::runtime::cuda_available()) {
    std::cout << "CUDA unavailable; owned attention gates skipped\n";
    return 0;
  }
#if !DIF_HAS_H3_OWNED_ATTENTION
  std::cout << "build has no in-tree owned attention kernel; gates skipped\n";
  return 0;
#else
  const auto probe =
      dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda, 0);
  if (probe.compute_major * 10U + probe.compute_minor != 86U) {
    std::cout << "owned attention kernel is sm_86 only; device is sm_"
              << probe.compute_major << probe.compute_minor << "; gates skipped\n";
    return 0;
  }
  test_two_routes_fail_closed();
  gate(1536U, 4U, 0.9999);
  gate(16880U, 4U, 0.9999);
  gate_center_k(1536U, 4U);
  gate_center_k(16880U, 4U);
  if (failures != 0) {
    std::cerr << failures << " owned attention test failure(s)\n";
    return 1;
  }
  std::cout << "owned attention tests passed\n";
  return 0;
#endif
}
