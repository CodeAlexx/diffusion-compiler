// Gate for AttentionBackward implementation 2 (cuDNN SDPA backward) against
// the CPU executor's F32-accumulating reference, with the CUDA generated
// math path (implementation 1) printed alongside. Bars: every element of
// dQ, dK, dV within one bf16 quantum at the tensor's magnitude (the Mojo
// shim's bar) and cosine >= 0.99999 (the Mojo 0.9999967 was measured against
// a matching bf16 math path; against an F32 reference the bf16 output
// rounding itself costs about 3e-6 under GQA accumulation), at S=1536 (head
// dim 128, 4 heads), under GQA (2 kv heads), and at a non-128-multiple S. dQ
// is not bit-repeatable on cuDNN's flash backward (a few dozen flips per
// million); dK and dV are, and the test reports both.
// Skips (exit 0) without CUDA or a cuDNN build.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

#include <algorithm>
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

dif::runtime::Tensor random_bf16(const std::vector<std::uint64_t> &dims,
                                 std::uint64_t seed, float amplitude) {
  std::uint64_t count = 1;
  for (const auto dim : dims)
    count *= dim;
  std::vector<std::uint8_t> bytes(count * 2U);
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (std::uint64_t i = 0; i < count; ++i) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto unit = static_cast<double>((state >> 33U) & 0x7fffffffU) /
                      static_cast<double>(0x7fffffffU);
    const auto value = bf16_bits(static_cast<float>((unit * 2.0 - 1.0) * amplitude));
    std::memcpy(bytes.data() + i * 2U, &value, 2U);
  }
  return dif::runtime::Tensor{dif::ir::DType::BF16, dims, std::move(bytes)};
}

struct Comparison {
  double cosine{};
  double max_quanta{};
  bool finite{true};
};

// One bf16 quantum at a given magnitude (8 significand bits).
double quantum_at(double magnitude) {
  if (magnitude == 0.0)
    return std::ldexp(1.0, -133);
  int exponent = 0;
  (void)std::frexp(magnitude, &exponent);
  return std::ldexp(1.0, exponent - 8);
}

Comparison compare(const dif::runtime::Tensor &a, const dif::runtime::Tensor &b) {
  Comparison result;
  double dot = 0.0, na = 0.0, nb = 0.0, max_abs = 0.0, magnitude = 0.0;
  const auto count = a.bytes.size() / 2U;
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t ba = 0, bb = 0;
    std::memcpy(&ba, a.bytes.data() + i * 2U, 2U);
    std::memcpy(&bb, b.bytes.data() + i * 2U, 2U);
    const float x = bf16_value(ba), y = bf16_value(bb);
    if (!std::isfinite(y))
      result.finite = false;
    dot += static_cast<double>(x) * y;
    na += static_cast<double>(x) * x;
    nb += static_cast<double>(y) * y;
    max_abs = std::max(max_abs, std::fabs(static_cast<double>(x) - y));
    magnitude = std::max(magnitude, static_cast<double>(std::fabs(x)));
  }
  // The Mojo bar: the largest disagreement is at most one bf16 quantum at
  // the tensor's own magnitude.
  result.max_quanta = max_abs / quantum_at(magnitude);
  result.cosine = (na > 0.0 && nb > 0.0) ? dot / std::sqrt(na * nb) : 0.0;
  return result;
}

// q,k,v,dO inputs; forward Attention (implementation 2) -> O, AttentionLse
// -> lse, AttentionBackward (implementation `backward`) -> dq, dk, dv.
dif::ir::Program program(std::uint64_t sequence, std::uint64_t heads,
                         std::uint64_t kv_heads, std::uint64_t backward) {
  using namespace dif::ir;
  Program p;
  const std::vector<std::uint64_t> q_shape{sequence, heads, 128U};
  const std::vector<std::uint64_t> kv_shape{sequence, kv_heads, 128U};
  p.tensors = {
      {1U, DType::BF16, TensorRole::Input, q_shape},
      {2U, DType::BF16, TensorRole::Input, kv_shape},
      {3U, DType::BF16, TensorRole::Input, kv_shape},
      {4U, DType::BF16, TensorRole::Input, q_shape},
      {5U, DType::BF16, TensorRole::Internal, q_shape},
      {6U, DType::F32, TensorRole::Internal, {sequence, heads}},
      {7U, DType::BF16, TensorRole::Output, q_shape},
      {8U, DType::BF16, TensorRole::Output, kv_shape},
      {9U, DType::BF16, TensorRole::Output, kv_shape},
  };
  std::vector<Attribute> gqa;
  if (kv_heads != heads)
    gqa.push_back(Attribute::u64(AttrKey::KvHeads, kv_heads));
  auto with = [&](std::vector<Attribute> extra) {
    auto attributes = gqa;
    for (auto &attribute : extra)
      attributes.push_back(attribute);
    return attributes;
  };
  p.operations = {
      {1U, Opcode::Attention, {1U, 2U, 3U}, {5U},
       with({Attribute::u64(AttrKey::Implementation, 2U)})},
      {2U, Opcode::AttentionLse, {1U, 2U}, {6U}, gqa},
      {3U, Opcode::AttentionBackward, {4U, 1U, 2U, 3U, 5U, 6U}, {7U, 8U, 9U},
       with({Attribute::u64(AttrKey::Implementation, backward)})},
  };
  dif::ir::verify(p);
  return p;
}

dif::runtime::RunOptions base_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  return options;
}

void gate(std::uint64_t sequence, std::uint64_t heads, std::uint64_t kv_heads) {
  const auto label = "S=" + std::to_string(sequence) + " H=" + std::to_string(heads) +
                     " KvH=" + std::to_string(kv_heads);
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, random_bf16({sequence, heads, 128U}, 11U + sequence, 1.0f));
  inputs.emplace(2U, random_bf16({sequence, kv_heads, 128U}, 23U + sequence, 1.0f));
  inputs.emplace(3U, random_bf16({sequence, kv_heads, 128U}, 37U + sequence, 1.0f));
  inputs.emplace(4U, random_bf16({sequence, heads, 128U}, 41U + sequence, 1.0f));
  auto cpu_program = program(sequence, heads, kv_heads, 1U);
  for (auto &operation : cpu_program.operations)
    operation.attributes.erase(
        std::remove_if(operation.attributes.begin(), operation.attributes.end(),
                       [](const dif::ir::Attribute &a) {
                         return a.key == dif::ir::AttrKey::Implementation;
                       }),
        operation.attributes.end());
  const auto reference = dif::runtime::make_cpu_executor()->run(
      cpu_program, inputs, base_options());
  const auto math = dif::runtime::make_cuda_executor()->run(
      program(sequence, heads, kv_heads, 1U), inputs, base_options());
  const auto cudnn = dif::runtime::make_cuda_executor()->run(
      program(sequence, heads, kv_heads, 2U), inputs, base_options());
  const auto repeat = dif::runtime::make_cuda_executor()->run(
      program(sequence, heads, kv_heads, 2U), inputs, base_options());
  const char *names[3] = {"dQ", "dK", "dV"};
  for (std::uint32_t index = 0U; index < 3U; ++index) {
    const auto id = 7U + index;
    const auto comparison = compare(reference.outputs.at(id), cudnn.outputs.at(id));
    const auto math_comparison = compare(reference.outputs.at(id), math.outputs.at(id));
    std::size_t flips = 0U;
    const auto &a = cudnn.outputs.at(id).bytes;
    const auto &b = repeat.outputs.at(id).bytes;
    for (std::size_t i = 0; i + 1U < a.size(); i += 2U)
      flips += (a[i] != b[i] || a[i + 1U] != b[i + 1U]) ? 1U : 0U;
    std::cout << "CUDNN_ATTENTION_BACKWARD " << label << " " << names[index]
              << " cudnn_vs_cpu cosine=" << comparison.cosine
              << " max_quanta=" << comparison.max_quanta
              << " | math_vs_cpu cosine=" << math_comparison.cosine
              << " max_quanta=" << math_comparison.max_quanta
              << " finite=" << (comparison.finite ? 1 : 0)
              << " repeat_flips=" << flips << "/" << a.size() / 2U << "\n";
    expect(comparison.finite, label + " " + names[index] + ": finite");
    expect(comparison.cosine >= 0.99999,
           label + " " + names[index] + ": cosine >= 0.99999 vs the CPU reference");
    expect(comparison.max_quanta <= 1.0,
           label + " " + names[index] + ": every element within one bf16 quantum");
    if (index != 0U)
      expect(flips == 0U, label + " " + names[index] + ": bit-repeatable");
  }
  std::cout << "CUDNN_ATTENTION_BACKWARD " << label
            << " math_ms=" << math.mean_milliseconds
            << " cudnn_ms=" << cudnn.mean_milliseconds << "\n";
  expect(cudnn.run_telemetry.cudnn_attention_dispatches >= 2U,
         label + ": receipt counts the cuDNN backward dispatch");
}

} // namespace

int main() {
  if (!dif::runtime::cuda_available()) {
    std::cout << "CUDA unavailable; cuDNN attention backward gates skipped\n";
    return 0;
  }
#if !DIF_HAS_CUDNN
  std::cout << "build has no cuDNN; gates skipped\n";
  return 0;
#else
  gate(1536U, 4U, 4U);
  gate(1536U, 4U, 2U);
  gate(1000U, 2U, 2U);
  if (failures != 0) {
    std::cerr << failures << " cuDNN attention backward test failure(s)\n";
    return 1;
  }
  std::cout << "cuDNN attention backward tests passed\n";
  return 0;
#endif
}
