// RotaryApply layouts: Interleaved (pair (2p, 2p+1)) and HalfSplit (pair
// (p, p+P), the rotate-half convention of Llama, Mistral and Qwen). Both are
// checked on the CPU executor against a direct computation, and on the CUDA
// executor against the CPU result when a device is present.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
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
void expect(bool c, const std::string &label) {
  if (!c) {
    ++failures;
    std::cerr << "FAIL: " << label << "\n";
  }
}

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<float> &values) {
  std::vector<std::uint8_t> bytes(values.size() * 4U);
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return dif::runtime::Tensor{dif::ir::DType::F32, std::move(dims), std::move(bytes)};
}

std::vector<float> values_of(const dif::runtime::Tensor &t) {
  std::vector<float> out(t.element_count());
  std::memcpy(out.data(), t.data(), out.size() * 4U);
  return out;
}

void run_layout(dif::ir::RotaryLayout layout, const char *label) {
  using namespace dif::ir;
  constexpr std::uint64_t B = 1, L = 3, H = 2, D = 8, P = 3; // partial: 6 of 8 rotated
  Program program;
  program.tensors = {
      {1U, DType::F32, TensorRole::Input, {B, L, H, D}},
      {2U, DType::F32, TensorRole::Input, {B, L, P}},
      {3U, DType::F32, TensorRole::Input, {B, L, P}},
      {4U, DType::F32, TensorRole::Output, {B, L, H, D}},
  };
  program.operations = {{1U, Opcode::RotaryApply, {1U, 2U, 3U}, {4U},
                         {Attribute::u64(AttrKey::RotaryLayout,
                                         static_cast<std::uint64_t>(layout))}}};
  verify(program);
  std::vector<float> x(B * L * H * D), c(B * L * P), s(B * L * P);
  for (std::size_t i = 0; i < x.size(); ++i)
    x[i] = 0.25f * static_cast<float>((i * 7) % 11) - 1.0f;
  for (std::uint64_t t = 0; t < L; ++t)
    for (std::uint64_t p = 0; p < P; ++p) {
      const double angle = 0.3 * static_cast<double>(t + 1) / static_cast<double>(p + 1);
      c[t * P + p] = static_cast<float>(std::cos(angle));
      s[t * P + p] = static_cast<float>(std::sin(angle));
    }
  // Direct computation.
  std::vector<float> want(x);
  const bool half = layout == RotaryLayout::HalfSplit;
  for (std::uint64_t t = 0; t < L; ++t)
    for (std::uint64_t h = 0; h < H; ++h)
      for (std::uint64_t p = 0; p < P; ++p) {
        const auto base = (t * H + h) * D;
        const auto a = half ? p : 2 * p, b = half ? p + P : 2 * p + 1;
        const float e = x[base + a], o = x[base + b];
        want[base + a] = e * c[t * P + p] - o * s[t * P + p];
        want[base + b] = e * s[t * P + p] + o * c[t * P + p];
      }
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, f32_tensor({B, L, H, D}, x));
  inputs.emplace(2U, f32_tensor({B, L, P}, c));
  inputs.emplace(3U, f32_tensor({B, L, P}, s));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto cpu = values_of(
      dif::runtime::make_cpu_executor()->run(program, inputs, options).outputs.at(4U));
  float max_cpu = 0.0f;
  for (std::size_t i = 0; i < want.size(); ++i)
    max_cpu = std::max(max_cpu, std::fabs(cpu[i] - want[i]));
  expect(max_cpu <= 1.0e-6f, std::string(label) + ": CPU matches the direct rotation");
  if (dif::runtime::cuda_available()) {
    const auto gpu = values_of(
        dif::runtime::make_cuda_executor()->run(program, inputs, options).outputs.at(4U));
    float max_gpu = 0.0f;
    for (std::size_t i = 0; i < want.size(); ++i)
      max_gpu = std::max(max_gpu, std::fabs(gpu[i] - want[i]));
    expect(max_gpu <= 1.0e-6f, std::string(label) + ": CUDA matches the direct rotation");
    std::cout << label << " max_abs cpu=" << max_cpu << " cuda=" << max_gpu << "\n";
  } else {
    std::cout << label << " max_abs cpu=" << max_cpu << " (CUDA unavailable)\n";
  }
}
} // namespace

int main() {
  run_layout(dif::ir::RotaryLayout::Interleaved, "interleaved");
  run_layout(dif::ir::RotaryLayout::HalfSplit, "half-split");
  // The unrotated tail (columns >= 2P) must pass through untouched in both.
  if (failures != 0) {
    std::cerr << failures << " rotary layout failure(s)\n";
    return 1;
  }
  std::cout << "rotary layout tests passed\n";
  return 0;
}
