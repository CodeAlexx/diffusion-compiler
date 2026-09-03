// Gates for the two H3 INT8 attention route refinements:
//   RunOptions::h3_int8_attention_exact_query_ranges  (exact query-row overlay)
//   RunOptions::h3_int8_attention_hybrid_first_layer/_layers (hybrid sub-range)
// Uses the in-tree owned dense INT8 kernel as the approximate route, so the
// gate runs without any DSO. Skips (exit 0) without CUDA or off sm_86.

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
    const auto value =
        bf16_bits(static_cast<float>((unit * 2.0 - 1.0) * amplitude));
    std::memcpy(bytes.data() + i * 2U, &value, 2U);
  }
  return dif::runtime::Tensor{dif::ir::DType::BF16, dims, std::move(bytes)};
}

struct Comparison {
  double cosine{};
  double max_abs{};
  bool finite{true};
  bool identical{true};
};

// Compare rows [begin, begin+count) of two [S,H,128] BF16 tensors.
Comparison compare_rows(const dif::runtime::Tensor &a,
                        const dif::runtime::Tensor &b, std::uint64_t begin,
                        std::uint64_t count) {
  Comparison result;
  const auto row_elements = a.dims.at(1) * a.dims.at(2);
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::uint64_t e = begin * row_elements; e < (begin + count) * row_elements;
       ++e) {
    std::uint16_t ba = 0, bb = 0;
    std::memcpy(&ba, a.bytes.data() + e * 2U, 2U);
    std::memcpy(&bb, b.bytes.data() + e * 2U, 2U);
    if (ba != bb)
      result.identical = false;
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

dif::ir::Program one_attention(std::uint64_t sequence, std::uint64_t heads) {
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

// Two chained attentions: op 2 consumes op 1's output as its query.
dif::ir::Program two_attentions(std::uint64_t sequence, std::uint64_t heads) {
  using namespace dif::ir;
  Program program;
  const std::vector<std::uint64_t> shape{sequence, heads, 128U};
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, shape},
      {2U, DType::BF16, TensorRole::Input, shape},
      {3U, DType::BF16, TensorRole::Input, shape},
      {4U, DType::BF16, TensorRole::Internal, shape},
      {5U, DType::BF16, TensorRole::Output, shape},
  };
  program.operations = {
      {1U, Opcode::Attention, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::Implementation, 2U)}},
      {2U, Opcode::Attention, {4U, 2U, 3U}, {5U},
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

dif::runtime::TensorMap inputs_for(std::uint64_t sequence, std::uint64_t heads,
                                   std::uint64_t seed) {
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, random_bf16({sequence, heads, 128U}, seed + 11U, 1.0f));
  inputs.emplace(2U, random_bf16({sequence, heads, 128U}, seed + 23U, 1.0f));
  inputs.emplace(3U, random_bf16({sequence, heads, 128U}, seed + 37U, 1.0f));
  return inputs;
}

bool refused(const dif::ir::Program &program, const dif::runtime::TensorMap &inputs,
             const dif::runtime::RunOptions &options, const char *needle) {
  try {
    (void)dif::runtime::make_cuda_executor()->run(program, inputs, options);
  } catch (const dif::Error &error) {
    return std::string(error.what()).find(needle) != std::string::npos;
  }
  return false;
}

void test_exact_query_rows(std::uint64_t sequence, std::uint64_t heads) {
  const auto label = "exact-rows S=" + std::to_string(sequence);
  const auto program = one_attention(sequence, heads);
  const auto inputs = inputs_for(sequence, heads, sequence);
  const auto exact =
      dif::runtime::make_cuda_executor()->run(program, inputs, base_options());
  auto owned_options = base_options();
  owned_options.h3_owned_attention = true;
  const auto plain =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  const std::vector<dif::runtime::RunOptions::QueryRowRange> ranges{
      {100U, 50U}, {600U, 200U}};
  owned_options.h3_int8_attention_exact_query_ranges = ranges;
  const auto overlay =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  const auto repeat =
      dif::runtime::make_cuda_executor()->run(program, inputs, owned_options);
  expect(overlay.h3_int8_attention_exact_row_dispatches == ranges.size(),
         label + ": one exact dispatch per range per routed operation, got " +
             std::to_string(overlay.h3_int8_attention_exact_row_dispatches));
  expect(overlay.h3_int8_attention_exact_query_ranges.size() == ranges.size(),
         label + ": receipt echoes the ranges");
  expect(overlay.run_telemetry.ck_attention_dispatches == 1U &&
             overlay.run_telemetry.cudnn_attention_dispatches == ranges.size(),
         label + ": telemetry counts one INT8 and two exact dispatches");
  expect(overlay.outputs.at(4U).bytes == repeat.outputs.at(4U).bytes,
         label + ": overlay route is repeat-bit-exact");
  // Inside the ranges the rows are exact cuDNN attention over the full K/V;
  // outside them the rows are untouched INT8 output.
  for (const auto &range : ranges) {
    const auto inside =
        compare_rows(exact.outputs.at(4U), overlay.outputs.at(4U), range.begin,
                     range.count);
    const auto was_int8 = compare_rows(exact.outputs.at(4U), plain.outputs.at(4U),
                                       range.begin, range.count);
    std::cout << "EXACT_ROWS " << label << " range=[" << range.begin << ",+"
              << range.count << ") overlay_cosine=" << inside.cosine
              << " overlay_max_abs=" << inside.max_abs
              << " int8_cosine=" << was_int8.cosine
              << " int8_max_abs=" << was_int8.max_abs << "\n";
    expect(inside.finite, label + ": overlay rows are finite");
    expect(inside.cosine >= 0.999999 && inside.max_abs <= 0.02,
           label + ": overlay rows match exact cuDNN attention");
    expect(inside.max_abs <= was_int8.max_abs,
           label + ": overlay rows are at least as close as the INT8 rows");
  }
  expect(compare_rows(plain.outputs.at(4U), overlay.outputs.at(4U), 0U, 100U)
             .identical,
         label + ": rows before the first range are untouched INT8 output");
  expect(compare_rows(plain.outputs.at(4U), overlay.outputs.at(4U), 150U, 450U)
             .identical,
         label + ": rows between the ranges are untouched INT8 output");
  expect(compare_rows(plain.outputs.at(4U), overlay.outputs.at(4U), 800U,
                      sequence - 800U)
             .identical,
         label + ": rows after the last range are untouched INT8 output");
}

void test_exact_query_rows_fail_closed() {
  const auto program = one_attention(1024U, 2U);
  const auto inputs = inputs_for(1024U, 2U, 5U);
  auto options = base_options();
  options.h3_int8_attention_exact_query_ranges = {{100U, 50U}};
  expect(refused(program, inputs, options, "require an H3 INT8 attention route"),
         "exact rows without an INT8 route are refused");
  options.h3_owned_attention = true;
  options.h3_int8_attention_exact_query_ranges = {{100U, 50U}, {120U, 10U}};
  expect(refused(program, inputs, options, "overlaps"),
         "overlapping exact row ranges are refused");
  options.h3_int8_attention_exact_query_ranges = {{1020U, 10U}};
  expect(refused(program, inputs, options, "out of bounds"),
         "out-of-bounds exact row range is refused");
  options.h3_int8_attention_exact_query_ranges = {{10U, 0U}};
  expect(refused(program, inputs, options, "empty"),
         "empty exact row range is refused");
}

void test_hybrid_subrange(std::uint64_t sequence, std::uint64_t heads) {
  const auto label = "hybrid-subrange S=" + std::to_string(sequence);
  const auto program = two_attentions(sequence, heads);
  const auto inputs = inputs_for(sequence, heads, 3U * sequence);
  // Expected "op 1 exact, op 2 INT8": run the single-attention program exact,
  // feed its output as op 2's query on the INT8 route.
  const auto single = one_attention(sequence, heads);
  const auto exact_first =
      dif::runtime::make_cuda_executor()->run(single, inputs, base_options());
  dif::runtime::TensorMap second_inputs;
  second_inputs.emplace(1U, exact_first.outputs.at(4U));
  second_inputs.emplace(2U, inputs.at(2U));
  second_inputs.emplace(3U, inputs.at(3U));
  auto owned = base_options();
  owned.h3_owned_attention = true;
  const auto expected =
      dif::runtime::make_cuda_executor()->run(single, second_inputs, owned);

  auto hybrid = owned;
  hybrid.h3_int8_attention_hybrid = true;
  hybrid.h3_int8_attention_hybrid_first_layer = 0U;
  hybrid.h3_int8_attention_hybrid_layers = 1U;
  hybrid.h3_int8_attention_active = false;
  const auto mixed =
      dif::runtime::make_cuda_executor()->run(program, inputs, hybrid);
  expect(mixed.run_telemetry.ck_attention_dispatches == 1U &&
             mixed.run_telemetry.cudnn_attention_dispatches == 1U,
         label + ": inactive run sends op 1 exact and op 2 INT8");
  expect(mixed.h3_int8_attention_hybrid_first_layer == 0U &&
             mixed.h3_int8_attention_hybrid_layers == 1U,
         label + ": receipt echoes the hybrid sub-range");
  expect(mixed.outputs.at(5U).bytes == expected.outputs.at(4U).bytes,
         label + ": mixed output is bit-identical to exact-then-INT8");

  hybrid.h3_int8_attention_active = true;
  const auto all_int8 =
      dif::runtime::make_cuda_executor()->run(program, inputs, hybrid);
  expect(all_int8.run_telemetry.ck_attention_dispatches == 2U &&
             all_int8.run_telemetry.cudnn_attention_dispatches == 0U,
         label + ": active run sends both operations INT8");

  hybrid.h3_int8_attention_active = false;
  hybrid.h3_int8_attention_hybrid_first_layer = 1U;
  const auto other =
      dif::runtime::make_cuda_executor()->run(program, inputs, hybrid);
  expect(other.run_telemetry.ck_attention_dispatches == 1U &&
             other.run_telemetry.cudnn_attention_dispatches == 1U &&
             other.outputs.at(5U).bytes != mixed.outputs.at(5U).bytes,
         label + ": sub-range [1,2) exact differs from [0,1) exact");

  auto bad = owned;
  bad.h3_int8_attention_hybrid_first_layer = 1U;
  expect(refused(program, inputs, bad, "require h3_int8_attention_hybrid"),
         "hybrid sub-range without the hybrid is refused");
  bad.h3_int8_attention_hybrid = true;
  bad.h3_int8_attention_hybrid_first_layer = 2U;
  expect(refused(program, inputs, bad, "leaves the routed"),
         "hybrid sub-range past the routed operations is refused");
  bad.h3_int8_attention_hybrid_first_layer = 1U;
  bad.h3_int8_attention_hybrid_layers = 2U;
  expect(refused(program, inputs, bad, "leaves the routed"),
         "hybrid sub-range overrunning the route is refused");
}

} // namespace

int main() {
  if (!dif::runtime::cuda_available()) {
    std::cout << "CUDA unavailable; H3 route refinement gates skipped\n";
    return 0;
  }
#if !DIF_HAS_H3_OWNED_ATTENTION || !DIF_HAS_CUDNN
  std::cout << "build lacks the in-tree owned attention or cuDNN; gates skipped\n";
  return 0;
#else
  const auto probe =
      dif::runtime::probe_target(dif::runtime::ProbeBackend::Cuda, 0);
  if (probe.compute_major * 10U + probe.compute_minor != 86U) {
    std::cout << "owned attention kernel is sm_86 only; gates skipped\n";
    return 0;
  }
  test_exact_query_rows_fail_closed();
  test_exact_query_rows(1536U, 4U);
  test_hybrid_subrange(1536U, 4U);
  if (failures != 0) {
    std::cerr << failures << " H3 route refinement test failure(s)\n";
    return 1;
  }
  std::cout << "H3 route refinement tests passed\n";
  return 0;
#endif
}
