// Optimizer and knob contracts.
// 1. AdamWUpdate ClipScale folds gradient clipping into the kernel: with
//    ClipScale=c the update must be bit-identical to the un-clipped update on
//    a gradient pre-multiplied by c (CPU), the default must be a provable
//    no-op, the verifier must reject out-of-range values, and CUDA must match
//    CPU within one BF16 ulp of the parameter magnitude.
// 2. Every RunOptions knob added this month is a no-op at its default
//    (Flame levers.rs contract): outputs byte-identical with and without it.
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<float> &values) {
  std::vector<std::uint8_t> bytes(values.size() * sizeof(float));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  return dif::runtime::Tensor(dif::ir::DType::F32, std::move(dims),
                              std::move(bytes));
}

dif::runtime::Tensor i32_tensor(std::int32_t value) {
  std::vector<std::uint8_t> bytes(sizeof(std::int32_t));
  std::memcpy(bytes.data(), &value, sizeof(value));
  return dif::runtime::Tensor(dif::ir::DType::I32, {1U}, std::move(bytes));
}

std::vector<float> pattern(std::size_t count, float phase, float amplitude) {
  std::vector<float> values(count);
  for (std::size_t index = 0; index < count; ++index)
    values[index] = amplitude * std::sin(0.37F * static_cast<float>(index) + phase);
  return values;
}

// parameter(1) gradient(2) first(3) second(4) step(5) -> updated(6,7,8)
dif::ir::Program adamw_program(std::uint64_t count, double clip_scale) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1U, DType::F32, TensorRole::Input, {count}},
      {2U, DType::F32, TensorRole::Input, {count}},
      {3U, DType::F32, TensorRole::Input, {count}},
      {4U, DType::F32, TensorRole::Input, {count}},
      {5U, DType::I32, TensorRole::Input, {1U}},
      {6U, DType::F32, TensorRole::Output, {count}},
      {7U, DType::F32, TensorRole::Output, {count}},
      {8U, DType::F32, TensorRole::Output, {count}},
  };
  std::vector<Attribute> attributes = {
      Attribute::f64(AttrKey::LearningRate, 1.0e-2),
      Attribute::f64(AttrKey::Beta1, 0.9),
      Attribute::f64(AttrKey::Beta2, 0.999),
      Attribute::f64(AttrKey::Epsilon, 1.0e-8),
      Attribute::f64(AttrKey::WeightDecay, 1.0e-2),
  };
  if (clip_scale > 0.0)
    attributes.push_back(Attribute::f64(AttrKey::ClipScale, clip_scale));
  program.operations = {
      {1U, Opcode::AdamWUpdate, {1U, 2U, 3U, 4U, 5U}, {6U, 7U, 8U},
       std::move(attributes)},
  };
  return program;
}

dif::runtime::RunOptions single_run_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return options;
}

bool same_bytes(const dif::runtime::Tensor &a, const dif::runtime::Tensor &b) {
  return a.byte_size() == b.byte_size() &&
         std::memcmp(a.data(), b.data(), a.byte_size()) == 0;
}

void test_clip_scale_semantics() {
  constexpr std::uint64_t count = 257U;
  const auto parameter = pattern(count, 0.1F, 0.5F);
  const auto gradient = pattern(count, 1.3F, 0.2F);
  const auto first = pattern(count, 2.1F, 0.05F);
  std::vector<float> second(count);
  for (std::size_t index = 0; index < count; ++index)
    second[index] = 1.0e-4F * (1.0F + static_cast<float>(index % 7U));
  const auto bind = [&](const std::vector<float> &grad) {
    dif::runtime::TensorMap map;
    map.emplace(1U, f32_tensor({count}, parameter));
    map.emplace(2U, f32_tensor({count}, grad));
    map.emplace(3U, f32_tensor({count}, first));
    map.emplace(4U, f32_tensor({count}, second));
    map.emplace(5U, i32_tensor(3));
    return map;
  };
  constexpr float clip = 0.25F;
  std::vector<float> scaled(count);
  for (std::size_t index = 0; index < count; ++index)
    scaled[index] = gradient[index] * clip;

  const auto clipped_program = adamw_program(count, clip);
  const auto plain_program = adamw_program(count, 0.0);
  dif::ir::verify(clipped_program);
  dif::ir::verify(plain_program);
  const auto cpu = dif::runtime::make_cpu_executor();
  const auto clipped = cpu->run(clipped_program, bind(gradient), single_run_options());
  const auto reference = cpu->run(plain_program, bind(scaled), single_run_options());
  for (const auto id : {6U, 7U, 8U})
    expect(same_bytes(clipped.outputs.at(id), reference.outputs.at(id)),
           "ClipScale=c is bit-identical to the un-clipped update on c*gradient (output " +
               std::to_string(id) + ")");

  const auto default_explicit = cpu->run(adamw_program(count, 1.0), bind(gradient), single_run_options());
  const auto default_absent = cpu->run(plain_program, bind(gradient), single_run_options());
  for (const auto id : {6U, 7U, 8U})
    expect(same_bytes(default_explicit.outputs.at(id), default_absent.outputs.at(id)),
           "ClipScale=1.0 is a no-op (output " + std::to_string(id) + ")");
  expect(!same_bytes(clipped.outputs.at(6U), default_absent.outputs.at(6U)),
         "ClipScale=0.25 changes the update (the gate can fail)");

  for (const double bad : {0.0, -0.5, 2.0}) {
    bool rejected = false;
    try {
      auto program = adamw_program(count, 1.0);
      program.operations[0].attributes.back() =
          dif::ir::Attribute::f64(dif::ir::AttrKey::ClipScale, bad);
      dif::ir::verify(program);
    } catch (const std::exception &) {
      rejected = true;
    }
    expect(rejected, "verifier rejects ClipScale " + std::to_string(bad));
  }

  if (dif::runtime::cuda_available()) {
    const auto cuda = dif::runtime::make_cuda_executor()->run(
        clipped_program, bind(gradient), single_run_options());
    float worst = 0.0F;
    for (const auto id : {6U, 7U, 8U})
      for (std::uint64_t index = 0; index < count; ++index)
        worst = std::max(worst, std::abs(dif::runtime::load_float(cuda.outputs.at(id), index) -
                                         dif::runtime::load_float(clipped.outputs.at(id), index)));
    expect(worst <= 1.0e-6F, "CUDA ClipScale update matches CPU (max_abs=" + std::to_string(worst) + ")");
    std::cout << "GATE adamw_clip_scale backend=" << cuda.backend_name
              << " max_abs=" << worst << "\n";
  }
}

// A tiny CUDA program whose outputs must not change under any default-valued
// knob: the two page-cache policies and the MXFP8/FP8 guards added this month
// are exercised through their default paths.
void test_knobs_noop_at_default() {
  if (!dif::runtime::cuda_available())
    return;
  using namespace dif::ir;
  constexpr std::uint64_t rows = 64U, inner = 96U, columns = 48U;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, inner}},
      {2U, DType::BF16, TensorRole::Constant, {columns, inner}},
      {3U, DType::BF16, TensorRole::Output, {rows, columns}},
  };
  program.operations = {{1U, Opcode::Linear, {1U, 2U}, {3U}, {}}};
  dif::ir::verify(program);
  const auto to_bf16 = [](const std::vector<float> &values, std::vector<std::uint64_t> dims) {
    std::vector<std::uint8_t> bytes(values.size() * 2U);
    for (std::size_t index = 0; index < values.size(); ++index) {
      std::uint32_t bits;
      std::memcpy(&bits, &values[index], sizeof(bits));
      const std::uint16_t half = static_cast<std::uint16_t>((bits + 0x7FFFU + ((bits >> 16U) & 1U)) >> 16U);
      std::memcpy(bytes.data() + 2U * index, &half, 2U);
    }
    return dif::runtime::Tensor(DType::BF16, std::move(dims), std::move(bytes));
  };
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U, to_bf16(pattern(rows * inner, 0.2F, 1.0F), {rows, inner}));
  bindings.emplace(2U, to_bf16(pattern(columns * inner, 0.9F, 0.1F), {columns, inner}));
  const auto executor = dif::runtime::make_cuda_executor();
  const auto baseline = executor->run(program, bindings, single_run_options());
  struct Knob {
    const char *name;
    void (*apply)(dif::runtime::RunOptions &);
  };
  const Knob knobs[] = {
      {"resident_evict_host_pages=false",
       [](dif::runtime::RunOptions &o) { o.resident_evict_host_pages = false; }},
      {"streamed_release_mapped_pages_per_copy=false",
       [](dif::runtime::RunOptions &o) { o.streamed_release_mapped_pages_per_copy = false; }},
      {"lazy_resident_upload=true",
       [](dif::runtime::RunOptions &o) { o.lazy_resident_upload = true; }},
  };
  for (const auto &knob : knobs) {
    auto options = single_run_options();
    knob.apply(options);
    const auto candidate = executor->run(program, bindings, options);
    expect(same_bytes(candidate.outputs.at(3U), baseline.outputs.at(3U)),
           std::string("knob is byte-identical to the default: ") + knob.name);
  }
  std::cout << "GATE knobs_noop_at_default backend=" << baseline.backend_name << " knobs=3\n";
}

} // namespace

int main() {
  try {
    test_clip_scale_semantics();
    test_knobs_noop_at_default();
  } catch (const std::exception &error) {
    std::cerr << "FAIL: exception: " << error.what() << "\n";
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " optimizer test failure(s)\n";
    return 1;
  }
  std::cout << "OPTIMIZER_TESTS PASS\n";
  return 0;
}
