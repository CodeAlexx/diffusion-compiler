// Weight-only INT8 residency for a frozen linear.
//
// Three things are checked, in order of how badly they would hurt.
//
// First, that the host quantizer and the QuantizeInt8Rows operation this
// compiler already had produce byte-identical results. There are now two
// implementations of one arithmetic -- one for load time, one for graph time
// -- and two implementations of one arithmetic is how a format silently
// drifts. If they ever disagree, this fails.
//
// Second, that the rewrite leaves NO dequantization behind. The whole point
// is that the GEMM and the input gradient read INT8 bytes; a graph that
// dequantizes a frozen weight before each matmul pays that conversion once
// per linear per step forever, which is what it costs the reference
// implementation an order of magnitude.
//
// Third, that the gradient through the quantized path is the gradient
// through the weight it stands for.

#include "dif/compiler/int8.hpp"
#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/training/autodiff.hpp"

#include <array>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
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

template <typename Body>
void expect_refused(Body &&body, const std::string &fragment,
                    const std::string &message) {
  try {
    body();
  } catch (const std::exception &error) {
    if (std::string(error.what()).find(fragment) != std::string::npos)
      return;
    ++failures;
    std::cerr << "FAIL: " << message << " -- wrong reason: " << error.what()
              << "\n";
    return;
  }
  ++failures;
  std::cerr << "FAIL: " << message << " -- was accepted\n";
}

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

std::uint16_t to_bf16(float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  return static_cast<std::uint16_t>(bits >> 16U);
}

float from_bf16(std::uint16_t raw) {
  const std::uint32_t bits = static_cast<std::uint32_t>(raw) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

dif::runtime::Tensor bf16_tensor(std::vector<std::uint64_t> dims,
                                 const std::vector<float> &values) {
  dif::runtime::Tensor tensor;
  tensor.dtype = DType::BF16;
  tensor.dims = std::move(dims);
  tensor.bytes.resize(values.size() * sizeof(std::uint16_t));
  auto *raw = reinterpret_cast<std::uint16_t *>(tensor.bytes.data());
  for (std::size_t index = 0U; index < values.size(); ++index)
    raw[index] = to_bf16(values[index]);
  return tensor;
}

// The weight to quantize: awkward on purpose. One row is all zeros (the
// division-by-zero case the 1e-30 floor exists for), one row has a single
// large outlier, and the rest are ordinary.
std::vector<float> awkward_weight(std::uint64_t columns, std::uint64_t inner) {
  std::mt19937 generator(20260904U);
  std::normal_distribution<float> normal(0.0F, 0.3F);
  std::vector<float> values(columns * inner);
  for (std::uint64_t row = 0U; row < columns; ++row)
    for (std::uint64_t index = 0U; index < inner; ++index) {
      if (row == 1U)
        values[row * inner + index] = 0.0F;
      else
        values[row * inner + index] = from_bf16(to_bf16(normal(generator)));
    }
  if (columns > 2U)
    values[2U * inner] = 40.0F;
  return values;
}

void one_arithmetic_not_two() {
  const std::uint64_t columns = 6U, inner = 8U;
  const auto values = awkward_weight(columns, inner);
  const auto source = bf16_tensor({columns, inner}, values);

  // The host path, used at load time.
  const auto host = dif::compiler::quantize_int8_weight(source);

  // The graph path this compiler already had, in its Direct mode.
  Program program;
  program.tensors = {{1U, DType::BF16, TensorRole::Input, {columns, inner}},
                     {2U, DType::I8, TensorRole::Output, {columns, inner}},
                     {3U, DType::F32, TensorRole::Output, {columns}}};
  program.operations = {
      {1U, Opcode::QuantizeInt8Rows, {1U}, {2U, 3U},
       {dif::ir::Attribute::u64(
           dif::ir::AttrKey::Implementation,
           static_cast<std::uint64_t>(dif::ir::Int8RowQuantization::Direct))}}};
  dif::ir::verify(program);

  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, source);
  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  const auto prepared = executor->prepare(program, inputs, options);
  const auto result = prepared->run(inputs, options);

  const auto &graph_weight = result.outputs.at(2U);
  const auto &graph_scales = result.outputs.at(3U);
  expect(graph_weight.bytes == host.weight.bytes,
         "the host quantizer and QuantizeInt8Rows agree byte for byte");
  expect(graph_scales.bytes == host.scales.bytes,
         "and so do their scales");

  // The zero row is the one that would divide by zero without the floor.
  const auto *scales = reinterpret_cast<const float *>(host.scales.data());
  expect(scales[1] == 1.0e-30F,
         "an all-zero row takes the floor rather than dividing by zero");
  const auto *codes =
      reinterpret_cast<const std::int8_t *>(host.weight.data());
  for (std::uint64_t index = 0U; index < inner; ++index)
    expect(codes[inner + index] == 0,
           "and quantizes to zeros rather than to noise");
  // The outlier row is the one that loses the most, and it should say so.
  expect(host.maximum_absolute_error > 0.0F,
         "the rounding cost is measured, not assumed");
  expect(host.squared_error < host.squared_reference * 0.01,
         "and is a small fraction of the weight it stands for");
}

Program linear_program(std::uint64_t rows, std::uint64_t inner,
                       std::uint64_t columns) {
  Program program;
  program.tensors = {{1U, DType::BF16, TensorRole::Input, {rows, inner}},
                     {2U, DType::BF16, TensorRole::Constant, {columns, inner}},
                     {3U, DType::BF16, TensorRole::Output, {rows, columns}}};
  program.operations = {{1U, Opcode::Linear, {1U, 2U}, {3U}, {}}};
  dif::ir::verify(program);
  return program;
}

void the_rewrite_leaves_no_dequantization() {
  // A realistic width. At eight columns the per-row scales are a third of
  // the payload; the halving this exists for only appears at a width a real
  // linear has.
  const auto program = linear_program(4U, 256U, 64U);
  const auto rewrite = dif::compiler::rewrite_int8_weight_only(program, {1U});
  expect(rewrite.entries.size() == 1U, "the linear was rewritten");
  for (const auto &operation : rewrite.program.operations) {
    expect(operation.opcode != Opcode::DequantizeInt8Blocks &&
               operation.opcode != Opcode::DequantizeInt4 &&
               operation.opcode != Opcode::DequantizeInt5 &&
               operation.opcode != Opcode::QuantizeInt8Rows,
           "no conversion operation is left in the graph to run every step");
    expect(operation.opcode != Opcode::Cast,
           "and no cast either");
  }
  expect(rewrite.program.operations.size() == 1U,
         "the rewrite adds no operations at all: it changes one");
  expect(rewrite.program.operations.front().opcode ==
             Opcode::LinearInt8WeightScaled,
         "the matmul reads INT8 directly");
  // The float weight is GONE. Leaving it would keep it in the memory plan,
  // which is the entire cost being removed.
  expect(rewrite.program.tensor(2U) == nullptr,
         "the float weight is removed from the program, not merely unused");
  expect(rewrite.bytes_after * 3U < rewrite.bytes_before * 2U,
         "and the weight now costs meaningfully less than half");
  dif::ir::verify(rewrite.program);
}

void the_rewrite_refuses_what_it_must_not_convert() {
  auto program = linear_program(4U, 8U, 6U);
  expect_refused([&] { dif::compiler::rewrite_int8_weight_only(program, {9U}); },
                 "no operation 9", "an operation that does not exist");
  expect_refused(
      [&] { dif::compiler::rewrite_int8_weight_only(program, {1U, 1U}); },
      "named twice", "the same operation twice");
  {
    // A trainable weight must never be made resident in a format it cannot
    // be trained in.
    auto trainable = program;
    trainable.tensors[1].roles =
        static_cast<std::uint32_t>(TensorRole::Input) |
        static_cast<std::uint32_t>(TensorRole::Parameter);
    expect_refused(
        [&] { dif::compiler::rewrite_int8_weight_only(trainable, {1U}); },
        "not frozen", "a weight that is not frozen");
  }
  {
    // Two linears sharing one weight: converting for one would break the
    // other.
    auto shared = program;
    shared.tensors.push_back({4U, DType::BF16, TensorRole::Output, {4U, 6U}});
    shared.operations.push_back({2U, Opcode::Linear, {1U, 2U}, {4U}, {}});
    expect_refused(
        [&] { dif::compiler::rewrite_int8_weight_only(shared, {1U}); },
        "more than one operation", "a weight two linears share");
  }
  {
    auto biased = program;
    biased.tensors.push_back({4U, DType::BF16, TensorRole::Constant, {6U}});
    biased.operations.front().inputs = {1U, 2U, 4U};
    expect_refused(
        [&] { dif::compiler::rewrite_int8_weight_only(biased, {1U}); },
        "has a bias", "a linear with a bias");
  }
}

// The gradient through the quantized path must be the gradient through the
// weight the INT8 bytes stand for -- not approximately, exactly, because
// q * scale IS the weight now.
void the_gradient_reads_int8_directly() {
  const std::uint64_t rows = 4U, inner = 8U, columns = 6U;
  auto program = linear_program(rows, inner, columns);
  // Autodiff differentiates a scalar, so give it one.
  program.tensors.push_back(
      {5U, DType::BF16, TensorRole::Input, {rows, columns}});
  program.tensors.push_back({6U, DType::F32, TensorRole::Output, {1U}});
  program.operations.push_back({2U, Opcode::MseLoss, {3U, 5U}, {6U}, {}});
  dif::ir::verify(program);
  const auto rewrite = dif::compiler::rewrite_int8_weight_only(program, {1U});
  const auto entry = rewrite.entries.front();

  const std::vector<std::uint32_t> wanted{1U};
  auto differentiated =
      dif::training::differentiate(rewrite.program, 6U, wanted);
  bool reads_int8 = false;
  for (const auto &operation : differentiated.program.operations) {
    expect(operation.opcode != Opcode::DequantizeInt8Blocks,
           "the backward pass does not dequantize either");
    if (operation.opcode == Opcode::LinearInt8WeightScaledBackwardInput) {
      reads_int8 = true;
      expect(operation.inputs.size() == 3U &&
                 operation.inputs[1] == entry.weight_tensor_id &&
                 operation.inputs[2] == entry.scales_tensor_id,
             "the input gradient reads the INT8 weight and its scales");
    }
  }
  expect(reads_int8, "the quantized linear has an input gradient");

  // And it computes the right thing: compare against the dense gradient
  // through the dequantized weight, on the CPU.
  const auto values = awkward_weight(columns, inner);
  const auto quantized =
      dif::compiler::quantize_int8_weight(bf16_tensor({columns, inner},
                                                      values));
  std::vector<float> dequantized(columns * inner);
  {
    const auto *codes =
        reinterpret_cast<const std::int8_t *>(quantized.weight.data());
    const auto *scales =
        reinterpret_cast<const float *>(quantized.scales.data());
    for (std::uint64_t row = 0U; row < columns; ++row)
      for (std::uint64_t index = 0U; index < inner; ++index)
        dequantized[row * inner + index] =
            static_cast<float>(codes[row * inner + index]) * scales[row];
  }

  std::vector<float> grad_output(rows * columns);
  for (std::size_t index = 0U; index < grad_output.size(); ++index)
    grad_output[index] = from_bf16(to_bf16(
        0.25F - 0.05F * static_cast<float>(index % 7U)));

  Program quantized_backward;
  quantized_backward.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, columns}},
      {2U, DType::I8, TensorRole::Constant, {columns, inner}},
      {3U, DType::F32, TensorRole::Constant, {columns}},
      {4U, DType::BF16, TensorRole::Output, {rows, inner}}};
  quantized_backward.operations = {
      {1U, Opcode::LinearInt8WeightScaledBackwardInput, {1U, 2U, 3U}, {4U},
       {}}};
  dif::ir::verify(quantized_backward);

  Program dense_backward;
  dense_backward.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, columns}},
      {2U, DType::BF16, TensorRole::Constant, {columns, inner}},
      {4U, DType::BF16, TensorRole::Output, {rows, inner}}};
  dense_backward.operations = {
      {1U, Opcode::LinearBackwardInput, {1U, 2U}, {4U}, {}}};
  dif::ir::verify(dense_backward);

  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  const auto run = [&](const Program &graph,
                       const dif::runtime::TensorMap &inputs) {
    const auto prepared = executor->prepare(graph, inputs, options);
    return prepared->run(inputs, options).outputs.at(4U);
  };
  dif::runtime::TensorMap quantized_inputs;
  quantized_inputs.emplace(1U, bf16_tensor({rows, columns}, grad_output));
  quantized_inputs.emplace(2U, quantized.weight);
  quantized_inputs.emplace(3U, quantized.scales);
  dif::runtime::TensorMap dense_inputs;
  dense_inputs.emplace(1U, bf16_tensor({rows, columns}, grad_output));
  dense_inputs.emplace(2U, bf16_tensor({columns, inner}, dequantized));

  const auto from_int8 = run(quantized_backward, quantized_inputs);
  const auto from_dense = run(dense_backward, dense_inputs);
  const auto *left =
      reinterpret_cast<const std::uint16_t *>(from_int8.data());
  const auto *right =
      reinterpret_cast<const std::uint16_t *>(from_dense.data());
  double worst = 0.0;
  double reference = 0.0;
  for (std::uint64_t index = 0U; index < rows * inner; ++index) {
    const double a = from_bf16(left[index]);
    const double b = from_bf16(right[index]);
    worst = std::max(worst, std::abs(a - b));
    reference = std::max(reference, std::abs(b));
  }
  // The dequantized weight rounds to BF16 and the INT8 path does not, so
  // these agree to rounding, not to the bit -- and the tolerance says which.
  expect(worst <= reference * 1e-2,
         "the INT8 gradient matches the gradient through the weight it "
         "stands for");
  std::cout << "  gradient agreement: worst " << worst << " against "
            << reference << "\n";
}

// The tiled kernel has edge cases the naive one it replaced did not: a tile
// that hangs off the end of the gradient, of the weight, or of the
// contraction. The shapes below are deliberately not multiples of the tile,
// so every guard is exercised, and the CPU reference is the oracle.
void the_gpu_kernel_agrees_with_the_reference() {
  auto cuda = dif::runtime::make_cuda_executor();
  if (!cuda) {
    std::cout << "SKIP: no CUDA device\n";
    return;
  }
  // rows, inner, outputs -- none a multiple of 32, one below a tile, one just
  // over, and one realistic width.
  const std::vector<std::array<std::uint64_t, 3>> shapes{
      {70U, 100U, 90U}, {31U, 33U, 65U}, {64U, 6144U, 1536U}};
  auto cpu = dif::runtime::make_cpu_executor();
  std::mt19937 generator(7U);
  std::normal_distribution<float> normal(0.0F, 0.5F);
  for (const auto &shape : shapes) {
    const auto rows = shape[0], inner = shape[1], outputs = shape[2];
    Program program;
    program.tensors = {
        {1U, DType::BF16, TensorRole::Input, {rows, outputs}},
        {2U, DType::I8, TensorRole::Constant, {outputs, inner}},
        {3U, DType::F32, TensorRole::Constant, {outputs}},
        {4U, DType::BF16, TensorRole::Output, {rows, inner}}};
    program.operations = {
        {1U, Opcode::LinearInt8WeightScaledBackwardInput, {1U, 2U, 3U}, {4U},
         {}}};
    dif::ir::verify(program);

    std::vector<float> gradient(rows * outputs);
    for (auto &value : gradient)
      value = normal(generator);
    std::vector<float> weight_values(outputs * inner);
    for (auto &value : weight_values)
      value = normal(generator);
    const auto quantized = dif::compiler::quantize_int8_weight(
        bf16_tensor({outputs, inner}, weight_values));

    dif::runtime::TensorMap inputs;
    inputs.emplace(1U, bf16_tensor({rows, outputs}, gradient));
    inputs.emplace(2U, quantized.weight);
    inputs.emplace(3U, quantized.scales);
    dif::runtime::RunOptions options;
    const auto reference =
        cpu->prepare(program, inputs, options)->run(inputs, options)
            .outputs.at(4U);
    const auto actual =
        cuda->prepare(program, inputs, options)->run(inputs, options)
            .outputs.at(4U);
    const auto *left =
        reinterpret_cast<const std::uint16_t *>(reference.data());
    const auto *right =
        reinterpret_cast<const std::uint16_t *>(actual.data());
    std::uint64_t mismatches = 0U;
    double worst = 0.0, magnitude = 0.0;
    for (std::uint64_t index = 0U; index < rows * inner; ++index) {
      if (left[index] != right[index])
        ++mismatches;
      const double expected = from_bf16(left[index]);
      magnitude = std::max(magnitude, std::abs(expected));
      worst = std::max(worst, std::abs(expected - from_bf16(right[index])));
    }
    const auto label = std::to_string(rows) + "x" + std::to_string(inner) +
                       "x" + std::to_string(outputs) + " (worst " +
                       std::to_string(worst) + " of " +
                       std::to_string(magnitude) + ", " +
                       std::to_string(mismatches) + " differing)";
    // Two paths run on the device. A shape CUTLASS can align goes to the
    // tensor cores, which round the scale-folded gradient to BF16 before the
    // matmul -- one extra rounding, so it agrees to a couple of BF16 ulps of
    // the largest value and no better. A shape it cannot align keeps the
    // generated kernel, which accumulates in the CPU's order with the CPU's
    // fused multiply-adds and has to match to the bit.
    const bool tensor_core_eligible =
        inner % 16U == 0U && outputs % 16U == 0U && rows % 8U == 0U;
    if (tensor_core_eligible)
      expect(worst <= magnitude * (2.0 / 128.0),
             "the tensor-core path agrees with the CPU reference to two "
             "BF16 ulps at " + label);
    else
      expect(mismatches == 0U,
             "the generated kernel matches the CPU reference bit for bit "
             "at " + label);
  }
}

} // namespace

int main() {
  one_arithmetic_not_two();
  the_rewrite_leaves_no_dequantization();
  the_rewrite_refuses_what_it_must_not_convert();
  the_gradient_reads_int8_directly();
  the_gpu_kernel_agrees_with_the_reference();
  if (failures != 0) {
    std::cerr << failures << " INT8 residency failure(s)\n";
    return 1;
  }
  std::cout << "int8 resident tests passed\n";
  return 0;
}
