// Finite-difference gradient tests for the training opcodes.
//
// Every rule the autodiff pass learns is checked the only way a gradient can
// honestly be checked: against a central difference of the forward program it
// claims to differentiate. The programs run in F32 on the CPU executor, where
// a difference quotient is meaningful, and each one ends in a scalar loss so
// the seed gradient is well defined.
//
// A rule that is subtly wrong -- a transposed index, a missing scale, a
// forgotten fan-in -- shows up here as a mismatch, not as a silently bad
// training run three days later.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/training/autodiff.hpp"

#include <algorithm>
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

using dif::ir::AttrKey;
using dif::ir::Attribute;
using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                std::uint64_t seed, float amplitude) {
  std::uint64_t count = 1U;
  for (const auto dim : dims)
    count *= dim;
  dif::runtime::Tensor tensor{DType::F32, std::move(dims), {}};
  tensor.bytes.resize(static_cast<std::size_t>(count) * sizeof(float));
  tensor.validate();
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  for (std::uint64_t index = 0U; index < count; ++index) {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    const auto unit = static_cast<double>((state >> 33U) & 0x7fffffffU) /
                      static_cast<double>(0x7fffffffU);
    dif::runtime::store_float(tensor, index,
                              static_cast<float>((unit * 2.0 - 1.0) * amplitude));
  }
  return tensor;
}

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
}

double loss_of(const Program &program, const dif::runtime::TensorMap &inputs,
               std::uint32_t loss_tensor) {
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto result =
      dif::runtime::make_cpu_executor()->run(program, inputs, options);
  return dif::runtime::load_float(result.outputs.at(loss_tensor), 0U);
}

// Central differences against the analytic gradient of every element of every
// requested target.
void check_gradients(const std::string &label, const Program &forward,
                     const dif::runtime::TensorMap &inputs,
                     std::uint32_t loss_tensor,
                     const std::vector<std::uint32_t> &targets,
                     double step = 8.0e-3, double absolute = 3.0e-5,
                     double relative = 3.0e-3) {
  const auto differentiated =
      dif::training::differentiate(forward, loss_tensor, targets);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto analytic =
      dif::runtime::make_cpu_executor()->run(differentiated.program, inputs,
                                             options);
  for (const auto target : targets) {
    const auto gradient_id = differentiated.gradients.at(target);
    const auto &gradient = analytic.outputs.at(gradient_id);
    auto perturbed = inputs;
    auto &tensor = perturbed.at(target);
    const auto count = tensor.element_count();
    double worst = 0.0;
    std::uint64_t worst_index = 0U;
    double worst_analytic = 0.0;
    double worst_numeric = 0.0;
    for (std::uint64_t index = 0U; index < count; ++index) {
      const auto original = dif::runtime::load_float(tensor, index);
      dif::runtime::store_float(tensor, index,
                                static_cast<float>(original + step));
      const auto up = loss_of(forward, perturbed, loss_tensor);
      dif::runtime::store_float(tensor, index,
                                static_cast<float>(original - step));
      const auto down = loss_of(forward, perturbed, loss_tensor);
      dif::runtime::store_float(tensor, index, original);
      const auto numeric = (up - down) / (2.0 * step);
      const auto exact =
          static_cast<double>(dif::runtime::load_float(gradient, index));
      // The gradcheck criterion: an absolute floor covering the difference
      // quotient's own noise (an F32 loss divided by a step of this size
      // resolves to a few times 1e-5) plus a relative term for the rest.
      const auto budget = absolute + relative * std::abs(exact);
      const auto error = std::abs(numeric - exact) / budget;
      if (error > worst) {
        worst = error;
        worst_index = index;
        worst_analytic = exact;
        worst_numeric = numeric;
      }
    }
    std::cout << "  " << label << " tensor " << target << ": worst "
              << worst << " of budget\n";
    expect(worst <= 1.0,
           label + " tensor " + std::to_string(target) +
               ": worst error is " + std::to_string(worst) +
               " times its budget at element " + std::to_string(worst_index) +
               " (analytic " + std::to_string(worst_analytic) + ", numeric " +
               std::to_string(worst_numeric) + ")");
  }
}

// Every program below is: inputs -> the operation under test -> MSE against a
// target -> scalar loss.
struct Case {
  Program program;
  dif::runtime::TensorMap inputs;
  std::uint32_t loss{};
  std::vector<std::uint32_t> targets;
  // Both bars scale with how long the reduction feeding each output is,
  // because that is what decides how much F32 rounding accumulates. A 3-D
  // convolution sums three times as many taps per output as the 2-D one
  // beside it, and cuDNN's 3-D algorithms accumulate in a different order
  // again, so those cases carry a measured multiplier rather than the whole
  // suite being loosened to fit them. Both are set with headroom over the
  // measured worst and both are proven to still catch an injected defect.
  double gradient_budget_scale{1.0};
  double parity_bar_scale{1.0};
};

Case gelu_case(dif::ir::GeluApproximation approximation) {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {2U, DType::F32, TensorRole::Internal, {4U, 5U}},
                       {3U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Gelu, {1U}, {2U},
       {Attribute::u64(AttrKey::Approximation,
                       static_cast<std::uint64_t>(approximation))}},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({4U, 5U}, 3U, 2.0F));
  c.inputs.emplace(3U, f32_tensor({4U, 5U}, 5U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

Case upsample_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {1U, 2U, 3U, 3U}},
                       {2U, DType::F32, TensorRole::Internal, {1U, 2U, 6U, 9U}},
                       {3U, DType::F32, TensorRole::Input, {1U, 2U, 6U, 9U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::UpsampleNearest2d, {1U}, {2U},
       {Attribute::u64(AttrKey::ScaleH, 2U), Attribute::u64(AttrKey::ScaleW, 3U)}},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({1U, 2U, 3U, 3U}, 7U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({1U, 2U, 6U, 9U}, 11U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

Case slice_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {3U, 8U}},
                       {2U, DType::F32, TensorRole::Internal, {3U, 3U}},
                       {3U, DType::F32, TensorRole::Input, {3U, 3U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Slice, {1U}, {2U},
       {Attribute::u64(AttrKey::Axis, 1U), Attribute::u64(AttrKey::Start, 4U)}},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({3U, 8U}, 13U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({3U, 3U}, 17U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

Case broadcast_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {1U, 4U}},
                       {2U, DType::F32, TensorRole::Internal, {2U, 3U, 4U}},
                       {3U, DType::F32, TensorRole::Input, {2U, 3U, 4U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {{1U, Opcode::BroadcastTo, {1U}, {2U}, {}},
                          {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({1U, 4U}, 19U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({2U, 3U, 4U}, 23U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

// Reshape, permute and concat in one program: their gradients are shape moves
// and a fan-in, which is exactly where an index slip hides.
Case shape_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {2U, 3U, 4U}},
                       {2U, DType::F32, TensorRole::Internal, {2U, 4U, 3U}},
                       {3U, DType::F32, TensorRole::Internal, {2U, 12U}},
                       {4U, DType::F32, TensorRole::Input, {2U, 5U}},
                       {5U, DType::F32, TensorRole::Internal, {2U, 17U}},
                       {6U, DType::F32, TensorRole::Input, {2U, 17U}},
                       {7U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Permute, {1U}, {2U},
       {Attribute::u64(AttrKey::Permutation0, 0U),
        Attribute::u64(AttrKey::Permutation1, 2U),
        Attribute::u64(AttrKey::Permutation2, 1U)}},
      {2U, Opcode::Reshape, {2U}, {3U}, {}},
      {3U, Opcode::Concat, {3U, 4U}, {5U},
       {Attribute::u64(AttrKey::Axis, 1U)}},
      {4U, Opcode::MseLoss, {5U, 6U}, {7U}, {}}};
  c.inputs.emplace(1U, f32_tensor({2U, 3U, 4U}, 29U, 1.0F));
  c.inputs.emplace(4U, f32_tensor({2U, 5U}, 31U, 1.0F));
  c.inputs.emplace(6U, f32_tensor({2U, 17U}, 37U, 1.0F));
  c.loss = 7U;
  c.targets = {1U, 4U};
  return c;
}

Case group_norm_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {2U, 4U, 3U, 3U}},
                       {2U, DType::F32, TensorRole::Input, {4U}},
                       {3U, DType::F32, TensorRole::Input, {4U}},
                       {4U, DType::F32, TensorRole::Internal, {2U, 4U, 3U, 3U}},
                       {5U, DType::F32, TensorRole::Input, {2U, 4U, 3U, 3U}},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::GroupNorm, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::Groups, 2U),
        Attribute::f64(AttrKey::Epsilon, 1.0e-5)}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor({2U, 4U, 3U, 3U}, 41U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({4U}, 43U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({4U}, 47U, 0.5F));
  c.inputs.emplace(5U, f32_tensor({2U, 4U, 3U, 3U}, 53U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U, 3U};
  return c;
}

// RMS norm with shared-vector modulation, the form the Krea 2 blocks use.
// The vector is broadcast across the rows of its group AND feeds both the
// scale and the shift, so its gradient is a sum over the group of (normalized
// + 1). Dropping either term, or confusing the vector's per-group reduction
// with delta's reduction over everything, fails here.
Case shared_vector_modulate_case(std::uint64_t vectors) {
  Case c;
  const std::uint64_t rows = 4U;
  const std::uint64_t columns = 5U;
  const std::vector<std::uint64_t> x_shape{rows, columns};
  const std::vector<std::uint64_t> vector_shape{vectors, columns};
  const std::vector<std::uint64_t> delta_shape{2U, columns};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Input, {columns}},
                       {3U, DType::F32, TensorRole::Input, vector_shape},
                       {4U, DType::F32, TensorRole::Input, delta_shape},
                       {5U, DType::F32, TensorRole::Internal, x_shape},
                       {6U, DType::F32, TensorRole::Input, x_shape},
                       {7U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::RmsNormModulate, {1U, 2U, 3U, 4U}, {5U},
       {Attribute::u64(AttrKey::ModulationLayout,
                       static_cast<std::uint64_t>(
                           dif::ir::ModulationLayout::SharedVectorDelta)),
        Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        // A nonzero weight offset, so a gradient that forgets it fails.
        Attribute::f64(AttrKey::WeightOffset, 1.0)}},
      {2U, Opcode::MseLoss, {5U, 6U}, {7U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 433U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({columns}, 439U, 0.5F));
  c.inputs.emplace(3U, f32_tensor(vector_shape, 443U, 0.5F));
  c.inputs.emplace(4U, f32_tensor(delta_shape, 449U, 0.5F));
  c.inputs.emplace(6U, f32_tensor(x_shape, 457U, 1.0F));
  c.loss = 7U;
  c.targets = {1U, 2U, 3U, 4U};
  return c;
}

// The rotary embedding, in both pairing conventions and with a rotated range
// shorter than the head so the untouched tail is exercised too. F32 tables
// under BF16 activations is the shape the frontends actually use, but a
// difference quotient in BF16 is noise, so the dtypes here are F32 and the
// mixed-storage case is checked for backend agreement instead.
Case rotary_apply_case(bool half_split, std::uint64_t pairs) {
  Case c;
  const std::uint64_t batch = 2U;
  const std::uint64_t sequence = 3U;
  const std::uint64_t heads = 2U;
  const std::uint64_t dim = 8U;
  const std::vector<std::uint64_t> x_shape{batch, sequence, heads, dim};
  const std::vector<std::uint64_t> table{batch, sequence, pairs};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Input, table},
                       {3U, DType::F32, TensorRole::Input, table},
                       {4U, DType::F32, TensorRole::Internal, x_shape},
                       {5U, DType::F32, TensorRole::Input, x_shape},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::RotaryApply, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::RotaryLayout,
                       static_cast<std::uint64_t>(
                           half_split ? dif::ir::RotaryLayout::HalfSplit
                                      : dif::ir::RotaryLayout::Interleaved))}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 409U, 1.0F));
  c.inputs.emplace(2U, f32_tensor(table, 419U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(table, 421U, 1.0F));
  c.inputs.emplace(5U, f32_tensor(x_shape, 431U, 1.0F));
  c.loss = 6U;
  c.targets = {1U};
  return c;
}

// The index and permutation operations a denoiser carries. Each produces
// several outputs, so each also exercises the reverse sweep collecting a
// gradient per output: the outputs are summed elementwise before the loss, so
// every one of them carries a gradient back.

// Chunked row gather: index 2 is chosen three times and index 0 twice, so the
// source rows they name accumulate several contributions; row 3 is never
// chosen and must come back exactly zero.
Case select_row_chunks_case(std::size_t chunks) {
  Case c;
  const std::uint64_t source_rows = 5U;
  const std::uint64_t width = 3U;
  const std::vector<std::int32_t> indices{2, 0, 2, 4, 0, 2};
  const auto rows = static_cast<std::uint64_t>(indices.size());
  const std::vector<std::uint64_t> values{source_rows, chunks * width};
  const std::vector<std::uint64_t> chunk_shape{rows, width};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, values},
                       {2U, DType::I32, TensorRole::Input, {rows}}};
  std::vector<std::uint32_t> outputs;
  std::uint32_t next = 3U;
  for (std::size_t chunk = 0U; chunk < chunks; ++chunk) {
    c.program.tensors.push_back(
        {next, DType::F32, TensorRole::Internal, chunk_shape});
    outputs.push_back(next++);
  }
  c.program.operations = {
      {1U, Opcode::SelectRowChunks, {1U, 2U}, outputs, {}}};
  std::uint32_t operation = 2U;
  auto total = outputs[0];
  for (std::size_t chunk = 1U; chunk < chunks; ++chunk) {
    c.program.tensors.push_back(
        {next, DType::F32, TensorRole::Internal, chunk_shape});
    c.program.operations.push_back(
        {operation++, Opcode::Add, {total, outputs[chunk]}, {next}, {}});
    total = next++;
  }
  const auto target = next++;
  const auto loss = next++;
  c.program.tensors.push_back(
      {target, DType::F32, TensorRole::Input, chunk_shape});
  c.program.tensors.push_back({loss, DType::F32, TensorRole::Output, {1U}});
  c.program.operations.push_back(
      {operation++, Opcode::MseLoss, {total, target}, {loss}, {}});
  c.inputs.emplace(1U, f32_tensor(values, 337U, 1.0F));
  c.inputs.emplace(2U, i32_tensor({rows}, indices));
  c.inputs.emplace(target, f32_tensor(chunk_shape, 347U, 1.0F));
  c.loss = loss;
  c.targets = {1U};
  return c;
}

// Mapped row update: a map of -1 keeps the base row, so the base gradient
// passes through exactly there and is zero everywhere else, while update row
// 1 is named twice and must sum.
Case indexed_update_rows_case() {
  Case c;
  const std::uint64_t width = 4U;
  const std::vector<std::int32_t> map{-1, 1, 0, 1, -1};
  const auto rows = static_cast<std::uint64_t>(map.size());
  const std::uint64_t update_rows = 3U;
  const std::vector<std::uint64_t> base{rows, width};
  const std::vector<std::uint64_t> updates{update_rows, width};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, base},
                       {2U, DType::F32, TensorRole::Input, updates},
                       {3U, DType::I32, TensorRole::Input, {rows}},
                       {4U, DType::F32, TensorRole::Internal, base},
                       {5U, DType::F32, TensorRole::Input, base},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::IndexedUpdateRows, {1U, 2U, 3U}, {4U}, {}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor(base, 349U, 1.0F));
  c.inputs.emplace(2U, f32_tensor(updates, 353U, 1.0F));
  c.inputs.emplace(3U, i32_tensor({rows}, map));
  c.inputs.emplace(5U, f32_tensor(base, 359U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U};
  return c;
}

// Packed QKV weight split: a pure permutation, so the gradient is its
// inverse. Nothing sums, and nothing may be dropped.
//
// The three components MUST reach the loss asymmetrically. Summing them and
// taking one loss gives all three the same upstream gradient, and a backward
// that sends K's gradient to V's slot then produces exactly the right answer
// -- verified: that injected swap passed a symmetric version of this case.
// Three separate losses against three different targets make the components
// distinguishable, and the same injection then fails.
Case deinterleave_qkv_case() {
  Case c;
  const std::uint64_t heads = 2U;
  const std::uint64_t head_dim = 3U;
  const std::uint64_t hidden = 4U;
  const auto n = heads * head_dim;
  const std::vector<std::uint64_t> packed{3U * n, hidden};
  const std::vector<std::uint64_t> part{n, hidden};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, packed},
                       {2U, DType::F32, TensorRole::Internal, part},
                       {3U, DType::F32, TensorRole::Internal, part},
                       {4U, DType::F32, TensorRole::Internal, part},
                       {5U, DType::F32, TensorRole::Input, part},
                       {6U, DType::F32, TensorRole::Input, part},
                       {7U, DType::F32, TensorRole::Input, part},
                       {8U, DType::F32, TensorRole::Internal, {1U}},
                       {9U, DType::F32, TensorRole::Internal, {1U}},
                       {10U, DType::F32, TensorRole::Internal, {1U}},
                       {11U, DType::F32, TensorRole::Internal, {1U}},
                       {12U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::H3DeinterleaveQkvWeight, {1U}, {2U, 3U, 4U},
       {Attribute::u64(AttrKey::Heads, heads),
        Attribute::u64(AttrKey::HeadDim, head_dim)}},
      {2U, Opcode::MseLoss, {2U, 5U}, {8U}, {}},
      {3U, Opcode::MseLoss, {3U, 6U}, {9U}, {}},
      {4U, Opcode::MseLoss, {4U, 7U}, {10U}, {}},
      {5U, Opcode::Add, {8U, 9U}, {11U}, {}},
      {6U, Opcode::Add, {11U, 10U}, {12U}, {}}};
  c.inputs.emplace(1U, f32_tensor(packed, 367U, 1.0F));
  c.inputs.emplace(5U, f32_tensor(part, 373U, 1.0F));
  c.inputs.emplace(6U, f32_tensor(part, 379U, 1.5F));
  c.inputs.emplace(7U, f32_tensor(part, 389U, 0.5F));
  c.loss = 12U;
  c.targets = {1U};
  return c;
}

// adaLN modulation selection: six chunks per token, with two tokens naming
// the same table row so that row's gradient is a sum.
Case adaln_select_case() {
  Case c;
  const std::uint64_t table = 2U;
  const std::uint64_t hidden = 3U;
  const std::vector<std::int32_t> indices{0, 5, 0, 3};
  const auto rows = static_cast<std::uint64_t>(indices.size());
  const std::vector<std::uint64_t> projected{table, 18U * hidden};
  const std::vector<std::uint64_t> chunk{rows, hidden};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, projected},
                       {2U, DType::I32, TensorRole::Input, {rows}}};
  std::vector<std::uint32_t> outputs;
  std::uint32_t next = 3U;
  for (std::size_t index = 0U; index < 6U; ++index) {
    c.program.tensors.push_back(
        {next, DType::F32, TensorRole::Internal, chunk});
    outputs.push_back(next++);
  }
  c.program.operations = {{1U, Opcode::H3AdaLNSelect, {1U, 2U}, outputs, {}}};
  std::uint32_t operation = 2U;
  auto total = outputs[0];
  for (std::size_t index = 1U; index < 6U; ++index) {
    c.program.tensors.push_back(
        {next, DType::F32, TensorRole::Internal, chunk});
    c.program.operations.push_back(
        {operation++, Opcode::Add, {total, outputs[index]}, {next}, {}});
    total = next++;
  }
  const auto target = next++;
  const auto loss = next++;
  c.program.tensors.push_back({target, DType::F32, TensorRole::Input, chunk});
  c.program.tensors.push_back({loss, DType::F32, TensorRole::Output, {1U}});
  c.program.operations.push_back(
      {operation++, Opcode::MseLoss, {total, target}, {loss}, {}});
  c.inputs.emplace(1U, f32_tensor(projected, 397U, 1.0F));
  c.inputs.emplace(2U, i32_tensor({rows}, indices));
  c.inputs.emplace(target, f32_tensor(chunk, 401U, 1.0F));
  c.loss = loss;
  c.targets = {1U};
  return c;
}

// Reflect padding. The elements just inside each edge are read more than
// once, so their gradient is a SUM -- and the corners are read by the
// reflections of two or three axes at once, which is where an inverse mapping
// that handles each axis but not their product goes wrong. Extents are
// asymmetric so no symmetry can hide a swapped edge.
Case pad_reflect_case(bool volumetric) {
  Case c;
  const std::uint64_t channels = 2U;
  const std::uint64_t depth = 4U;
  const std::uint64_t height = 5U;
  const std::uint64_t width = 6U;
  const std::uint64_t front = 2U, back = 1U, top = 1U, bottom = 3U;
  const std::uint64_t west = 2U, east = 2U;
  std::vector<std::uint64_t> x_shape{1U, channels};
  std::vector<std::uint64_t> y_shape{1U, channels};
  if (volumetric) {
    x_shape.push_back(depth);
    y_shape.push_back(depth + front + back);
  }
  x_shape.push_back(height);
  x_shape.push_back(width);
  y_shape.push_back(height + top + bottom);
  y_shape.push_back(width + west + east);
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Internal, y_shape},
                       {3U, DType::F32, TensorRole::Input, y_shape},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<Attribute> attributes{Attribute::u64(AttrKey::PadTop, top),
                                    Attribute::u64(AttrKey::PadBottom, bottom),
                                    Attribute::u64(AttrKey::PadWest, west),
                                    Attribute::u64(AttrKey::PadEast, east)};
  if (volumetric) {
    attributes.push_back(Attribute::u64(AttrKey::PadFront, front));
    attributes.push_back(Attribute::u64(AttrKey::PadBack, back));
  }
  c.program.operations = {
      {1U, Opcode::PadReflect, {1U}, {2U}, std::move(attributes)},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 317U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(y_shape, 331U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

// RMS normalization across a channel axis. The fiber runs along a strided
// axis rather than the contiguous last one, so an index that confuses the
// channel stride with the trailing one is caught here and nowhere else.
Case channel_rms_norm_case(std::uint64_t axis, bool train_gamma) {
  Case c;
  const std::vector<std::uint64_t> shape{2U, 3U, 4U};
  const auto channels = shape[axis];
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, shape},
                       {2U, DType::F32, TensorRole::Input, {channels}},
                       {3U, DType::F32, TensorRole::Internal, shape},
                       {4U, DType::F32, TensorRole::Input, shape},
                       {5U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::ChannelRmsNorm, {1U, 2U}, {3U},
       {Attribute::u64(AttrKey::Axis, axis),
        Attribute::f64(AttrKey::Epsilon, 1.0e-12)}},
      {2U, Opcode::MseLoss, {3U, 4U}, {5U}, {}}};
  c.inputs.emplace(1U, f32_tensor(shape, 307U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({channels}, 311U, 1.0F));
  c.inputs.emplace(4U, f32_tensor(shape, 313U, 1.0F));
  c.loss = 5U;
  c.targets = train_gamma ? std::vector<std::uint32_t>{1U, 2U}
                          : std::vector<std::uint32_t>{1U};
  return c;
}

// Constant padding: the gradient is the crop back to the input's own region.
// Asymmetric amounts on every axis, so a rule that confuses the low pad with
// the high one, or crops the wrong axis, cannot pass by symmetry.
Case pad_constant_case(bool volumetric) {
  Case c;
  const std::uint64_t channels = 2U;
  const std::uint64_t depth = 3U;
  const std::uint64_t height = 4U;
  const std::uint64_t width = 5U;
  const std::uint64_t front = 1U, back = 2U, top = 2U, bottom = 1U;
  const std::uint64_t west = 1U, east = 3U;
  std::vector<std::uint64_t> x_shape{1U, channels};
  std::vector<std::uint64_t> y_shape{1U, channels};
  if (volumetric) {
    x_shape.push_back(depth);
    y_shape.push_back(depth + front + back);
  }
  x_shape.push_back(height);
  x_shape.push_back(width);
  y_shape.push_back(height + top + bottom);
  y_shape.push_back(width + west + east);
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Internal, y_shape},
                       {3U, DType::F32, TensorRole::Input, y_shape},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<Attribute> attributes{Attribute::u64(AttrKey::PadTop, top),
                                    Attribute::u64(AttrKey::PadBottom, bottom),
                                    Attribute::u64(AttrKey::PadWest, west),
                                    Attribute::u64(AttrKey::PadEast, east),
                                    Attribute::f64(AttrKey::Value, 0.25)};
  if (volumetric) {
    attributes.push_back(Attribute::u64(AttrKey::PadFront, front));
    attributes.push_back(Attribute::u64(AttrKey::PadBack, back));
  }
  c.program.operations = {
      {1U, Opcode::PadConstant, {1U}, {2U}, std::move(attributes)},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 283U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(y_shape, 293U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

// A 3-D convolution, in the geometries a video model uses: a temporal axis
// that may be strided or padded differently from the spatial ones, and
// grouped channels. The depthwise case (groups == channels) is the one a
// naive weight gradient gets wrong, because every group has exactly one
// input channel and the group arithmetic degenerates.
Case conv3d_case(std::uint64_t stride_t, std::uint64_t stride_hw,
                 std::uint64_t pad_t, std::uint64_t pad_hw,
                 std::uint64_t groups, std::uint64_t kernel_t) {
  Case c;
  const std::uint64_t in_channels = 4U;
  const std::uint64_t out_channels = 4U;
  const std::uint64_t depth = 5U;
  const std::uint64_t height = 4U;
  const std::uint64_t width = 4U;
  const std::uint64_t kernel = 3U;
  const auto out_d = (depth + 2U * pad_t - kernel_t) / stride_t + 1U;
  const auto out_h = (height + 2U * pad_hw - kernel) / stride_hw + 1U;
  const auto out_w = (width + 2U * pad_hw - kernel) / stride_hw + 1U;
  const std::vector<std::uint64_t> x_shape{1U, in_channels, depth, height,
                                           width};
  const std::vector<std::uint64_t> w_shape{
      out_channels, in_channels / groups, kernel_t, kernel, kernel};
  const std::vector<std::uint64_t> y_shape{1U, out_channels, out_d, out_h,
                                           out_w};
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Input, w_shape},
                       {3U, DType::F32, TensorRole::Input, {out_channels}},
                       {4U, DType::F32, TensorRole::Internal, y_shape},
                       {5U, DType::F32, TensorRole::Input, y_shape},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Conv3d, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::StrideT, stride_t),
        Attribute::u64(AttrKey::StrideH, stride_hw),
        Attribute::u64(AttrKey::StrideW, stride_hw),
        Attribute::u64(AttrKey::DilationT, 1U),
        Attribute::u64(AttrKey::DilationH, 1U),
        Attribute::u64(AttrKey::DilationW, 1U),
        Attribute::u64(AttrKey::PadFront, pad_t),
        Attribute::u64(AttrKey::PadBack, pad_t),
        Attribute::u64(AttrKey::PadTop, pad_hw),
        Attribute::u64(AttrKey::PadBottom, pad_hw),
        Attribute::u64(AttrKey::PadWest, pad_hw),
        Attribute::u64(AttrKey::PadEast, pad_hw),
        Attribute::u64(AttrKey::Groups, groups)}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 269U, 1.0F));
  c.inputs.emplace(2U, f32_tensor(w_shape, 271U, 0.5F));
  c.inputs.emplace(3U, f32_tensor({out_channels}, 277U, 0.3F));
  c.inputs.emplace(5U, f32_tensor(y_shape, 281U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U, 3U};
  // Measured over all seven geometries: the difference quotient reached 2.00
  // of the unscaled budget and the two backends differed by 3.7e-4 of the
  // range. Both bars are set with headroom over that, and both still catch a
  // one-percent weight-gradient error by a wide margin.
  c.gradient_budget_scale = 4.0;
  c.parity_bar_scale = 40.0;
  return c;
}

// Fused per-head RMS norm plus partial rotary, in the shapes a real
// transformer uses: interleaved adjacent pairs rather than a half split, a
// rotation table longer than the rows being rotated with a start offset into
// it, and the batched [B,S,H,D] form. Each of those was a separate way for
// the backward to index the wrong table entry.
Case qk_norm_rope_case(bool interleaved, std::uint64_t table_start,
                       bool batched) {
  Case c;
  const std::uint64_t batch = batched ? 2U : 1U;
  const std::uint64_t sequence = 3U;
  const std::uint64_t heads = 2U;
  const std::uint64_t dim = 4U;
  const std::uint64_t table_width = interleaved ? dim / 2U : dim;
  const std::uint64_t table_sequence = sequence + table_start;
  auto shape = [&](std::vector<std::uint64_t> tail) {
    std::vector<std::uint64_t> dims;
    if (batched)
      dims.push_back(batch);
    for (const auto value : tail)
      dims.push_back(value);
    return dims;
  };
  const auto x_shape = shape({sequence, heads, dim});
  const auto table_shape = shape({table_sequence, table_width});
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                       {2U, DType::F32, TensorRole::Input, {dim}},
                       {3U, DType::F32, TensorRole::Input, table_shape},
                       {4U, DType::F32, TensorRole::Input, table_shape},
                       {5U, DType::F32, TensorRole::Internal, x_shape},
                       {6U, DType::F32, TensorRole::Input, x_shape},
                       {7U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<Attribute> attributes{
      Attribute::f64(AttrKey::Epsilon, 1.0e-6),
      Attribute::u64(AttrKey::RotaryDim, dim),
      Attribute::u64(AttrKey::Start, table_start)};
  if (interleaved) {
    attributes.push_back(Attribute::u64(AttrKey::Implementation, 2U));
    attributes.push_back(Attribute::u64(
        AttrKey::RotaryLayout,
        static_cast<std::uint64_t>(dif::ir::RotaryLayout::Interleaved)));
  }
  c.program.operations = {
      {1U, Opcode::QkNormPartialRope, {1U, 2U, 3U, 4U}, {5U}, attributes},
      {2U, Opcode::MseLoss, {5U, 6U}, {7U}, {}}};
  c.inputs.emplace(1U, f32_tensor(x_shape, 211U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({dim}, 223U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(table_shape, 227U, 1.0F));
  c.inputs.emplace(4U, f32_tensor(table_shape, 229U, 1.0F));
  c.inputs.emplace(6U, f32_tensor(x_shape, 233U, 1.0F));
  c.loss = 7U;
  c.targets = {1U, 2U};
  return c;
}

// The gated residual add, with the gate governing several rows at once --
// how a DiT block gates every token with one per-sample value. The gate
// gradient is a sum over the rows it governs; the square case is the
// degenerate one where that sum has a single term.
Case residual_gate_case(std::uint64_t gate_rows) {
  Case c;
  const std::uint64_t rows = 4U;
  const std::uint64_t columns = 5U;
  c.program.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, columns}},
      {2U, DType::F32, TensorRole::Input, {rows, columns}},
      {3U, DType::F32, TensorRole::Input, {gate_rows, columns}},
      {4U, DType::F32, TensorRole::Internal, {rows, columns}},
      {5U, DType::F32, TensorRole::Input, {rows, columns}},
      {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::ResidualGate, {1U, 2U, 3U}, {4U}, {}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor({rows, columns}, 191U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({rows, columns}, 193U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({gate_rows, columns}, 197U, 1.0F));
  c.inputs.emplace(5U, f32_tensor({rows, columns}, 199U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U, 3U};
  return c;
}

// Layer normalization with an adaptive scale and shift -- the modulation
// every DiT block applies. The modulation rows are BROADCAST: two rows share
// each scale/shift row here, so the modulation gradients are sums over a
// group and the affine gradients are sums over everything. A rule that
// confuses those two reductions passes on a square case and fails here.
Case layer_norm_modulate_case(std::uint64_t modulation_rows) {
  Case c;
  const std::uint64_t rows = 4U;
  const std::uint64_t columns = 5U;
  c.program.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, columns}},
      {2U, DType::F32, TensorRole::Input, {columns}},
      {3U, DType::F32, TensorRole::Input, {columns}},
      {4U, DType::F32, TensorRole::Input, {modulation_rows, columns}},
      {5U, DType::F32, TensorRole::Input, {modulation_rows, columns}},
      {6U, DType::F32, TensorRole::Internal, {rows, columns}},
      {7U, DType::F32, TensorRole::Input, {rows, columns}},
      {8U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::LayerNormModulate, {1U, 2U, 3U, 4U, 5U}, {6U},
       {Attribute::f64(AttrKey::Epsilon, 1.0e-5)}},
      {2U, Opcode::MseLoss, {6U, 7U}, {8U}, {}}};
  c.inputs.emplace(1U, f32_tensor({rows, columns}, 157U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({columns}, 163U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({columns}, 167U, 0.5F));
  c.inputs.emplace(4U, f32_tensor({modulation_rows, columns}, 173U, 0.5F));
  c.inputs.emplace(5U, f32_tensor({modulation_rows, columns}, 179U, 0.5F));
  c.inputs.emplace(7U, f32_tensor({rows, columns}, 181U, 1.0F));
  c.loss = 8U;
  c.targets = {1U, 2U, 3U, 4U, 5U};
  return c;
}

// An embedding lookup. Index 2 is chosen three times and index 0 twice, so
// the gradient of those table rows is a SUM of several contributions -- the
// behaviour that makes a token embedding trainable, and the one a
// scatter-add gets wrong if it drops or double-counts a hit. Row 3 is never
// chosen, so its gradient must come out exactly zero.
Case gather_case() {
  Case c;
  const std::uint64_t table_rows = 5U;
  const std::uint64_t width = 4U;
  const std::vector<std::int32_t> indices{2, 0, 2, 4, 0, 2};
  const auto gathered = static_cast<std::uint64_t>(indices.size());
  c.program.tensors = {
      {1U, DType::F32, TensorRole::Input, {table_rows, width}},
      {2U, DType::I32, TensorRole::Input, {gathered}},
      {3U, DType::F32, TensorRole::Internal, {gathered, width}},
      {4U, DType::F32, TensorRole::Input, {gathered, width}},
      {5U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {{1U, Opcode::GatherRows, {1U, 2U}, {3U}, {}},
                          {2U, Opcode::MseLoss, {3U, 4U}, {5U}, {}}};
  c.inputs.emplace(1U, f32_tensor({table_rows, width}, 149U, 1.0F));
  c.inputs.emplace(2U, i32_tensor({gathered}, indices));
  c.inputs.emplace(4U, f32_tensor({gathered, width}, 151U, 1.0F));
  c.loss = 5U;
  c.targets = {1U};
  return c;
}

// The elementwise and affine rules the VAE and text frontends need. Clamp is
// checked with the sampled values straddling both bounds, so the saturated
// region is actually exercised rather than assumed.
Case clamp_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {2U, DType::F32, TensorRole::Internal, {4U, 5U}},
                       {3U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Clamp, {1U}, {2U},
       {Attribute::f64(AttrKey::Lower, -0.5), Attribute::f64(AttrKey::Upper, 0.5)}},
      {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({4U, 5U}, 101U, 1.5F));
  c.inputs.emplace(3U, f32_tensor({4U, 5U}, 103U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

Case sigmoid_case() {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {2U, DType::F32, TensorRole::Internal, {4U, 5U}},
                       {3U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {4U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {{1U, Opcode::Sigmoid, {1U}, {2U}, {}},
                          {2U, Opcode::MseLoss, {2U, 3U}, {4U}, {}}};
  c.inputs.emplace(1U, f32_tensor({4U, 5U}, 107U, 2.0F));
  c.inputs.emplace(3U, f32_tensor({4U, 5U}, 109U, 1.0F));
  c.loss = 4U;
  c.targets = {1U};
  return c;
}

// The affine rule adds no opcode: it composes from AffineLastDim, Multiply
// and BiasBackward. That makes it more worth checking, not less -- a
// composition can be wrong in ways a fused kernel cannot.
Case affine_case(bool with_bias) {
  Case c;
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {2U, DType::F32, TensorRole::Input, {5U}},
                       {3U, DType::F32, TensorRole::Input, {5U}},
                       {4U, DType::F32, TensorRole::Internal, {4U, 5U}},
                       {5U, DType::F32, TensorRole::Input, {4U, 5U}},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<std::uint32_t> inputs{1U, 2U};
  if (with_bias)
    inputs.push_back(3U);
  c.program.operations = {
      {1U, Opcode::AffineLastDim, inputs, {4U}, {}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor({4U, 5U}, 113U, 1.0F));
  c.inputs.emplace(2U, f32_tensor({5U}, 127U, 1.0F));
  c.inputs.emplace(3U, f32_tensor({5U}, 131U, 0.5F));
  c.inputs.emplace(5U, f32_tensor({4U, 5U}, 137U, 1.0F));
  c.loss = 6U;
  c.targets = with_bias ? std::vector<std::uint32_t>{1U, 2U, 3U}
                        : std::vector<std::uint32_t>{1U, 2U};
  return c;
}

// Attention over every geometry a real model carries: unbatched [S,H,D] and
// batched [B,S,H,D], square self attention and cross attention whose keys
// carry their own row count, plain and grouped-query.  The backward kernel
// guards the query and key geometries independently, so each combination
// exercises a different thread-to-element mapping.
Case attention_case(bool batched, std::uint64_t kv_rows,
                    std::uint64_t kv_heads, bool causal) {
  Case c;
  const std::uint64_t batch = 2U;
  const std::uint64_t rows = 4U;
  const std::uint64_t heads = 2U;
  const std::uint64_t dim = 4U;
  auto shape = [&](std::uint64_t r, std::uint64_t h) {
    std::vector<std::uint64_t> dims;
    if (batched)
      dims.push_back(batch);
    dims.push_back(r);
    dims.push_back(h);
    dims.push_back(dim);
    return dims;
  };
  const auto q_shape = shape(rows, heads);
  const auto kv_shape = shape(kv_rows, kv_heads);
  c.program.tensors = {{1U, DType::F32, TensorRole::Input, q_shape},
                       {2U, DType::F32, TensorRole::Input, kv_shape},
                       {3U, DType::F32, TensorRole::Input, kv_shape},
                       {4U, DType::F32, TensorRole::Internal, q_shape},
                       {5U, DType::F32, TensorRole::Input, q_shape},
                       {6U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<Attribute> attributes;
  if (causal)
    attributes.push_back(Attribute::boolean(AttrKey::Causal, true));
  if (kv_heads != heads)
    attributes.push_back(Attribute::u64(AttrKey::KvHeads, kv_heads));
  c.program.operations = {
      {1U, Opcode::Attention, {1U, 2U, 3U}, {4U}, std::move(attributes)},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor(q_shape, 61U, 1.0F));
  c.inputs.emplace(2U, f32_tensor(kv_shape, 67U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(kv_shape, 71U, 1.0F));
  c.inputs.emplace(5U, f32_tensor(q_shape, 73U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U, 3U};
  return c;
}

Case conv2d_case(std::uint64_t stride, std::uint64_t pad,
                 std::uint64_t groups) {
  Case c;
  const std::uint64_t in_channels = 4U;
  const std::uint64_t out_channels = 4U;
  const std::uint64_t height = 5U;
  const std::uint64_t width = 5U;
  const std::uint64_t kernel = 3U;
  const auto out_h = (height + 2U * pad - kernel) / stride + 1U;
  const auto out_w = (width + 2U * pad - kernel) / stride + 1U;
  c.program.tensors = {
      {1U, DType::F32, TensorRole::Input, {1U, in_channels, height, width}},
      {2U, DType::F32, TensorRole::Input,
       {out_channels, in_channels / groups, kernel, kernel}},
      {3U, DType::F32, TensorRole::Input, {out_channels}},
      {4U, DType::F32, TensorRole::Internal, {1U, out_channels, out_h, out_w}},
      {5U, DType::F32, TensorRole::Input, {1U, out_channels, out_h, out_w}},
      {6U, DType::F32, TensorRole::Output, {1U}}};
  c.program.operations = {
      {1U, Opcode::Conv2d, {1U, 2U, 3U}, {4U},
       {Attribute::u64(AttrKey::StrideH, stride),
        Attribute::u64(AttrKey::StrideW, stride),
        Attribute::u64(AttrKey::DilationH, 1U),
        Attribute::u64(AttrKey::DilationW, 1U),
        Attribute::u64(AttrKey::PadTop, pad),
        Attribute::u64(AttrKey::PadBottom, pad),
        Attribute::u64(AttrKey::PadWest, pad),
        Attribute::u64(AttrKey::PadEast, pad),
        Attribute::u64(AttrKey::Groups, groups)}},
      {2U, Opcode::MseLoss, {4U, 5U}, {6U}, {}}};
  c.inputs.emplace(1U, f32_tensor({1U, in_channels, height, width}, 59U, 1.0F));
  c.inputs.emplace(
      2U, f32_tensor({out_channels, in_channels / groups, kernel, kernel}, 61U,
                     0.5F));
  c.inputs.emplace(3U, f32_tensor({out_channels}, 67U, 0.3F));
  c.inputs.emplace(5U, f32_tensor({1U, out_channels, out_h, out_w}, 71U, 1.0F));
  c.loss = 6U;
  c.targets = {1U, 2U, 3U};
  return c;
}

// The same differentiated program on both executors. The CPU reference is
// the oracle the finite differences validated; this says the CUDA kernels
// compute the same thing, which is the second half of trusting a gradient.
void check_backends(const std::string &label, const Case &c) {
  if (!dif::runtime::cuda_available())
    return;
  const auto differentiated =
      dif::training::differentiate(c.program, c.loss, c.targets);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference = dif::runtime::make_cpu_executor()->run(
      differentiated.program, c.inputs, options);
  const auto candidate = dif::runtime::make_cuda_executor()->run(
      differentiated.program, c.inputs, options);
  for (const auto target : c.targets) {
    const auto gradient = differentiated.gradients.at(target);
    const auto &expected = reference.outputs.at(gradient);
    const auto &actual = candidate.outputs.at(gradient);
    double worst = 0.0;
    double scale = 1.0e-6;
    for (std::uint64_t index = 0U; index < expected.element_count(); ++index)
      scale = std::max(
          scale, static_cast<double>(
                     std::abs(dif::runtime::load_float(expected, index))));
    for (std::uint64_t index = 0U; index < expected.element_count(); ++index)
      worst = std::max(
          worst,
          static_cast<double>(std::abs(dif::runtime::load_float(expected, index) -
                                       dif::runtime::load_float(actual, index))) /
              scale);
    // The bar follows the storage the gradient actually lives in. Two
    // executors running the same program in F32 should agree to F32 rounding;
    // in BF16 they cannot, because BF16 carries eight mantissa bits and the
    // two round their intermediates at different points. Holding a BF16
    // program to an F32 bar would mean either a permanently red test or a
    // silently skipped one.
    const double bar = expected.dtype == DType::F32   ? 2.0e-5
                       : expected.dtype == DType::F16 ? 2.0e-3
                                                      : 2.0e-2;
    expect(worst <= bar * c.parity_bar_scale,
           label + " tensor " + std::to_string(target) +
               ": CPU and CUDA differ by " + std::to_string(worst) +
               " of the range (bar " +
               std::to_string(bar * c.parity_bar_scale) + ")");
  }
}

// Runs a program AS WRITTEN on both backends and compares its outputs. Some
// backward kernels cannot be reached through their own forward in a test: the
// CUDA shared-vector rms_norm_modulate forward admits only the production
// 6144-wide BF16 reduction, so a differentiable F32 case cannot execute on
// CUDA at all. The gradient itself is still checked on the CPU by finite
// differences; this is what keeps the generated kernel honest alongside it.
void check_program_backends(const std::string &label, const Program &program,
                            const dif::runtime::TensorMap &inputs,
                            const std::vector<std::uint32_t> &outputs) {
  if (!dif::runtime::cuda_available())
    return;
  dif::ir::verify(program);
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  const auto reference =
      dif::runtime::make_cpu_executor()->run(program, inputs, options);
  const auto candidate =
      dif::runtime::make_cuda_executor()->run(program, inputs, options);
  for (const auto id : outputs) {
    const auto &expected = reference.outputs.at(id);
    const auto &actual = candidate.outputs.at(id);
    double worst = 0.0;
    double scale = 1.0e-6;
    for (std::uint64_t index = 0U; index < expected.element_count(); ++index)
      scale = std::max(scale,
                       static_cast<double>(std::abs(
                           dif::runtime::load_float(expected, index))));
    for (std::uint64_t index = 0U; index < expected.element_count(); ++index)
      worst = std::max(
          worst, static_cast<double>(
                     std::abs(dif::runtime::load_float(expected, index) -
                              dif::runtime::load_float(actual, index))) /
                     scale);
    expect(worst <= 2.0e-5,
           label + " output " + std::to_string(id) +
               ": CPU and CUDA differ by " + std::to_string(worst) +
               " of the range");
    std::cout << "  " << label << " output " << id << ": backends agree to "
              << worst << "\n";
  }
}

// A gradient whose forward cannot execute on CUDA at the test's shape: the
// finite-difference check still runs on the CPU, and the generated kernel is
// covered by check_program_backends instead.
void run_gradients_only(const std::string &label, const Case &c) {
  dif::ir::verify(c.program);
  check_gradients(label, c.program, c.inputs, c.loss, c.targets, 8.0e-3,
                  3.0e-5 * c.gradient_budget_scale,
                  3.0e-3 * c.gradient_budget_scale);
}

void run(const std::string &label, const Case &c) {
  dif::ir::verify(c.program);
  check_gradients(label, c.program, c.inputs, c.loss, c.targets, 8.0e-3,
                  3.0e-5 * c.gradient_budget_scale,
                  3.0e-3 * c.gradient_budget_scale);
  check_backends(label, c);
}

// Some shapes cannot be gradient-checked but must still agree between the
// backends. A half-precision program is one: a central difference in BF16 is
// noise, while the two executors running the SAME program still have to
// produce the same answer. This is how a kernel that reads an F32 table with
// a BF16 loader gets caught -- the CPU reference reads the table by its own
// dtype and the generated kernel did not.
void run_parity(const std::string &label, const Case &c) {
  dif::ir::verify(c.program);
  check_backends(label, c);
}

dif::runtime::Tensor storage_tensor(DType dtype,
                                    std::vector<std::uint64_t> dims,
                                    std::uint64_t seed, float amplitude) {
  const auto reference = f32_tensor(dims, seed, amplitude);
  std::uint64_t count = 1U;
  for (const auto dim : dims)
    count *= dim;
  dif::runtime::Tensor tensor{dtype, std::move(dims), {}};
  tensor.bytes.resize(static_cast<std::size_t>(count) *
                      dif::ir::dtype_size(dtype));
  tensor.validate();
  for (std::uint64_t index = 0U; index < reference.element_count(); ++index)
    dif::runtime::store_float(tensor, index,
                              dif::runtime::load_float(reference, index));
  return tensor;
}

// BF16 q/k with F32 rotation tables, through the generic path: the exact
// combination the frontends use and the one nothing numerical covered.
Case mixed_dtype_rope_case(bool interleaved) {
  Case c;
  const std::uint64_t sequence = 3U;
  const std::uint64_t heads = 2U;
  const std::uint64_t dim = 8U;
  const std::uint64_t table_width = interleaved ? dim / 2U : dim;
  const std::vector<std::uint64_t> x_shape{sequence, heads, dim};
  const std::vector<std::uint64_t> table_shape{sequence, table_width};
  c.program.tensors = {{1U, DType::BF16, TensorRole::Input, x_shape},
                       {2U, DType::BF16, TensorRole::Input, {dim}},
                       {3U, DType::F32, TensorRole::Input, table_shape},
                       {4U, DType::F32, TensorRole::Input, table_shape},
                       {5U, DType::BF16, TensorRole::Internal, x_shape},
                       {6U, DType::BF16, TensorRole::Input, x_shape},
                       {7U, DType::F32, TensorRole::Output, {1U}}};
  std::vector<Attribute> attributes{
      Attribute::f64(AttrKey::Epsilon, 1.0e-6),
      Attribute::u64(AttrKey::RotaryDim, dim)};
  if (interleaved) {
    attributes.push_back(Attribute::u64(AttrKey::Implementation, 2U));
    attributes.push_back(Attribute::u64(
        AttrKey::RotaryLayout,
        static_cast<std::uint64_t>(dif::ir::RotaryLayout::Interleaved)));
  }
  c.program.operations = {
      {1U, Opcode::QkNormPartialRope, {1U, 2U, 3U, 4U}, {5U}, attributes},
      {2U, Opcode::MseLoss, {5U, 6U}, {7U}, {}}};
  c.inputs.emplace(1U, storage_tensor(DType::BF16, x_shape, 239U, 1.0F));
  c.inputs.emplace(2U, storage_tensor(DType::BF16, {dim}, 241U, 1.0F));
  c.inputs.emplace(3U, f32_tensor(table_shape, 251U, 1.0F));
  c.inputs.emplace(4U, f32_tensor(table_shape, 257U, 1.0F));
  c.inputs.emplace(6U, storage_tensor(DType::BF16, x_shape, 263U, 1.0F));
  c.loss = 7U;
  c.targets = {1U, 2U};
  return c;
}

// The shared-vector backward on its own, run on both backends. Its forward
// cannot reach CUDA at this shape, but the backward operation has no such
// restriction, so the generated kernel is compared against the CPU reference
// directly rather than left unexercised.
void shared_vector_modulate_backend_check() {
  const std::uint64_t rows = 4U;
  const std::uint64_t columns = 6U;
  const std::uint64_t vectors = 2U;
  const std::vector<std::uint64_t> x_shape{rows, columns};
  const std::vector<std::uint64_t> vector_shape{vectors, columns};
  const std::vector<std::uint64_t> delta_shape{2U, columns};
  Program program;
  program.tensors = {{1U, DType::F32, TensorRole::Input, x_shape},
                     {2U, DType::F32, TensorRole::Input, x_shape},
                     {3U, DType::F32, TensorRole::Input, {columns}},
                     {4U, DType::F32, TensorRole::Input, vector_shape},
                     {5U, DType::F32, TensorRole::Input, delta_shape},
                     {6U, DType::F32, TensorRole::Output, x_shape},
                     {7U, DType::F32, TensorRole::Output, {columns}},
                     {8U, DType::F32, TensorRole::Output, vector_shape},
                     {9U, DType::F32, TensorRole::Output, delta_shape}};
  program.operations = {
      {1U, Opcode::RmsNormModulateBackward, {1U, 2U, 3U, 4U, 5U},
       {6U, 7U, 8U, 9U},
       {Attribute::u64(AttrKey::ModulationLayout,
                       static_cast<std::uint64_t>(
                           dif::ir::ModulationLayout::SharedVectorDelta)),
        Attribute::f64(AttrKey::Epsilon, 1.0e-5),
        Attribute::f64(AttrKey::WeightOffset, 1.0)}}};
  dif::runtime::TensorMap inputs;
  inputs.emplace(1U, f32_tensor(x_shape, 461U, 1.0F));
  inputs.emplace(2U, f32_tensor(x_shape, 463U, 1.0F));
  inputs.emplace(3U, f32_tensor({columns}, 467U, 0.5F));
  inputs.emplace(4U, f32_tensor(vector_shape, 479U, 0.5F));
  inputs.emplace(5U, f32_tensor(delta_shape, 487U, 0.5F));
  check_program_backends("shared vector modulate backward", program, inputs,
                         {6U, 7U, 8U, 9U});
}

} // namespace

int main() {
  run("gelu tanh", gelu_case(dif::ir::GeluApproximation::Tanh));
  run("gelu exact erf", gelu_case(dif::ir::GeluApproximation::ExactErf));
  run("gelu quick sigmoid",
      gelu_case(dif::ir::GeluApproximation::QuickSigmoid));
  run("upsample nearest 2d", upsample_case());
  run("slice", slice_case());
  run("broadcast to", broadcast_case());
  run("permute reshape concat", shape_case());
  run("group norm", group_norm_case());
  run("conv2d", conv2d_case(1U, 1U, 1U));
  run("conv2d strided", conv2d_case(2U, 1U, 1U));
  run("conv2d unpadded", conv2d_case(1U, 0U, 1U));
  run("conv2d grouped", conv2d_case(1U, 1U, 2U));
  run_gradients_only("shared vector modulate",
                     shared_vector_modulate_case(4U));
  run_gradients_only("shared vector modulate broadcast",
                     shared_vector_modulate_case(2U));
  shared_vector_modulate_backend_check();
  run("rotary apply half split", rotary_apply_case(true, 4U));
  run("rotary apply interleaved", rotary_apply_case(false, 4U));
  run("rotary apply partial", rotary_apply_case(false, 3U));
  run("rotary apply half split partial", rotary_apply_case(true, 3U));
  run("select row chunks", select_row_chunks_case(2U));
  run("select row chunks four", select_row_chunks_case(4U));
  run("indexed update rows", indexed_update_rows_case());
  run("deinterleave qkv weight", deinterleave_qkv_case());
  run("adaln select", adaln_select_case());
  run("pad reflect", pad_reflect_case(false));
  run("pad reflect volumetric", pad_reflect_case(true));
  run("channel rms norm", channel_rms_norm_case(1U, true));
  run("channel rms norm frozen gamma", channel_rms_norm_case(1U, false));
  run("channel rms norm last axis", channel_rms_norm_case(2U, true));
  run("pad constant", pad_constant_case(false));
  run("pad constant volumetric", pad_constant_case(true));
  run("conv3d", conv3d_case(1U, 1U, 1U, 1U, 1U, 3U));
  run("conv3d strided time", conv3d_case(2U, 1U, 1U, 1U, 1U, 3U));
  run("conv3d strided space", conv3d_case(1U, 2U, 1U, 1U, 1U, 3U));
  run("conv3d unpadded", conv3d_case(1U, 1U, 0U, 0U, 1U, 3U));
  run("conv3d grouped", conv3d_case(1U, 1U, 1U, 1U, 2U, 3U));
  run("conv3d depthwise", conv3d_case(1U, 1U, 1U, 1U, 4U, 3U));
  // A temporal kernel of one is the "2-D convolution applied per frame" shape
  // a video VAE uses for its spatial layers.
  run("conv3d flat time", conv3d_case(1U, 1U, 0U, 1U, 1U, 1U));
  run_parity("qk norm rope bf16 f32 tables", mixed_dtype_rope_case(false));
  run_parity("qk norm rope bf16 f32 tables interleaved",
             mixed_dtype_rope_case(true));
  run("qk norm rope", qk_norm_rope_case(false, 0U, false));
  run("qk norm rope interleaved", qk_norm_rope_case(true, 0U, false));
  run("qk norm rope table offset", qk_norm_rope_case(true, 2U, false));
  run("qk norm rope batched", qk_norm_rope_case(true, 2U, true));
  run("residual gate", residual_gate_case(4U));
  run("residual gate broadcast", residual_gate_case(1U));
  run("layer norm modulate", layer_norm_modulate_case(4U));
  run("layer norm modulate broadcast", layer_norm_modulate_case(2U));
  run("gather rows", gather_case());
  run("clamp", clamp_case());
  run("sigmoid", sigmoid_case());
  run("affine last dim", affine_case(true));
  run("affine last dim no bias", affine_case(false));
  run("attention", attention_case(false, 4U, 2U, false));
  run("attention causal", attention_case(false, 4U, 2U, true));
  run("attention gqa", attention_case(false, 4U, 1U, false));
  run("attention batched", attention_case(true, 4U, 2U, false));
  run("attention cross", attention_case(false, 6U, 2U, false));
  run("attention batched cross", attention_case(true, 6U, 2U, false));
  run("attention batched cross gqa", attention_case(true, 6U, 1U, false));
  if (failures != 0) {
    std::cerr << failures << " gradient failure(s)\n";
    return 1;
  }
  std::cout << "training gradient tests passed\n";
  return 0;
}
