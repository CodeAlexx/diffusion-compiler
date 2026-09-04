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

dif::runtime::Tensor i32_tensor(std::vector<std::uint64_t> dims,
                                const std::vector<std::int32_t> &values) {
  dif::runtime::Tensor tensor{DType::I32, std::move(dims), {}};
  tensor.bytes.resize(values.size() * sizeof(std::int32_t));
  std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
  tensor.validate();
  return tensor;
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
    expect(worst <= 2.0e-5, label + " tensor " + std::to_string(target) +
                                ": CPU and CUDA differ by " +
                                std::to_string(worst) + " of the range");
  }
}

void run(const std::string &label, const Case &c) {
  dif::ir::verify(c.program);
  check_gradients(label, c.program, c.inputs, c.loss, c.targets);
  check_backends(label, c);
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
