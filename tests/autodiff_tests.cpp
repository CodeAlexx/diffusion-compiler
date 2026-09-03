// Deterministic gradient fan-in: a tensor consumed by several operations
// receives its gradient as a fold of every consumer's contribution. The fold
// order is sorted by consumer operation id, so two topological orderings of
// the same forward program produce bit-identical gradients on the CPU
// executor. The backward programs must also verify and expose the same
// gradient tensors.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/training/autodiff.hpp"

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

// x feeds three Linear layers of very different scales; their outputs are
// summed and compared with a target. `order` lists the three Linear
// operations in the position they take in the operation list (any order is
// topologically valid).
dif::ir::Program forward_program(const std::vector<std::uint32_t> &order) {
  using namespace dif::ir;
  constexpr std::uint64_t rows = 8U, width = 16U;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, width}},
      {2U, DType::BF16, TensorRole::Constant, {width, width}},
      {3U, DType::BF16, TensorRole::Constant, {width}},
      {4U, DType::BF16, TensorRole::Constant, {width, width}},
      {5U, DType::BF16, TensorRole::Constant, {width}},
      {6U, DType::BF16, TensorRole::Constant, {width, width}},
      {7U, DType::BF16, TensorRole::Constant, {width}},
      {8U, DType::BF16, TensorRole::Internal, {rows, width}},
      {9U, DType::BF16, TensorRole::Internal, {rows, width}},
      {10U, DType::BF16, TensorRole::Internal, {rows, width}},
      {11U, DType::BF16, TensorRole::Internal, {rows, width}},
      {12U, DType::BF16, TensorRole::Internal, {rows, width}},
      {13U, DType::BF16, TensorRole::Input, {rows, width}},
      {14U, DType::F32, TensorRole::Output, {1U}},
  };
  const Operation linears[3] = {
      {1U, Opcode::Linear, {1U, 2U, 3U}, {8U}, {}},
      {2U, Opcode::Linear, {1U, 4U, 5U}, {9U}, {}},
      {3U, Opcode::Linear, {1U, 6U, 7U}, {10U}, {}},
  };
  for (const auto index : order)
    program.operations.push_back(linears[index]);
  program.operations.push_back({4U, Opcode::Add, {8U, 9U}, {11U}, {}});
  program.operations.push_back({5U, Opcode::Add, {11U, 10U}, {12U}, {}});
  program.operations.push_back({6U, Opcode::MseLoss, {12U, 13U}, {14U}, {}});
  dif::ir::verify(program);
  return program;
}

dif::runtime::TensorMap inputs() {
  dif::runtime::TensorMap map;
  map.emplace(1U, random_bf16({8U, 16U}, 3U, 1.0f));
  map.emplace(2U, random_bf16({16U, 16U}, 5U, 1.0f));
  map.emplace(3U, random_bf16({16U}, 7U, 0.5f));
  map.emplace(4U, random_bf16({16U, 16U}, 11U, 0.001f));
  map.emplace(5U, random_bf16({16U}, 13U, 0.001f));
  map.emplace(6U, random_bf16({16U, 16U}, 17U, 30.0f));
  map.emplace(7U, random_bf16({16U}, 19U, 3.0f));
  map.emplace(13U, random_bf16({8U, 16U}, 23U, 1.0f));
  return map;
}

struct Run {
  std::vector<std::uint8_t> grad_x;
  std::size_t add_operations;
};

Run run(const std::vector<std::uint32_t> &order) {
  const auto program = forward_program(order);
  const std::vector<std::uint32_t> targets{1U};
  const auto differentiated =
      dif::training::differentiate(program, 14U, targets);
  dif::ir::verify(differentiated.program);
  std::size_t adds = 0U;
  for (const auto &operation : differentiated.program.operations)
    adds += operation.opcode == dif::ir::Opcode::Add ? 1U : 0U;
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  const auto result = dif::runtime::make_cpu_executor()->run(
      differentiated.program, inputs(), options);
  const auto gradient = differentiated.gradients.at(1U);
  return {result.outputs.at(gradient).bytes, adds};
}

} // namespace

int main() {
  const auto a = run({0U, 1U, 2U});
  const auto b = run({2U, 0U, 1U});
  const auto c = run({1U, 2U, 0U});
  expect(!a.grad_x.empty(), "gradient of the shared input is produced");
  expect(a.grad_x == b.grad_x,
         "gradient bit-identical across consumer orderings (0,1,2) vs (2,0,1)");
  expect(a.grad_x == c.grad_x,
         "gradient bit-identical across consumer orderings (0,1,2) vs (1,2,0)");
  // Three consumers of x plus two Add forwards: the backward folds x's three
  // contributions with exactly two Adds; the forward Adds contribute none.
  expect(a.add_operations == 2U + 2U,
         "backward emits one Add per extra contribution (" +
             std::to_string(a.add_operations) + ")");
  bool nonzero = false;
  for (const auto byte : a.grad_x)
    nonzero = nonzero || byte != 0U;
  expect(nonzero, "gradient is not identically zero");
  if (failures != 0) {
    std::cerr << failures << " autodiff test failure(s)\n";
    return 1;
  }
  std::cout << "autodiff tests passed\n";
  return 0;
}
