// Low-rank adaptation as a transform over an arbitrary program.
//
// The load-bearing property is that LoRA at initialisation is the IDENTITY:
// with the down-projection zeroed, the adapted program must produce exactly
// what the base program produced -- bit for bit, not close. That single check
// catches a mis-scaled delta, a wrong factorization order, and, most
// importantly, a consumer left reading the un-adapted value, because any of
// those would perturb an output that must not move.

#include "dif/ir/ir.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/lora.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"

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

using dif::ir::DType;
using dif::ir::Opcode;
using dif::ir::Program;
using dif::ir::TensorRole;
using dif::runtime::TensorMap;

dif::runtime::Tensor f32_tensor(std::vector<std::uint64_t> dims,
                                std::uint64_t seed, float amplitude = 1.0F) {
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

dif::runtime::Tensor zeros_like(const dif::ir::TensorDesc &desc) {
  dif::runtime::Tensor tensor{desc.dtype, desc.dims, {}};
  tensor.bytes.assign(static_cast<std::size_t>(desc.byte_count()), 0);
  tensor.validate();
  return tensor;
}

// Two Linears with something between them, and a THIRD operation reading the
// first Linear's output. That third reader is the rewiring test: if the
// transform forgets it, the adapted value never reaches the loss.
struct Fixture {
  Program program;
  std::uint32_t input{}, first_linear{}, second_linear{}, output{};
  TensorMap inputs;
};

Fixture make_fixture() {
  Fixture fixture;
  const std::uint64_t rows = 4U, width = 6U, hidden = 8U;
  fixture.program.tensors = {
      {1U, DType::F32, TensorRole::Input, {rows, width}},
      {2U, DType::F32, TensorRole::Constant, {hidden, width}},
      {3U, DType::F32, TensorRole::Constant, {width, hidden}},
      {4U, DType::F32, TensorRole::Internal, {rows, hidden}},
      {5U, DType::F32, TensorRole::Internal, {rows, hidden}},
      {6U, DType::F32, TensorRole::Internal, {rows, hidden}},
      {7U, DType::F32, TensorRole::Output, {rows, width}}};
  fixture.program.operations = {
      {1U, Opcode::Linear, {1U, 2U}, {4U}, {}},
      {2U, Opcode::SiLU, {4U}, {5U}, {}},
      // A SECOND reader of the first Linear's output. If the transform
      // rewires only the first reader, this one keeps the frozen value and a
      // zeroed adapter stops being the identity.
      {3U, Opcode::Add, {5U, 4U}, {6U}, {}},
      {4U, Opcode::Linear, {6U, 3U}, {7U}, {}}};
  dif::ir::verify(fixture.program);
  fixture.input = 1U;
  fixture.first_linear = 1U;
  fixture.second_linear = 4U;
  fixture.output = 7U;
  fixture.inputs.emplace(1U, f32_tensor({rows, width}, 3U));
  fixture.inputs.emplace(2U, f32_tensor({hidden, width}, 5U));
  fixture.inputs.emplace(3U, f32_tensor({width, hidden}, 7U));
  return fixture;
}

TensorMap run(const Program &program, const TensorMap &inputs) {
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  return dif::runtime::make_cpu_executor()->run(program, inputs, options).outputs;
}

dif::opt::LoraSpec spec_for(const Fixture &fixture) {
  dif::opt::LoraSpec spec;
  spec.rank = 2U;
  spec.alpha = 4.0;
  spec.operations = {fixture.first_linear, fixture.second_linear};
  return spec;
}

void at_initialisation_lora_is_the_identity() {
  const auto fixture = make_fixture();
  const auto adapted = dif::opt::insert_lora(fixture.program, spec_for(fixture));
  expect(adapted.sites.size() == 2U, "both named Linears were adapted");
  expect(adapted.parameters.size() == 4U, "two adapters per site");

  auto inputs = fixture.inputs;
  for (const auto &site : adapted.sites) {
    inputs.emplace(site.down, zeros_like(*adapted.program.tensor(site.down)));
    // The up-projection is deliberately NOT zero: a transform that happened
    // to zero the wrong factor would still pass if both were zero.
    inputs.emplace(site.up,
                   f32_tensor(adapted.program.tensor(site.up)->dims, 11U));
  }

  const auto base = run(fixture.program, fixture.inputs);
  const auto with_lora = run(adapted.program, inputs);
  // The output id is the same in both programs by construction: the adapted
  // value takes over the id the Linear used to produce.
  const auto expected = base.at(fixture.output);
  const auto actual = with_lora.at(fixture.output);
  expect(expected.bytes == actual.bytes,
         "a zeroed down-projection leaves the output BIT-identical");
}

void a_nonzero_adapter_actually_reaches_the_output() {
  const auto fixture = make_fixture();
  const auto adapted = dif::opt::insert_lora(fixture.program, spec_for(fixture));
  auto inputs = fixture.inputs;
  std::uint64_t seed = 13U;
  for (const auto &site : adapted.sites) {
    inputs.emplace(site.down,
                   f32_tensor(adapted.program.tensor(site.down)->dims, seed++));
    inputs.emplace(site.up,
                   f32_tensor(adapted.program.tensor(site.up)->dims, seed++));
  }
  const auto base = run(fixture.program, fixture.inputs);
  const auto with_lora = run(adapted.program, inputs);
  expect(base.at(fixture.output).bytes != with_lora.at(fixture.output).bytes,
         "a nonzero adapter changes the output");
}

// Adapt ONLY the first Linear, whose output has two readers. Both must see
// the ADAPTED value, and only the transform's own Add may see the frozen one.
// Because the adapted value keeps the original tensor id, this is a check
// that the frozen path was moved aside correctly rather than that consumers
// were rewritten.
void every_reader_of_an_adapted_output_sees_the_adapted_value() {
  const auto fixture = make_fixture();
  dif::opt::LoraSpec spec;
  spec.rank = 2U;
  spec.alpha = 4.0;
  spec.operations = {fixture.first_linear};
  const auto adapted = dif::opt::insert_lora(fixture.program, spec);
  const auto &site = adapted.sites.front();

  // Count the readers of the adapted value in the rewritten program. The
  // original had two readers of tensor 4; both must now read the combined
  // value, and none may still read the frozen one.
  std::size_t reads_combined = 0U, reads_base = 0U;
  for (const auto &operation : adapted.program.operations)
    for (const auto input : operation.inputs) {
      if (input == site.output)
        ++reads_combined;
      else if (input == site.base_output)
        ++reads_base;
    }
  // The Add the transform itself inserted reads the frozen value once, and
  // that is the only permitted reader.
  expect(reads_base == 1U,
         "only the transform's own Add still reads the frozen output, saw " +
             std::to_string(reads_base));
  expect(reads_combined == 2U,
         "both original readers now see the adapted value, saw " +
             std::to_string(reads_combined));

  auto inputs = fixture.inputs;
  inputs.emplace(site.down, zeros_like(*adapted.program.tensor(site.down)));
  inputs.emplace(site.up,
                 f32_tensor(adapted.program.tensor(site.up)->dims, 17U));
  expect(run(fixture.program, fixture.inputs).at(fixture.output).bytes ==
             run(adapted.program, inputs).at(fixture.output).bytes,
         "with one site adapted and zeroed, the output is still bit-identical");
}

void the_scale_is_alpha_over_rank() {
  const auto fixture = make_fixture();
  dif::opt::LoraSpec spec;
  spec.rank = 4U;
  spec.alpha = 8.0;
  spec.operations = {fixture.first_linear};
  const auto adapted = dif::opt::insert_lora(fixture.program, spec);
  bool found = false;
  for (const auto &operation : adapted.program.operations)
    if (operation.opcode == Opcode::Fill &&
        operation.f64(dif::ir::AttrKey::Value, 0.0) == 2.0)
      found = true;
  expect(found, "the in-graph scale is alpha/rank = 2");
}

void mixed_precision_adapters_cross_an_explicit_cast() {
  auto fixture = make_fixture();
  // A BF16 program with F32 adapters: the dtype boundary must be a Cast, not
  // a verifier failure.
  for (auto &tensor : fixture.program.tensors)
    tensor.dtype = DType::BF16;
  dif::ir::verify(fixture.program);
  dif::opt::LoraSpec spec;
  spec.rank = 2U;
  spec.alpha = 4.0;
  spec.parameter_dtype = DType::F32;
  spec.operations = {fixture.first_linear};
  const auto adapted = dif::opt::insert_lora(fixture.program, spec);
  std::size_t casts = 0U;
  for (const auto &operation : adapted.program.operations)
    if (operation.opcode == Opcode::Cast)
      ++casts;
  expect(casts == 2U, "both adapters cross a Cast, saw " + std::to_string(casts));
  for (const auto &site : adapted.sites) {
    expect(adapted.program.tensor(site.down)->dtype == DType::F32,
           "the stored adapter keeps its own dtype");
  }
}

void bad_requests_are_refused() {
  const auto fixture = make_fixture();
  const auto refused = [&](dif::opt::LoraSpec spec, const std::string &why) {
    bool threw = false;
    try {
      dif::opt::insert_lora(fixture.program, spec);
    } catch (const dif::Error &) {
      threw = true;
    }
    expect(threw, "refused: " + why);
  };
  auto base = spec_for(fixture);
  auto zero_rank = base;
  zero_rank.rank = 0U;
  refused(zero_rank, "a rank of zero");
  auto zero_alpha = base;
  zero_alpha.alpha = 0.0;
  refused(zero_alpha, "an alpha of zero");
  auto empty = base;
  empty.operations.clear();
  refused(empty, "no sites at all");
  auto unknown = base;
  unknown.operations = {9999U};
  refused(unknown, "an operation the program does not have");
  auto not_linear = base;
  not_linear.operations = {2U};  // the SiLU
  refused(not_linear, "an operation that is not a Linear");
  auto twice = base;
  twice.operations = {fixture.first_linear, fixture.first_linear};
  refused(twice, "the same site named twice");
}

} // namespace

int main() {
  at_initialisation_lora_is_the_identity();
  a_nonzero_adapter_actually_reaches_the_output();
  every_reader_of_an_adapted_output_sees_the_adapted_value();
  the_scale_is_alpha_over_rank();
  mixed_precision_adapters_cross_an_explicit_cast();
  bad_requests_are_refused();
  if (failures != 0) {
    std::cerr << failures << " LoRA transform failure(s)\n";
    return 1;
  }
  std::cout << "LoRA transform tests passed\n";
  return 0;
}
