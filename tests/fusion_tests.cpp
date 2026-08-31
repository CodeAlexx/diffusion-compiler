// Elementwise region fusion tests.
//
// Covers: opt-in candidate identity (Implementation=2 stamping), region
// detection eligibility and its fail-safe declines (multi-consumer
// intermediates, dying external inputs with index remaps, streamed
// programs, the 16-argument launch cap), fused CUDA emission shape, and the
// regression gate this feature must never lose: fused-vs-unfused outputs
// BYTE-IDENTICAL on CUDA, with the expected kernel-launch reduction.

#include "dif/compiler/compiler.hpp"
#include "dif/frontend/dit_block.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"

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

// Deterministic pseudo-random tensor in [-1, 1]; zeros for I32 and when
// zeroed is requested (optimizer state).
dif::runtime::Tensor make_tensor(dif::ir::DType dtype,
                                 const std::vector<std::uint64_t> &dims,
                                 std::uint64_t seed, bool zeroed = false) {
  dif::runtime::Tensor tensor{dtype, dims, {}};
  std::uint64_t count = 1;
  for (const auto dim : dims)
    count *= dim;
  std::uint64_t state =
      seed * 6364136223846793005ULL + 1442695040888963407ULL;
  const auto next = [&]() {
    state = state * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<float>(
        (static_cast<double>((state >> 33U) & 0x7fffffffU) /
         static_cast<double>(0x7fffffffU)) *
            2.0 -
        1.0);
  };
  if (dtype == dif::ir::DType::I32) {
    tensor.bytes.assign(count * 4U, 0);
    return tensor;
  }
  if (dtype == dif::ir::DType::F32) {
    tensor.bytes.resize(count * 4U);
    for (std::uint64_t index = 0; index < count; ++index) {
      const float value = zeroed ? 0.0F : next();
      std::memcpy(tensor.bytes.data() + index * 4U, &value, 4U);
    }
    return tensor;
  }
  if (dtype == dif::ir::DType::BF16) {
    tensor.bytes.resize(count * 2U);
    for (std::uint64_t index = 0; index < count; ++index) {
      const std::uint16_t value = zeroed ? 0U : bf16_bits(next());
      std::memcpy(tensor.bytes.data() + index * 2U, &value, 2U);
    }
    return tensor;
  }
  expect(false, "unsupported test tensor dtype");
  return tensor;
}

// Stamp Implementation=2 on every elementwise-fusable operation (the difc
// set-elementwise-fusion transform, in-memory).
dif::ir::Program stamped(const dif::ir::Program &program) {
  auto copy = program;
  for (auto &operation : copy.operations) {
    using dif::ir::Opcode;
    const auto fusable =
        operation.opcode == Opcode::Add ||
        operation.opcode == Opcode::Multiply ||
        operation.opcode == Opcode::SiLU ||
        operation.opcode == Opcode::Clamp ||
        operation.opcode == Opcode::Cast ||
        operation.opcode == Opcode::BiasAdd ||
        operation.opcode == Opcode::ResidualGate ||
        operation.opcode == Opcode::SwiGlu;
    if (!fusable)
      continue;
    auto *attribute = const_cast<dif::ir::Attribute *>(
        operation.find(dif::ir::AttrKey::Implementation));
    if (attribute)
      *attribute =
          dif::ir::Attribute::u64(dif::ir::AttrKey::Implementation, 2U);
    else
      operation.attributes.push_back(
          dif::ir::Attribute::u64(dif::ir::AttrKey::Implementation, 2U));
  }
  dif::ir::verify(copy);
  return copy;
}

bool same_output_bytes(const dif::runtime::TensorMap &left,
                       const dif::runtime::TensorMap &right,
                       const std::string &label) {
  bool same = left.size() == right.size();
  expect(same, label + ": output tensor count");
  for (const auto &[id, tensor] : left) {
    const auto found = right.find(id);
    if (found == right.end()) {
      expect(false, label + ": missing output id " + std::to_string(id));
      same = false;
      continue;
    }
    if (tensor.bytes.size() != found->second.bytes.size() ||
        std::memcmp(tensor.bytes.data(), found->second.bytes.data(),
                    tensor.bytes.size()) != 0) {
      expect(false, label + ": output id " + std::to_string(id) +
                        " is not byte-identical");
      same = false;
    }
  }
  return same;
}

dif::runtime::RunOptions single_run_options() {
  dif::runtime::RunOptions options;
  options.warmups = 0;
  options.iterations = 1;
  options.minimum_free_bytes = 0;
  return options;
}

// A six-operation full-family chain whose external operands are all
// dedicated (Input/Constant): Add -> SiLU -> BiasAdd -> Multiply -> Clamp ->
// ResidualGate.
dif::ir::Program full_family_chain(dif::ir::DType dtype) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, dtype, TensorRole::Input, {64, 128}},
      {2, dtype, TensorRole::Constant, {64, 128}},
      {3, dtype, TensorRole::Constant, {128}},
      {4, dtype, TensorRole::Internal, {64, 128}},
      {5, dtype, TensorRole::Internal, {64, 128}},
      {6, dtype, TensorRole::Internal, {64, 128}},
      {7, dtype, TensorRole::Internal, {64, 128}},
      {8, dtype, TensorRole::Internal, {64, 128}},
      {9, dtype, TensorRole::Output, {64, 128}},
  };
  program.operations = {
      {1, Opcode::Add, {1, 2}, {4}, {}},
      {2, Opcode::SiLU, {4}, {5}, {}},
      {3, Opcode::BiasAdd, {5, 3}, {6}, {}},
      {4, Opcode::Multiply, {6, 1}, {7}, {}},
      {5, Opcode::Clamp, {7}, {8},
       {Attribute::f64(AttrKey::Lower, -0.75),
        Attribute::f64(AttrKey::Upper, 0.85)}},
      {6, Opcode::ResidualGate, {1, 8, 2}, {9}, {}},
  };
  dif::ir::verify(program);
  return program;
}

dif::runtime::TensorMap full_family_bindings(dif::ir::DType dtype) {
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, make_tensor(dtype, {64, 128}, 11));
  bindings.emplace(2, make_tensor(dtype, {64, 128}, 23));
  bindings.emplace(3, make_tensor(dtype, {128}, 37));
  return bindings;
}

void test_detection_and_emission() {
  const auto program = full_family_chain(dif::ir::DType::BF16);
  const auto candidate = stamped(program);

  const auto unstamped_census =
      dif::compiler::census_elementwise_fusion(program);
  expect(unstamped_census.regions == 0U &&
             unstamped_census.eliminated_launches == 0U,
         "unstamped program reports zero fusion regions (default OFF)");
  const auto census = dif::compiler::census_elementwise_fusion(candidate);
  expect(census.regions == 1U && census.fused_operations == 6U &&
             census.eliminated_launches == 5U,
         "stamped full-family chain censuses as one six-op region");

  expect(dif::ir::fingerprint(program) != dif::ir::fingerprint(candidate),
         "stamping is a fingerprinted candidate transform");

  const auto unfused = dif::compiler::emit_cuda(program);
  expect(unfused.skipped_operations.empty() &&
             unfused.launch_inputs.empty() &&
             unfused.entrypoints.size() == 6U,
         "unstamped emission is untouched by the fusion pass");

  const auto fused = dif::compiler::emit_cuda(candidate);
  expect(fused.entrypoints.size() == 1U && fused.entrypoints.contains(6U),
         "fused emission keeps only the anchor entrypoint");
  expect(fused.skipped_operations.size() == 5U &&
             fused.skipped_operations.contains(1U) &&
             fused.skipped_operations.contains(5U) &&
             !fused.skipped_operations.contains(6U),
         "fused emission subsumes the five interior operations");
  const auto arguments = fused.launch_inputs.find(6U);
  expect(arguments != fused.launch_inputs.end() &&
             arguments->second.size() == 3U,
         "fused anchor launches with the three deduplicated external inputs");
  expect(fused.source.find("asm(\"\" : \"+f\"") != std::string::npos,
         "fused stages carry the contraction value barrier");
  std::size_t kernels = 0;
  for (std::size_t at = fused.source.find("__global__ void");
       at != std::string::npos;
       at = fused.source.find("__global__ void", at + 1U))
    ++kernels;
  expect(kernels == 1U, "fully fused program emits exactly one kernel");
}

void test_declines() {
  using namespace dif::ir;
  // (a) Multi-consumer intermediate: no merge edge, zero regions.
  {
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {8, 8}},
        {2, DType::F32, TensorRole::Internal, {8, 8}},
        {3, DType::F32, TensorRole::Output, {8, 8}},
        {4, DType::F32, TensorRole::Output, {8, 8}},
    };
    program.operations = {
        {1, Opcode::SiLU, {1}, {2}, {}},
        {2, Opcode::SiLU, {2}, {3}, {}},
        {3, Opcode::SiLU, {2}, {4}, {}},
    };
    dif::ir::verify(program);
    const auto census =
        dif::compiler::census_elementwise_fusion(stamped(program));
    expect(census.regions == 0U,
           "multi-consumer intermediate declines fusion");
  }
  // (b) Dying external input read through a SwiGlu index remap with a
  // non-dedicated anchor output: declined.  The identical region with an
  // Output-role anchor output is accepted (nothing writes the dead slot).
  {
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {8, 64}},
        {2, DType::F32, TensorRole::Internal, {8, 64}}, // dies inside region
        {3, DType::F32, TensorRole::Constant, {8, 64}},
        {4, DType::F32, TensorRole::Internal, {8, 64}},
        {5, DType::F32, TensorRole::Internal, {8, 32}}, // anchor output
        {6, DType::F32, TensorRole::Input, {8, 32}},
        {7, DType::F32, TensorRole::Output, {8, 32}},
    };
    program.operations = {
        {1, Opcode::SiLU, {1}, {2}, {}}, // unstamped producer (see below)
        {2, Opcode::Add, {2, 3}, {4}, {}},
        {3, Opcode::SwiGlu, {4}, {5},
         {Attribute::boolean(AttrKey::GateFirst, true)}},
        {4, Opcode::Multiply, {5, 6}, {7}, {}},
    };
    dif::ir::verify(program);
    auto candidate = stamped(program);
    // Keep the SiLU producer and the tail Multiply out of the region so the
    // region is exactly {Add, SwiGlu} with tensor 2 dying inside it and the
    // anchor output internal (consumed by the unstamped Multiply).
    for (auto &operation : candidate.operations) {
      if (operation.id == 1U || operation.id == 4U)
        operation.attributes.clear();
    }
    dif::ir::verify(candidate);
    const auto census = dif::compiler::census_elementwise_fusion(candidate);
    expect(census.regions == 0U,
           "dying remap-read input with internal anchor output declines");

    auto accepted = candidate;
    for (auto &tensor : accepted.tensors)
      if (tensor.id == 5U)
        tensor.roles = TensorRole::Output;
    dif::ir::verify(accepted);
    const auto accepted_census =
        dif::compiler::census_elementwise_fusion(accepted);
    expect(accepted_census.regions == 1U &&
               accepted_census.fused_operations == 2U,
           "same region with a dedicated anchor output is accepted");
  }
  // (c) Any dying input in a program with streamed constants: declined.
  {
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {8, 8}},
        {2, DType::F32, TensorRole::Internal, {8, 8}},
        {3, DType::F32, TensorRole::Internal, {8, 8}},
        {4, DType::F32, TensorRole::Output, {8, 8}},
        {5, DType::F32,
         static_cast<std::uint32_t>(TensorRole::Constant) |
             static_cast<std::uint32_t>(TensorRole::Streamed),
         {8, 8}},
    };
    program.operations = {
        {1, Opcode::Multiply, {1, 5}, {2}, {}}, // unstamped producer
        {2, Opcode::SiLU, {2}, {3}, {}},
        {3, Opcode::SiLU, {3}, {4}, {}},
    };
    dif::ir::verify(program);
    auto candidate = stamped(program);
    for (auto &operation : candidate.operations)
      if (operation.id == 1U)
        operation.attributes.clear();
    dif::ir::verify(candidate);
    const auto census = dif::compiler::census_elementwise_fusion(candidate);
    expect(census.regions == 0U,
           "dying input inside a streamed program declines");
  }
  // (d) Launch-argument cap: a 15-add chain needs 16 external pointers plus
  // the output and is declined; a 14-add chain (15 externals) fuses.
  const auto add_chain = [](std::uint32_t adds) {
    Program program;
    std::uint32_t next_tensor = 1U;
    std::vector<std::uint32_t> inputs;
    for (std::uint32_t index = 0; index <= adds; ++index) {
      program.tensors.push_back(
          {next_tensor, DType::F32, TensorRole::Input, {4, 4}});
      inputs.push_back(next_tensor++);
    }
    std::uint32_t carry = inputs[0];
    for (std::uint32_t index = 0; index < adds; ++index) {
      const auto role = index + 1U == adds
                            ? static_cast<std::uint32_t>(TensorRole::Output)
                            : static_cast<std::uint32_t>(TensorRole::Internal);
      program.tensors.push_back({next_tensor, DType::F32, role, {4, 4}});
      program.operations.push_back(
          {index + 1U, Opcode::Add, {carry, inputs[index + 1U]},
           {next_tensor}, {}});
      carry = next_tensor++;
    }
    dif::ir::verify(program);
    return program;
  };
  const auto over = dif::compiler::census_elementwise_fusion(
      stamped(add_chain(15U)));
  expect(over.regions == 0U, "sixteen external inputs decline (16-arg cap)");
  const auto under = dif::compiler::census_elementwise_fusion(
      stamped(add_chain(14U)));
  expect(under.regions == 1U && under.fused_operations == 14U,
         "fifteen external inputs fuse under the launch-argument cap");
}

void run_byte_identity(const dif::ir::Program &program,
                       const dif::runtime::TensorMap &bindings,
                       std::uint64_t expected_eliminated,
                       const std::string &label) {
  const auto candidate = stamped(program);
  const auto census = dif::compiler::census_elementwise_fusion(candidate);
  expect(census.eliminated_launches == expected_eliminated,
         label + ": expected launch elimination census");

  const auto options = single_run_options();
  const auto cpu_reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto cpu_candidate =
      dif::runtime::make_cpu_executor()->run(candidate, bindings, options);
  same_output_bytes(cpu_reference.outputs, cpu_candidate.outputs,
                    label + ": CPU executor ignores the stamp");

  if (!dif::runtime::cuda_available())
    return;
  const auto cuda_reference =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const auto cuda_candidate =
      dif::runtime::make_cuda_executor()->run(candidate, bindings, options);
  same_output_bytes(cuda_reference.outputs, cuda_candidate.outputs,
                    label + ": CUDA fused vs unfused byte identity");
  const auto reference_launches =
      cuda_reference.run_telemetry.kernel_launches;
  const auto candidate_launches =
      cuda_candidate.run_telemetry.kernel_launches;
  expect(reference_launches ==
             candidate_launches + expected_eliminated,
         label + ": kernel launches drop by the eliminated-op count");
  std::cout << "LAUNCHES " << label << " unfused=" << reference_launches
            << " fused=" << candidate_launches << "\n";
}

void test_byte_identity_full_family() {
  for (const auto dtype : {dif::ir::DType::BF16, dif::ir::DType::F32}) {
    const auto label = dtype == dif::ir::DType::BF16
                           ? std::string("full-family-bf16")
                           : std::string("full-family-f32");
    run_byte_identity(full_family_chain(dtype), full_family_bindings(dtype),
                      5U, label);
  }
}

void test_byte_identity_contraction_hazard() {
  // Multiply feeding Add is the exact FMA-contraction hazard: without the
  // stage barrier NVRTC would contract round(a*b)+c into fma(a,b,c) in F32.
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {32, 32}},
      {2, DType::F32, TensorRole::Input, {32, 32}},
      {3, DType::F32, TensorRole::Input, {32, 32}},
      {4, DType::F32, TensorRole::Internal, {32, 32}},
      {5, DType::F32, TensorRole::Output, {32, 32}},
  };
  program.operations = {
      {1, Opcode::Multiply, {1, 2}, {4}, {}},
      {2, Opcode::Add, {4, 3}, {5}, {}},
  };
  dif::ir::verify(program);
  dif::runtime::TensorMap bindings;
  bindings.emplace(1, make_tensor(DType::F32, {32, 32}, 101));
  bindings.emplace(2, make_tensor(DType::F32, {32, 32}, 103));
  bindings.emplace(3, make_tensor(DType::F32, {32, 32}, 107));
  run_byte_identity(program, bindings, 1U, "multiply-add-f32");
}

void test_byte_identity_swiglu_and_cast() {
  using namespace dif::ir;
  // Add -> SwiGlu(terminal-interior) -> Multiply, both SwiGlu orderings.
  for (const auto gate_first : {false, true}) {
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {8, 64}},
        {2, DType::F32, TensorRole::Input, {8, 64}},
        {3, DType::F32, TensorRole::Internal, {8, 64}},
        {4, DType::F32, TensorRole::Internal, {8, 32}},
        {5, DType::F32, TensorRole::Input, {8, 32}},
        {6, DType::F32, TensorRole::Output, {8, 32}},
    };
    program.operations = {
        {1, Opcode::Add, {1, 2}, {3}, {}},
        {2, Opcode::SwiGlu, {3}, {4},
         {Attribute::boolean(AttrKey::GateFirst, gate_first)}},
        {3, Opcode::Multiply, {4, 5}, {6}, {}},
    };
    dif::ir::verify(program);
    dif::runtime::TensorMap bindings;
    bindings.emplace(1, make_tensor(DType::F32, {8, 64}, 211));
    bindings.emplace(2, make_tensor(DType::F32, {8, 64}, 223));
    bindings.emplace(5, make_tensor(DType::F32, {8, 32}, 227));
    run_byte_identity(program, bindings, 2U,
                      gate_first ? "swiglu-gate-first" : "swiglu-gate-last");
  }
  // F32 Add -> Cast(bf16) -> SiLU: a dtype boundary inside the region.
  {
    Program program;
    program.tensors = {
        {1, DType::F32, TensorRole::Input, {16, 32}},
        {2, DType::F32, TensorRole::Input, {16, 32}},
        {3, DType::F32, TensorRole::Internal, {16, 32}},
        {4, DType::BF16, TensorRole::Internal, {16, 32}},
        {5, DType::BF16, TensorRole::Output, {16, 32}},
    };
    program.operations = {
        {1, Opcode::Add, {1, 2}, {3}, {}},
        {2, Opcode::Cast, {3}, {4}, {}},
        {3, Opcode::SiLU, {4}, {5}, {}},
    };
    dif::ir::verify(program);
    dif::runtime::TensorMap bindings;
    bindings.emplace(1, make_tensor(DType::F32, {16, 32}, 307));
    bindings.emplace(2, make_tensor(DType::F32, {16, 32}, 311));
    run_byte_identity(program, bindings, 2U, "add-cast-silu");
  }
}

void test_composed_dit_block_program() {
  // The composed DiT-block training program (forward + backward + AdamW).
  // The census here is a measurement, not an assertion: the honest expected
  // value may be zero because pointwise operations in this graph are
  // separated by Linear/Attention operations and gradient-accumulation Adds
  // die at their consumers.  The byte-identity requirement is absolute.
  dif::frontend::DitBlockTrainingConfig config;
  config.sequence = 16;
  config.heads = 2;
  config.head_dim = 8;
  config.mlp_width = 16;
  config.blocks = 2;
  config.rotary_dim = 8;
  config.full_rope_table = true;
  config.causal = false;
  config.learning_rate = 1.0e-3;
  config.beta1 = 0.9;
  config.beta2 = 0.999;
  config.epsilon_adam = 1.0e-8;
  config.weight_decay = 0.01;
  config.epsilon_norm = 1.0e-5;
  const auto build = dif::frontend::make_dit_block_training(config);
  const auto &program = build.program;
  const auto candidate = stamped(program);
  const auto census = dif::compiler::census_elementwise_fusion(candidate);
  std::cout << "DIT_BLOCK_CENSUS regions=" << census.regions
            << " fused_ops=" << census.fused_operations
            << " eliminated_launches=" << census.eliminated_launches << "\n";

  dif::runtime::TensorMap bindings;
  for (const auto &tensor : program.tensors) {
    if (!tensor.has_role(dif::ir::TensorRole::Input))
      continue;
    const bool zeroed = tensor.has_role(dif::ir::TensorRole::OptimizerState);
    bindings.emplace(tensor.id,
                     make_tensor(tensor.dtype, tensor.dims,
                                 401U + tensor.id, zeroed));
  }
  const auto options = single_run_options();
  const auto cpu_reference =
      dif::runtime::make_cpu_executor()->run(program, bindings, options);
  const auto cpu_candidate =
      dif::runtime::make_cpu_executor()->run(candidate, bindings, options);
  same_output_bytes(cpu_reference.outputs, cpu_candidate.outputs,
                    "dit-block: CPU executor ignores the stamp");
  if (!dif::runtime::cuda_available())
    return;
  const auto cuda_reference =
      dif::runtime::make_cuda_executor()->run(program, bindings, options);
  const auto cuda_candidate =
      dif::runtime::make_cuda_executor()->run(candidate, bindings, options);
  same_output_bytes(cuda_reference.outputs, cuda_candidate.outputs,
                    "dit-block: CUDA fused vs unfused byte identity");
  expect(cuda_reference.run_telemetry.kernel_launches ==
             cuda_candidate.run_telemetry.kernel_launches +
                 census.eliminated_launches,
         "dit-block: launch count drops by exactly the census");
  std::cout << "LAUNCHES dit-block unfused="
            << cuda_reference.run_telemetry.kernel_launches << " fused="
            << cuda_candidate.run_telemetry.kernel_launches << "\n";
}

} // namespace

int main() {
  test_detection_and_emission();
  test_declines();
  test_byte_identity_full_family();
  test_byte_identity_contraction_hazard();
  test_byte_identity_swiglu_and_cast();
  test_composed_dit_block_program();
  if (!dif::runtime::cuda_available())
    std::cout << "NOTE: CUDA unavailable; GPU byte-identity gates skipped\n";
  if (failures != 0) {
    std::cerr << failures << " fusion test failure(s)\n";
    return 1;
  }
  std::cout << "fusion tests passed\n";
  return 0;
}
