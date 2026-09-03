// Gate for the opcode registry (include/dif/ir/opcodes.def): the single
// declaration every opcode table derives from. Checks that the registry is
// well formed, that the codec names round-trip, that decode fails closed on an
// unregistered opcode value, and that the optimizer's semantic classification
// is exactly the registry's traits (so nobody re-grows a hand-written list).
#include "dif/ir/ir.hpp"
#include "dif/ir/codec.hpp"
#include "dif/opt/semantics.hpp"
#include "dif/support/error.hpp"

#include <cctype>
#include <cstdint>
#include <iostream>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &what) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << what << '\n';
  }
}

bool snake_case(std::string_view name) {
  if (name.empty() || name.front() == '_' || name.back() == '_')
    return false;
  for (const char c : name)
    if (!(std::islower(static_cast<unsigned char>(c)) ||
          std::isdigit(static_cast<unsigned char>(c)) || c == '_'))
      return false;
  return true;
}

dif::ir::Program tiny_program(dif::ir::Opcode opcode) {
  using namespace dif::ir;
  Program program;
  program.tensors = {
      {1, DType::F32, TensorRole::Input, {4}},
      {2, DType::F32, TensorRole::Input, {4}},
      {3, DType::F32, TensorRole::Output, {4}},
  };
  program.operations = {{1, opcode, {1, 2}, {3}, {}}};
  return program;
}

} // namespace

int main() {
  using namespace dif::ir;
  std::set<std::uint32_t> values;
  std::set<std::string_view> names;
  std::uint32_t largest = 0;
  for (const auto &info : kOpcodeRegistry) {
    expect(info.value != 0U, "opcode value 0 is reserved");
    expect(values.insert(info.value).second,
           "duplicate opcode value " + std::to_string(info.value));
    expect(names.insert(info.name).second,
           "duplicate opcode name " + std::string(info.name));
    expect(snake_case(info.name),
           "opcode name is not snake_case: " + std::string(info.name));
    expect(static_cast<std::uint32_t>(info.opcode) == info.value,
           "enum value disagrees with registry value for " +
               std::string(info.name));
    expect(opcode_name(info.opcode) == info.name,
           "opcode_name round trip for " + std::string(info.name));
    const auto parsed = opcode_from_name(info.name);
    expect(parsed.has_value() && *parsed == info.opcode,
           "opcode_from_name round trip for " + std::string(info.name));
    expect(opcode_is_registered(info.value),
           "opcode_is_registered for " + std::string(info.name));
    expect(opcode_info(info.opcode) == &info,
           "opcode_info identity for " + std::string(info.name));
    // The semantic classification must be the registry, nothing else.
    Operation probe{1, info.opcode, {}, {}, {}};
    expect(dif::opt::pinned_numeric_semantics(probe) ==
               opcode_has_trait(info.opcode, opcode_trait::PinnedNumerics),
           "pinned-numerics trait vs semantics for " + std::string(info.name));
    expect(dif::opt::bit_exact_data_movement(info.opcode) ==
               opcode_has_trait(info.opcode, opcode_trait::DataMovement),
           "data-movement trait vs semantics for " + std::string(info.name));
    expect(dif::opt::dtype_uniform(info.opcode) ==
               opcode_has_trait(info.opcode, opcode_trait::DtypeUniform),
           "dtype-uniform trait vs semantics for " + std::string(info.name));
    expect(dif::opt::pure_operation(info.opcode) ==
               (info.opcode != Opcode::Barrier),
           "purity for " + std::string(info.name));
    largest = std::max(largest, info.value);
  }
  expect(kOpcodeRegistry.size() == kOpcodeCount, "registry size vs count");
  expect(kOpcodeCount >= 75U, "registry lost opcodes");
  expect(!opcode_is_registered(0U), "value 0 must not be registered");
  expect(!opcode_is_registered(largest + 1U),
         "value past the registry must not be registered");
  expect(opcode_name(static_cast<Opcode>(largest + 1U)) == "invalid",
         "opcode_name of an unregistered value is 'invalid'");
  expect(!opcode_from_name("no_such_op").has_value(),
         "opcode_from_name rejects unknown names");

  // The wire format fails closed on an unregistered opcode value: encode
  // refuses to write one, and decode refuses to read one (defense in depth
  // for bytes produced elsewhere).
  bool rejected = false;
  try {
    (void)decode(encode(tiny_program(static_cast<Opcode>(largest + 1U))));
  } catch (const dif::Error &) {
    rejected = true;
  }
  expect(rejected, "codec rejects an unregistered opcode value");
  const auto valid = decode(encode(tiny_program(Opcode::Add)));
  expect(valid.operations.size() == 1U &&
             valid.operations.front().opcode == Opcode::Add,
         "decode accepts a registered opcode");

  if (failures != 0) {
    std::cerr << failures << " opcode registry assertion(s) failed\n";
    return 1;
  }
  std::cout << "PASS: opcode registry (" << kOpcodeCount
            << " opcodes) is unique, round-trips, fails closed, and matches "
               "the semantic tables\n";
  return 0;
}
