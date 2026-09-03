#include "dif/ir/ir.hpp"

#include <algorithm>

namespace dif::ir {

// The registry itself lives in ir.hpp (constexpr, generated from opcodes.def);
// these are the runtime lookups built on it.

const OpcodeInfo *opcode_info(Opcode opcode) {
  for (const auto &info : kOpcodeRegistry)
    if (info.opcode == opcode)
      return &info;
  return nullptr;
}

std::string_view opcode_name(Opcode opcode) {
  const auto *info = opcode_info(opcode);
  return info ? info->name : std::string_view{"invalid"};
}

std::optional<Opcode> opcode_from_name(std::string_view name) {
  for (const auto &info : kOpcodeRegistry)
    if (info.name == name)
      return info.opcode;
  return std::nullopt;
}

bool opcode_is_registered(std::uint32_t value) {
  return std::any_of(kOpcodeRegistry.begin(), kOpcodeRegistry.end(),
                     [value](const OpcodeInfo &info) {
                       return info.value == value;
                     });
}

bool opcode_has_trait(Opcode opcode, std::uint32_t trait) {
  const auto *info = opcode_info(opcode);
  return info != nullptr && (info->traits & trait) == trait;
}

} // namespace dif::ir
