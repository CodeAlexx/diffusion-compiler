#include "dif/compiler/slice.hpp"
#include "dif/ir/codec.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"

#include <cstdlib>
#include <iostream>

namespace {

std::uint32_t operation_id(const char *text) {
  char *end = nullptr;
  const auto value = std::strtoul(text, &end, 10);
  if (!end || *end != '\0' || value == 0U)
    dif::fail("invalid operation id");
  return static_cast<std::uint32_t>(value);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: difslice IN.difir OUT.difir FIRST_OP LAST_OP\n";
      return 2;
    }
    const auto program = dif::ir::read_file(argv[1]);
    const auto sliced = dif::compiler::slice_operations(
        program, operation_id(argv[3]), operation_id(argv[4]));
    dif::ir::write_file(sliced, argv[2]);
    std::cout << "SLICE path=" << argv[2]
              << " operations=" << sliced.operations.size()
              << " fingerprint="
              << dif::hex_digest(dif::ir::fingerprint(sliced));
    for (const auto &tensor : sliced.tensors) {
      if (tensor.has_role(dif::ir::TensorRole::Input))
        std::cout << " input=" << tensor.id;
      if (tensor.has_role(dif::ir::TensorRole::Output))
        std::cout << " output=" << tensor.id;
      if (tensor.has_role(dif::ir::TensorRole::Constant))
        std::cout << " constant=" << tensor.id;
    }
    std::cout << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difslice: " << error.what() << "\n";
    return 1;
  }
}
