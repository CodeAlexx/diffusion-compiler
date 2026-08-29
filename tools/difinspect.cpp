#include "dif/compiler/memory_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/support/sha256.hpp"

#include <iostream>
#include <string>

int main(int argc, char **argv) {
  try {
    if (argc < 2 || argc > 5) {
      std::cerr << "usage: difinspect FILE.difir [--prefetch-distance N]"
                   " [--assignments]\n";
      return 2;
    }
    std::uint64_t prefetch_distance = 0U;
    bool assignments = false;
    for (int argument = 2; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--assignments") {
        assignments = true;
      } else if (option == "--prefetch-distance" && argument + 1 < argc) {
        std::size_t consumed = 0U;
        prefetch_distance = std::stoull(argv[++argument], &consumed, 10);
        if (consumed != std::string(argv[argument]).size())
          throw std::runtime_error("invalid prefetch distance");
      } else {
        throw std::runtime_error("unknown difinspect option");
      }
    }
    const auto program = dif::ir::read_file(argv[1]);
    std::cout << "DiffIR version=" << program.version
              << " fingerprint=" << dif::hex_digest(dif::ir::fingerprint(program))
              << " tensors=" << program.tensors.size()
              << " operations=" << program.operations.size() << "\n";
    for (const auto &tensor : program.tensors) {
      std::cout << "tensor id=" << tensor.id << " dtype=" << dif::ir::dtype_name(tensor.dtype)
                << " roles=" << tensor.roles << " shape=";
      for (std::size_t i = 0; i < tensor.dims.size(); ++i)
        std::cout << (i ? "x" : "") << tensor.dims[i];
      std::cout << " bytes=" << tensor.byte_count() << "\n";
    }
    for (const auto &op : program.operations) {
      std::cout << "op id=" << op.id << " code=" << dif::ir::opcode_name(op.opcode)
                << " inputs=" << op.inputs.size() << " outputs=" << op.outputs.size()
                << " attrs=" << op.attributes.size() << "\n";
    }
    const auto memory =
        dif::compiler::plan_memory(program, 256U, prefetch_distance);
    std::cout << "memory slots=" << memory.slots.size()
              << " planned_bytes=" << memory.total_bytes
              << " naive_bytes=" << memory.naive_bytes
              << " saved_bytes=" << memory.naive_bytes - memory.total_bytes
              << "\n";
    if (assignments) {
      for (const auto &value : memory.assignments)
        std::cout << "assignment tensor=" << value.tensor_id
                  << " slot=" << value.slot_id << " bytes=" << value.bytes
                  << " first=" << value.first_operation
                  << " last=" << value.last_operation << "\n";
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difinspect: " << error.what() << "\n";
    return 1;
  }
}
