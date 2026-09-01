#include "dif/compiler/memory_plan.hpp"
#include "dif/compiler/layout_plan.hpp"
#include "dif/compiler/residency_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/support/sha256.hpp"

#include <iostream>
#include <optional>
#include <string>
#include <unordered_set>

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: difinspect FILE.difir [--prefetch-distance N]"
                   " [--resident-plan-mib N] [--fixed-runtime-mib N]"
                   " [--resident-order first|largest]"
                   " [--alias-reshapes] [--assignments]\n";
      return 2;
    }
    std::uint64_t prefetch_distance = 0U;
    std::uint64_t resident_plan_mib = 0U;
    std::uint64_t fixed_runtime_mib = 0U;
    bool assignments = false;
    bool alias_reshapes = false;
    auto residency_order =
        dif::compiler::StreamedResidencyOrder::FirstConsumer;
    for (int argument = 2; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--assignments") {
        assignments = true;
      } else if (option == "--alias-reshapes") {
        alias_reshapes = true;
      } else if (option == "--prefetch-distance" && argument + 1 < argc) {
        std::size_t consumed = 0U;
        prefetch_distance = std::stoull(argv[++argument], &consumed, 10);
        if (consumed != std::string(argv[argument]).size())
          throw std::runtime_error("invalid prefetch distance");
      } else if (option == "--resident-plan-mib" && argument + 1 < argc) {
        resident_plan_mib = std::stoull(argv[++argument]);
      } else if (option == "--fixed-runtime-mib" && argument + 1 < argc) {
        fixed_runtime_mib = std::stoull(argv[++argument]);
      } else if (option == "--resident-order" && argument + 1 < argc) {
        const std::string value = argv[++argument];
        if (value == "first")
          residency_order =
              dif::compiler::StreamedResidencyOrder::FirstConsumer;
        else if (value == "largest")
          residency_order =
              dif::compiler::StreamedResidencyOrder::LargestFirst;
        else
          throw std::runtime_error("resident order accepts first or largest");
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
                << " inputs=";
      for (std::size_t i = 0; i < op.inputs.size(); ++i)
        std::cout << (i ? "," : "") << op.inputs[i];
      std::cout << " outputs=";
      for (std::size_t i = 0; i < op.outputs.size(); ++i)
        std::cout << (i ? "," : "") << op.outputs[i];
      std::cout << " attrs=" << op.attributes.size() << "\n";
    }
    std::optional<dif::compiler::StreamedResidencyPlan> residency;
    std::optional<dif::compiler::ReshapeAliasPlan> reshape_alias_plan;
    if (alias_reshapes) {
      reshape_alias_plan.emplace(
          dif::compiler::plan_reshape_aliases(program));
      std::cout << "reshape_aliases operations="
                << reshape_alias_plan->operation_ids.size()
                << " eliminated_materialization_bytes="
                << reshape_alias_plan->eliminated_materialization_bytes
                << "\n";
    }
    std::unordered_set<std::uint32_t> resident_ids;
    if (resident_plan_mib != 0U) {
      residency.emplace(dif::compiler::plan_streamed_residency(
          program, resident_plan_mib * 1024ULL * 1024ULL,
          fixed_runtime_mib * 1024ULL * 1024ULL, prefetch_distance,
          reshape_alias_plan
              ? reshape_alias_plan->output_to_root_input
              : std::unordered_map<std::uint32_t, std::uint32_t>{},
          residency_order));
      resident_ids.insert(residency->resident_tensor_ids.begin(),
                          residency->resident_tensor_ids.end());
      std::cout << "residency resident_tensors="
                << residency->resident_tensor_ids.size()
                << " resident_bytes=" << residency->resident_constant_bytes
                << " streamed_bytes=" << residency->streamed_constant_bytes
                << " memory_plan_bytes=" << residency->memory_plan_bytes
                << " fixed_runtime_bytes=" << residency->fixed_runtime_bytes
                << " required_bytes=" << residency->required_bytes << "\n";
    }
    const auto memory = dif::compiler::plan_memory(
        program, 256U, prefetch_distance, {}, resident_ids,
        reshape_alias_plan
            ? reshape_alias_plan->output_to_root_input
            : std::unordered_map<std::uint32_t, std::uint32_t>{});
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
