// difinspect: structural DiffIR listing, memory/residency planning views,
// and (--source) provenance navigation that joins each operation with the
// creator module and section its frontend recorded, its checkpoint weight
// identity from a sealed bundle, the compiler transforms and decisions that
// name it, and the backend implementations a runtime trace observed for it.
// Absent links are reported as absent; nothing is inferred from names.

#include "dif/compiler/layout_plan.hpp"
#include "dif/compiler/memory_plan.hpp"
#include "dif/compiler/residency_plan.hpp"
#include "dif/frontend/provenance.hpp"
#include "dif/ir/codec.hpp"
#include "dif/opt/plan.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/vocabulary.hpp"
#include "dif/weights/bundle.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace {

void usage() {
  std::cerr << "usage: difinspect FILE.difir [--prefetch-distance N]"
               " [--resident-plan-mib N] [--fixed-runtime-mib N]"
               " [--resident-order first|largest]"
               " [--alias-reshapes] [--assignments] [--json]\n"
               "       difinspect FILE.difir --source [--provenance FILE.provenance.json]"
               " [--bundle FILE.difbind] [--plan FILE.difplan]"
               " [--trace FILE.jsonl] [--op ID] [--json]\n"
               "\n"
               "--source joins creator provenance (frontend sidecar, default\n"
               "FILE.difir.provenance.json), checkpoint weight identity (bundle),\n"
               "compiler transforms/decisions (plan), and observed backend\n"
               "implementations (runtime-trace JSON lines) per operation.\n";
}

struct BackendObservation {
  std::map<std::string, std::uint64_t> apis;
  std::uint64_t documents{};
};

std::map<std::uint32_t, BackendObservation>
read_trace(const std::filesystem::path &path, std::uint64_t &documents) {
  std::map<std::uint32_t, BackendObservation> out;
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    throw std::runtime_error("cannot open trace file " + path.string());
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty())
      continue;
    const auto document = dif::json::parse(line);
    ++documents;
    const auto *trace = document.find("trace");
    if (!trace)
      continue;
    const auto *events = trace->find("run_events");
    if (!events || !events->is_array())
      continue;
    for (const auto &event : events->array()) {
      const auto *category = event.find("category");
      const auto *operation = event.find("operation_id");
      const auto *name = event.find("name");
      if (!category || !operation || !name)
        continue;
      const auto &kind = category->string();
      if (kind != dif::telemetry::category::gemm &&
          kind != dif::telemetry::category::attention &&
          kind != dif::telemetry::category::convolution &&
          kind != dif::telemetry::category::generated_kernel &&
          kind != dif::telemetry::category::layout)
        continue;
      const auto id = static_cast<std::uint32_t>(operation->number());
      if (id == 0U)
        continue;
      ++out[id].apis[kind + ":" + name->string()];
    }
  }
  return out;
}

dif::telemetry::Object operation_row(
    const dif::ir::Program &program, std::size_t position,
    const dif::frontend::ProvenanceTable *provenance,
    const dif::weights::WeightBundle *bundle,
    const dif::opt::OptimizationPlan *plan,
    const std::map<std::uint32_t, BackendObservation> *backend) {
  const auto &op = program.operations[position];
  dif::telemetry::Object row;
  row.set("operation", op.id);
  row.set("position", position);
  row.set("opcode", dif::ir::opcode_name(op.opcode));
  dif::telemetry::Array inputs;
  for (const auto id : op.inputs)
    inputs.push_back(id);
  row.set("inputs", std::move(inputs));
  dif::telemetry::Array outputs;
  for (const auto id : op.outputs)
    outputs.push_back(id);
  row.set("outputs", std::move(outputs));
  if (const auto *implementation = op.find(dif::ir::AttrKey::Implementation))
    row.set("implementation_attribute", implementation->as_u64());
  else
    row.set("implementation_attribute", nullptr);

  // Creator semantic: only what the frontend recorded.
  if (provenance) {
    if (const auto *record = provenance->find(op.id)) {
      dif::telemetry::Object creator;
      creator.set("frontend", provenance->frontend);
      creator.set("creator", provenance->creator);
      creator.set("creator_revision", provenance->creator_revision);
      creator.set("module", record->creator_module);
      creator.set("block", record->block);
      creator.set("tag", record->semantic_tag);
      row.set("provenance", std::move(creator));
    } else {
      row.set("provenance", nullptr);
    }
  } else {
    row.set("provenance", nullptr);
  }

  // Weight identity for constant inputs: creator checkpoint name from the
  // frontend, storage identity from the sealed bundle.
  dif::telemetry::Array weights;
  for (const auto id : op.inputs) {
    const auto *tensor = program.tensor(id);
    if (!tensor || !tensor->has_role(dif::ir::TensorRole::Constant))
      continue;
    dif::telemetry::Object weight;
    weight.set("tensor", id);
    weight.set("dtype", dif::ir::dtype_name(tensor->dtype));
    weight.set("bytes", tensor->byte_count());
    weight.set("streamed", tensor->has_role(dif::ir::TensorRole::Streamed));
    const std::string *name = provenance ? provenance->weight_name(id) : nullptr;
    weight.set("creator_name", name ? dif::telemetry::Value(*name)
                                    : dif::telemetry::Value(nullptr));
    bool bound = false;
    if (bundle) {
      for (const auto &binding : bundle->bindings) {
        if (binding.tensor_id != id)
          continue;
        dif::telemetry::Object storage;
        storage.set("checkpoint_name", binding.tensor_name);
        const auto &shard = bundle->shards.at(binding.shard_index);
        storage.set("shard", shard.path.string());
        storage.set("shard_sha256", dif::hex_digest(shard.digest));
        storage.set("file_offset", binding.file_offset);
        storage.set("byte_count", binding.byte_count);
        weight.set("bundle", std::move(storage));
        bound = true;
        break;
      }
    }
    if (!bound)
      weight.set("bundle", nullptr);
    weights.push_back(std::move(weight));
  }
  row.set("weights", std::move(weights));

  // Compiler region: transforms and decisions that name this operation.
  dif::telemetry::Object compiler;
  if (plan) {
    dif::telemetry::Array transforms;
    for (const auto &transform : plan->transforms) {
      const bool explicit_site =
          std::find(transform.operations.begin(), transform.operations.end(),
                    op.id) != transform.operations.end();
      const bool whole_program =
          transform.operations.empty() && transform.tensors.empty();
      if (!explicit_site && !whole_program)
        continue;
      dif::telemetry::Object entry;
      entry.set("encoded", dif::opt::encode_transform(transform));
      entry.set("class", dif::opt::transform_class_name(
                             dif::opt::transform_class(transform.kind)));
      entry.set("scope", explicit_site ? "explicit" : "whole-program");
      transforms.push_back(std::move(entry));
    }
    compiler.set("transforms", std::move(transforms));
    dif::telemetry::Array decisions;
    for (const auto &decision : plan->decisions) {
      bool relevant = decision.subject == "operation" && decision.subject_id == op.id;
      if (!relevant && decision.subject == "candidate") {
        for (const auto &[key, value] : decision.evidence) {
          if (key != "transforms")
            continue;
          try {
            for (const auto &transform :
                 dif::opt::decode_transform_sequence(value))
              relevant = relevant ||
                         std::find(transform.operations.begin(),
                                   transform.operations.end(),
                                   op.id) != transform.operations.end();
          } catch (const std::exception &) {
          }
        }
      }
      if (!relevant)
        continue;
      dif::telemetry::Object entry;
      entry.set("subject", decision.subject);
      entry.set("id", decision.subject_id);
      entry.set("decision", decision.decision);
      entry.set("reason", decision.reason);
      decisions.push_back(std::move(entry));
    }
    compiler.set("decisions", std::move(decisions));
    row.set("compiler", std::move(compiler));
  } else {
    row.set("compiler", nullptr);
  }

  // Backend implementation actually observed for this operation.
  if (backend) {
    const auto found = backend->find(op.id);
    dif::telemetry::Object observed;
    if (found == backend->end()) {
      observed.set("observed", false);
      observed.set("implementations", dif::telemetry::Object{});
    } else {
      observed.set("observed", true);
      dif::telemetry::Object implementations;
      for (const auto &[api, count] : found->second.apis)
        implementations.set(api, count);
      observed.set("implementations", std::move(implementations));
    }
    row.set("backend", std::move(observed));
  } else {
    row.set("backend", nullptr);
  }
  return row;
}

int source_mode(const std::filesystem::path &program_path, int argc,
                char **argv) {
  std::filesystem::path provenance_path;
  std::filesystem::path bundle_path;
  std::filesystem::path plan_path;
  std::filesystem::path trace_path;
  std::optional<std::uint32_t> only;
  bool json = false;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        throw std::runtime_error("missing value after " + option);
      return argv[++index];
    };
    if (option == "--source")
      continue;
    if (option == "--provenance")
      provenance_path = value();
    else if (option == "--bundle")
      bundle_path = value();
    else if (option == "--plan")
      plan_path = value();
    else if (option == "--trace")
      trace_path = value();
    else if (option == "--op")
      only = static_cast<std::uint32_t>(std::stoul(value()));
    else if (option == "--json")
      json = true;
    else
      throw std::runtime_error("unknown difinspect --source option: " + option);
  }
  const auto program = dif::ir::read_file(program_path);
  std::optional<dif::frontend::ProvenanceTable> provenance;
  if (provenance_path.empty()) {
    const auto sidecar = dif::frontend::provenance_sidecar_path(program_path);
    if (std::filesystem::exists(sidecar))
      provenance_path = sidecar;
  }
  if (!provenance_path.empty())
    provenance.emplace(dif::frontend::read_provenance(provenance_path));
  std::optional<dif::weights::WeightBundle> bundle;
  if (!bundle_path.empty())
    bundle.emplace(dif::weights::read_weight_bundle(bundle_path));
  std::optional<dif::opt::OptimizationPlan> plan;
  if (!plan_path.empty())
    plan.emplace(dif::opt::read_plan(plan_path));
  std::optional<std::map<std::uint32_t, BackendObservation>> backend;
  std::uint64_t trace_documents = 0U;
  if (!trace_path.empty())
    backend.emplace(read_trace(trace_path, trace_documents));

  auto document = dif::telemetry::make_document("provenance-report");
  dif::telemetry::Object sources;
  sources.set("program", std::filesystem::absolute(program_path).string());
  sources.set("program_fingerprint",
              dif::hex_digest(dif::ir::fingerprint(program)));
  sources.set("provenance", provenance_path.empty()
                                ? dif::telemetry::Value(nullptr)
                                : dif::telemetry::Value(provenance_path.string()));
  sources.set("bundle", bundle_path.empty()
                            ? dif::telemetry::Value(nullptr)
                            : dif::telemetry::Value(bundle_path.string()));
  sources.set("plan", plan_path.empty() ? dif::telemetry::Value(nullptr)
                                        : dif::telemetry::Value(plan_path.string()));
  sources.set("trace", trace_path.empty() ? dif::telemetry::Value(nullptr)
                                          : dif::telemetry::Value(trace_path.string()));
  sources.set("trace_documents", trace_documents);
  if (provenance) {
    dif::telemetry::Object creator;
    creator.set("frontend", provenance->frontend);
    creator.set("creator", provenance->creator);
    creator.set("creator_revision", provenance->creator_revision);
    creator.set("recorded_operations", provenance->records.size());
    creator.set("recorded_weights", provenance->weight_names.size());
    sources.set("creator", std::move(creator));
  } else {
    sources.set("creator", nullptr);
  }
  document.set("sources", std::move(sources));

  dif::telemetry::Array rows;
  std::size_t with_provenance = 0U;
  std::size_t with_backend = 0U;
  for (std::size_t position = 0; position < program.operations.size(); ++position) {
    const auto &op = program.operations[position];
    if (only && op.id != *only)
      continue;
    auto row = operation_row(program, position, provenance ? &*provenance : nullptr,
                             bundle ? &*bundle : nullptr, plan ? &*plan : nullptr,
                             backend ? &*backend : nullptr);
    if (const auto *value = row.find("provenance"); value && !value->is_null())
      ++with_provenance;
    if (const auto *value = row.find("backend");
        value && value->is_object() &&
        value->object().find("observed")->boolean())
      ++with_backend;
    rows.push_back(std::move(row));
  }
  dif::telemetry::Object summary;
  summary.set("operations", rows.size());
  summary.set("with_recorded_provenance", with_provenance);
  summary.set("with_observed_backend", with_backend);
  document.set("summary", std::move(summary));
  document.set("operations", std::move(rows));
  if (json) {
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return 0;
  }
  for (const auto &entry : document.find("operations")->array()) {
    const auto &row = entry.object();
    std::cout << "op " << row.find("operation")->number() << " "
              << row.find("opcode")->string();
    if (const auto *prov = row.find("provenance"); prov && prov->is_object()) {
      const auto &p = prov->object();
      std::cout << "  creator=" << p.find("module")->string()
                << " block=" << p.find("block")->number()
                << " tag=" << p.find("tag")->string();
    } else {
      std::cout << "  creator=unrecorded";
    }
    if (const auto *back = row.find("backend"); back && back->is_object()) {
      const auto &b = back->object();
      if (b.find("observed")->boolean()) {
        std::cout << "  backend=";
        bool first = true;
        for (const auto &[api, count] :
             b.find("implementations")->object().members()) {
          std::cout << (first ? "" : ",") << api << "x" << count.number();
          first = false;
        }
      } else {
        std::cout << "  backend=unobserved";
      }
    }
    std::cout << "\n";
    for (const auto &weight : row.find("weights")->array()) {
      const auto &w = weight.object();
      std::cout << "    weight tensor=" << w.find("tensor")->number();
      if (const auto *name = w.find("creator_name"); name && name->is_string())
        std::cout << " creator_name=" << name->string();
      if (const auto *storage = w.find("bundle"); storage && storage->is_object())
        std::cout << " shard=" << storage->object().find("shard")->string();
      std::cout << "\n";
    }
    if (const auto *compiler = row.find("compiler"); compiler && compiler->is_object()) {
      for (const auto &transform :
           compiler->object().find("transforms")->array())
        std::cout << "    transform " << transform.object().find("encoded")->string()
                  << " (" << transform.object().find("scope")->string() << ")\n";
      for (const auto &decision :
           compiler->object().find("decisions")->array())
        std::cout << "    decision " << decision.object().find("decision")->string()
                  << ": " << decision.object().find("reason")->string() << "\n";
    }
  }
  const auto &summary_object = document.find("summary")->object();
  std::cout << "summary operations=" << summary_object.find("operations")->number()
            << " with_recorded_provenance="
            << summary_object.find("with_recorded_provenance")->number()
            << " with_observed_backend="
            << summary_object.find("with_observed_backend")->number() << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    for (int index = 2; index < argc; ++index)
      if (std::string(argv[index]) == "--source")
        return source_mode(argv[1], argc, argv);
    std::uint64_t prefetch_distance = 0U;
    std::uint64_t resident_plan_mib = 0U;
    std::uint64_t fixed_runtime_mib = 0U;
    bool assignments = false;
    bool alias_reshapes = false;
    bool json = false;
    auto residency_order =
        dif::compiler::StreamedResidencyOrder::FirstConsumer;
    for (int argument = 2; argument < argc; ++argument) {
      const std::string option = argv[argument];
      if (option == "--assignments") {
        assignments = true;
      } else if (option == "--alias-reshapes") {
        alias_reshapes = true;
      } else if (option == "--json") {
        json = true;
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
    std::optional<dif::compiler::StreamedResidencyPlan> residency;
    std::optional<dif::compiler::ReshapeAliasPlan> reshape_alias_plan;
    if (alias_reshapes)
      reshape_alias_plan.emplace(
          dif::compiler::plan_reshape_aliases(program));
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
    }
    const auto memory = dif::compiler::plan_memory(
        program, 256U, prefetch_distance, {}, resident_ids,
        reshape_alias_plan
            ? reshape_alias_plan->output_to_root_input
            : std::unordered_map<std::uint32_t, std::uint32_t>{});
    if (json) {
      auto document = dif::telemetry::make_document("program-report");
      dif::telemetry::Object header;
      header.set("path", std::filesystem::absolute(argv[1]).string());
      header.set("version", program.version);
      header.set("fingerprint", dif::hex_digest(dif::ir::fingerprint(program)));
      header.set("tensors", program.tensors.size());
      header.set("operations", program.operations.size());
      document.set("program", std::move(header));
      dif::telemetry::Array tensors;
      for (const auto &tensor : program.tensors) {
        dif::telemetry::Object entry;
        entry.set("id", tensor.id);
        entry.set("dtype", dif::ir::dtype_name(tensor.dtype));
        entry.set("roles", tensor.roles);
        dif::telemetry::Array dims;
        for (const auto dim : tensor.dims)
          dims.push_back(dim);
        entry.set("dims", std::move(dims));
        entry.set("bytes", tensor.byte_count());
        tensors.push_back(std::move(entry));
      }
      document.set("tensors", std::move(tensors));
      dif::telemetry::Array operations;
      for (std::size_t position = 0; position < program.operations.size(); ++position)
        operations.push_back(operation_row(program, position, nullptr, nullptr,
                                           nullptr, nullptr));
      document.set("operations", std::move(operations));
      dif::telemetry::Object memory_section;
      memory_section.set("slots", memory.slots.size());
      memory_section.set("planned_bytes", memory.total_bytes);
      memory_section.set("naive_bytes", memory.naive_bytes);
      document.set("memory", std::move(memory_section));
      if (residency) {
        dif::telemetry::Object section;
        section.set("selection_order", residency->selection_order);
        section.set("resident_tensors", residency->resident_tensor_ids.size());
        section.set("resident_bytes", residency->resident_constant_bytes);
        section.set("streamed_bytes", residency->streamed_constant_bytes);
        section.set("required_bytes", residency->required_bytes);
        document.set("residency", std::move(section));
      } else {
        document.set("residency", nullptr);
      }
      std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
      return 0;
    }
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
    if (reshape_alias_plan)
      std::cout << "reshape_aliases operations="
                << reshape_alias_plan->operation_ids.size()
                << " eliminated_materialization_bytes="
                << reshape_alias_plan->eliminated_materialization_bytes
                << "\n";
    if (residency)
      std::cout << "residency resident_tensors="
                << residency->resident_tensor_ids.size()
                << " resident_bytes=" << residency->resident_constant_bytes
                << " streamed_bytes=" << residency->streamed_constant_bytes
                << " memory_plan_bytes=" << residency->memory_plan_bytes
                << " fixed_runtime_bytes=" << residency->fixed_runtime_bytes
                << " required_bytes=" << residency->required_bytes << "\n";
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
