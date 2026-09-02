// difplan: inspect, compare, and explain optimization plans from the
// decisions the compiler actually recorded. It never invents a reason: when
// a plan carries no decision for a subject, that absence is reported.

#include "dif/compiler/residency_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/opt/plan.hpp"
#include "dif/opt/transform.hpp"
#include "dif/support/error.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: difplan show PLAN.difplan [--json]\n"
         "       difplan diff A.difplan B.difplan [--json]\n"
         "       difplan residency PLAN.difplan [--program FILE.difir] [--json]\n"
         "       difplan residency --program FILE.difir --budget-mib N\n"
         "                [--fixed-runtime-mib N] [--order first|largest]\n"
         "                [--prefetch-distance N] [--json]\n"
         "       difplan explain tensor ID PLAN.difplan [--program FILE.difir] [--json]\n"
         "       difplan explain op ID PLAN.difplan [--program FILE.difir] [--json]\n"
         "\n"
         "Decisions are the compiler's own recorded provenance (residency\n"
         "admission arithmetic, candidate verdicts, precision policy, target\n"
         "requirements). A subject without a recorded decision is reported as\n"
         "such, never explained by inference.\n";
}

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

dif::telemetry::Object decision_section(const dif::opt::PlanDecision &decision) {
  dif::telemetry::Object out;
  out.set("subject", decision.subject);
  out.set("id", decision.subject_id);
  out.set("decision", decision.decision);
  out.set("reason", decision.reason);
  dif::telemetry::Object evidence;
  for (const auto &[key, value] : decision.evidence)
    evidence.set(key, value);
  out.set("evidence", std::move(evidence));
  return out;
}

dif::telemetry::Object transform_section(const dif::opt::Transform &transform) {
  dif::telemetry::Object out;
  out.set("kind", dif::opt::transform_kind_name(transform.kind));
  out.set("class", dif::opt::transform_class_name(
                       dif::opt::transform_class(transform.kind)));
  out.set("changes_numerics", dif::opt::changes_numerics(transform.kind));
  dif::telemetry::Array operations;
  for (const auto id : transform.operations)
    operations.push_back(id);
  out.set("operations", std::move(operations));
  dif::telemetry::Array tensors;
  for (const auto id : transform.tensors)
    tensors.push_back(id);
  out.set("tensors", std::move(tensors));
  dif::telemetry::Array parameters;
  for (const auto value : transform.parameters)
    parameters.push_back(value);
  out.set("parameters", std::move(parameters));
  out.set("encoded", dif::opt::encode_transform(transform));
  return out;
}

dif::telemetry::Object plan_section(const dif::opt::OptimizationPlan &plan,
                                    const std::filesystem::path &path) {
  dif::telemetry::Object out;
  out.set("path", std::filesystem::absolute(path).string());
  out.set("plan_fingerprint", dif::opt::plan_fingerprint(plan));
  out.set("base_program_fingerprint", plan.base_program_fingerprint);
  out.set("base_fingerprint", plan.base_fingerprint);
  out.set("candidate_program_fingerprint", plan.candidate_program_fingerprint);
  out.set("candidate_fingerprint", plan.candidate_fingerprint);
  if (plan.compatibility) {
    dif::telemetry::Object compatibility;
    compatibility.set("compiler_revision", plan.compatibility->compiler_revision);
    compatibility.set("target_fingerprint",
                      plan.compatibility->target_fingerprint);
    compatibility.set("runtime_budget_class",
                      plan.compatibility->runtime_budget_class);
    compatibility.set("precision_policy", plan.compatibility->precision_policy);
    compatibility.set("minimum_usable_device_bytes",
                      plan.compatibility->minimum_usable_device_bytes);
    compatibility.set("required_workspace_bytes",
                      plan.compatibility->required_workspace_bytes);
    out.set("compatibility", std::move(compatibility));
  } else {
    out.set("compatibility", nullptr);
  }
  dif::telemetry::Array transforms;
  for (const auto &transform : plan.transforms)
    transforms.push_back(transform_section(transform));
  out.set("transforms", std::move(transforms));
  std::map<std::string, std::uint64_t> counts;
  for (const auto &decision : plan.decisions)
    ++counts[decision.subject + ":" + decision.decision];
  dif::telemetry::Object decision_counts;
  for (const auto &[key, count] : counts)
    decision_counts.set(key, count);
  out.set("decision_counts", std::move(decision_counts));
  out.set("decision_total", plan.decisions.size());
  return out;
}

void print_plan(const dif::telemetry::Object &section) {
  const auto text = [&](const char *key) {
    const auto *value = section.find(key);
    return value && value->is_string() ? value->string() : std::string("-");
  };
  std::cout << "plan                  " << text("path") << "\n"
            << "plan fingerprint      " << text("plan_fingerprint") << "\n"
            << "base program          " << text("base_program_fingerprint") << "\n"
            << "base candidate        " << text("base_fingerprint") << "\n"
            << "candidate program     " << text("candidate_program_fingerprint")
            << "\n"
            << "candidate             " << text("candidate_fingerprint") << "\n";
  if (const auto *compatibility = section.find("compatibility");
      compatibility && compatibility->is_object()) {
    const auto &object = compatibility->object();
    std::cout << "compiler revision     "
              << object.find("compiler_revision")->string() << "\n"
              << "target fingerprint    "
              << object.find("target_fingerprint")->string() << "\n"
              << "budget class          "
              << object.find("runtime_budget_class")->string() << "\n"
              << "precision policy      "
              << object.find("precision_policy")->string() << "\n"
              << "min usable VRAM       "
              << object.find("minimum_usable_device_bytes")->number() << "\n"
              << "required workspace    "
              << object.find("required_workspace_bytes")->number() << "\n";
  } else {
    std::cout << "compatibility         unbound (version-1 or historical plan)\n";
  }
  const auto &transforms = section.find("transforms")->array();
  std::cout << "transforms            " << transforms.size() << "\n";
  for (const auto &transform : transforms)
    std::cout << "  " << transform.object().find("encoded")->string() << "\n";
  std::cout << "decisions             "
            << section.find("decision_total")->number() << "\n";
  for (const auto &[key, count] :
       section.find("decision_counts")->object().members())
    std::cout << "  " << key << " = " << count.number() << "\n";
}

int command_show(int argc, char **argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const std::filesystem::path path = argv[2];
  bool json = false;
  for (int index = 3; index < argc; ++index)
    if (std::string(argv[index]) == "--json")
      json = true;
  const auto plan = dif::opt::read_plan(path);
  auto section = plan_section(plan, path);
  if (json) {
    auto document =
        dif::telemetry::make_document(dif::telemetry::kind::plan_report);
    document.set("report", "show");
    document.set("plan", std::move(section));
    dif::telemetry::Array decisions;
    for (const auto &decision : plan.decisions)
      decisions.push_back(decision_section(decision));
    document.set("decisions", std::move(decisions));
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return 0;
  }
  print_plan(section);
  return 0;
}

int command_diff(int argc, char **argv) {
  if (argc < 4) {
    usage();
    return 2;
  }
  const std::filesystem::path left_path = argv[2];
  const std::filesystem::path right_path = argv[3];
  bool json = false;
  for (int index = 4; index < argc; ++index)
    if (std::string(argv[index]) == "--json")
      json = true;
  const auto left = dif::opt::read_plan(left_path);
  const auto right = dif::opt::read_plan(right_path);
  dif::telemetry::Array differences;
  const auto field = [&](const char *name, const std::string &a,
                         const std::string &b) {
    if (a == b)
      return;
    dif::telemetry::Object entry;
    entry.set("field", name);
    entry.set("left", a);
    entry.set("right", b);
    differences.push_back(std::move(entry));
  };
  field("base_program_fingerprint", left.base_program_fingerprint,
        right.base_program_fingerprint);
  field("base_fingerprint", left.base_fingerprint, right.base_fingerprint);
  field("candidate_program_fingerprint", left.candidate_program_fingerprint,
        right.candidate_program_fingerprint);
  field("candidate_fingerprint", left.candidate_fingerprint,
        right.candidate_fingerprint);
  const auto compat = [](const dif::opt::OptimizationPlan &plan,
                         auto member) -> std::string {
    if (!plan.compatibility)
      return "<unbound>";
    return member(*plan.compatibility);
  };
  field("compiler_revision",
        compat(left, [](const auto &c) { return c.compiler_revision; }),
        compat(right, [](const auto &c) { return c.compiler_revision; }));
  field("target_fingerprint",
        compat(left, [](const auto &c) { return c.target_fingerprint; }),
        compat(right, [](const auto &c) { return c.target_fingerprint; }));
  field("runtime_budget_class",
        compat(left, [](const auto &c) { return c.runtime_budget_class; }),
        compat(right, [](const auto &c) { return c.runtime_budget_class; }));
  field("precision_policy",
        compat(left, [](const auto &c) { return c.precision_policy; }),
        compat(right, [](const auto &c) { return c.precision_policy; }));
  field("minimum_usable_device_bytes",
        compat(left, [](const auto &c) {
          return std::to_string(c.minimum_usable_device_bytes);
        }),
        compat(right, [](const auto &c) {
          return std::to_string(c.minimum_usable_device_bytes);
        }));
  field("required_workspace_bytes",
        compat(left, [](const auto &c) {
          return std::to_string(c.required_workspace_bytes);
        }),
        compat(right, [](const auto &c) {
          return std::to_string(c.required_workspace_bytes);
        }));
  std::vector<std::string> left_transforms;
  std::vector<std::string> right_transforms;
  for (const auto &transform : left.transforms)
    left_transforms.push_back(dif::opt::encode_transform(transform));
  for (const auto &transform : right.transforms)
    right_transforms.push_back(dif::opt::encode_transform(transform));
  dif::telemetry::Array removed;
  dif::telemetry::Array added;
  {
    std::multiset<std::string> right_set(right_transforms.begin(),
                                         right_transforms.end());
    for (const auto &encoded : left_transforms) {
      const auto found = right_set.find(encoded);
      if (found == right_set.end())
        removed.push_back(encoded);
      else
        right_set.erase(found);
    }
    for (const auto &encoded : right_set)
      added.push_back(encoded);
  }
  const bool order_changed = left_transforms != right_transforms &&
                             removed.empty() && added.empty();
  const auto key = [](const dif::opt::PlanDecision &decision) {
    return decision.subject + "#" + std::to_string(decision.subject_id);
  };
  std::map<std::string, const dif::opt::PlanDecision *> left_decisions;
  std::map<std::string, const dif::opt::PlanDecision *> right_decisions;
  for (const auto &decision : left.decisions)
    left_decisions.emplace(key(decision), &decision);
  for (const auto &decision : right.decisions)
    right_decisions.emplace(key(decision), &decision);
  dif::telemetry::Array decision_changes;
  for (const auto &[subject, decision] : left_decisions) {
    const auto found = right_decisions.find(subject);
    dif::telemetry::Object entry;
    entry.set("subject", subject);
    if (found == right_decisions.end()) {
      entry.set("change", "removed");
      entry.set("left", decision->decision);
      entry.set("right", nullptr);
    } else if (found->second->decision != decision->decision) {
      entry.set("change", "changed");
      entry.set("left", decision->decision);
      entry.set("right", found->second->decision);
    } else {
      continue;
    }
    decision_changes.push_back(std::move(entry));
  }
  for (const auto &[subject, decision] : right_decisions) {
    if (left_decisions.contains(subject))
      continue;
    dif::telemetry::Object entry;
    entry.set("subject", subject);
    entry.set("change", "added");
    entry.set("left", nullptr);
    entry.set("right", decision->decision);
    decision_changes.push_back(std::move(entry));
  }
  const bool identical = differences.empty() && removed.empty() &&
                         added.empty() && !order_changed &&
                         decision_changes.empty();
  if (json) {
    auto document =
        dif::telemetry::make_document(dif::telemetry::kind::plan_report);
    document.set("report", "diff");
    document.set("left", plan_section(left, left_path));
    document.set("right", plan_section(right, right_path));
    document.set("identical", identical);
    document.set("field_differences", std::move(differences));
    document.set("transforms_removed", std::move(removed));
    document.set("transforms_added", std::move(added));
    document.set("transform_order_changed", order_changed);
    document.set("decision_changes", std::move(decision_changes));
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return identical ? 0 : 1;
  }
  std::cout << "left   " << left_path.string() << "\n"
            << "right  " << right_path.string() << "\n";
  for (const auto &entry : differences) {
    const auto &object = entry.object();
    std::cout << "FIELD " << object.find("field")->string() << "\n  left  "
              << object.find("left")->string() << "\n  right "
              << object.find("right")->string() << "\n";
  }
  for (const auto &entry : removed)
    std::cout << "TRANSFORM removed " << entry.string() << "\n";
  for (const auto &entry : added)
    std::cout << "TRANSFORM added   " << entry.string() << "\n";
  if (order_changed)
    std::cout << "TRANSFORM order changed\n";
  for (const auto &entry : decision_changes) {
    const auto &object = entry.object();
    std::cout << "DECISION " << object.find("change")->string() << " "
              << object.find("subject")->string() << "\n";
  }
  std::cout << (identical ? "plans are identical\n" : "plans differ\n");
  return identical ? 0 : 1;
}

struct ProgramFacts {
  std::optional<dif::ir::Program> program;
  std::map<std::uint32_t, std::vector<std::uint32_t>> consumers;
  std::map<std::uint32_t, std::uint32_t> producers;

  void load(const std::filesystem::path &path) {
    program = dif::ir::read_file(path);
    for (const auto &operation : program->operations) {
      for (const auto id : operation.inputs)
        consumers[id].push_back(operation.id);
      for (const auto id : operation.outputs)
        producers[id] = operation.id;
    }
  }
};

dif::telemetry::Object tensor_facts(const ProgramFacts &facts,
                                    std::uint32_t id) {
  dif::telemetry::Object out;
  out.set("id", id);
  if (!facts.program) {
    out.set("program", nullptr);
    return out;
  }
  const auto *tensor = facts.program->tensor(id);
  if (!tensor) {
    out.set("present", false);
    return out;
  }
  out.set("present", true);
  out.set("dtype", dif::ir::dtype_name(tensor->dtype));
  dif::telemetry::Array dims;
  for (const auto dim : tensor->dims)
    dims.push_back(dim);
  out.set("dims", std::move(dims));
  out.set("bytes", tensor->byte_count());
  dif::telemetry::Array roles;
  if (tensor->has_role(dif::ir::TensorRole::Input))
    roles.push_back("input");
  if (tensor->has_role(dif::ir::TensorRole::Output))
    roles.push_back("output");
  if (tensor->has_role(dif::ir::TensorRole::Constant))
    roles.push_back("constant");
  if (tensor->has_role(dif::ir::TensorRole::Streamed))
    roles.push_back("streamed");
  if (tensor->has_role(dif::ir::TensorRole::Parameter))
    roles.push_back("parameter");
  if (tensor->has_role(dif::ir::TensorRole::OptimizerState))
    roles.push_back("optimizer_state");
  if (roles.empty())
    roles.push_back("internal");
  out.set("roles", std::move(roles));
  const auto producer = facts.producers.find(id);
  out.set("producer_operation",
          dif::telemetry::nullable_number(
              producer != facts.producers.end(),
              producer == facts.producers.end()
                  ? 0.0
                  : static_cast<double>(producer->second)));
  dif::telemetry::Array consumers;
  if (const auto found = facts.consumers.find(id);
      found != facts.consumers.end())
    for (const auto operation : found->second)
      consumers.push_back(operation);
  out.set("consumer_operations", std::move(consumers));
  return out;
}

dif::telemetry::Object operation_facts(const ProgramFacts &facts,
                                       std::uint32_t id) {
  dif::telemetry::Object out;
  out.set("id", id);
  if (!facts.program) {
    out.set("program", nullptr);
    return out;
  }
  const dif::ir::Operation *operation = nullptr;
  std::size_t position = 0U;
  for (std::size_t index = 0; index < facts.program->operations.size(); ++index)
    if (facts.program->operations[index].id == id) {
      operation = &facts.program->operations[index];
      position = index;
    }
  if (!operation) {
    out.set("present", false);
    return out;
  }
  out.set("present", true);
  out.set("opcode", dif::ir::opcode_name(operation->opcode));
  out.set("position", position);
  dif::telemetry::Array inputs;
  for (const auto input : operation->inputs)
    inputs.push_back(input);
  out.set("inputs", std::move(inputs));
  dif::telemetry::Array outputs;
  for (const auto output : operation->outputs)
    outputs.push_back(output);
  out.set("outputs", std::move(outputs));
  dif::telemetry::Array attributes;
  for (const auto &attribute : operation->attributes) {
    dif::telemetry::Object entry;
    entry.set("key", static_cast<std::uint32_t>(attribute.key));
    switch (attribute.kind) {
    case dif::ir::AttrKind::U64:
      entry.set("value", attribute.as_u64());
      break;
    case dif::ir::AttrKind::I64:
      entry.set("value", static_cast<long long>(attribute.as_i64()));
      break;
    case dif::ir::AttrKind::F64:
      entry.set("value", attribute.as_f64());
      break;
    case dif::ir::AttrKind::Bool:
      entry.set("value", attribute.as_bool());
      break;
    }
    attributes.push_back(std::move(entry));
  }
  out.set("attributes", std::move(attributes));
  if (const auto *implementation =
          operation->find(dif::ir::AttrKey::Implementation))
    out.set("implementation_attribute", implementation->as_u64());
  else
    out.set("implementation_attribute", nullptr);
  return out;
}

std::string transform_mentions(const dif::opt::Transform &transform,
                               const std::string &kind, std::uint32_t id) {
  const auto &ids = kind == "tensor" ? transform.tensors : transform.operations;
  if (std::find(ids.begin(), ids.end(), id) != ids.end())
    return "explicit";
  if (ids.empty() && transform.operations.empty() && transform.tensors.empty())
    return "whole-program";
  return "";
}

int command_explain(int argc, char **argv) {
  if (argc < 5) {
    usage();
    return 2;
  }
  const std::string kind = argv[2];
  if (kind != "tensor" && kind != "op")
    dif::fail("difplan explain accepts tensor or op");
  const auto id = static_cast<std::uint32_t>(number(argv[3], "subject id"));
  const std::filesystem::path plan_path = argv[4];
  std::filesystem::path program_path;
  bool json = false;
  for (int index = 5; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--json")
      json = true;
    else if (option == "--program" && index + 1 < argc)
      program_path = argv[++index];
    else
      dif::fail("unknown difplan option: " + option);
  }
  const auto plan = dif::opt::read_plan(plan_path);
  ProgramFacts facts;
  if (!program_path.empty())
    facts.load(program_path);
  const std::string subject = kind == "tensor" ? "tensor" : "operation";
  dif::telemetry::Array decisions;
  for (const auto &decision : plan.decisions)
    if (decision.subject == subject && decision.subject_id == id)
      decisions.push_back(decision_section(decision));
  // Candidate decisions whose transforms touch the subject explain why an
  // implementation was tried, accepted, or rejected at this site.
  dif::telemetry::Array candidate_decisions;
  for (const auto &decision : plan.decisions) {
    if (decision.subject != "candidate")
      continue;
    for (const auto &[key, value] : decision.evidence) {
      if (key != "transforms")
        continue;
      bool touches = false;
      try {
        for (const auto &transform :
             dif::opt::decode_transform_sequence(value))
          touches = touches || !transform_mentions(transform, kind, id).empty();
      } catch (const std::exception &) {
      }
      if (touches)
        candidate_decisions.push_back(decision_section(decision));
    }
  }
  dif::telemetry::Array transforms;
  for (const auto &transform : plan.transforms) {
    const auto mention = transform_mentions(transform, kind, id);
    if (mention.empty())
      continue;
    auto section = transform_section(transform);
    section.set("scope", mention);
    transforms.push_back(std::move(section));
  }
  auto document = dif::telemetry::make_document(dif::telemetry::kind::plan_report);
  document.set("report", "explain");
  document.set("subject", subject);
  document.set("id", id);
  document.set("plan", plan_section(plan, plan_path));
  document.set("facts", kind == "tensor" ? tensor_facts(facts, id)
                                         : operation_facts(facts, id));
  document.set("applied_transforms", std::move(transforms));
  document.set("decisions", std::move(decisions));
  document.set("candidate_decisions", std::move(candidate_decisions));
  const auto *recorded = document.find("decisions");
  const bool has_decision = !recorded->array().empty();
  document.set("recorded_decision", has_decision);
  if (!has_decision)
    document.set("note", "no recorded decision names this subject; the plan "
                         "cannot explain it beyond the transforms above");
  if (json) {
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return 0;
  }
  std::cout << subject << " " << id << " in " << plan_path.string() << "\n";
  if (const auto *program_facts = document.find("facts")) {
    const auto &object = program_facts->object();
    if (const auto *present = object.find("present")) {
      if (!present->boolean())
        std::cout << "  not present in the supplied program\n";
      else if (kind == "tensor")
        std::cout << "  " << object.find("dtype")->string()
                  << " bytes=" << object.find("bytes")->number()
                  << " roles=" << object.find("roles")->array().size()
                  << " consumers="
                  << object.find("consumer_operations")->array().size() << "\n";
      else
        std::cout << "  " << object.find("opcode")->string()
                  << " position=" << object.find("position")->number()
                  << " inputs=" << object.find("inputs")->array().size()
                  << " outputs=" << object.find("outputs")->array().size()
                  << "\n";
    }
  }
  for (const auto &transform : document.find("applied_transforms")->array())
    std::cout << "  transform (" << transform.object().find("scope")->string()
              << ") " << transform.object().find("encoded")->string() << "\n";
  for (const auto &decision : document.find("decisions")->array()) {
    const auto &object = decision.object();
    std::cout << "  decision " << object.find("decision")->string() << ": "
              << object.find("reason")->string() << "\n";
    for (const auto &[key, value] :
         object.find("evidence")->object().members())
      std::cout << "    " << key << " = " << value.string() << "\n";
  }
  for (const auto &decision :
       document.find("candidate_decisions")->array()) {
    const auto &object = decision.object();
    std::cout << "  candidate " << object.find("id")->number() << " "
              << object.find("decision")->string() << ": "
              << object.find("reason")->string() << "\n";
  }
  if (!has_decision)
    std::cout << "  " << document.find("note")->string() << "\n";
  return 0;
}

int command_residency(int argc, char **argv) {
  std::filesystem::path plan_path;
  std::filesystem::path program_path;
  std::uint64_t budget_mib = 0U;
  std::uint64_t fixed_runtime_mib = 0U;
  std::uint64_t prefetch_distance = 1U;
  auto order = dif::compiler::StreamedResidencyOrder::FirstConsumer;
  bool json = false;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value after " + option);
      return argv[++index];
    };
    if (option == "--json")
      json = true;
    else if (option == "--program")
      program_path = value();
    else if (option == "--budget-mib")
      budget_mib = number(value(), "budget");
    else if (option == "--fixed-runtime-mib")
      fixed_runtime_mib = number(value(), "fixed runtime");
    else if (option == "--prefetch-distance")
      prefetch_distance = number(value(), "prefetch distance");
    else if (option == "--order") {
      const auto text = value();
      if (text == "first")
        order = dif::compiler::StreamedResidencyOrder::FirstConsumer;
      else if (text == "largest")
        order = dif::compiler::StreamedResidencyOrder::LargestFirst;
      else
        dif::fail("--order accepts first or largest");
    } else if (option.rfind("--", 0U) == 0U)
      dif::fail("unknown difplan option: " + option);
    else
      plan_path = option;
  }
  auto document = dif::telemetry::make_document(dif::telemetry::kind::plan_report);
  document.set("report", "residency");
  ProgramFacts facts;
  if (!program_path.empty())
    facts.load(program_path);
  dif::telemetry::Array entries;
  std::uint64_t resident_bytes = 0U;
  std::uint64_t streamed_bytes = 0U;
  if (!plan_path.empty()) {
    const auto plan = dif::opt::read_plan(plan_path);
    document.set("plan", plan_section(plan, plan_path));
    // Residency from recorded tensor decisions first, then from explicit
    // SetConstantResidency transforms; whole-program transforms are
    // reported as policy rather than expanded without the program.
    std::map<std::uint32_t, const dif::opt::PlanDecision *> decided;
    for (const auto &decision : plan.decisions)
      if (decision.subject == "tensor" &&
          (decision.decision == "resident" || decision.decision == "streamed"))
        decided[static_cast<std::uint32_t>(decision.subject_id)] = &decision;
    std::map<std::uint32_t, std::string> by_transform;
    dif::telemetry::Array policies;
    for (const auto &transform : plan.transforms) {
      if (transform.kind != dif::opt::TransformKind::SetConstantResidency)
        continue;
      const bool streamed =
          !transform.parameters.empty() && transform.parameters[0] != 0U;
      if (transform.tensors.empty()) {
        dif::telemetry::Object policy;
        policy.set("scope", "whole-program");
        policy.set("residency", streamed ? "streamed" : "resident");
        policy.set("encoded", dif::opt::encode_transform(transform));
        policies.push_back(std::move(policy));
        continue;
      }
      for (const auto id : transform.tensors)
        by_transform[id] = streamed ? "streamed" : "resident";
    }
    document.set("whole_program_policies", std::move(policies));
    std::set<std::uint32_t> ids;
    for (const auto &[id, decision] : decided)
      ids.insert(id);
    for (const auto &[id, residency] : by_transform)
      ids.insert(id);
    for (const auto id : ids) {
      dif::telemetry::Object entry;
      entry.set("tensor", id);
      std::string residency;
      if (const auto found = decided.find(id); found != decided.end()) {
        residency = found->second->decision;
        entry.set("source", "recorded-decision");
        entry.set("reason", found->second->reason);
      } else {
        residency = by_transform.at(id);
        entry.set("source", "transform");
        entry.set("reason", nullptr);
      }
      entry.set("residency", residency);
      const auto tensor = tensor_facts(facts, id);
      if (const auto *bytes = tensor.find("bytes")) {
        const auto count = static_cast<std::uint64_t>(bytes->number());
        if (residency == "resident")
          resident_bytes += count;
        else
          streamed_bytes += count;
      }
      entry.set("facts", tensor);
      entries.push_back(std::move(entry));
    }
  } else {
    if (program_path.empty() || budget_mib == 0U)
      dif::fail("difplan residency needs a plan file, or --program with "
                "--budget-mib to run the planner");
    const auto plan = dif::compiler::plan_streamed_residency(
        *facts.program, budget_mib * 1024ULL * 1024ULL,
        fixed_runtime_mib * 1024ULL * 1024ULL, prefetch_distance, {}, order);
    dif::telemetry::Object planner;
    planner.set("program", std::filesystem::absolute(program_path).string());
    planner.set("selection_order", plan.selection_order);
    planner.set("maximum_total_bytes", plan.maximum_total_bytes);
    planner.set("fixed_runtime_bytes", plan.fixed_runtime_bytes);
    planner.set("memory_plan_bytes", plan.memory_plan_bytes);
    planner.set("resident_constant_bytes", plan.resident_constant_bytes);
    planner.set("streamed_constant_bytes", plan.streamed_constant_bytes);
    planner.set("required_bytes", plan.required_bytes);
    planner.set("resident_tensor_count", plan.resident_tensor_ids.size());
    document.set("planner", std::move(planner));
    resident_bytes = plan.resident_constant_bytes;
    streamed_bytes = plan.streamed_constant_bytes;
    for (const auto &decision : plan.decisions) {
      dif::telemetry::Object entry;
      entry.set("tensor", decision.tensor_id);
      entry.set("residency", decision.resident ? "resident" : "streamed");
      entry.set("source", "planner");
      entry.set("reason", decision.reason);
      entry.set("bytes", decision.bytes);
      entry.set("first_consumer_operation",
                decision.first_consumer_operation);
      entry.set("required_bytes_if_resident",
                decision.required_bytes_if_resident);
      entry.set("maximum_total_bytes", decision.maximum_total_bytes);
      entry.set("facts", tensor_facts(facts, decision.tensor_id));
      entries.push_back(std::move(entry));
    }
  }
  dif::telemetry::Object totals;
  totals.set("resident_bytes", resident_bytes);
  totals.set("streamed_bytes", streamed_bytes);
  totals.set("entries", entries.size());
  document.set("totals", std::move(totals));
  document.set("tensors", std::move(entries));
  if (json) {
    std::cout << dif::telemetry::serialize(dif::telemetry::Value(document));
    return 0;
  }
  if (const auto *planner = document.find("planner")) {
    const auto &object = planner->object();
    std::cout << "planner order=" << object.find("selection_order")->string()
              << " ceiling=" << object.find("maximum_total_bytes")->number()
              << " required=" << object.find("required_bytes")->number()
              << " resident=" << object.find("resident_constant_bytes")->number()
              << " streamed=" << object.find("streamed_constant_bytes")->number()
              << "\n";
  }
  for (const auto &entry : document.find("tensors")->array()) {
    const auto &object = entry.object();
    std::cout << "tensor " << object.find("tensor")->number() << " "
              << object.find("residency")->string() << " ("
              << object.find("source")->string() << ")";
    if (const auto *reason = object.find("reason"); reason && reason->is_string())
      std::cout << ": " << reason->string();
    std::cout << "\n";
  }
  std::cout << "totals resident_bytes=" << resident_bytes
            << " streamed_bytes=" << streamed_bytes << "\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "show")
      return command_show(argc, argv);
    if (command == "diff")
      return command_diff(argc, argv);
    if (command == "residency")
      return command_residency(argc, argv);
    if (command == "explain")
      return command_explain(argc, argv);
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "difplan: " << error.what() << "\n";
    return 1;
  }
}
