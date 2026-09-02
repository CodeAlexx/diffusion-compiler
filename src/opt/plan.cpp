#include "dif/opt/plan.hpp"

#include "dif/ir/verify.hpp"
#include "dif/build_info.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <ios>
#include <limits>
#include <sstream>
#include <span>

namespace dif::opt {
namespace {

constexpr std::string_view kPlanKind = "diffusion-compiler-optimization-plan";
constexpr int kPlanVersion = 2;

std::string encode_ids(const std::vector<std::uint32_t> &values) {
  std::string out = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      out += ',';
    out += std::to_string(values[index]);
  }
  out += ']';
  return out;
}

std::string encode_parameters(const std::vector<std::uint64_t> &values) {
  std::string out = "[";
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index != 0U)
      out += ',';
    out += std::to_string(values[index]);
  }
  out += ']';
  return out;
}

template <typename T>
std::vector<T> decode_integers(const json::Value &value, const char *label) {
  if (!value.is_array())
    fail(std::string("optimization plan ") + label + " must be an array");
  std::vector<T> result;
  for (const auto &element : value.array()) {
    const auto number = element.number();
    if (!(number >= 0.0) ||
        number > static_cast<double>(std::numeric_limits<T>::max()) ||
        number != std::floor(number))
      fail(std::string("optimization plan ") + label +
           " has a non-integral element");
    result.push_back(static_cast<T>(number));
  }
  return result;
}

const json::Value &required(const json::Value &object, const char *key) {
  const auto *found = object.find(key);
  if (!found)
    fail(std::string("optimization plan is missing \"") + key + "\"");
  return *found;
}

std::uint64_t unsigned_integer(const json::Value &value, const char *label) {
  const auto number = value.number();
  if (!(number >= 0.0) ||
      number > static_cast<double>(std::numeric_limits<std::uint64_t>::max()) ||
      number != std::floor(number))
    fail(std::string("optimization plan ") + label +
         " must be a nonnegative integer");
  return static_cast<std::uint64_t>(number);
}

RewriteContext replay_program(const OptimizationPlan &plan,
                              const RewriteContext &base) {
  const auto base_program = program_fingerprint(base.program);
  if (!plan.base_program_fingerprint.empty() &&
      plan.base_program_fingerprint != base_program)
    fail("optimization plan was recorded against program " +
         plan.base_program_fingerprint + " but was replayed against " +
         base_program);
  const auto base_candidate = candidate_fingerprint(base);
  if (!plan.base_fingerprint.empty() &&
      plan.base_fingerprint != base_candidate)
    fail("optimization plan base bindings do not match: recorded " +
         plan.base_fingerprint + ", replayed " + base_candidate);
  RewriteContext context = base;
  for (const auto &transform : plan.transforms)
    apply(transform, context);
  ir::verify(context.program);
  const auto produced = candidate_fingerprint(context);
  if (!plan.candidate_fingerprint.empty() &&
      plan.candidate_fingerprint != produced)
    fail("replayed optimization plan produced candidate " + produced +
         " but recorded " + plan.candidate_fingerprint);
  return context;
}

} // namespace

std::string json_quote(std::string_view text) {
  std::string out = "\"";
  for (const auto character : text) {
    switch (character) {
    case '"':
      out += "\\\"";
      break;
    case '\\':
      out += "\\\\";
      break;
    case '\n':
      out += "\\n";
      break;
    case '\r':
      out += "\\r";
      break;
    case '\t':
      out += "\\t";
      break;
    default:
      if (static_cast<unsigned char>(character) < 0x20U) {
        char buffer[7];
        std::snprintf(buffer, sizeof(buffer), "\\u%04x",
                      static_cast<unsigned>(static_cast<unsigned char>(character)));
        out += buffer;
      } else {
        out += character;
      }
    }
  }
  out += '"';
  return out;
}

std::string json_number(double value) {
  if (!std::isfinite(value))
    return "null";
  char buffer[40];
  std::snprintf(buffer, sizeof(buffer), "%.17g", value);
  return buffer;
}

std::string serialize_plan(const OptimizationPlan &plan) {
  std::string out = "{\n";
  out += "  \"kind\": " + json_quote(kPlanKind) + ",\n";
  out += "  \"version\": " + std::to_string(kPlanVersion) + ",\n";
  out += "  \"base_program_fingerprint\": " +
         json_quote(plan.base_program_fingerprint) + ",\n";
  out += "  \"base_fingerprint\": " + json_quote(plan.base_fingerprint) + ",\n";
  out += "  \"candidate_program_fingerprint\": " +
         json_quote(plan.candidate_program_fingerprint) + ",\n";
  out += "  \"candidate_fingerprint\": " +
         json_quote(plan.candidate_fingerprint) + ",\n";
  if (plan.compatibility) {
    const auto &compatibility = *plan.compatibility;
    out += "  \"compatibility\": {\n";
    out += "    \"compiler_revision\": " +
           json_quote(compatibility.compiler_revision) + ",\n";
    out += "    \"target_fingerprint\": " +
           json_quote(compatibility.target_fingerprint) + ",\n";
    out += "    \"runtime_budget_class\": " +
           json_quote(compatibility.runtime_budget_class) + ",\n";
    out += "    \"precision_policy\": " +
           json_quote(compatibility.precision_policy) + ",\n";
    out += "    \"minimum_usable_device_bytes\": " +
           std::to_string(compatibility.minimum_usable_device_bytes) + ",\n";
    out += "    \"required_workspace_bytes\": " +
           std::to_string(compatibility.required_workspace_bytes) + "\n";
    out += "  },\n";
  }
  out += "  \"transforms\": [";
  for (std::size_t index = 0; index < plan.transforms.size(); ++index) {
    const auto &transform = plan.transforms[index];
    out += index == 0U ? "\n" : ",\n";
    out += "    {\"kind\": " +
           json_quote(transform_kind_name(transform.kind)) +
           ", \"class\": " +
           json_quote(transform_class_name(transform_class(transform.kind))) +
           ", \"operations\": " + encode_ids(transform.operations) +
           ", \"tensors\": " + encode_ids(transform.tensors) +
           ", \"parameters\": " + encode_parameters(transform.parameters) + "}";
  }
  out += plan.transforms.empty() ? "]\n" : "\n  ]\n";
  out += "}\n";
  return out;
}

OptimizationPlan parse_plan(std::string_view text) {
  const auto document = json::parse(text);
  if (!document.is_object())
    fail("optimization plan is not a JSON object");
  if (required(document, "kind").string() != kPlanKind)
    fail("file is not a diffusion-compiler optimization plan");
  const auto version = required(document, "version").number();
  if (version != 1.0 && version != static_cast<double>(kPlanVersion))
    fail("unsupported optimization plan version");
  OptimizationPlan plan;
  plan.base_program_fingerprint =
      required(document, "base_program_fingerprint").string();
  plan.base_fingerprint = required(document, "base_fingerprint").string();
  plan.candidate_program_fingerprint =
      required(document, "candidate_program_fingerprint").string();
  plan.candidate_fingerprint =
      required(document, "candidate_fingerprint").string();
  if (const auto *compatibility = document.find("compatibility")) {
    if (!compatibility->is_object())
      fail("optimization plan compatibility must be an object");
    PlanCompatibility value;
    value.compiler_revision =
        required(*compatibility, "compiler_revision").string();
    value.target_fingerprint =
        required(*compatibility, "target_fingerprint").string();
    value.runtime_budget_class =
        required(*compatibility, "runtime_budget_class").string();
    value.precision_policy =
        required(*compatibility, "precision_policy").string();
    value.minimum_usable_device_bytes = unsigned_integer(
        required(*compatibility, "minimum_usable_device_bytes"),
        "minimum usable device bytes");
    value.required_workspace_bytes = unsigned_integer(
        required(*compatibility, "required_workspace_bytes"),
        "required workspace bytes");
    plan.compatibility = std::move(value);
  }
  const auto &transforms = required(document, "transforms");
  if (!transforms.is_array())
    fail("optimization plan transforms must be an array");
  for (const auto &entry : transforms.array()) {
    if (!entry.is_object())
      fail("optimization plan transform is not an object");
    Transform transform;
    if (!transform_kind_from_name(required(entry, "kind").string(),
                                  transform.kind))
      fail("optimization plan names an unknown transform kind");
    transform.operations =
        decode_integers<std::uint32_t>(required(entry, "operations"),
                                       "operations");
    transform.tensors =
        decode_integers<std::uint32_t>(required(entry, "tensors"), "tensors");
    transform.parameters =
        decode_integers<std::uint64_t>(required(entry, "parameters"),
                                       "parameters");
    plan.transforms.push_back(std::move(transform));
  }
  return plan;
}

void write_plan(const OptimizationPlan &plan,
                const std::filesystem::path &path) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream)
    fail("cannot open optimization plan for writing: " + path.string());
  const auto text = serialize_plan(plan);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  if (!stream)
    fail("cannot write optimization plan: " + path.string());
}

OptimizationPlan read_plan(const std::filesystem::path &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    fail("cannot open optimization plan: " + path.string());
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return parse_plan(buffer.str());
}

void bind_plan_compatibility(
    OptimizationPlan &plan, const target::TargetProfile &profile,
    const target::RuntimeBudget &budget, std::string precision_policy,
    std::uint64_t minimum_usable_device_bytes,
    std::uint64_t required_workspace_bytes) {
  if (minimum_usable_device_bytes > budget.usable_device_memory_bytes)
    fail("cannot bind a plan requiring more usable device memory than the "
         "current runtime budget");
  if (required_workspace_bytes > budget.workspace_budget_bytes)
    fail("cannot bind a plan requiring more workspace than the current "
         "runtime budget");
  plan.compatibility = PlanCompatibility{
      std::string(build::compiler_revision()),
      target::target_fingerprint(profile),
      target::runtime_budget_class(budget),
      std::move(precision_policy),
      minimum_usable_device_bytes,
      required_workspace_bytes,
  };
}

void validate_plan_compatibility(
    const OptimizationPlan &plan, const target::TargetProfile &profile,
    const target::RuntimeBudget &budget, std::string_view precision_policy) {
  if (!plan.compatibility)
    return;
  const auto &expected = *plan.compatibility;
  if (expected.compiler_revision != build::compiler_revision())
    fail("optimization plan compiler revision mismatch: recorded " +
         expected.compiler_revision + ", current " +
         std::string(build::compiler_revision()));
  const auto target = target::target_fingerprint(profile);
  if (expected.target_fingerprint != target)
    fail("optimization plan target mismatch: recorded " +
         expected.target_fingerprint + ", current " + target);
  if (expected.precision_policy != precision_policy)
    fail("optimization plan precision policy mismatch: recorded " +
         expected.precision_policy + ", current " +
         std::string(precision_policy));
  const auto budget_class = target::runtime_budget_class(budget);
  if (expected.runtime_budget_class != budget_class)
    fail("optimization plan runtime budget class mismatch: recorded " +
         expected.runtime_budget_class + ", current " + budget_class);
  if (budget.usable_device_memory_bytes <
      expected.minimum_usable_device_bytes)
    fail("optimization plan runtime budget has insufficient usable device "
         "memory");
  if (budget.workspace_budget_bytes < expected.required_workspace_bytes)
    fail("optimization plan runtime budget has insufficient workspace");
}

std::string plan_fingerprint(const OptimizationPlan &plan) {
  const auto text = serialize_plan(plan);
  const auto bytes = std::span<const std::uint8_t>(
      reinterpret_cast<const std::uint8_t *>(text.data()), text.size());
  return hex_digest(sha256(bytes));
}

RewriteContext replay(const OptimizationPlan &plan,
                      const RewriteContext &base) {
  if (plan.compatibility)
    fail("target-bound optimization plan requires a TargetProfile and "
         "RuntimeBudget for replay");
  return replay_program(plan, base);
}

RewriteContext replay(const OptimizationPlan &plan, const RewriteContext &base,
                      const target::TargetProfile &profile,
                      const target::RuntimeBudget &budget,
                      std::string_view precision_policy) {
  validate_plan_compatibility(plan, profile, budget, precision_policy);
  return replay_program(plan, base);
}

RewriteContext apply_global_strategy(const OptimizationPlan &plan,
                                     const RewriteContext &target) {
  RewriteContext context = target;
  for (auto transform : plan.transforms) {
    transform.operations.clear();
    transform.tensors.clear();
    apply(transform, context);
  }
  ir::verify(context.program);
  return context;
}

} // namespace dif::opt
