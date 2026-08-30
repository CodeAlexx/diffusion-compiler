#include "dif/opt/plan.hpp"

#include "dif/ir/verify.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <ios>
#include <limits>
#include <sstream>

namespace dif::opt {
namespace {

constexpr std::string_view kPlanKind = "diffusion-compiler-optimization-plan";
constexpr int kPlanVersion = 1;

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
  if (version != static_cast<double>(kPlanVersion))
    fail("unsupported optimization plan version");
  OptimizationPlan plan;
  plan.base_program_fingerprint =
      required(document, "base_program_fingerprint").string();
  plan.base_fingerprint = required(document, "base_fingerprint").string();
  plan.candidate_program_fingerprint =
      required(document, "candidate_program_fingerprint").string();
  plan.candidate_fingerprint =
      required(document, "candidate_fingerprint").string();
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

RewriteContext replay(const OptimizationPlan &plan,
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
