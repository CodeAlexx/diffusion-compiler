// difbisect: generic first-divergence finder between native captures and an
// oracle fixture. Boundaries are compared in order; the report names the last
// boundary that passed and the first that failed, and states plainly which
// operations between them were not captured. It never asserts a divergence
// at a boundary nobody observed.

#include "dif/ir/codec.hpp"
#include "dif/opt/gate.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/json.hpp"
#include "dif/support/sha256.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/vocabulary.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

void usage() {
  std::cerr
      << "usage: difbisect pairs --native FILE.safetensors --oracle FILE.safetensors\n"
         "                       --order NAME[,NAME...] [bars] [--json] [--report FILE]\n"
         "       difbisect manifest MANIFEST.json [bars] [--json] [--report FILE]\n"
         "       difbisect validate-oracle MANIFEST.json [--json]\n"
         "       difbisect program --backend cpu|cuda --program FILE.difir\n"
         "                       [--weight-bundle FILE.difbind] [--input ID=FILE ...]\n"
         "                       --oracle FILE.safetensors --map TENSOR_ID=NAME [--map ...]\n"
         "                       [--oracle-manifest FILE.json] [bars] [--json] [--report FILE]\n"
         "bars: [--min-cos F] [--max-rel-l2 F] [--min-norm-ratio F] [--max-norm-ratio F]\n"
         "      [--max-abs F]   (defaults 0.999, 0.02, 0.98, 1.02, unbounded)\n"
         "\n"
         "pairs: native and oracle files hold the same boundary names; --order gives\n"
         "the semantic order. manifest: an explicit ordered boundary list with\n"
         "native/oracle tensor specs (FILE.diftensor or FILE.safetensors::NAME).\n"
         "program: executes the program, captures the mapped tensors at their\n"
         "producers, and compares them with the oracle in program order.\n";
}

double number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtod(text.c_str(), &end);
  if (!end || *end != '\0' || !std::isfinite(value))
    dif::fail(std::string("invalid ") + label);
  return value;
}

std::uint64_t integer(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

struct Bars {
  double min_cosine{0.999};
  double max_relative_l2{0.02};
  double min_norm_ratio{0.98};
  double max_norm_ratio{1.02};
  double max_absolute{std::numeric_limits<double>::infinity()};
};

bool parse_bar(Bars &bars, const std::string &option, const std::string &value) {
  if (option == "--min-cos")
    bars.min_cosine = number(value, "minimum cosine");
  else if (option == "--max-rel-l2")
    bars.max_relative_l2 = number(value, "maximum relative L2");
  else if (option == "--min-norm-ratio")
    bars.min_norm_ratio = number(value, "minimum norm ratio");
  else if (option == "--max-norm-ratio")
    bars.max_norm_ratio = number(value, "maximum norm ratio");
  else if (option == "--max-abs")
    bars.max_absolute = number(value, "maximum absolute error");
  else
    return false;
  return true;
}

dif::telemetry::Object bars_section(const Bars &bars) {
  dif::telemetry::Object out;
  out.set("min_cosine", bars.min_cosine);
  out.set("max_relative_l2", bars.max_relative_l2);
  out.set("min_norm_ratio", bars.min_norm_ratio);
  out.set("max_norm_ratio", bars.max_norm_ratio);
  out.set("max_absolute", std::isfinite(bars.max_absolute)
                              ? dif::telemetry::Value(bars.max_absolute)
                              : dif::telemetry::Value(nullptr));
  return out;
}

struct Boundary {
  std::string name;
  // Producer position in the program when known; otherwise the index in
  // the declared order.
  std::uint64_t order{};
  std::optional<std::uint32_t> tensor_id;
  std::optional<std::uint32_t> producer_operation;
  std::optional<std::uint64_t> producer_position;
  std::string native_spec;
  std::string oracle_spec;
  dif::runtime::Tensor native;
  dif::runtime::Tensor oracle;
  // A boundary declared in the order but absent on one side is reported,
  // never compared.
  bool missing{};
  bool comparable{};
  std::string problem;
  dif::opt::NumericalMeasurement numerics;
  bool passed{};
  std::vector<std::string> failed_bars;
};

dif::runtime::Tensor load_spec(const std::string &spec) {
  const auto separator = spec.find("::");
  if (separator == std::string::npos)
    return dif::runtime::read_tensor(std::filesystem::path(spec));
  if (separator == 0U || separator + 2U == spec.size())
    dif::fail("invalid SafeTensors tensor specification: " + spec);
  const auto path = std::filesystem::path(spec.substr(0U, separator));
  const auto name = spec.substr(separator + 2U);
  return dif::weights::map_safetensor(dif::weights::read_safetensors(path),
                                      name);
}

void compare(Boundary &boundary, const Bars &bars) {
  if (boundary.missing) {
    boundary.comparable = false;
    return;
  }
  if (!dif::runtime::is_float_dtype(boundary.native.dtype) ||
      !dif::runtime::is_float_dtype(boundary.oracle.dtype)) {
    boundary.comparable = false;
    boundary.problem = "non-floating tensors are not compared";
    return;
  }
  if (boundary.native.dims != boundary.oracle.dims) {
    boundary.comparable = false;
    boundary.problem = "native and oracle shapes differ";
    return;
  }
  if (boundary.native.dtype != boundary.oracle.dtype) {
    // Measure at the oracle's declared precision by converting the native
    // capture; a dtype mismatch is reported, not hidden.
    boundary.native = dif::runtime::convert_float_tensor(boundary.native,
                                                         boundary.oracle.dtype);
    boundary.problem = "native dtype converted to the oracle dtype for comparison";
  }
  dif::runtime::TensorMap reference;
  dif::runtime::TensorMap candidate;
  reference.emplace(1U, boundary.oracle);
  candidate.emplace(1U, boundary.native);
  const dif::opt::AcceptanceGate gate(dif::opt::AcceptanceBars{});
  boundary.numerics = gate.measure(reference, candidate);
  boundary.comparable = true;
  const auto &n = boundary.numerics;
  if (n.nonfinite_count != 0U)
    boundary.failed_bars.push_back("nonfinite");
  if (!(n.cosine_similarity >= bars.min_cosine))
    boundary.failed_bars.push_back("min_cosine");
  if (!(n.relative_l2 <= bars.max_relative_l2))
    boundary.failed_bars.push_back("max_relative_l2");
  if (!(n.norm_ratio >= bars.min_norm_ratio))
    boundary.failed_bars.push_back("min_norm_ratio");
  if (!(n.norm_ratio <= bars.max_norm_ratio))
    boundary.failed_bars.push_back("max_norm_ratio");
  if (!(n.max_absolute_error <= bars.max_absolute))
    boundary.failed_bars.push_back("max_absolute");
  boundary.passed = boundary.failed_bars.empty();
}

dif::telemetry::Object boundary_section(const Boundary &boundary) {
  dif::telemetry::Object out;
  out.set("name", boundary.name);
  out.set("order", boundary.order);
  out.set("tensor_id", boundary.tensor_id
                           ? dif::telemetry::Value(*boundary.tensor_id)
                           : dif::telemetry::Value(nullptr));
  out.set("producer_operation",
          boundary.producer_operation
              ? dif::telemetry::Value(*boundary.producer_operation)
              : dif::telemetry::Value(nullptr));
  out.set("producer_position",
          boundary.producer_position
              ? dif::telemetry::Value(*boundary.producer_position)
              : dif::telemetry::Value(nullptr));
  out.set("native", boundary.native_spec);
  out.set("oracle", boundary.oracle_spec);
  out.set("observed", !boundary.missing);
  out.set("comparable", boundary.comparable);
  out.set("note", boundary.problem);
  if (boundary.comparable) {
    dif::telemetry::Object numerics;
    numerics.set("elements", boundary.numerics.compared_elements);
    numerics.set("exact_mismatches", boundary.numerics.exact_mismatch_count);
    numerics.set("nonfinite", boundary.numerics.nonfinite_count);
    numerics.set("cosine", boundary.numerics.cosine_similarity);
    numerics.set("relative_l2", boundary.numerics.relative_l2);
    numerics.set("norm_ratio", boundary.numerics.norm_ratio);
    numerics.set("max_absolute", boundary.numerics.max_absolute_error);
    out.set("numerics", std::move(numerics));
    out.set("verdict", boundary.passed ? "pass" : "fail");
    dif::telemetry::Array failed;
    for (const auto &bar : boundary.failed_bars)
      failed.push_back(bar);
    out.set("failed_bars", std::move(failed));
  } else {
    out.set("verdict", "not-comparable");
  }
  return out;
}

struct Verdict {
  std::optional<std::size_t> last_good;
  std::optional<std::size_t> first_bad;
  std::string text;
};

Verdict decide(const std::vector<Boundary> &boundaries) {
  Verdict verdict;
  for (std::size_t index = 0; index < boundaries.size(); ++index) {
    const auto &boundary = boundaries[index];
    if (!boundary.comparable)
      continue;
    if (boundary.passed) {
      if (!verdict.first_bad)
        verdict.last_good = index;
    } else if (!verdict.first_bad) {
      verdict.first_bad = index;
    }
  }
  if (!verdict.first_bad) {
    verdict.text = "no divergence observed at any captured boundary";
    if (verdict.last_good)
      verdict.text += "; last captured boundary '" +
                      boundaries[*verdict.last_good].name + "' passes";
    verdict.text += "; boundaries that were not captured remain unobserved";
    return verdict;
  }
  verdict.text = "first observed divergence at boundary '" +
                 boundaries[*verdict.first_bad].name + "'";
  if (verdict.last_good)
    verdict.text += "; last known good boundary '" +
                    boundaries[*verdict.last_good].name + "'";
  else
    verdict.text += "; no earlier captured boundary passed";
  return verdict;
}

dif::telemetry::Object write_report(const std::string &mode,
                                    const Bars &bars,
                                    std::vector<Boundary> &boundaries,
                                    const dif::ir::Program *program,
                                    dif::telemetry::Value oracle_metadata) {
  for (auto &boundary : boundaries)
    compare(boundary, bars);
  const auto verdict = decide(boundaries);
  auto document = dif::telemetry::make_document("bisect");
  document.set("mode", mode);
  document.set("bars", bars_section(bars));
  document.set("oracle_metadata", std::move(oracle_metadata));
  dif::telemetry::Array entries;
  for (const auto &boundary : boundaries)
    entries.push_back(boundary_section(boundary));
  document.set("boundaries", std::move(entries));
  dif::telemetry::Object result;
  result.set("last_known_good",
             verdict.last_good
                 ? dif::telemetry::Value(boundaries[*verdict.last_good].name)
                 : dif::telemetry::Value(nullptr));
  result.set("first_known_bad",
             verdict.first_bad
                 ? dif::telemetry::Value(boundaries[*verdict.first_bad].name)
                 : dif::telemetry::Value(nullptr));
  // The unobserved span: operations strictly between the last good producer
  // and the first bad producer that had no capture. Reported as a location
  // range, never as a located divergence.
  dif::telemetry::Object unobserved;
  if (verdict.first_bad && program && boundaries[*verdict.first_bad].producer_position) {
    const auto bad_position = *boundaries[*verdict.first_bad].producer_position;
    std::uint64_t good_position = 0U;
    bool has_good = false;
    if (verdict.last_good && boundaries[*verdict.last_good].producer_position) {
      good_position = *boundaries[*verdict.last_good].producer_position;
      has_good = true;
    }
    dif::telemetry::Array operations;
    const auto begin = has_good ? good_position + 1U : 0U;
    for (std::uint64_t position = begin; position < bad_position; ++position) {
      const auto &op = program->operations[static_cast<std::size_t>(position)];
      dif::telemetry::Object entry;
      entry.set("position", position);
      entry.set("operation", op.id);
      entry.set("opcode", dif::ir::opcode_name(op.opcode));
      operations.push_back(std::move(entry));
    }
    unobserved.set("operation_count", operations.size());
    unobserved.set("operations", std::move(operations));
    unobserved.set(
        "statement",
        "the divergence lies at or before the first bad boundary's producer "
        "and after the last good boundary's producer; operations listed here "
        "were not captured and are not individually convicted");
  } else if (verdict.first_bad) {
    const auto bad = *verdict.first_bad;
    const std::size_t good = verdict.last_good ? *verdict.last_good + 1U : 0U;
    unobserved.set("declared_boundaries_between", bad - good);
    unobserved.set("statement",
                   "boundaries between the last good and the first bad "
                   "capture that were not compared remain unobserved; without "
                   "a program the span cannot be enumerated");
  } else {
    unobserved.set("statement", "no divergence observed");
  }
  result.set("unobserved_span", std::move(unobserved));
  result.set("verdict", verdict.text);
  std::size_t compared = 0U;
  std::size_t passed = 0U;
  for (const auto &boundary : boundaries) {
    if (!boundary.comparable)
      continue;
    ++compared;
    if (boundary.passed)
      ++passed;
  }
  result.set("compared", compared);
  result.set("passed", passed);
  result.set("failed", compared - passed);
  document.set("result", std::move(result));
  document.set("status", verdict.first_bad ? "diverged" : "no-divergence-observed");
  return document;
}

std::string seconds(double value) {
  std::ostringstream out;
  out << std::setprecision(8) << value;
  return out.str();
}

void print_human(const dif::telemetry::Object &document) {
  const auto &result = document.find("result")->object();
  for (const auto &entry : document.find("boundaries")->array()) {
    const auto &object = entry.object();
    std::cout << std::left << std::setw(8)
              << object.find("verdict")->string() << " "
              << object.find("name")->string();
    if (const auto *numerics = object.find("numerics")) {
      const auto &n = numerics->object();
      std::cout << " cos=" << seconds(n.find("cosine")->number())
                << " rel_l2=" << seconds(n.find("relative_l2")->number())
                << " norm_ratio=" << seconds(n.find("norm_ratio")->number())
                << " max_abs=" << seconds(n.find("max_absolute")->number())
                << " nonfinite=" << n.find("nonfinite")->number();
    } else {
      std::cout << " " << object.find("note")->string();
    }
    std::cout << "\n";
  }
  const auto &unobserved = result.find("unobserved_span")->object();
  if (const auto *count = unobserved.find("operation_count"))
    std::cout << "unobserved operations between good and bad: "
              << count->number() << "\n";
  std::cout << "BISECT " << result.find("verdict")->string() << "\n";
}

void emit(const dif::telemetry::Object &document, bool json,
          const std::filesystem::path &report) {
  const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
  if (!report.empty()) {
    std::ofstream stream(report, std::ios::binary | std::ios::trunc);
    stream.write(text.data(), static_cast<std::streamsize>(text.size()));
  }
  if (json)
    std::cout << text;
  else
    print_human(document);
}

dif::telemetry::Value read_oracle_manifest(const std::filesystem::path &path,
                                           std::vector<std::string> *order) {
  if (path.empty())
    return dif::telemetry::Value(nullptr);
  std::ifstream stream(path, std::ios::binary);
  if (!stream)
    dif::fail("cannot open oracle manifest " + path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const auto parsed = dif::json::parse(buffer.str());
  if (order) {
    if (const auto *boundaries = parsed.find("boundaries");
        boundaries && boundaries->is_array())
      for (const auto &entry : boundaries->array())
        order->push_back(entry.is_object() ? entry.find("name")->string()
                                           : entry.string());
  }
  return dif::telemetry::from_parsed(parsed);
}

int command_pairs(int argc, char **argv) {
  std::filesystem::path native_path;
  std::filesystem::path oracle_path;
  std::filesystem::path oracle_manifest;
  std::vector<std::string> order;
  Bars bars;
  bool json = false;
  std::filesystem::path report;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value after " + option);
      return argv[++index];
    };
    if (option == "--native")
      native_path = value();
    else if (option == "--oracle")
      oracle_path = value();
    else if (option == "--oracle-manifest")
      oracle_manifest = value();
    else if (option == "--order") {
      std::stringstream names(value());
      std::string name;
      while (std::getline(names, name, ','))
        if (!name.empty())
          order.push_back(name);
    } else if (option == "--json")
      json = true;
    else if (option == "--report")
      report = value();
    else if (parse_bar(bars, option, index + 1 < argc ? argv[index + 1] : ""))
      ++index;
    else
      dif::fail("unknown difbisect option: " + option);
  }
  if (native_path.empty() || oracle_path.empty())
    dif::fail("difbisect pairs requires --native and --oracle");
  auto metadata = read_oracle_manifest(oracle_manifest,
                                       order.empty() ? &order : nullptr);
  const auto native = dif::weights::read_safetensors(native_path);
  const auto oracle = dif::weights::read_safetensors(oracle_path);
  if (order.empty())
    dif::fail("difbisect pairs requires --order (or an oracle manifest with "
              "a boundaries list); file order is not a semantic order");
  std::vector<Boundary> boundaries;
  for (std::size_t index = 0; index < order.size(); ++index) {
    Boundary boundary;
    boundary.name = order[index];
    boundary.order = index;
    boundary.native_spec = native_path.string() + "::" + boundary.name;
    boundary.oracle_spec = oracle_path.string() + "::" + boundary.name;
    if (!native.find(boundary.name)) {
      boundary.missing = true;
      boundary.problem = "native file has no tensor with this name";
    } else if (!oracle.find(boundary.name)) {
      boundary.missing = true;
      boundary.problem = "oracle file has no tensor with this name";
    } else {
      boundary.native = dif::weights::map_safetensor(native, boundary.name);
      boundary.oracle = dif::weights::map_safetensor(oracle, boundary.name);
    }
    boundaries.push_back(std::move(boundary));
  }
  auto out = write_report("pairs", bars, boundaries, nullptr, std::move(metadata));
  emit(out, json, report);
  return out.find("status")->string() == "diverged" ? 1 : 0;
}

int command_manifest(int argc, char **argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const std::filesystem::path manifest_path = argv[2];
  Bars bars;
  bool json = false;
  std::filesystem::path report;
  for (int index = 3; index < argc; ++index) {
    const std::string option = argv[index];
    if (option == "--json")
      json = true;
    else if (option == "--report" && index + 1 < argc)
      report = argv[++index];
    else if (parse_bar(bars, option, index + 1 < argc ? argv[index + 1] : ""))
      ++index;
    else
      dif::fail("unknown difbisect option: " + option);
  }
  std::ifstream stream(manifest_path, std::ios::binary);
  if (!stream)
    dif::fail("cannot open bisect manifest " + manifest_path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const auto parsed = dif::json::parse(buffer.str());
  if (!parsed.is_object() || !parsed.find("boundaries") ||
      !parsed.find("boundaries")->is_array())
    dif::fail("bisect manifest must be an object with a boundaries array");
  if (const auto *manifest_bars = parsed.find("bars");
      manifest_bars && manifest_bars->is_object()) {
    for (const auto &[key, value] : manifest_bars->object())
      if (!parse_bar(bars, "--" + key, std::to_string(value.number())))
        dif::fail("bisect manifest names an unknown bar: " + key);
  }
  std::vector<Boundary> boundaries;
  std::size_t index = 0U;
  for (const auto &entry : parsed.find("boundaries")->array()) {
    Boundary boundary;
    boundary.name = entry.find("name")->string();
    boundary.order = index++;
    boundary.native_spec = entry.find("native")->string();
    boundary.oracle_spec = entry.find("oracle")->string();
    if (const auto *operation = entry.find("operation_id"))
      boundary.producer_operation = static_cast<std::uint32_t>(operation->number());
    boundary.native = load_spec(boundary.native_spec);
    boundary.oracle = load_spec(boundary.oracle_spec);
    boundary.comparable = true;
    boundaries.push_back(std::move(boundary));
  }
  dif::telemetry::Value metadata(nullptr);
  if (const auto *oracle = parsed.find("oracle"))
    metadata = dif::telemetry::from_parsed(*oracle);
  auto out = write_report("manifest", bars, boundaries, nullptr, std::move(metadata));
  emit(out, json, report);
  return out.find("status")->string() == "diverged" ? 1 : 0;
}

std::pair<std::uint32_t, std::string> mapping(const std::string &text) {
  const auto split = text.find('=');
  if (split == std::string::npos || split == 0 || split + 1 >= text.size())
    dif::fail("--map expects TENSOR_ID=ORACLE_NAME");
  return {static_cast<std::uint32_t>(integer(text.substr(0, split), "tensor id")),
          text.substr(split + 1)};
}

int command_program(int argc, char **argv) {
  std::string backend = "cpu";
  std::filesystem::path program_path;
  std::filesystem::path weight_bundle;
  std::filesystem::path oracle_path;
  std::filesystem::path oracle_manifest;
  std::vector<std::pair<std::uint32_t, std::filesystem::path>> inputs;
  std::vector<std::pair<std::uint32_t, std::string>> maps;
  Bars bars;
  bool json = false;
  std::filesystem::path report;
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    const auto value = [&]() -> std::string {
      if (index + 1 >= argc)
        dif::fail("missing value after " + option);
      return argv[++index];
    };
    if (option == "--backend")
      backend = value();
    else if (option == "--program")
      program_path = value();
    else if (option == "--weight-bundle")
      weight_bundle = value();
    else if (option == "--oracle")
      oracle_path = value();
    else if (option == "--oracle-manifest")
      oracle_manifest = value();
    else if (option == "--input") {
      const auto text = value();
      const auto split = text.find('=');
      if (split == std::string::npos)
        dif::fail("--input expects ID=FILE");
      inputs.emplace_back(
          static_cast<std::uint32_t>(integer(text.substr(0, split), "input id")),
          text.substr(split + 1));
    } else if (option == "--map")
      maps.push_back(mapping(value()));
    else if (option == "--json")
      json = true;
    else if (option == "--report")
      report = value();
    else if (parse_bar(bars, option, index + 1 < argc ? argv[index + 1] : ""))
      ++index;
    else
      dif::fail("unknown difbisect option: " + option);
  }
  if (program_path.empty() || oracle_path.empty() || maps.empty())
    dif::fail("difbisect program requires --program, --oracle, and --map");
  const auto program = dif::ir::read_file(program_path);
  dif::runtime::TensorMap bindings;
  if (!weight_bundle.empty()) {
    const auto bundle = dif::weights::read_weight_bundle(weight_bundle);
    bindings = dif::weights::load_weight_bundle(bundle, program, false);
  }
  for (const auto &[id, path] : inputs)
    bindings.insert_or_assign(id, dif::runtime::read_tensor(path));
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 1U;
  options.minimum_free_bytes = 0U;
  std::map<std::uint32_t, std::uint64_t> producer_position;
  std::map<std::uint32_t, std::uint32_t> producer_operation;
  for (std::size_t position = 0; position < program.operations.size(); ++position)
    for (const auto id : program.operations[position].outputs) {
      producer_position[id] = position;
      producer_operation[id] = program.operations[position].id;
    }
  for (const auto &[id, name] : maps) {
    const auto *tensor = program.tensor(id);
    if (!tensor)
      dif::fail("--map names tensor " + std::to_string(id) +
                " which the program does not declare");
    if (!producer_position.contains(id))
      dif::fail("--map names tensor " + std::to_string(id) +
                " which no operation produces");
    if (!tensor->has_role(dif::ir::TensorRole::Output))
      options.capture_intermediate_tensors.push_back(id);
  }
  std::unique_ptr<dif::runtime::Executor> executor;
  if (backend == "cpu")
    executor = dif::runtime::make_cpu_executor();
  else if (backend == "cuda")
    executor = dif::runtime::make_cuda_executor();
  else
    dif::fail("difbisect backend accepts cpu or cuda");
  auto prepared = executor->prepare(program, bindings, options);
  const auto result = prepared->run(bindings, options);
  const auto oracle = dif::weights::read_safetensors(oracle_path);
  std::vector<std::string> manifest_order;
  auto metadata = read_oracle_manifest(oracle_manifest, &manifest_order);
  std::vector<Boundary> boundaries;
  for (const auto &[id, name] : maps) {
    Boundary boundary;
    boundary.name = name;
    boundary.tensor_id = id;
    boundary.producer_operation = producer_operation.at(id);
    boundary.producer_position = producer_position.at(id);
    boundary.order = *boundary.producer_position;
    boundary.native_spec = "captured tensor " + std::to_string(id);
    boundary.oracle_spec = oracle_path.string() + "::" + name;
    const auto captured = result.captured_intermediates.find(id);
    const auto output = result.outputs.find(id);
    if (captured != result.captured_intermediates.end())
      boundary.native = captured->second;
    else if (output != result.outputs.end())
      boundary.native = output->second;
    else
      dif::fail("runtime did not capture tensor " + std::to_string(id));
    if (!oracle.find(name))
      dif::fail("oracle file has no tensor named " + name);
    boundary.oracle = dif::weights::map_safetensor(oracle, name);
    boundary.comparable = true;
    boundaries.push_back(std::move(boundary));
  }
  std::sort(boundaries.begin(), boundaries.end(),
            [](const Boundary &left, const Boundary &right) {
              return left.order < right.order;
            });
  auto out = write_report("program", bars, boundaries, &program, std::move(metadata));
  dif::telemetry::Object execution;
  execution.set("program", std::filesystem::absolute(program_path).string());
  execution.set("fingerprint", dif::hex_digest(dif::ir::fingerprint(program)));
  execution.set("backend", result.backend_name);
  execution.set("device", result.device_name);
  execution.set("operations", program.operations.size());
  execution.set("captured_boundaries", boundaries.size());
  out.set("execution", std::move(execution));
  emit(out, json, report);
  return out.find("status")->string() == "diverged" ? 1 : 0;
}

} // namespace

// --- oracle fixture protocol ------------------------------------------------
// Per-model oracle scripts emit a manifest beside their safetensors payload;
// this validates the standard fields and the payload identity so a bisect
// consumes a fixture whose provenance is stated, not assumed.

int command_validate_oracle(int argc, char **argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const std::filesystem::path manifest_path = argv[2];
  bool json = false;
  for (int index = 3; index < argc; ++index)
    if (std::string(argv[index]) == "--json")
      json = true;
  std::ifstream stream(manifest_path, std::ios::binary);
  if (!stream)
    dif::fail("cannot open oracle manifest " + manifest_path.string());
  std::stringstream buffer;
  buffer << stream.rdbuf();
  const auto parsed = dif::json::parse(buffer.str());
  auto document = dif::telemetry::make_document("oracle-fixture-report");
  document.set("manifest", std::filesystem::absolute(manifest_path).string());
  dif::telemetry::Array checks;
  bool valid = true;
  const auto record = [&](const std::string &name, bool ok, const std::string &detail) {
    dif::telemetry::Object entry;
    entry.set("check", name);
    entry.set("ok", ok);
    entry.set("detail", detail);
    checks.push_back(std::move(entry));
    valid = valid && ok;
  };
  const auto string_field = [&](const dif::json::Value &object, const char *key,
                                const std::string &label) -> std::string {
    const auto *value = object.find(key);
    if (!value || !std::holds_alternative<std::string>(value->storage) ||
        value->string().empty()) {
      record(label, false, std::string("missing or empty '") + key + "'");
      return {};
    }
    record(label, true, value->string());
    return value->string();
  };
  if (!parsed.is_object()) {
    record("manifest", false, "not a JSON object");
  } else {
    const auto *kind = parsed.find("kind");
    record("kind", kind && std::holds_alternative<std::string>(kind->storage) &&
                       kind->string() == "diffusion-compiler-oracle-fixture",
           "kind must be diffusion-compiler-oracle-fixture");
    const auto *version = parsed.find("version");
    record("version", version && std::holds_alternative<double>(version->storage) &&
                          version->number() == 1.0,
           "version must be 1");
    if (const auto *creator = parsed.find("creator"); creator && creator->is_object()) {
      string_field(*creator, "repository", "creator.repository");
      string_field(*creator, "revision", "creator.revision");
    } else {
      record("creator", false, "missing 'creator' object");
    }
    if (const auto *model = parsed.find("model"); model && model->is_object())
      string_field(*model, "name", "model.name");
    else
      record("model", false, "missing 'model' object");
    string_field(parsed, "semantic_boundary", "semantic_boundary");
    string_field(parsed, "dtype", "dtype");
    string_field(parsed, "fixture_version", "fixture_version");
    if (const auto *inputs = parsed.find("inputs"); inputs && inputs->is_array()) {
      for (const auto &input : inputs->array()) {
        const auto *name = input.find("name");
        const auto *sha = input.find("sha256");
        record("input", name && sha && std::holds_alternative<std::string>(sha->storage) &&
                            sha->string().size() == 64U,
               name ? "input " + name->string() : "input without a name");
      }
    } else {
      record("inputs", false, "missing 'inputs' array");
    }
    std::optional<dif::weights::SafeTensorFile> payload_file;
    if (const auto *payload = parsed.find("payload"); payload && payload->is_object()) {
      const auto path_text = string_field(*payload, "path", "payload.path");
      const auto expected = string_field(*payload, "sha256", "payload.sha256");
      if (!path_text.empty()) {
        auto path = std::filesystem::path(path_text);
        if (!path.is_absolute())
          path = manifest_path.parent_path() / path;
        std::error_code error;
        if (!std::filesystem::is_regular_file(path, error)) {
          record("payload.exists", false, path.string());
        } else {
          record("payload.exists", true, path.string());
          const auto actual = dif::hex_digest(dif::sha256_file(path));
          record("payload.sha256_matches", actual == expected,
                 actual == expected ? actual : "actual " + actual);
          try {
            payload_file.emplace(dif::weights::read_safetensors(path));
          } catch (const std::exception &error) {
            record("payload.safetensors", false, error.what());
          }
        }
      }
    } else {
      record("payload", false, "missing 'payload' object");
    }
    if (const auto *boundaries = parsed.find("boundaries");
        boundaries && boundaries->is_array() && !boundaries->array().empty()) {
      for (const auto &boundary : boundaries->array()) {
        const auto *name = boundary.find("name");
        const auto *tensor = boundary.find("tensor");
        if (!name || !tensor) {
          record("boundary", false, "boundary without name or tensor");
          continue;
        }
        if (!payload_file) {
          record("boundary " + name->string(), false, "payload not readable");
          continue;
        }
        const auto *entry = payload_file->find(tensor->string());
        if (!entry) {
          record("boundary " + name->string(), false,
                 "payload has no tensor " + tensor->string());
          continue;
        }
        bool shape_ok = true;
        if (const auto *shape = boundary.find("shape"); shape && shape->is_array()) {
          std::vector<std::uint64_t> dims;
          for (const auto &dim : shape->array())
            dims.push_back(static_cast<std::uint64_t>(dim.number()));
          shape_ok = dims == entry->dims;
        }
        bool dtype_ok = true;
        if (const auto *dtype = boundary.find("dtype");
            dtype && std::holds_alternative<std::string>(dtype->storage))
          dtype_ok = dtype->string() == dif::ir::dtype_name(entry->dtype);
        record("boundary " + name->string(), shape_ok && dtype_ok,
               shape_ok && dtype_ok ? "tensor " + tensor->string() + " present with declared shape and dtype"
                                    : "declared shape or dtype differs from the payload");
        // Harness validity (Flame lesson: a reference generator with an
        // all-zero conditioning input faked a 4.2% divergence): an oracle
        // boundary must be finite and non-degenerate before it can convict.
        if (shape_ok && dtype_ok && payload_file->mapping &&
            (entry->dtype == dif::ir::DType::F32 ||
             entry->dtype == dif::ir::DType::BF16 ||
             entry->dtype == dif::ir::DType::F16)) {
          const auto *base = payload_file->mapping->data() + entry->file_offset;
          std::vector<std::uint8_t> bytes(base, base + entry->byte_count);
          dif::runtime::Tensor values(entry->dtype, entry->dims, std::move(bytes));
          const auto count = values.element_count();
          std::uint64_t nonfinite = 0U;
          bool constant = count > 0U;
          const float first = count > 0U ? dif::runtime::load_float(values, 0U) : 0.0F;
          for (std::uint64_t index = 0U; index < count; ++index) {
            const auto value = dif::runtime::load_float(values, index);
            if (!std::isfinite(value))
              ++nonfinite;
            else if (value != first)
              constant = false;
          }
          const bool degenerate = nonfinite != 0U || constant;
          record("boundary " + name->string() + " validity", !degenerate,
                 nonfinite != 0U ? std::to_string(nonfinite) + " non-finite values"
                 : constant       ? "constant tensor (every value " +
                                        std::to_string(first) + "); a degenerate oracle cannot convict"
                                  : "finite and non-constant");
        }
      }
    } else {
      record("boundaries", false, "missing or empty 'boundaries' array");
    }
  }
  document.set("checks", std::move(checks));
  document.set("valid", valid);
  document.set("oracle", dif::telemetry::from_parsed(parsed));
  const auto text = dif::telemetry::serialize(dif::telemetry::Value(document));
  if (json) {
    std::cout << text;
  } else {
    for (const auto &entry : document.find("checks")->array())
      std::cout << (entry.object().find("ok")->boolean() ? "ok    " : "FAIL  ")
                << entry.object().find("check")->string() << ": "
                << entry.object().find("detail")->string() << "\n";
    std::cout << "ORACLE_FIXTURE " << (valid ? "VALID" : "INVALID") << "\n";
  }
  return valid ? 0 : 1;
}

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      usage();
      return 2;
    }
    const std::string command = argv[1];
    if (command == "validate-oracle")
      return command_validate_oracle(argc, argv);
    if (command == "pairs")
      return command_pairs(argc, argv);
    if (command == "manifest")
      return command_manifest(argc, argv);
    if (command == "program")
      return command_program(argc, argv);
    usage();
    return 2;
  } catch (const std::exception &error) {
    std::cerr << "difbisect: " << error.what() << "\n";
    return 1;
  }
}
