// Phase-B truth-surface gate: the ordered telemetry document model, runtime
// trace events on the reference executor, the environment trace sink, plan
// decision provenance and residency-planner decisions, and the difbench /
// diftrace / difplan command lines driven on synthetic fixtures. Nothing here
// measures a model; the subject is that the surfaces exist, agree on one
// vocabulary, and report only what was recorded.

#include "dif/compiler/memory_plan.hpp"
#include "dif/compiler/residency_plan.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/plan.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/json.hpp"
#include "dif/support/png.hpp"
#include "dif/telemetry/document.hpp"
#include "dif/telemetry/schema.hpp"
#include "dif/telemetry/trace_sink.hpp"
#include "dif/telemetry/vocabulary.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <vector>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

std::filesystem::path workspace() {
  static const auto root =
      std::filesystem::temp_directory_path() / "dif_telemetry_tests";
  return root;
}

std::string quote(const std::string &value) { return "'" + value + "'"; }

struct Outcome {
  int exit_code{};
  std::string output;
};

Outcome run(const std::string &program, const std::vector<std::string> &arguments) {
  const auto log = workspace() / "command.log";
  std::string command = quote(program);
  for (const auto &argument : arguments)
    command += " " + quote(argument);
  command += " > " + quote(log.string()) + " 2>" +
             quote((workspace() / "command.err").string());
  const auto status = std::system(command.c_str());
  Outcome outcome;
  outcome.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  std::ifstream input(log);
  std::stringstream buffer;
  buffer << input.rdbuf();
  outcome.output = buffer.str();
  return outcome;
}

const dif::json::Value &required(const dif::json::Value &object,
                                 const char *key) {
  const auto *value = object.find(key);
  if (!value)
    throw std::runtime_error(std::string("missing JSON field ") + key);
  return *value;
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

// x + w1 -> t1 ; t1 + w2 -> t2 ; t2 + w3 -> y, with three streamed constants
// so the residency planner's slot sharing can reject an admission.
constexpr std::size_t kTinyOperations = 3U;

dif::ir::Program tiny_program() {
  dif::ir::Program program;
  const auto tensor = [&](std::uint32_t id, std::uint32_t roles) {
    dif::ir::TensorDesc desc;
    desc.id = id;
    desc.dtype = dif::ir::DType::F32;
    desc.roles = roles;
    desc.dims = {8U};
    program.tensors.push_back(desc);
  };
  using dif::ir::TensorRole;
  tensor(1U, TensorRole::Input);
  tensor(2U, TensorRole::Constant | TensorRole::Streamed);
  tensor(3U, TensorRole::Internal);
  tensor(4U, TensorRole::Constant | TensorRole::Streamed);
  tensor(5U, TensorRole::Internal);
  tensor(6U, TensorRole::Constant | TensorRole::Streamed);
  tensor(7U, TensorRole::Output);
  const auto add = [](std::uint32_t id, std::uint32_t left, std::uint32_t right,
                      std::uint32_t out) {
    dif::ir::Operation operation;
    operation.id = id;
    operation.opcode = dif::ir::Opcode::Add;
    operation.inputs = {left, right};
    operation.outputs = {out};
    return operation;
  };
  program.operations = {add(1U, 1U, 2U, 3U), add(2U, 3U, 4U, 5U),
                        add(3U, 5U, 6U, 7U)};
  dif::ir::verify(program);
  return program;
}

dif::runtime::Tensor filled(float value) {
  std::vector<std::uint8_t> bytes(8U * sizeof(float));
  for (std::size_t index = 0; index < 8U; ++index)
    std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
  return dif::runtime::Tensor(dif::ir::DType::F32, {8U}, std::move(bytes));
}

void test_document_model() {
  dif::telemetry::Object object;
  object.set("zulu", 1);
  object.set("alpha", "text");
  object.set("bytes", 18446744073709551615ULL);
  object.set("ratio", 0.5);
  object.set("flag", true);
  object.set("nothing", nullptr);
  dif::telemetry::Array list;
  list.push_back(1);
  list.push_back(2);
  object.set("list", list);
  object.set("zulu", 2);
  const auto text = dif::telemetry::serialize(dif::telemetry::Value(object));
  expect(text.find("\"zulu\"") < text.find("\"alpha\""),
         "document members keep insertion order");
  expect(text.find("\"zulu\": 2") != std::string::npos,
         "set() replaces an existing member in place");
  expect(text.find("18446744073709551615") != std::string::npos,
         "unsigned 64-bit values serialize exactly");
  expect(text.find("[1, 2]") != std::string::npos,
         "scalar arrays serialize on one line");
  const auto compact =
      dif::telemetry::serialize_compact(dif::telemetry::Value(object));
  expect(compact.find('\n') == std::string::npos,
         "compact serialization is a single line");
  const auto parsed = dif::json::parse(text);
  expect(required(parsed, "alpha").string() == "text",
         "serialized document parses back");
  const auto restored = dif::telemetry::from_parsed(parsed);
  expect(restored.object().find("list")->array().size() == 2U,
         "from_parsed restores arrays");
  expect(dif::telemetry::number_text(0.1) == "0.10000000000000001",
         "doubles serialize with round-trip precision");
  auto document = dif::telemetry::make_document("unit-test");
  const auto head = dif::json::parse(
      dif::telemetry::serialize(dif::telemetry::Value(document)));
  expect(required(required(head, "schema"), "name").string() ==
             dif::telemetry::kSchemaName,
         "make_document carries the shared schema name");
  expect(required(head, "kind").string() == "unit-test",
         "make_document carries the kind");
  expect(required(required(head, "provenance"), "revision").is_object() == false,
         "provenance revision is a scalar");
}

void test_runtime_trace_cpu() {
  const auto program = tiny_program();
  dif::runtime::TensorMap bindings;
  bindings.emplace(1U, filled(1.0F));
  bindings.emplace(2U, filled(2.0F));
  bindings.emplace(4U, filled(3.0F));
  bindings.emplace(6U, filled(4.0F));
  auto executor = dif::runtime::make_cpu_executor();
  dif::runtime::RunOptions options;
  options.warmups = 0U;
  options.iterations = 2U;
  options.trace_events = true;
  auto prepared = executor->prepare(program, bindings, options);
  const auto result = prepared->run(bindings, options);
  expect(result.trace_events.size() == kTinyOperations * 2U,
         "CPU executor records one operation event per operation per iteration");
  expect(!result.trace_events.empty() &&
             result.trace_events.front().category ==
                 dif::telemetry::category::operation,
         "CPU trace events use the shared operation category");
  expect(!result.trace_events.empty() &&
             result.trace_events.front().opcode ==
                 dif::ir::opcode_name(dif::ir::Opcode::Add),
         "CPU trace events name the opcode");
  expect(result.trace_milliseconds >= 0.0, "trace wall is recorded");
  const auto attribution =
      dif::telemetry::trace_attribution_section(result.trace_events);
  expect(attribution.find("operation") != nullptr &&
             attribution.find("operation")->object().find("count")->number() ==
                 static_cast<double>(kTinyOperations * 2U),
         "attribution rolls events up by category");
  const auto document = dif::telemetry::runtime_trace_document(
      result, "fingerprint", program.operations.size(), options);
  const auto parsed = dif::json::parse(
      dif::telemetry::serialize(dif::telemetry::Value(document)));
  expect(required(parsed, "kind").string() == dif::telemetry::kind::runtime_trace,
         "runtime trace document kind");
  expect(required(required(parsed, "trace"), "run_events").array().size() ==
             kTinyOperations * 2U,
         "runtime trace document lists the events");

  // Environment sink: any tool, no flags.
  const auto sink = workspace() / "sink.jsonl";
  std::filesystem::remove(sink);
  setenv(dif::telemetry::kTraceFileVariable, sink.c_str(), 1);
  dif::runtime::RunOptions plain;
  plain.warmups = 0U;
  plain.iterations = 1U;
  expect(dif::telemetry::trace_events_requested(plain),
         "DIF_TRACE_FILE requests trace events without a flag");
  auto prepared_plain = executor->prepare(program, bindings, plain);
  (void)prepared_plain->run(bindings, plain);
  (void)prepared_plain->run(bindings, plain);
  unsetenv(dif::telemetry::kTraceFileVariable);
  std::ifstream stream(sink);
  std::string line;
  std::size_t lines = 0U;
  bool parsed_ok = true;
  while (std::getline(stream, line)) {
    if (line.empty())
      continue;
    ++lines;
    try {
      const auto entry = dif::json::parse(line);
      parsed_ok = parsed_ok && required(entry, "kind").string() ==
                                   dif::telemetry::kind::runtime_trace;
    } catch (const std::exception &) {
      parsed_ok = false;
    }
  }
  expect(lines == 2U, "trace sink appends one document per run()");
  expect(parsed_ok, "trace sink lines are runtime-trace documents");
  expect(!dif::telemetry::trace_events_requested(plain),
         "without DIF_TRACE_FILE nothing is requested");
}

void test_plan_decisions() {
  dif::opt::OptimizationPlan plan;
  plan.base_program_fingerprint = "a";
  plan.base_fingerprint = "b";
  plan.candidate_program_fingerprint = "c";
  plan.candidate_fingerprint = "d";
  dif::opt::Transform residency;
  residency.kind = dif::opt::TransformKind::SetConstantResidency;
  residency.tensors = {2U};
  residency.parameters = {0U};
  plan.transforms.push_back(residency);
  const auto identity = dif::opt::plan_fingerprint(plan);
  dif::opt::PlanDecision decision;
  decision.subject = "tensor";
  decision.subject_id = 2U;
  decision.decision = "resident";
  decision.reason = "admitted as resident: complete plan needs 100 of 200 bytes";
  decision.evidence.emplace_back("bytes", "32");
  decision.evidence.emplace_back("ratio", "0.5");
  decision.evidence.emplace_back("order", "largest_first");
  plan.decisions.push_back(decision);
  expect(dif::opt::plan_fingerprint(plan) == identity,
         "decisions do not change plan identity");
  const auto text = dif::opt::serialize_plan(plan);
  expect(text.find("\"decisions\"") != std::string::npos,
         "decisions serialize into the plan");
  expect(text.find("\"bytes\": 32") != std::string::npos,
         "numeric evidence serializes as a JSON number");
  expect(text.find("\"order\": \"largest_first\"") != std::string::npos,
         "text evidence serializes quoted");
  const auto restored = dif::opt::parse_plan(text);
  expect(restored.decisions.size() == 1U, "decisions parse back");
  {
    // The parser keys objects alphabetically, so look evidence up by name.
    std::string bytes;
    std::string order;
    std::string ratio;
    if (restored.decisions.size() == 1U)
      for (const auto &[key, value] : restored.decisions.front().evidence) {
        if (key == "bytes")
          bytes = value;
        else if (key == "order")
          order = value;
        else if (key == "ratio")
          ratio = value;
      }
    expect(bytes == "32" && order == "largest_first" && ratio == "0.5",
           "decision evidence round-trips");
  }
  expect(dif::opt::plan_fingerprint(restored) == identity,
         "restored plan keeps its identity");
  const auto legacy = dif::opt::parse_plan(
      "{\"kind\":\"diffusion-compiler-optimization-plan\",\"version\":2,"
      "\"base_program_fingerprint\":\"a\",\"base_fingerprint\":\"b\","
      "\"candidate_program_fingerprint\":\"c\",\"candidate_fingerprint\":\"d\","
      "\"transforms\":[]}");
  expect(legacy.decisions.empty(), "plans without decisions still parse");
}

void test_residency_decisions() {
  const auto program = tiny_program();
  const auto baseline = dif::compiler::plan_streamed_residency(
      program, 1ULL << 30U, 0U, 1U, {},
      dif::compiler::StreamedResidencyOrder::LargestFirst);
  expect(baseline.decisions.size() == kTinyOperations,
         "residency planner records one decision per streamed constant");
  expect(baseline.selection_order == "largest_first",
         "residency plan records its selection order");
  expect(baseline.resident_tensor_ids.size() == kTinyOperations,
         "generous ceiling admits every constant");
  for (const auto &decision : baseline.decisions)
    expect(decision.resident && !decision.reason.empty() &&
               decision.required_bytes_if_resident <= decision.maximum_total_bytes,
           "admitted decisions carry the admission arithmetic");
  // The minimum legal ceiling is the unmodified plan. Constants whose
  // streaming slot is shared with another live constant cannot be admitted
  // without growing the plan, so at least one must be recorded as rejected.
  const auto unmodified = dif::compiler::plan_memory(program, 256U, 1U);
  const auto tight = dif::compiler::plan_streamed_residency(
      program, unmodified.total_bytes, 0U, 1U, {},
      dif::compiler::StreamedResidencyOrder::FirstConsumer);
  expect(tight.decisions.size() == kTinyOperations,
         "tight plan still records every constant");
  std::size_t rejected = 0U;
  std::size_t admitted = 0U;
  for (const auto &decision : tight.decisions) {
    if (decision.resident) {
      ++admitted;
      expect(decision.reason.find("admitted") != std::string::npos,
             "admitted decision says so");
    } else {
      ++rejected;
      expect(decision.reason.find("stays streamed") != std::string::npos,
             "rejected decision explains the ceiling");
      expect(decision.required_bytes_if_resident > decision.maximum_total_bytes,
             "rejected decision records the arithmetic that rejected it");
    }
  }
  expect(admitted == tight.resident_tensor_ids.size(),
         "admitted decisions match the resident set");
  expect(rejected >= 1U, "tight ceiling rejects at least one constant");
}

std::filesystem::path write_fixture_png() {
  const auto path = workspace() / "fixture.png";
  std::vector<std::uint8_t> rgb(4U * 3U * 3U, 0x40);
  dif::write_png_rgb8(path, 4U, 3U, rgb);
  return path;
}

void test_difbench_cli() {
  const auto fixture = write_fixture_png();
  const auto missing_model = workspace() / "missing-model.safetensors";
  std::filesystem::remove(missing_model);
  const auto recipe = workspace() / "image.json";
  write_text(recipe, R"({
  "kind": "diffusion-compiler-benchmark-recipe",
  "version": 1,
  "name": "synthetic-image",
  "output_kind": "image",
  "description": "synthetic fixture chain",
  "workload": {"model_family": "synthetic", "geometry": "4x3"},
  "comparator": {"name": "synthetic comparator", "wall_seconds": 10.0, "target_ratio": 2.0},
  "variables": {"FIXTURE": ")" + fixture.string() + R"("},
  "prompt": {"file": ")" + fixture.string() + R"("},
  "model_files": [")" + fixture.string() + R"("],
  "stages": [
    {"name": "prepare", "argv": ["sh", "-c", "sleep 0.05"], "after": []},
    {"name": "side", "argv": ["sh", "-c", "true"], "after": []},
    {"name": "write", "argv": ["cp", "${FIXTURE}", "${output}"], "after": ["prepare", "side"]}
  ],
  "output": "${workdir}/out.png"
})");
  const auto workdir = workspace() / "bench";
  std::filesystem::remove_all(workdir);
  const auto outcome =
      run(DIF_DIFBENCH_PATH, {"run", recipe.string(), "--workdir",
                              workdir.string(), "--json", "--no-ffprobe",
                              "--cooldown-seconds", "0", "--repeat", "2"});
  expect(outcome.exit_code == 0, "difbench run completes on the synthetic recipe");
  try {
    const auto document = dif::json::parse(outcome.output);
    expect(required(document, "kind").string() == dif::telemetry::kind::benchmark,
           "difbench emits the benchmark kind");
    expect(required(document, "status").string() == "completed",
           "difbench status completed");
    const auto &runs = required(document, "runs").array();
    expect(runs.size() == 2U, "difbench records every repetition");
    const auto &first = runs.at(0);
    expect(required(first, "status").string() == "completed",
           "run status completed");
    expect(required(first, "wall_seconds").number() >= 0.05,
           "complete wall covers the whole chain");
    const auto &stages = required(first, "stages").array();
    expect(stages.size() == 3U, "every stage is recorded");
    expect(required(stages.at(2), "start_offset_seconds").number() >=
               required(stages.at(0), "wall_seconds").number(),
           "dependent stage starts after its dependency exits");
    const auto &output = required(first, "output");
    expect(required(output, "format").string() == "png", "PNG output detected");
    expect(required(required(output, "image"), "width").number() == 4.0 &&
               required(required(output, "image"), "height").number() == 3.0,
           "PNG geometry parsed");
    expect(required(required(first, "conditions"), "process").string() ==
               "fresh",
           "process condition is fresh");
    expect(required(required(first, "filesystem_before"), "condition").is_object() == false,
           "filesystem condition is reported");
    const auto &summary = required(document, "summary");
    expect(required(summary, "completed_runs").number() == 2.0,
           "summary counts completed runs");
    const auto &comparator = required(summary, "comparator");
    expect(required(comparator, "ratio_minimum").number() > 1.0,
           "comparator ratio computed against the minimum wall");
    expect(required(comparator, "meets_target").boolean(),
           "synthetic run meets the synthetic target");
    expect(required(document, "hardware").is_object(),
           "hardware section embedded");
    expect(std::filesystem::exists(workdir / "difbench.json"),
           "difbench writes its report into the work directory");
  } catch (const std::exception &error) {
    expect(false, std::string("difbench JSON: ") + error.what());
  }

  // Preflight refusal: a declared model file that does not exist.
  const auto refused_recipe = workspace() / "refused.json";
  write_text(refused_recipe, R"({
  "kind": "diffusion-compiler-benchmark-recipe",
  "version": 1,
  "name": "refused",
  "output_kind": "image",
  "model_files": [")" + missing_model.string() + R"("],
  "stages": [{"name": "write", "argv": ["cp", ")" + fixture.string() +
                                R"(", "${output}"]}],
  "output": "${workdir}/out.png"
})");
  const auto refused_dir = workspace() / "refused";
  std::filesystem::remove_all(refused_dir);
  const auto refused = run(DIF_DIFBENCH_PATH,
                           {"run", refused_recipe.string(), "--workdir",
                            refused_dir.string(), "--json", "--no-ffprobe"});
  expect(refused.exit_code == 2, "difbench refuses a recipe with a missing model file");
  try {
    const auto document = dif::json::parse(refused.output);
    expect(required(document, "status").string() == "refused",
           "refusal is stated in the document");
    expect(!required(document, "preflight").array().empty(),
           "preflight problems are listed");
  } catch (const std::exception &error) {
    expect(false, std::string("difbench refusal JSON: ") + error.what());
  }

  const auto inspect = run(DIF_DIFBENCH_PATH,
                           {"inspect", fixture.string(), "--json", "--no-ffprobe"});
  expect(inspect.exit_code == 0, "difbench inspect reads the PNG");
}

void test_diftrace_cli() {
  const auto program = tiny_program();
  const auto program_path = workspace() / "tiny.difir";
  dif::ir::write_file(program, program_path);
  dif::runtime::write_tensor(filled(1.0F), workspace() / "x.diftensor");
  dif::runtime::write_tensor(filled(2.0F), workspace() / "w1.diftensor");
  dif::runtime::write_tensor(filled(3.0F), workspace() / "w2.diftensor");
  dif::runtime::write_tensor(filled(4.0F), workspace() / "w3.diftensor");
  const auto fixture = write_fixture_png();
  const auto recipe = workspace() / "trace.json";
  write_text(recipe, R"({
  "kind": "diffusion-compiler-benchmark-recipe",
  "version": 1,
  "name": "synthetic-trace",
  "output_kind": "image",
  "stages": [
    {"name": "run", "argv": [")" + std::string(DIF_DIFRUN_PATH) +
                       R"(", "--backend", "cpu", "--program", ")" +
                       program_path.string() + R"(", "--input", "1=)" +
                       (workspace() / "x.diftensor").string() +
                       R"(", "--input", "2=)" +
                       (workspace() / "w1.diftensor").string() +
                       R"(", "--input", "4=)" +
                       (workspace() / "w2.diftensor").string() +
                       R"(", "--input", "6=)" +
                       (workspace() / "w3.diftensor").string() +
                       R"(", "--output", "7=${workdir}/y.diftensor", "--warmups", "0", "--iterations", "3"]},
    {"name": "write", "argv": ["cp", ")" + fixture.string() + R"(", "${output}"]}
  ],
  "output": "${workdir}/out.png"
})");
  const auto workdir = workspace() / "trace";
  std::filesystem::remove_all(workdir);
  const auto outcome = run(DIF_DIFTRACE_PATH,
                           {"recipe", recipe.string(), "--workdir",
                            workdir.string(), "--json", "--no-ffprobe"});
  expect(outcome.exit_code == 0, "diftrace recipe completes");
  try {
    const auto document = dif::json::parse(outcome.output);
    expect(required(document, "kind").string() == dif::telemetry::kind::trace,
           "diftrace emits the trace kind");
    const auto &stages = required(document, "stages").array();
    expect(stages.size() == 2U, "diftrace reports every stage");
    const auto &runtime = required(stages.at(0), "runtime");
    expect(required(runtime, "runtime_trace_documents").number() == 1.0,
           "the traced stage produced one runtime-trace document");
    const auto &attribution = required(runtime, "run_attribution");
    expect(required(required(attribution, "operation"), "count").number() ==
               static_cast<double>(kTinyOperations * 3U),
           "runtime attribution counts operation submissions across iterations");
    const auto &chain = required(required(document, "attribution"),
                                 "runtime_categories");
    expect(chain.find("operation") != nullptr,
           "chain attribution merges runtime categories");
    expect(required(required(document, "attribution"), "stages_by_wall")
               .array()
               .size() == 2U,
           "stages ranked by wall");
    expect(std::filesystem::exists(workdir / "trace" / "run.jsonl"),
           "per-stage trace sink written");
  } catch (const std::exception &error) {
    expect(false, std::string("diftrace JSON: ") + error.what());
  }
  const auto direct = run(
      DIF_DIFTRACE_PATH,
      {"program", "--backend", "cpu", "--program", program_path.string(),
       "--input", "1=" + (workspace() / "x.diftensor").string(), "--input",
       "2=" + (workspace() / "w1.diftensor").string(), "--input",
       "4=" + (workspace() / "w2.diftensor").string(), "--input",
       "6=" + (workspace() / "w3.diftensor").string(), "--iterations", "2",
       "--json"});
  expect(direct.exit_code == 0, "diftrace program completes");
  try {
    const auto document = dif::json::parse(direct.output);
    expect(required(required(document, "trace"), "run_events").array().size() ==
               kTinyOperations * 2U,
           "diftrace program lists the runtime events");
    expect(!required(document, "operation_mix").array().empty(),
           "diftrace program reports the opcode mix");
  } catch (const std::exception &error) {
    expect(false, std::string("diftrace program JSON: ") + error.what());
  }
}

void test_difplan_cli() {
  const auto program = tiny_program();
  const auto program_path = workspace() / "plan-program.difir";
  dif::ir::write_file(program, program_path);
  dif::opt::OptimizationPlan plan;
  plan.base_program_fingerprint = "a";
  plan.base_fingerprint = "b";
  plan.candidate_program_fingerprint = "c";
  plan.candidate_fingerprint = "d";
  dif::opt::Transform residency;
  residency.kind = dif::opt::TransformKind::SetConstantResidency;
  residency.tensors = {2U};
  residency.parameters = {0U};
  plan.transforms.push_back(residency);
  dif::opt::PlanDecision decision;
  decision.subject = "tensor";
  decision.subject_id = 2U;
  decision.decision = "resident";
  decision.reason = "admitted as resident: complete plan needs 100 of 200 bytes";
  decision.evidence.emplace_back("bytes", "32");
  plan.decisions.push_back(decision);
  dif::opt::PlanDecision candidate;
  candidate.subject = "candidate";
  candidate.subject_id = 1U;
  candidate.decision = "selected";
  candidate.reason = "Accepted: fastest accepted candidate under the objective";
  candidate.evidence.emplace_back(
      "transforms", dif::opt::encode_transform_sequence(plan.transforms));
  plan.decisions.push_back(candidate);
  const auto plan_a = workspace() / "a.difplan";
  const auto plan_b = workspace() / "b.difplan";
  dif::opt::write_plan(plan, plan_a);
  auto other = plan;
  other.transforms.clear();
  other.decisions.clear();
  other.candidate_fingerprint = "e";
  dif::opt::write_plan(other, plan_b);

  const auto show = run(DIF_DIFPLAN_PATH, {"show", plan_a.string(), "--json"});
  expect(show.exit_code == 0, "difplan show succeeds");
  try {
    const auto document = dif::json::parse(show.output);
    expect(required(document, "kind").string() == dif::telemetry::kind::plan_report,
           "difplan emits the plan-report kind");
    expect(required(required(document, "plan"), "decision_total").number() == 2.0,
           "difplan show counts decisions");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan show JSON: ") + error.what());
  }
  const auto diff = run(DIF_DIFPLAN_PATH,
                        {"diff", plan_a.string(), plan_b.string(), "--json"});
  expect(diff.exit_code == 1, "difplan diff exits 1 for differing plans");
  try {
    const auto document = dif::json::parse(diff.output);
    expect(!required(document, "identical").boolean(), "plans differ");
    expect(required(document, "transforms_removed").array().size() == 1U,
           "removed transform reported");
    expect(required(document, "decision_changes").array().size() == 2U,
           "removed decisions reported");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan diff JSON: ") + error.what());
  }
  const auto explain = run(DIF_DIFPLAN_PATH,
                           {"explain", "tensor", "2", plan_a.string(),
                            "--program", program_path.string(), "--json"});
  expect(explain.exit_code == 0, "difplan explain tensor succeeds");
  try {
    const auto document = dif::json::parse(explain.output);
    expect(required(document, "recorded_decision").boolean(),
           "explain finds the recorded tensor decision");
    expect(required(required(document, "facts"), "bytes").number() == 32.0,
           "explain reports program facts for the tensor");
    expect(required(document, "applied_transforms").array().size() == 1U,
           "explain lists the transform naming the tensor");
    expect(required(document, "candidate_decisions").array().size() == 1U,
           "explain lists candidates whose transforms touch the tensor");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan explain JSON: ") + error.what());
  }
  const auto unexplained = run(DIF_DIFPLAN_PATH,
                               {"explain", "op", "1", plan_a.string(), "--json"});
  expect(unexplained.exit_code == 0, "difplan explain op succeeds");
  try {
    const auto document = dif::json::parse(unexplained.output);
    expect(!required(document, "recorded_decision").boolean(),
           "explain does not invent a decision for an unrecorded subject");
    expect(document.find("note") != nullptr, "absence is stated explicitly");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan explain op JSON: ") + error.what());
  }
  const auto residency_report =
      run(DIF_DIFPLAN_PATH, {"residency", "--program", program_path.string(),
                             "--budget-mib", "64", "--order", "largest", "--json"});
  expect(residency_report.exit_code == 0, "difplan residency runs the planner");
  try {
    const auto document = dif::json::parse(residency_report.output);
    expect(required(document, "tensors").array().size() == kTinyOperations,
           "planner residency lists every streamed constant");
    expect(required(required(document, "planner"), "selection_order").string() ==
               "largest_first",
           "planner residency reports the selection order");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan residency JSON: ") + error.what());
  }
  const auto from_plan = run(DIF_DIFPLAN_PATH,
                             {"residency", plan_a.string(), "--program",
                              program_path.string(), "--json"});
  expect(from_plan.exit_code == 0, "difplan residency reads a plan");
  try {
    const auto document = dif::json::parse(from_plan.output);
    const auto &tensors = required(document, "tensors").array();
    expect(tensors.size() == 1U && required(tensors.at(0), "source").string() ==
                                       "recorded-decision",
           "plan residency prefers the recorded decision");
  } catch (const std::exception &error) {
    expect(false, std::string("difplan residency plan JSON: ") + error.what());
  }
}

} // namespace

int main() {
  try {
    std::filesystem::remove_all(workspace());
    std::filesystem::create_directories(workspace());
    test_document_model();
    test_runtime_trace_cpu();
    test_plan_decisions();
    test_residency_decisions();
    test_difbench_cli();
    test_diftrace_cli();
    test_difplan_cli();
  } catch (const std::exception &error) {
    std::cerr << "telemetry tests: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " telemetry test failure(s)\n";
    return 1;
  }
  std::cout << "TELEMETRY_TESTS PASS\n";
  return 0;
}
