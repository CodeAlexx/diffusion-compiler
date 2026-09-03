// Phase-D gate: physical-format registry legality against architecture
// capabilities, SquareQ/FP8 hooks reported as hook-only, discovery gated on
// the probed target, format decisions recorded into difopt plans, and
// difweights storage statistics without name semantics.

#include "dif/compiler/compiler.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/physical_format.hpp"
#include "dif/opt/plan.hpp"
#include "dif/opt/rewrite.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/json.hpp"
#include "dif/weights/safetensors.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
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
      std::filesystem::temp_directory_path() / "dif_format_tests";
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

dif::target::TargetProfile nvidia_profile(dif::target::ArchitectureFamily family,
                                          bool fp8, bool nvfp4) {
  dif::target::TargetProfile profile;
  profile.backend = "cuda";
  profile.vendor = dif::target::Vendor::Nvidia;
  profile.architecture = family;
  profile.product_name = "synthetic";
  profile.precision.tensor_cores = true;
  profile.precision.fp16_tensor_cores = true;
  profile.precision.bf16_tensor_cores = true;
  profile.precision.int8_tensor_cores = true;
  profile.precision.fp8_tensor_cores = fp8;
  profile.precision.nvfp4_tensor_cores = nvfp4;
  return profile;
}

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
  tensor(2U, TensorRole::Constant);
  tensor(3U, TensorRole::Internal);
  tensor(4U, TensorRole::Constant);
  tensor(5U, TensorRole::Output);
  const auto add = [](std::uint32_t id, std::uint32_t left, std::uint32_t right,
                      std::uint32_t out) {
    dif::ir::Operation operation;
    operation.id = id;
    operation.opcode = dif::ir::Opcode::Add;
    operation.inputs = {left, right};
    operation.outputs = {out};
    return operation;
  };
  program.operations = {add(1U, 1U, 2U, 3U), add(2U, 3U, 4U, 5U)};
  dif::ir::verify(program);
  return program;
}

dif::runtime::Tensor filled(float value) {
  std::vector<std::uint8_t> bytes(8U * sizeof(float));
  for (std::size_t index = 0; index < 8U; ++index)
    std::memcpy(bytes.data() + index * sizeof(float), &value, sizeof(float));
  return dif::runtime::Tensor(dif::ir::DType::F32, {8U}, std::move(bytes));
}

// MXFP8/E8M0 scale reconstruction must use ldexp, never exp2: under
// fast-math `ex2.approx.ftz` flushes small scales to zero on SM 9x (Mojo
// lesson, serenitymojo/ops/mxfp4.mojo:99-105). Inspect the generated source.
void test_mxfp8_scale_reconstruction_uses_ldexp() {
  using namespace dif::ir;
  constexpr std::uint64_t rows = 128U;
  constexpr std::uint64_t inner = 128U;
  constexpr std::uint64_t columns = 128U;
  constexpr std::uint64_t scale_blocks = inner / 32U;
  Program program;
  program.tensors = {
      {1U, DType::BF16, TensorRole::Input, {rows, inner}},
      {2U, DType::FP8E4M3, TensorRole::Output, {rows, inner}},
      {3U, DType::FP8E8M0, TensorRole::Output, {rows, scale_blocks}},
      {4U, DType::FP8E4M3, TensorRole::Constant, {columns, inner}},
      {5U, DType::FP8E8M0, TensorRole::Constant, {columns, scale_blocks}},
      {6U, DType::BF16, TensorRole::Output, {rows, columns}},
  };
  program.operations = {
      {1U, Opcode::QuantizeFp8Blocks32, {1U}, {2U, 3U},
       {Attribute::u64(AttrKey::BlockSize, 256U)}},
      {2U, Opcode::LinearFp8BlockScaled, {2U, 4U, 3U, 5U}, {6U}, {}},
  };
  dif::ir::verify(program);
  const auto generated = dif::compiler::emit_cuda(program);
  const auto &source = generated.source;
  expect(source.find("ldexpf(") != std::string::npos,
         "MXFP8 block quantization reconstructs the E8M0 scale with ldexpf");
  expect(source.find("exp2f(") == std::string::npos &&
             source.find("ex2.approx") == std::string::npos &&
             source.find("powf(2") == std::string::npos,
         "generated FP8 code never reconstructs a scale with exp2/ex2.approx/powf");
  expect(dif::runtime::fp8_e8m0_to_float(0U) > 0.0F &&
             dif::runtime::fp8_e8m0_to_float(0U) < 1.0e-37F,
         "CPU reference keeps the smallest E8M0 scale non-zero");
}

void test_format_registry() {
  using dif::opt::PhysicalFormat;
  expect(dif::opt::all_physical_formats().size() == 11U, "eleven registered formats");
  PhysicalFormat parsed;
  expect(dif::opt::physical_format_from_name("squareq-nvfp4", parsed) &&
             parsed == PhysicalFormat::SquareQNvfp4,
         "format names parse");
  expect(!dif::opt::physical_format_from_name("fp4", parsed),
         "unknown format names are rejected");

  const auto ampere = nvidia_profile(dif::target::ArchitectureFamily::Ampere, false, false);
  const auto fp8 = dif::opt::physical_format_status(PhysicalFormat::Fp8E4M3, &ampere);
  expect(!fp8.legal_on_target && fp8.legality_reason.find("FP8 tensor cores") != std::string::npos,
         "FP8 is illegal on Ampere with the missing capability named");
  const auto w8 = dif::opt::physical_format_status(PhysicalFormat::SquareQW8, &ampere);
  expect(w8.legal_on_target && !w8.competes &&
             w8.availability == dif::opt::FormatAvailability::HookOnly,
         "SquareQ W8 is legal on Ampere but hook-only, so it does not compete");
  const auto nvfp4 = dif::opt::physical_format_status(PhysicalFormat::SquareQNvfp4, &ampere);
  expect(!nvfp4.legal_on_target, "SquareQ NVFP4 is illegal without NVFP4 tensor cores");
  const auto int4 = dif::opt::physical_format_status(PhysicalFormat::Int4Group, &ampere);
  expect(int4.competes, "INT4 grouped quantization competes as a search candidate");
  const auto convrot = dif::opt::physical_format_status(PhysicalFormat::Int8ConvRot, &ampere);
  expect(convrot.legal_on_target && !convrot.competes &&
             convrot.availability == dif::opt::FormatAvailability::ExecutionPolicy,
         "ConvRot INT8 is legal but execution policy, not a search candidate");

  const auto hopper = nvidia_profile(dif::target::ArchitectureFamily::Hopper, true, false);
  expect(dif::opt::physical_format_status(PhysicalFormat::Fp8E4M3, &hopper).legal_on_target,
         "FP8 is legal on an FP8-capable target");
  expect(!dif::opt::physical_format_status(PhysicalFormat::Fp8E4M3, &hopper).competes,
         "FP8 still does not compete without an implementation");
  const auto blackwell = nvidia_profile(dif::target::ArchitectureFamily::Blackwell, true, true);
  expect(dif::opt::physical_format_status(PhysicalFormat::SquareQNvfp4, &blackwell).legal_on_target,
         "SquareQ NVFP4 is legal on an NVFP4-capable target");
  const auto mxfp8_ampere =
      dif::opt::physical_format_status(PhysicalFormat::Fp8BlockScaled, &ampere);
  expect(!mxfp8_ampere.legal_on_target &&
             mxfp8_ampere.legality_reason.find("FP8 tensor cores") != std::string::npos,
         "MXFP8 is illegal without FP8 tensor cores");
  auto old_library = blackwell;
  old_library.cublaslt_version = 120403U;
  const auto mxfp8_old = dif::opt::physical_format_status(PhysicalFormat::Fp8BlockScaled, &old_library);
  expect(!mxfp8_old.legal_on_target &&
             mxfp8_old.legality_reason.find("cuBLASLt >= 120800") != std::string::npos &&
             mxfp8_old.legality_reason.find("linked 120403") != std::string::npos,
         "MXFP8 is illegal with a cuBLASLt older than block-scaled matmul, naming both versions");
  auto new_library = blackwell;
  new_library.cublaslt_version = 130100U;
  const auto mxfp8_new = dif::opt::physical_format_status(PhysicalFormat::Fp8BlockScaled, &new_library);
  expect(mxfp8_new.legal_on_target && !mxfp8_new.competes &&
             mxfp8_new.availability == dif::opt::FormatAvailability::ExecutionPolicy,
         "MXFP8 is legal with FP8 tensor cores and cuBLASLt >= 12.8 but is execution policy");
  expect(dif::opt::physical_format_from_name("mxfp8-block-scaled", parsed) &&
             parsed == PhysicalFormat::Fp8BlockScaled,
         "MXFP8 format name parses");

  const auto host = dif::runtime::probe_target(dif::runtime::ProbeBackend::Host);
  expect(!dif::opt::physical_format_status(PhysicalFormat::Int8ConvRot, &host).legal_on_target,
         "device-only formats are illegal on the host target");
  expect(dif::opt::physical_format_status(PhysicalFormat::Bf16, &host).competes,
         "BF16 precision competes on the host");
  const auto none = dif::opt::physical_format_status(PhysicalFormat::Bf16, nullptr);
  expect(!none.legal_on_target && !none.competes &&
             none.legality_reason.find("no target profile") != std::string::npos,
         "without a probed target nothing competes");
}

void test_discovery_gating() {
  using dif::opt::PhysicalFormat;
  dif::opt::RewriteContext context;
  context.program = tiny_program();
  context.bindings.emplace(2U, filled(2.0F));
  context.bindings.emplace(4U, filled(3.0F));
  dif::opt::DiscoveryOptions options;
  options.structural = false;
  options.schedule = false;
  options.memory = false;
  options.physical_formats = {PhysicalFormat::Bf16, PhysicalFormat::Fp16,
                              PhysicalFormat::Fp8E4M3, PhysicalFormat::SquareQW4,
                              PhysicalFormat::Fp32};
  options.target = dif::runtime::probe_target(dif::runtime::ProbeBackend::Host);
  const auto statuses = dif::opt::format_statuses(options);
  expect(statuses.size() == 5U, "every requested format has a status");
  std::size_t competing = 0U;
  for (const auto &status : statuses)
    competing += status.competes ? 1U : 0U;
  expect(competing == 3U, "bf16, fp16, and fp32 compete on the host; fp8 and SquareQ do not");
  const auto transforms = dif::opt::discover(context, options);
  std::size_t bf16 = 0U;
  std::size_t f16 = 0U;
  std::size_t other_precision = 0U;
  for (const auto &transform : transforms) {
    if (transform.kind != dif::opt::TransformKind::SetOperationPrecision)
      continue;
    const auto code = transform.parameters.at(0);
    if (code == static_cast<std::uint64_t>(dif::ir::DType::BF16))
      ++bf16;
    else if (code == static_cast<std::uint64_t>(dif::ir::DType::F16))
      ++f16;
    else
      ++other_precision;
  }
  expect(bf16 > 0U && f16 > 0U, "competing formats produce precision candidates");
  expect(other_precision == 0U, "no precision candidate outside the competing formats");
  options.target.reset();
  const auto blind = dif::opt::discover(context, options);
  bool any_precision = false;
  for (const auto &transform : blind)
    any_precision = any_precision ||
                    transform.kind == dif::opt::TransformKind::SetOperationPrecision;
  expect(!any_precision, "without a probed target no format competes");
}

void test_difopt_formats_cli() {
  const auto table = run(DIF_DIFOPT_PATH, {"--formats-table", "--backend", "cpu", "--json"});
  expect(table.exit_code == 0, "difopt --formats-table succeeds");
  try {
    const auto document = dif::json::parse(table.output);
    expect(required(document, "kind").string() == "physical-formats",
           "formats table kind");
    const auto &formats = required(document, "formats").array();
    expect(formats.size() == 11U, "formats table lists every format");
    bool fp8_illegal = false;
    bool bf16_competes = false;
    for (const auto &entry : formats) {
      const auto &name = required(entry, "format").string();
      if (name == "fp8-e4m3")
        fp8_illegal = !required(entry, "legal_on_target").boolean();
      if (name == "bf16")
        bf16_competes = required(entry, "competes").boolean();
    }
    expect(fp8_illegal && bf16_competes, "formats table reports host legality");
  } catch (const std::exception &error) {
    expect(false, std::string("formats table JSON: ") + error.what());
  }

  const auto program = tiny_program();
  const auto program_path = workspace() / "tiny.difir";
  dif::ir::write_file(program, program_path);
  dif::runtime::write_tensor(filled(1.0F), workspace() / "x.diftensor");
  dif::runtime::write_tensor(filled(2.0F), workspace() / "w1.diftensor");
  dif::runtime::write_tensor(filled(3.0F), workspace() / "w2.diftensor");
  const auto plan_path = workspace() / "formats.difplan";
  const auto search = run(
      DIF_DIFOPT_PATH,
      {"--program", program_path.string(), "--backend", "cpu",
       "--bind", "1=" + (workspace() / "x.diftensor").string(),
       "--bind", "2=" + (workspace() / "w1.diftensor").string(),
       "--bind", "4=" + (workspace() / "w2.diftensor").string(),
       "--formats", "bf16,fp8-e4m3,squareq-w8", "--no-structural",
       "--no-schedule", "--no-memory", "--depth", "1", "--max-candidates", "6",
       "--warmups", "0", "--iterations", "1", "--plan", plan_path.string()});
  expect(search.exit_code == 0, "difopt search with --formats completes");
  expect(search.output.find("FORMAT bf16 competes") != std::string::npos &&
             search.output.find("FORMAT fp8-e4m3 excluded") != std::string::npos &&
             search.output.find("FORMAT squareq-w8 excluded") != std::string::npos,
         "difopt prints one FORMAT line per requested format");
  try {
    const auto plan = dif::opt::read_plan(plan_path);
    std::size_t format_decisions = 0U;
    bool bf16_competes = false;
    bool squareq_excluded = false;
    for (const auto &decision : plan.decisions) {
      if (decision.subject != "physical-format")
        continue;
      ++format_decisions;
      for (const auto &[key, value] : decision.evidence) {
        if (key != "format")
          continue;
        if (value == "bf16")
          bf16_competes = decision.decision == "competes";
        if (value == "squareq-w8")
          squareq_excluded = decision.decision == "excluded" &&
                             decision.reason.find("no backend implementation") !=
                                 std::string::npos;
      }
    }
    expect(format_decisions == 3U, "plan records one decision per requested format");
    expect(bf16_competes && squareq_excluded,
           "plan records why bf16 competed and why SquareQ did not");
  } catch (const std::exception &error) {
    expect(false, std::string("formats plan: ") + error.what());
  }
}

void test_difweights_stats() {
  const auto path = workspace() / "weights.safetensors";
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  std::vector<std::pair<std::string, std::vector<std::uint64_t>>> layout = {
      {"a.0.w", {4U, 8U}}, {"a.1.w", {4U, 8U}}, {"a.2.w", {4U, 8U}},
      {"a.0.b", {8U}},     {"a.1.b", {8U}},     {"a.2.b", {8U}},
      {"big", {64U, 64U}}};
  for (const auto &[name, dims] : layout)
    specs.push_back({name, dif::ir::DType::F32, dims});
  dif::weights::SafeTensorWriter writer(path, specs);
  for (const auto &[name, dims] : layout) {
    std::uint64_t elements = 1U;
    for (const auto dim : dims)
      elements *= dim;
    std::vector<std::uint8_t> bytes(elements * sizeof(float), 0U);
    writer.append(name, std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  }
  (void)writer.finish();
  const auto outcome = run(DIF_DIFWEIGHTS_PATH, {"stats", path.string(), "--json"});
  expect(outcome.exit_code == 0, "difweights stats succeeds");
  try {
    const auto document = dif::json::parse(outcome.output);
    expect(required(document, "kind").string() == "weights-report", "weights report kind");
    const auto &totals = required(document, "totals");
    expect(required(totals, "tensors").number() == 7.0, "stats counts tensors");
    expect(required(totals, "bytes").number() == (3 * 32 + 3 * 8 + 4096) * 4.0,
           "stats sums bytes");
    expect(required(totals, "distinct_shape_patterns").number() == 3.0,
           "stats groups shape patterns");
    const auto &patterns = required(document, "shape_patterns").array();
    expect(required(patterns.at(0), "count").number() == 3.0 &&
               required(patterns.at(0), "repeated").boolean(),
           "most repeated pattern first");
    const auto &largest = required(document, "largest").array();
    expect(required(largest.at(0), "name").string() == "big", "largest tensor named");
    expect(required(document, "note").string().find("no model semantics") != std::string::npos,
           "stats disclaims semantic inference");
  } catch (const std::exception &error) {
    expect(false, std::string("difweights stats JSON: ") + error.what());
  }
}

} // namespace

int main() {
  try {
    std::filesystem::remove_all(workspace());
    std::filesystem::create_directories(workspace());
    test_format_registry();
    test_mxfp8_scale_reconstruction_uses_ldexp();
    test_discovery_gating();
    test_difopt_formats_cli();
    test_difweights_stats();
  } catch (const std::exception &error) {
    std::cerr << "format tests: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " format test failure(s)\n";
    return 1;
  }
  std::cout << "FORMAT_TESTS PASS\n";
  return 0;
}
