// Phase-C gate: frontend-recorded provenance (table round trip and the Krea2
// builder's own records), difinspect --source joins, and difbisect's
// last-known-good / first-known-bad reporting with explicit unobserved spans.
// Synthetic fixtures only; nothing here claims model parity.

#include "dif/frontend/krea2.hpp"
#include "dif/frontend/provenance.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/json.hpp"
#include "dif/weights/safetensors.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
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
      std::filesystem::temp_directory_path() / "dif_bisect_tests";
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

void write_safetensors(const std::filesystem::path &path,
                       const std::vector<std::pair<std::string, float>> &values) {
  std::vector<dif::weights::SafeTensorWriteSpec> specs;
  for (const auto &[name, value] : values)
    specs.push_back({name, dif::ir::DType::F32, {8U}});
  dif::weights::SafeTensorWriter writer(path, std::move(specs));
  for (const auto &[name, value] : values) {
    const auto tensor = filled(value);
    writer.append(name, std::span<const std::uint8_t>(tensor.data(),
                                                      tensor.byte_size()));
  }
  (void)writer.finish();
}

void test_provenance_table() {
  dif::frontend::ProvenanceTable table;
  table.frontend = "unit";
  table.creator = "creator/repo";
  table.creator_revision = "abc123";
  table.records.push_back({1U, "first", -1, "patch.first"});
  table.records.push_back({2U, "blocks.0.attn", 0, "attention.qkv"});
  table.weight_names.emplace_back(2U, "blocks.0.attn.wq.weight");
  const auto text = dif::frontend::serialize_provenance(table);
  const auto restored = dif::frontend::parse_provenance(text);
  expect(restored.records.size() == 2U && restored.weight_names.size() == 1U,
         "provenance table round-trips");
  expect(restored.find(2U) && restored.find(2U)->semantic_tag == "attention.qkv" &&
             restored.find(2U)->block == 0,
         "provenance lookup by operation");
  expect(restored.find(3U) == nullptr, "unrecorded operations are absent");
  expect(restored.weight_name(2U) &&
             *restored.weight_name(2U) == "blocks.0.attn.wq.weight",
         "weight names round-trip");
  expect(dif::frontend::provenance_sidecar_path("x/y.difir").string() ==
             "x/y.difir.provenance.json",
         "sidecar path convention");
}

void test_krea2_provenance() {
  const auto block = dif::frontend::make_krea2_block(dif::frontend::Krea2Config{}, 3U, false);
  expect(block.provenance.records.size() == block.program.operations.size(),
         "Krea2 block records provenance for every operation");
  expect(block.provenance.creator_revision ==
             std::string(dif::frontend::kKrea2CreatorRevision),
         "Krea2 block carries the pinned creator revision");
  std::set<std::string> tags;
  bool all_block_three = true;
  bool attention_tagged = false;
  for (const auto &record : block.provenance.records) {
    tags.insert(record.semantic_tag);
    all_block_three = all_block_three && record.block == 3;
    const auto &op = block.program.operations[record.operation_id - 1U];
    if (op.opcode == dif::ir::Opcode::Attention)
      attention_tagged = record.semantic_tag == "attention.core" &&
                         record.creator_module == "blocks.3.attn";
  }
  expect(all_block_three, "every block record names block 3");
  expect(attention_tagged, "the Attention operation is tagged attention.core");
  for (const char *tag : {"modulation", "attention.qkv", "attention.qknorm",
                          "attention.rotary", "attention.core",
                          "attention.gate", "attention.out",
                          "attention.residual", "modulation.post",
                          "mlp.gate_up", "mlp.activation", "mlp.down",
                          "mlp.residual"})
    expect(tags.contains(tag), std::string("Krea2 block records tag ") + tag);
  expect(block.provenance.weight_names.size() == block.checkpoint_tensors.size(),
         "block weight names cover every checkpoint tensor");

  const auto denoiser =
      dif::frontend::make_krea2_denoiser(dif::frontend::Krea2Config{}, false);
  expect(denoiser.provenance.records.size() == denoiser.program.operations.size(),
         "Krea2 denoiser records provenance for every operation");
  std::set<std::uint32_t> ids;
  std::set<std::int64_t> blocks;
  bool ids_match = true;
  for (const auto &record : denoiser.provenance.records) {
    ids.insert(record.operation_id);
    blocks.insert(record.block);
    bool present = false;
    for (const auto &op : denoiser.program.operations)
      present = present || op.id == record.operation_id;
    ids_match = ids_match && present;
  }
  expect(ids.size() == denoiser.program.operations.size() && ids_match,
         "denoiser provenance names each operation exactly once");
  expect(blocks.contains(-1) && blocks.contains(0) && blocks.contains(27) &&
             blocks.size() == 29U,
         "denoiser provenance spans pre-block, 28 blocks, and post-block");
  expect(denoiser.provenance.records.front().semantic_tag == "patch.first" &&
             denoiser.provenance.records.back().semantic_tag == "final.head",
         "denoiser first/last operations are patch.first and final.head");
  expect(denoiser.provenance.weight_names.size() ==
             denoiser.checkpoint_tensors.size(),
         "denoiser weight names cover every checkpoint tensor");
}

void test_difbisect_pairs() {
  const auto native = workspace() / "native.safetensors";
  const auto oracle = workspace() / "oracle.safetensors";
  write_safetensors(native, {{"a", 1.0F}, {"b", 2.0F}, {"c", 3.0F}});
  write_safetensors(oracle, {{"a", 1.0F}, {"b", 4.0F}, {"c", 3.0F}, {"d", 9.0F}});
  const auto outcome = run(DIF_DIFBISECT_PATH,
                           {"pairs", "--native", native.string(), "--oracle",
                            oracle.string(), "--order", "a,b,c,d", "--json"});
  expect(outcome.exit_code == 1, "difbisect pairs exits 1 on divergence");
  try {
    const auto document = dif::json::parse(outcome.output);
    const auto &result = required(document, "result");
    expect(required(result, "last_known_good").string() == "a",
           "pairs: last known good is a");
    expect(required(result, "first_known_bad").string() == "b",
           "pairs: first known bad is b");
    const auto &boundaries = required(document, "boundaries").array();
    expect(boundaries.size() == 4U, "pairs: every declared boundary listed");
    expect(required(boundaries.at(2), "verdict").string() == "pass",
           "pairs: a later passing boundary stays reported as pass");
    expect(!required(boundaries.at(3), "observed").boolean() &&
               required(boundaries.at(3), "verdict").string() == "not-comparable",
           "pairs: a boundary missing on one side is unobserved, not judged");
    expect(required(document, "status").string() == "diverged",
           "pairs: status diverged");
  } catch (const std::exception &error) {
    expect(false, std::string("difbisect pairs JSON: ") + error.what());
  }
  const auto clean = run(DIF_DIFBISECT_PATH,
                         {"pairs", "--native", native.string(), "--oracle",
                          native.string(), "--order", "a,b,c", "--json"});
  expect(clean.exit_code == 0, "difbisect pairs exits 0 without divergence");
  try {
    const auto document = dif::json::parse(clean.output);
    expect(required(required(document, "result"), "first_known_bad").is_object() == false,
           "pairs: no first bad when nothing diverges");
    expect(required(document, "status").string() == "no-divergence-observed",
           "pairs: verdict never claims absence of divergence beyond captures");
  } catch (const std::exception &error) {
    expect(false, std::string("difbisect clean JSON: ") + error.what());
  }
}

void test_difbisect_program() {
  const auto program = tiny_program();
  const auto program_path = workspace() / "tiny.difir";
  dif::ir::write_file(program, program_path);
  dif::runtime::write_tensor(filled(1.0F), workspace() / "x.diftensor");
  dif::runtime::write_tensor(filled(2.0F), workspace() / "w1.diftensor");
  dif::runtime::write_tensor(filled(3.0F), workspace() / "w2.diftensor");
  dif::runtime::write_tensor(filled(4.0F), workspace() / "w3.diftensor");
  // Native values: t1 = 3, t2 = 6, y = 10. The oracle disagrees from t2 on.
  const auto oracle = workspace() / "oracle-program.safetensors";
  write_safetensors(oracle, {{"t1", 3.0F}, {"t2", 7.0F}, {"y", 11.0F}});
  const std::vector<std::string> common = {
      "program", "--backend", "cpu", "--program", program_path.string(),
      "--input", "1=" + (workspace() / "x.diftensor").string(),
      "--input", "2=" + (workspace() / "w1.diftensor").string(),
      "--input", "4=" + (workspace() / "w2.diftensor").string(),
      "--input", "6=" + (workspace() / "w3.diftensor").string(),
      "--oracle", oracle.string(), "--json"};
  auto full = common;
  full.insert(full.end(), {"--map", "3=t1", "--map", "5=t2", "--map", "7=y"});
  const auto outcome = run(DIF_DIFBISECT_PATH, full);
  expect(outcome.exit_code == 1, "difbisect program exits 1 on divergence");
  try {
    const auto document = dif::json::parse(outcome.output);
    const auto &result = required(document, "result");
    expect(required(result, "last_known_good").string() == "t1",
           "program: last known good t1");
    expect(required(result, "first_known_bad").string() == "t2",
           "program: first known bad t2");
    expect(required(required(result, "unobserved_span"), "operation_count").number() == 0.0,
           "program: adjacent captures leave no unobserved operation");
    const auto &boundaries = required(document, "boundaries").array();
    expect(required(boundaries.at(0), "producer_operation").number() == 1.0 &&
               required(boundaries.at(1), "producer_operation").number() == 2.0,
           "program: boundaries ordered by producer");
    expect(required(required(boundaries.at(1), "numerics"), "relative_l2").number() > 0.1,
           "program: the diverging boundary carries its metrics");
  } catch (const std::exception &error) {
    expect(false, std::string("difbisect program JSON: ") + error.what());
  }
  auto sparse = common;
  sparse.insert(sparse.end(), {"--map", "3=t1", "--map", "7=y"});
  const auto gap = run(DIF_DIFBISECT_PATH, sparse);
  expect(gap.exit_code == 1, "difbisect sparse program exits 1");
  try {
    const auto document = dif::json::parse(gap.output);
    const auto &result = required(document, "result");
    expect(required(result, "first_known_bad").string() == "y",
           "sparse: first known bad y");
    const auto &span = required(result, "unobserved_span");
    expect(required(span, "operation_count").number() == 1.0,
           "sparse: one uncaptured operation between good and bad");
    expect(required(required(span, "operations").array().at(0), "operation").number() == 2.0,
           "sparse: the uncaptured operation is named, not convicted");
    expect(required(span, "statement").string().find("not individually convicted") !=
               std::string::npos,
           "sparse: statement refuses to locate the divergence further");
  } catch (const std::exception &error) {
    expect(false, std::string("difbisect sparse JSON: ") + error.what());
  }
}

void test_difinspect_source() {
  const auto program = tiny_program();
  const auto program_path = workspace() / "inspect.difir";
  dif::ir::write_file(program, program_path);
  dif::frontend::ProvenanceTable table;
  table.frontend = "unit";
  table.creator = "creator/repo";
  table.creator_revision = "abc123";
  table.records.push_back({1U, "first", -1, "patch.first"});
  table.records.push_back({2U, "blocks.0.mlp", 0, "mlp.down"});
  table.weight_names.emplace_back(4U, "blocks.0.mlp.down.weight");
  dif::frontend::write_provenance(
      table, dif::frontend::provenance_sidecar_path(program_path));
  // A hand-written runtime-trace line: op 2 observed on cuBLASLt.
  const auto trace = workspace() / "trace.jsonl";
  write_text(trace,
             "{\"kind\":\"runtime-trace\",\"trace\":{\"run_events\":["
             "{\"category\":\"gemm\",\"name\":\"cublasLtMatmul\",\"operation_id\":2},"
             "{\"category\":\"synchronization\",\"name\":\"cuEventRecord\",\"operation_id\":2},"
             "{\"category\":\"generated_kernel\",\"name\":\"cuLaunchKernel\",\"operation_id\":0}"
             "]}}\n");
  const auto outcome = run(DIF_DIFINSPECT_PATH,
                           {program_path.string(), "--source", "--trace",
                            trace.string(), "--json"});
  expect(outcome.exit_code == 0, "difinspect --source succeeds");
  try {
    const auto document = dif::json::parse(outcome.output);
    expect(required(document, "kind").string() == "provenance-report",
           "difinspect --source emits the provenance-report kind");
    const auto &summary = required(document, "summary");
    expect(required(summary, "with_recorded_provenance").number() == 2.0,
           "two operations carry recorded provenance");
    expect(required(summary, "with_observed_backend").number() == 1.0,
           "one operation has an observed backend implementation");
    const auto &rows = required(document, "operations").array();
    expect(required(rows.at(2), "provenance").is_object() == false,
           "unrecorded operation reports no provenance");
    const auto &second = rows.at(1);
    expect(required(required(second, "provenance"), "tag").string() == "mlp.down",
           "recorded tag joined to the operation");
    expect(required(required(second, "backend"), "observed").boolean(),
           "observed backend joined to the operation");
    const auto &weights = required(second, "weights").array();
    expect(weights.size() == 1U &&
               required(weights.at(0), "creator_name").string() ==
                   "blocks.0.mlp.down.weight",
           "constant input joined to its creator weight name");
    expect(required(required(rows.at(0), "backend"), "observed").boolean() == false,
           "operation without trace events is unobserved");
    expect(required(required(document, "sources"), "creator").is_object(),
           "sources name the frontend creator");
  } catch (const std::exception &error) {
    expect(false, std::string("difinspect source JSON: ") + error.what());
  }
  const auto one = run(DIF_DIFINSPECT_PATH,
                       {program_path.string(), "--source", "--op", "2", "--json"});
  expect(one.exit_code == 0, "difinspect --source --op succeeds");
  try {
    const auto document = dif::json::parse(one.output);
    expect(required(document, "operations").array().size() == 1U,
           "--op filters to one operation");
  } catch (const std::exception &error) {
    expect(false, std::string("difinspect --op JSON: ") + error.what());
  }
  const auto structural = run(DIF_DIFINSPECT_PATH, {program_path.string(), "--json"});
  expect(structural.exit_code == 0, "difinspect --json succeeds");
  try {
    const auto document = dif::json::parse(structural.output);
    expect(required(document, "kind").string() == "program-report" &&
               required(document, "operations").array().size() == 3U,
           "difinspect --json lists the program");
  } catch (const std::exception &error) {
    expect(false, std::string("difinspect --json: ") + error.what());
  }
}

} // namespace


// revert-check: a temporary git repository whose second commit breaks a
// file-content gate. HEAD fails, HEAD minus that commit passes: CONFIRMED.
// A gate that fails regardless is NOT_ISOLATED; one that already passes on
// HEAD is HEAD_PASSES; an unknown commit is BLOCKED.
void test_difbisect_revert_check() {
#ifdef DIF_DIFBISECT_PATH
  const auto repo = workspace() / "revert-repo";
  std::filesystem::remove_all(repo);
  std::filesystem::create_directories(repo);
  auto git = [&](const std::string &arguments) {
    const auto command = "git -C " + quote(repo.string()) + " " + arguments +
                         " > /dev/null 2>&1";
    return std::system(command.c_str()) == 0;
  };
  if (!git("init -q") || !git("config user.email t@example.com") ||
      !git("config user.name test")) {
    std::cout << "revert-check test skipped: git unavailable\n";
    return;
  }
  { std::ofstream(repo / "gate.txt") << "ok\n"; }
  expect(git("add gate.txt") && git("commit -q -m good"), "revert-check: good commit");
  { std::ofstream(repo / "gate.txt") << "bad\n"; }
  expect(git("commit -q -am bad"), "revert-check: bad commit");
  auto confirmed = run(DIF_DIFBISECT_PATH,
                       {"revert-check", "--repo", repo.string(), "--commit", "HEAD",
                        "--no-build", "--", "grep", "-q", "ok", "{repo}/gate.txt"});
  expect(confirmed.exit_code == 0 &&
             confirmed.output.find("REVERT_CHECK CONFIRMED") != std::string::npos,
         "revert-check confirms the breaking commit: " + confirmed.output.substr(0, 160));
  auto not_isolated = run(DIF_DIFBISECT_PATH,
                          {"revert-check", "--repo", repo.string(), "--commit", "HEAD",
                           "--no-build", "--", "false"});
  expect(not_isolated.exit_code == 1 &&
             not_isolated.output.find("NOT_ISOLATED") != std::string::npos,
         "revert-check reports NOT_ISOLATED when the gate fails either way");
  auto head_passes = run(DIF_DIFBISECT_PATH,
                         {"revert-check", "--repo", repo.string(), "--commit", "HEAD",
                          "--no-build", "--", "true"});
  expect(head_passes.exit_code == 1 &&
             head_passes.output.find("HEAD_PASSES") != std::string::npos,
         "revert-check reports HEAD_PASSES when the premise is wrong");
  auto blocked = run(DIF_DIFBISECT_PATH,
                     {"revert-check", "--repo", repo.string(), "--commit",
                      "0000000000000000000000000000000000000000", "--no-build",
                      "--", "true"});
  expect(blocked.exit_code == 3 && blocked.output.find("BLOCKED") != std::string::npos,
         "revert-check is BLOCKED on an unknown commit");
  expect(std::system(("test \"$(git -C " + quote(repo.string()) +
                      " worktree list | wc -l)\" = 1")
                         .c_str()) == 0,
         "revert-check removes its worktree");
#endif
}

int main() {
  try {
    std::filesystem::remove_all(workspace());
    std::filesystem::create_directories(workspace());
    test_provenance_table();
    test_krea2_provenance();
    test_difbisect_pairs();
    test_difbisect_program();
    test_difinspect_source();
    test_difbisect_revert_check();
  } catch (const std::exception &error) {
    std::cerr << "bisect tests: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " bisect test failure(s)\n";
    return 1;
  }
  std::cout << "BISECT_TESTS PASS\n";
  return 0;
}
