// CPU-only integration gate for difopt's checkpoint-binding command line.
//
// These cases cover how a sealed weight bundle composes with explicit tensors
// and with the synthetic fixture, and that each way of presenting a wrong
// checkpoint is refused. They drive the real difopt binary rather than the
// library, because the composition rules being checked live in the tool's
// argument handling and cannot be reached any other way.
//
// The bundled values here come from the deterministic experiment fixture. They
// are not a model and nothing in this file measures numerical quality; the
// subject is which bindings reach the search and which are rejected.

#include "dif/frontend/h3.hpp"
#include "dif/ir/codec.hpp"
#include "dif/ir/verify.hpp"
#include "dif/opt/bindings.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/support/error.hpp"
#include "dif/support/sha256.hpp"
#include "dif/weights/bundle.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <unordered_set>
#include <vector>

namespace {

using namespace dif;

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

std::filesystem::path workspace() {
  static const auto root =
      std::filesystem::temp_directory_path() / "dif_difopt_cli_tests";
  return root;
}

std::string quote(const std::string &value) { return "'" + value + "'"; }

struct RunOutcome {
  int exit_code{};
  std::string output;
};

RunOutcome run_difopt(const std::vector<std::string> &arguments) {
  const auto log = workspace() / "difopt.log";
  std::string command = quote(DIF_DIFOPT_PATH);
  for (const auto &argument : arguments)
    command += " " + quote(argument);
  command += " > " + quote(log.string()) + " 2>&1";
  const auto status = std::system(command.c_str());
  RunOutcome outcome;
  outcome.exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  std::ifstream input(log);
  std::stringstream buffer;
  buffer << input.rdbuf();
  outcome.output = buffer.str();
  return outcome;
}

// The candidate fingerprint difopt prints for the base context. It covers the
// program, the streaming policy and every bound constant value, so two runs
// that differ only in one bound tensor must print different values here.
std::string base_candidate(const std::string &output) {
  const auto marker = output.find(" candidate=");
  if (marker == std::string::npos)
    return {};
  const auto start = marker + std::string(" candidate=").size();
  const auto end = output.find_first_of(" \n", start);
  return output.substr(start, end == std::string::npos ? end : end - start);
}

// A small real-shaped H3 transformer: one layer, packed QKV, two timestep
// tables. Dimensions are deliberately tiny because the subject is binding
// composition, not model behaviour.
ir::Program fixture_program(std::uint64_t block_size) {
  auto program = frontend::make_h3_transformer_bf16(
      /*sequence=*/8U, /*hidden=*/32U, /*heads=*/2U, /*head_dim=*/16U,
      /*ffn=*/64U, /*rotary=*/8U, /*layers=*/1U, /*timestep_tables=*/2U,
      /*time_embed_dim=*/16U, block_size, /*streamed_constants=*/false,
      /*source_shaped_qkv=*/false);
  ir::verify(program);
  return program;
}

// Rotary cosine/sine tables are computed, not checkpoint data, so a real bundle
// does not carry them. Identifying them from the graph keeps this independent
// of tensor numbering.
std::unordered_set<std::uint32_t> computed_tables(const ir::Program &program) {
  std::unordered_set<std::uint32_t> tables;
  for (const auto &operation : program.operations) {
    if (operation.opcode == ir::Opcode::QkNormPartialRope &&
        operation.inputs.size() == 4U) {
      tables.insert(operation.inputs[2]);
      tables.insert(operation.inputs[3]);
    }
  }
  return tables;
}

std::string shard_tensor_name(std::uint32_t tensor_id) {
  // difweights applies the real checkpoint names when it seals a bundle from a
  // SafeTensors index. This test builds its bundle directly, so the names only
  // have to round-trip.
  return "block.0.constant." + std::to_string(tensor_id);
}

struct Fixture {
  std::filesystem::path program_path;
  std::filesystem::path other_program_path;
  std::filesystem::path bundle_path;
  std::filesystem::path tampered_bundle_path;
  // Inputs and computed tables, which a bundle never covers.
  std::vector<std::pair<std::uint32_t, std::filesystem::path>> explicit_binds;
  // One tensor the bundle does carry, rebound to a different value.
  std::uint32_t overridden_tensor{};
  std::filesystem::path overridden_path;
};

Fixture build_fixture() {
  const auto root = workspace();
  std::filesystem::remove_all(root);
  std::filesystem::create_directories(root);

  Fixture fixture;
  const auto program = fixture_program(64U);
  expect(program.tensors.size() == 37U && program.operations.size() == 15U,
         "the fixture program has the expected shape");
  fixture.program_path = root / "block.difir";
  ir::write_file(program, fixture.program_path);

  // Same geometry, different launch block size, so it is a valid program with a
  // different fingerprint.
  fixture.other_program_path = root / "block_other.difir";
  ir::write_file(fixture_program(128U), fixture.other_program_path);

  const auto bindings = opt::synthesize_bindings(program, 5U);
  const auto tables = computed_tables(program);

  std::vector<std::uint32_t> bundled;
  for (const auto &tensor : program.tensors) {
    if (tensor.has_role(ir::TensorRole::Constant) &&
        !tables.contains(tensor.id))
      bundled.push_back(tensor.id);
  }
  expect(!bundled.empty(), "the fixture has checkpoint constants to seal");

  // Everything the bundle does not carry has to be supplied explicitly.
  for (const auto &tensor : program.tensors) {
    if (!tensor.has_role(ir::TensorRole::Input) && !tables.contains(tensor.id))
      continue;
    const auto path = root / ("input_" + std::to_string(tensor.id) + ".diftensor");
    runtime::write_tensor(bindings.at(tensor.id), path);
    fixture.explicit_binds.emplace_back(tensor.id, path);
  }

  std::vector<weights::SafeTensorWriteSpec> specs;
  for (const auto id : bundled) {
    const auto *description = program.tensor(id);
    specs.push_back({shard_tensor_name(id), description->dtype,
                     description->dims});
  }
  const auto shard_path = root / "model-00001.safetensors";
  weights::SafeTensorFile shard;
  {
    weights::SafeTensorWriter writer(shard_path, specs);
    for (const auto id : bundled) {
      const auto &tensor = bindings.at(id);
      writer.append(shard_tensor_name(id),
                    {tensor.data(), tensor.byte_size()});
    }
    shard = writer.finish();
  }

  const auto seal = [&](const std::filesystem::path &path,
                        const Sha256Digest &digest,
                        const std::filesystem::path &out) {
    weights::WeightBundle bundle;
    bundle.program_fingerprint = ir::fingerprint(program);
    bundle.index_fingerprint = sha256_file(shard_path);
    bundle.shards.push_back(
        {path, std::filesystem::file_size(path), digest});
    for (const auto id : bundled) {
      const auto *entry = shard.find(shard_tensor_name(id));
      bundle.bindings.push_back({id, 0U, shard_tensor_name(id), entry->dtype,
                                 entry->dims, entry->file_offset,
                                 entry->byte_count});
    }
    weights::write_weight_bundle(bundle, out);
  };

  fixture.bundle_path = root / "block.difbind";
  seal(shard_path, sha256_file(shard_path), fixture.bundle_path);

  // A shard whose payload changed after sealing, with its size untouched so
  // only the digest can catch it.
  const auto tampered_shard = root / "model-00001-tampered.safetensors";
  std::filesystem::copy_file(shard_path, tampered_shard);
  {
    std::fstream stream(tampered_shard,
                        std::ios::binary | std::ios::in | std::ios::out);
    stream.seekp(static_cast<std::streamoff>(shard.data_offset));
    const char flipped = 0x7F;
    stream.write(&flipped, 1);
  }
  expect(std::filesystem::file_size(tampered_shard) ==
             std::filesystem::file_size(shard_path),
         "tampering preserved the shard size, so only the digest differs");
  fixture.tampered_bundle_path = root / "block_tampered.difbind";
  seal(tampered_shard, sha256_file(shard_path), fixture.tampered_bundle_path);

  // A different value for a tensor the bundle already carries.
  const auto other_values = opt::synthesize_bindings(program, 99U);
  fixture.overridden_tensor = bundled.front();
  fixture.overridden_path = root / "override.diftensor";
  runtime::write_tensor(other_values.at(fixture.overridden_tensor),
                        fixture.overridden_path);
  return fixture;
}

// A minimal, fast search. The point of every case below is which bindings are
// admitted, so the search itself is kept as small as difopt allows.
std::vector<std::string> search_arguments(const std::filesystem::path &program) {
  return {"--program",   program.string(), "--objective", "memory",
          "--warmups",   "0",              "--iterations", "1",
          "--depth",     "1",              "--beam",       "1",
          "--max-candidates", "2",         "--no-numeric", "--no-memory",
          "--blocks",    "64"};
}

void append_explicit_binds(std::vector<std::string> &arguments,
                           const Fixture &fixture) {
  for (const auto &[id, path] : fixture.explicit_binds) {
    arguments.push_back("--bind");
    arguments.push_back(std::to_string(id) + "=" + path.string());
  }
}

void test_bundle_binding_succeeds(const Fixture &fixture) {
  auto arguments = search_arguments(fixture.program_path);
  arguments.push_back("--weight-bundle");
  arguments.push_back(fixture.bundle_path.string());
  arguments.push_back("--verify-shards");
  append_explicit_binds(arguments, fixture);
  const auto outcome = run_difopt(arguments);
  expect(outcome.exit_code == 0,
         "a sealed bundle plus the tensors it does not carry runs: " +
             outcome.output);
  expect(outcome.output.find("BUNDLE path=") != std::string::npos,
         "the run reports the bundle it bound");
}

void test_explicit_binding_overrides_the_bundle(const Fixture &fixture) {
  auto plain = search_arguments(fixture.program_path);
  plain.push_back("--weight-bundle");
  plain.push_back(fixture.bundle_path.string());
  append_explicit_binds(plain, fixture);

  auto overridden = plain;
  overridden.push_back("--bind");
  overridden.push_back(std::to_string(fixture.overridden_tensor) + "=" +
                       fixture.overridden_path.string());

  const auto without = run_difopt(plain);
  const auto with = run_difopt(overridden);
  expect(without.exit_code == 0 && with.exit_code == 0,
         "both the plain and overridden runs succeed");
  const auto plain_hash = base_candidate(without.output);
  const auto overridden_hash = base_candidate(with.output);
  expect(!plain_hash.empty() && !overridden_hash.empty(),
         "both runs report a base candidate fingerprint");
  // The candidate fingerprint covers bound constant values, so a genuine
  // override has to move it. Equal fingerprints would mean the explicit tensor
  // was silently discarded.
  expect(plain_hash != overridden_hash,
         "an explicit tensor overrides the same tensor from the bundle");
}

void test_synthetic_bindings_are_refused_alongside_real_ones(
    const Fixture &fixture) {
  auto arguments = search_arguments(fixture.program_path);
  arguments.push_back("--weight-bundle");
  arguments.push_back(fixture.bundle_path.string());
  arguments.push_back("--synthetic-bindings");
  arguments.push_back("7");
  const auto outcome = run_difopt(arguments);
  expect(outcome.exit_code == 2,
         "a bundle plus the synthetic fixture is a usage error");

  auto with_binds = search_arguments(fixture.program_path);
  append_explicit_binds(with_binds, fixture);
  with_binds.push_back("--synthetic-bindings");
  with_binds.push_back("7");
  expect(run_difopt(with_binds).exit_code == 2,
         "explicit tensors plus the synthetic fixture is a usage error");

  expect(run_difopt(search_arguments(fixture.program_path)).exit_code == 2,
         "a run with no binding source at all is a usage error");
}

void test_mismatched_fingerprint_is_refused(const Fixture &fixture) {
  auto arguments = search_arguments(fixture.other_program_path);
  arguments.push_back("--weight-bundle");
  arguments.push_back(fixture.bundle_path.string());
  append_explicit_binds(arguments, fixture);
  const auto outcome = run_difopt(arguments);
  expect(outcome.exit_code == 1,
         "a bundle sealed against another program is refused");
  expect(outcome.output.find("different DiffIR fingerprint") !=
             std::string::npos,
         "the refusal names the fingerprint mismatch: " + outcome.output);
}

void test_missing_constant_is_refused(const Fixture &fixture) {
  // The bundle alone. It carries the checkpoint constants but never the
  // computed rotary tables, so the program is not fully bound.
  auto arguments = search_arguments(fixture.program_path);
  arguments.push_back("--weight-bundle");
  arguments.push_back(fixture.bundle_path.string());
  const auto outcome = run_difopt(arguments);
  expect(outcome.exit_code == 1,
         "a program whose constants are not fully bound is refused");
  expect(outcome.output.find("missing constant tensor") != std::string::npos,
         "the refusal names the unbound constant: " + outcome.output);
}

void test_verify_shards_rejects_a_modified_shard(const Fixture &fixture) {
  auto arguments = search_arguments(fixture.program_path);
  arguments.push_back("--weight-bundle");
  arguments.push_back(fixture.tampered_bundle_path.string());
  append_explicit_binds(arguments, fixture);

  auto checked = arguments;
  checked.push_back("--verify-shards");
  const auto rejected = run_difopt(checked);
  expect(rejected.exit_code == 1,
         "--verify-shards refuses a shard modified after sealing");
  expect(rejected.output.find("shard SHA-256 mismatch") != std::string::npos,
         "the refusal names the digest mismatch: " + rejected.output);

  // Advisory by design: a search that already verified once may skip re-hashing
  // every shard for every candidate. The size and metadata checks still run.
  const auto unchecked = run_difopt(arguments);
  expect(unchecked.exit_code == 0,
         "without --verify-shards the same shard loads, and the run warns");
  expect(unchecked.output.find("without --verify-shards") != std::string::npos,
         "skipping shard digests is warned about: " + unchecked.output);
}

} // namespace

int main() {
  try {
    const auto fixture = build_fixture();
    test_bundle_binding_succeeds(fixture);
    test_explicit_binding_overrides_the_bundle(fixture);
    test_synthetic_bindings_are_refused_alongside_real_ones(fixture);
    test_mismatched_fingerprint_is_refused(fixture);
    test_missing_constant_is_refused(fixture);
    test_verify_shards_rejects_a_modified_shard(fixture);
    std::filesystem::remove_all(workspace());
  } catch (const std::exception &error) {
    std::cerr << "FAIL: unexpected exception: " << error.what() << "\n";
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " difopt CLI test failure(s)\n";
    return 1;
  }
  std::cout << "all difopt CLI tests passed\n";
  return 0;
}
