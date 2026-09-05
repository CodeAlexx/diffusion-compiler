#include "dif/opt/plan.hpp"
#include "dif/runtime/device_probe.hpp"
#include "dif/support/json.hpp"
#include "dif/target/profile.hpp"
#include "dif/telemetry/schema.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    ++failures;
  }
}

template <typename Function>
void expect_failure(Function function, const std::string &message) {
  try {
    function();
    expect(false, message);
  } catch (const std::exception &) {
    expect(true, message);
  }
}

const dif::json::Value &required(const dif::json::Value &object,
                                 const char *key) {
  const auto *value = object.find(key);
  if (!value)
    throw std::runtime_error(std::string("missing JSON field ") + key);
  return *value;
}

dif::target::TargetProfile synthetic_target() {
  dif::target::TargetProfile profile;
  profile.backend = "cuda";
  profile.vendor = dif::target::Vendor::Nvidia;
  profile.architecture = dif::target::ArchitectureFamily::Ampere;
  profile.product_name = "synthetic GPU name";
  profile.device_ordinal = 0;
  profile.compute_major = 8;
  profile.compute_minor = 6;
  profile.multiprocessor_count = 84;
  profile.warp_size = 32;
  profile.total_device_memory_bytes = 24ULL * 1024ULL * 1024ULL * 1024ULL;
  profile.l2_cache_bytes = 6ULL * 1024ULL * 1024ULL;
  profile.shared_memory_per_block_bytes = 49152;
  profile.shared_memory_per_block_optin_bytes = 101376;
  profile.shared_memory_per_multiprocessor_bytes = 102400;
  profile.precision = {true, true, true, false, true, false};
  profile.execution = {true, true, false};
  profile.cuda_driver_version = 12080;
  profile.cuda_runtime_version = 12060;
  profile.cublaslt_version = 120603;
  profile.cudnn_version = 90701;
  return profile;
}

dif::target::RuntimeBudget synthetic_budget() {
  dif::target::RuntimeBudget budget;
  budget.free_device_memory_bytes = 22ULL * 1024ULL * 1024ULL * 1024ULL;
  budget.reserved_device_memory_bytes = 512ULL * 1024ULL * 1024ULL;
  budget.usable_device_memory_bytes =
      budget.free_device_memory_bytes - budget.reserved_device_memory_bytes;
  budget.total_host_memory_bytes = 64ULL * 1024ULL * 1024ULL * 1024ULL;
  budget.available_host_memory_bytes = 48ULL * 1024ULL * 1024ULL * 1024ULL;
  budget.host_memory_budget_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
  budget.pinned_host_memory_budget_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  budget.workspace_budget_bytes = 256ULL * 1024ULL * 1024ULL;
  budget.staging_budget_bytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;
  return budget;
}

void test_host_cache_admission() {
  using dif::runtime::host_cache_working_set_fits;
  constexpr auto gib = 1ULL << 30U;
  expect(!host_cache_working_set_fits(50 * gib, 40 * gib,
                                     {22 * gib, 2 * gib}, 4 * gib),
         "a non-fitting conditioner must not reread weights to warm cache");
  expect(host_cache_working_set_fits(8 * gib, 40 * gib,
                                    {22 * gib, 2 * gib}, 4 * gib),
         "a fitting iterative denoiser retains cache warming");
  expect(!host_cache_working_set_fits(8 * gib, 10 * gib, {}, 4 * gib),
         "uncapped processes still respect available host RAM");
  expect(host_cache_working_set_fits(16 * gib, 40 * gib,
                                    {22 * gib, 2 * gib}, 4 * gib),
         "exact-fit cache admission preserves the reserve");
  expect(!host_cache_working_set_fits(1, 40 * gib,
                                     {22 * gib, 23 * gib}, 0),
         "an over-limit cgroup does not underflow available memory");
  expect(!host_cache_working_set_fits(1, 0, {}, 0),
         "unknown available host memory fails closed");
  expect(!host_cache_working_set_fits(0, 40 * gib, {}, 0),
         "empty working sets do not need cache warming");
  expect(!host_cache_working_set_fits(UINT64_MAX, UINT64_MAX,
                                     {UINT64_MAX, 1}, 4 * gib),
         "cache admission does not overflow working set plus reserve");
}

void test_architecture_classification() {
  using dif::target::ArchitectureFamily;
  expect(dif::target::classify_nvidia_architecture(8, 6) ==
             ArchitectureFamily::Ampere,
         "sm_86 classifies as Ampere");
  expect(dif::target::classify_nvidia_architecture(8, 9) ==
             ArchitectureFamily::Ada,
         "sm_89 classifies as Ada");
  expect(dif::target::classify_nvidia_architecture(9, 0) ==
             ArchitectureFamily::Hopper,
         "sm_90 classifies as Hopper");
  expect(dif::target::classify_nvidia_architecture(12, 0) ==
             ArchitectureFamily::Blackwell,
         "sm_120 classifies as Blackwell");
}

void test_target_identity() {
  const auto profile = synthetic_target();
  const auto fingerprint = dif::target::target_fingerprint(profile);
  auto diagnostic_change = profile;
  diagnostic_change.product_name = "another equivalent product label";
  diagnostic_change.device_ordinal = 3;
  expect(dif::target::target_fingerprint(diagnostic_change) == fingerprint,
         "product name and ordinal do not become compiler semantics");
  auto capability_change = profile;
  capability_change.cublaslt_version += 1;
  expect(dif::target::target_fingerprint(capability_change) != fingerprint,
         "library capability drift changes target identity");
}

void test_probe_schema() {
  const auto profile = synthetic_target();
  const auto budget = synthetic_budget();
  const auto text = dif::telemetry::serialize_probe(profile, budget);
  const auto document = dif::json::parse(text);
  expect(required(required(document, "schema"), "name").string() ==
             dif::telemetry::kSchemaName,
         "probe uses the shared telemetry schema name");
  expect(required(required(document, "schema"), "version").number() == 1.0,
         "probe reports telemetry schema version 1");
  expect(required(document, "kind").string() == "device-probe",
         "probe document kind is stable");
  const auto &hardware = required(document, "hardware");
  expect(required(hardware, "architecture_family").string() == "ampere",
         "hardware architecture is machine readable");
  expect(required(required(hardware, "precision_features"),
                  "bf16_tensor_cores")
             .boolean(),
         "precision capabilities are machine readable");
  const auto &runtime = required(document, "runtime_budget");
  expect(static_cast<std::uint64_t>(required(runtime, "usable_vram_bytes").number()) ==
             budget.usable_device_memory_bytes,
         "dynamic usable VRAM is distinct from static target capacity");
  expect(!required(runtime, "budget_class").string().empty(),
         "runtime budget has a stable compatibility class");
}

void test_plan_compatibility() {
  const auto profile = synthetic_target();
  const auto budget = synthetic_budget();
  dif::opt::OptimizationPlan plan{"base-program", "base", "candidate-program",
                                  "candidate", {}};
  constexpr std::uint64_t required_device = 20ULL * 1024ULL * 1024ULL * 1024ULL;
  constexpr std::uint64_t required_workspace = 128ULL * 1024ULL * 1024ULL;
  dif::opt::bind_plan_compatibility(plan, profile, budget,
                                    "approximate-int8-v1", required_device,
                                    required_workspace);
  const auto encoded = dif::opt::serialize_plan(plan);
  const auto parsed = dif::opt::parse_plan(encoded);
  expect(parsed.compatibility.has_value(),
         "plan v2 preserves target compatibility identity");
  expect(parsed.compatibility->target_fingerprint ==
             dif::target::target_fingerprint(profile),
         "plan target fingerprint round trips");
  expect(dif::opt::plan_fingerprint(parsed) ==
             dif::opt::plan_fingerprint(plan),
         "plan fingerprint is stable across serialization");
  dif::opt::validate_plan_compatibility(parsed, profile, budget,
                                        "approximate-int8-v1");

  auto wrong_target = profile;
  wrong_target.compute_minor = 9;
  wrong_target.architecture = dif::target::ArchitectureFamily::Ada;
  expect_failure(
      [&] {
        dif::opt::validate_plan_compatibility(
            parsed, wrong_target, budget, "approximate-int8-v1");
      },
      "plan reuse fails closed on target drift");
  auto pressure = budget;
  pressure.usable_device_memory_bytes = required_device - 1U;
  expect_failure(
      [&] {
        dif::opt::validate_plan_compatibility(
            parsed, profile, pressure, "approximate-int8-v1");
      },
      "plan reuse fails closed when current VRAM is insufficient");
  expect_failure(
      [&] {
        dif::opt::validate_plan_compatibility(parsed, profile, budget,
                                              "exact-bf16-v1");
      },
      "plan reuse fails closed on precision policy drift");

  const auto legacy = dif::opt::parse_plan(
      "{\"kind\":\"diffusion-compiler-optimization-plan\","
      "\"version\":1,\"base_program_fingerprint\":\"a\","
      "\"base_fingerprint\":\"b\","
      "\"candidate_program_fingerprint\":\"c\","
      "\"candidate_fingerprint\":\"d\",\"transforms\":[]}");
  expect(!legacy.compatibility,
         "plan v1 remains readable but explicitly unbound");
}

void test_live_host_probe() {
  const auto report = dif::runtime::probe_device(
      dif::runtime::ProbeBackend::Host);
  expect(report.target.backend == "cpu",
         "host probe provides an explicit CPU target");
  expect(report.budget.total_host_memory_bytes > 0U &&
             report.budget.available_host_memory_bytes > 0U,
         "host probe measures total and available RAM");

  const std::string command = std::string(DIF_DIFPROBE_PATH) +
                              " --backend cpu --json";
  std::array<char, 4096> buffer{};
  std::string output;
  auto *pipe = popen(command.c_str(), "r");
  if (!pipe)
    throw std::runtime_error("cannot launch difprobe test command");
  while (std::fgets(buffer.data(), static_cast<int>(buffer.size()), pipe))
    output += buffer.data();
  const auto status = pclose(pipe);
  expect(status == 0, "difprobe --backend cpu --json exits successfully");
  const auto document = dif::json::parse(output);
  expect(required(required(document, "hardware"), "backend").string() ==
             "cpu",
         "difprobe CLI consumes the shared host DeviceProbe");
}

} // namespace

int main() {
  try {
    test_host_cache_admission();
    test_architecture_classification();
    test_target_identity();
    test_probe_schema();
    test_plan_compatibility();
    test_live_host_probe();
  } catch (const std::exception &error) {
    std::cerr << "target tests: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " target test failure(s)\n";
    return 1;
  }
  std::cout << "TARGET_TESTS PASS\n";
  return 0;
}
