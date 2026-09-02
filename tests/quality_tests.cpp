// Phase-E gate: quality metrics and the difquality verdict rules, the
// difregress tiers with baselines, and the oracle fixture protocol
// validation. Synthetic artifacts only.

#include "dif/quality/metrics.hpp"
#include "dif/support/json.hpp"
#include "dif/support/png.hpp"
#include "dif/support/sha256.hpp"
#include "dif/support/wav.hpp"
#include "dif/weights/safetensors.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
      std::filesystem::temp_directory_path() / "dif_quality_tests";
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

const dif::json::Value &required(const dif::json::Value &object, const char *key) {
  const auto *value = object.find(key);
  if (!value)
    throw std::runtime_error(std::string("missing JSON field ") + key);
  return *value;
}

void write_text(const std::filesystem::path &path, const std::string &text) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(text.data(), static_cast<std::streamsize>(text.size()));
}

dif::RgbImage gradient(std::uint32_t width, std::uint32_t height, int shift) {
  dif::RgbImage image;
  image.width = width;
  image.height = height;
  image.pixels.resize(static_cast<std::size_t>(width) * height * 3U);
  for (std::uint32_t y = 0; y < height; ++y)
    for (std::uint32_t x = 0; x < width; ++x) {
      auto *pixel = image.pixels.data() + (static_cast<std::size_t>(y) * width + x) * 3U;
      // Clamp instead of wrapping so a small shift is a small difference.
      const auto clamp = [](long value) {
        return static_cast<std::uint8_t>(std::min(255L, std::max(0L, value)));
      };
      pixel[0] = clamp(static_cast<long>(x * 255U / width) + shift);
      pixel[1] = clamp(static_cast<long>(y * 255U / height) + shift);
      pixel[2] = clamp(static_cast<long>(((x + y) * 7U) % 200U) + shift);
    }
  return image;
}

void write_image(const std::filesystem::path &path, const dif::RgbImage &image) {
  dif::write_png_rgb8(path, image.width, image.height, image.pixels);
}

void test_image_metrics() {
  const auto reference = gradient(32U, 24U, 0);
  const auto same = dif::quality::compare_images(reference, reference);
  expect(same.comparable && same.bit_identical && std::isinf(same.psnr_db) &&
             same.ssim_8x8 > 0.999,
         "identical images: infinite PSNR, SSIM 1");
  const auto shifted = gradient(32U, 24U, 3);
  const auto near = dif::quality::compare_images(reference, shifted);
  expect(near.comparable && !near.bit_identical && near.psnr_db > 30.0 &&
             near.max_absolute_difference == 3U,
         "slightly shifted image: high PSNR, max abs 3");
  const auto far = dif::quality::compare_images(reference, gradient(32U, 24U, 120));
  expect(far.comparable && far.psnr_db < 20.0, "heavily shifted image: low PSNR");
  const auto other = dif::quality::compare_images(reference, gradient(16U, 24U, 0));
  expect(!other.comparable, "geometry mismatch is not comparable");
  dif::RgbImage flat;
  flat.width = 8U;
  flat.height = 8U;
  flat.pixels.assign(8U * 8U * 3U, 77U);
  const auto sanity = dif::quality::image_sanity(flat);
  expect(sanity.decodable && sanity.constant_fraction == 1.0 && !sanity.problem.empty(),
         "constant image is flagged by sanity");
}

std::vector<float> tone(std::size_t samples, float amplitude, float noise_amplitude) {
  std::vector<float> out(samples);
  for (std::size_t index = 0; index < samples; ++index) {
    const auto t = static_cast<float>(index) / 16000.0F;
    out[index] = amplitude * std::sin(2.0F * 3.14159265F * 440.0F * t) +
                 noise_amplitude * (static_cast<float>((index * 7919U) % 1000U) / 500.0F - 1.0F);
  }
  return out;
}

void test_audio_metrics() {
  const auto clean_path = workspace() / "clean.wav";
  const auto noisy_path = workspace() / "noisy.wav";
  const auto silent_path = workspace() / "silent.wav";
  const auto clean = tone(16000U, 0.5F, 0.0F);
  const auto noisy = tone(16000U, 0.5F, 0.001F);
  const std::vector<float> silent(16000U, 0.0F);
  dif::support::write_wav_pcm16(clean_path, clean, 1U, clean.size(), 16000U);
  dif::support::write_wav_pcm16(noisy_path, noisy, 1U, noisy.size(), 16000U);
  dif::support::write_wav_pcm16(silent_path, silent, 1U, silent.size(), 16000U);
  const auto a = dif::quality::read_wav_pcm16(clean_path);
  const auto b = dif::quality::read_wav_pcm16(noisy_path);
  expect(a.sample_rate == 16000U && a.channels == 1U && a.samples_per_channel == 16000U,
         "WAV reader recovers the header");
  const auto same = dif::quality::compare_audio(a, a);
  expect(same.comparable && same.bit_identical && std::isinf(same.snr_db),
         "identical audio: infinite SNR");
  const auto near = dif::quality::compare_audio(a, b);
  expect(near.comparable && !near.bit_identical && near.snr_db > 40.0,
         "slightly noisy audio: high SNR");
  const auto sanity = dif::quality::audio_sanity(dif::quality::read_wav_pcm16(silent_path));
  expect(sanity.decodable && !sanity.problem.empty(), "silent audio is flagged by sanity");
}

void test_difquality_cli() {
  const auto reference = workspace() / "reference.png";
  const auto candidate = workspace() / "candidate.png";
  const auto broken = workspace() / "broken.png";
  const auto flat = workspace() / "flat.png";
  write_image(reference, gradient(64U, 48U, 0));
  write_image(candidate, gradient(64U, 48U, 2));
  write_image(broken, gradient(64U, 48U, 120));
  dif::RgbImage constant;
  constant.width = 64U;
  constant.height = 48U;
  constant.pixels.assign(64U * 48U * 3U, 9U);
  write_image(flat, constant);

  const auto manual = run(DIF_DIFQUALITY_PATH,
                          {candidate.string(), "--reference", reference.string(), "--json"});
  expect(manual.exit_code == 0, "difquality exits 0 when review is required");
  try {
    const auto document = dif::json::parse(manual.output);
    const auto &result = required(document, "result");
    expect(required(result, "numeric_admission").string() == "pass",
           "numeric admission passes for the near image");
    expect(required(result, "verdict").string() == "MANUAL REVIEW REQUIRED",
           "metrics alone never yield PASS");
  } catch (const std::exception &error) {
    expect(false, std::string("difquality manual JSON: ") + error.what());
  }
  const auto accepted = run(DIF_DIFQUALITY_PATH,
                            {candidate.string(), "--reference", reference.string(),
                             "--reviewer", "tester", "--review", "accept", "--json"});
  try {
    const auto document = dif::json::parse(accepted.output);
    expect(required(required(document, "result"), "verdict").string() == "PASS",
           "recorded acceptance plus numeric admission yields PASS");
    expect(required(required(document, "perceptual_review"), "recorded").boolean(),
           "the review is recorded in the report");
  } catch (const std::exception &error) {
    expect(false, std::string("difquality accepted JSON: ") + error.what());
  }
  const auto rejected = run(DIF_DIFQUALITY_PATH,
                            {candidate.string(), "--reference", reference.string(),
                             "--reviewer", "tester", "--review", "reject", "--json"});
  expect(rejected.exit_code == 1, "a recorded rejection fails");
  const auto failed = run(DIF_DIFQUALITY_PATH,
                          {broken.string(), "--reference", reference.string(),
                           "--reviewer", "tester", "--review", "accept", "--json"});
  expect(failed.exit_code == 1, "numeric failure fails even with an accepting review");
  try {
    const auto document = dif::json::parse(failed.output);
    expect(required(required(document, "result"), "verdict").string() == "FAIL",
           "verdict FAIL on metrics below bar");
  } catch (const std::exception &error) {
    expect(false, std::string("difquality failed JSON: ") + error.what());
  }
  const auto constant_run = run(DIF_DIFQUALITY_PATH, {flat.string(), "--json"});
  expect(constant_run.exit_code == 1, "a constant image fails sanity");
  const auto missing = run(DIF_DIFQUALITY_PATH, {(workspace() / "nope.png").string(), "--json"});
  expect(missing.exit_code == 1, "a missing artifact fails");
  const auto audio = run(DIF_DIFQUALITY_PATH,
                         {(workspace() / "noisy.wav").string(), "--reference",
                          (workspace() / "clean.wav").string(), "--json"});
  try {
    const auto document = dif::json::parse(audio.output);
    expect(required(document, "kind_under_test").string() == "audio",
           "WAV candidates are treated as audio");
    expect(required(required(document, "result"), "numeric_admission").string() == "pass",
           "audio numeric admission passes for the near waveform");
  } catch (const std::exception &error) {
    expect(false, std::string("difquality audio JSON: ") + error.what());
  }
}

void test_difregress_cli() {
  const auto suite = workspace() / "suite.json";
  const auto baseline = workspace() / "baselines.json";
  std::filesystem::remove(baseline);
  write_text(suite, R"({
  "kind": "diffusion-compiler-regression-suite",
  "version": 1,
  "name": "synthetic",
  "variables": {"MESSAGE": "hello"},
  "checks": [
    {"name": "passes", "tier": "smoke", "argv": ["sh", "-c", "echo '{\"kind\":\"x\",\"n\":3}'"],
     "expect_json": {"kind": "x", "n": 3},
     "performance": {"metric": "json:n", "samples": 2, "tolerance": 0.10}},
    {"name": "fails", "tier": "smoke", "argv": ["sh", "-c", "exit 1"]},
    {"name": "blocked", "tier": "smoke", "argv": ["sh", "-c", "exit 2"], "blocked_exit": [2]},
    {"name": "timed", "tier": "model", "model": "toy", "argv": ["sh", "-c", "sleep 0.02"],
     "performance": {"metric": "wall_seconds", "samples": 2, "tolerance": 0.50}}
  ]
})");
  const auto smoke = run(DIF_DIFREGRESS_PATH, {"run", suite.string(), "--tier", "smoke", "--json"});
  expect(smoke.exit_code == 1, "a failing check makes the tier fail");
  try {
    const auto document = dif::json::parse(smoke.output);
    const auto &summary = required(document, "summary");
    expect(required(summary, "verdict").string() == "FAIL", "tier verdict FAIL");
    expect(required(summary, "passed").number() == 1.0 &&
               required(summary, "failed").number() == 1.0 &&
               required(summary, "blocked").number() == 1.0,
           "pass/fail/blocked counted");
    const auto &checks = required(document, "checks").array();
    expect(required(required(checks.at(0), "performance"), "verdict").string() == "unbaselined",
           "without a baseline the performance verdict is unbaselined");
  } catch (const std::exception &error) {
    expect(false, std::string("difregress smoke JSON: ") + error.what());
  }
  const auto record = run(DIF_DIFREGRESS_PATH,
                          {"record", suite.string(), "--tier", "model", "toy",
                           "--baseline", baseline.string()});
  expect(record.exit_code == 0 && std::filesystem::exists(baseline),
         "difregress record writes a baseline file");
  const auto model = run(DIF_DIFREGRESS_PATH,
                         {"run", suite.string(), "--tier", "model", "toy",
                          "--baseline", baseline.string(), "--json"});
  expect(model.exit_code == 0, "model tier passes against its own baseline");
  try {
    const auto document = dif::json::parse(model.output);
    const auto &check = required(document, "checks").array().at(0);
    const auto &performance = required(check, "performance");
    const auto verdict = required(performance, "verdict").string();
    expect(verdict == "within" || verdict == "improved",
           "re-measurement against a fresh baseline is within tolerance");
    expect(required(performance, "baseline").is_object(), "baseline embedded in the report");
  } catch (const std::exception &error) {
    expect(false, std::string("difregress model JSON: ") + error.what());
  }
  // A slower command against the recorded baseline must be REGRESSED.
  write_text(suite, R"({
  "kind": "diffusion-compiler-regression-suite",
  "version": 1,
  "name": "synthetic",
  "checks": [
    {"name": "timed", "tier": "model", "model": "toy", "argv": ["sh", "-c", "sleep 0.20"],
     "performance": {"metric": "wall_seconds", "samples": 1, "tolerance": 0.50}}
  ]
})");
  const auto slower = run(DIF_DIFREGRESS_PATH,
                          {"run", suite.string(), "--tier", "model", "toy",
                           "--baseline", baseline.string(), "--json"});
  expect(slower.exit_code == 1, "a regression exits 1");
  try {
    const auto document = dif::json::parse(slower.output);
    expect(required(required(document, "summary"), "verdict").string() == "REGRESSED",
           "tier verdict REGRESSED");
  } catch (const std::exception &error) {
    expect(false, std::string("difregress regressed JSON: ") + error.what());
  }
  const auto missing = run(DIF_DIFREGRESS_PATH,
                           {"run", suite.string(), "--tier", "model", "absent", "--json"});
  expect(missing.exit_code == 3, "a tier with no selected checks is BLOCKED");
}

void test_oracle_protocol() {
  const auto payload = workspace() / "oracle.safetensors";
  std::vector<dif::weights::SafeTensorWriteSpec> specs = {
      {"step_1", dif::ir::DType::F32, {4U}}, {"step_2", dif::ir::DType::F32, {4U}}};
  dif::weights::SafeTensorWriter writer(payload, specs);
  std::vector<std::uint8_t> bytes(16U, 0U);
  writer.append("step_1", std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  writer.append("step_2", std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  (void)writer.finish();
  const auto digest = dif::hex_digest(dif::sha256_file(payload));
  const auto manifest = workspace() / "oracle.json";
  write_text(manifest, R"({
  "kind": "diffusion-compiler-oracle-fixture",
  "version": 1,
  "creator": {"repository": "creator/repo", "revision": "abc123"},
  "model": {"name": "synthetic"},
  "semantic_boundary": "step outputs",
  "dtype": "f32",
  "fixture_version": "v1",
  "inputs": [{"name": "prompt", "sha256": ")" + std::string(64U, 'a') + R"("}],
  "payload": {"path": "oracle.safetensors", "sha256": ")" + digest + R"("},
  "boundaries": [
    {"name": "step_1", "tensor": "step_1", "shape": [4], "dtype": "f32"},
    {"name": "step_2", "tensor": "step_2", "shape": [4], "dtype": "f32"}
  ]
})");
  const auto valid = run(DIF_DIFBISECT_PATH, {"validate-oracle", manifest.string(), "--json"});
  expect(valid.exit_code == 0, "a complete manifest with a matching payload validates");
  try {
    const auto document = dif::json::parse(valid.output);
    expect(required(document, "valid").boolean(), "valid flag set");
  } catch (const std::exception &error) {
    expect(false, std::string("validate-oracle JSON: ") + error.what());
  }
  write_text(manifest, R"({
  "kind": "diffusion-compiler-oracle-fixture",
  "version": 1,
  "creator": {"repository": "creator/repo", "revision": "abc123"},
  "model": {"name": "synthetic"},
  "semantic_boundary": "step outputs",
  "dtype": "f32",
  "fixture_version": "v1",
  "inputs": [],
  "payload": {"path": "oracle.safetensors", "sha256": ")" + std::string(64U, '0') + R"("},
  "boundaries": [{"name": "step_3", "tensor": "step_3", "shape": [4], "dtype": "f32"}]
})");
  const auto invalid = run(DIF_DIFBISECT_PATH, {"validate-oracle", manifest.string(), "--json"});
  expect(invalid.exit_code == 1, "a wrong payload hash and missing tensor invalidate");
  try {
    const auto document = dif::json::parse(invalid.output);
    expect(!required(document, "valid").boolean(), "invalid flag set");
    bool hash_flagged = false;
    bool tensor_flagged = false;
    for (const auto &check : required(document, "checks").array()) {
      const auto &name = required(check, "check").string();
      if (name == "payload.sha256_matches")
        hash_flagged = !required(check, "ok").boolean();
      if (name == "boundary step_3")
        tensor_flagged = !required(check, "ok").boolean();
    }
    expect(hash_flagged && tensor_flagged, "hash and missing tensor named");
  } catch (const std::exception &error) {
    expect(false, std::string("validate-oracle invalid JSON: ") + error.what());
  }
  // The pairs mode consumes the manifest's boundary order.
  const auto native = workspace() / "native.safetensors";
  dif::weights::SafeTensorWriter native_writer(native, specs);
  native_writer.append("step_1", std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  native_writer.append("step_2", std::span<const std::uint8_t>(bytes.data(), bytes.size()));
  (void)native_writer.finish();
  write_text(manifest, R"({
  "kind": "diffusion-compiler-oracle-fixture", "version": 1,
  "creator": {"repository": "creator/repo", "revision": "abc123"},
  "model": {"name": "synthetic"}, "semantic_boundary": "step outputs", "dtype": "f32",
  "fixture_version": "v1", "inputs": [],
  "payload": {"path": "oracle.safetensors", "sha256": ")" + digest + R"("},
  "boundaries": [{"name": "step_1", "tensor": "step_1"}, {"name": "step_2", "tensor": "step_2"}]
})");
  const auto pairs = run(DIF_DIFBISECT_PATH,
                         {"pairs", "--native", native.string(), "--oracle", payload.string(),
                          "--oracle-manifest", manifest.string(), "--json"});
  expect(pairs.exit_code == 0, "pairs mode takes the boundary order from the manifest");
  try {
    const auto document = dif::json::parse(pairs.output);
    expect(required(document, "boundaries").array().size() == 2U,
           "manifest boundaries drive the comparison");
    expect(required(document, "oracle_metadata").is_object(),
           "oracle metadata is embedded in the bisect report");
  } catch (const std::exception &error) {
    expect(false, std::string("pairs manifest JSON: ") + error.what());
  }
}

} // namespace

int main() {
  try {
    std::filesystem::remove_all(workspace());
    std::filesystem::create_directories(workspace());
    test_image_metrics();
    test_audio_metrics();
    test_difquality_cli();
    test_difregress_cli();
    test_oracle_protocol();
  } catch (const std::exception &error) {
    std::cerr << "quality tests: " << error.what() << "\n";
    return 1;
  }
  if (failures != 0) {
    std::cerr << failures << " quality test failure(s)\n";
    return 1;
  }
  std::cout << "QUALITY_TESTS PASS\n";
  return 0;
}
