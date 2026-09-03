// Noise statistics gate (Mojo lesson: serenitymojo/training/noise_stats_smoke.mojo).
// A Box-Muller with a 2^53 divisor bug produced mean +0.56, std 1.19 and
// theta spanning only [0, pi), silently inflating training loss. Every noise
// source the compiler ships must pass: mean, std, sign balance overall and on
// the odd/even halves of each Box-Muller pair, and 3-sigma tail mass. The gate
// proves it can fail by running a deliberately poisoned generator.
#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/runtime/tensor.hpp"
#include "dif/runtime/scalar.hpp"
#include "dif/support/torch_cpu_rng.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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

struct Stats {
  double mean{};
  double std{};
  double positive_fraction{};
  double odd_positive_fraction{};
  double even_positive_fraction{};
  double tail_fraction{};
  bool finite{true};
};

Stats measure(const std::vector<float> &values) {
  Stats stats;
  double sum = 0.0;
  double sum_squares = 0.0;
  std::size_t positive = 0U, odd_positive = 0U, even_positive = 0U, tail = 0U;
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const double value = values[index];
    if (!std::isfinite(value))
      stats.finite = false;
    sum += value;
    sum_squares += value * value;
    if (value > 0.0) {
      ++positive;
      (index % 2U == 0U ? even_positive : odd_positive) += 1U;
    }
    if (std::fabs(value) > 3.0)
      ++tail;
  }
  const double n = static_cast<double>(values.size());
  stats.mean = sum / n;
  stats.std = std::sqrt(std::max(0.0, sum_squares / n - stats.mean * stats.mean));
  stats.positive_fraction = static_cast<double>(positive) / n;
  stats.odd_positive_fraction = static_cast<double>(odd_positive) / (n / 2.0);
  stats.even_positive_fraction = static_cast<double>(even_positive) / (n / 2.0);
  stats.tail_fraction = static_cast<double>(tail) / n;
  return stats;
}

// Bars for 2^20 draws: the standard error of the mean is ~1e-3, of a
// fraction ~5e-4; the bars are 10x those so a real generator never trips
// them and the historical bug (mean +0.56, std 1.19, sine half all
// non-negative) fails every clause.
bool gate(const Stats &stats, const std::string &label, bool report) {
  const bool pass = stats.finite && std::fabs(stats.mean) < 0.01 &&
                    std::fabs(stats.std - 1.0) < 0.01 &&
                    std::fabs(stats.positive_fraction - 0.5) < 0.01 &&
                    std::fabs(stats.odd_positive_fraction - 0.5) < 0.015 &&
                    std::fabs(stats.even_positive_fraction - 0.5) < 0.015 &&
                    stats.tail_fraction > 0.0020 && stats.tail_fraction < 0.0035;
  if (report)
    std::cout << "GATE noise_stats source=" << label << " mean=" << stats.mean
              << " std=" << stats.std << " positive=" << stats.positive_fraction
              << " odd_positive=" << stats.odd_positive_fraction
              << " even_positive=" << stats.even_positive_fraction
              << " tail3sigma=" << stats.tail_fraction << " pass=" << pass
              << "\n";
  return pass;
}

// The historical bug, reproduced on purpose: a 52-bit mantissa divided by
// 2^53 gives u in [0, 0.5), so theta = 2*pi*u2 spans only [0, pi) and the
// sine half is never negative.
std::vector<float> poisoned_box_muller(std::size_t count, std::uint64_t seed) {
  std::vector<float> out;
  out.reserve(count);
  std::uint64_t state = seed * 6364136223846793005ULL + 1442695040888963407ULL;
  const auto next_bits = [&]() {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    return state;
  };
  while (out.size() < count) {
    const double u1 = static_cast<double>(next_bits() >> 12U) / 9007199254740992.0;
    const double u2 = static_cast<double>(next_bits() >> 12U) / 9007199254740992.0;
    const double radius = std::sqrt(-2.0 * std::log(std::max(u1, 1.0e-300)));
    const double theta = 2.0 * M_PI * u2;
    out.push_back(static_cast<float>(radius * std::cos(theta)));
    out.push_back(static_cast<float>(radius * std::sin(theta)));
  }
  out.resize(count);
  return out;
}

std::filesystem::path workspace() {
  const auto path = std::filesystem::temp_directory_path() / "dif-noise-tests";
  std::filesystem::create_directories(path);
  return path;
}

} // namespace

int main() {
  constexpr std::size_t count = std::size_t{1} << 20U;
  try {
    for (const std::uint64_t seed : {42ULL, 4242ULL, 4243ULL}) {
      const auto values = dif::torch_cpu_normal(count, seed);
      expect(gate(measure(values), "torch-cpu seed=" + std::to_string(seed), true),
             "torch-cpu normal generator passes the noise statistics gate");
    }
    // The gate must be able to fail: the poisoned generator trips it.
    expect(!gate(measure(poisoned_box_muller(count, 42ULL)), "poisoned-box-muller", true),
           "the poisoned Box-Muller generator fails the gate (bitrot control)");

#ifdef DIF_DIFH3NOISE_PATH
    if (dif::runtime::cuda_available()) {
      // The H3 initial-state generator (GPU ChaCha + Box-Muller with
      // cos/sin.approx.ftz) that produced the recorded seeds 4242/4243.
      const auto output = workspace() / "serenity-4242.diftensor";
      std::filesystem::remove(output);
      const std::string command =
          std::string(DIF_DIFH3NOISE_PATH) + " --rng serenity --seed 4242 --layout flat --rows 1024 --cols 1024 --output " +
          output.string() + " > " + (workspace() / "serenity.log").string() + " 2>&1";
      const int status = std::system(command.c_str());
      expect(WIFEXITED(status) && WEXITSTATUS(status) == 0,
             "difh3noise --rng serenity succeeds");
      if (WIFEXITED(status) && WEXITSTATUS(status) == 0) {
        const auto tensor = dif::runtime::read_tensor(output);
        std::vector<float> values(tensor.element_count());
        for (std::size_t index = 0U; index < values.size(); ++index)
          values[index] = dif::runtime::load_float(tensor, index);
        expect(gate(measure(values), "serenity-cuda seed=4242", true),
               "GPU serenity generator passes the noise statistics gate");
      }
    }
#endif
  } catch (const std::exception &error) {
    std::cerr << "FAIL: exception: " << error.what() << "\n";
    ++failures;
  }
  if (failures != 0) {
    std::cerr << failures << " noise test failure(s)\n";
    return 1;
  }
  std::cout << "NOISE_TESTS PASS\n";
  return 0;
}
