// SDXL discrete sigma schedule tests.
//
// Compares include/dif/sampling/discrete_schedule.hpp against fixtures
// recorded from the reference sampler by
// tools/export_sdxl_schedule_reference.py (tests/sdxl_schedule_fixtures.inc).
//
// Gate policy. Everything the reference computes with IEEE add/sub/mul/div
// or its own control flow is compared bit-for-bit as float32: the 1000-entry
// sigma and log-sigma tables, sigma_min/max, the fractional timestep grid of
// every schedule, the log-sigma interpolant behind sigma(t), the argmin
// timesteps, the UNet timesteps and max_denoise. The reference's final
// float32 exp and sqrt run through its vector math library (Intel MKL VML,
// high-accuracy mode), which is within one ulp but not correctly rounded and
// cannot be reproduced from libm; those outputs (sigma(t), the schedule
// sigmas, the epsilon-input divisor) are gated at a maximum of one float32
// ulp, and the test reports how many of them matched exactly. The terminal
// zero and every integer-valued output stay bit-exact.

#include "dif/sampling/discrete_schedule.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <string>
#include <vector>

#include "sdxl_schedule_fixtures.inc"

namespace {

int failures = 0;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    ++failures;
    std::cerr << "FAIL: " << message << "\n";
  }
}

std::uint32_t bits_of(float value) {
  std::uint32_t bits{};
  std::memcpy(&bits, &value, sizeof bits);
  return bits;
}

bool bit_equal(float lhs, float rhs) { return bits_of(lhs) == bits_of(rhs); }

// Distance in float32 ulps between two finite values of the same sign; a
// sign change or a non-finite operand counts as unbounded.
std::uint64_t ulp_distance(float lhs, float rhs) {
  if (!std::isfinite(lhs) || !std::isfinite(rhs))
    return UINT64_MAX;
  if (bit_equal(lhs, rhs))
    return 0U;
  if ((lhs < 0.0F) != (rhs < 0.0F) && lhs != 0.0F && rhs != 0.0F)
    return UINT64_MAX;
  const auto a = static_cast<std::int64_t>(bits_of(std::fabs(lhs)));
  const auto b = static_cast<std::int64_t>(bits_of(std::fabs(rhs)));
  return static_cast<std::uint64_t>(a > b ? a - b : b - a);
}

std::string hex(float value) {
  char buffer[32];
  std::snprintf(buffer, sizeof buffer, "%a", static_cast<double>(value));
  return buffer;
}

struct UlpTally {
  std::uint64_t exact{};
  std::uint64_t one_ulp{};
  std::uint64_t worst{};
};

void expect_within_one_ulp(float actual, float reference,
                           const std::string &message, UlpTally &tally) {
  const auto distance = ulp_distance(actual, reference);
  tally.worst = std::max(tally.worst, distance);
  if (distance == 0U)
    ++tally.exact;
  else if (distance == 1U)
    ++tally.one_ulp;
  expect(distance <= 1U, message + ": got " + hex(actual) + " want " +
                             hex(reference) + " (" +
                             std::to_string(distance) + " ulp)");
}

void report(const char *what, const UlpTally &tally) {
  std::cout << what << ": " << tally.exact << " bit-exact, " << tally.one_ulp
            << " within one ulp, worst " << tally.worst << " ulp\n";
}

void test_tables(const dif::sampling::DiscreteSigmaTable &table) {
  using namespace sdxl_schedule_fixtures;
  expect(table.sigmas.size() == kTimesteps, "sigma table size");
  expect(table.log_sigmas.size() == kTimesteps, "log-sigma table size");
  if (table.sigmas.size() != kTimesteps ||
      table.log_sigmas.size() != kTimesteps)
    return;
  std::uint32_t sigma_mismatches = 0;
  std::uint32_t log_mismatches = 0;
  for (std::uint32_t index = 0; index < kTimesteps; ++index) {
    if (!bit_equal(table.sigmas[index], kSigmas[index])) {
      ++sigma_mismatches;
      expect(false, "sigmas[" + std::to_string(index) + "] got " +
                        hex(table.sigmas[index]) + " want " +
                        hex(kSigmas[index]));
    }
    if (!bit_equal(table.log_sigmas[index], kLogSigmas[index])) {
      ++log_mismatches;
      expect(false, "log_sigmas[" + std::to_string(index) + "] got " +
                        hex(table.log_sigmas[index]) + " want " +
                        hex(kLogSigmas[index]));
    }
  }
  std::cout << "tables: " << (kTimesteps - sigma_mismatches)
            << "/1000 sigmas and " << (kTimesteps - log_mismatches)
            << "/1000 log-sigmas bit-exact\n";
  expect(bit_equal(table.sigma_min(), kSigmaMin), "sigma_min");
  expect(bit_equal(table.sigma_max(), kSigmaMax), "sigma_max");
  // The table ascends strictly, so argmin lookups are unambiguous.
  for (std::uint32_t index = 1; index < kTimesteps; ++index)
    expect(table.sigmas[index - 1U] < table.sigmas[index],
           "sigma table ascends at " + std::to_string(index));
}

void test_sigma_at(const dif::sampling::DiscreteSigmaTable &table) {
  using namespace sdxl_schedule_fixtures;
  UlpTally tally;
  for (const auto &item : kSigmaAtCases) {
    const std::string tag = "sigma_at(" + hex(item.timestep) + ")";
    expect(bit_equal(table.log_sigma_at(item.timestep), item.log_sigma),
           tag + " log-sigma interpolant got " +
               hex(table.log_sigma_at(item.timestep)) + " want " +
               hex(item.log_sigma));
    expect_within_one_ulp(table.sigma_at(item.timestep), item.sigma, tag,
                          tally);
  }
  report("sigma_at spot checks", tally);
}

void test_timestep_for(const dif::sampling::DiscreteSigmaTable &table) {
  using namespace sdxl_schedule_fixtures;
  for (const auto &item : kTimestepForCases) {
    const auto got = table.timestep_for(item.sigma);
    expect(got == item.timestep, "timestep_for(" + hex(item.sigma) +
                                     ") got " + std::to_string(got) +
                                     " want " + std::to_string(item.timestep));
  }
  // Endpoints used by the normal scheduler.
  expect(table.timestep_for(table.sigma_max()) == kTimesteps - 1U,
         "timestep_for(sigma_max)");
  expect(table.timestep_for(table.sigma_min()) == 0U,
         "timestep_for(sigma_min)");
}

void test_schedules(const dif::sampling::DiscreteSigmaTable &table) {
  using namespace sdxl_schedule_fixtures;
  UlpTally sigma_tally;
  UlpTally divisor_tally;
  for (const auto &item : kScheduleCases) {
    const std::string tag = "steps=" + std::to_string(item.steps) + " ";
    const std::span<const float> want_timesteps(item.timesteps, item.steps);
    const std::span<const float> want_log_sigmas(item.log_sigmas, item.steps);
    const std::span<const float> want_sigmas(item.sigmas, item.sigma_count);
    const std::span<const float> want_unet(item.unet_timesteps, item.steps);
    const std::span<const float> want_divisors(item.input_divisors,
                                               item.steps);

    expect(table.timestep_for(table.sigma_max()) == item.start_timestep,
           tag + "start timestep");
    expect(table.timestep_for(table.sigma_min()) == item.end_timestep,
           tag + "end timestep");

    const auto grid = dif::sampling::normal_schedule_timesteps(table,
                                                               item.steps);
    expect(grid.size() == item.steps, tag + "grid size");
    for (std::size_t index = 0; index < grid.size() && index < item.steps;
         ++index) {
      expect(bit_equal(grid[index], want_timesteps[index]),
             tag + "grid[" + std::to_string(index) + "] got " +
                 hex(grid[index]) + " want " + hex(want_timesteps[index]));
      expect(bit_equal(table.log_sigma_at(grid[index]),
                       want_log_sigmas[index]),
             tag + "log-sigma[" + std::to_string(index) + "] got " +
                 hex(table.log_sigma_at(grid[index])) + " want " +
                 hex(want_log_sigmas[index]));
    }

    const auto sigmas = dif::sampling::normal_schedule(table, item.steps);
    expect(sigmas.size() == item.sigma_count, tag + "sigma count");
    expect(sigmas.size() == item.steps + 1U, tag + "steps + 1 sigmas");
    if (sigmas.size() != item.sigma_count)
      continue;
    for (std::size_t index = 0; index + 1U < sigmas.size(); ++index)
      expect_within_one_ulp(sigmas[index], want_sigmas[index],
                            tag + "sigma[" + std::to_string(index) + "]",
                            sigma_tally);
    expect(bit_equal(sigmas.back(), 0.0F) &&
               bit_equal(want_sigmas.back(), 0.0F),
           tag + "terminal sigma is +0");
    for (std::size_t index = 1; index < sigmas.size(); ++index)
      expect(sigmas[index - 1U] > sigmas[index],
             tag + "sigmas descend at " + std::to_string(index));

    // The UNet timesteps must match whether computed from the native
    // schedule or from the recorded reference sigmas.
    const auto unet = dif::sampling::unet_timesteps(table, sigmas);
    const auto unet_from_reference = dif::sampling::unet_timesteps(
        table, std::vector<float>(want_sigmas.begin(), want_sigmas.end()));
    expect(unet.size() == item.steps, tag + "unet timestep count");
    for (std::size_t index = 0; index < unet.size() && index < item.steps;
         ++index) {
      expect(bit_equal(unet[index], want_unet[index]),
             tag + "unet timestep[" + std::to_string(index) + "] got " +
                 hex(unet[index]) + " want " + hex(want_unet[index]));
      expect(bit_equal(unet_from_reference[index], want_unet[index]),
             tag + "unet timestep from reference sigma[" +
                 std::to_string(index) + "]");
      expect(unet[index] == std::floor(unet[index]),
             tag + "unet timestep is integer-valued");
    }

    expect(dif::sampling::max_denoise(table, sigmas) == item.max_denoise,
           tag + "max_denoise");

    for (std::size_t index = 0; index < item.steps; ++index) {
      const float sigma = sigmas[index];
      expect_within_one_ulp(dif::sampling::eps_input_divisor(sigma),
                            want_divisors[index],
                            tag + "input divisor[" + std::to_string(index) +
                                "]",
                            divisor_tally);
      expect(bit_equal(dif::sampling::initial_noise_scale(sigma, true),
                       dif::sampling::eps_input_divisor(sigma)),
             tag + "max-denoise noise scale equals the input divisor");
      expect(bit_equal(dif::sampling::initial_noise_scale(sigma, false),
                       sigma),
             tag + "plain noise scale is sigma");
      expect(bit_equal(dif::sampling::eps_input_scale(sigma),
                       1.0F / dif::sampling::eps_input_divisor(sigma)),
             tag + "eps_input_scale is the reciprocal divisor");
    }
  }
  report("normal schedule sigmas", sigma_tally);
  report("epsilon input divisors", divisor_tally);
}

void test_edge_cases(const dif::sampling::DiscreteSigmaTable &table) {
  // Clamping: out-of-range timesteps land on the end entries.
  expect(bit_equal(table.log_sigma_at(-1.0F), table.log_sigmas.front()),
         "clamp below zero");
  expect(bit_equal(table.log_sigma_at(5000.0F), table.log_sigmas.back()),
         "clamp above last");
  // Integer timesteps interpolate to the table entry itself.
  for (std::uint32_t index : {0U, 1U, 499U, 998U, 999U})
    expect(bit_equal(table.log_sigma_at(static_cast<float>(index)),
                     table.log_sigmas[index]),
           "integer timestep " + std::to_string(index));
  // Every table sigma maps back to its own index.
  for (std::uint32_t index = 0; index < table.sigmas.size(); ++index)
    expect(table.timestep_for(table.sigmas[index]) == index,
           "round trip timestep " + std::to_string(index));
  bool threw = false;
  try {
    (void)table.timestep_for(-1.0F);
  } catch (const dif::Error &) {
    threw = true;
  }
  expect(threw, "negative sigma rejected");
  threw = false;
  try {
    (void)dif::sampling::normal_schedule(table, 0U);
  } catch (const dif::Error &) {
    threw = true;
  }
  expect(threw, "zero steps rejected");
}

} // namespace

int main() {
  const auto table = dif::sampling::DiscreteSigmaTable::linear();
  test_tables(table);
  test_sigma_at(table);
  test_timestep_for(table);
  test_schedules(table);
  test_edge_cases(table);
  if (failures != 0) {
    std::cerr << failures << " failure(s)\n";
    return 1;
  }
  std::cout << "sdxl schedule tests passed\n";
  return 0;
}
