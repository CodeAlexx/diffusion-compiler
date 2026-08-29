#include "dif/runtime/tensor.hpp"
#include "dif/sampling/rectified_flow.hpp"
#include "dif/support/error.hpp"

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

std::uint32_t points(const char *text) {
  char *end = nullptr;
  errno = 0;
  const auto value = std::strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || value < 2U ||
      value > std::numeric_limits<std::uint32_t>::max())
    dif::fail("schedule points must be an integer in [2,2^32-1]");
  return static_cast<std::uint32_t>(value);
}

float positive_float(const char *text) {
  char *end = nullptr;
  errno = 0;
  const auto value = std::strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0' || !(value > 0.0F) ||
      !std::isfinite(value))
    dif::fail("schedule shift must be finite and positive");
  return value;
}

float sigma_float(const char *text) {
  char *end = nullptr;
  errno = 0;
  const auto value = std::strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0' || value < 0.0F ||
      !std::isfinite(value))
    dif::fail("sigma must be finite and nonnegative");
  return value;
}

dif::runtime::Tensor tensor(const std::vector<float> &values) {
  dif::runtime::Tensor result{dif::ir::DType::F32,
                              {static_cast<std::uint64_t>(values.size())}, {}};
  result.bytes.resize(values.size() * sizeof(float));
  std::memcpy(result.bytes.data(), values.data(), result.bytes.size());
  result.validate();
  return result;
}

void usage() {
  std::cerr << "usage: difschedule make-exponential-shifted SIGMAS.diftensor "
               "TIMESTEPS.diftensor POINTS SHIFT\n"
               "       difschedule make-explicit SIGMAS.diftensor SIGMA SIGMA [SIGMA ...]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc >= 5 && std::string(argv[1]) == "make-explicit") {
      std::vector<float> sigmas;
      sigmas.reserve(static_cast<std::size_t>(argc - 3));
      for (int index = 3; index < argc; ++index)
        sigmas.push_back(sigma_float(argv[index]));
      for (std::size_t index = 1U; index < sigmas.size(); ++index) {
        if (sigmas[index] > sigmas[index - 1U])
          dif::fail("explicit sigmas must be nonincreasing");
      }
      if (sigmas.back() != 0.0F)
        dif::fail("explicit sigma schedule must end at zero");
      dif::runtime::write_tensor(tensor(sigmas), argv[2]);
      std::cout << "SCHEDULE sigmas=" << argv[2]
                << " points=" << sigmas.size()
                << " evaluations=" << sigmas.size() - 1U
                << " kind=explicit dtype=f32\n";
      return 0;
    }
    if (argc != 6 || std::string(argv[1]) != "make-exponential-shifted") {
      usage();
      return 2;
    }
    const auto point_count = points(argv[4]);
    const auto shift = positive_float(argv[5]);
    const auto schedule =
        dif::sampling::make_exponential_shifted_schedule(point_count, shift);
    dif::runtime::write_tensor(tensor(schedule.sigmas), argv[2]);
    dif::runtime::write_tensor(tensor(schedule.timesteps), argv[3]);
    std::cout << "SCHEDULE sigmas=" << argv[2] << " timesteps=" << argv[3]
              << " requested_points=" << point_count
              << " retained_points=" << schedule.sigmas.size()
              << " evaluations=" << schedule.timesteps.size()
              << " shift=" << shift << " dtype=f32\n";
    return 0;
  } catch (const std::exception &exception) {
    std::cerr << "difschedule: " << exception.what() << "\n";
    return 1;
  }
}
