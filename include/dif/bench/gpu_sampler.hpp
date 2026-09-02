#pragma once

// Background NVML sampler for the duration of a benchmark run. It needs no
// CUDA context in the measuring process, so it never perturbs the free VRAM
// the workload sees. Everything degrades to "unavailable" without NVML.

#include <cstdint>
#include <memory>
#include <string>

namespace dif::bench {

struct GpuSample {
  bool available{};
  std::string product_name;
  std::uint64_t total_memory_bytes{};
  // Memory in use before the run and the maximum seen during it.
  std::uint64_t used_memory_bytes_before{};
  std::uint64_t peak_used_memory_bytes{};
  // Milliwatts from NVML converted to watts.
  double power_limit_watts{};
  double mean_power_watts{};
  double max_power_watts{};
  std::uint32_t max_temperature_celsius{};
  std::uint64_t sample_count{};
  double interval_seconds{};
};

class GpuSampler {
public:
  explicit GpuSampler(unsigned device_index = 0U,
                      double interval_seconds = 0.2);
  ~GpuSampler();
  GpuSampler(const GpuSampler &) = delete;
  GpuSampler &operator=(const GpuSampler &) = delete;

  bool available() const;
  void start();
  GpuSample stop();

private:
  struct State;
  std::unique_ptr<State> state_;
};

} // namespace dif::bench
