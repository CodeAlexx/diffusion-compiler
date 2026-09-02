#include "dif/bench/gpu_sampler.hpp"

#include <atomic>
#include <chrono>
#include <dlfcn.h>
#include <thread>

namespace dif::bench {
namespace {

using nvmlReturn = int;
struct nvmlDeviceOpaque;
using nvmlDevice = nvmlDeviceOpaque *;
struct nvmlMemory {
  unsigned long long total;
  unsigned long long free;
  unsigned long long used;
};

struct Nvml {
  void *handle{nullptr};
  nvmlReturn (*init)(){nullptr};
  nvmlReturn (*shutdown)(){nullptr};
  nvmlReturn (*device_by_index)(unsigned, nvmlDevice *){nullptr};
  nvmlReturn (*memory_info)(nvmlDevice, nvmlMemory *){nullptr};
  nvmlReturn (*power_usage)(nvmlDevice, unsigned *){nullptr};
  nvmlReturn (*power_limit)(nvmlDevice, unsigned *){nullptr};
  nvmlReturn (*temperature)(nvmlDevice, int, unsigned *){nullptr};
  nvmlReturn (*name)(nvmlDevice, char *, unsigned){nullptr};

  bool load() {
    handle = ::dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!handle)
      return false;
    const auto resolve = [&](const char *symbol) {
      return ::dlsym(handle, symbol);
    };
    init = reinterpret_cast<decltype(init)>(resolve("nvmlInit_v2"));
    shutdown = reinterpret_cast<decltype(shutdown)>(resolve("nvmlShutdown"));
    device_by_index = reinterpret_cast<decltype(device_by_index)>(
        resolve("nvmlDeviceGetHandleByIndex_v2"));
    memory_info = reinterpret_cast<decltype(memory_info)>(
        resolve("nvmlDeviceGetMemoryInfo"));
    power_usage = reinterpret_cast<decltype(power_usage)>(
        resolve("nvmlDeviceGetPowerUsage"));
    power_limit = reinterpret_cast<decltype(power_limit)>(
        resolve("nvmlDeviceGetPowerManagementLimit"));
    temperature = reinterpret_cast<decltype(temperature)>(
        resolve("nvmlDeviceGetTemperature"));
    name = reinterpret_cast<decltype(name)>(resolve("nvmlDeviceGetName"));
    return init && shutdown && device_by_index && memory_info &&
           power_usage && power_limit && temperature && name;
  }
};

} // namespace

struct GpuSampler::State {
  Nvml nvml;
  nvmlDevice device{nullptr};
  bool available{};
  unsigned device_index{};
  double interval_seconds{};
  std::atomic<bool> running{false};
  std::thread worker;
  GpuSample sample;
  double power_sum_watts{};

  void take() {
    nvmlMemory memory{};
    if (nvml.memory_info(device, &memory) == 0) {
      sample.total_memory_bytes = memory.total;
      if (memory.used > sample.peak_used_memory_bytes)
        sample.peak_used_memory_bytes = memory.used;
    }
    unsigned milliwatts = 0U;
    if (nvml.power_usage(device, &milliwatts) == 0) {
      const auto watts = static_cast<double>(milliwatts) / 1000.0;
      power_sum_watts += watts;
      if (watts > sample.max_power_watts)
        sample.max_power_watts = watts;
    }
    unsigned celsius = 0U;
    if (nvml.temperature(device, 0, &celsius) == 0 &&
        celsius > sample.max_temperature_celsius)
      sample.max_temperature_celsius = celsius;
    ++sample.sample_count;
  }
};

GpuSampler::GpuSampler(unsigned device_index, double interval_seconds)
    : state_(std::make_unique<State>()) {
  state_->device_index = device_index;
  state_->interval_seconds = interval_seconds;
  if (!state_->nvml.load())
    return;
  if (state_->nvml.init() != 0)
    return;
  if (state_->nvml.device_by_index(device_index, &state_->device) != 0) {
    state_->nvml.shutdown();
    return;
  }
  state_->available = true;
  char name[96] = {};
  if (state_->nvml.name(state_->device, name, sizeof(name)) == 0)
    state_->sample.product_name = name;
  unsigned limit = 0U;
  if (state_->nvml.power_limit(state_->device, &limit) == 0)
    state_->sample.power_limit_watts = static_cast<double>(limit) / 1000.0;
  nvmlMemory memory{};
  if (state_->nvml.memory_info(state_->device, &memory) == 0) {
    state_->sample.total_memory_bytes = memory.total;
    state_->sample.used_memory_bytes_before = memory.used;
  }
}

GpuSampler::~GpuSampler() {
  if (state_->running.load())
    (void)stop();
  if (state_->available)
    state_->nvml.shutdown();
  if (state_->nvml.handle)
    ::dlclose(state_->nvml.handle);
}

bool GpuSampler::available() const { return state_->available; }

void GpuSampler::start() {
  if (!state_->available || state_->running.load())
    return;
  state_->sample.available = true;
  state_->sample.interval_seconds = state_->interval_seconds;
  state_->sample.sample_count = 0U;
  state_->sample.peak_used_memory_bytes = 0U;
  state_->sample.max_power_watts = 0.0;
  state_->sample.max_temperature_celsius = 0U;
  state_->power_sum_watts = 0.0;
  state_->running.store(true);
  state_->worker = std::thread([this] {
    const auto interval = std::chrono::duration<double>(
        state_->interval_seconds);
    while (state_->running.load()) {
      state_->take();
      std::this_thread::sleep_for(interval);
    }
  });
}

GpuSample GpuSampler::stop() {
  if (state_->running.load()) {
    state_->running.store(false);
    if (state_->worker.joinable())
      state_->worker.join();
    state_->take();
  }
  auto sample = state_->sample;
  sample.available = state_->available;
  if (sample.sample_count != 0U)
    sample.mean_power_watts =
        state_->power_sum_watts / static_cast<double>(sample.sample_count);
  return sample;
}

} // namespace dif::bench
