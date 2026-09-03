#include "dif/runtime/device_probe.hpp"

#include "dif/support/error.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <sys/sysinfo.h>

#if DIF_HAS_CUDA
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublasLt.h>
#if DIF_HAS_CUDNN
#include <cudnn.h>
#endif
#endif

namespace dif::runtime {
namespace {

std::uint64_t checked_scale(unsigned long value, unsigned long unit) {
  if (unit != 0UL &&
      value > std::numeric_limits<std::uint64_t>::max() / unit)
    fail("host memory probe overflow");
  return static_cast<std::uint64_t>(value) * unit;
}

std::uint64_t available_host_memory() {
  std::ifstream input("/proc/meminfo");
  std::string key;
  std::uint64_t value = 0U;
  std::string unit;
  while (input >> key >> value >> unit) {
    if (key == "MemAvailable:") {
      if (unit != "kB")
        fail("unexpected MemAvailable unit");
      return value * 1024ULL;
    }
  }
  struct sysinfo info {};
  if (sysinfo(&info) != 0)
    fail("cannot query available host memory");
  return checked_scale(info.freeram + info.bufferram, info.mem_unit);
}

std::uint64_t read_cgroup_value(const std::string &path) {
  std::ifstream input(path);
  std::string token;
  if (!(input >> token) || token == "max")
    return 0U;
  try {
    return static_cast<std::uint64_t>(std::stoull(token));
  } catch (...) {
    return 0U;
  }
}

std::uint64_t total_host_memory() {
  struct sysinfo info {};
  if (sysinfo(&info) != 0)
    fail("cannot query total host memory");
  return checked_scale(info.totalram, info.mem_unit);
}

#if DIF_HAS_CUDA
void check(CUresult result, const char *what) {
  if (result == CUDA_SUCCESS)
    return;
  const char *name = nullptr;
  const char *text = nullptr;
  (void)cuGetErrorName(result, &name);
  (void)cuGetErrorString(result, &text);
  fail(std::string(what) + ": " + (name ? name : "CUDA_ERROR") +
       (text ? std::string(" (") + text + ")" : ""));
}

int attribute(CUdevice device, CUdevice_attribute attribute_value,
              const char *what) {
  int value = 0;
  check(cuDeviceGetAttribute(&value, attribute_value, device), what);
  return value;
}

struct ScopedPrimaryContext {
  explicit ScopedPrimaryContext(CUdevice device) : device_(device) {
    check(cuCtxGetCurrent(&previous_), "cuCtxGetCurrent");
    check(cuDevicePrimaryCtxRetain(&context_, device_),
          "cuDevicePrimaryCtxRetain");
    check(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
  }

  ~ScopedPrimaryContext() {
    (void)cuCtxSetCurrent(previous_);
    if (context_)
      (void)cuDevicePrimaryCtxRelease(device_);
  }

  CUdevice device_{};
  CUcontext previous_{};
  CUcontext context_{};
};

CUdevice cuda_device(int ordinal) {
  check(cuInit(0), "cuInit");
  int count = 0;
  check(cuDeviceGetCount(&count), "cuDeviceGetCount");
  if (ordinal < 0 || ordinal >= count)
    fail("CUDA device ordinal is out of range");
  CUdevice device{};
  check(cuDeviceGet(&device, ordinal), "cuDeviceGet");
  return device;
}
#endif

} // namespace

HostCgroupMemory probe_host_cgroup_memory() {
  HostCgroupMemory result;
  std::ifstream input("/proc/self/cgroup");
  std::string line;
  std::string relative;
  while (std::getline(input, line)) {
    if (line.rfind("0::", 0) == 0) {
      relative = line.substr(3);
      break;
    }
  }
  if (relative.empty())
    return result;
  const std::string root = "/sys/fs/cgroup";
  result.current_bytes = read_cgroup_value(root + relative + "/memory.current");
  auto path = relative;
  while (!path.empty()) {
    for (const char *file : {"/memory.max", "/memory.high"}) {
      const auto value = read_cgroup_value(root + path + file);
      if (value != 0U &&
          (result.limit_bytes == 0U || value < result.limit_bytes))
        result.limit_bytes = value;
    }
    const auto slash = path.find_last_of('/');
    if (slash == std::string::npos || slash == 0U)
      break;
    path.resize(slash);
  }
  return result;
}

target::TargetProfile probe_target(ProbeBackend backend, int device_ordinal) {
  target::TargetProfile profile;
  if (backend == ProbeBackend::Host) {
    profile.backend = "cpu";
    profile.vendor = target::Vendor::Host;
    profile.architecture = target::ArchitectureFamily::Host;
    profile.product_name = "host-cpu";
    profile.device_ordinal = -1;
    return profile;
  }
#if DIF_HAS_CUDA
  const auto device = cuda_device(device_ordinal);
  profile.backend = "cuda";
  profile.vendor = target::Vendor::Nvidia;
  profile.device_ordinal = device_ordinal;
  std::array<char, 256> name{};
  check(cuDeviceGetName(name.data(), static_cast<int>(name.size()), device),
        "cuDeviceGetName");
  profile.product_name = name.data();
  profile.compute_major = static_cast<std::uint32_t>(attribute(
      device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
      "cuDeviceGetAttribute compute major"));
  profile.compute_minor = static_cast<std::uint32_t>(attribute(
      device, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
      "cuDeviceGetAttribute compute minor"));
  profile.architecture = target::classify_nvidia_architecture(
      profile.compute_major, profile.compute_minor);
  profile.multiprocessor_count = static_cast<std::uint32_t>(attribute(
      device, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
      "cuDeviceGetAttribute multiprocessor count"));
  profile.warp_size = static_cast<std::uint32_t>(attribute(
      device, CU_DEVICE_ATTRIBUTE_WARP_SIZE,
      "cuDeviceGetAttribute warp size"));
  std::size_t total = 0U;
  check(cuDeviceTotalMem(&total, device), "cuDeviceTotalMem");
  profile.total_device_memory_bytes = total;
  profile.l2_cache_bytes = static_cast<std::uint64_t>(attribute(
      device, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
      "cuDeviceGetAttribute L2 cache size"));
  profile.shared_memory_per_block_bytes =
      static_cast<std::uint64_t>(attribute(
          device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK,
          "cuDeviceGetAttribute shared memory per block"));
  profile.shared_memory_per_block_optin_bytes =
      static_cast<std::uint64_t>(attribute(
          device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
          "cuDeviceGetAttribute opt-in shared memory per block"));
  profile.shared_memory_per_multiprocessor_bytes =
      static_cast<std::uint64_t>(attribute(
          device, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,
          "cuDeviceGetAttribute shared memory per multiprocessor"));
  const auto major = profile.compute_major;
  const auto minor = profile.compute_minor;
  profile.precision.tensor_cores = major >= 7U;
  profile.precision.fp16_tensor_cores = major >= 7U;
  profile.precision.bf16_tensor_cores = major >= 8U;
  profile.precision.fp8_tensor_cores =
      major >= 9U || (major == 8U && minor >= 9U);
  profile.precision.int8_tensor_cores =
      major >= 8U || (major == 7U && minor >= 2U);
  profile.precision.nvfp4_tensor_cores = major >= 10U;
  int driver = 0;
  check(cuDriverGetVersion(&driver), "cuDriverGetVersion");
  profile.cuda_driver_version = static_cast<std::uint32_t>(driver);
  profile.cuda_runtime_version = CUDART_VERSION;
  profile.cublaslt_version = cublasLtGetVersion();
#if DIF_HAS_CUDNN
  profile.cudnn_version = cudnnGetVersion();
#endif
  profile.execution.cuda_graphs = profile.cuda_driver_version >= 10000U;
  profile.execution.async_copy = major >= 8U;
  profile.execution.tensor_memory_accelerator = major >= 9U;
  return profile;
#else
  (void)device_ordinal;
  fail("CUDA device probe requested but CUDA was not built");
#endif
}

target::RuntimeBudget probe_runtime_budget(
    const target::TargetProfile &profile, const BudgetRequest &request) {
  target::RuntimeBudget budget;
  budget.total_host_memory_bytes = total_host_memory();
  budget.available_host_memory_bytes = available_host_memory();
  budget.host_memory_budget_bytes =
      request.host_memory_budget_bytes == 0U
          ? budget.available_host_memory_bytes
          : std::min(request.host_memory_budget_bytes,
                     budget.available_host_memory_bytes);
  budget.pinned_host_memory_budget_bytes = std::min(
      request.pinned_host_memory_budget_bytes, budget.host_memory_budget_bytes);
  budget.workspace_budget_bytes = request.workspace_budget_bytes;
  budget.staging_budget_bytes =
      std::min({request.staging_budget_bytes,
                budget.pinned_host_memory_budget_bytes,
                budget.host_memory_budget_bytes});
  if (profile.backend != "cuda")
    return budget;
#if DIF_HAS_CUDA
  const auto device = cuda_device(profile.device_ordinal);
  ScopedPrimaryContext context(device);
  std::size_t free = 0U;
  std::size_t total = 0U;
  check(cuMemGetInfo(&free, &total), "cuMemGetInfo");
  if (profile.total_device_memory_bytes != 0U &&
      total != profile.total_device_memory_bytes)
    fail("CUDA target total memory changed between target and budget probes");
  budget.free_device_memory_bytes = free;
  budget.reserved_device_memory_bytes = request.reserved_device_memory_bytes;
  budget.usable_device_memory_bytes =
      free > request.reserved_device_memory_bytes
          ? free - request.reserved_device_memory_bytes
          : 0U;
  return budget;
#else
  fail("CUDA runtime budget requested but CUDA was not built");
#endif
}

DeviceProbeResult probe_device(ProbeBackend backend, int device_ordinal,
                               const BudgetRequest &request) {
  DeviceProbeResult result;
  result.target = probe_target(backend, device_ordinal);
  result.budget = probe_runtime_budget(result.target, request);
  return result;
}

} // namespace dif::runtime
