#include "dif/runtime/device_probe.hpp"
#include "dif/runtime/executor.hpp"
#include "dif/support/error.hpp"
#include "dif/target/profile.hpp"
#include "dif/telemetry/schema.hpp"

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>

namespace {

constexpr std::uint64_t mib = 1024ULL * 1024ULL;

std::uint64_t number(const std::string &text, const char *label) {
  char *end = nullptr;
  const auto value = std::strtoull(text.c_str(), &end, 10);
  if (!end || *end != '\0')
    dif::fail(std::string("invalid ") + label);
  return value;
}

std::uint64_t bytes_from_mib(const std::string &text, const char *label) {
  const auto value = number(text, label);
  if (value > std::numeric_limits<std::uint64_t>::max() / mib)
    dif::fail(std::string(label) + " overflows bytes");
  return value * mib;
}

void usage() {
  std::cerr << "usage: difprobe [--backend auto|cpu|cuda] [--device N] [--json]"
               " [--reserve-mib N] [--host-budget-mib N]"
               " [--pinned-budget-mib N] [--workspace-budget-mib N]"
               " [--staging-budget-mib N]\n";
}

} // namespace

int main(int argc, char **argv) {
  try {
    std::string backend = "auto";
    int device = 0;
    bool json = false;
    dif::runtime::BudgetRequest request;
    for (int index = 1; index < argc; ++index) {
      const std::string option = argv[index];
      const auto value = [&]() -> std::string {
        if (index + 1 >= argc)
          dif::fail("missing value after " + option);
        return argv[++index];
      };
      if (option == "--backend")
        backend = value();
      else if (option == "--device") {
        const auto parsed = number(value(), "device ordinal");
        if (parsed > static_cast<std::uint64_t>(std::numeric_limits<int>::max()))
          dif::fail("device ordinal exceeds int range");
        device = static_cast<int>(parsed);
      } else if (option == "--json")
        json = true;
      else if (option == "--reserve-mib")
        request.reserved_device_memory_bytes =
            bytes_from_mib(value(), "device reserve");
      else if (option == "--host-budget-mib")
        request.host_memory_budget_bytes =
            bytes_from_mib(value(), "host budget");
      else if (option == "--pinned-budget-mib")
        request.pinned_host_memory_budget_bytes =
            bytes_from_mib(value(), "pinned budget");
      else if (option == "--workspace-budget-mib")
        request.workspace_budget_bytes =
            bytes_from_mib(value(), "workspace budget");
      else if (option == "--staging-budget-mib")
        request.staging_budget_bytes =
            bytes_from_mib(value(), "staging budget");
      else if (option == "--help" || option == "-h") {
        usage();
        return 0;
      } else {
        dif::fail("unknown difprobe option: " + option);
      }
    }
    dif::runtime::ProbeBackend selected;
    if (backend == "auto")
      selected = dif::runtime::cuda_available()
                     ? dif::runtime::ProbeBackend::Cuda
                     : dif::runtime::ProbeBackend::Host;
    else if (backend == "cuda")
      selected = dif::runtime::ProbeBackend::Cuda;
    else if (backend == "cpu")
      selected = dif::runtime::ProbeBackend::Host;
    else
      dif::fail("difprobe backend accepts auto, cpu, or cuda");
    const auto report = dif::runtime::probe_device(selected, device, request);
    if (json) {
      std::cout << dif::telemetry::serialize_probe(report.target, report.budget);
      return 0;
    }
    const auto &target = report.target;
    const auto &budget = report.budget;
    std::cout << "GPU             " << target.product_name << "\n"
              << "backend         " << target.backend << "\n"
              << "vendor          " << dif::target::vendor_name(target.vendor)
              << "\n"
              << "architecture    "
              << dif::target::architecture_name(target.architecture) << "\n"
              << "compute         ";
    if (target.backend == "cuda")
      std::cout << "sm_" << target.compute_major << target.compute_minor;
    else
      std::cout << "not-applicable";
    std::cout << "\n"
              << "SM count        " << target.multiprocessor_count << "\n"
              << "VRAM total      " << target.total_device_memory_bytes << "\n"
              << "VRAM free       " << budget.free_device_memory_bytes << "\n"
              << "VRAM usable     " << budget.usable_device_memory_bytes << "\n"
              << "BF16 tensor     "
              << (target.precision.bf16_tensor_cores ? "yes" : "no") << "\n"
              << "FP8 tensor      "
              << (target.precision.fp8_tensor_cores ? "yes" : "no") << "\n"
              << "NVFP4 tensor    "
              << (target.precision.nvfp4_tensor_cores ? "yes" : "no") << "\n"
              << "INT8 tensor     "
              << (target.precision.int8_tensor_cores ? "yes" : "no") << "\n"
              << "CUDA driver     " << target.cuda_driver_version << "\n"
              << "CUDA runtime    " << target.cuda_runtime_version << "\n"
              << "cuBLASLt        " << target.cublaslt_version << "\n"
              << "cuDNN           " << target.cudnn_version << "\n"
              << "target          "
              << dif::target::target_fingerprint(target) << "\n"
              << "budget class    "
              << dif::target::runtime_budget_class(budget) << "\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "difprobe: " << error.what() << "\n";
    return 1;
  }
}
