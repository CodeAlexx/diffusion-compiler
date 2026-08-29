#include "dif/backend/abi.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <vector>

namespace {

struct Executable {
  std::vector<float> constant;
};

void set_error(dif_backend_error *error, const char *message) {
  if (error && error->data && error->capacity)
    std::snprintf(error->data, error->capacity, "%s", message);
}

dif_backend_status create(int32_t device, dif_backend_context *context,
                          dif_backend_error *error) {
  if (!context || device != 0) {
    set_error(error, "mock v2 backend accepts device zero");
    return DIF_BACKEND_INVALID_ARGUMENT;
  }
  *context = new (std::nothrow) int(2);
  return *context ? DIF_BACKEND_OK : DIF_BACKEND_OUT_OF_MEMORY;
}

void destroy(dif_backend_context context) { delete static_cast<int *>(context); }

dif_backend_status compile(
    dif_backend_context context, const uint8_t *diffir, size_t size,
    const dif_backend_tensor_view *constants, size_t constant_count,
    const dif_backend_compile_options_v2 *options,
    dif_backend_executable *executable,
    dif_backend_compile_telemetry_v2 *telemetry, dif_backend_error *error) {
  constexpr char magic[] = "DIFIR001";
  if (!context || !diffir || size < 8U || !constants ||
      constant_count != 1U || !options ||
      (options->flags & DIF_BACKEND_COMPILE_COPY_CONSTANTS) == 0U ||
      !executable || !telemetry || std::memcmp(diffir, magic, 8U) != 0 ||
      constants[0].tensor_id != 2U || constants[0].dtype != 1U ||
      constants[0].byte_count % sizeof(float) != 0U) {
    set_error(error, "mock v2 compile contract mismatch");
    return DIF_BACKEND_COMPILE_ERROR;
  }
  auto *result = new (std::nothrow) Executable;
  if (!result)
    return DIF_BACKEND_OUT_OF_MEMORY;
  const auto *source = static_cast<const float *>(constants[0].host_data);
  result->constant.assign(
      source, source + constants[0].byte_count / sizeof(float));
  *executable = result;
  telemetry->preparation_milliseconds = 0.002;
  telemetry->resident_bytes = constants[0].byte_count;
  std::snprintf(telemetry->device_name, sizeof(telemetry->device_name), "%s",
                "mock-v2-device");
  return DIF_BACKEND_OK;
}

void destroy_executable(dif_backend_executable executable) {
  delete static_cast<Executable *>(executable);
}

dif_backend_status execute(dif_backend_executable executable,
                           const dif_backend_tensor_view *inputs,
                           size_t input_count, dif_backend_tensor_view *outputs,
                           size_t output_count,
                           const dif_backend_run_options *options,
                           dif_backend_telemetry *telemetry,
                           dif_backend_error *error) {
  const auto *compiled = static_cast<const Executable *>(executable);
  if (!compiled || !inputs || input_count != 1U || !outputs ||
      output_count != 1U || !options || options->iterations == 0U ||
      !telemetry || inputs[0].tensor_id != 1U || inputs[0].dtype != 1U ||
      outputs[0].dtype != 1U ||
      inputs[0].byte_count != outputs[0].byte_count ||
      inputs[0].byte_count != compiled->constant.size() * sizeof(float)) {
    set_error(error, "mock v2 execute contract mismatch");
    return DIF_BACKEND_INVALID_ARGUMENT;
  }
  const auto *input = static_cast<const float *>(inputs[0].host_data);
  auto *output = static_cast<float *>(outputs[0].host_data);
  for (std::size_t i = 0; i < compiled->constant.size(); ++i)
    output[i] = input[i] + compiled->constant[i];
  telemetry->mean_milliseconds = 0.0005;
  telemetry->minimum_milliseconds = 0.0005;
  telemetry->maximum_milliseconds = 0.0005;
  std::snprintf(telemetry->device_name, sizeof(telemetry->device_name), "%s",
                "mock-v2-device");
  return DIF_BACKEND_OK;
}

const dif_backend_api_v2 api = {
    sizeof(dif_backend_api_v2), DIF_BACKEND_ABI_VERSION_V2, "mock-v2",
    DIF_BACKEND_CAP_F32 | DIF_BACKEND_CAP_ASYNC, create, destroy, compile,
    destroy_executable, execute};

} // namespace

extern "C" DIF_BACKEND_EXPORT const dif_backend_api_v2 *
dif_backend_get_v2(void) {
  return &api;
}
