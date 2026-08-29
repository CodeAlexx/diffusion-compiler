#include "dif/backend/abi.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <new>

namespace {

void set_error(dif_backend_error *error, const char *message) {
  if (error && error->data && error->capacity)
    std::snprintf(error->data, error->capacity, "%s", message);
}

dif_backend_status create(int32_t device, dif_backend_context *context,
                          dif_backend_error *error) {
  if (!context || device != 0) {
    set_error(error, "mock backend accepts device zero");
    return DIF_BACKEND_INVALID_ARGUMENT;
  }
  *context = new (std::nothrow) int(1);
  return *context ? DIF_BACKEND_OK : DIF_BACKEND_OUT_OF_MEMORY;
}

void destroy(dif_backend_context context) { delete static_cast<int *>(context); }

dif_backend_status compile(dif_backend_context context, const uint8_t *diffir,
                           size_t size, dif_backend_executable *executable,
                           dif_backend_error *error) {
  constexpr char magic[] = "DIFIR001";
  if (!context || !diffir || size < 8U || !executable ||
      std::memcmp(diffir, magic, 8U) != 0) {
    set_error(error, "mock compile rejected input");
    return DIF_BACKEND_COMPILE_ERROR;
  }
  *executable = new (std::nothrow) int(2);
  return *executable ? DIF_BACKEND_OK : DIF_BACKEND_OUT_OF_MEMORY;
}

void destroy_executable(dif_backend_executable executable) {
  delete static_cast<int *>(executable);
}

dif_backend_status execute(dif_backend_executable executable,
                           const dif_backend_tensor_view *inputs,
                           size_t input_count, dif_backend_tensor_view *outputs,
                           size_t output_count,
                           const dif_backend_run_options *options,
                           dif_backend_telemetry *telemetry,
                           dif_backend_error *error) {
  if (!executable || !inputs || input_count != 2U || !outputs ||
      output_count != 1U || !options || options->iterations == 0U || !telemetry) {
    set_error(error, "mock execute contract mismatch");
    return DIF_BACKEND_INVALID_ARGUMENT;
  }
  if (inputs[0].dtype != 1U || inputs[1].dtype != 1U || outputs[0].dtype != 1U ||
      inputs[0].byte_count != inputs[1].byte_count ||
      inputs[0].byte_count != outputs[0].byte_count ||
      (inputs[0].byte_count % sizeof(float)) != 0U) {
    set_error(error, "mock execute tensor mismatch");
    return DIF_BACKEND_INVALID_ARGUMENT;
  }
  const auto *a = static_cast<const float *>(inputs[0].host_data);
  const auto *b = static_cast<const float *>(inputs[1].host_data);
  auto *output = static_cast<float *>(outputs[0].host_data);
  const auto count = inputs[0].byte_count / sizeof(float);
  for (uint64_t i = 0; i < count; ++i)
    output[i] = a[i] + b[i];
  telemetry->mean_milliseconds = 0.001;
  telemetry->minimum_milliseconds = 0.001;
  telemetry->maximum_milliseconds = 0.001;
  std::snprintf(telemetry->device_name, sizeof(telemetry->device_name),
                "%s", "mock-device");
  return DIF_BACKEND_OK;
}

const dif_backend_api_v1 api = {
    sizeof(dif_backend_api_v1), DIF_BACKEND_ABI_VERSION, "mock-v1",
    DIF_BACKEND_CAP_F32, create, destroy, compile, destroy_executable, execute};

} // namespace

extern "C" DIF_BACKEND_EXPORT const dif_backend_api_v1 *dif_backend_get_v1(void) {
  return &api;
}
