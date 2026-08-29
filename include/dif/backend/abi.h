#ifndef DIF_BACKEND_ABI_H
#define DIF_BACKEND_ABI_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DIF_BACKEND_ABI_VERSION_V1 1U
#define DIF_BACKEND_ABI_VERSION_V2 2U
/* Kept for source compatibility with v1 plugins. */
#define DIF_BACKEND_ABI_VERSION DIF_BACKEND_ABI_VERSION_V1

#if defined(_WIN32)
#define DIF_BACKEND_EXPORT __declspec(dllexport)
#else
#define DIF_BACKEND_EXPORT __attribute__((visibility("default")))
#endif

typedef enum dif_backend_status {
  DIF_BACKEND_OK = 0,
  DIF_BACKEND_INVALID_ARGUMENT = 1,
  DIF_BACKEND_UNSUPPORTED = 2,
  DIF_BACKEND_OUT_OF_MEMORY = 3,
  DIF_BACKEND_COMPILE_ERROR = 4,
  DIF_BACKEND_EXECUTION_ERROR = 5,
  DIF_BACKEND_INTERNAL_ERROR = 6
} dif_backend_status;

typedef enum dif_backend_capability {
  DIF_BACKEND_CAP_F32 = 1ULL << 0U,
  DIF_BACKEND_CAP_BF16 = 1ULL << 1U,
  DIF_BACKEND_CAP_F16 = 1ULL << 2U,
  DIF_BACKEND_CAP_I8 = 1ULL << 3U,
  DIF_BACKEND_CAP_TRAINING = 1ULL << 4U,
  DIF_BACKEND_CAP_ASYNC = 1ULL << 5U
} dif_backend_capability;

typedef struct dif_backend_error {
  char *data;
  size_t capacity;
} dif_backend_error;

typedef struct dif_backend_tensor_view {
  uint32_t tensor_id;
  uint32_t dtype;
  uint32_t rank;
  const uint64_t *dims;
  void *host_data;
  uint64_t byte_count;
} dif_backend_tensor_view;

typedef struct dif_backend_run_options {
  uint32_t warmups;
  uint32_t iterations;
  uint64_t minimum_free_bytes;
} dif_backend_run_options;

typedef struct dif_backend_telemetry {
  double mean_milliseconds;
  double minimum_milliseconds;
  double maximum_milliseconds;
  uint64_t free_bytes_before;
  uint64_t free_bytes_after;
  char device_name[256];
} dif_backend_telemetry;

typedef void *dif_backend_context;
typedef void *dif_backend_executable;

typedef struct dif_backend_api_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char *backend_name;
  uint64_t capabilities;

  dif_backend_status (*create)(int32_t device_ordinal,
                               dif_backend_context *context,
                               dif_backend_error *error);
  void (*destroy)(dif_backend_context context);
  dif_backend_status (*compile)(dif_backend_context context,
                                const uint8_t *diffir,
                                size_t diffir_size,
                                dif_backend_executable *executable,
                                dif_backend_error *error);
  void (*destroy_executable)(dif_backend_executable executable);
  dif_backend_status (*execute)(dif_backend_executable executable,
                                const dif_backend_tensor_view *inputs,
                                size_t input_count,
                                dif_backend_tensor_view *outputs,
                                size_t output_count,
                                const dif_backend_run_options *options,
                                dif_backend_telemetry *telemetry,
                                dif_backend_error *error);
} dif_backend_api_v1;

typedef const dif_backend_api_v1 *(*dif_backend_get_v1_fn)(void);

DIF_BACKEND_EXPORT const dif_backend_api_v1 *dif_backend_get_v1(void);

typedef enum dif_backend_compile_flag {
  /* The executable owns/copies constants before compile returns. */
  DIF_BACKEND_COMPILE_COPY_CONSTANTS = 1ULL << 0U
} dif_backend_compile_flag;

typedef struct dif_backend_compile_options_v2 {
  uint64_t minimum_free_bytes;
  uint64_t flags;
} dif_backend_compile_options_v2;

typedef struct dif_backend_compile_telemetry_v2 {
  double preparation_milliseconds;
  uint64_t resident_bytes;
  uint64_t free_bytes_before;
  uint64_t free_bytes_after;
  char device_name[256];
} dif_backend_compile_telemetry_v2;

/*
 * v2 moves immutable constants into compile(). This is the ownership boundary
 * a GPU plugin needs to create a reusable, resident executable. execute()
 * receives only dynamic graph inputs and synchronously materializes requested
 * host outputs; a backend may schedule asynchronous work internally.
 */
typedef struct dif_backend_api_v2 {
  uint32_t struct_size;
  uint32_t abi_version;
  const char *backend_name;
  uint64_t capabilities;

  dif_backend_status (*create)(int32_t device_ordinal,
                               dif_backend_context *context,
                               dif_backend_error *error);
  void (*destroy)(dif_backend_context context);
  dif_backend_status (*compile)(
      dif_backend_context context, const uint8_t *diffir, size_t diffir_size,
      const dif_backend_tensor_view *constants, size_t constant_count,
      const dif_backend_compile_options_v2 *options,
      dif_backend_executable *executable,
      dif_backend_compile_telemetry_v2 *telemetry,
      dif_backend_error *error);
  void (*destroy_executable)(dif_backend_executable executable);
  dif_backend_status (*execute)(dif_backend_executable executable,
                                const dif_backend_tensor_view *dynamic_inputs,
                                size_t dynamic_input_count,
                                dif_backend_tensor_view *outputs,
                                size_t output_count,
                                const dif_backend_run_options *options,
                                dif_backend_telemetry *telemetry,
                                dif_backend_error *error);
} dif_backend_api_v2;

typedef const dif_backend_api_v2 *(*dif_backend_get_v2_fn)(void);

/* Optional symbol. Hosts fall back to dif_backend_get_v1 when it is absent. */
DIF_BACKEND_EXPORT const dif_backend_api_v2 *dif_backend_get_v2(void);

#ifdef __cplusplus
}
#endif

#endif
