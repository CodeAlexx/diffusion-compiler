cmake_minimum_required(VERSION 3.24)

project(diffusion_compiler VERSION 0.1.0 LANGUAGES CXX)

execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE DIF_COMPILER_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
string(LENGTH "${DIF_COMPILER_REVISION}" DIF_COMPILER_REVISION_LENGTH)
if(NOT DIF_COMPILER_REVISION_LENGTH EQUAL 40)
  set(DIF_COMPILER_REVISION "unknown")
endif()

find_package(PNG REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DIF_ENABLE_CUDA "Build the CUDA Driver/NVRTC backend" ON)
option(DIF_ENABLE_CUDNN "Build optional cuDNN SDPA lowering" ON)
option(DIF_BUILD_TESTS "Build unit tests" ON)
option(DIF_ENABLE_OPENCL "Build the vendor-neutral OpenCL reference backend" ON)
set(DIF_CUDNN_ROOT "" CACHE PATH "cuDNN installation root (include/ and lib/)")
set(DIF_CUTLASS_ROOT "" CACHE PATH "CUTLASS source root (contains include/cutlass)")
set(DIF_FLASH_ATTENTION_ROOT "" CACHE PATH
    "FlashAttention csrc/flash_attn root (contains src/flash.h)")
set(DIF_CUDA_ARCHITECTURES "native" CACHE STRING
    "CUDA architectures for compiled kernels (for example 86 or 120)")

# CUDA kernel templates (src/compiler/kernels/*.cu) are embedded at build time;
# the generated table is rebuilt whenever a template or the script changes.
file(GLOB DIF_KERNEL_TEMPLATE_FILES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/kernels/*.cu")
set(DIF_KERNEL_TEMPLATES_CPP "${CMAKE_CURRENT_BINARY_DIR}/generated/kernel_templates_generated.cpp")
add_custom_command(
    OUTPUT "${DIF_KERNEL_TEMPLATES_CPP}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
    COMMAND ${CMAKE_COMMAND}
            -DKERNEL_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/compiler/kernels
            -DOUTPUT=${DIF_KERNEL_TEMPLATES_CPP}
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernels.cmake"
    DEPENDS ${DIF_KERNEL_TEMPLATE_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_kernels.cmake"
    COMMENT "Embedding CUDA kernel templates"
    VERBATIM)

add_library(difcore
    "${DIF_KERNEL_TEMPLATES_CPP}"
    src/compiler/kernel_template.cpp
    src/build_info.cpp
    src/backend/plugin.cpp
    src/compiler/compiler.cpp
    src/compiler/int4.cpp
    src/compiler/layout_plan.cpp
    src/compiler/memory_plan.cpp
    src/compiler/residency_plan.cpp
    src/compiler/slice.cpp
  src/frontend/h3.cpp
  src/frontend/h3_conditioning.cpp
  src/frontend/h3_latents.cpp
  src/frontend/h3_media.cpp
  src/frontend/h3_vae.cpp
  src/frontend/h3_video_encoder.cpp
  src/frontend/h3_audio_vae.cpp
  src/frontend/qwen3vl_conditioner.cpp
  src/frontend/qwen3vl_vision.cpp
  src/frontend/krea2.cpp
  src/frontend/provenance.cpp
  src/frontend/krea2_vae.cpp
  src/frontend/flux2.cpp
  src/frontend/squareq_w4.cpp
  src/frontend/flux2_vae.cpp
  src/frontend/sdxl_vae.cpp
  src/frontend/sdxl_clip.cpp
  src/frontend/sdxl_unet.cpp
  src/frontend/dit_block.cpp
  src/frontend/dit_lora.cpp
  src/frontend/lora.cpp
  src/frontend/training.cpp
    src/ir/codec.cpp
    src/ir/opcodes.cpp
    src/ir/verify.cpp
    src/opt/bindings.cpp
    src/opt/gate.cpp
    src/opt/physical_format.cpp
    src/opt/plan.cpp
    src/opt/rewrite.cpp
    src/opt/search.cpp
    src/opt/semantics.cpp
    src/opt/transform.cpp
    src/runtime/cpu_executor.cpp
    src/runtime/device_probe.cpp
    src/runtime/executor.cpp
    src/runtime/scalar.cpp
    src/runtime/tensor_io.cpp
    src/sampling/rectified_flow.cpp
    src/support/sha256.cpp
    src/support/wav.cpp
    src/support/json.cpp
    src/support/png.cpp
    src/support/torch_cpu_rng.cpp
    src/target/profile.cpp
    src/telemetry/document.cpp
    src/telemetry/schema.cpp
    src/telemetry/trace_sink.cpp
    src/bench/artifact.cpp
    src/bench/gpu_sampler.cpp
    src/bench/host.cpp
    src/bench/process.cpp
    src/bench/recipe.cpp
    src/bench/report.cpp
    src/quality/metrics.cpp
    src/tune/database.cpp
    src/training/autodiff.cpp
    src/training/checkpoint.cpp
    src/weights/safetensors.cpp
    src/weights/bundle.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_sources(difcore PRIVATE src/support/torch_cpu_rng_avx2.cpp)
  set_source_files_properties(
      src/support/torch_cpu_rng_avx2.cpp
      PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
  target_compile_definitions(difcore PRIVATE DIF_TORCH_CPU_RNG_AVX2=1)
else()
  target_compile_definitions(difcore PRIVATE DIF_TORCH_CPU_RNG_AVX2=0)
endif()
target_include_directories(difcore PUBLIC include)
target_compile_definitions(
    difcore PRIVATE
    DIF_COMPILER_VERSION="${PROJECT_VERSION}"
    DIF_COMPILER_REVISION="${DIF_COMPILER_REVISION}")
target_compile_options(
    difcore PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    # NVCC's generated host stubs use GCC-style line directives that trip
    # host -Wpedantic. Keep all actionable CUDA/host warnings fatal.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
target_link_libraries(difcore PRIVATE ${CMAKE_DL_LIBS} PNG::PNG)

if(DIF_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
endif()

if(DIF_ENABLE_CUDA AND CUDAToolkit_FOUND)
  # FindCUDAToolkit may expose multiple include directories (CUDA 13 also
  # appends include/cccl).  The NVRTC helper expects one quoted directory;
  # injecting the CMake list into a compile definition breaks shell quoting.
  list(GET CUDAToolkit_INCLUDE_DIRS 0 DIF_CUDA_PRIMARY_INCLUDE_DIR)
  target_sources(difcore PRIVATE src/runtime/cuda_executor.cpp)
  target_link_libraries(difcore PRIVATE CUDA::cuda_driver CUDA::nvrtc
                                        CUDA::cublas CUDA::cublasLt)
  target_compile_definitions(
      difcore PUBLIC DIF_HAS_CUDA=1
      PRIVATE DIF_CUDA_INCLUDE_DIR="${DIF_CUDA_PRIMARY_INCLUDE_DIR}")
  # NVTX v3 is header-only; ranges are inert unless a tool such as Nsight
  # Systems injects an implementation at run time.
  find_path(DIF_NVTX_INCLUDE_DIR nvtx3/nvToolsExt.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS})
  if(DIF_NVTX_INCLUDE_DIR)
    target_include_directories(difcore SYSTEM PRIVATE "${DIF_NVTX_INCLUDE_DIR}")
    target_compile_definitions(difcore PRIVATE DIF_HAS_NVTX=1)
    message(STATUS "Diffusion compiler NVTX ranges: ${DIF_NVTX_INCLUDE_DIR}")
  else()
    target_compile_definitions(difcore PRIVATE DIF_HAS_NVTX=0)
    message(STATUS "Diffusion compiler NVTX ranges: disabled")
  endif()
  find_path(DIF_CUTLASS_INCLUDE_DIR cutlass/cutlass.h
            HINTS "${DIF_CUTLASS_ROOT}" ENV CUTLASS_ROOT
            PATH_SUFFIXES include)
  if(DIF_CUTLASS_INCLUDE_DIR)
    enable_language(CUDA)
    target_sources(difcore PRIVATE src/runtime/cutlass_gemm.cu)
    # CUTLASS 2.x epilogue visitors use CUTE constexpr tuple/move helpers from
    # device callbacks and require relaxed constexpr under CUDA 12.x.
    set_source_files_properties(src/runtime/cutlass_gemm.cu PROPERTIES
                                COMPILE_OPTIONS "--expt-relaxed-constexpr")
    target_include_directories(difcore SYSTEM PRIVATE "${DIF_CUTLASS_INCLUDE_DIR}")
    target_link_libraries(difcore PRIVATE CUDA::cudart)
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUTLASS=1)
    # In-tree owned H3 dense INT8 attention (sm_86 kernel; admitted at prepare
    # only on that SM). Compiled whenever the CUDA language is enabled.
    target_sources(difcore PRIVATE src/runtime/h3_owned_attention.cu)
    target_compile_definitions(difcore PUBLIC DIF_HAS_H3_OWNED_ATTENTION=1)
    set_property(TARGET difcore PROPERTY CUDA_ARCHITECTURES
                                         "${DIF_CUDA_ARCHITECTURES}")
    set_property(TARGET difcore PROPERTY CUDA_STANDARD 17)
    set_property(TARGET difcore PROPERTY CUDA_STANDARD_REQUIRED ON)
    message(STATUS "Diffusion compiler CUTLASS GEMM: ${DIF_CUTLASS_INCLUDE_DIR}")
    message(STATUS "Diffusion compiler CUDA architectures: ${DIF_CUDA_ARCHITECTURES}")

    find_path(DIF_FLASH_ATTENTION_SOURCE_DIR
              flash_fwd_hdim128_bf16_sm80.cu
              HINTS "${DIF_FLASH_ATTENTION_ROOT}" ENV FLASH_ATTENTION_ROOT
              PATH_SUFFIXES src)
    if(DIF_FLASH_ATTENTION_SOURCE_DIR)
      target_sources(
          difcore PRIVATE
          src/runtime/flash_attention.cu
          "${DIF_FLASH_ATTENTION_SOURCE_DIR}/flash_fwd_hdim128_bf16_sm80.cu")
      target_include_directories(
          difcore BEFORE PRIVATE
          "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flash_attention_shim")
      target_include_directories(
          difcore SYSTEM PRIVATE "${DIF_FLASH_ATTENTION_SOURCE_DIR}")
      set_source_files_properties(
          src/runtime/flash_attention.cu
          "${DIF_FLASH_ATTENTION_SOURCE_DIR}/flash_fwd_hdim128_bf16_sm80.cu"
          PROPERTIES COMPILE_OPTIONS
          "--expt-relaxed-constexpr;--expt-extended-lambda")
      target_compile_definitions(
          difcore PUBLIC DIF_HAS_FLASH_ATTENTION=1
          PRIVATE FLASH_NAMESPACE=pytorch_flash
                  FLASHATTENTION_DISABLE_ALIBI=1
                  FLASHATTENTION_DISABLE_SOFTCAP=1
                  UNFUSE_FMA=1)
      message(STATUS
              "Diffusion compiler native FlashAttention: ${DIF_FLASH_ATTENTION_SOURCE_DIR}")
    else()
      target_compile_definitions(difcore PUBLIC DIF_HAS_FLASH_ATTENTION=0)
      message(STATUS "Diffusion compiler native FlashAttention: disabled")
    endif()
  else()
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUTLASS=0
                                              DIF_HAS_FLASH_ATTENTION=0)
    message(STATUS "Diffusion compiler CUTLASS GEMM: disabled")
  endif()
  if(DIF_ENABLE_CUDNN)
    find_path(DIF_CUDNN_INCLUDE_DIR cudnn.h
              HINTS "${DIF_CUDNN_ROOT}" ENV CUDNN_ROOT
              PATH_SUFFIXES include)
    find_library(DIF_CUDNN_LIBRARY
                 NAMES cudnn libcudnn.so.9
                 HINTS "${DIF_CUDNN_ROOT}" ENV CUDNN_ROOT
                 PATH_SUFFIXES lib lib64)
  endif()
  if(DIF_ENABLE_CUDNN AND DIF_CUDNN_INCLUDE_DIR AND DIF_CUDNN_LIBRARY)
    target_sources(difcore PRIVATE src/runtime/cudnn_attention.cpp
                                   src/runtime/cudnn_conv.cpp
                                   src/runtime/cudnn_handle.cpp)
    target_include_directories(
        difcore SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cudnn_frontend/include"
        "${DIF_CUDNN_INCLUDE_DIR}")
    target_link_libraries(difcore PRIVATE CUDA::cudart "${DIF_CUDNN_LIBRARY}")
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUDNN=1)
    message(STATUS "Diffusion compiler cuDNN SDPA: ${DIF_CUDNN_LIBRARY}")
  else()
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUDNN=0)
    message(STATUS "Diffusion compiler cuDNN SDPA: disabled")
  endif()
else()
  target_sources(difcore PRIVATE src/runtime/cuda_executor_stub.cpp)
  target_compile_definitions(difcore PUBLIC DIF_HAS_CUDA=0 DIF_HAS_CUDNN=0
                                            DIF_HAS_CUTLASS=0
                                            DIF_HAS_FLASH_ATTENTION=0)
endif()

add_executable(difc tools/difc.cpp)
target_link_libraries(difc PRIVATE difcore)

add_executable(difrun tools/difrun.cpp)
target_link_libraries(difrun PRIVATE difcore)

add_executable(difinspect tools/difinspect.cpp)
target_link_libraries(difinspect PRIVATE difcore)

add_executable(difprobe tools/difprobe.cpp)
target_link_libraries(difprobe PRIVATE difcore)

add_executable(difbench tools/difbench.cpp)
target_link_libraries(difbench PRIVATE difcore)

add_executable(diftrace tools/diftrace.cpp)
target_link_libraries(diftrace PRIVATE difcore)

add_executable(difplan tools/difplan.cpp)
target_link_libraries(difplan PRIVATE difcore)

add_executable(difbisect tools/difbisect.cpp)
target_link_libraries(difbisect PRIVATE difcore)

add_executable(difquality tools/difquality.cpp)
target_link_libraries(difquality PRIVATE difcore)

add_executable(difregress tools/difregress.cpp)
target_link_libraries(difregress PRIVATE difcore)

add_executable(diftune tools/diftune.cpp)
target_link_libraries(diftune PRIVATE difcore)

add_executable(difopt tools/difopt.cpp)
target_link_libraries(difopt PRIVATE difcore)

add_executable(difcast tools/difcast.cpp)
target_link_libraries(difcast PRIVATE difcore)

add_executable(difcompare tools/difcompare.cpp)
target_link_libraries(difcompare PRIVATE difcore)

add_executable(difschedule tools/difschedule.cpp)
target_link_libraries(difschedule PRIVATE difcore)

add_executable(difh3layout tools/difh3layout.cpp)
target_link_libraries(difh3layout PRIVATE difcore)

add_executable(difweights tools/difweights.cpp)
target_link_libraries(difweights PRIVATE difcore)

add_executable(difquant tools/difquant.cpp)
target_link_libraries(difquant PRIVATE difcore)

add_executable(difimage tools/difimage.cpp)
target_link_libraries(difimage PRIVATE difcore)

add_executable(difvaedecode tools/difvaedecode.cpp)
target_link_libraries(difvaedecode PRIVATE difcore)

add_executable(difditops tools/difditops.cpp)
target_link_libraries(difditops PRIVATE difcore)

add_executable(difdittrain tools/difdittrain.cpp)
target_link_libraries(difdittrain PRIVATE difcore)

add_executable(diftrain tools/diftrain.cpp)
target_link_libraries(diftrain PRIVATE difcore)

add_executable(difh3infer tools/difh3infer.cpp)
target_link_libraries(difh3infer PRIVATE difcore)

add_executable(difh3media tools/difh3media.cpp)
target_link_libraries(difh3media PRIVATE difcore)

add_executable(difh3encode tools/difh3encode.cpp)
target_link_libraries(difh3encode PRIVATE difcore)

add_executable(difh3vision tools/difh3vision.cpp)
target_link_libraries(difh3vision PRIVATE difcore dif_text)

add_executable(difh3state tools/difh3state.cpp)
target_link_libraries(difh3state PRIVATE difcore)

add_executable(difslice tools/difslice.cpp)
target_link_libraries(difslice PRIVATE difcore)

add_executable(difaudioops tools/difaudioops.cpp)
target_link_libraries(difaudioops PRIVATE difcore)

add_executable(difaudiodecode tools/difaudiodecode.cpp)
target_link_libraries(difaudiodecode PRIVATE difcore)

add_executable(difimport tools/difimport.cpp)
target_link_libraries(difimport PRIVATE difcore)

add_executable(difcondition tools/difcondition.cpp)
target_link_libraries(difcondition PRIVATE difcore)

add_executable(difmodcache tools/difmodcache.cpp)
target_link_libraries(difmodcache PRIVATE difcore)
if(TARGET CUDA::cuda_driver)
  target_link_libraries(difmodcache PRIVATE CUDA::cuda_driver CUDA::nvrtc
                                            ${CMAKE_DL_LIBS})
endif()

add_executable(difh3noise tools/difh3noise.cpp)
target_link_libraries(difh3noise PRIVATE difcore)
if(TARGET CUDA::cuda_driver)
  target_link_libraries(difh3noise PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()

add_executable(difh3convrot tools/difh3convrot.cpp)
target_link_libraries(difh3convrot PRIVATE difcore)
if(TARGET CUDA::cuda_driver)
  target_link_libraries(difh3convrot PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()
add_executable(difconvrot tools/difh3convrot.cpp)
target_link_libraries(difconvrot PRIVATE difcore)
if(TARGET CUDA::cuda_driver)
  target_link_libraries(difconvrot PRIVATE CUDA::cuda_driver CUDA::nvrtc)
endif()

add_executable(difkrea2block tools/difkrea2block.cpp)
target_link_libraries(difkrea2block PRIVATE difcore)

add_executable(difflux2block tools/difflux2block.cpp)
target_link_libraries(difflux2block PRIVATE difcore)

add_executable(difsdxlstage tools/difsdxlstage.cpp)
target_link_libraries(difsdxlstage PRIVATE difcore)
install(TARGETS difsdxlstage RUNTIME DESTINATION bin)

add_executable(difsdxlsample tools/difsdxlsample.cpp)
install(TARGETS difsdxlsample RUNTIME DESTINATION bin)

add_executable(difflux2sample tools/difflux2sample.cpp)
target_link_libraries(difflux2sample PRIVATE difcore)
add_executable(difkrea2text tools/difkrea2text.cpp)
target_link_libraries(difkrea2text PRIVATE difcore)
add_executable(difkrea2denoise tools/difkrea2denoise.cpp)
target_link_libraries(difkrea2denoise PRIVATE difcore)
add_executable(difkrea2sample tools/difkrea2sample.cpp)
target_link_libraries(difkrea2sample PRIVATE difcore)
add_executable(difkrea2vae tools/difkrea2vae.cpp)
target_link_libraries(difkrea2vae PRIVATE difcore)

if(DIF_ENABLE_OPENCL)
  find_path(DIF_OPENCL_INCLUDE_DIR CL/cl.h
            HINTS ${CUDAToolkit_INCLUDE_DIRS}
                  /usr/local/cuda/targets/x86_64-linux/include)
  find_library(DIF_OPENCL_LIBRARY OpenCL
               HINTS /usr/local/cuda/targets/x86_64-linux/lib)
endif()
if(DIF_ENABLE_OPENCL AND DIF_OPENCL_INCLUDE_DIR AND DIF_OPENCL_LIBRARY)
  add_library(dif_backend_opencl SHARED
      backends/opencl/opencl_backend.cpp
      src/compiler/memory_plan.cpp
      src/ir/codec.cpp
      src/ir/opcodes.cpp
      src/ir/verify.cpp
      src/support/sha256.cpp)
  target_include_directories(dif_backend_opencl PRIVATE include
                                                     "${DIF_OPENCL_INCLUDE_DIR}")
  target_compile_options(dif_backend_opencl PRIVATE -Wall -Wextra -Wpedantic -Werror)
  target_link_libraries(dif_backend_opencl PRIVATE "${DIF_OPENCL_LIBRARY}")
  set_target_properties(dif_backend_opencl PROPERTIES CXX_VISIBILITY_PRESET hidden)
  install(TARGETS dif_backend_opencl LIBRARY DESTINATION lib)
  message(STATUS "Diffusion compiler OpenCL backend: ${DIF_OPENCL_LIBRARY}")
else()
  message(STATUS "Diffusion compiler OpenCL backend: disabled")
endif()

install(TARGETS difc difrun difinspect difprobe diftune difopt difcast difcompare difschedule difh3layout difweights difquant difimage difvaedecode diftrain difh3infer difh3media difh3encode difh3vision difh3state difh3convrot difconvrot difslice
        RUNTIME DESTINATION bin)

if(DIF_BUILD_TESTS)
  enable_testing()
  add_executable(dif_tests tests/dif_tests.cpp)
  target_link_libraries(dif_tests PRIVATE difcore)
  add_test(NAME dif_tests COMMAND dif_tests)

  add_executable(dif_flux2_frontend_tests tests/flux2_frontend_tests.cpp)
  target_link_libraries(dif_flux2_frontend_tests PRIVATE difcore)
  add_test(NAME dif_flux2_frontend_tests COMMAND dif_flux2_frontend_tests)
  add_executable(dif_rotary_tests tests/rotary_tests.cpp)
  target_link_libraries(dif_rotary_tests PRIVATE difcore)
  add_test(NAME dif_rotary_tests COMMAND dif_rotary_tests)
  add_executable(dif_squareq_w4_tests tests/squareq_w4_tests.cpp)
  target_link_libraries(dif_squareq_w4_tests PRIVATE difcore)
  add_test(NAME dif_squareq_w4_tests COMMAND dif_squareq_w4_tests
           ${CMAKE_SOURCE_DIR}/perf/regress/fixtures/squareq-w4-tiny)

  add_executable(dif_lora_tests tests/lora_tests.cpp)
  target_link_libraries(dif_lora_tests PRIVATE difcore)
  add_test(NAME dif_lora_tests COMMAND dif_lora_tests)

  add_executable(dif_cudnn_attention_backward_tests tests/cudnn_attention_backward_tests.cpp)
  target_link_libraries(dif_cudnn_attention_backward_tests PRIVATE difcore)
  add_test(NAME dif_cudnn_attention_backward_tests COMMAND dif_cudnn_attention_backward_tests)
  add_executable(dif_training_gradient_tests tests/training_gradient_tests.cpp)
  target_link_libraries(dif_training_gradient_tests PRIVATE difcore)
  add_test(NAME dif_training_gradient_tests COMMAND dif_training_gradient_tests)

  add_executable(dif_autodiff_tests tests/autodiff_tests.cpp)
  target_link_libraries(dif_autodiff_tests PRIVATE difcore)
  add_test(NAME dif_autodiff_tests COMMAND dif_autodiff_tests)
  add_executable(dif_dit_backward_tests tests/dit_backward_tests.cpp)
  target_link_libraries(dif_dit_backward_tests PRIVATE difcore)
  add_test(NAME dif_dit_backward_tests COMMAND dif_dit_backward_tests)

  add_executable(dif_dit_lora_tests tests/dit_lora_tests.cpp)
  target_link_libraries(dif_dit_lora_tests PRIVATE difcore)
  add_test(NAME dif_dit_lora_tests COMMAND dif_dit_lora_tests)

  add_executable(dif_fusion_tests tests/fusion_tests.cpp)
  target_link_libraries(dif_fusion_tests PRIVATE difcore)
  add_test(NAME dif_fusion_tests COMMAND dif_fusion_tests)

  add_executable(dif_owned_attention_tests tests/owned_attention_tests.cpp)
  target_link_libraries(dif_owned_attention_tests PRIVATE difcore)
  add_test(NAME dif_owned_attention_tests COMMAND dif_owned_attention_tests)
  add_executable(dif_h3_route_refinement_tests
                 tests/h3_route_refinement_tests.cpp)
  target_link_libraries(dif_h3_route_refinement_tests PRIVATE difcore)
  add_test(NAME dif_h3_route_refinement_tests
           COMMAND dif_h3_route_refinement_tests)
  add_executable(dif_epilogue_tests tests/epilogue_tests.cpp)
  target_link_libraries(dif_epilogue_tests PRIVATE difcore)
  add_test(NAME dif_epilogue_tests COMMAND dif_epilogue_tests)

  add_library(dif_mock_backend SHARED tests/mock_backend.cpp)
  target_include_directories(dif_mock_backend PRIVATE include)
  target_compile_options(dif_mock_backend PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(dif_mock_backend PROPERTIES CXX_VISIBILITY_PRESET hidden)

  add_library(dif_mock_backend_v2 SHARED tests/mock_backend_v2.cpp)
  target_include_directories(dif_mock_backend_v2 PRIVATE include)
  target_compile_options(dif_mock_backend_v2 PRIVATE -Wall -Wextra -Wpedantic -Werror)
  set_target_properties(dif_mock_backend_v2 PROPERTIES CXX_VISIBILITY_PRESET hidden)

  add_executable(dif_opt_tests tests/opt_tests.cpp)
  target_link_libraries(dif_opt_tests PRIVATE difcore)
  add_test(NAME dif_opt_tests COMMAND dif_opt_tests)

  add_executable(dif_target_tests tests/target_tests.cpp)
  target_link_libraries(dif_target_tests PRIVATE difcore)
  target_compile_definitions(
      dif_target_tests PRIVATE
      DIF_DIFPROBE_PATH="$<TARGET_FILE:difprobe>")
  add_dependencies(dif_target_tests difprobe)
  add_test(NAME dif_target_tests COMMAND dif_target_tests)

  add_executable(dif_telemetry_tests tests/telemetry_tests.cpp)
  target_link_libraries(dif_telemetry_tests PRIVATE difcore)
  target_compile_definitions(
      dif_telemetry_tests PRIVATE
      DIF_DIFBENCH_PATH="$<TARGET_FILE:difbench>"
      DIF_DIFTRACE_PATH="$<TARGET_FILE:diftrace>"
      DIF_DIFPLAN_PATH="$<TARGET_FILE:difplan>"
      DIF_DIFRUN_PATH="$<TARGET_FILE:difrun>"
      DIF_DIFOPT_PATH="$<TARGET_FILE:difopt>")
  add_dependencies(dif_telemetry_tests difbench diftrace difplan difrun difopt)
  add_test(NAME dif_telemetry_tests COMMAND dif_telemetry_tests)

  add_executable(dif_quality_tests tests/quality_tests.cpp)
  target_link_libraries(dif_quality_tests PRIVATE difcore)
  target_compile_definitions(
      dif_quality_tests PRIVATE
      DIF_DIFQUALITY_PATH="$<TARGET_FILE:difquality>"
      DIF_DIFREGRESS_PATH="$<TARGET_FILE:difregress>"
      DIF_DIFBISECT_PATH="$<TARGET_FILE:difbisect>")
  add_dependencies(dif_quality_tests difquality difregress difbisect)
  add_test(NAME dif_quality_tests COMMAND dif_quality_tests)

  add_executable(dif_optimizer_tests tests/optimizer_tests.cpp)
  target_link_libraries(dif_optimizer_tests PRIVATE difcore)
  add_test(NAME dif_optimizer_tests COMMAND dif_optimizer_tests)

  add_executable(dif_bench_tests tests/bench_tests.cpp)
  target_link_libraries(dif_bench_tests PRIVATE difcore)
  add_test(NAME dif_bench_tests COMMAND dif_bench_tests)

  add_executable(dif_noise_tests tests/noise_tests.cpp)
  target_link_libraries(dif_noise_tests PRIVATE difcore)
  target_compile_definitions(
      dif_noise_tests PRIVATE DIF_DIFH3NOISE_PATH="$<TARGET_FILE:difh3noise>")
  add_dependencies(dif_noise_tests difh3noise)
  add_test(NAME dif_noise_tests COMMAND dif_noise_tests)

  add_executable(dif_format_tests tests/format_tests.cpp)
  target_link_libraries(dif_format_tests PRIVATE difcore)
  target_compile_definitions(
      dif_format_tests PRIVATE
      DIF_DIFOPT_PATH="$<TARGET_FILE:difopt>"
      DIF_DIFWEIGHTS_PATH="$<TARGET_FILE:difweights>")
  add_dependencies(dif_format_tests difopt difweights)
  add_test(NAME dif_format_tests COMMAND dif_format_tests)

  add_executable(dif_bisect_tests tests/bisect_tests.cpp)
  target_link_libraries(dif_bisect_tests PRIVATE difcore)
  target_compile_definitions(
      dif_bisect_tests PRIVATE
      DIF_DIFBISECT_PATH="$<TARGET_FILE:difbisect>"
      DIF_DIFINSPECT_PATH="$<TARGET_FILE:difinspect>"
      DIF_DIFRUN_PATH="$<TARGET_FILE:difrun>")
  add_dependencies(dif_bisect_tests difbisect difinspect difrun)
  add_test(NAME dif_bisect_tests COMMAND dif_bisect_tests)

  add_executable(dif_difopt_cli_tests tests/difopt_cli_tests.cpp)
  target_link_libraries(dif_difopt_cli_tests PRIVATE difcore)
  target_compile_definitions(
      dif_difopt_cli_tests PRIVATE
      DIF_DIFOPT_PATH="$<TARGET_FILE:difopt>")
  add_dependencies(dif_difopt_cli_tests difopt)
  add_test(NAME dif_difopt_cli_tests COMMAND dif_difopt_cli_tests)

  add_executable(dif_plugin_tests tests/plugin_tests.cpp)
  target_link_libraries(dif_plugin_tests PRIVATE difcore)
  target_compile_definitions(
      dif_plugin_tests PRIVATE
      DIF_MOCK_BACKEND_V1_PATH="$<TARGET_FILE:dif_mock_backend>"
      DIF_MOCK_BACKEND_V2_PATH="$<TARGET_FILE:dif_mock_backend_v2>")
  add_dependencies(dif_plugin_tests dif_mock_backend dif_mock_backend_v2)
  add_test(NAME dif_plugin_tests COMMAND dif_plugin_tests)

  if(TARGET dif_backend_opencl)
    add_executable(dif_opencl_plugin_tests tests/opencl_plugin_tests.cpp)
    target_link_libraries(dif_opencl_plugin_tests PRIVATE difcore)
    target_compile_definitions(
        dif_opencl_plugin_tests PRIVATE
        DIF_OPENCL_BACKEND_PATH="$<TARGET_FILE:dif_backend_opencl>")
    add_dependencies(dif_opencl_plugin_tests dif_backend_opencl)
    add_test(NAME dif_opencl_plugin_tests COMMAND dif_opencl_plugin_tests)
    set_tests_properties(dif_opencl_plugin_tests PROPERTIES SKIP_RETURN_CODE 77)
  endif()
endif()

# --- native text tokenizer (w3-conditioner) ---------------------------------
add_library(dif_text STATIC
    src/text/unicode.cpp
    src/text/qwen_bpe_tokenizer.cpp
    src/text/clip_bpe_tokenizer.cpp
    src/text/flux2_prompt.cpp)
target_include_directories(dif_text PUBLIC include)
target_compile_options(
    dif_text PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
target_link_libraries(dif_text PUBLIC difcore)
target_link_libraries(difflux2sample PRIVATE dif_text)
target_link_libraries(difsdxlsample PRIVATE dif_text)

add_executable(diftokenize tools/diftokenize.cpp)
target_link_libraries(diftokenize PRIVATE dif_text)
install(TARGETS diftokenize RUNTIME DESTINATION bin)

if(DIF_BUILD_TESTS)
  add_executable(dif_kernel_source_tests tests/kernel_source_tests.cpp)
  target_link_libraries(dif_kernel_source_tests PRIVATE difcore)
  add_test(NAME dif_kernel_source_tests
           COMMAND dif_kernel_source_tests
                   "${CMAKE_CURRENT_SOURCE_DIR}/perf/regress/fixtures/kernel-sources")
  add_executable(dif_opcode_registry_tests tests/opcode_registry_tests.cpp)
  target_link_libraries(dif_opcode_registry_tests PRIVATE difcore)
  add_test(NAME dif_opcode_registry_tests COMMAND dif_opcode_registry_tests)
  add_executable(dif_tokenizer_tests tests/tokenizer_tests.cpp)
  target_link_libraries(dif_tokenizer_tests PRIVATE dif_text)
  add_test(NAME dif_tokenizer_tests COMMAND dif_tokenizer_tests
           ${CMAKE_SOURCE_DIR}/perf/regress/fixtures/tokenizer-mistral-tekken)
  set_tests_properties(dif_tokenizer_tests PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(dif_sdxl_schedule_tests tests/sdxl_schedule_tests.cpp)
  target_include_directories(dif_sdxl_schedule_tests PRIVATE tests)
  target_link_libraries(dif_sdxl_schedule_tests PRIVATE difcore)
  add_test(NAME dif_sdxl_schedule_tests COMMAND dif_sdxl_schedule_tests)

  add_executable(dif_clip_tokenizer_tests tests/clip_tokenizer_tests.cpp)
  target_link_libraries(dif_clip_tokenizer_tests PRIVATE dif_text)
  add_test(NAME dif_clip_tokenizer_tests COMMAND dif_clip_tokenizer_tests
           ${CMAKE_SOURCE_DIR}/perf/regress/fixtures/clip-tokenizer)
  set_tests_properties(dif_clip_tokenizer_tests PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(dif_flux2_prompt_tests tests/flux2_prompt_tests.cpp)
  target_link_libraries(dif_flux2_prompt_tests PRIVATE dif_text)
  add_test(NAME dif_flux2_prompt_tests COMMAND dif_flux2_prompt_tests)
  set_tests_properties(dif_flux2_prompt_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
