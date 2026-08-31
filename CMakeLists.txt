cmake_minimum_required(VERSION 3.24)

project(diffusion_compiler VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(DIF_ENABLE_CUDA "Build the CUDA Driver/NVRTC backend" ON)
option(DIF_ENABLE_CUDNN "Build optional cuDNN SDPA lowering" ON)
option(DIF_BUILD_TESTS "Build unit tests" ON)
option(DIF_ENABLE_OPENCL "Build the vendor-neutral OpenCL reference backend" ON)
set(DIF_CUDNN_ROOT "" CACHE PATH "cuDNN installation root (include/ and lib/)")
set(DIF_CUTLASS_ROOT "" CACHE PATH "CUTLASS source root (contains include/cutlass)")

add_library(difcore
    src/backend/plugin.cpp
    src/compiler/compiler.cpp
    src/compiler/int4.cpp
    src/compiler/memory_plan.cpp
    src/compiler/slice.cpp
  src/frontend/h3.cpp
  src/frontend/h3_conditioning.cpp
  src/frontend/h3_latents.cpp
  src/frontend/h3_media.cpp
  src/frontend/h3_vae.cpp
  src/frontend/h3_audio_vae.cpp
  src/frontend/dit_block.cpp
  src/frontend/dit_lora.cpp
  src/frontend/lora.cpp
  src/frontend/training.cpp
    src/ir/codec.cpp
    src/ir/verify.cpp
    src/opt/bindings.cpp
    src/opt/gate.cpp
    src/opt/plan.cpp
    src/opt/rewrite.cpp
    src/opt/search.cpp
    src/opt/semantics.cpp
    src/opt/transform.cpp
    src/runtime/cpu_executor.cpp
    src/runtime/executor.cpp
    src/runtime/scalar.cpp
    src/runtime/tensor_io.cpp
    src/sampling/rectified_flow.cpp
    src/support/sha256.cpp
    src/support/wav.cpp
    src/support/json.cpp
    src/tune/database.cpp
    src/training/autodiff.cpp
    src/training/checkpoint.cpp
    src/weights/safetensors.cpp
    src/weights/bundle.cpp
)
target_include_directories(difcore PUBLIC include)
target_compile_options(
    difcore PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>
    # NVCC's generated host stubs use GCC-style line directives that trip
    # host -Wpedantic. Keep all actionable CUDA/host warnings fatal.
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Werror>)
target_link_libraries(difcore PRIVATE ${CMAKE_DL_LIBS})

if(DIF_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
endif()

if(DIF_ENABLE_CUDA AND CUDAToolkit_FOUND)
  target_sources(difcore PRIVATE src/runtime/cuda_executor.cpp)
  target_link_libraries(difcore PRIVATE CUDA::cuda_driver CUDA::nvrtc
                                        CUDA::cublas CUDA::cublasLt)
  target_compile_definitions(
      difcore PUBLIC DIF_HAS_CUDA=1
      PRIVATE DIF_CUDA_INCLUDE_DIR="${CUDAToolkit_INCLUDE_DIRS}")
  find_path(DIF_CUTLASS_INCLUDE_DIR cutlass/cutlass.h
            HINTS "${DIF_CUTLASS_ROOT}" ENV CUTLASS_ROOT
            PATH_SUFFIXES include)
  if(DIF_CUTLASS_INCLUDE_DIR)
    enable_language(CUDA)
    target_sources(difcore PRIVATE src/runtime/cutlass_gemm.cu)
    target_include_directories(difcore SYSTEM PRIVATE "${DIF_CUTLASS_INCLUDE_DIR}")
    target_link_libraries(difcore PRIVATE CUDA::cudart)
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUTLASS=1)
    set_property(TARGET difcore PROPERTY CUDA_ARCHITECTURES 86)
    set_property(TARGET difcore PROPERTY CUDA_STANDARD 17)
    set_property(TARGET difcore PROPERTY CUDA_STANDARD_REQUIRED ON)
    message(STATUS "Diffusion compiler CUTLASS GEMM: ${DIF_CUTLASS_INCLUDE_DIR}")
  else()
    target_compile_definitions(difcore PUBLIC DIF_HAS_CUTLASS=0)
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
    target_sources(difcore PRIVATE src/runtime/cudnn_attention.cpp)
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
                                            DIF_HAS_CUTLASS=0)
endif()

add_executable(difc tools/difc.cpp)
target_link_libraries(difc PRIVATE difcore)

add_executable(difrun tools/difrun.cpp)
target_link_libraries(difrun PRIVATE difcore)

add_executable(difinspect tools/difinspect.cpp)
target_link_libraries(difinspect PRIVATE difcore)

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

add_executable(difslice tools/difslice.cpp)
target_link_libraries(difslice PRIVATE difcore)

add_executable(difaudioops tools/difaudioops.cpp)
target_link_libraries(difaudioops PRIVATE difcore)

add_executable(difaudiodecode tools/difaudiodecode.cpp)
target_link_libraries(difaudiodecode PRIVATE difcore)

add_executable(difimport tools/difimport.cpp)
target_link_libraries(difimport PRIVATE difcore)

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

install(TARGETS difc difrun difinspect diftune difopt difcast difcompare difschedule difh3layout difweights difquant difimage difvaedecode diftrain difh3infer difh3media difslice
        RUNTIME DESTINATION bin)

if(DIF_BUILD_TESTS)
  enable_testing()
  add_executable(dif_tests tests/dif_tests.cpp)
  target_link_libraries(dif_tests PRIVATE difcore)
  add_test(NAME dif_tests COMMAND dif_tests)

  add_executable(dif_lora_tests tests/lora_tests.cpp)
  target_link_libraries(dif_lora_tests PRIVATE difcore)
  add_test(NAME dif_lora_tests COMMAND dif_lora_tests)

  add_executable(dif_dit_backward_tests tests/dit_backward_tests.cpp)
  target_link_libraries(dif_dit_backward_tests PRIVATE difcore)
  add_test(NAME dif_dit_backward_tests COMMAND dif_dit_backward_tests)

  add_executable(dif_dit_lora_tests tests/dit_lora_tests.cpp)
  target_link_libraries(dif_dit_lora_tests PRIVATE difcore)
  add_test(NAME dif_dit_lora_tests COMMAND dif_dit_lora_tests)

  add_executable(dif_fusion_tests tests/fusion_tests.cpp)
  target_link_libraries(dif_fusion_tests PRIVATE difcore)
  add_test(NAME dif_fusion_tests COMMAND dif_fusion_tests)

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
    src/text/qwen_bpe_tokenizer.cpp)
target_include_directories(dif_text PUBLIC include)
target_compile_options(
    dif_text PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic;-Werror>)
target_link_libraries(dif_text PUBLIC difcore)

add_executable(diftokenize tools/diftokenize.cpp)
target_link_libraries(diftokenize PRIVATE dif_text)
install(TARGETS diftokenize RUNTIME DESTINATION bin)

if(DIF_BUILD_TESTS)
  add_executable(dif_tokenizer_tests tests/tokenizer_tests.cpp)
  target_link_libraries(dif_tokenizer_tests PRIVATE dif_text)
  add_test(NAME dif_tokenizer_tests COMMAND dif_tokenizer_tests)
  set_tests_properties(dif_tokenizer_tests PROPERTIES SKIP_RETURN_CODE 77)
endif()
