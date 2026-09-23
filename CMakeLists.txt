cmake_minimum_required(VERSION 3.25)

project(gewell LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
option(GEWELL_ENABLE_CUDA "Build the SM120 runtime and GPU tests" ON)
option(GEWELL_BUILD_DIAGNOSTICS "Build fixed oracle and profiling runners" ON)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(Threads REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)

add_library(gewell_gemma4_image_processor STATIC src/models/gemma4/text/image_processor.cc)
target_include_directories(gewell_gemma4_image_processor PUBLIC include)
target_compile_features(gewell_gemma4_image_processor PUBLIC cxx_std_17)
target_link_libraries(gewell_gemma4_image_processor PRIVATE PNG::PNG JPEG::JPEG)

add_library(gewell_text_codec STATIC
  src/tokenizer.cc
  src/chat_codec.cc
  src/stop_matcher.cc
  src/models/gemma4/31b/serving_assets.cc
  src/models/gemma4/text/contract.cc
  src/models/gemma4/text/chat_template.cc
  src/text_codec_cli.cc
)
target_include_directories(gewell_text_codec PUBLIC include)
target_include_directories(gewell_text_codec SYSTEM PUBLIC vendor/nlohmann)
target_compile_features(gewell_text_codec PUBLIC cxx_std_17)
target_link_libraries(gewell_text_codec PUBLIC OpenSSL::Crypto)

# Build only XGrammar's native library. Its top-level configuration enables
# Python bindings and changes global build flags, neither of which we need.
include(FetchContent)
find_program(GEWELL_PATCH_EXECUTABLE patch REQUIRED)
FetchContent_Declare(xgrammar
  URL https://files.pythonhosted.org/packages/26/38/0c38a8a7614bc02edad2834fa60365fc70f5451ccf1ba713c11dfa69ae8c/xgrammar-0.2.5.post1.tar.gz
  URL_HASH SHA256=bcb977d2e6801e51fc4d2b48b83820e6cef7a0a81894928b9df0e12bd801ba64
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  PATCH_COMMAND "${GEWELL_PATCH_EXECUTABLE}" -p1 --input "${CMAKE_CURRENT_SOURCE_DIR}/vendor/xgrammar/unicode-escapes.patch"
  SOURCE_SUBDIR gewell-native-only
)
FetchContent_MakeAvailable(xgrammar)
file(GLOB_RECURSE GEWELL_XGRAMMAR_SOURCES CONFIGURE_DEPENDS "${xgrammar_SOURCE_DIR}/cpp/*.cc")
list(FILTER GEWELL_XGRAMMAR_SOURCES EXCLUDE REGEX "/tvm_ffi/")
add_library(gewell_xgrammar STATIC ${GEWELL_XGRAMMAR_SOURCES})
target_include_directories(gewell_xgrammar SYSTEM PUBLIC
  "${xgrammar_SOURCE_DIR}/include"
  "${xgrammar_SOURCE_DIR}/3rdparty/dlpack/include"
)
target_include_directories(gewell_xgrammar SYSTEM PRIVATE "${xgrammar_SOURCE_DIR}/3rdparty/picojson")
target_compile_features(gewell_xgrammar PUBLIC cxx_std_17)
target_compile_options(gewell_xgrammar PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-O3>)
target_compile_definitions(gewell_xgrammar PRIVATE XGRAMMAR_ENABLE_CPPTRACE=0 XGRAMMAR_ENABLE_INTERNAL_CHECK=0)
target_link_libraries(gewell_xgrammar PUBLIC Threads::Threads)

add_library(gewell_json_constraint STATIC src/json_constraint.cc)
target_link_libraries(gewell_json_constraint PUBLIC gewell_text_codec PRIVATE gewell_xgrammar)
target_compile_features(gewell_json_constraint PUBLIC cxx_std_17)

add_library(gewell_console STATIC src/console.cc)
target_include_directories(gewell_console PUBLIC include)
target_include_directories(gewell_console SYSTEM PUBLIC vendor/nlohmann)
target_compile_features(gewell_console PUBLIC cxx_std_17)

add_library(gewell_offline_io STATIC src/app/offline_io.cc)
target_include_directories(gewell_offline_io PUBLIC src)
target_compile_features(gewell_offline_io PUBLIC cxx_std_17)

add_library(gewell_metrics STATIC src/metrics.cc)
target_include_directories(gewell_metrics PUBLIC include)
target_include_directories(gewell_metrics SYSTEM PUBLIC vendor/nlohmann)
target_compile_features(gewell_metrics PUBLIC cxx_std_17)

add_library(gewell_http STATIC src/http_api.cc src/http_server.cc src/models/gemma4/text/tool_output.cc)
target_link_libraries(gewell_http PUBLIC gewell_text_codec gewell_json_constraint gewell_console gewell_metrics Threads::Threads)
target_compile_features(gewell_http PUBLIC cxx_std_17)

# These libraries and tests configure without a CUDA toolkit.
add_library(gewell_artifact STATIC src/models/gemma4/31b/artifact.cc src/models/gemma4/31b/component_weights.cc)
target_include_directories(gewell_artifact PUBLIC include)
target_include_directories(gewell_artifact SYSTEM PRIVATE vendor/nlohmann)
target_compile_features(gewell_artifact PUBLIC cxx_std_17)
target_link_libraries(gewell_artifact PUBLIC OpenSSL::Crypto)

add_library(gewell_cache STATIC src/kv_cache.cc src/prefix_index.cc src/pending_prefix.cc)
target_include_directories(gewell_cache PUBLIC include)
target_compile_features(gewell_cache PUBLIC cxx_std_17)

add_library(gewell_runtime STATIC src/runtime/cache.cc src/runtime/scheduler.cc src/runtime/image.cc)
target_include_directories(gewell_runtime PUBLIC include)
target_include_directories(gewell_runtime SYSTEM PUBLIC vendor/nlohmann)
target_compile_features(gewell_runtime PUBLIC cxx_std_17)
target_link_libraries(gewell_runtime PUBLIC gewell_cache gewell_json_constraint gewell_metrics gewell_console Threads::Threads PRIVATE OpenSSL::Crypto)

add_library(gewell_gemma4_31b_cache_config STATIC
  src/models/gemma4/31b/sm120/cache_config.cc)
target_include_directories(gewell_gemma4_31b_cache_config PRIVATE src)
target_compile_features(gewell_gemma4_31b_cache_config PUBLIC cxx_std_17)
target_link_libraries(gewell_gemma4_31b_cache_config PUBLIC gewell_cache)

if(BUILD_TESTING)
  add_executable(gewell_offline_io_test tests/offline_io_test.cc)
  target_link_libraries(gewell_offline_io_test PRIVATE gewell_offline_io)
  add_test(NAME gewell_offline_io_test COMMAND gewell_offline_io_test)
  add_executable(gewell_image_processor_test tests/image_processor_test.cc)
  target_include_directories(gewell_image_processor_test SYSTEM PRIVATE vendor/nlohmann)
  target_link_libraries(gewell_image_processor_test PRIVATE gewell_gemma4_image_processor OpenSSL::Crypto)
  add_test(NAME gewell_image_processor_test COMMAND gewell_image_processor_test
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/image_processor.json")
  foreach(suite IN ITEMS runtime_cache runtime_scheduler image_identity)
    add_executable(gewell_${suite}_test tests/${suite}_test.cc)
    target_link_libraries(gewell_${suite}_test PRIVATE gewell_runtime)
    add_test(NAME gewell_${suite}_test COMMAND gewell_${suite}_test)
  endforeach()
  foreach(suite IN ITEMS artifact kv_cache prefix_index pending_prefix model_contract metrics stop_matcher vision_engine_contract)
    add_executable(gewell_${suite}_test tests/${suite}_test.cc)
    target_link_libraries(gewell_${suite}_test PRIVATE gewell_artifact gewell_cache gewell_http)
    add_test(NAME gewell_${suite}_test COMMAND gewell_${suite}_test)
  endforeach()
  target_sources(gewell_vision_engine_contract_test PRIVATE
    src/models/gemma4/31b/sm120/vision/engine.cc)

  set(GEWELL_TOKENIZER_DIRECTORY
      ""
      CACHE PATH "Pinned local tokenizer assets for the CPU text codec oracle test")
  foreach(suite IN ITEMS text_codec json_constraint http_api http_server)
    add_executable(gewell_${suite}_test tests/${suite}_test.cc)
    target_link_libraries(gewell_${suite}_test PRIVATE gewell_artifact gewell_http)
  endforeach()
  add_test(NAME gewell_text_codec_test COMMAND gewell_text_codec_test
    "${GEWELL_TOKENIZER_DIRECTORY}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/text_codec.json")
  add_test(NAME gewell_json_constraint_test COMMAND gewell_json_constraint_test
    "${GEWELL_TOKENIZER_DIRECTORY}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures/dsgen_schemas.json")
  add_test(NAME gewell_http_api_test COMMAND gewell_http_api_test "${GEWELL_TOKENIZER_DIRECTORY}")
  set_tests_properties(gewell_text_codec_test gewell_json_constraint_test gewell_http_api_test
    PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 180)
endif()

if(NOT GEWELL_ENABLE_CUDA)
  return()
endif()
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# Only headers are needed for the fixed SM120 GEMM instantiations.
FetchContent_Declare(cutlass
  URL https://codeload.github.com/NVIDIA/cutlass/tar.gz/refs/tags/v4.5.0
  URL_HASH SHA256=d1ce5d785ba45e2ec1bbfc0fad64b5542878b8329a7d7dfdbfa681a545d35295
  DOWNLOAD_EXTRACT_TIMESTAMP FALSE
  SOURCE_SUBDIR gewell-headers-only
)
FetchContent_MakeAvailable(cutlass)
set_property(SOURCE src/nvfp4_linear.cu src/fp8_linear.cu
  src/models/gemma4/31b/sm120/kernels/prefill_fp8_attention.cu APPEND PROPERTY
  INCLUDE_DIRECTORIES "${cutlass_SOURCE_DIR}/include" "${cutlass_SOURCE_DIR}/tools/util/include")
set_property(SOURCE src/nvfp4_linear.cu src/fp8_linear.cu
  src/models/gemma4/31b/sm120/kernels/prefill_fp8_attention.cu APPEND PROPERTY
  COMPILE_OPTIONS --expt-relaxed-constexpr)

function(gewell_cuda_target target)
  target_include_directories(${target} PRIVATE include src)
  target_compile_features(${target} PRIVATE cxx_std_17 cuda_std_17)
  target_compile_definitions(${target} PRIVATE GEWELL_CUDA_ARCHITECTURE="120a-real")
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(${target} PROPERTIES
    CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED YES CUDA_EXTENSIONS NO
    CUDA_ARCHITECTURES "120a-real" CUDA_RUNTIME_LIBRARY Shared CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

# Test bodies and fixed-fixture runners are excluded from this library.
add_library(gewell_gemma4_31b_sm120 STATIC
  src/models/gemma4/31b/sm120/runtime_backend.cu
  src/models/gemma4/31b/sm120/cache.cu
  src/models/gemma4/31b/sm120/kernels/linear.cu
  src/models/gemma4/31b/sm120/kernels/embedding.cu
  src/models/gemma4/31b/sm120/kernels/norm.cu
  src/models/gemma4/31b/sm120/kernels/rope.cu
  src/models/gemma4/31b/sm120/kernels/kv_write.cu
  src/models/gemma4/31b/sm120/kernels/attention.cu
  src/models/gemma4/31b/sm120/kernels/activation.cu
  src/models/gemma4/31b/sm120/kernels/sampling.cu
  src/models/gemma4/31b/sm120/kernels/prefill_rope.cu
  src/models/gemma4/31b/sm120/kernels/prefill_kv_write.cu
  src/models/gemma4/31b/sm120/kernels/prefill_attention.cu
  src/models/gemma4/31b/sm120/kernels/prefill_tensor_attention.cu
  src/models/gemma4/31b/sm120/kernels/prefill_fp8_attention.cu
  src/models/gemma4/31b/sm120/mtp/assistant.cu
  src/models/gemma4/31b/sm120/mtp/attention.cu
  src/models/gemma4/31b/sm120/mtp/sampling.cu
  src/models/gemma4/31b/sm120/mtp/target.cu
  src/models/gemma4/31b/sm120/mtp/cycle.cu
  src/nvfp4_linear.cu
  src/fp8_linear.cu
  src/replay_metrics.cu
  src/models/gemma4/31b/sm120/vision/engine.cc
  src/models/gemma4/31b/sm120/vision/executor.cu
  src/models/gemma4/31b/sm120/vision/primitives.cu
  src/models/gemma4/31b/sm120/vision/prepared_image.cu
  src/weight_qdq.cu
)
gewell_cuda_target(gewell_gemma4_31b_sm120)
target_link_libraries(gewell_gemma4_31b_sm120 PUBLIC
  gewell_artifact gewell_gemma4_31b_cache_config gewell_runtime gewell_console
  CUDA::cudart CUDA::cublas CUDA::cublasLt)

add_executable(gewell src/app/runner.cu src/app/main.cu)
target_sources(gewell PRIVATE src/app/offline_runner.cu)
install(TARGETS gewell RUNTIME DESTINATION bin)
target_link_libraries(gewell PRIVATE gewell_offline_io)
gewell_cuda_target(gewell)
target_link_libraries(gewell PRIVATE gewell_gemma4_31b_sm120 gewell_http gewell_gemma4_image_processor)

if(GEWELL_BUILD_DIAGNOSTICS)
  add_executable(gewell_diagnostics
    src/diagnostics/main.cc src/diagnostics/bos_runner.cu
    src/diagnostics/pair_fixtures.cu src/diagnostics/vision_runner.cu)
  gewell_cuda_target(gewell_diagnostics)
  target_link_libraries(gewell_diagnostics PRIVATE gewell_gemma4_31b_sm120)
endif()

if(BUILD_TESTING)
  foreach(suite IN ITEMS fp8_kv fp8_linear nvfp4_linear replay_metrics vision_primitives
      mtp_sampling mtp_assistant mtp_target mtp_attention mtp_norm
      bf16_primitives weight_qdq cuda_target bos_runner app)
    add_executable(gewell_${suite}_test tests/${suite}_test.cu)
    gewell_cuda_target(gewell_${suite}_test)
    target_link_libraries(gewell_${suite}_test PRIVATE gewell_gemma4_31b_sm120)
    add_test(NAME gewell_${suite}_test COMMAND gewell_${suite}_test)
    set_tests_properties(gewell_${suite}_test PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
  endforeach()
  target_link_libraries(gewell_app_test PRIVATE gewell_http gewell_gemma4_image_processor)
  add_test(NAME gewell_weight_qdq_device_test COMMAND gewell_weight_qdq_test --device)
  set_tests_properties(gewell_weight_qdq_device_test PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)
  add_executable(gewell_prefill_primitive_self_test tests/prefill_primitives_test.cu)
  gewell_cuda_target(gewell_prefill_primitive_self_test)
  target_link_libraries(gewell_prefill_primitive_self_test PRIVATE gewell_gemma4_31b_sm120)
  add_test(NAME gewell_prefill_primitive_self_test COMMAND gewell_prefill_primitive_self_test)
  set_tests_properties(gewell_prefill_primitive_self_test PROPERTIES RUN_SERIAL TRUE TIMEOUT 300)

  # Actual-weight oracle is opt-in; its fixture is generated locally.
  add_executable(gewell_mtp_assistant_oracle tests/mtp_assistant_oracle.cu)
  gewell_cuda_target(gewell_mtp_assistant_oracle)
  target_link_libraries(gewell_mtp_assistant_oracle PRIVATE gewell_gemma4_31b_sm120)
  foreach(benchmark IN ITEMS compact_attention prefill_chunk mtp_sampling)
    add_executable(gewell_${benchmark}_benchmark EXCLUDE_FROM_ALL tests/${benchmark}_benchmark.cu)
    gewell_cuda_target(gewell_${benchmark}_benchmark)
    target_link_libraries(gewell_${benchmark}_benchmark PRIVATE gewell_gemma4_31b_sm120)
  endforeach()
  add_executable(gewell_teacher_replay_benchmark EXCLUDE_FROM_ALL tests/teacher_replay_benchmark.cu)
  gewell_cuda_target(gewell_teacher_replay_benchmark)
  target_link_libraries(gewell_teacher_replay_benchmark PRIVATE gewell_gemma4_31b_sm120)
endif()

# Local research targets are omitted from the public source export.
include(experiments/CMakeLists.txt OPTIONAL)
