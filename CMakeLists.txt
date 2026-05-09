cmake_minimum_required(VERSION 3.20)

project(edgespeech_rt VERSION 0.1.0 LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(EDGESPEECH_ENABLE_ORT "Build the ONNX Runtime C++ inference backend" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(edgespeech_rt
  cpp/src/audio_buffer.cpp
  cpp/src/enhancer.cpp
  cpp/src/stft.cpp
  cpp/src/wav_io.cpp
)

target_include_directories(edgespeech_rt
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cpp/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(edgespeech_rt PUBLIC cxx_std_17)

if(EDGESPEECH_ENABLE_ORT)
  set(ONNXRUNTIME_ROOT "" CACHE PATH "Path to an ONNX Runtime C/C++ distribution")
  find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS
      ${ONNXRUNTIME_ROOT}/include
      ${ONNXRUNTIME_ROOT}/include/onnxruntime
  )
  find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime libonnxruntime.1.23.2.dylib libonnxruntime.dylib
    HINTS
      ${ONNXRUNTIME_ROOT}/lib
      ${ONNXRUNTIME_ROOT}/runtimes/osx-arm64/native
      ${ONNXRUNTIME_ROOT}/runtimes/osx-x64/native
  )

  if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
    message(FATAL_ERROR "ONNX Runtime C++ headers/library not found. Set ONNXRUNTIME_ROOT or build with EDGESPEECH_ENABLE_ORT=OFF.")
  endif()

  target_include_directories(edgespeech_rt PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
  target_link_libraries(edgespeech_rt PRIVATE ${ONNXRUNTIME_LIBRARY})
  target_compile_definitions(edgespeech_rt PRIVATE EDGESPEECH_HAS_ONNXRUNTIME=1)
else()
  target_compile_definitions(edgespeech_rt PRIVATE EDGESPEECH_HAS_ONNXRUNTIME=0)
endif()

if(MSVC)
  target_compile_options(edgespeech_rt PRIVATE /W4)
else()
  target_compile_options(edgespeech_rt PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(edgespeech_wav_cli cpp/src/main_wav_cli.cpp)
target_link_libraries(edgespeech_wav_cli PRIVATE edgespeech_rt)

enable_testing()

add_executable(test_stft_roundtrip tests/test_stft_roundtrip.cpp)
target_link_libraries(test_stft_roundtrip PRIVATE edgespeech_rt)
add_test(NAME stft_roundtrip COMMAND test_stft_roundtrip)

add_executable(test_streaming_state_reset tests/test_streaming_state_reset.cpp)
target_link_libraries(test_streaming_state_reset PRIVATE edgespeech_rt)
add_test(NAME streaming_state_reset COMMAND test_streaming_state_reset)

add_executable(test_latency_budget tests/test_latency_budget.cpp)
target_link_libraries(test_latency_budget PRIVATE edgespeech_rt)
add_test(NAME latency_budget COMMAND test_latency_budget)
