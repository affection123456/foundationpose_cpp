cmake_minimum_required(VERSION 3.8)
project(foundationpose_cpp)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 避免 CMP0104：未设置时默认 86 (Ampere)，可通过 -DCMAKE_CUDA_ARCHITECTURES="86;87" 覆盖
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "CUDA architectures")
endif()

add_subdirectory(easy_deploy_tool)
add_subdirectory(detection_6d_foundationpose)

add_subdirectory(simple_tests)
