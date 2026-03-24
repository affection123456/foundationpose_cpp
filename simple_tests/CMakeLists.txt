cmake_minimum_required(VERSION 3.8)
project(simple_tests)

add_compile_options(-std=c++17)
add_compile_options(-O3 -Wextra -Wdeprecated -fPIC)
set(CMAKE_CXX_STANDARD 17)

find_package(GTest REQUIRED)
find_package(glog REQUIRED)
# 测试需要 highgui(videoio) 用于 imshow/waitKey/VideoWriter
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui videoio)

# Add nlohmann/json - use single header file approach
# Try to find system-installed version first
find_package(nlohmann_json QUIET)
if(NOT nlohmann_json_FOUND)
  # Download single header file
  set(JSON_HEADER_DIR ${CMAKE_BINARY_DIR}/_deps/json-single/include)
  set(JSON_HEADER_FILE ${JSON_HEADER_DIR}/nlohmann/json.hpp)
  
  file(MAKE_DIRECTORY ${JSON_HEADER_DIR}/nlohmann)
  
  if(NOT EXISTS ${JSON_HEADER_FILE})
    message(STATUS "Downloading nlohmann/json single header file...")
    file(DOWNLOAD 
      https://github.com/nlohmann/json/releases/download/v3.11.2/json.hpp
      ${JSON_HEADER_FILE}
      TIMEOUT 30
      TLS_VERIFY ON
    )
  endif()
  
  # Create interface library with alias
  add_library(nlohmann_json INTERFACE)
  target_include_directories(nlohmann_json INTERFACE ${JSON_HEADER_DIR})
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
endif()

set(source_file
  src/test_foundationpose.cpp
)

enable_testing()

include_directories(
  include
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME} ${source_file})

# 显式链接 libcurl，满足 libgdal/libnetcdf 对 curl@CURL_OPENSSL_4 的符号（由 OpenCV imgcodecs 间接拉入）
find_package(CURL QUIET)
set(extra_libs "")
if(TARGET CURL::libcurl)
  list(APPEND extra_libs CURL::libcurl)
endif()

target_link_libraries(${PROJECT_NAME} PUBLIC
  GTest::gtest_main
  glog::glog
  ${OpenCV_LIBS}
  deploy_core
  trt_core
  detection_6d_foundationpose
  nlohmann_json::nlohmann_json
  ${extra_libs}
)



target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)
