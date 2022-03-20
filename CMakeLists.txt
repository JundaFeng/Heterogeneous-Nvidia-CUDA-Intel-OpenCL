cmake_minimum_required(VERSION 3.10)
project(test_cuda_opencl LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_MAKEFILE ON)
set(VERBOSE_BUILD ON)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CUDA_STANDARD 14)
set(DEFAULT_BUILD_TYPE "Release")

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
add_executable(test_cuda_opencl main.cu cxxtimer.hpp)

set_target_properties(test_cuda_opencl PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_include_directories (test_cuda_opencl PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries (test_cuda_opencl ${OpenCL_LIBRARY})
target_link_libraries(test_cuda_opencl ${CUDA_LIBRARIES})
include_directories(test_cuda_opencl ${CUDA_INCLUDE_DIRS})
set_target_properties(test_cuda_opencl PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON)
