project(learning-cuda)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(ROOT_DIR "${CMAKE_SOURCE_DIR}")

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_subdirectory(src/lib lib)
add_subdirectory(src/apps apps)
