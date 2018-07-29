project(learning-cuda)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(${CUDA_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)

add_subdirectory(src)
