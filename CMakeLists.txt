project(cuda-learning)

cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++11" )

include_directories(${CUDA_INCLUDE_DIRS})

add_subdirectory(src)
