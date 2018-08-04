project (gpulib)
cmake_minimum_required (VERSION 2.8)

file(GLOB_RECURSE KERNELS_SOURCES "kernels/*.cu")
file(GLOB_RECURSE WRAPPERS_SOURCES "wrappers/*.cpp")
file(GLOB_RECURSE UTIL_SOURCES "util/*.cpp")

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

cuda_add_library(gpulib SHARED ${KERNELS_SOURCES} ${WRAPPERS_SOURCES} ${UTIL_SOURCES})

target_link_libraries(gpulib ${OpenCV_LIBRARIES})
