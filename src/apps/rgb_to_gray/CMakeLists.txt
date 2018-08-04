include_directories(
  "${ROOT_DIR}/src/lib"
)

cuda_add_executable(rgb_to_gray main.cpp)

target_link_libraries(rgb_to_gray gpulib)