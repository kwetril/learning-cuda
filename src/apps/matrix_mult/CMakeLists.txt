include_directories(
  "${ROOT_DIR}/src/lib"
)

cuda_add_executable(matrix_mult main.cpp)

target_link_libraries(matrix_mult gpulib)