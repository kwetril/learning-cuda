include_directories(
  "${ROOT_DIR}/src/lib"
)

cuda_add_executable(reduce_vector main.cpp)

target_link_libraries(reduce_vector gpulib)