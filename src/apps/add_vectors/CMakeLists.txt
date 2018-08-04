include_directories(
  "${ROOT_DIR}/src/lib"
)

cuda_add_executable(add_vectors main.cpp)

target_link_libraries(add_vectors gpulib)