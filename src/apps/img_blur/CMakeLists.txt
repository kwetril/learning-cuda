include_directories(
  "${ROOT_DIR}/src/lib"
)

cuda_add_executable(img_blur main.cpp)

target_link_libraries(img_blur gpulib)