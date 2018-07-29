cuda_add_executable(rgb_to_gray main.cpp rgb_to_gray.hpp rgb_to_gray.cu)

target_link_libraries(rgb_to_gray ${OpenCV_LIBRARIES})