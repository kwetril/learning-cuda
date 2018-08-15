#pragma once

#include <cstddef>
#include <opencv2/opencv.hpp>


namespace learn_cuda
{

cv::Mat BlurImg(cv::Mat const & image, size_t kernelSize);

}
