#pragma once

#include <cstddef>
#include <opencv2/opencv.hpp>


namespace learn_cuda
{

cv::Mat RgbToGray(cv::Mat const & image);

}
