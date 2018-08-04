#pragma once

#include "wrappers/rgb_to_gray.hpp"
#include "kernels/rgb_to_gray.hpp"

#include <cstddef>

namespace learn_cuda
{

cv::Mat RgbToGray(cv::Mat const & image)
{
  cv::Size const size = image.size();
  cv::Mat grayImage(size.height, size.width, CV_8U);
  CudaRgbToGray(image.data, size.width, size.height, grayImage.data);
  return grayImage;
}

}
