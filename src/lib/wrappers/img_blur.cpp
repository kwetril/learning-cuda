#include "wrappers/img_blur.hpp"
#include "kernels/img_blur.hpp"

#include <cstddef>

namespace learn_cuda
{

cv::Mat BlurImg(cv::Mat const & image, size_t kernelSize)
{
  cv::Size const size = image.size();
  cv::Mat bluredImage(size.height, size.width, CV_8UC3);
  CudaBlurImg(image.data, size.width, size.height, kernelSize, bluredImage.data);
  return bluredImage;
}

}
