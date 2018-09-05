#include "wrappers/matrix_mult.hpp"
#include "kernels/matrix_mult.hpp"

#include <cstddef>
#include <random>

namespace learn_cuda
{

cv::Mat MatrixMult(cv::Mat const & first, cv::Mat const & second)
{
  cv::Size const firstSize = first.size();
  cv::Size const secondSize = second.size();
  cv::Mat result(firstSize.height, secondSize.width, CV_32F);
  CudaMatrixMult(reinterpret_cast<float *>(first.data), reinterpret_cast<float *>(second.data),
      firstSize.height, firstSize.width, secondSize.width, reinterpret_cast<float *>(result.data));
  return result;
}

}
