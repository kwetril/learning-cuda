#include "util/matrix_utils.hpp"

namespace learn_cuda
{

void MatrixFill(cv::Mat & matrix, float minValue, float maxValue, int randomSeed)
{
  float * ptr = reinterpret_cast<float *>(matrix.data);
  float const * endPtr = reinterpret_cast<float const *>(matrix.dataend);

  std::mt19937 randomGenerator(randomSeed);
  std::uniform_real_distribution<float> uniform(minValue, maxValue);
  while (ptr != endPtr)
  {
    *ptr = uniform(randomGenerator);
    ++ptr;
  }
}

double MatrixDiffMSE(cv::Mat const & first, cv::Mat const & second)
{
  assert(first.size() == second.size());

  float * firstPtr = reinterpret_cast<float *>(first.data);
  float const * firstEnd = reinterpret_cast<float const *>(first.dataend);
  float * secondPtr = reinterpret_cast<float *>(second.data);

  double res = 0.;
  while (firstPtr != firstEnd)
  {
    double diff = *firstPtr - *secondPtr;
    res += diff * diff;
    ++firstPtr;
    ++secondPtr;
  }

  return res;
}

}

