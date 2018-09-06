#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>

#include "util/timer.hpp"
#include "util/matrix_utils.hpp"
#include "wrappers/matrix_mult.hpp"

float constexpr MinValue = -10.;
float constexpr MaxValue = 10.;

int main()
{
  using namespace learn_cuda;

  size_t const n = 1600;
  size_t const m = 600;
  size_t const k = 1280;

  cv::Mat first(n, m, CV_32F);
  MatrixFill(first, MinValue, MaxValue, 42);
  cv::Mat second(m, k, CV_32F);
  MatrixFill(second, MinValue, MaxValue, 123);

  Timer timer;

  timer.Start();
  cv::Mat const cvMult = first * second;
  double const cvTime = timer.Stop();

  timer.Start();
  cv::Mat const cudaMult = MatrixMult(first, second);
  double const cudaTime = timer.Stop();

  timer.Start();
  cv::Mat const cudaTiledMult = TiledMatrixMult(first, second);
  double const cudaTiledTime = timer.Stop();

  std::cout << "MSE[OpenCV, CUDA]: " << MatrixDiffMSE(cvMult, cudaMult) << std::endl;
  std::cout << "MSE[OpenCV, CUDA tiled]: " << MatrixDiffMSE(cvMult, cudaTiledMult) << std::endl;
  std::cout << "OpenCV time: " << cvTime << " ms" << std::endl;
  std::cout << "CUDA time: " << cudaTime << " ms" << std::endl;
  std::cout << "CUDA tiled time: " << cudaTiledTime << " ms" << std::endl;
  return 0;
}
