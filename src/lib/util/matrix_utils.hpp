#pragma once

#include <opencv2/opencv.hpp>

namespace learn_cuda
{

void MatrixFill(cv::Mat & matrix, float minValue, float maxValue, int randomSeed);

double MatrixDiffMSE(cv::Mat const & first, cv::Mat const & second);

}
