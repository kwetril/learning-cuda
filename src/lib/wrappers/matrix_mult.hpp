#pragma once

#include <cstddef>
#include <opencv2/opencv.hpp>


namespace learn_cuda
{

cv::Mat MatrixFill(cv::Mat & matrix, double minValue, double maxValue);

cv::Mat MatrixMult(cv::Mat const & first, cv::Mat const & second);

}
