#pragma once

#include <cstddef>
#include <opencv2/opencv.hpp>


namespace learn_cuda
{

cv::Mat MatrixMult(cv::Mat const & first, cv::Mat const & second);

cv::Mat TiledMatrixMult(cv::Mat const & first, cv::Mat const & second);

}
