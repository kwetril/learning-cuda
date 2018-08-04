#include "wrappers/rgb_to_gray.hpp"

#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>


int main()
{
  using namespace learn_cuda;
  std::string const rootDir = "..";
  std::string const imgPath = rootDir + "/data/test.png";
  cv::Mat image = cv::imread(imgPath, cv::IMREAD_COLOR);
  if(!image.data)
  {
    std::cout << "Couldn't open or find the image: " << imgPath << std::endl;
    return -1;
  }
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
  cv::Size const size = image.size();
  std::cout << "Image " << imgPath << ": width=" << size.width << ", height=" << size.height
    << ", channels=" << image.channels() << std::endl;

  cv::Mat const grayImage = RgbToGray(image);
  std::string outputPath = "out.png";
  cv::imwrite(outputPath, grayImage);
  std::cout << "Done" << std::endl;

  return 0;
}
