#include "wrappers/img_blur.hpp"

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
  cv::Size const size = image.size();
  std::cout << "Image " << imgPath << ": width=" << size.width << ", height=" << size.height
    << ", channels=" << image.channels() << std::endl;

  cv::Mat const bluredImage = BlurImg(image, 7);
  std::string outputPath = "out.png";
  cv::imwrite(outputPath, bluredImage);
  cv::imshow("Blured image", bluredImage);
  cv::waitKey();
  return 0;
}
