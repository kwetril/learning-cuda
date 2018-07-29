#include "rgb_to_gray.hpp"

#include <stdio.h>
#include <opencv2/opencv.hpp>


int main()
{
  char imgPath[] = "lena.png";
  cv::Mat image;
  image = cv::imread(imgPath, cv::IMREAD_COLOR);
  cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

  if(!image.data)
  {
      printf("Couldn't open or find the image: %s", imgPath);
      return -1;
  }

  int width = image.size().width;
  int height = image.size().height;
  printf("Image info: width=%d, height=%d, channels=%d\n", width, height, image.channels());

  cv::Mat grayImage(height, width, CV_8U);
  char outputPath[] = "out.png";
  RgbToGray(image.data, width, height, grayImage.data);
  cv::imwrite(outputPath, grayImage);
  printf("Done\n");

  return 0;
}
