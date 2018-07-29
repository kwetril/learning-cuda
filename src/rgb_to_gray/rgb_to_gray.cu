#include "rgb_to_gray.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>


int const kNumChannels = 3;
int const kBlockSize = 16;


__global__
void RgbToGrayKernel(unsigned char const * rgb, int width, int height, unsigned char * gray)
{
  int pixelCol = blockDim.x * blockIdx.x + threadIdx.x;
  int pixelRow = blockDim.y * blockIdx.y + threadIdx.y;
  if (pixelCol < width && pixelRow < height)
  {
    int offset = pixelRow * width + pixelCol;
    int rgbOffset = offset * kNumChannels;
    unsigned char r = rgb[rgbOffset];
    unsigned char g = rgb[rgbOffset + 1];
    unsigned char b = rgb[rgbOffset + 2];
    gray[offset] = static_cast<unsigned char>(0.21f * r + 0.71f * g + 0.07f * b);
  }
}


void RgbToGray(unsigned char const * rgb, int width, int height, unsigned char * gray)
{
  int rgbSize = width * height * kNumChannels;
  int graySize = width * height;
  unsigned char * d_rgb, * d_gray;
  cudaMalloc(reinterpret_cast<void **>(&d_rgb), rgbSize);
  cudaMemcpy(d_rgb, rgb, rgbSize, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_gray), graySize);

  dim3 gridDimensions(ceil(width / static_cast<double>(kBlockSize)), ceil(height / static_cast<double>(kBlockSize)), 1);
  dim3 blockDimensions(kBlockSize, kBlockSize, 1);
  RgbToGrayKernel<<<gridDimensions,blockDimensions>>>(d_rgb, width, height, d_gray);

  cudaMemcpy(gray, d_gray, graySize, cudaMemcpyDeviceToHost);
  cudaFree(d_rgb);
  cudaFree(d_gray);
}
