#include "kernels/rgb_to_gray.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

int const kNumChannels = 3;
int const kBlockSize = 16;


__global__
void RgbToGrayKernel(uint8_t const * rgb, size_t width, size_t height, uint8_t * gray)
{
  size_t pixelCol = blockDim.x * blockIdx.x + threadIdx.x;
  size_t pixelRow = blockDim.y * blockIdx.y + threadIdx.y;
  if (pixelCol < width && pixelRow < height)
  {
    size_t offset = pixelRow * width + pixelCol;
    size_t rgbOffset = offset * kNumChannels;
    uint8_t r = rgb[rgbOffset];
    uint8_t g = rgb[rgbOffset + 1];
    uint8_t b = rgb[rgbOffset + 2];
    gray[offset] = static_cast<uint8_t>(0.21f * r + 0.71f * g + 0.07f * b);
  }
}


void CudaRgbToGray(uint8_t const * rgb, size_t width, size_t height, uint8_t * gray)
{
  size_t rgbSize = width * height * kNumChannels;
  size_t graySize = width * height;
  uint8_t * d_rgb, * d_gray;
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

}
