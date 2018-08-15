#include "kernels/rgb_to_gray.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

int const kBlockSize = 16;
int const kNumChannels = 3;


__global__
void BlurImgKernel(uint8_t const * img, int width, int height, int kernelSize, uint8_t * blurImg)
{
  int pixelCol = blockDim.x * blockIdx.x + threadIdx.x;
  int pixelRow = blockDim.y * blockIdx.y + threadIdx.y;
  if (pixelCol < width && pixelRow < height)
  {
    int bluePixelSum = 0;
    int greenPixelSum = 0;
    int redPixelSum = 0;
    size_t numPixels = 0;
    int const halfKernelSize = kernelSize / 2;
    for (int dRow = -halfKernelSize; dRow <= halfKernelSize; ++dRow)
    {
      for (int dCol = -halfKernelSize; dCol <= halfKernelSize; ++dCol)
      {
        int col = pixelCol + dCol;
        int row = pixelRow + dRow;
        if (0 <= col && col < width && 0 <= row && row < height)
        {
          int offset = (row * width + col) * kNumChannels;
          bluePixelSum += img[offset];
          greenPixelSum += img[offset + 1];
          redPixelSum += img[offset + 2];
          numPixels++;
        }
      }
    }
    int pixelOffset = (pixelRow * width + pixelCol) * kNumChannels;
    blurImg[pixelOffset] = static_cast<uint8_t>(bluePixelSum / numPixels);
    blurImg[pixelOffset + 1] = static_cast<uint8_t>(greenPixelSum / numPixels);
    blurImg[pixelOffset + 2] = static_cast<uint8_t>(redPixelSum / numPixels);
  }
}


void CudaBlurImg(uint8_t const * img, size_t width, size_t height, size_t kernelSize, uint8_t * bluredImg)
{
  size_t imgSize = width * height * kNumChannels;
  uint8_t * d_image, * d_bluredImg;
  cudaMalloc(reinterpret_cast<void **>(&d_image), imgSize);
  cudaMemcpy(d_image, img, imgSize, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_bluredImg), imgSize);

  dim3 gridDimensions(ceil(width / static_cast<double>(kBlockSize)), ceil(height / static_cast<double>(kBlockSize)), 1);
  dim3 blockDimensions(kBlockSize, kBlockSize, 1);
  BlurImgKernel<<<gridDimensions,blockDimensions>>>(d_image, width, height, kernelSize, d_bluredImg);

  cudaMemcpy(bluredImg, d_bluredImg, imgSize, cudaMemcpyDeviceToHost);
  cudaFree(d_image);
  cudaFree(d_bluredImg);
}

}
