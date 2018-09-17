#include "kernels/add_vectors.hpp"

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

size_t constexpr BlockSize = 16;

__global__
void VectorSumKernel(float const * data, size_t size, float * tmpSum)
{
  __shared__ float partialSum[BlockSize];

  size_t const tid = threadIdx.x;
  size_t const bid = blockIdx.x;
  size_t const dataIdx = 2 * blockDim.x * bid + tid;
  if (dataIdx < size)
  {
    partialSum[tid] = data[dataIdx];
  }
  else
  {
    partialSum[tid] = 0.f;
  }

  if (dataIdx + BlockSize < size)
  {
    partialSum[tid] += data[dataIdx + BlockSize];
  }

  for (size_t stride = blockDim.x / 2; stride >= 1; stride >>= 1)
  {
    __syncthreads();
    if (tid < stride)
    {
      partialSum[tid] += partialSum[tid + stride];
    }
  }

  if (tid == 0)
  {
    tmpSum[bid] = partialSum[0];
  }
}

float CudaVectorSum(float const * data, size_t size)
{
  size_t sizeBytes = size * sizeof(float);
  float * d_data, * d_tmpSum, * d_tmpPtr;

  cudaMalloc(reinterpret_cast<void **>(&d_data), sizeBytes);
  cudaMemcpy(d_data, data, sizeBytes, cudaMemcpyHostToDevice);

  size_t numBlocks = ceil(size / 2.f / (float) BlockSize);
  cudaMalloc(reinterpret_cast<void **>(&d_tmpSum), sizeof(float) * numBlocks);

  do
  {
    numBlocks = ceil(size / 2.f / (float) BlockSize);
    VectorSumKernel<<<numBlocks, BlockSize>>>(d_data, size, d_tmpSum);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
        break;
    }

    d_tmpPtr = d_data;
    d_data = d_tmpSum;
    d_tmpSum = d_tmpPtr;
    size = numBlocks;
  } while (numBlocks > 1);

  float res;
  cudaMemcpy(&res, d_data, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_data);
  cudaFree(d_tmpSum);

  return res;
}

}
