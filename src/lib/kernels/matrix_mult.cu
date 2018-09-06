#include "kernels/matrix_mult.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

int const BlockSize = 16;

__global__
void MatrixMultKernel(float const * first, float const * second, size_t n, size_t m, size_t k, float * result)
{
  size_t resultRow = BlockSize * blockIdx.y + threadIdx.y;
  size_t resultCol = BlockSize * blockIdx.x + threadIdx.x;

  if (resultRow < n && resultCol < k)
  {
    double sum = 0.;
    for (size_t i = 0; i < m; ++i)
    {
      sum += static_cast<double>(first[resultRow * m + i]) * static_cast<double>(second[i * k + resultCol]);
    }
    result[resultRow * k + resultCol] = static_cast<float>(sum);
  }
}

__global__
void TiledMatrixMultKernel(float const * first, float const * second, size_t n, size_t m, size_t k, float * result)
{
  __shared__ float firstTile[BlockSize][BlockSize];
  __shared__ float secondTile[BlockSize][BlockSize];

  size_t const tx = threadIdx.x;
  size_t const ty = threadIdx.y;
  size_t const bx = blockIdx.x;
  size_t const by = blockIdx.y;

  size_t const resultRow = BlockSize * by + ty;
  size_t const resultCol = BlockSize * bx + tx;

  double sum = 0.;
  size_t const numTiles = ceil(m / static_cast<float>(BlockSize));
  for (size_t tile = 0; tile < numTiles; ++tile)
  {
    // each thread in block loads its matrix elements to shared memory
    if (resultRow < n && (tile * BlockSize + tx) < m)
    {
      firstTile[ty][tx] = first[(resultRow * m) + (tile * BlockSize + tx)];
    }
    else
    {
      firstTile[ty][tx] = 0.f;
    }

    if ((tile * BlockSize + ty) < m && resultCol < k)
    {
      secondTile[ty][tx] = second[((tile * BlockSize + ty) * k) + (resultCol)];
    }
    else
    {
      secondTile[ty][tx] = 0.f;
    }
    __syncthreads();

    for (size_t i = 0; i < BlockSize; ++i)
    {
      sum += static_cast<double>(firstTile[ty][i]) * static_cast<double>(secondTile[i][tx]);
    }
    __syncthreads();
  }

  if (resultRow < n && resultCol < k)
  {
    result[resultRow * k + resultCol] = static_cast<float>(sum);
  }
}

void CudaMatrixMult(float const * first, float const * second, size_t n, size_t m, size_t k, float * result, bool useTiles)
{
  size_t const firstSize = n * m * sizeof(float);
  size_t const secondSize = m * k * sizeof(float);
  size_t const resultSize = n * k * sizeof(float);
  float * d_first, * d_second, * d_result;
  cudaMalloc(reinterpret_cast<void **>(&d_first), firstSize);
  cudaMemcpy(d_first, first, firstSize, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_second), secondSize);
  cudaMemcpy(d_second, second, secondSize, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_result), resultSize);

  dim3 gridDimensions(ceil(k / static_cast<double>(BlockSize)), ceil(n / static_cast<double>(BlockSize)), 1);
  dim3 blockDimensions(BlockSize, BlockSize, 1);
  if (useTiles)
  {
    TiledMatrixMultKernel<<<gridDimensions,blockDimensions>>>(d_first, d_second, n, m, k, d_result);
  }
  else
  {
    MatrixMultKernel<<<gridDimensions,blockDimensions>>>(d_first, d_second, n, m, k, d_result);
  }

  cudaMemcpy(result, d_result, resultSize, cudaMemcpyDeviceToHost);
  cudaFree(d_first);
  cudaFree(d_second);
  cudaFree(d_result);
}

}
