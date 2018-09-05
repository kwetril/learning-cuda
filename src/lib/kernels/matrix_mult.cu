#include "kernels/matrix_mult.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

int const kBlockSize = 16;

__global__
void MatrixMultKernel(float const * first, float const * second, size_t n, size_t m, size_t k, float * result)
{
  size_t resultRow = blockDim.y * blockIdx.y + threadIdx.y;
  size_t resultCol = blockDim.x * blockIdx.x + threadIdx.x;

  if (resultRow < n && resultCol < k)
  {
    double sum = 0.;
    for (size_t i = 0; i < m; ++i)
    {
      sum += first[resultRow * m + i] * second[i * k + resultCol];
    }
    result[resultRow * k + resultCol] = static_cast<float>(sum);
  }
}

void CudaMatrixMult(float const * first, float const * second, size_t n, size_t m, size_t k, float * result)
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

  dim3 gridDimensions(ceil(k / static_cast<double>(kBlockSize)), ceil(n / static_cast<double>(kBlockSize)), 1);
  dim3 blockDimensions(kBlockSize, kBlockSize, 1);
  MatrixMultKernel<<<gridDimensions,blockDimensions>>>(d_first, d_second, n, m, k, d_result);

  cudaMemcpy(result, d_result, resultSize, cudaMemcpyDeviceToHost);
  cudaFree(d_first);
  cudaFree(d_second);
  cudaFree(d_result);
}

}
