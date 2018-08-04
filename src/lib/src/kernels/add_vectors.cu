#include "kernels/add_vectors.hpp"

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace learn_cuda
{

__global__
void VectorAddKernel(float const * fst, float const * snd, float * res, size_t size)
{
  size_t i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
    res[i] = fst[i] + snd[i];
}

void CudaVectorAdd(float const * fst, float const * snd, float * res, size_t size)
{
  size_t size_bytes = size * sizeof(float);
  float * d_fst, * d_snd, * d_res;
  cudaMalloc(reinterpret_cast<void **>(&d_fst), size_bytes);
  cudaMemcpy(d_fst, fst, size_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_snd), size_bytes);
  cudaMemcpy(d_snd, snd, size_bytes, cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&d_res), size_bytes);

  // run n/256 blocks of 256 threads each
  VectorAddKernel<<<ceil(size / 256.0), 256>>>(d_fst, d_snd, d_res, size);

  cudaMemcpy(res, d_res, size_bytes, cudaMemcpyDeviceToHost);
  cudaFree(d_fst);
  cudaFree(d_snd);
  cudaFree(d_res);
}

}
