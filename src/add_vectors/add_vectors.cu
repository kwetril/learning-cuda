#include "add_vectors.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

void VectorFill(float * v, int size)
{
  for (size_t i = 0; i < size; ++i)
     v[i] = static_cast<float>(rand()) / RAND_MAX;
}


__global__
void VectorAddKernel(float const * fst, float const * snd, float * res, int size)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < size)
    res[i] = fst[i] + snd[i];
}


void VectorAdd(float const * fst, float const * snd, float * res, int size, bool use_gpu)
{
  if (use_gpu)
  {
    printf("using gpu\n");
    int size_bytes = size * sizeof(float);
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
  else
  {
    printf("using cpu\n");
    for (int i = 0; i < size; ++i)
      res[i] = fst[i] + snd[i];
  }
}

void VectorPrint(char const * name, float const * v, int size)
{
  printf("%s [", name);
  for (size_t i = 0; i < size; ++i)
     printf(" %.3f", v[i]);
  printf("]\n");
}
