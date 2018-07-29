#include <stdio.h>
#include <cuda_runtime_api.h>


int main()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  printf("Found %d cuda devices:\n", deviceCount);
  for (int i = 0; i < deviceCount; ++i)
  {
    printf("Device %d:\n", i);
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, i);
    printf("\tName: %s\n", deviceProperties.name);
    printf("\tNumber of multiprocessors: %d\n", deviceProperties.multiProcessorCount);
    printf("\tClock frequency: %.3f MHz\n", deviceProperties.clockRate / 1000000.);
    printf("\tNumber of threads in a warp: %d\n", deviceProperties.warpSize);
    printf("\tMax threads per block: %d\n", deviceProperties.maxThreadsPerBlock);
    printf("\tMax number of blocks in grid for each dimension: [%d, %d, %d]\n",
        deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1], deviceProperties.maxGridSize[2]);
    printf("\tMax number of treads in block for each dimension: [%d, %d, %d]\n",
        deviceProperties.maxThreadsDim[0], deviceProperties.maxThreadsDim[1], deviceProperties.maxThreadsDim[2]);

    printf("\tMemory info:\n");
    printf("\t\tTotal global memory: %.2f Gb\n", deviceProperties.totalGlobalMem / 1024. / 1024. / 1024.);
    printf("\t\tMemory clock rate: %.3f MHz\n", deviceProperties.memoryClockRate / 1000000. );
    printf("\t\tMemory bus width: %d\n", deviceProperties.memoryBusWidth);
    printf("\t\tShared memory per block: %.2f Kb\n", deviceProperties.sharedMemPerBlock / 1024.);
  }

  return 0;
}
