#include <iostream>
#include <cuda_runtime_api.h>


int main()
{
  int deviceCount;
  cudaGetDeviceCount(&deviceCount);
  std::cout.setf(std::ios::fixed);
  std::cout << "Found " << deviceCount << " cuda devices:" << std::endl;
  for (int i = 0; i < deviceCount; ++i)
  {
    std::cout << "Device " << i << ":" << std::endl;
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, i);
    std::cout << "\tName: " << deviceProperties.name << std::endl;
    std::cout << "\tNumber of multiprocessors: " << deviceProperties.multiProcessorCount << std::endl;
    std::cout.precision(3);
    std::cout << "\tClock frequency: " << deviceProperties.clockRate / 1000000. << " MHz" << std::endl;
    std::cout << "\tNumber of threads in a warp: " << deviceProperties.warpSize << std::endl;
    std::cout << "\tMax threads per block: " << deviceProperties.maxThreadsPerBlock << std::endl;
    std::cout << "\tMax number of blocks in grid for each dimension: [" << deviceProperties.maxGridSize[0] << ", "
      << deviceProperties.maxGridSize[1] << ", " << deviceProperties.maxGridSize[2] << "]" << std::endl;
    std::cout << "\tMax number of treads in block for each dimension: [" << deviceProperties.maxThreadsDim[0] << ", "
      << deviceProperties.maxThreadsDim[1] << ", " << deviceProperties.maxThreadsDim[2] << "]" << std::endl;

    std::cout << "\tMemory info:" << std::endl;
    std::cout.precision(2);
    std::cout << "\t\tTotal global memory: " << deviceProperties.totalGlobalMem / 1024. / 1024. / 1024. << " Gb" << std::endl;
    std::cout.precision(3);
    std::cout << "\t\tMemory clock rate: " << deviceProperties.memoryClockRate / 1000000. << " MHz" << std::endl;
    std::cout << "\t\tMemory bus width: " << deviceProperties.memoryBusWidth << std::endl;
    std::cout.precision(2);
    std::cout << "\t\tShared memory per block: " << deviceProperties.sharedMemPerBlock / 1024. << " Kb\n";
  }

  return 0;
}
