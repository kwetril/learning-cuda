#include <algorithm>
#include <iostream>

#include "wrappers/vector_sum.hpp"
#include "util/vector_utils.hpp"
#include "util/timer.hpp"

int main()
{
  using namespace learn_cuda;

  size_t const size = 100000000;
  size_t const seed = 42;
  std::vector<float> v;
  Timer timer;

  VectorFill(v, size, seed);

  timer.Start();
  float const resCpu = std::accumulate(v.begin(), v.end(), 0.);
  double const cpuTimeMs = timer.Stop();

  timer.Start();
  float const resGpu = VectorSum(v);
  double const gpuTimeMs = timer.Stop();

  std::cout.setf(std::ios::fixed);
  std::cout.precision(3);
  std::cout << "Results:" << std::endl;
  std::cout << "cpu sum: " << resCpu << std::endl;
  std::cout << "gpu sum: " << resGpu << std::endl;
  std::cout << std::endl;
  std::cout << "Execution time:" << std::endl;
  std::cout << "cpu: " << cpuTimeMs << " ms" << std::endl;
  std::cout << "gpu: " << gpuTimeMs << " ms" << std::endl;


  return 0;
}
