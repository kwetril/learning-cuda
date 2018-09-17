#include "util/vector_utils.hpp"

#include <iostream>
#include <random>

namespace learn_cuda
{

void VectorFill(std::vector<float> & v, size_t size, size_t seed)
{
  std::mt19937 generator(seed);
  std::uniform_real_distribution<float> distribution(-1.f, 1.f);
  v.resize(size);
  for (size_t i = 0; i < size; ++i)
     v[i] = distribution(generator);
}

void VectorPrint(std::string const & name, std::vector<float> const & v)
{
  std::cout << name << " [";
  std::cout.setf(std::ios::fixed);
  std::cout.precision(3);
  for (size_t i = 0; i < v.size(); ++i)
    std::cout << " " << v[i];
  std::cout << "]" << std::endl;
}

}
