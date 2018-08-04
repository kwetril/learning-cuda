#include "util/vector_utils.hpp"

#include <iostream>
#include <stdlib.h>

namespace learn_cuda
{

void VectorFill(std::vector<float> & v, size_t size)
{
  v.resize(size);
  for (size_t i = 0; i < size; ++i)
     v[i] = static_cast<float>(rand()) / RAND_MAX;
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
