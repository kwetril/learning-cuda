#pragma once

#include <vector>
#include <string>

namespace learn_cuda
{

void VectorFill(std::vector<float> & v, size_t size, size_t seed);

void VectorPrint(std::string const & name, std::vector<float> const & v);

}
