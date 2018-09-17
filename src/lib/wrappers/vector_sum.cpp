#include "wrappers/vector_sum.hpp"
#include "kernels/vector_sum.hpp"

namespace learn_cuda
{

float VectorSum(std::vector<float> const & v)
{
  return CudaVectorSum(v.data(), v.size());
}

}
