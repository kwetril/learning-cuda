#include "wrappers/add_vectors.hpp"
#include "kernels/add_vectors.hpp"

namespace learn_cuda
{

std::vector<float> VectorAdd(std::vector<float> const & fst, std::vector<float> const & snd)
{
  std::vector<float> res(fst.size());
  CudaVectorAdd(fst.data(), snd.data(), res.data(), fst.size());
  return res;
}

}
