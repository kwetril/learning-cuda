#include "wrappers/add_vectors.hpp"
#include "util/vector_utils.hpp"

#include <iostream>


int main()
{
  using namespace learn_cuda;

  size_t const size = 100;
  size_t const fstSeed = 42;
  size_t const sndSeed = 123;
  std::vector<float> fst, snd;

  VectorFill(fst, size, fstSeed);
  VectorPrint("fst", fst);

  VectorFill(snd, size, sndSeed);
  VectorPrint("snd", snd);

  std::vector<float> res = VectorAdd(fst, snd);

  VectorPrint("res", res);

  return 0;
}
