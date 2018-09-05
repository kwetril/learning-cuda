#include "wrappers/add_vectors.hpp"
#include "util/vector_utils.hpp"

#include <iostream>
#include <cstdlib>
#include <ctime>


int main()
{
  using namespace learn_cuda;

  int size = 100;
  std::vector<float> fst, snd;

  srand(static_cast<unsigned int>(time(NULL)));

  VectorFill(fst, size);
  VectorPrint("fst", fst);

  VectorFill(snd, size);
  VectorPrint("snd", snd);

  std::vector<float> res = VectorAdd(fst, snd);

  VectorPrint("res", res);

  return 0;
}
