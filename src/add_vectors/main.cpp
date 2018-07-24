#include "add_vectors.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>


int main()
{
  int size = 10000;
  float * fst = reinterpret_cast<float *>(malloc(size * sizeof(float)));
  float * snd = reinterpret_cast<float *>(malloc(size * sizeof(float)));
  float * res = reinterpret_cast<float *>(malloc(size * sizeof(float)));

  srand(static_cast<unsigned int>(time(NULL)));

  VectorFill(fst, size);
  VectorPrint("fst", fst, size);

  VectorFill(snd, size);
  VectorPrint("snd", snd, size);

  VectorAdd(fst, snd, res, size, true);

  VectorPrint("res", res, size);

  return 0;
}
