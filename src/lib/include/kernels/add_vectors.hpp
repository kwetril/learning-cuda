#pragma once

#include <cstddef>

namespace learn_cuda
{

void CudaVectorAdd(float const * fst, float const * snd, float * res, size_t size);

}
