#pragma once

#include <cstddef>
#include <cstdint>

namespace learn_cuda
{

void CudaMatrixMult(float const * first, float const * second, size_t n, size_t m, size_t k, float * result);

}
