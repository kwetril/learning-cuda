#pragma once

#include <cstddef>
#include <cstdint>

namespace learn_cuda
{

void CudaRgbToGray(uint8_t const * rgb, size_t width, size_t height, uint8_t * gray);

}
