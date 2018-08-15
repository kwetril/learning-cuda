#pragma once

#include <cstddef>
#include <cstdint>

namespace learn_cuda
{

void CudaBlurImg(uint8_t const * img, size_t width, size_t height, size_t kernelSize, uint8_t * bluredImg);

}
