#pragma once

#include <cuda_runtime.h>

namespace gfx {

bool cuda_fill_gradient(uchar4* pixels, int width, int height);

} // namespace gfx
