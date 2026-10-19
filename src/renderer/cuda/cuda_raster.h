#pragma once

#include <cuda_runtime.h>

#include "core/mat4.h"
#include "renderer/cuda/cuda_mesh.h"

namespace gfx {

bool cuda_clear(uchar4* color, int width, int height, uchar4 value);

bool cuda_draw_points(uchar4* color, int width, int height,
                      const DeviceMesh& mesh, Mat4 mvp);

} // namespace gfx
