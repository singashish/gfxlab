#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "core/mat4.h"
#include "renderer/cuda/cuda_mesh.h"

namespace gfx {

bool cuda_clear(uchar4* color, int width, int height, uchar4 value);

bool cuda_clear_visibility(std::uint64_t* visibility,
                           int width, int height);

bool cuda_transform_vertices(const DeviceMesh& mesh, Mat4 mvp,
                             int width, int height, float4* out);

bool cuda_rasterize_triangles(std::uint64_t* visibility, int width,
            int height, const DeviceMesh& mesh, const float4* vertices,
            std::uint32_t triangle_base);

bool cuda_resolve_ids(uchar4* color, const std::uint64_t* visibility,
                      int width, int height);

} // namespace gfx
