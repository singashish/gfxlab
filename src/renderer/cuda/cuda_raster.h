#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "core/mat4.h"
#include "renderer/cuda/cuda_mesh.h"

namespace gfx {

struct ProjectedVertex {
    float4 screen;       // screen x, screen y, depth01, clip-space w
    float4 view_pos;     // view-space position
    float4 view_normal;  // view-space normal
};

bool cuda_clear(uchar4* color, int width, int height, uchar4 value);

bool cuda_clear_visibility(std::uint64_t* visibility, int width, int height);

bool cuda_transform_vertices(
                const DeviceMesh& mesh,
                Mat4 mvp,
                Mat4 model_view,
                Mat4 normal_to_view,
                int width,
                int height,
                ProjectedVertex* out);

bool cuda_rasterize_triangles(
                std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base);

bool cuda_shade_item(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base,
                Vec3 up_view);

bool cuda_resolve_ids(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height);

} // namespace gfx
