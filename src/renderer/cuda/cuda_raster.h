#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "core/mat4.h"
#include "core/raster_math.h"
#include "renderer/cuda/cuda_mesh.h"

namespace gfx {

struct ProjectedVertex {
    float4 screen;       // screen x, screen y, depth01, clip-space w
    float4 view_pos;     // view-space position
    float4 view_normal;  // view-space normal
};

// One triangle that survived setup, ready for rasterization and shading.
// All per-vertex arrays are in the setup's positive-area order, so
// screen[k] belongs to (setup.sx[k], setup.sy[k])
struct RasterTri {
    RasterTriangleSetup setup;  // snapped coordinates, original signed area
    int min_x, max_x, min_y, max_y; // pixel bounds, clamped to the screen

    float4 screen[3];        // x, y, depth01, clip-space w
    float4 view_pos[3];
    float4 view_normal[3];
};

// Per-frame counters, written only by the GPU.
// Note: "small" is a macro in <windows.h> (rpcndr.h), so the
// fields carry a _count suffix
struct RasterCounts {
    unsigned int record_count;
    unsigned int small_count;
    unsigned int large_count;
    unsigned int overflow_count;
};

// Device buffers shared by the setup, rasterize and shade kernels
struct RasterWork {
    RasterTri* records = nullptr;
    unsigned int* small_list = nullptr;
    unsigned int* large_list = nullptr;
    RasterCounts* counts = nullptr;
    unsigned int capacity = 0;
};

struct CudaGridParams {
    Mat4 view_proj;
    Vec3 eye;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float fade_end;
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

// Zeroes the counters on the default stream (no CPU wait)
bool cuda_reset_raster_counts(RasterCounts* counts);

// One thread per triangle of one draw item: validates, culls
// off-screen triangles, writes a record and appends its index to the
// small or the large list
bool cuda_setup_triangles(
                const RasterWork& work,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices);

// Rasterizes every record in the small and large lists into the
// visibility buffer. Visibility id = record index + 1
bool cuda_rasterize_worklist(
                std::uint64_t* visibility,
                int width,
                const RasterWork& work,
                int sm_count);

bool cuda_draw_edges(
                std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices);

bool cuda_resolve_wire(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height);

// One pass over all pixels, shading the record that won each pixel
bool cuda_shade(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                const RasterWork& work,
                Vec3 up_view);

bool cuda_resolve_ids(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height);

bool cuda_draw_grid(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                CudaGridParams params);

} // namespace gfx
