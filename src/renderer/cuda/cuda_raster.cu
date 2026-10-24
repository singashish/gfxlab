#include "renderer/cuda/cuda_raster.h"

#include "core/log.h"
#include "core/bvh.h"
#include "core/brdf.h"
#include "core/light_sampling.h"
#include "core/bvh4.h"
#include "core/path_math.h"
#include "core/raster_math.h"
#include "renderer/cuda/cuda_api.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <climits>

namespace gfx {

namespace {

constexpr unsigned int kThreads = 256;
constexpr int kTile = 16;
constexpr float kMaxScreenCoordinate = 16384.0f;
constexpr long long kLargeTriangleArea = 1024;
constexpr unsigned int kLargeSlices = 32;
constexpr unsigned int kLargeThreads = 128;

// Clipping: the near plane plus four guard-band planes
constexpr int kClipPlanes = 5;

// Each plane can add at most one vertex to a convex polygon
constexpr int kMaxClipVertices = 3 + kClipPlanes;

// Guard band in NDC units: clipped vertices stay inside
// +/- kMaxScreenCoordinate pixels, with a 2x safety margin
float guard_band(int width, int height) {
    const int largest = width > height ? width : height;
    return 0.5f * (2.0f * kMaxScreenCoordinate /
                   static_cast<float>(largest) - 1.0f);
}

bool check_launch(const char* operation) {
    return cuda_ok(cudaGetLastError(), operation);
}

__device__ __forceinline__ std::size_t pixel_index(
    int x, int y, int width) {
    return static_cast<std::size_t>(y) *
               static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x);
}

__device__ __forceinline__ int clamp_int(
    int value, int low, int high) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ bool clip_test(
                float p,
                float q,
                float& t_enter,
                float& t_exit) {

    if (p == 0.0f) {
        return q >= 0.0f;
    }

    const float r = q / p;

    if (p < 0.0f) {
        if (r > t_exit) {
            return false;
        }
        t_enter = fmaxf(t_enter, r);
    } else {
        if (r < t_enter) {
            return false;
        }
        t_exit = fminf(t_exit, r);
    }

    return true;
}

// One source of truth for rasterization and shading
// sx/sy and order are normalized to positive area, while
// signed_area retains the original winding
__device__ __forceinline__ bool setup_triangle(
                const float4& a,
                const float4& b,
                const float4& c,
                RasterTriangleSetup& result) {

    const float4 points[3] = {a, b, c};

    // Safety net: after clipping these checks should never fail
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float4 p = points[i];

        if (!isfinite(p.x) || !isfinite(p.y) ||
            !isfinite(p.z) || !isfinite(p.w) ||
            p.w <= 0.0f ||
            fabsf(p.x) > kMaxScreenCoordinate ||
            fabsf(p.y) > kMaxScreenCoordinate) {
            return false;
        }
    }

    result = setup_raster_triangle(
        a.x, a.y,
        b.x, b.y,
        c.x, c.y);

    return result.valid;
}

// Perspective divide and viewport transform. Row 0 is the top of the
// screen. The single place where clip space becomes screen space
__device__ __forceinline__ float4 clip_to_screen(
                const float4& clip,
                int width,
                int height) {

    const float inv_w = 1.0f / clip.w;
    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;
    const float ndc_z = clip.z * inv_w;

    return make_float4(
        (ndc_x * 0.5f + 0.5f) * width,
        (1.0f - (ndc_y * 0.5f + 0.5f)) * height,
        ndc_z * 0.5f + 0.5f,
        clip.w);
}

// Signed distance to a clip plane: >= 0 means the point is kept.
// Plane 0 is the OpenGL near plane (z >= -w), which also forces w > 0.
// Planes 1-4 are the guard band |x| <= g*w and |y| <= g*w
__device__ __forceinline__ float plane_distance(
                const float4& c,
                int plane,
                float guard) {

    switch (plane) {
    case 0:  return c.z + c.w;
    case 1:  return guard * c.w + c.x;
    case 2:  return guard * c.w - c.x;
    case 3:  return guard * c.w + c.y;
    default: return guard * c.w - c.y;
    }
}

__device__ __forceinline__ float4 lerp4(
                const float4& a,
                const float4& b,
                float t) {

    return make_float4(
        a.x + t * (b.x - a.x),
        a.y + t * (b.y - a.y),
        a.z + t * (b.z - a.z),
        a.w + t * (b.w - a.w));
}

// A polygon corner during clipping. Every attribute is linear in clip
// space, so one t interpolates all of them correctly. That includes the
// texture coordinates in the w components of view_pos and view_normal
struct ClipVertex {
    float4 clip;
    float4 view_pos;
    float4 view_normal;
};

__device__ __forceinline__ ClipVertex lerp_vertex(
                const ClipVertex& a,
                const ClipVertex& b,
                float t) {

    ClipVertex v;
    v.clip = lerp4(a.clip, b.clip, t);
    v.view_pos = lerp4(a.view_pos, b.view_pos, t);
    v.view_normal = lerp4(a.view_normal, b.view_normal, t);
    return v;
}

__device__ __forceinline__ bool triangle_pixel_bounds(
                const RasterTriangleSetup& s,
                int width,
                int height,
                int& min_x,
                int& max_x,
                int& min_y,
                int& max_y) {

    const int min_sx = min(s.sx[0], min(s.sx[1], s.sx[2]));
    const int max_sx = max(s.sx[0], max(s.sx[1], s.sx[2]));
    const int min_sy = min(s.sy[0], min(s.sy[1], s.sy[2]));
    const int max_sy = max(s.sy[0], max(s.sy[1], s.sy[2]));

    // Entirely outside the screen: clamping alone would squash the
    // box onto the border and keep the triangle alive
    if (max_sx < 0 || max_sy < 0 ||
        min_sx > width * 16 || min_sy > height * 16) {
        return false;
    }

    min_x = clamp_int(
        static_cast<int>(floorf(min_sx / 16.0f)),
        0, width - 1);

    max_x = clamp_int(
        static_cast<int>(ceilf(max_sx / 16.0f)),
        0, width - 1);

    min_y = clamp_int(
        static_cast<int>(floorf(min_sy / 16.0f)),
        0, height - 1);

    max_y = clamp_int(
        static_cast<int>(ceilf(max_sy / 16.0f)),
        0, height - 1);

    return min_x <= max_x && min_y <= max_y;
}

__device__ __forceinline__ void rasterize_pixel(
                unsigned long long* visibility,
                int width,
                int x,
                int y,
                const RasterTriangleSetup& s,
                const float4* v,
                float inv_area,
                unsigned int id) {

    const int px = x * 16 + 8;
    const int py = y * 16 + 8;

    const long long e0 = edge_function(
        s.sx[1], s.sy[1], s.sx[2], s.sy[2], px, py);

    const long long e1 = edge_function(
        s.sx[2], s.sy[2], s.sx[0], s.sy[0], px, py);

    const long long e2 = edge_function(
        s.sx[0], s.sy[0], s.sx[1], s.sy[1], px, py);

    if (!raster_triangle_covers_sample(s, px, py)) {
        return;
    }

    const float b0 = static_cast<float>(e0) * inv_area;
    const float b1 = static_cast<float>(e1) * inv_area;
    const float b2 = static_cast<float>(e2) * inv_area;

    const float z =
        b0 * v[0].z +
        b1 * v[1].z +
        b2 * v[2].z;

    if (!isfinite(z) || z < 0.0f || z > 1.0f) {
        return;
    }

    const unsigned int depth_bits = __float_as_uint(z);

    const unsigned long long packed =
        (static_cast<unsigned long long>(depth_bits) << 32) |
        static_cast<unsigned long long>(id);

    atomicMin(
        &visibility[pixel_index(x, y, width)],
        packed);
}

__global__ void clear_kernel(
    uchar4* color, int width, int height, uchar4 value) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    color[pixel_index(x, y, width)] = value;
}

__global__ void clear_visibility_kernel(
    unsigned long long* visibility, std::size_t count) {

    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x +
        threadIdx.x;

    if (i < count) {
        visibility[i] = ~0ULL;
    }
}

__global__ void transform_vertices_kernel(
                const DeviceMesh mesh,
                Mat4 mvp,
                Mat4 model_view,
                Mat4 normal_to_view,
                int width,
                int height,
                ProjectedVertex* out) {

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= mesh.vertex_count) {
        return;
    }

    const Vec3 position = mesh.positions[i];

    const Vec3 view_position =
        transform_point(model_view, position);

    // The texture coordinates (0, 0 for a mesh without) ride in the w
    // components of the view position and normal; see ProjectedVertex
    const Vec2 uv = mesh.texcoords ? mesh.texcoords[i] : Vec2{0.0f, 0.0f};

    out[i].view_pos = make_float4(
        view_position.x,
        view_position.y,
        view_position.z,
        uv.x);

    if (mesh.normals) {
        const Vec3 view_normal =
            transform_direction(normal_to_view, mesh.normals[i]);

        out[i].view_normal = make_float4(
            view_normal.x,
            view_normal.y,
            view_normal.z,
            uv.y);
    } else {
        out[i].view_normal = make_float4(0, 0, 0, uv.y);
    }

    const Vec4 clip_v = mvp * make_vec4(position, 1.0f);
    const float4 clip =
        make_float4(clip_v.x, clip_v.y, clip_v.z, clip_v.w);

    // Always stored: the setup and edge kernels clip from it
    out[i].clip = clip;

    // A vertex behind the camera has no meaningful screen position.
    // Triangles using it go through clipping and never read it
    if (!isfinite(clip.w) || clip.w <= 0.0f) {
        out[i].screen = make_float4(0, 0, 0, clip.w);
        return;
    }

    out[i].screen = clip_to_screen(clip, width, height);
}

__global__ void draw_edges_kernel(
                unsigned long long* visibility,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices,
                float guard) {

    const unsigned int edge =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int edge_count =
        mesh.edge_index_count / 2;

    if (edge >= edge_count) {
        return;
    }

    const std::uint32_t i0 = mesh.edges[edge * 2];
    const std::uint32_t i1 = mesh.edges[edge * 2 + 1];

    const float4 ca = vertices[i0].clip;
    const float4 cb = vertices[i1].clip;

    if (!isfinite(ca.x) || !isfinite(ca.y) ||
        !isfinite(ca.z) || !isfinite(ca.w) ||
        !isfinite(cb.x) || !isfinite(cb.y) ||
        !isfinite(cb.z) || !isfinite(cb.w)) {
        return;
    }

    // Clip the segment in clip space against the near plane and the
    // guard band. t0..t1 is the part of the original segment that
    // survives every plane
    float t0 = 0.0f;
    float t1 = 1.0f;

    #pragma unroll
    for (int plane = 0; plane < kClipPlanes; ++plane) {
        const float da = plane_distance(ca, plane, guard);
        const float db = plane_distance(cb, plane, guard);

        if (da < 0.0f && db < 0.0f) {
            return;
        }

        if (da < 0.0f) {
            t0 = fmaxf(t0, da / (da - db));
        } else if (db < 0.0f) {
            t1 = fminf(t1, da / (da - db));
        }
    }

    if (t0 > t1) {
        return;
    }

    const float4 a = clip_to_screen(lerp4(ca, cb, t0), width, height);
    const float4 b = clip_to_screen(lerp4(ca, cb, t1), width, height);

    if (!isfinite(a.x) || !isfinite(a.y) || !isfinite(a.z) ||
        !isfinite(b.x) || !isfinite(b.y) || !isfinite(b.z)) {
        return;
    }

    const float dx = b.x - a.x;
    const float dy = b.y - a.y;

    if (!isfinite(dx) || !isfinite(dy)) {
        return;
    }

    // Liang-Barsky clipping against the viewport.
    float t_enter = 0.0f;
    float t_exit = 1.0f;

    if (!clip_test(-dx, a.x, t_enter, t_exit) ||
        !clip_test( dx, width - a.x, t_enter, t_exit) ||
        !clip_test(-dy, a.y, t_enter, t_exit) ||
        !clip_test( dy, height - a.y, t_enter, t_exit) ||
        t_enter > t_exit) {
        return;
    }

    // Interpolate the clipped endpoints, including depth.
    const float x0 = a.x + t_enter * dx;
    const float y0 = a.y + t_enter * dy;
    const float z0 = a.z + t_enter * (b.z - a.z);

    const float x1 = a.x + t_exit * dx;
    const float y1 = a.y + t_exit * dy;
    const float z1 = a.z + t_exit * (b.z - a.z);

    if (!isfinite(x0) || !isfinite(y0) ||
        !isfinite(z0) || !isfinite(x1) ||
        !isfinite(y1) || !isfinite(z1)) {
        return;
    }

    const float line_dx = x1 - x0;
    const float line_dy = y1 - y0;

    const float max_delta =
        fmaxf(fabsf(line_dx), fabsf(line_dy));

    // Even a zero-length edge must produce one sample.
    const int steps = static_cast<int>(ceilf(max_delta));

    for (int i = 0; i <= steps; ++i) {
        const float t = steps > 0
            ? static_cast<float>(i) / steps
            : 0.0f;

        const float x = x0 + t * line_dx;
        const float y = y0 + t * line_dy;
        const float z = z0 + t * (z1 - z0);

        if (!isfinite(x) || !isfinite(y) ||
            !isfinite(z) || z < 0.0f || z > 1.0f) {
            continue;
        }

        const int px = static_cast<int>(floorf(x));
        const int py = static_cast<int>(floorf(y));

        // Clipping to width/height can still produce a coordinate
        // equal to the dimension itself. Such a pixel is outside.
        if (px < 0 || px >= width ||
            py < 0 || py >= height) {
            continue;
        }

        const unsigned int depth_bits = __float_as_uint(z);

        constexpr unsigned int kWireId = 1;

        const unsigned long long packed =
            (static_cast<unsigned long long>(depth_bits) << 32) |
            static_cast<unsigned long long>(kWireId);

        atomicMin(
            &visibility[pixel_index(px, py, width)],
            packed);
    }
}

__global__ void resolve_wire_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height) {

    const int x =
        blockIdx.x * blockDim.x + threadIdx.x;
    const int y =
        blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel = pixel_index(x, y, width);

    if (visibility[pixel] == ~0ULL) {
        return;
    }

    color[pixel] = make_uchar4(
        217, 217, 217, 255);
}

// Appends one screen-space triangle to the worklist.
// screen/view_pos/view_normal are in the triangle's own vertex order;
// material is the triangle's index in the global material table
__device__ __forceinline__ void emit_triangle(
                const RasterWork& work,
                int width,
                int height,
                const float4 (&screen)[3],
                const float4 (&view_pos)[3],
                const float4 (&view_normal)[3],
                int material) {

    RasterTriangleSetup s{};

    if (!setup_triangle(screen[0], screen[1], screen[2], s)) {
        return;
    }

    int min_x, max_x, min_y, max_y;

    // Off-screen triangles are culled here, once
    if (!triangle_pixel_bounds(
            s, width, height,
            min_x, max_x, min_y, max_y)) {
        return;
    }

    const unsigned int idx =
        atomicAdd(&work.counts->record_count, 1u);

    if (idx >= work.capacity) {
        atomicAdd(&work.counts->overflow_count, 1u);
        return;
    }

    RasterTri& r = work.records[idx];
    r.setup = s;
    r.min_x = min_x;
    r.max_x = max_x;
    r.min_y = min_y;
    r.max_y = max_y;
    r.material = material;
    r.padding = 0;

    // Store the vertices in the setup's positive-area order
    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        r.screen[k] = screen[s.order[k]];
        r.view_pos[k] = view_pos[s.order[k]];
        r.view_normal[k] = view_normal[s.order[k]];
    }

    const long long area_px =
        static_cast<long long>(max_x - min_x + 1) *
        static_cast<long long>(max_y - min_y + 1);

    // Both lists have the same capacity as the records,
    // and each record is appended to exactly one list
    if (area_px > kLargeTriangleArea) {
        const unsigned int slot =
            atomicAdd(&work.counts->large_count, 1u);
        work.large_list[slot] = idx;
    } else {
        const unsigned int slot =
            atomicAdd(&work.counts->small_count, 1u);
        work.small_list[slot] = idx;
    }
}

__global__ void setup_triangles_kernel(
                RasterWork work,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices,
                float guard) {

    const unsigned int tri =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int triangle_count = mesh.index_count / 3;

    if (tri >= triangle_count) {
        return;
    }

    const ProjectedVertex* v[3] = {
        &vertices[mesh.indices[tri * 3 + 0]],
        &vertices[mesh.indices[tri * 3 + 1]],
        &vertices[mesh.indices[tri * 3 + 2]]
    };

    // Every piece of a clipped triangle keeps the triangle's material
    const int material = static_cast<int>(
        mesh.material_base + mesh.triangle_materials[tri]);

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        const float4 c = v[k]->clip;

        if (!isfinite(c.x) || !isfinite(c.y) ||
            !isfinite(c.z) || !isfinite(c.w)) {
            return;
        }
    }

    // Classify the three corners against every plane
    bool all_inside = true;

    #pragma unroll
    for (int plane = 0; plane < kClipPlanes; ++plane) {
        int outside = 0;

        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            if (plane_distance(v[k]->clip, plane, guard) < 0.0f) {
                ++outside;
            }
        }

        // Every corner behind the same plane: nothing survives
        if (outside == 3) {
            return;
        }

        if (outside > 0) {
            all_inside = false;
        }
    }

    // Fast path, almost every triangle: no clipping needed
    if (all_inside) {
        const float4 screen[3] = {
            v[0]->screen, v[1]->screen, v[2]->screen
        };
        const float4 view_pos[3] = {
            v[0]->view_pos, v[1]->view_pos, v[2]->view_pos
        };
        const float4 view_normal[3] = {
            v[0]->view_normal, v[1]->view_normal, v[2]->view_normal
        };

        emit_triangle(
            work, width, height, screen, view_pos, view_normal,
            material);
        return;
    }

    // Sutherland-Hodgman: clip the polygon against one plane at a
    // time, ping-ponging between two small local arrays
    ClipVertex poly_a[kMaxClipVertices];
    ClipVertex poly_b[kMaxClipVertices];

    ClipVertex* in = poly_a;
    ClipVertex* out = poly_b;
    int count = 3;

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        in[k].clip = v[k]->clip;
        in[k].view_pos = v[k]->view_pos;
        in[k].view_normal = v[k]->view_normal;
    }

    for (int plane = 0; plane < kClipPlanes; ++plane) {
        int out_count = 0;

        for (int i = 0; i < count; ++i) {
            const ClipVertex& a = in[i];
            const ClipVertex& b = in[(i + 1) % count];

            const float da = plane_distance(a.clip, plane, guard);
            const float db = plane_distance(b.clip, plane, guard);

            // Keep corners on the inside
            if (da >= 0.0f && out_count < kMaxClipVertices) {
                out[out_count++] = a;
            }

            // The edge crosses the plane: add the crossing point
            if ((da >= 0.0f) != (db >= 0.0f) &&
                out_count < kMaxClipVertices) {
                const float t = da / (da - db);
                out[out_count++] = lerp_vertex(a, b, t);
            }
        }

        ClipVertex* swap = in;
        in = out;
        out = swap;
        count = out_count;

        if (count < 3) {
            return;
        }
    }

    // Fan-triangulate the clipped polygon: (0, i, i + 1).
    // Clipping keeps the winding, so every piece faces the same way
    // as the original triangle
    const float4 screen0 = clip_to_screen(in[0].clip, width, height);

    for (int i = 1; i + 1 < count; ++i) {
        const float4 screen[3] = {
            screen0,
            clip_to_screen(in[i].clip, width, height),
            clip_to_screen(in[i + 1].clip, width, height)
        };
        const float4 view_pos[3] = {
            in[0].view_pos, in[i].view_pos, in[i + 1].view_pos
        };
        const float4 view_normal[3] = {
            in[0].view_normal, in[i].view_normal, in[i + 1].view_normal
        };

        emit_triangle(
            work, width, height, screen, view_pos, view_normal,
            material);
    }
}

__device__ __forceinline__ void load_record_screen(
                const RasterTri& r,
                float4 (&v)[3],
                float& inv_area) {

    v[0] = r.screen[0];
    v[1] = r.screen[1];
    v[2] = r.screen[2];

    inv_area = 1.0f / static_cast<float>(r.setup.area);
}

// Fixed-size grid: each thread walks the small list with a grid stride.
// The count is read from device memory; the CPU never sees it
__global__ void rasterize_small_kernel(
                unsigned long long* visibility,
                int width,
                RasterWork work) {

    const unsigned int count = work.counts->small_count;
    const unsigned int stride = gridDim.x * blockDim.x;

    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < count;
         i += stride) {

        const unsigned int idx = work.small_list[i];
        const RasterTri& r = work.records[idx];

        float4 v[3];
        float inv_area;
        load_record_screen(r, v, inv_area);

        for (int y = r.min_y; y <= r.max_y; ++y) {
            for (int x = r.min_x; x <= r.max_x; ++x) {
                rasterize_pixel(
                    visibility, width, x, y, r.setup, v,
                    inv_area, idx + 1);
            }
        }
    }
}

// Fixed-size grid: blocks walk the (large triangle, row slice) pairs.
// Threads of a block share the columns of each row
__global__ void rasterize_large_kernel(
                unsigned long long* visibility,
                int width,
                RasterWork work) {

    const unsigned int count = work.counts->large_count;

    const unsigned long long items =
        static_cast<unsigned long long>(count) * kLargeSlices;

    for (unsigned long long item = blockIdx.x;
         item < items;
         item += gridDim.x) {

        const unsigned int list_slot =
            static_cast<unsigned int>(item / kLargeSlices);
        const int slice =
            static_cast<int>(item % kLargeSlices);

        const unsigned int idx = work.large_list[list_slot];
        const RasterTri& r = work.records[idx];

        float4 v[3];
        float inv_area;
        load_record_screen(r, v, inv_area);

        for (int y = r.min_y + slice;
             y <= r.max_y;
             y += static_cast<int>(kLargeSlices)) {

            for (int x = r.min_x + static_cast<int>(threadIdx.x);
                 x <= r.max_x;
                 x += static_cast<int>(blockDim.x)) {

                rasterize_pixel(
                    visibility, width, x, y, r.setup, v,
                    inv_area, idx + 1);
            }
        }
    }
}

__device__ __forceinline__ unsigned char to_byte(float c) {
    int value = __float2int_rn(c * 255.0f);
    value = max(0, min(255, value));
    return static_cast<unsigned char>(value);
}

// The viewport lighting, shared by the rasterizer and the ray tracer so
// the two paths cannot drift apart. n is the view-space normal, already
// flipped towards the viewer for back faces; view_pos is the shaded
// point in view space. Ported from the OpenGL mesh shader
// key_dir is the direction towards the key light in view space: the
// camera-attached light for the rasterizer, the sun for the ray tracer.
// key_visibility is 1 when the key light reaches the point and 0 when a
// shadow ray found something in the way. Only the key light's diffuse
// and highlight are removed; the ambient fill stays, so shadows are
// dark but never black. ambient_scale is the ambient occlusion factor
// (1 = fully open, 0 = fully enclosed). The rasterizer passes 1 for both.
// base is the surface colour the light is multiplied with: the material
// colour times the texel, or the grey of unlit_base_color(). The
// highlight is white and is not tinted by it, as in the OpenGL shader
__device__ __forceinline__ uchar4 lit_color(
                Vec3 n,
                Vec3 view_pos,
                Vec3 up_view,
                Vec3 key_dir,
                float key_visibility,
                float ambient_scale,
                Vec3 base) {

    n = normalize(n);
    const Vec3 v = normalize(-view_pos);

    const float diff =
        fmaxf(dot(n, key_dir), 0.0f) * key_visibility;

    const float hemi = dot(n, up_view) * 0.5f + 0.5f;

    const Vec3 ambient_low{0.10f, 0.10f, 0.12f};
    const Vec3 ambient_high{0.30f, 0.32f, 0.36f};

    const Vec3 ambient = Vec3{
        ambient_low.x + (ambient_high.x - ambient_low.x) * hemi,
        ambient_low.y + (ambient_high.y - ambient_low.y) * hemi,
        ambient_low.z + (ambient_high.z - ambient_low.z) * hemi
    } * ambient_scale;

    const Vec3 h = normalize(key_dir + v);

    const float spec = diff > 0.0f
        ? powf(fmaxf(dot(n, h), 0.0f), 32.0f) * 0.25f
        : 0.0f;

    const float r = fminf(
        base.x * (ambient.x + 0.75f * diff) + spec, 1.0f);
    const float g = fminf(
        base.y * (ambient.y + 0.75f * diff) + spec, 1.0f);
    const float b = fminf(
        base.z * (ambient.z + 0.75f * diff) + spec, 1.0f);

    return make_uchar4(to_byte(r), to_byte(g), to_byte(b), 255);
}

// The perspective-correct weights of a record's three corners at the
// sample position (px, py), given in 1/16 pixel units like the rasterizer's
// sample points. Returns false when they cannot be computed (the position
// is at or beyond the horizon of the triangle's plane).
//
// The position does not have to be inside the triangle: the edge functions
// are plain linear functions of the position, so outside they simply keep
// going and give weights below 0 or above 1. Shading needs that to look at
// the neighbouring pixels (see raster_base_color)
__device__ __forceinline__ bool perspective_weights(
                const RasterTri& rec,
                int px,
                int py,
                float inv_area,
                float& l0,
                float& l1,
                float& l2) {

    const RasterTriangleSetup& s = rec.setup;

    const long long e0 = edge_function(
        s.sx[1], s.sy[1], s.sx[2], s.sy[2], px, py);
    const long long e1 = edge_function(
        s.sx[2], s.sy[2], s.sx[0], s.sy[0], px, py);
    const long long e2 = edge_function(
        s.sx[0], s.sy[0], s.sx[1], s.sy[1], px, py);

    // Screen-space barycentrics: linear in the screen position, but
    // attributes are linear in the 3D position, not in the screen position
    const float b0 = static_cast<float>(e0) * inv_area;
    const float b1 = static_cast<float>(e1) * inv_area;
    const float b2 = static_cast<float>(e2) * inv_area;

    // Perspective-correct interpolation: weight each corner by 1 / w
    // (w = view depth) and renormalize. That undoes the squeezing of far
    // parts of the triangle on screen
    const float q0 = b0 / rec.screen[0].w;
    const float q1 = b1 / rec.screen[1].w;
    const float q2 = b2 / rec.screen[2].w;
    const float qsum = q0 + q1 + q2;

    // Every w is positive, so qsum is positive inside the triangle;
    // outside it can reach 0 where the surface runs off to the horizon
    if (!isfinite(qsum) || !(qsum > 1.0e-20f)) {
        return false;
    }

    l0 = q0 / qsum;
    l1 = q1 / qsum;
    l2 = q2 / qsum;
    return true;
}

// Texture coordinates of a record at weights (l0, l1, l2); they are
// stored in the w components of the corners' view_pos and view_normal
__device__ __forceinline__ float2 interpolate_uv(
                const RasterTri& rec,
                float l0,
                float l1,
                float l2) {

    return make_float2(
        l0 * rec.view_pos[0].w +
            l1 * rec.view_pos[1].w + l2 * rec.view_pos[2].w,
        l0 * rec.view_normal[0].w +
            l1 * rec.view_normal[1].w + l2 * rec.view_normal[2].w);
}

// Perspective-correct texture coordinates of a record at the sample
// position (px, py), 1/16 pixel units. False where there are none
__device__ __forceinline__ bool uv_at_sample(
                const RasterTri& rec,
                int px,
                int py,
                float inv_area,
                float2& uv) {

    float l0 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;

    if (!perspective_weights(rec, px, py, inv_area, l0, l1, l2)) {
        return false;
    }

    uv = interpolate_uv(rec, l0, l1, l2);
    return true;
}

// The base colour of a rasterized pixel: material colour times the
// base-colour texture, sampled trilinearly. uv is the pixel's texture
// coordinate; (px, py) its centre, in 1/16 pixel units
__device__ __forceinline__ Vec3 raster_base_color(
                const RasterTri& rec,
                const MaterialTables& tables,
                float2 uv,
                int px,
                int py,
                float inv_area) {

    const CudaMaterial material = tables.materials[rec.material];
    const TextureView& texture = tables.textures[material.texture];

    // Which mip level fits this pixel. A texture with one level (the white
    // texture of untextured materials) needs no derivatives at all
    float lod = 0.0f;

    if (texture.levels > 1) {
        // How fast the coordinates change per pixel. The hardware would
        // compare the values of neighbouring fragments of a 2x2 quad
        // (dFdx, dFdy), but a deferred shader has no such quads: every
        // thread shades one pixel on its own, from the visibility buffer.
        // The triangle is known though, so the thread evaluates the same
        // interpolation one pixel to the right and one pixel down by itself
        // (16 sub-pixel units = 1 pixel) and takes the differences
        float2 uv_right = uv;
        float2 uv_down = uv;

        const bool ok =
            uv_at_sample(rec, px + 16, py, inv_area, uv_right) &&
            uv_at_sample(rec, px, py + 16, inv_area, uv_down);

        if (ok) {
            // The texture is sampled at v_image = 1 - v, so that is the
            // coordinate whose change counts (only its size matters)
            lod = texture_lod(
                texture,
                uv_right.x - uv.x, -(uv_right.y - uv.y),
                uv_down.x - uv.x, -(uv_down.y - uv.y));
        } else {
            // A neighbour is past the horizon of the surface: the texture
            // is squeezed infinitely there, so use the blurriest level
            lod = static_cast<float>(texture.levels - 1);
        }
    }

    const Vec3 texel = sample_trilinear(texture, uv.x, 1.0f - uv.y, lod);

    return Vec3{material.base_color.x * texel.x,
                material.base_color.y * texel.y,
                material.base_color.z * texel.z};
}

__global__ void shade_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height,
                RasterWork work,
                MaterialTables tables,
                Vec3 up_view) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel = pixel_index(x, y, width);
    const unsigned long long packed = visibility[pixel];

    if (packed == ~0ULL) {
        return;
    }

    const std::uint32_t id =
        static_cast<std::uint32_t>(packed & 0xFFFFFFFFULL);

    if (id == 0) {
        return;
    }

    const std::uint32_t idx = id - 1;

    // Records beyond the capacity were never written
    const unsigned int written = min(
        work.counts->record_count, work.capacity);

    if (idx >= written) {
        return;
    }

    // Named rec, not r: the lighting code below uses r for red
    const RasterTri& rec = work.records[idx];
    const RasterTriangleSetup& s = rec.setup;

    const int px = x * 16 + 8;
    const int py = y * 16 + 8;

    if (!raster_triangle_covers_sample(s, px, py)) {
        return;
    }

    const float inv_area = 1.0f / static_cast<float>(s.area);

    // Perspective-correct weights of the three corners at the pixel centre
    float l0 = 0.0f;
    float l1 = 0.0f;
    float l2 = 0.0f;

    if (!perspective_weights(rec, px, py, inv_area, l0, l1, l2)) {
        return;
    }

    Vec3 view_pos = {
        l0 * rec.view_pos[0].x +
            l1 * rec.view_pos[1].x + l2 * rec.view_pos[2].x,
        l0 * rec.view_pos[0].y +
            l1 * rec.view_pos[1].y + l2 * rec.view_pos[2].y,
        l0 * rec.view_pos[0].z +
            l1 * rec.view_pos[1].z + l2 * rec.view_pos[2].z
    };

    Vec3 n = {
        l0 * rec.view_normal[0].x +
            l1 * rec.view_normal[1].x + l2 * rec.view_normal[2].x,
        l0 * rec.view_normal[0].y +
            l1 * rec.view_normal[1].y + l2 * rec.view_normal[2].y,
        l0 * rec.view_normal[0].z +
            l1 * rec.view_normal[1].z + l2 * rec.view_normal[2].z
    };

    // Screen coordinates are y-down. CCW front faces therefore
    // have a negative original signed area
    const bool front_facing = s.signed_area < 0;

    if (!front_facing) {
        n = -n;
    }

    // Surface colour: material colour times texture, or the plain grey
    Vec3 base = unlit_base_color();

    if (tables.show_materials) {
        base = raster_base_color(
            rec, tables, interpolate_uv(rec, l0, l1, l2),
            px, py, inv_area);
    }

    color[pixel] = lit_color(
        n, view_pos, up_view, key_light_view_dir(), 1.0f, 1.0f, base);
}

// Does anything in the scene block the ray origin + t * dir for
// 0 < t < t_max? Used for shadow and AO rays, so any hit is enough.
// dir is unit length in world space, so t_max is a world distance; the
// object-space direction is not renormalized, so t means the same there
__device__ __forceinline__ bool scene_occluded(
                const RtScene& scene,
                Vec3 origin,
                Vec3 dir,
                float t_max) {

    for (int i = 0; i < scene.count; ++i) {
        const RtInstance& inst = scene.items[i];

        const Vec3 local_origin =
            transform_point(inst.world_to_object, origin);
        const Vec3 local_dir =
            transform_direction(inst.world_to_object, dir);

        // Both trees hold the same triangles in the same order, so
        // either answers the same question
        const bool blocked = scene.use_bvh4
            ? bvh4_occluded(inst.mesh.bvh4_nodes,
                            inst.mesh.bvh_tri_indices,
                            inst.mesh.positions,
                            inst.mesh.indices,
                            local_origin, local_dir, 0.0f, t_max)
            : bvh_occluded(inst.mesh.bvh_nodes,
                           inst.mesh.bvh_tri_indices,
                           inst.mesh.positions,
                           inst.mesh.indices,
                           local_origin, local_dir, 0.0f, t_max);

        if (blocked) {
            return true;
        }
    }

    return false;
}

// Distance to move a shadow ray's start off the surface, so it does not
// hit the triangle it starts on ("shadow acne"). Scaled with the size of
// the coordinates because float precision is relative
__device__ __forceinline__ float shadow_offset(Vec3 p) {
    const float largest =
        fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fabsf(p.z)));

    return 1.0e-4f * fmaxf(1.0f, largest);
}

// PCG hash: turns any 32-bit number into a well-scrambled one. Chained,
// it gives each pixel and sample its own repeatable random sequence
__device__ __forceinline__ unsigned int pcg_hash(unsigned int v) {
    const unsigned int state = v * 747796405u + 2891336453u;
    const unsigned int word =
        ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Next random number in [0, 1) from the seed, advancing the seed
__device__ __forceinline__ float random01(unsigned int& seed) {
    seed = pcg_hash(seed);
    return static_cast<float>(seed >> 8) * (1.0f / 16777216.0f);
}

// ---------------------------------------------------------------------
// Ray tracer building blocks, shared by both ways of running it:
//  - the "megakernel": one thread does everything for one pixel
//  - the "wavefront": the work is split into passes, and each shadow or
//    AO ray gets its own thread (see cuda_ray_trace below)
// ---------------------------------------------------------------------

// Camera ray direction through the point (px, py) of the image, in
// pixels: (0, 0) is the top-left corner of the first pixel and
// (x + 0.5, y + 0.5) the centre of pixel (x, y). Not unit length: its
// view-space z is exactly -1, so t is the view depth (for any px, py)
__device__ __forceinline__ Vec3 camera_ray_dir_at(
                const RtScene& scene,
                float px,
                float py,
                int width,
                int height) {

    const float ndc_x = 2.0f * px / width - 1.0f;
    const float ndc_y = 1.0f - 2.0f * py / height;

    return scene.forward + ndc_x * scene.right + ndc_y * scene.up;
}

// The ray through the centre of pixel (x, y)
__device__ __forceinline__ Vec3 camera_ray_dir(
                const RtScene& scene,
                int x,
                int y,
                int width,
                int height) {

    return camera_ray_dir_at(scene, x + 0.5f, y + 0.5f, width, height);
}

// The point a ray hit, with everything shading needs, in world space.
// Both normals are turned towards the side the camera sees
struct SurfacePoint {
    float t;         // distance along the ray (view depth for camera rays)
    Vec3 position;
    Vec3 normal;     // interpolated shading normal (not unit length)
    Vec3 geometric;  // the triangle's own normal (unit length)
    Vec3 base;       // surface colour: material colour times texel

    // What the path tracer needs to look up the rest of the material
    // (see resolve_surface_material). Both are only filled while
    // materials are shown; with them off nothing reads them. They are
    // cheap facts the hit has found out anyway, so recording them costs
    // the ray tracer nothing; the expensive part (more texture reads) is
    // left to the path tracer
    std::uint32_t material;  // index into the renderer-wide table
    Vec2 uv;                 // texture coordinates at the hit (v up)

    // Which instance (index into RtScene::items) and which triangle of
    // its mesh was hit. The path tracer needs them to work out how likely
    // the light sampling would have been to pick this very point of a
    // lamp (multiple importance sampling). Two plain stores
    std::uint32_t instance;
    std::uint32_t triangle;
};

// The texture coordinates at a hit on the triangle with corners
// (i0, i1, i2) and barycentric weights (w0, w1, w2). A mesh without
// texture coordinates uses uv (0, 0)
__device__ __forceinline__ Vec2 hit_uv(
                const DeviceMesh& mesh,
                std::uint32_t i0,
                std::uint32_t i1,
                std::uint32_t i2,
                float w0,
                float w1,
                float w2) {

    Vec2 uv{0.0f, 0.0f};

    if (mesh.texcoords) {
        uv = mesh.texcoords[i0] * w0 +
             mesh.texcoords[i1] * w1 +
             mesh.texcoords[i2] * w2;
    }

    return uv;
}

// The surface colour at a ray hit: the material's colour times its
// base-colour texture, at the texture coordinates uv. (The caller uses
// the plain grey instead when materials are off.)
//
// Camera rays have no pixel footprint to measure, so the texture is
// sampled at level 0 only (bilinear, no mip levels). The price is some
// shimmer on finely detailed textures far away or at grazing angles,
// which the rasterizer avoids with trilinear filtering. Mip levels for
// rays would need ray differentials, which is a step for later
__device__ __forceinline__ Vec3 hit_base_color(
                const MaterialTables& tables,
                const CudaMaterial& material,
                Vec2 uv) {

    // Image rows run top to bottom, v points up: sample at 1 - v
    const Vec3 texel = sample_bilinear(
        tables.textures[material.texture], 0, uv.x, 1.0f - uv.y);

    return Vec3{material.base_color.x * texel.x,
                material.base_color.y * texel.y,
                material.base_color.z * texel.z};
}

// Closest hit of the ray origin + t * dir with t_min < t < t_max over
// every instance. Camera rays start at the eye and use the near and far
// plane as their range; the rays of a path tracer start at an offset
// surface point (see shadow_offset), with t_min 0 and a huge t_max.
// The direction is moved into object space without normalizing it, so t
// means the same distance in both spaces and hits can be compared
// directly. Fills point on a hit
__device__ __forceinline__ bool closest_surface(
                const RtScene& scene,
                Vec3 origin,
                Vec3 dir,
                float t_min,
                float t_max,
                SurfacePoint& point) {

    float best_t = t_max;
    int best_item = -1;
    BvhHit best{};

    for (int i = 0; i < scene.count; ++i) {
        const RtInstance& inst = scene.items[i];

        BvhHit hit;

        const Vec3 local_origin =
            transform_point(inst.world_to_object, origin);
        const Vec3 local_dir =
            transform_direction(inst.world_to_object, dir);

        const bool found = scene.use_bvh4
            ? bvh4_trace(inst.mesh.bvh4_nodes,
                         inst.mesh.bvh_tri_indices,
                         inst.mesh.positions,
                         inst.mesh.indices,
                         local_origin, local_dir,
                         t_min, best_t, hit)
            : bvh_trace(inst.mesh.bvh_nodes,
                        inst.mesh.bvh_tri_indices,
                        inst.mesh.positions,
                        inst.mesh.indices,
                        local_origin, local_dir,
                        t_min, best_t, hit);

        if (found) {
            best_t = hit.t;
            best = hit;
            best_item = i;
        }
    }

    if (best_item < 0) {
        return false;
    }

    const RtInstance& inst = scene.items[best_item];
    const DeviceMesh& mesh = inst.mesh;

    const std::uint32_t i0 = mesh.indices[best.triangle * 3 + 0];
    const std::uint32_t i1 = mesh.indices[best.triangle * 3 + 1];
    const std::uint32_t i2 = mesh.indices[best.triangle * 3 + 2];

    const Vec3 p0 = mesh.positions[i0];
    const Vec3 p1 = mesh.positions[i1];
    const Vec3 p2 = mesh.positions[i2];

    // Geometric normal in object space, from the triangle's winding
    const Vec3 geometric = cross(p1 - p0, p2 - p0);

    // Interpolated vertex normal (barycentrics u, v from the hit)
    const float w0 = 1.0f - best.u - best.v;

    Vec3 n_object = geometric;

    if (mesh.normals) {
        n_object = mesh.normals[i0] * w0 +
                   mesh.normals[i1] * best.u +
                   mesh.normals[i2] * best.v;
    }

    // Front-facing when the ray arrives against the counter-clockwise
    // normal. The sign of this dot product is the same in object and
    // world space, so the object-space direction can be used
    const Vec3 local_dir =
        transform_direction(inst.world_to_object, dir);
    const bool front_facing = dot(geometric, local_dir) < 0.0f;

    point.t = best_t;
    point.instance = static_cast<std::uint32_t>(best_item);
    point.triangle = best.triangle;
    point.position = origin + best_t * dir;
    point.normal = transform_direction(inst.normal_to_world, n_object);
    point.geometric = normalize(
        transform_direction(inst.normal_to_world, geometric));

    if (!front_facing) {
        point.normal = -point.normal;
        point.geometric = -point.geometric;
    }

    if (scene.surfaces.show_materials) {
        point.material =
            mesh.material_base + mesh.triangle_materials[best.triangle];
        point.uv = hit_uv(mesh, i0, i1, i2, w0, best.u, best.v);
        point.base = hit_base_color(
            scene.surfaces, scene.surfaces.materials[point.material],
            point.uv);
    } else {
        point.material = 0;
        point.uv = Vec2{0.0f, 0.0f};
        point.base = unlit_base_color();
    }

    return true;
}

// Does the sun reach this point? Returns true when a shadow ray was
// needed (the surface faces the sun) and sets blocked
__device__ __forceinline__ bool trace_shadow(
                const RtScene& scene,
                const SurfacePoint& point,
                bool& blocked) {

    blocked = false;

    if (!scene.shadows ||
        dot(normalize(point.normal), scene.sun_world) <= 0.0f) {
        return false;  // surfaces facing away are unlit by the sun anyway
    }

    const Vec3 origin =
        point.position + point.geometric * shadow_offset(point.position);

    blocked = scene_occluded(scene, origin, scene.sun_world, 1.0e30f);
    return true;
}

// One ambient occlusion ray for the given pixel and sample number.
// Returns true when a ray was traced and sets escaped when it got away
// without hitting anything within the AO radius
__device__ __forceinline__ bool trace_ao_ray(
                const RtScene& scene,
                const SurfacePoint& point,
                unsigned int pixel,
                unsigned int sample,
                bool& escaped) {

    escaped = false;

    unsigned int seed = pcg_hash(pixel ^ pcg_hash(sample));

    const float u1 = random01(seed);
    const float u2 = random01(seed);

    const Vec3 d = cosine_hemisphere(normalize(point.normal), u1, u2);

    // A direction that dips below the actual triangle (possible where
    // smooth normals bend away from it) counts as blocked
    if (dot(d, point.geometric) <= 0.0f) {
        return false;
    }

    const Vec3 origin =
        point.position + point.geometric * shadow_offset(point.position);

    escaped = !scene_occluded(scene, origin, d, scene.ao_radius);
    return true;
}

// The visibility buffer entry (depth in the high 32 bits, id 1 in the
// low ones) for a ray-traced surface point. Same depth the rasterizer
// would have written, so the grid's depth test treats all paths alike
__device__ __forceinline__ unsigned long long visibility_entry(
                const RtScene& scene,
                Vec3 position) {

    const Vec4 clip =
        scene.view_proj * make_vec4(position, 1.0f);

    float depth01 = (clip.z / clip.w) * 0.5f + 0.5f;
    depth01 = fminf(fmaxf(depth01, 0.0f), 1.0f);

    return (static_cast<unsigned long long>(__float_as_uint(depth01))
                << 32) |
           1ULL;
}

// Combines the shadow and AO results into the pixel's colour and depth.
// new_escaped is how many of this frame's AO rays got away
__device__ __forceinline__ void finish_pixel(
                uchar4* color,
                unsigned long long* visibility,
                unsigned int pixel,
                const RtScene& scene,
                const SurfacePoint& point,
                bool sun_blocked,
                unsigned int new_escaped) {

    float ao = 1.0f;

    if (scene.ao) {
        float escaped = scene.ao_samples_before > 0
            ? scene.ao_accum[pixel]
            : 0.0f;

        escaped += static_cast<float>(new_escaped);

        const unsigned int total =
            scene.ao_samples_before +
            static_cast<unsigned int>(scene.ao_rays);

        if (scene.ao_rays > 0) {
            scene.ao_accum[pixel] = escaped;
        }

        ao = total > 0 ? escaped / static_cast<float>(total) : 1.0f;
    }

    const Vec3 n_view = transform_direction(scene.view, point.normal);
    const Vec3 view_pos = transform_point(scene.view, point.position);

    color[pixel] = lit_color(
        n_view, view_pos, scene.up_view, scene.sun_view,
        sun_blocked ? 0.0f : 1.0f, ao, point.base);

    visibility[pixel] = visibility_entry(scene, point.position);
}

// Adds a thread's ray counts to the frame's counters: first per block
// in shared memory, then one global atomic per counter per block.
// Every thread of the block must call it (it contains barriers)
__device__ __forceinline__ void add_ray_counts(
                unsigned long long* ray_counts,
                unsigned int primary,
                unsigned int shadow,
                unsigned int ao) {

    __shared__ unsigned int block_counts[3];

    const bool first_thread = threadIdx.x == 0 && threadIdx.y == 0;

    if (first_thread) {
        block_counts[0] = 0;
        block_counts[1] = 0;
        block_counts[2] = 0;
    }

    __syncthreads();

    if (ray_counts != nullptr) {
        if (primary) atomicAdd(&block_counts[0], primary);
        if (shadow)  atomicAdd(&block_counts[1], shadow);
        if (ao)      atomicAdd(&block_counts[2], ao);
    }

    __syncthreads();

    if (first_thread && ray_counts != nullptr) {
        atomicAdd(&ray_counts[0],
                  static_cast<unsigned long long>(block_counts[0]));
        atomicAdd(&ray_counts[1],
                  static_cast<unsigned long long>(block_counts[1]));
        atomicAdd(&ray_counts[2],
                  static_cast<unsigned long long>(block_counts[2]));
    }
}

// ---------------------------------------------------------------------
// Megakernel: one thread per pixel does camera ray, shadow ray and all
// AO rays in a row
// ---------------------------------------------------------------------

__global__ void ray_trace_kernel(
                uchar4* color,
                unsigned long long* visibility,
                int width,
                int height,
                RtScene scene) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int primary = 0;
    unsigned int shadow = 0;
    unsigned int ao = 0;

    // No early return: every thread must reach add_ray_counts
    if (x < width && y < height) {
        const unsigned int pixel =
            static_cast<unsigned int>(pixel_index(x, y, width));

        primary = 1;

        SurfacePoint point;

        if (closest_surface(
                scene, scene.eye,
                camera_ray_dir(scene, x, y, width, height),
                scene.t_min, scene.t_max, point)) {
            bool blocked = false;
            shadow = trace_shadow(scene, point, blocked) ? 1u : 0u;

            unsigned int escaped = 0;

            for (int i = 0; i < scene.ao_rays && scene.ao; ++i) {
                bool got_away = false;

                if (trace_ao_ray(
                        scene, point, pixel,
                        scene.ao_samples_before +
                            static_cast<unsigned int>(i),
                        got_away)) {
                    ++ao;
                    escaped += got_away ? 1u : 0u;
                }
            }

            finish_pixel(color, visibility, pixel, scene, point,
                         blocked, escaped);
        }
    }

    add_ray_counts(scene.ray_counts, primary, shadow, ao);
}

// ---------------------------------------------------------------------
// Wavefront: three passes connected by a list of hits
//  1. camera rays: one thread per pixel; hits are appended to the list
//  2. shadow and AO rays: one thread per RAY, over the hit list
//  3. finish: one thread per hit combines the results
// ---------------------------------------------------------------------

__global__ void rt_camera_kernel(
                int width,
                int height,
                RtScene scene,
                RtWavefront wave) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int primary = 0;

    if (x < width && y < height) {
        primary = 1;

        SurfacePoint point;

        if (closest_surface(
                scene, scene.eye,
                camera_ray_dir(scene, x, y, width, height),
                scene.t_min, scene.t_max, point)) {
            const unsigned int index =
                atomicAdd(&wave.counts->hit_count, 1u);

            if (index < wave.capacity) {
                RtHit& hit = wave.hits[index];
                hit.pixel =
                    static_cast<unsigned int>(pixel_index(x, y, width));
                hit.position = point.position;
                hit.normal = point.normal;
                hit.geometric = point.geometric;
                hit.base = point.base;

                // Results start empty; pass 2 fills them in
                wave.shadow_blocked[index] = 0;
                wave.ao_escaped[index] = 0;
            }
        }
    }

    add_ray_counts(scene.ray_counts, primary, 0, 0);
}

// Work item i belongs to hit i / rays_per_hit. Slot 0 is the shadow
// ray, slots 1.. are AO rays. Fixed grid, grid-stride loop: the number
// of hits is only known on the GPU
__global__ void rt_secondary_kernel(RtScene scene, RtWavefront wave) {
    const unsigned int hits =
        min(wave.counts->hit_count, wave.capacity);

    const unsigned int rays_per_hit =
        1u + static_cast<unsigned int>(scene.ao ? scene.ao_rays : 0);

    const unsigned long long total =
        static_cast<unsigned long long>(hits) * rays_per_hit;

    const unsigned long long stride =
        static_cast<unsigned long long>(gridDim.x) * blockDim.x;

    unsigned int shadow = 0;
    unsigned int ao = 0;

    for (unsigned long long item =
             static_cast<unsigned long long>(blockIdx.x) * blockDim.x +
             threadIdx.x;
         item < total;
         item += stride) {

        const unsigned int index =
            static_cast<unsigned int>(item / rays_per_hit);
        const unsigned int slot =
            static_cast<unsigned int>(item % rays_per_hit);

        const RtHit& hit = wave.hits[index];

        SurfacePoint point;
        point.position = hit.position;
        point.normal = hit.normal;
        point.geometric = hit.geometric;

        if (slot == 0) {
            bool blocked = false;

            if (trace_shadow(scene, point, blocked)) {
                ++shadow;
                wave.shadow_blocked[index] = blocked ? 1u : 0u;
            }
        } else {
            bool escaped = false;

            if (trace_ao_ray(scene, point, hit.pixel,
                             scene.ao_samples_before + (slot - 1),
                             escaped)) {
                ++ao;

                if (escaped) {
                    atomicAdd(&wave.ao_escaped[index], 1u);
                }
            }
        }
    }

    add_ray_counts(scene.ray_counts, 0, shadow, ao);
}

__global__ void rt_finish_kernel(
                uchar4* color,
                unsigned long long* visibility,
                RtScene scene,
                RtWavefront wave) {

    const unsigned int hits =
        min(wave.counts->hit_count, wave.capacity);

    for (unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
         index < hits;
         index += gridDim.x * blockDim.x) {

        const RtHit& hit = wave.hits[index];

        SurfacePoint point;
        point.position = hit.position;
        point.normal = hit.normal;
        point.geometric = hit.geometric;
        point.base = hit.base;

        finish_pixel(color, visibility, hit.pixel, scene, point,
                     wave.shadow_blocked[index] != 0,
                     wave.ao_escaped[index]);
    }
}

// ---------------------------------------------------------------------
// Path tracer ("Rendered" mode). The maths and the ideas (linear light,
// Monte Carlo, cosine sampling, next-event estimation, Russian roulette)
// are explained in core/path_math.h; the constants (bounces, sun, sky)
// are defined there. Surfaces are a metal/dielectric mix: a Lambert
// diffuse layer and a GGX specular lobe (the maths and the reasons are in
// core/brdf.h), plus emission.
//
// Colours: the stored colours (material colour times texel, the values
// the viewport shows) are display (sRGB) values. The path tracer decodes
// them to linear light for the albedo, one rule for textures and
// factors alike, and encodes only the final picture.
//
// The sky and the sun are lights, not scenery: bounce rays that miss
// collect the sky, surfaces get the sun through a shadow ray, but a ray
// that leaves the camera and misses everything sees neither. It gets the
// viewport background (PtParams::background), which goes through the same
// accumulation and tone mapping as any radiance, so the edge pixels of an
// object blend with the background correctly.
// ---------------------------------------------------------------------

// The material of a path-traced surface point, in the units the path
// tracer computes in (linear light).
struct PathMaterial {
    Vec3 albedo;      // linear, 0..1: the fraction of light reflected
    float roughness;  // 0..1: 0 = mirror-like, 1 = fully rough
    float metallic;   // 0..1: 0 = dielectric (plastic, ...), 1 = metal

    // Reflectance of a dielectric looking straight at it (4% for the
    // usual materials, kDielectricF0). 0 means "no glossy layer at all":
    // together with metallic 0 the surface is then plain Lambert paint
    // (that is what turning the materials off gives)
    float dielectric_f0;

    // Linear radiance the surface gives off by itself, per colour
    // channel, in the same units as the sky and the sun. Radiance is
    // "how much light travels along one ray", so this is exactly what
    // a ray that arrives at the surface from the camera side brings
    // back: the surface shines this much, seen from any direction and
    // from both sides. May be far above 1 (a lamp)
    Vec3 emission;
};

// The light a material gives off at the texture coordinates uv: the
// emission factor times the emissive texture (a colour, stored as sRGB,
// so decoded to linear light first). Level 0 only and the same v flip as
// every other texture lookup of the path tracer. Most surfaces give off
// no light: those skip the texture read and the decoding
__device__ __forceinline__ Vec3 surface_emission(
                const MaterialTables& tables,
                const CudaMaterial& material,
                Vec2 uv) {

    if (material.emission.x > 0.0f || material.emission.y > 0.0f ||
        material.emission.z > 0.0f) {
        const Vec3 texel = decode_srgb(
            sample_bilinear(tables.textures[material.emissive_texture],
                            0, uv.x, 1.0f - uv.y));

        return modulate(material.emission, texel);
    }

    return Vec3{0.0f, 0.0f, 0.0f};
}

// Everything the path tracer needs to know about the surface at `point`
// (a hit found by closest_surface). The expensive part of the lookup,
// the extra texture reads, happens here and only here, so the ray tracer
// that shares closest_surface does not pay for it.
//
// Three textures play three different roles, and that decides how their
// texels are read:
//   - base colour: a COLOUR, stored as sRGB. closest_surface already
//     multiplied it into point.base; the albedo is decode_srgb of that.
//   - emissive: also a colour in sRGB, so its texel is decoded to
//     linear light before it scales the (linear) emission factor.
//   - metallic-roughness: DATA, not a colour. The texel values are the
//     numbers themselves (G = roughness, B = metallic, both 0..1 as
//     stored), so decoding them as sRGB would bend the values: a
//     roughness of 0.5 stored as 128 must come out as 0.5, not 0.21.
//
// With materials off (the M key) the surface is plain grey paint: the
// grey albedo that closest_surface put in point.base, no glossy layer
// (dielectric_f0 0, not metallic) and no emission. "No materials" has to
// mean no lamps and no shine either
__device__ __forceinline__ PathMaterial resolve_surface_material(
                const MaterialTables& tables,
                const SurfacePoint& point) {

    PathMaterial result;
    result.albedo = decode_srgb(point.base);

    if (!tables.show_materials) {
        result.roughness = 0.5f;
        result.metallic = 0.0f;
        result.dielectric_f0 = 0.0f;
        result.emission = Vec3{0.0f, 0.0f, 0.0f};
        return result;
    }

    const CudaMaterial& material = tables.materials[point.material];

    // Same texture coordinates and the same v flip as the base colour:
    // image rows run top to bottom while v points up. Level 0 only,
    // like every ray (no pixel footprint to choose a mip level from)
    const float u = point.uv.x;
    const float v = 1.0f - point.uv.y;

    // Data texture: used as stored. The product of two numbers in 0..1
    // stays in 0..1; clamp01 only guards against a table entry that was
    // not made by CudaMeshStore
    const Vec3 packed =
        sample_bilinear(tables.textures[material.metallic_roughness_texture],
                        0, u, v);

    result.roughness = clamp01(material.roughness * packed.y);
    result.metallic = clamp01(material.metallic * packed.z);
    result.dielectric_f0 = kDielectricF0;

    result.emission = surface_emission(tables, material, point.uv);

    return result;
}

// ---------------------------------------------------------------------
// Area lights: sampling the emissive triangles (see core/light_sampling.h)
// ---------------------------------------------------------------------

// A point chosen on a lamp triangle
struct LightSample {
    Vec3 position;
    Vec3 normal;     // unit, orientation arbitrary (lamps are two-sided)
    float area;      // world-space area of the triangle
    float prob;      // probability of having picked this triangle
    Vec3 emission;   // linear radiance leaving the sampled point
};

// The world-space corners of a triangle of an instance
__device__ __forceinline__ void instance_triangle(
                const RtInstance& inst,
                std::uint32_t triangle,
                std::uint32_t& i0,
                std::uint32_t& i1,
                std::uint32_t& i2,
                Vec3& w0,
                Vec3& w1,
                Vec3& w2) {

    const DeviceMesh& mesh = inst.mesh;

    i0 = mesh.indices[triangle * 3 + 0];
    i1 = mesh.indices[triangle * 3 + 1];
    i2 = mesh.indices[triangle * 3 + 2];

    w0 = transform_point(inst.object_to_world, mesh.positions[i0]);
    w1 = transform_point(inst.object_to_world, mesh.positions[i1]);
    w2 = transform_point(inst.object_to_world, mesh.positions[i2]);
}

// Picks a lamp point: the instance by a scan over the instance weights
// (u0), the triangle by binary search in the mesh's cdf (u1), the point
// uniformly by area inside it (u2, u3). Four independent random numbers
// keep the three choices from being correlated. False when the scene has
// no usable lamp
__device__ __forceinline__ bool sample_light(
                const RtScene& scene,
                float u0,
                float u1,
                float u2,
                float u3,
                std::uint32_t& instance,
                std::uint32_t& triangle,
                LightSample& out) {

    const float target = u0 * scene.light_total;
    float running = 0.0f;
    int chosen = -1;
    int last = -1;

    for (int i = 0; i < scene.count; ++i) {
        const float w = scene.items[i].light_weight;

        if (w > 0.0f) {
            last = i;
            running += w;

            if (running >= target) {
                chosen = i;
                break;
            }
        }
    }

    if (chosen < 0) {
        chosen = last;  // rounding: the scan ended just short of target
    }

    if (chosen < 0) {
        return false;
    }

    const RtInstance& inst = scene.items[chosen];
    const DeviceMesh& mesh = inst.mesh;

    float tri_prob = 0.0f;
    const unsigned int pick =
        pick_from_cdf(mesh.light_cdf, mesh.light_count, u1, tri_prob);

    if (!(tri_prob > 0.0f)) {
        return false;
    }

    instance = static_cast<std::uint32_t>(chosen);
    triangle = mesh.light_tris[pick];

    std::uint32_t i0;
    std::uint32_t i1;
    std::uint32_t i2;
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;

    instance_triangle(inst, triangle, i0, i1, i2, p0, p1, p2);
    triangle_area_normal(p0, p1, p2, out.area, out.normal);

    if (!(out.area > 0.0f)) {
        return false;
    }

    float b0;
    float b1;
    float b2;

    sample_triangle_barycentric(u2, u3, b0, b1, b2);

    out.position = p0 * b0 + p1 * b1 + p2 * b2;
    out.prob = (inst.light_weight / scene.light_total) * tri_prob;

    // The emission at the sampled point: factor times emissive texture
    const CudaMaterial& material = scene.surfaces.materials[
        mesh.material_base + mesh.triangle_materials[triangle]];

    out.emission = surface_emission(
        scene.surfaces, material, hit_uv(mesh, i0, i1, i2, b0, b1, b2));

    return true;
}

// The solid-angle density with which sample_light would have produced
// the point where a ray from the surface hit the lamp triangle in
// `point`: exactly the same formula, evaluated from the hit. Returns 0
// when the hit is not a lamp triangle of the lists, or it is a case
// the light sampling skips (grazing, or at the surface); the caller then
// gives the bounce ray the full weight 1
__device__ __forceinline__ float light_hit_pdf(
                const RtScene& scene,
                const SurfacePoint& point,
                Vec3 unit_dir) {

    const RtInstance& inst = scene.items[point.instance];
    const DeviceMesh& mesh = inst.mesh;

    const unsigned int index = find_in_sorted(
        mesh.light_tris, mesh.light_count, point.triangle);

    if (index >= mesh.light_count || !(inst.light_weight > 0.0f)) {
        return 0.0f;
    }

    std::uint32_t i0;
    std::uint32_t i1;
    std::uint32_t i2;
    Vec3 p0;
    Vec3 p1;
    Vec3 p2;
    float area;
    Vec3 normal;

    instance_triangle(inst, point.triangle, i0, i1, i2, p0, p1, p2);
    triangle_area_normal(p0, p1, p2, area, normal);

    const float cos_light = fabsf(dot(normal, unit_dir));

    if (!(area > 0.0f) || cos_light < kMinLightCosine ||
        point.t < kMinLightDistance) {
        return 0.0f;
    }

    const float prob =
        (inst.light_weight / scene.light_total) *
        cdf_probability(mesh.light_cdf, mesh.light_count, index);

    return area_to_solid_angle_pdf(prob / area, point.t, cos_light);
}

// Kd / pi + f_s(l, v): the whole BRDF of the surface for the light
// direction l, and the cosine of h = normalize(l + v) with the normal
// (the density of the bounce needs it too)
__device__ __forceinline__ Vec3 surface_brdf(
                Vec3 n,
                Vec3 v,
                Vec3 l,
                float cos_v,
                float cos_l,
                float alpha,
                Vec3 kd,
                Vec3 f0,
                bool has_specular,
                float& n_dot_h) {

    const Vec3 h = normalize(l + v);

    n_dot_h = dot(n, h);

    Vec3 f = kd * (1.0f / kPi);

    if (has_specular) {
        f = f + ggx_specular_brdf(
            alpha, f0, cos_l, cos_v, n_dot_h, dot(v, h));
    }

    return f;
}

// The radiance (linear light) arriving at the camera through the point
// (x + jx, y + jy) of the image, for one sample, found by following one
// random path of up to max_bounces surfaces.
//
// For one sample the estimate is
//     L = sum over the path's vertices of (throughput there) * (light
//         added there),
// where the throughput is the product of the weights of the bounces so
// far (for a diffuse bounce the diffuse colour, for a glossy one the
// Fresnel and shadowing factors; see below) and the light added is what
// the vertex's surface emits, the sun it receives through the shadow ray,
// or the sky a ray that left the scene sees.
// Averaging many samples (the accumulation buffer) gives the picture.
//
// Random numbers: every sample of every pixel gets its own seed,
// pcg_hash(pixel ^ pcg_hash(sample_index * 0x9E3779B9u + 1u)), and
// random01 advances it. sample_index is the number of the sample at that
// pixel (the frames since the last restart), so each frame is a new
// independent sample. The multiplication by the golden-ratio constant
// spreads consecutive sample numbers over the 32-bit range before
// hashing; the +1 keeps sample 0 from hashing the number 0.
//
// depth receives the view depth of the camera ray's hit, or -1 for a miss
__device__ __forceinline__ Vec3 trace_path(
                const RtScene& scene,
                const PtParams& pt,
                unsigned int pixel,
                int x,
                int y,
                int width,
                int height,
                unsigned int sample_index,
                float& depth) {

    unsigned int seed =
        pcg_hash(pixel ^ pcg_hash(sample_index * 0x9E3779B9u + 1u));

    // Where inside the pixel the camera ray goes: a uniform point of the
    // pixel's square. Over many samples this averages the pixel's whole
    // footprint, which is free anti-aliasing
    const float jitter_x = random01(seed);
    const float jitter_y = random01(seed);

    Vec3 origin = scene.eye;
    Vec3 dir = camera_ray_dir_at(
        scene, static_cast<float>(x) + jitter_x,
        static_cast<float>(y) + jitter_y, width, height);

    // The camera ray is limited by the near and far plane like in the
    // other modes; the rays after it are not
    float t_min = scene.t_min;
    float t_max = scene.t_max;

    Vec3 radiance{0.0f, 0.0f, 0.0f};
    Vec3 throughput{1.0f, 1.0f, 1.0f};

    depth = -1.0f;

    // The sun as seen by multiple importance sampling (MIS; the long
    // explanation is in core/brdf.h). The sun is found in two ways: the
    // shadow ray below aims at it, and a bounce ray may escape through
    // it. Each way gets a weight from the densities of both, and the two
    // weights add up to 1 for every direction in the cone, so the sun is
    // counted exactly once:
    //     shadow ray:  w_sun  = 1 / (1 + (pdf_bsdf / pdf_sun)^2)
    //     bounce ray:  w_bsdf = 1 / (1 + (pdf_sun / pdf_bsdf)^2)
    // pdf_sun = 1 / sun_solid_angle is the density of the shadow ray's
    // direction (uniform in the cone), pdf_bsdf the density with which
    // the surface's bounce would generate that same direction.
    // sun_radiance = E / solid angle is the radiance inside the cone.
    // prev_pdf is the density of the bounce that produced the current
    // ray (0 for the camera ray, which has no bounce behind it)
    const float sun_solid_angle =
        2.0f * kPi * (1.0f - pt.sun_cos_half_angle);
    const bool sun_mis = pt.sun_enabled && sun_solid_angle > 0.0f;
    const float sun_radiance =
        sun_mis ? pt.sun_irradiance / sun_solid_angle : 0.0f;
    float prev_pdf = 0.0f;

    // Area lights (emissive triangles) get the same treatment as the sun:
    // a shadow ray aimed at a random point of a lamp, and the bounce rays
    // that hit a lamp by chance, weighed against each other so that the
    // light is counted once. With no lamps (or materials off, which
    // switches emission off) none of this runs
    const bool lights_on =
        scene.light_total > 0.0f && scene.surfaces.show_materials;

    // One iteration per ray. Rays 0 .. max_bounces - 1 end on surfaces
    // that are shaded; ray max_bounces only collects the sky and the
    // emission of the surface it hits, because it is the continuation of
    // the last shaded surface (and a miss of the camera ray, ray 0,
    // collects the viewport background instead)
    for (int bounce = 0; bounce <= pt.max_bounces; ++bounce) {
        SurfacePoint point;

        if (!closest_surface(scene, origin, dir, t_min, t_max, point)) {
            const Vec3 unit_dir = normalize(dir);
            const Vec3 seen = bounce == 0
                ? pt.background
                : sky_radiance(unit_dir);

            radiance = radiance + modulate(throughput, seen);

            // A bounce ray that escapes through the sun's cone found the
            // sun by chance. It counts, with the weight w_bsdf of the MIS
            // above; the shadow ray of the surface it left counts the
            // same light with w_sun, and the two weights add up to 1.
            // The final ray (bounce == max_bounces) must count too:
            // the shadow ray of the last shaded surface already used a
            // weight w_sun < 1, and the rest of the sun's light is only
            // accounted for by this ray. The camera ray shows no sun disc
            // (the plain background), and a bounce ray that hits
            // geometry is not an escape: the surface blocks the sun
            if (bounce > 0 && sun_mis && prev_pdf > 0.0f &&
                dot(unit_dir, scene.sun_world) >= pt.sun_cos_half_angle) {
                const float ratio = 1.0f / (prev_pdf * sun_solid_angle);

                radiance = radiance + modulate(throughput,
                    Vec3{sun_radiance, sun_radiance, sun_radiance}) *
                    (1.0f / (1.0f + ratio * ratio));
            }

            break;
        }

        if (bounce == 0) {
            depth = point.t;
        }

        // The albedo is the display colour decoded to linear light. Base
        // colours are clamped to 0..1 by the decoding, so a surface never
        // reflects more than it receives
        const PathMaterial material =
            resolve_surface_material(scene.surfaces, point);
        const Vec3 albedo = material.albedo;

        // Emission: this surface is itself a source of light. The ray
        // that found it travels backwards along the path, so whatever
        // the surface emits towards us is light that reaches the camera
        // after being reflected by all the surfaces between (the
        // throughput, the product of their weights, says how much of it
        // survives). The camera ray's first hit has throughput 1, so a
        // lamp seen directly shows its full emission.
        //
        // This comes BEFORE the "out of bounces" test: the last ray of
        // the path may well end on a lamp, and that light is as real as
        // any other. Only the shading below (the sun, the next
        // direction) is limited to max_bounces surfaces; adding what the
        // hit surface emits costs no further ray, so it is free to count
        // every hit. (Dropping it would make small lamps vanish from the
        // image whenever the path that finds them is already long.)
        // A bounce ray that hits a lamp by chance gets the weight w_bsdf
        // = 1 / (1 + (pdf_light / pdf_bsdf)^2) of the MIS: pdf_light is
        // the density with which the light sampling of the previous
        // surface would have picked this very point, pdf_bsdf (prev_pdf)
        // the density of the bounce that found it. Where light sampling
        // is better (small, distant lamps) the weight is near 0 and
        // that strategy's share is used instead (below). The camera ray
        // (bounce 0) sees the lamp directly: weight 1. This applies to
        // the final ray too: the previous surface's shadow ray used a
        // weight below 1, and this ray supplies the rest
        float emission_weight = 1.0f;

        if (bounce > 0 && lights_on && prev_pdf > 0.0f &&
            (material.emission.x > 0.0f || material.emission.y > 0.0f ||
             material.emission.z > 0.0f)) {
            const float pdf_light =
                light_hit_pdf(scene, point, normalize(dir));

            if (pdf_light > 0.0f) {
                const float ratio = pdf_light / prev_pdf;

                emission_weight = 1.0f / (1.0f + ratio * ratio);
            }
        }

        radiance = radiance +
            modulate(throughput, material.emission) * emission_weight;

        if (bounce == pt.max_bounces) {
            break;  // out of bounces: this surface is not shaded
        }

        Vec3 n = normalize(point.normal);

        if (dot(n, n) == 0.0f) {
            n = point.geometric;  // degenerate vertex normals
        }

        const Vec3 start =
            point.position + point.geometric * shadow_offset(point.position);

        // The surface is a mix of two lobes, a diffuse (Lambert) one and a
        // glossy (GGX) one; core/brdf.h explains the model. Everything
        // below is in world space. v points back to where the ray came
        // from, and its cosine with the normal is kept above a tiny
        // number, because interpolated normals can face slightly away
        // from the viewer
        const Vec3 v = -normalize(dir);
        const float cos_v = fmaxf(dot(n, v), kMinCosView);
        const float alpha = ggx_alpha_from_roughness(material.roughness);

        // No glossy layer at all (materials off): the specular lobe is
        // skipped entirely. A dielectric f0 of 0 is the "no layer" marker
        // of PathMaterial; Schlick's curve through 0 would still rise to 1
        // at glancing angles
        const bool has_specular =
            material.dielectric_f0 > 0.0f || material.metallic > 0.0f;

        // Reflectance at normal incidence: a dielectric reflects 4% of
        // every colour, a metal reflects its colour (the base colour)
        Vec3 f0{0.0f, 0.0f, 0.0f};

        if (has_specular) {
            const float m = material.metallic;
            const float d0 = material.dielectric_f0 * (1.0f - m);

            f0 = Vec3{d0 + albedo.x * m, d0 + albedo.y * m,
                      d0 + albedo.z * m};
        }

        // The diffuse colour: nothing for a metal (its electrons reflect
        // or absorb all light at the surface), and for a dielectric the
        // albedo reduced by what the clear glossy layer on top already
        // reflected. The more it reflects at this view angle (glancing
        // views), the less is left to enter the diffuse layer. A cheap
        // approximation of the energy balance, not a measured model
        const float fd =
            fresnel_dielectric_layer(material.dielectric_f0, cos_v);
        const Vec3 kd = albedo * ((1.0f - material.metallic) * (1.0f - fd));

        // The probability of bouncing off the glossy lobe (see below); it
        // is also needed here, to weigh the shadow ray against the bounce
        const float p_spec = specular_lobe_probability(
            has_specular ? fresnel_schlick(f0, cos_v)
                         : Vec3{0.0f, 0.0f, 0.0f},
            kd);

        // Next-event estimation: one shadow ray towards a random point
        // of the sun's disc. The disc has radiance E / solid_angle and a
        // uniformly chosen direction has density 1 / solid_angle, so the
        // solid angle cancels and the surface receives
        //     (Kd / pi + f_s(l, v)) * E * cos(angle to the normal),
        // both lobes at once (the lobes are not picked here: the sun is
        // a light to aim at, not a direction to bounce into). This
        // is multiplied by the MIS weight w_sun, which is close to 1
        // where the bounce rays would rarely find the sun (diffuse and
        // rough surfaces: pdf_bsdf is about a thousand times smaller than
        // pdf_sun, so the weight is 1 to six digits and nothing changes
        // for them) and close to 0 on a smooth glossy highlight, where
        // the bounce rays do the job better. Directions below the
        // surface (shading normal or the real triangle) get nothing
        if (pt.sun_enabled) {
            const float u1 = random01(seed);
            const float u2 = random01(seed);

            const Vec3 l = sample_sun_cone(
                scene.sun_world, pt.sun_cos_half_angle, u1, u2);

            const float cos_l = dot(n, l);

            if (cos_l > 0.0f && dot(point.geometric, l) > 0.0f &&
                !scene_occluded(scene, start, l, 1.0e30f)) {
                float n_dot_h;
                const Vec3 f = surface_brdf(
                    n, v, l, cos_v, cos_l, alpha, kd, f0, has_specular,
                    n_dot_h);

                // pdf_bsdf / pdf_sun = pdf_bsdf * solid angle
                float w_sun = 1.0f;

                if (sun_mis) {
                    const float ratio = sun_solid_angle *
                        bsdf_mixture_pdf(p_spec, alpha, cos_v, cos_l,
                                         n_dot_h);

                    w_sun = 1.0f / (1.0f + ratio * ratio);
                }

                radiance = radiance + modulate(throughput, f) *
                    (pt.sun_irradiance * cos_l * w_sun);
            }
        }

        // Next-event estimation for the area lights: one shadow ray
        // towards a random point of a random lamp triangle. A point on
        // the lamp has the solid-angle density pdf_light (see
        // core/light_sampling.h), so the light the surface receives is
        //     f(l) cos(l) L_e / pdf_light,
        // times the MIS weight w_light = 1 / (1 + (pdf_bsdf / pdf_light)^2)
        // (pdf_bsdf: the density with which this surface's bounce would
        // have generated l). Written as one fraction,
        //     f cos L_e / (pdf_light + pdf_bsdf^2 / pdf_light),
        // so a tiny pdf_light cannot turn "huge times zero" into a NaN.
        // Lamps are two-sided, and the lamp triangle the surface point
        // itself is on is skipped (it cannot light itself)
        if (lights_on) {
            const float lu0 = random01(seed);
            const float lu1 = random01(seed);
            const float lu2 = random01(seed);
            const float lu3 = random01(seed);

            std::uint32_t light_instance = 0u;
            std::uint32_t light_triangle = 0u;
            LightSample ls;

            if (sample_light(scene, lu0, lu1, lu2, lu3,
                             light_instance, light_triangle, ls) &&
                !(light_instance == point.instance &&
                  light_triangle == point.triangle)) {
                const Vec3 to_light = ls.position - start;
                const float dist = length(to_light);

                if (dist >= kMinLightDistance) {
                    const Vec3 l = to_light * (1.0f / dist);
                    const float cos_l = dot(n, l);
                    const float cos_light = fabsf(dot(ls.normal, l));

                    if (cos_l > 0.0f && dot(point.geometric, l) > 0.0f &&
                        cos_light >= kMinLightCosine &&
                        !scene_occluded(
                            scene, start, l,
                            dist * (1.0f - kLightShadowShortening))) {
                        const float pdf_light = area_to_solid_angle_pdf(
                            ls.prob / ls.area, dist, cos_light);

                        float n_dot_h;
                        const Vec3 f = surface_brdf(
                            n, v, l, cos_v, cos_l, alpha, kd, f0,
                            has_specular, n_dot_h);
                        const float pdf_bsdf = bsdf_mixture_pdf(
                            p_spec, alpha, cos_v, cos_l, n_dot_h);

                        if (pdf_light > 0.0f && finite_float(pdf_light)) {
                            const float scale = cos_l /
                                (pdf_light + pdf_bsdf * pdf_bsdf /
                                                 pdf_light);

                            radiance = radiance +
                                modulate(throughput,
                                         modulate(f, ls.emission)) * scale;
                        }
                    }
                }
            }
        }

        // Continue the path in ONE direction, sampled from ONE of the two
        // lobes. The two random numbers v1 and v2 are used by whichever
        // lobe is picked. The lobe is picked with probability p_spec for
        // the glossy one and 1 - p_spec for the diffuse one (the
        // probability follows how much light each lobe reflects). A third
        // random number is only drawn when there is really a choice
        // (0 < p_spec < 1), so a surface with a single lobe uses the same
        // random numbers as before the glossy lobe existed
        const float v1 = random01(seed);
        const float v2 = random01(seed);
        const float v3 = (p_spec > 0.0f && p_spec < 1.0f)
            ? random01(seed) : 0.0f;

        const BsdfSample bounce_sample =
            sample_bsdf_mixture(n, v, alpha, p_spec, v1, v2, v3);

        const Vec3 d = bounce_sample.d;
        const float cos_out = dot(n, d);

        // A glossy bounce can end up below the surface (the microfacet's
        // mirror points into the surface), and a diffuse one can dip below
        // the actual triangle (smooth normals bend away from it): such a
        // sample carries no light, the path ends
        if (cos_out <= 0.0f || dot(d, point.geometric) <= 0.0f) {
            break;
        }

        // The density with which this bounce generated d: the mixture of
        // both lobes, whichever one was picked (core/brdf.h). It is the
        // divisor of the weight below and, kept for the next ray, the
        // pdf_bsdf of the MIS weight if that ray escapes through the sun
        const Vec3 h_out = normalize(d + v);
        const float n_dot_h_out = dot(n, h_out);
        const float pdf = bsdf_mixture_pdf(
            p_spec, alpha, cos_v, cos_out, n_dot_h_out);

        if (!finite_float(pdf) || !(pdf > 0.0f)) {
            break;
        }

        // The weight of the bounce, f(d) cos(d) / pdf(d), with the
        // mixture density. Written out like this it is the same for any
        // lobe that gets picked, which is what makes it consistent with
        // the MIS weights. When the surface has a single lobe the density
        // cancels the BRDF almost completely and the plain forms are used
        // (they are exact and well behaved):
        //  - glossy only: F(v.h) G2 / G1(v), at most 1 per channel (the
        //    cancellation is explained in core/brdf.h)
        //  - diffuse only: Kd (the density cos / pi cancels Kd / pi cos)
        // so for them the result is the same as before MIS. With both
        // lobes the full f = Kd / pi + f_s is divided by the mixture
        Vec3 weight;

        if (p_spec >= 1.0f) {
            weight = fresnel_schlick(f0, dot(v, bounce_sample.h)) *
                     smith_g2_over_g1(alpha, cos_out, cos_v);
        } else if (p_spec <= 0.0f) {
            weight = kd;
        } else {
            const Vec3 f = kd * (1.0f / kPi) +
                ggx_specular_brdf(alpha, f0, cos_out, cos_v, n_dot_h_out,
                                  dot(v, h_out));

            weight = f * (cos_out / pdf);
        }

        prev_pdf = pdf;

        throughput = modulate(throughput, weight);

        // Russian roulette: terminate with probability 1 - q, otherwise
        // divide by q, which keeps the average the same
        if (bounce >= kPtRouletteStart) {
            const float q = russian_roulette_survival(throughput);

            if (random01(seed) >= q) {
                break;
            }

            throughput = throughput * (1.0f / q);
        }

        origin = start;
        dir = d;
        t_min = 0.0f;
        t_max = 1.0e30f;
    }

    return radiance;
}

// Adds pt.samples_to_add samples to every pixel's accumulator. A sum is
// a running mean in disguise: the mean after N samples is sum / N, so
// storing the sum and the count (not the mean) needs no rounding
// compensation, and one more sample is just one more addition
__global__ void path_trace_kernel(
                int width,
                int height,
                RtScene scene,
                PtParams pt) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const unsigned int pixel =
        static_cast<unsigned int>(pixel_index(x, y, width));

    // Starting over: whatever is in the buffer is from an older picture
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, -1.0f);

    if (pt.samples_before > 0) {
        acc = pt.accum[pixel];
    }

    for (unsigned int i = 0; i < pt.samples_to_add; ++i) {
        float depth = -1.0f;

        const Vec3 radiance = trace_path(
            scene, pt, pixel, x, y, width, height,
            pt.samples_before + i, depth);

        // A NaN or infinite sample adds nothing (it still counts as a
        // sample), so one bad value can never poison the sum
        if (finite_vec3(radiance)) {
            acc.x += radiance.x;
            acc.y += radiance.y;
            acc.z += radiance.z;
        }

        // The depth the grid is tested against: the closest camera hit
        // of any sample, so a pixel on an object's silhouette keeps
        // hiding the grid instead of flickering with the jitter
        if (depth > 0.0f && (acc.w < 0.0f || depth < acc.w)) {
            acc.w = depth;
        }
    }

    pt.accum[pixel] = acc;
}

// Always runs: turns the accumulated sums into the picture. Idempotent
// (it only reads the accumulator), so a converged picture is just
// resolved again every frame
__global__ void pt_resolve_kernel(
                uchar4* color,
                unsigned long long* visibility,
                int width,
                int height,
                RtScene scene,
                PtParams pt) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel = pixel_index(x, y, width);

    const unsigned int count = pt.samples_before + pt.samples_to_add;

    if (count == 0) {
        return;  // nothing accumulated yet: keep the cleared background
    }

    const float4 acc = pt.accum[pixel];

    // Every sample so far missed everything: the pixel is background and
    // keeps exactly the colour the clear pass gave it, with no visibility
    // entry, so the grid draws over it. (Their mean would be the same
    // colour up to rounding; this makes it exact.) A pixel with at least
    // one hit is resolved below, its mean including the background
    // radiance of the samples that missed: the anti-aliased edge
    if (acc.w < 0.0f) {
        return;
    }

    const Vec3 mean =
        Vec3{acc.x, acc.y, acc.z} * (1.0f / static_cast<float>(count));

    const Vec3 display = linear_to_display(mean, pt.exposure);

    color[pixel] = make_uchar4(
        to_byte(display.x), to_byte(display.y), to_byte(display.z), 255);

    // Covered pixel: depth + id 1 as the ray tracer writes them. The
    // world position is on the pixel centre's ray at the stored view
    // depth; the depth value only depends on the view depth, so the
    // jitter of the sample that found it does not matter
    if (acc.w > 0.0f) {
        const Vec3 position =
            scene.eye + acc.w * camera_ray_dir(scene, x, y, width, height);

        visibility[pixel] = visibility_entry(scene, position);
    }
}

__global__ void resolve_ids_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel = pixel_index(x, y, width);
    const unsigned long long packed = visibility[pixel];

    if (packed == ~0ULL) {
        return;
    }

    const unsigned int id = static_cast<unsigned int>(packed);
    const unsigned int hash = id * 0x9E3779B9u;

    color[pixel] = make_uchar4(
        static_cast<unsigned char>(96u + (hash & 0x9Fu)),
        static_cast<unsigned char>(96u + ((hash >> 8) & 0x9Fu)),
        static_cast<unsigned char>(96u + ((hash >> 16) & 0x9Fu)),
        255);
}

__device__ __forceinline__ float grid_smoothstep(
    float edge0, float edge1, float x) {

    float t = fminf(
        fmaxf((x - edge0) / (edge1 - edge0), 0.0f),
        1.0f);

    return t * t * (3.0f - 2.0f * t);
}

__device__ __forceinline__ float aa_line(
    float dist_px, float width_px) {

    return fminf(
        fmaxf(0.5f * width_px + 0.5f - dist_px,
              0.0f),
        1.0f);
}

__device__ __forceinline__ float grid_fract(float x) {
    return x - floorf(x);
}

__device__ __forceinline__ bool grid_hit(
                const CudaGridParams& params,
                float sx,
                float sy,
                int width,
                int height,
                Vec3& ray,
                float& t,
                Vec3& p) {

    const float ndc_x = 2.0f * sx / width - 1.0f;
    const float ndc_y = 1.0f - 2.0f * sy / height;

    ray = params.forward
        + ndc_x * params.right
        + ndc_y * params.up;

    if (fabsf(ray.z) <= 1.0e-6f) {
        return false;
    }

    t = -params.eye.z / ray.z;

    if (t <= 0.0f) {
        return false;
    }

    p = params.eye + t * ray;
    return true;
}

__device__ __forceinline__ float grid_level(
                float2 p,
                float spacing,
                float width_px,
                float fw_x,
                float fw_y) {

    const float cx = p.x / spacing;
    const float cy = p.y / spacing;

    const float wx = fmaxf(fw_x / spacing, 1.0e-6f);
    const float wy = fmaxf(fw_y / spacing, 1.0e-6f);

    const float dx =
        fabsf(grid_fract(cx - 0.5f) - 0.5f) / wx;

    const float dy =
        fabsf(grid_fract(cy - 0.5f) - 0.5f) / wy;

    const float lod_x =
        grid_smoothstep(2.0f, 8.0f, 1.0f / wx);

    const float lod_y =
        grid_smoothstep(2.0f, 8.0f, 1.0f / wy);

    return fmaxf(
        aa_line(dx, width_px) * lod_x,
        aa_line(dy, width_px) * lod_y);
}

__device__ __forceinline__ float4 grid_layer(
    float4 dst, float3 src_rgb, float src_a) {

    const float a = src_a + dst.w * (1.0f - src_a);

    const float3 rgb = make_float3(
        (src_rgb.x * src_a +
         dst.x * dst.w * (1.0f - src_a)) / fmaxf(a, 1.0e-6f),

        (src_rgb.y * src_a +
         dst.y * dst.w * (1.0f - src_a)) / fmaxf(a, 1.0e-6f),

        (src_rgb.z * src_a +
         dst.z * dst.w * (1.0f - src_a)) / fmaxf(a, 1.0e-6f));

    return make_float4(rgb.x, rgb.y, rgb.z, a);
}

__global__ void draw_grid_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height,
                CudaGridParams params) {

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t pixel =
        static_cast<std::size_t>(y) * width + x;

    // Intersect the ray through the pixel centre with z = 0
    Vec3 ray{};
    Vec3 p{};
    float t = 0.0f;

    if (!grid_hit(
            params,
            x + 0.5f,
            y + 0.5f,
            width,
            height,
            ray,
            t,
            p)) {
        return;
    }

    // Estimate the world-space footprint of one pixel
    Vec3 ray_dx{};
    Vec3 p_dx{};
    float t_dx = 0.0f;

    Vec3 ray_dy{};
    Vec3 p_dy{};
    float t_dy = 0.0f;

    const bool hit_dx = grid_hit(
        params, x + 1.5f, y + 0.5f,
        width, height, ray_dx, t_dx, p_dx);

    const bool hit_dy = grid_hit(
        params, x + 0.5f, y + 1.5f,
        width, height, ray_dy, t_dy, p_dy);

    float fw_x = 1.0e6f;
    float fw_y = 1.0e6f;

    if (hit_dx && hit_dy) {
        fw_x =
            fabsf(p_dx.x - p.x) +
            fabsf(p_dy.x - p.x);

        fw_y =
            fabsf(p_dx.y - p.y) +
            fabsf(p_dy.y - p.y);
    }

    fw_x = fmaxf(fw_x, 1.0e-6f);
    fw_y = fmaxf(fw_y, 1.0e-6f);

    // Match the OpenGL grid shader's spacing and colours
    const float minor =
        grid_level(make_float2(p.x, p.y), 1.0f,
                   1.0f, fw_x, fw_y);

    const float major =
        grid_level(make_float2(p.x, p.y), 10.0f,
                   1.0f, fw_x, fw_y);

    const float axis_x =
        aa_line(fabsf(p.y) / fw_y, 1.25f);

    const float axis_y =
        aa_line(fabsf(p.x) / fw_x, 1.25f);

    float4 c = make_float4(0, 0, 0, 0);

    c = grid_layer(
        c, make_float3(0.55f, 0.55f, 0.55f),
        0.10f * minor);

    c = grid_layer(
        c, make_float3(0.65f, 0.65f, 0.65f),
        0.24f * major);

    c = grid_layer(
        c, make_float3(0.85f, 0.25f, 0.30f),
        0.75f * axis_x);

    c = grid_layer(
        c, make_float3(0.45f, 0.78f, 0.25f),
        0.75f * axis_y);

    // Fade the grid with distance from the camera
    const float ray_length = sqrtf(
        ray.x * ray.x +
        ray.y * ray.y +
        ray.z * ray.z);

    const float dist = t * ray_length;

    c.w *= 1.0f - grid_smoothstep(
        0.4f * params.fade_end,
        params.fade_end,
        dist);

    // Push only the depth-test position slightly farther away
    // Keep the true hit point for colour and fade
    const Vec3 p_depth =
        params.eye + (t * 1.001f) * ray;

    const Vec4 clip =
        params.view_proj * make_vec4(p_depth, 1.0f);

    if (fabsf(clip.w) <= 1.0e-6f) {
        return;
    }

    const float ndc_z = clip.z / clip.w;

    if (ndc_z < -1.0f ||
        ndc_z > 1.0f ||
        c.w < 0.004f) {
        return;
    }

    const float grid_depth = ndc_z * 0.5f + 0.5f;

    // The visibility buffer packs depth in the high 32 bits
    // and the triangle ID in the low 32 bits
    const unsigned long long packed = visibility[pixel];

    if (packed != ~0ULL) {
        const unsigned int depth_bits =
            static_cast<unsigned int>(packed >> 32);

        const float scene_depth = __uint_as_float(depth_bits);

        // GL_LESS: the grid must be closer than the mesh.
        if (!(grid_depth < scene_depth)) {
            return;
        }
    }

    // Standard SRC_ALPHA, ONE_MINUS_SRC_ALPHA blending
    const uchar4 dst = color[pixel];

    const float inv_255 = 1.0f / 255.0f;
    const float one_minus_a = 1.0f - c.w;

    const float out_r =
        c.x * c.w + dst.x * inv_255 * one_minus_a;

    const float out_g =
        c.y * c.w + dst.y * inv_255 * one_minus_a;

    const float out_b =
        c.z * c.w + dst.z * inv_255 * one_minus_a;

    color[pixel] = make_uchar4(
        to_byte(out_r),
        to_byte(out_g),
        to_byte(out_b),
        255);
}

} // namespace

bool cuda_clear(
    uchar4* color, int width, int height, uchar4 value) {

    if (!color || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA clear: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    clear_kernel<<<blocks, threads>>>(color, width, height, value);
    return check_launch("CUDA clear kernel launch");
}

bool cuda_clear_visibility(
    std::uint64_t* visibility, int width, int height) {

    if (!visibility || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA visibility clear: invalid arguments");
        return false;
    }

    const std::size_t count =
        static_cast<std::size_t>(width) * height;

    const unsigned int blocks = static_cast<unsigned int>(
        (count + kThreads - 1) / kThreads);

    clear_visibility_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility), count);

    return check_launch("CUDA visibility clear kernel launch");
}

bool cuda_transform_vertices(
                const DeviceMesh& mesh,
                Mat4 mvp,
                Mat4 model_view,
                Mat4 normal_to_view,
                int width,
                int height,
                ProjectedVertex* out) {

    if (!mesh.positions || !out || mesh.vertex_count == 0 ||
        width <= 0 || height <= 0) {
        LOG_ERROR("CUDA vertex transform: invalid arguments");
        return false;
    }

    const unsigned int blocks =
        (mesh.vertex_count + kThreads - 1) / kThreads;

    transform_vertices_kernel<<<blocks, kThreads>>>(
        mesh, mvp, model_view, normal_to_view,
        width, height, out);

    return check_launch("CUDA vertex transform kernel launch");
}

bool cuda_reset_raster_counts(RasterCounts* counts) {
    if (!counts) {
        LOG_ERROR("CUDA raster counts: null pointer");
        return false;
    }

    // Stream 0 keeps the reset ordered with the kernels around it
    return cuda_ok(
        cudaMemsetAsync(counts, 0, sizeof(RasterCounts), 0),
        "cudaMemsetAsync raster counts");
}

bool cuda_setup_triangles(
                const RasterWork& work,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices) {

    if (!work.records || !work.small_list || !work.large_list ||
        !work.counts || work.capacity == 0 ||
        !mesh.indices || !vertices ||
        width <= 0 || height <= 0 ||
        mesh.index_count < 3 || mesh.index_count % 3 != 0) {
        LOG_ERROR("CUDA triangle setup: invalid arguments");
        return false;
    }

    const unsigned int triangle_count = mesh.index_count / 3;
    const unsigned int blocks =
        (triangle_count + kThreads - 1) / kThreads;

    setup_triangles_kernel<<<blocks, kThreads>>>(
        work, width, height, mesh, vertices,
        guard_band(width, height));

    return check_launch("CUDA triangle setup kernel launch");
}

bool cuda_rasterize_worklist(
                std::uint64_t* visibility,
                int width,
                const RasterWork& work,
                int sm_count) {

    if (!visibility || width <= 0 ||
        !work.records || !work.small_list ||
        !work.large_list || !work.counts) {
        LOG_ERROR("CUDA worklist rasterizer: invalid arguments");
        return false;
    }

    // A few resident blocks per SM is enough to keep every SM busy.
    // Blocks that find no work exit immediately
    const unsigned int sms =
        sm_count > 0 ? static_cast<unsigned int>(sm_count) : 1u;

    rasterize_small_kernel<<<sms * 4, kThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility),
        width, work);

    if (!check_launch("CUDA small-triangle rasterizer launch")) {
        return false;
    }

    rasterize_large_kernel<<<sms * 8, kLargeThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility),
        width, work);

    return check_launch("CUDA large-triangle rasterizer launch");
}

bool cuda_draw_edges(
                std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices) {

    if (!visibility || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA edge rasterizer: invalid arguments");
        return false;
    }

    if (mesh.edge_index_count == 0) {
        return true;
    }

    if (!mesh.edges || !vertices ||
        mesh.edge_index_count % 2 != 0) {
        LOG_ERROR("CUDA edge rasterizer: invalid edge data");
        return false;
    }

    const unsigned int edge_count =
        mesh.edge_index_count / 2;

    const unsigned int blocks =
        (edge_count + kThreads - 1) / kThreads;

    draw_edges_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility),
        width,
        height,
        mesh,
        vertices,
        guard_band(width, height));

    return check_launch("CUDA edge rasterizer launch");
}

bool cuda_resolve_wire(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height) {

    if (!color || !visibility ||
        width <= 0 || height <= 0) {
        LOG_ERROR("CUDA wire resolve: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    resolve_wire_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width,
        height);

    return check_launch("CUDA wire resolve kernel launch");
}

bool cuda_shade(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                const RasterWork& work,
                const MaterialTables& tables,
                Vec3 up_view) {

    if (!color || !visibility || !work.records || !work.counts ||
        !tables.materials || !tables.textures ||
        width <= 0 || height <= 0) {
        LOG_ERROR("CUDA shade: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    shade_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width, height, work, tables, up_view);

    return check_launch("CUDA shade kernel launch");
}

bool cuda_ray_trace(
                uchar4* color,
                std::uint64_t* visibility,
                int width,
                int height,
                const RtScene& scene,
                const RtWavefront* wave,
                int sm_count) {

    if (!color || !visibility || width <= 0 || height <= 0 ||
        scene.count < 0 || scene.count > kMaxRtInstances ||
        (scene.ao && !scene.ao_accum) ||
        !scene.surfaces.materials || !scene.surfaces.textures ||
        (wave && (!wave->hits || !wave->counts ||
                  !wave->shadow_blocked || !wave->ao_escaped))) {
        LOG_ERROR("CUDA ray tracer: invalid arguments");
        return false;
    }

    auto* vis = reinterpret_cast<unsigned long long*>(visibility);

    const dim3 tile_threads(kTile, kTile);
    const dim3 tile_blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    if (wave == nullptr) {
        ray_trace_kernel<<<tile_blocks, tile_threads>>>(
            color, vis, width, height, scene);

        return check_launch("CUDA ray trace kernel launch");
    }

    // Wavefront: reset the hit count, then the three passes in order on
    // the default stream, so each pass sees the previous one's results
    if (!cuda_ok(cudaMemsetAsync(wave->counts, 0,
                                 sizeof(RtWavefrontCounts), 0),
                 "cudaMemsetAsync ray tracer hit count")) {
        return false;
    }

    rt_camera_kernel<<<tile_blocks, tile_threads>>>(
        width, height, scene, *wave);

    if (!check_launch("CUDA ray tracer camera pass launch")) {
        return false;
    }

    const unsigned int sms =
        sm_count > 0 ? static_cast<unsigned int>(sm_count) : 1u;

    // Plenty of threads: one per shadow or AO ray, many per SM
    rt_secondary_kernel<<<sms * 16, kThreads>>>(scene, *wave);

    if (!check_launch("CUDA ray tracer shadow/AO pass launch")) {
        return false;
    }

    rt_finish_kernel<<<sms * 4, kThreads>>>(color, vis, scene, *wave);

    return check_launch("CUDA ray tracer finish pass launch");
}

bool cuda_path_trace(
                uchar4* color,
                std::uint64_t* visibility,
                int width,
                int height,
                const RtScene& scene,
                const PtParams& params) {

    if (!color || !visibility || !params.accum ||
        width <= 0 || height <= 0 ||
        scene.count < 0 || scene.count > kMaxRtInstances ||
        (scene.count > 0 && (!scene.surfaces.materials ||
                             !scene.surfaces.textures)) ||
        params.max_bounces < 1) {
        LOG_ERROR("CUDA path tracer: invalid arguments");
        return false;
    }

    const dim3 tile_threads(kTile, kTile);
    const dim3 tile_blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    auto* vis = reinterpret_cast<unsigned long long*>(visibility);

    // Same stream, so the resolve sees the finished sums
    if (params.samples_to_add > 0) {
        path_trace_kernel<<<tile_blocks, tile_threads>>>(
            width, height, scene, params);

        if (!check_launch("CUDA path trace kernel launch")) {
            return false;
        }
    }

    pt_resolve_kernel<<<tile_blocks, tile_threads>>>(
        color, vis, width, height, scene, params);

    return check_launch("CUDA path trace resolve kernel launch");
}

bool cuda_resolve_ids(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height) {

    if (!color || !visibility || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA resolve: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    resolve_ids_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width, height);

    return check_launch("CUDA resolve kernel launch");
}

bool cuda_draw_grid(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                CudaGridParams params) {

    if (!color || !visibility ||
        width <= 0 || height <= 0) {
        LOG_ERROR("CUDA grid: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    draw_grid_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width,
        height,
        params);

    return check_launch("CUDA grid kernel launch");
}

} // namespace gfx
