#include "renderer/cuda/cuda_raster.h"

#include "core/log.h"
#include "core/bvh.h"
#include "core/bvh4.h"
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
// space, so one t interpolates all of them correctly
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

    out[i].view_pos = make_float4(
        view_position.x,
        view_position.y,
        view_position.z,
        1.0f);

    if (mesh.normals) {
        const Vec3 view_normal =
            transform_direction(normal_to_view, mesh.normals[i]);

        out[i].view_normal = make_float4(
            view_normal.x,
            view_normal.y,
            view_normal.z,
            0.0f);
    } else {
        out[i].view_normal = make_float4(0, 0, 0, 0);
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
// screen/view_pos/view_normal are in the triangle's own vertex order
__device__ __forceinline__ void emit_triangle(
                const RasterWork& work,
                int width,
                int height,
                const float4 (&screen)[3],
                const float4 (&view_pos)[3],
                const float4 (&view_normal)[3]) {

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
            work, width, height, screen, view_pos, view_normal);
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
            work, width, height, screen, view_pos, view_normal);
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
// (1 = fully open, 0 = fully enclosed). The rasterizer passes 1 for both
__device__ __forceinline__ uchar4 lit_color(
                Vec3 n,
                Vec3 view_pos,
                Vec3 up_view,
                Vec3 key_dir,
                float key_visibility,
                float ambient_scale) {

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

    const Vec3 base{0.72f, 0.74f, 0.78f};

    const float r = fminf(
        base.x * (ambient.x + 0.75f * diff) + spec, 1.0f);
    const float g = fminf(
        base.y * (ambient.y + 0.75f * diff) + spec, 1.0f);
    const float b = fminf(
        base.z * (ambient.z + 0.75f * diff) + spec, 1.0f);

    return make_uchar4(to_byte(r), to_byte(g), to_byte(b), 255);
}

__global__ void shade_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height,
                RasterWork work,
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

    // Same layout as a projected vertex, already in setup order.
    // Shading does not need the clip position
    const float4 kUnused = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    const ProjectedVertex p[3] = {
        {rec.screen[0], rec.view_pos[0], rec.view_normal[0], kUnused},
        {rec.screen[1], rec.view_pos[1], rec.view_normal[1], kUnused},
        {rec.screen[2], rec.view_pos[2], rec.view_normal[2], kUnused}
    };

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

    const float inv_area = 1.0f / static_cast<float>(s.area);

    const float b0 = static_cast<float>(e0) * inv_area;
    const float b1 = static_cast<float>(e1) * inv_area;
    const float b2 = static_cast<float>(e2) * inv_area;

    // Perspective-correct interpolation
    const float q0 = b0 / p[0].screen.w;
    const float q1 = b1 / p[1].screen.w;
    const float q2 = b2 / p[2].screen.w;
    const float qsum = q0 + q1 + q2;

    if (!isfinite(qsum) || fabsf(qsum) < 1.0e-20f) {
        return;
    }

    const float l0 = q0 / qsum;
    const float l1 = q1 / qsum;
    const float l2 = q2 / qsum;

    Vec3 view_pos = {
        l0 * p[0].view_pos.x +
            l1 * p[1].view_pos.x + l2 * p[2].view_pos.x,
        l0 * p[0].view_pos.y +
            l1 * p[1].view_pos.y + l2 * p[2].view_pos.y,
        l0 * p[0].view_pos.z +
            l1 * p[1].view_pos.z + l2 * p[2].view_pos.z
    };

    Vec3 n = {
        l0 * p[0].view_normal.x +
            l1 * p[1].view_normal.x + l2 * p[2].view_normal.x,
        l0 * p[0].view_normal.y +
            l1 * p[1].view_normal.y + l2 * p[2].view_normal.y,
        l0 * p[0].view_normal.z +
            l1 * p[1].view_normal.z + l2 * p[2].view_normal.z
    };

    // Screen coordinates are y-down. CCW front faces therefore
    // have a negative original signed area
    const bool front_facing = s.signed_area < 0;

    if (!front_facing) {
        n = -n;
    }

    color[pixel] = lit_color(
        n, view_pos, up_view, key_light_view_dir(), 1.0f, 1.0f);
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

// A random direction above the surface with normal n (unit length),
// cosine-weighted: directions near the normal are picked more often,
// matching how much each direction contributes to ambient light
__device__ __forceinline__ Vec3 cosine_hemisphere(
                Vec3 n,
                float u1,
                float u2) {

    // Two tangents perpendicular to n (Duff et al. 2017, branchless)
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;

    const Vec3 t{1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x};
    const Vec3 bt{b, sign + n.y * n.y * a, -n.y};

    // A uniform point on a disc, lifted onto the hemisphere
    const float r = sqrtf(u1);
    const float phi = 6.2831853f * u2;

    return t * (r * cosf(phi)) +
           bt * (r * sinf(phi)) +
           n * sqrtf(fmaxf(0.0f, 1.0f - u1));
}

// ---------------------------------------------------------------------
// Ray tracer building blocks, shared by both ways of running it:
//  - the "megakernel": one thread does everything for one pixel
//  - the "wavefront": the work is split into passes, and each shadow or
//    AO ray gets its own thread (see cuda_ray_trace below)
// ---------------------------------------------------------------------

// Camera ray direction through the centre of pixel (x, y). Not unit
// length: its view-space z is exactly -1, so t is the view depth
__device__ __forceinline__ Vec3 camera_ray_dir(
                const RtScene& scene,
                int x,
                int y,
                int width,
                int height) {

    const float ndc_x = 2.0f * (x + 0.5f) / width - 1.0f;
    const float ndc_y = 1.0f - 2.0f * (y + 0.5f) / height;

    return scene.forward + ndc_x * scene.right + ndc_y * scene.up;
}

// The point a ray hit, with everything shading needs, in world space.
// Both normals are turned towards the side the camera sees
struct SurfacePoint {
    Vec3 position;
    Vec3 normal;     // interpolated shading normal (not unit length)
    Vec3 geometric;  // the triangle's own normal (unit length)
};

// Closest hit over every instance. The direction is moved into object
// space without normalizing it, so t means the same distance in both
// spaces and hits can be compared directly. Fills point on a hit
__device__ __forceinline__ bool closest_surface(
                const RtScene& scene,
                Vec3 dir,
                SurfacePoint& point) {

    float best_t = scene.t_max;
    int best_item = -1;
    BvhHit best{};

    for (int i = 0; i < scene.count; ++i) {
        const RtInstance& inst = scene.items[i];

        BvhHit hit;

        const Vec3 local_origin =
            transform_point(inst.world_to_object, scene.eye);
        const Vec3 local_dir =
            transform_direction(inst.world_to_object, dir);

        const bool found = scene.use_bvh4
            ? bvh4_trace(inst.mesh.bvh4_nodes,
                         inst.mesh.bvh_tri_indices,
                         inst.mesh.positions,
                         inst.mesh.indices,
                         local_origin, local_dir,
                         scene.t_min, best_t, hit)
            : bvh_trace(inst.mesh.bvh_nodes,
                        inst.mesh.bvh_tri_indices,
                        inst.mesh.positions,
                        inst.mesh.indices,
                        local_origin, local_dir,
                        scene.t_min, best_t, hit);

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

    point.position = scene.eye + best_t * dir;
    point.normal = transform_direction(inst.normal_to_world, n_object);
    point.geometric = normalize(
        transform_direction(inst.normal_to_world, geometric));

    if (!front_facing) {
        point.normal = -point.normal;
        point.geometric = -point.geometric;
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
        sun_blocked ? 0.0f : 1.0f, ao);

    // Same depth the rasterizer would have written, so the grid's
    // depth test treats both paths alike
    const Vec4 clip =
        scene.view_proj * make_vec4(point.position, 1.0f);

    float depth01 = (clip.z / clip.w) * 0.5f + 0.5f;
    depth01 = fminf(fmaxf(depth01, 0.0f), 1.0f);

    visibility[pixel] =
        (static_cast<unsigned long long>(__float_as_uint(depth01))
             << 32) |
        1ULL;
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
                scene, camera_ray_dir(scene, x, y, width, height),
                point)) {
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
                scene, camera_ray_dir(scene, x, y, width, height),
                point)) {
            const unsigned int index =
                atomicAdd(&wave.counts->hit_count, 1u);

            if (index < wave.capacity) {
                RtHit& hit = wave.hits[index];
                hit.pixel =
                    static_cast<unsigned int>(pixel_index(x, y, width));
                hit.position = point.position;
                hit.normal = point.normal;
                hit.geometric = point.geometric;

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

        finish_pixel(color, visibility, hit.pixel, scene, point,
                     wave.shadow_blocked[index] != 0,
                     wave.ao_escaped[index]);
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
                Vec3 up_view) {

    if (!color || !visibility || !work.records || !work.counts ||
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
        width, height, work, up_view);

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
