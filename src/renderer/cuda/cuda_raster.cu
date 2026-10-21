#include "renderer/cuda/cuda_raster.h"

#include "core/log.h"
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

__device__ __forceinline__ bool edge_inside(
    long long edge, bool top_left) {
    return edge + (top_left ? 0LL : -1LL) >= 0;
}

// One source of truth for rasterization and shading
// sx/sy and order are normalized to positive area, while
// signed_area retains the original winding
__device__ __forceinline__ bool setup_triangle(
                const ProjectedVertex* vertices,
                std::uint32_t i0,
                std::uint32_t i1,
                std::uint32_t i2,
                RasterTriangleSetup& result) {

    const std::uint32_t indices[3] = {i0, i1, i2};

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        const float4 p = vertices[indices[i]].screen;

        if (!isfinite(p.x) || !isfinite(p.y) ||
            !isfinite(p.z) || !isfinite(p.w) ||
            p.w <= 0.0f ||
            fabsf(p.x) > kMaxScreenCoordinate ||
            fabsf(p.y) > kMaxScreenCoordinate) {
            return false;
        }
    }

    result = setup_raster_triangle(
        vertices[i0].screen.x, vertices[i0].screen.y,
        vertices[i1].screen.x, vertices[i1].screen.y,
        vertices[i2].screen.x, vertices[i2].screen.y);

    return result.valid;
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

    const Vec4 clip = mvp * make_vec4(position, 1.0f);

    if (!isfinite(clip.w) || clip.w <= 0.0f) {
        out[i].screen = make_float4(0, 0, 0, clip.w);
        return;
    }

    const float inv_w = 1.0f / clip.w;
    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;
    const float ndc_z = clip.z * inv_w;

    const float screen_x = (ndc_x * 0.5f + 0.5f) * width;
    const float screen_y =
        (1.0f - (ndc_y * 0.5f + 0.5f)) * height;
    const float depth01 = ndc_z * 0.5f + 0.5f;

    out[i].screen = make_float4(
        screen_x, screen_y, depth01, clip.w);
}

__global__ void draw_edges_kernel(
                unsigned long long* visibility,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices) {

    const unsigned int edge =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int edge_count =
        mesh.edge_index_count / 2;

    if (edge >= edge_count) {
        return;
    }

    const std::uint32_t i0 = mesh.edges[edge * 2];
    const std::uint32_t i1 = mesh.edges[edge * 2 + 1];

    const float4 a = vertices[i0].screen;
    const float4 b = vertices[i1].screen;

    // Step 5 will handle edges crossing the near plane.
    // For now, reject invalid endpoints.
    if (!isfinite(a.x) || !isfinite(a.y) ||
        !isfinite(a.z) || !isfinite(a.w) ||
        !isfinite(b.x) || !isfinite(b.y) ||
        !isfinite(b.z) || !isfinite(b.w) ||
        a.w <= 0.0f || b.w <= 0.0f) {
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

__global__ void setup_triangles_kernel(
                RasterWork work,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices) {

    const unsigned int tri =
        blockIdx.x * blockDim.x + threadIdx.x;

    const unsigned int triangle_count = mesh.index_count / 3;

    if (tri >= triangle_count) {
        return;
    }

    const std::uint32_t indices[3] = {
        mesh.indices[tri * 3 + 0],
        mesh.indices[tri * 3 + 1],
        mesh.indices[tri * 3 + 2]
    };

    RasterTriangleSetup s{};

    if (!setup_triangle(
            vertices, indices[0], indices[1], indices[2], s)) {
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
        const ProjectedVertex& v = vertices[indices[s.order[k]]];
        r.screen[k] = v.screen;
        r.view_pos[k] = v.view_pos;
        r.view_normal[k] = v.view_normal;
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

    // Same layout as a projected vertex, already in setup order
    const ProjectedVertex p[3] = {
        {rec.screen[0], rec.view_pos[0], rec.view_normal[0]},
        {rec.screen[1], rec.view_pos[1], rec.view_normal[1]},
        {rec.screen[2], rec.view_pos[2], rec.view_normal[2]}
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

    n = normalize(n);
    const Vec3 v = normalize(-view_pos);
    const Vec3 key_dir = normalize(Vec3{-0.5f, 0.6f, 0.7f});

    const float diff = fmaxf(dot(n, key_dir), 0.0f);

    const float hemi = dot(n, up_view) * 0.5f + 0.5f;

    const Vec3 ambient_low{0.10f, 0.10f, 0.12f};
    const Vec3 ambient_high{0.30f, 0.32f, 0.36f};

    const Vec3 ambient = {
        ambient_low.x + (ambient_high.x - ambient_low.x) * hemi,
        ambient_low.y + (ambient_high.y - ambient_low.y) * hemi,
        ambient_low.z + (ambient_high.z - ambient_low.z) * hemi
    };

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

    color[pixel] = make_uchar4(
        to_byte(r), to_byte(g), to_byte(b), 255);
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
        work, width, height, mesh, vertices);

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
        vertices);

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
