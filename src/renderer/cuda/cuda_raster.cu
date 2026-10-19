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

__global__ void rasterize_triangles_kernel(
                unsigned long long* visibility,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base) {

    const unsigned int tri = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int triangle_count = mesh.index_count / 3;

    if (tri >= triangle_count || tri >= UINT_MAX - triangle_base) {
        return;
    }

    const std::uint32_t i0 = mesh.indices[tri * 3 + 0];
    const std::uint32_t i1 = mesh.indices[tri * 3 + 1];
    const std::uint32_t i2 = mesh.indices[tri * 3 + 2];

    RasterTriangleSetup s{};

    if (!setup_triangle(vertices, i0, i1, i2, s)) {
        return; // use continue instead if inside the pixel loop
    }

    const std::uint32_t indices[3] = {i0, i1, i2};

    float4 v[3] = {
        vertices[indices[s.order[0]]].screen,
        vertices[indices[s.order[1]]].screen,
        vertices[indices[s.order[2]]].screen
    };

    const int min_sx = min(s.sx[0], min(s.sx[1], s.sx[2]));
    const int max_sx = max(s.sx[0], max(s.sx[1], s.sx[2]));
    const int min_sy = min(s.sy[0], min(s.sy[1], s.sy[2]));
    const int max_sy = max(s.sy[0], max(s.sy[1], s.sy[2]));

    int min_x = clamp_int(
        static_cast<int>(floorf(min_sx / 16.0f)), 0, width - 1);
    int max_x = clamp_int(
        static_cast<int>(ceilf(max_sx / 16.0f)), 0, width - 1);
    int min_y = clamp_int(
        static_cast<int>(floorf(min_sy / 16.0f)), 0, height - 1);
    int max_y = clamp_int(
        static_cast<int>(ceilf(max_sy / 16.0f)), 0, height - 1);

    if (min_x > max_x || min_y > max_y) {
        return;
    }

    const float inv_area = 1.0f / static_cast<float>(s.area);

    for (int y = min_y; y <= max_y; ++y) {
        const int py = y * 16 + 8;

        for (int x = min_x; x <= max_x; ++x) {
            const int px = x * 16 + 8;

            const long long e0 = edge_function(
                s.sx[1], s.sy[1], s.sx[2], s.sy[2], px, py);
            const long long e1 = edge_function(
                s.sx[2], s.sy[2], s.sx[0], s.sy[0], px, py);
            const long long e2 = edge_function(
                s.sx[0], s.sy[0], s.sx[1], s.sy[1], px, py);

            if (!raster_triangle_covers_sample(s, px, py)) {
                continue;
            }

            const float b0 = static_cast<float>(e0) * inv_area;
            const float b1 = static_cast<float>(e1) * inv_area;
            const float b2 = static_cast<float>(e2) * inv_area;

            const float z =
                b0 * v[0].z + b1 * v[1].z + b2 * v[2].z;

            if (!isfinite(z) || z < 0.0f || z > 1.0f) {
                continue;
            }

            const unsigned int depth_bits = __float_as_uint(z);
            const unsigned int id = triangle_base + tri + 1;

            const unsigned long long packed =
                (static_cast<unsigned long long>(depth_bits) << 32) |
                static_cast<unsigned long long>(id);

            atomicMin(
                &visibility[pixel_index(x, y, width)], packed);
        }
    }
}

__device__ __forceinline__ unsigned char to_byte(float c) {
    int value = __float2int_rn(c * 255.0f);
    value = max(0, min(255, value));
    return static_cast<unsigned char>(value);
}

__global__ void shade_item_kernel(
                uchar4* color,
                const unsigned long long* visibility,
                int width,
                int height,
                DeviceMesh mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base,
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

    const std::uint32_t global_tri = id - 1;
    const std::uint32_t triangle_count = mesh.index_count / 3;

    // The winning triangle may belong to a different draw item
    if (global_tri < triangle_base ||
        global_tri - triangle_base >= triangle_count) {
        return;
    }

    const std::uint32_t tri = global_tri - triangle_base;

    const std::uint32_t i0 = mesh.indices[tri * 3 + 0];
    const std::uint32_t i1 = mesh.indices[tri * 3 + 1];
    const std::uint32_t i2 = mesh.indices[tri * 3 + 2];

    RasterTriangleSetup s{};

    if (!setup_triangle(vertices, i0, i1, i2, s)) {
        return; // use continue instead if inside the pixel loop
    }

    const std::uint32_t indices[3] = {i0, i1, i2};

    const ProjectedVertex p[3] = {
        vertices[indices[s.order[0]]],
        vertices[indices[s.order[1]]],
        vertices[indices[s.order[2]]]
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

bool cuda_rasterize_triangles(
                std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base) {

    if (!visibility || !mesh.indices || !vertices ||
        width <= 0 || height <= 0 ||
        mesh.index_count < 3 || mesh.index_count % 3 != 0) {
        LOG_ERROR("CUDA triangle rasterizer: invalid arguments");
        return false;
    }

    const unsigned int triangle_count = mesh.index_count / 3;
    const unsigned int blocks =
        (triangle_count + kThreads - 1) / kThreads;

    rasterize_triangles_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility),
        width, height, mesh, vertices, triangle_base);

    return check_launch("CUDA triangle rasterizer kernel launch");
}

bool cuda_shade_item(
                uchar4* color,
                const std::uint64_t* visibility,
                int width,
                int height,
                const DeviceMesh& mesh,
                const ProjectedVertex* vertices,
                std::uint32_t triangle_base,
                Vec3 up_view) {

    if (!color || !visibility || !mesh.indices || !vertices ||
        width <= 0 || height <= 0 ||
        mesh.index_count == 0 || mesh.index_count % 3 != 0) {
        LOG_ERROR("CUDA shade item: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile);

    shade_item_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width, height, mesh, vertices, triangle_base, up_view);

    return check_launch("CUDA shade item kernel launch");
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

} // namespace gfx
