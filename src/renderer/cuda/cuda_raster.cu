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

__global__ void clear_kernel(uchar4* color, int width,
                             int height, uchar4 value) {
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

    if (i >= count) {
        return;
    }

    visibility[i] = ~0ULL;
}

__global__ void transform_vertices_kernel(DeviceMesh mesh,
                Mat4 mvp, int width, int height, float4* out) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= mesh.vertex_count) {
        return;
    }

    const Vec4 clip = mvp * make_vec4(mesh.positions[i], 1.0f);

    // Preserve an invalid w so rasterization can reject triangles
    // crossing or behind the camera. Never divide by invalid w
    if (!isfinite(clip.w) || clip.w <= 0.0f) {
        out[i] = make_float4(0.0f, 0.0f, 0.0f, clip.w);
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

    out[i] = make_float4(screen_x, screen_y, depth01, clip.w);
}

__device__ __forceinline__ int clamp_int(
    int value, int low, int high
) {
    return value < low ? low : (value > high ? high : value);
}

__device__ __forceinline__ bool edge_inside(
    long long edge, bool top_left
) {
    return edge + (top_left ? 0LL : -1LL) >= 0;
}

__global__ void rasterize_triangles_kernel(
    unsigned long long* visibility,
    int width,
    int height,
    DeviceMesh mesh,
    const float4* vertices,
    std::uint32_t triangle_base
) {
    const unsigned int tri = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int triangle_count = mesh.index_count / 3;

    if (tri >= triangle_count) {
        return;
    }

    // Reserve zero as the invalid triangle ID
    if (tri >= UINT_MAX - triangle_base) {
        return;
    }

    const std::uint32_t i0 = mesh.indices[tri * 3 + 0];
    const std::uint32_t i1 = mesh.indices[tri * 3 + 1];
    const std::uint32_t i2 = mesh.indices[tri * 3 + 2];

    float4 v[3] = {
        vertices[i0],
        vertices[i1],
        vertices[i2]
    };

    // Near-plane clipping is not implemented in this milestone
    if (!isfinite(v[0].w) || !isfinite(v[1].w) ||
        !isfinite(v[2].w) ||
        v[0].w <= 0.0f || v[1].w <= 0.0f || v[2].w <= 0.0f) {
        return;
    }

    // Reject non-finite or extreme screen coordinates before
    // converting them to fixed-point integers
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        if (!isfinite(v[i].x) || !isfinite(v[i].y) ||
            !isfinite(v[i].z) ||
            fabsf(v[i].x) > kMaxScreenCoordinate ||
            fabsf(v[i].y) > kMaxScreenCoordinate) {
            return;
        }
    }

    int sx[3];
    int sy[3];

    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        sx[i] = snap_subpixel(v[i].x);
        sy[i] = snap_subpixel(v[i].y);
    }

    long long area = edge_function(
        sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]
    );

    if (area == 0) {
        return;
    }

    // Make the signed area positive, keeping depth associated
    // with its corresponding vertex
    if (area < 0) {
        const float4 tmp_v = v[1];
        v[1] = v[2];
        v[2] = tmp_v;

        const int tmp_x = sx[1];
        sx[1] = sx[2];
        sx[2] = tmp_x;

        const int tmp_y = sy[1];
        sy[1] = sy[2];
        sy[2] = tmp_y;

        area = -area;
    }

    const int min_sx = min(sx[0], min(sx[1], sx[2]));
    const int max_sx = max(sx[0], max(sx[1], sx[2]));
    const int min_sy = min(sy[0], min(sy[1], sy[2]));
    const int max_sy = max(sy[0], max(sy[1], sy[2]));

    // Convert the fixed-point bounds to pixel bounds.
    int min_x = static_cast<int>(floorf(min_sx / 16.0f));
    int max_x = static_cast<int>(ceilf(max_sx / 16.0f));
    int min_y = static_cast<int>(floorf(min_sy / 16.0f));
    int max_y = static_cast<int>(ceilf(max_sy / 16.0f));

    min_x = clamp_int(min_x, 0, width - 1);
    max_x = clamp_int(max_x, 0, width - 1);
    min_y = clamp_int(min_y, 0, height - 1);
    max_y = clamp_int(max_y, 0, height - 1);

    if (min_x > max_x || min_y > max_y) {
        return;
    }

    const bool tl0 = is_top_left(sx[1], sy[1], sx[2], sy[2]);
    const bool tl1 = is_top_left(sx[2], sy[2], sx[0], sy[0]);
    const bool tl2 = is_top_left(sx[0], sy[0], sx[1], sy[1]);

    const float inv_area = 1.0f / static_cast<float>(area);

    for (int y = min_y; y <= max_y; ++y) {
        const int py = y * 16 + 8;

        for (int x = min_x; x <= max_x; ++x) {
            const int px = x * 16 + 8;

            const long long e0 = edge_function(
                sx[1], sy[1], sx[2], sy[2], px, py
            );
            const long long e1 = edge_function(
                sx[2], sy[2], sx[0], sy[0], px, py
            );
            const long long e2 = edge_function(
                sx[0], sy[0], sx[1], sy[1], px, py
            );

            if (!edge_inside(e0, tl0) ||
                !edge_inside(e1, tl1) ||
                !edge_inside(e2, tl2)) {
                continue;
            }

            // Use unbiased edge values for interpolation.
            const float b0 = static_cast<float>(e0) * inv_area;
            const float b1 = static_cast<float>(e1) * inv_area;
            const float b2 = static_cast<float>(e2) * inv_area;

            float z = b0 * v[0].z + b1 * v[1].z + b2 * v[2].z;

            if (!isfinite(z)) {
                continue;
            }

            z = fminf(1.0f, fmaxf(0.0f, z));

            // Non-negative IEEE-754 float bit patterns are ordered
            // in the same way as their unsigned integer values.
            const unsigned int depth_bits = __float_as_uint(z);
            const unsigned int id = triangle_base + tri + 1;

            const unsigned long long packed =
                (static_cast<unsigned long long>(depth_bits) << 32) |
                static_cast<unsigned long long>(id);

            const std::size_t index = pixel_index(x, y, width);

            atomicMin(&visibility[index], packed);
        }
    }
}

__global__ void resolve_ids_kernel(uchar4* color,
        const unsigned long long* visibility, int width, int height) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t index = pixel_index(x, y, width);
    const unsigned long long packed = visibility[index];

    if (packed == ~0ULL) {
        // Preserve the background produced by cuda_clear
        return;
    }

    const unsigned int id = static_cast<unsigned int>(packed);

    // A deterministic, bright-enough per-triangle diagnostic colour
    const unsigned int hash = id * 0x9E3779B9u;

    const unsigned char r =
        static_cast<unsigned char>(96u + (hash & 0x9Fu));
    const unsigned char g =
        static_cast<unsigned char>(96u + ((hash >> 8) & 0x9Fu));
    const unsigned char b =
        static_cast<unsigned char>(96u + ((hash >> 16) & 0x9Fu));

    color[index] = make_uchar4(r, g, b, 255);
}

} // namespace

bool cuda_clear(uchar4* color, int width, int height, uchar4 value) {
    if (!color || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA clear: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile
    );

    clear_kernel<<<blocks, threads>>>(color, width, height, value);
    return check_launch("CUDA clear kernel launch");
}

bool cuda_clear_visibility(std::uint64_t* visibility,
                           int width, int height) {
    if (!visibility || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA visibility clear: invalid arguments");
        return false;
    }

    const std::size_t count =
        static_cast<std::size_t>(width) * height;

    const unsigned int blocks = static_cast<unsigned int>(
        (count + kThreads - 1) / kThreads
    );

    clear_visibility_kernel<<<blocks, kThreads>>>(
        reinterpret_cast<unsigned long long*>(visibility), count
    );

    return check_launch("CUDA visibility clear kernel launch");
}

bool cuda_transform_vertices(const DeviceMesh& mesh, Mat4 mvp,
                             int width, int height, float4* out) {
    if (!mesh.positions || !out || mesh.vertex_count == 0 ||
        width <= 0 || height <= 0) {
        LOG_ERROR("CUDA vertex transform: invalid arguments");
        return false;
    }

    const unsigned int blocks =
        (mesh.vertex_count + kThreads - 1) / kThreads;

    transform_vertices_kernel<<<blocks, kThreads>>>(
        mesh, mvp, width, height, out
    );

    return check_launch("CUDA vertex transform kernel launch");
}

bool cuda_rasterize_triangles(std::uint64_t* visibility, int width,
            int height, const DeviceMesh& mesh, const float4* vertices,
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
        width,
        height,
        mesh,
        vertices,
        triangle_base
    );

    return check_launch("CUDA triangle rasterizer kernel launch");
}

bool cuda_resolve_ids(uchar4* color, const std::uint64_t* visibility,
                      int width, int height) {
    if (!color || !visibility || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA resolve: invalid arguments");
        return false;
    }

    const dim3 threads(kTile, kTile);
    const dim3 blocks(
        (width + kTile - 1) / kTile,
        (height + kTile - 1) / kTile
    );

    resolve_ids_kernel<<<blocks, threads>>>(
        color,
        reinterpret_cast<const unsigned long long*>(visibility),
        width,
        height
    );

    return check_launch("CUDA resolve kernel launch");
}

} // namespace gfx
