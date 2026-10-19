#include "renderer/cuda/cuda_raster.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

#include <cstddef>

namespace gfx {

namespace {

constexpr int kThreadsPerBlock = 256;

__global__ void clear_kernel(uchar4* color, int width,
                             int height, uchar4 value) {
    const int x =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int y =
        static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height) {
        return;
    }

    const std::size_t index =
        static_cast<std::size_t>(y) * width + x;

    color[index] = value;
}

__global__ void draw_points_kernel(uchar4* color, int width,
                            int height, DeviceMesh mesh, Mat4 mvp) {
    const unsigned int i =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= mesh.vertex_count) {
        return;
    }

    const Vec3 position = mesh.positions[i];

    // Model -> world -> view -> clip coordinates
    const Vec4 clip = mvp * make_vec4(position, 1.0f);

    // Reject vertices on or behind the camera
    if (clip.w <= 0.0f) {
        return;
    }

    // Perspective divide
    const float inv_w = 1.0f / clip.w;

    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;
    const float ndc_z = clip.z * inv_w;

    // OpenGL NDC clipping: all three axes are [-1, 1]
    if (ndc_x < -1.0f || ndc_x > 1.0f ||
        ndc_y < -1.0f || ndc_y > 1.0f ||
        ndc_z < -1.0f || ndc_z > 1.0f) {
        return;
    }

    // NDC -> screen pixels
    const float screen_x =
        (ndc_x * 0.5f + 0.5f) * width;

    // Flip Y because CUDA's color buffer starts at the top
    const float screen_y =
        (1.0f - (ndc_y * 0.5f + 0.5f)) * height;

    int px = static_cast<int>(floorf(screen_x));
    int py = static_cast<int>(floorf(screen_y));

    px = max(0, min(px, width - 1));
    py = max(0, min(py, height - 1));

    const std::size_t pixel_index =
        static_cast<std::size_t>(py) * width + px;

    color[pixel_index] = make_uchar4(255, 255, 255, 255);
}

bool check_kernel(const char* name) {
    if (!cuda_ok(cudaGetLastError(), name)) {
        return false;
    }

    if (!cuda_ok(cudaDeviceSynchronize(), name)) {
        return false;
    }

    return true;
}

} // namespace

bool cuda_clear(uchar4* color, int width, int height, uchar4 value) {
    if (!color || width <= 0 || height <= 0) {
        LOG_ERROR("CUDA clear: invalid arguments");
        return false;
    }

    const dim3 threads(16, 16);

    const dim3 blocks(
        (width + 15) / 16,
        (height + 15) / 16
    );

    clear_kernel<<<blocks, threads>>>(
        color, width, height, value
    );

    return check_kernel("CUDA clear kernel");
}

bool cuda_draw_points(uchar4* color, int width, int height,
                      const DeviceMesh& mesh, Mat4 mvp) {
    if (!color || !mesh.positions ||
        width <= 0 || height <= 0 ||
        mesh.vertex_count == 0) {
        LOG_ERROR("CUDA draw points: invalid arguments");
        return false;
    }

    const unsigned int blocks =
        (mesh.vertex_count + kThreadsPerBlock - 1) /
        kThreadsPerBlock;

    draw_points_kernel<<<blocks, kThreadsPerBlock>>>(
        color, width, height, mesh, mvp
    );

    return check_kernel("CUDA draw points kernel");
}

} // namespace gfx
