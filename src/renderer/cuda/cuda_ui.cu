#include "renderer/cuda/cuda_ui.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

namespace gfx {

namespace {

constexpr int kUiTile = 16;

// Triangles cached in shared memory per pass: one per thread
constexpr int kUiChunk = kUiTile * kUiTile;

// One thread per pixel of the UI's bounding rectangle. Every pixel walks
// all triangles in order, because blending is order dependent: the panel
// background must be under its text. The triangles are read in chunks
// into shared memory, so each block fetches every triangle once instead
// of each thread fetching it from global memory
__global__ void ui_composite_kernel(uchar4* color,
                                    int width,
                                    const UiRasterTri* tris,
                                    int count,
                                    const std::uint8_t* atlas,
                                    int atlas_width,
                                    int atlas_height,
                                    UiRasterBounds bounds) {
    __shared__ UiRasterTri cache[kUiChunk];

    const int x = bounds.x0 + static_cast<int>(blockIdx.x) * kUiTile +
                  static_cast<int>(threadIdx.x);
    const int y = bounds.y0 + static_cast<int>(blockIdx.y) * kUiTile +
                  static_cast<int>(threadIdx.y);
    const int thread =
        static_cast<int>(threadIdx.y) * kUiTile + static_cast<int>(threadIdx.x);

    const bool inside = x <= bounds.x1 && y <= bounds.y1;

    const std::size_t index =
        static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
        static_cast<std::size_t>(x);

    uchar4 pixel = make_uchar4(0, 0, 0, 255);
    float rgb[3] = {0.0f, 0.0f, 0.0f};

    if (inside) {
        pixel = color[index];
        rgb[0] = static_cast<float>(pixel.x);
        rgb[1] = static_cast<float>(pixel.y);
        rgb[2] = static_cast<float>(pixel.z);
    }

    for (int base = 0; base < count; base += kUiChunk) {
        const int n = min(kUiChunk, count - base);

        // Every thread takes part in loading, even those outside the
        // bounds, so the barriers are reached by the whole block
        __syncthreads();

        if (thread < n) {
            cache[thread] = tris[base + thread];
        }

        __syncthreads();

        if (inside) {
            for (int i = 0; i < n; ++i) {
                ui_blend_triangle(cache[i], x, y, atlas,
                                  atlas_width, atlas_height, rgb);
            }
        }
    }

    if (inside) {
        color[index] = make_uchar4(
            static_cast<unsigned char>(rgb[0]),
            static_cast<unsigned char>(rgb[1]),
            static_cast<unsigned char>(rgb[2]),
            pixel.w);
    }
}

} // namespace

bool cuda_draw_ui(uchar4* color,
                  int width,
                  int height,
                  const UiRasterTri* tris,
                  int count,
                  const std::uint8_t* atlas,
                  int atlas_width,
                  int atlas_height,
                  UiRasterBounds bounds) {
    if (color == nullptr || tris == nullptr || atlas == nullptr ||
        width <= 0 || height <= 0 || count <= 0 ||
        atlas_width <= 0 || atlas_height <= 0 ||
        bounds.x0 < 0 || bounds.y0 < 0 ||
        bounds.x1 >= width || bounds.y1 >= height ||
        bounds.x0 > bounds.x1 || bounds.y0 > bounds.y1) {
        LOG_ERROR("CUDA UI: invalid arguments");
        return false;
    }

    const int span_x = bounds.x1 - bounds.x0 + 1;
    const int span_y = bounds.y1 - bounds.y0 + 1;

    const dim3 threads(kUiTile, kUiTile);
    const dim3 blocks((span_x + kUiTile - 1) / kUiTile,
                      (span_y + kUiTile - 1) / kUiTile);

    ui_composite_kernel<<<blocks, threads>>>(
        color, width, tris, count, atlas, atlas_width, atlas_height, bounds);

    return cuda_ok(cudaGetLastError(), "CUDA UI kernel launch");
}

} // namespace gfx
