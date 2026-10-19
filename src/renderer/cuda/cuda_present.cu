#include "renderer/cuda/cuda_present.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

namespace gfx {

namespace {

__global__ void fill_gradient_kernel(uchar4* pixels,
                                     int width, int height) {
    const int x =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    const int y =
        static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);

    if (x >= width || y >= height) {
        return;
    }

    const unsigned char red = static_cast<unsigned char>(
        255.0f * static_cast<float>(x) / static_cast<float>(width)
    );

    const unsigned char green = static_cast<unsigned char>(
        255.0f * static_cast<float>(y) / static_cast<float>(height)
    );

    const unsigned char blue = 128;
    const unsigned char alpha = 255;

    pixels[static_cast<std::size_t>(y) *
               static_cast<std::size_t>(width) +
           static_cast<std::size_t>(x)] =
        make_uchar4(red, green, blue, alpha);
}

} // namespace

bool cuda_fill_gradient(uchar4* pixels, int width, int height) {
    if (pixels == nullptr || width <= 0 || height <= 0) {
        LOG_ERROR("Invalid CUDA gradient parameters");
        return false;
    }

    constexpr int kBlockWidth = 16;
    constexpr int kBlockHeight = 16;

    const dim3 threads(kBlockWidth, kBlockHeight);

    const dim3 blocks(
        (static_cast<unsigned int>(width) + kBlockWidth - 1) /
            kBlockWidth,
        (static_cast<unsigned int>(height) + kBlockHeight - 1) /
            kBlockHeight
    );

    fill_gradient_kernel<<<blocks, threads>>>(
        pixels,
        width,
        height
    );

    // Check 1: kernel launch configuration and submission
    if (!cuda_ok(
            cudaGetLastError(),
            "fill_gradient_kernel launch")) {
        return false;
    }

    // Check 2: kernel execution completion and runtime errors
    if (!cuda_ok(
            cudaDeviceSynchronize(),
            "fill_gradient_kernel execution")) {
        return false;
    }

    return true;
}

} // namespace gfx
