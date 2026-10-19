// The first two includes are deliberately ordered as required
#include <windows.h>
#include <cuda_gl_interop.h>

#include "renderer/cuda/cuda_gl_image.h"
#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

namespace gfx {

bool cuda_select_gl_device() {
    constexpr unsigned int kMaxDevices = 8;

    unsigned int count = 0;
    int devices[kMaxDevices] = {};

    const cudaError_t result = cudaGLGetDevices(
        &count,
        devices,
        kMaxDevices,
        cudaGLDeviceListAll
    );

    if (!cuda_ok(result, "cudaGLGetDevices")) {
        return false;
    }

    if (count == 0) {
        LOG_ERROR(
            "CUDA/OpenGL interop: no CUDA device is associated "
            "with the current OpenGL context"
        );
        return false;
    }

    if (!cuda_ok(
            cudaSetDevice(devices[0]),
            "cudaSetDevice for OpenGL interop")) {
        return false;
    }

    cudaDeviceProp properties{};

    if (!cuda_ok(
            cudaGetDeviceProperties(&properties, devices[0]),
            "cudaGetDeviceProperties for OpenGL interop")) {
        return false;
    }

    LOG_INFO(
        "CUDA/OpenGL interop device: %s (compute capability %d.%d)",
        properties.name,
        properties.major,
        properties.minor
    );

    return true;
}

CudaGLImage::~CudaGLImage() {
    unregister();
}

bool CudaGLImage::register_texture(unsigned int gl_texture) {
    if (gl_texture == 0) {
        LOG_ERROR("Cannot register OpenGL texture 0 with CUDA");
        return false;
    }

    if (m_resource != nullptr) {
        LOG_ERROR("CUDA/OpenGL image is already registered");
        return false;
    }

    const cudaError_t result = cudaGraphicsGLRegisterImage(
        &m_resource,
        static_cast<GLuint>(gl_texture),
        GL_TEXTURE_2D,
        cudaGraphicsRegisterFlagsWriteDiscard
    );

    if (!cuda_ok(result, "cudaGraphicsGLRegisterImage")) {
        m_resource = nullptr;
        return false;
    }

    return true;
}

bool CudaGLImage::copy_from_device(const void* d_pixels, int width,
                                   int height, std::size_t pitch_bytes) {
    if (m_resource == nullptr ||
        d_pixels == nullptr ||
        width <= 0 ||
        height <= 0 ||
        pitch_bytes < static_cast<std::size_t>(width) * 4) {
        LOG_ERROR("Invalid CUDA/OpenGL image copy parameters");
        return false;
    }

    cudaGraphicsResource* resource = m_resource;

    if (!cuda_ok(
            cudaGraphicsMapResources(1, &resource, 0),
            "cudaGraphicsMapResources")) {
        return false;
    }

    bool success = true;
    cudaArray_t array = nullptr;

    if (!cuda_ok(
            cudaGraphicsSubResourceGetMappedArray(
                &array,
                resource,
                0,
                0),
            "cudaGraphicsSubResourceGetMappedArray")) {
        success = false;
    }

    if (success) {
        const std::size_t row_bytes =
            static_cast<std::size_t>(width) * 4;

        if (!cuda_ok(
                cudaMemcpy2DToArray(
                    array,
                    0,
                    0,
                    d_pixels,
                    pitch_bytes,
                    row_bytes,
                    static_cast<std::size_t>(height),
                    cudaMemcpyDeviceToDevice),
                "cudaMemcpy2DToArray")) {
            success = false;
        }
    }

    // Always attempt to unmap after a successful map.
    if (!cuda_ok(
            cudaGraphicsUnmapResources(1, &resource, 0),
            "cudaGraphicsUnmapResources")) {
        success = false;
    }

    return success;
}

void CudaGLImage::unregister() {
    if (m_resource == nullptr) {
        return;
    }

    cudaGraphicsResource* resource = m_resource;
    m_resource = nullptr;

    (void)cuda_ok(
        cudaGraphicsUnregisterResource(resource),
        "cudaGraphicsUnregisterResource"
    );
}

} // namespace gfx
