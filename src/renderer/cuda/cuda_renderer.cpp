#include "renderer/cuda/cuda_renderer.h"

#include <cuda_runtime.h>
#include "renderer/cuda/cuda_api.h"
#include "renderer/cuda/cuda_present.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

namespace gfx {

CudaRenderer::~CudaRenderer() {
    if (m_color != nullptr) {
        (void)cuda_ok(cudaFree(m_color), "cudaFree renderer color buffer");
        m_color = nullptr;
    }

    m_width = 0;
    m_height = 0;
}

bool CudaRenderer::init(void* native_window) {
    if (!m_context.create(native_window)) {
        return false;
    }

    // The OpenGL context must be current before querying interop devices
    if (!cuda_select_gl_device()) {
        return false;
    }

    if (!m_presenter.init()) {
        return false;
    }

    LOG_INFO("CUDA renderer initialized");
    return true;
}

bool CudaRenderer::resize_targets(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }

    // 1. Release CUDA's registration of the old texture
    m_image.unregister();

    // 2. Create the OpenGL texture at the new dimensions
    if (!m_presenter.resize(width, height)) {
        return false;
    }

    // 3. Register the new texture with CUDA
    if (!m_image.register_texture(m_presenter.texture())) {
        return false;
    }

    // 4. Release the previous linear CUDA pixel buffer
    if (m_color != nullptr) {
        if (!cuda_ok(
                cudaFree(m_color),
                "cudaFree old renderer color buffer")) {
            return false;
        }

        m_color = nullptr;
    }

    // 5. Allocate the new pixel buffer
    const std::size_t pixel_count =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_color),
                pixel_count * sizeof(uchar4)),
            "cudaMalloc renderer color buffer")) {
        m_color = nullptr;
        return false;
    }

    m_width = width;
    m_height = height;

    LOG_INFO("CUDA render targets resized: %dx%d", width, height);
    return true;
}

void CudaRenderer::render(const FrameData& frame) {
    if (frame.width <= 0 || frame.height <= 0) {
        return;
    }

    if (frame.width != m_width || frame.height != m_height) {
        if (!resize_targets(frame.width, frame.height)) {
            LOG_ERROR("Failed to resize CUDA render targets");
            return;
        }
    }

    // Generate the image entirely on the GPU
    if (!cuda_fill_gradient(static_cast<uchar4*>(m_color),
                            m_width, m_height)) {
        return;
    }

    // Copy the linear device buffer into the registered OpenGL texture
    if (!m_image.copy_from_device(
            m_color,
            m_width,
            m_height,
            static_cast<std::size_t>(m_width) * sizeof(uchar4))) {
        return;
    }

    // OpenGL is only responsible for presenting the completed image
    m_presenter.draw(m_width, m_height);
    m_context.swap_buffers();
}

void CudaRenderer::set_vsync(bool enabled) {
    m_context.set_vsync(enabled);
}

MeshId CudaRenderer::create_mesh(const Mesh&) {
    LOG_WARN("CUDA mesh rendering is not implemented yet (M5 Step 3)");
    return 0;
}

} // namespace gfx
