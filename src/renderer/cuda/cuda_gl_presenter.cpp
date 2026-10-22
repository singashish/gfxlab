#include "renderer/cuda/cuda_gl_presenter.h"

#include "core/log.h"

#include <cstddef>

namespace gfx {

bool CudaGLPresenter::init(void* native_window) {
    if (!m_context.create(native_window)) {
        return false;
    }

    // The OpenGL context must be current before querying interop devices
    if (!cuda_select_gl_device()) {
        return false;
    }

    return m_presenter.init();
}

bool CudaGLPresenter::resize(int width, int height) {
    // Release the old registration before replacing the texture
    m_image.unregister();

    if (!m_presenter.resize(width, height)) {
        return false;
    }

    return m_image.register_texture(m_presenter.texture());
}

bool CudaGLPresenter::copy(const void* pixels, int width, int height) {
    return m_image.copy_from_device(
        pixels,
        width,
        height,
        static_cast<std::size_t>(width) * 4);
}

void CudaGLPresenter::present(int width, int height) {
    m_presenter.draw(width, height);
    m_context.swap_buffers();
}

void CudaGLPresenter::set_vsync(bool enabled) {
    m_context.set_vsync(enabled);
}

std::unique_ptr<ICudaPresenter> create_cuda_presenter() {
    return std::make_unique<CudaGLPresenter>();
}

} // namespace gfx
