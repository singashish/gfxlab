#pragma once

#include "renderer/cuda/cuda_presenter.h"
#include "renderer/cuda/cuda_gl_image.h"
#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_presenter.h"

namespace gfx {

// Presents through CUDA/OpenGL interop: CUDA copies the frame into an
// OpenGL texture it has registered, then OpenGL draws that texture as a
// full-screen triangle and swaps buffers
class CudaGLPresenter final : public ICudaPresenter {
public:
    CudaGLPresenter() = default;
    ~CudaGLPresenter() override = default;

    CudaGLPresenter(const CudaGLPresenter&) = delete;
    CudaGLPresenter& operator=(const CudaGLPresenter&) = delete;

    const char* name() const override {
        return "OpenGL interop";
    }

    bool init(void* native_window) override;
    bool resize(int width, int height) override;
    bool copy(const void* pixels, int width, int height) override;
    void present(int width, int height) override;
    void set_vsync(bool enabled) override;

private:
    // Declaration order is intentional: the texture registration is
    // released first and the OpenGL context last
    GLContext m_context;
    GLPresenter m_presenter;
    CudaGLImage m_image;
};

} // namespace gfx
