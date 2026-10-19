#pragma once

#include "renderer/renderer.h"
#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_presenter.h"
#include "renderer/cuda/cuda_gl_image.h"
#include "renderer/cuda/cuda_mesh.h"

namespace gfx {

class CudaRenderer final : public IRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer() override;

    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    const char* name() const override {
        return "CUDA software rasterizer";
    }

    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    void set_vsync(bool enabled) override;
    MeshId create_mesh(const Mesh& mesh) override;

private:
    bool resize_targets(int width, int height);

    // Declaration order is intentional
    GLContext m_context;
    GLPresenter m_presenter;
    CudaGLImage m_image;
    CudaMeshStore m_meshes;

    void* m_color = nullptr;
    int m_width = 0;
    int m_height = 0;
};

} // namespace gfx
