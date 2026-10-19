#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#include "renderer/renderer.h"
#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_presenter.h"
#include "renderer/cuda/cuda_gl_image.h"
#include "renderer/cuda/cuda_mesh.h"
#include "renderer/cuda/cuda_raster.h"
#include "renderer/cuda/cuda_timer.h"

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
    bool ensure_scratch_capacity(std::uint32_t vertex_count);

    // Declaration order is intentional
    GLContext m_context;
    GLPresenter m_presenter;
    CudaGLImage m_image;
    CudaMeshStore m_meshes;

    CudaStageTimer m_stage_timer;

    void* m_color = nullptr;
    void* m_visibility = nullptr;

    ProjectedVertex* m_scratch = nullptr;
    std::uint32_t m_scratch_capacity = 0;

    int m_width = 0;
    int m_height = 0;
};

} // namespace gfx
