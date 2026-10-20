#pragma once

#include "renderer/renderer.h"
#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_grid.h"
#include "renderer/opengl/gl_mesh.h"
#include "renderer/opengl/gl_timer.h"
#include "renderer/opengl/gl_triangle_demo.h"

namespace gfx {

class GLRenderer final : public IRenderer {
public:
    GLRenderer() = default;
    ~GLRenderer() override = default;

    GLRenderer(const GLRenderer&) = delete;
    GLRenderer& operator=(const GLRenderer&) = delete;

    const char* name() const override {
        return "OpenGL 4.6";
    }

    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    bool last_gpu_ms(double& ms) override;
    void set_vsync(bool enabled) override;
    MeshId create_mesh(const Mesh& mesh) override;

private:
    // Declaration order is intentional
    // C++ destroys members in reverse order
    GLContext m_context;
    GLGpuTimer m_gpu_timer;
    GLMeshRenderer m_meshes;
    TriangleDemo m_triangle;
    GridRenderer m_grid;
};

} // namespace gfx
