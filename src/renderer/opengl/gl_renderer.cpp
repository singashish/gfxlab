#include "renderer/opengl/gl_renderer.h"
#include "renderer/opengl/gl_api.h"

#include <cmath>

namespace gfx {

bool GLRenderer::init(void* native_window) {
    if (!m_context.create(native_window)) {
        return false;
    }

    // GPU timing is optional; rendering must still work without it
    (void)m_gpu_timer.init();

    if (!m_meshes.init()) {
        return false;
    }

    if (!m_triangle.init()) {
        return false;
    }

    if (!m_grid.init()) {
        return false;
    }

    return true;
}

void GLRenderer::set_vsync(bool enabled) {
    m_context.set_vsync(enabled);
}

void GLRenderer::render(const FrameData& frame) {
    if (frame.width <= 0 || frame.height <= 0) {
        return;
    }

    const float aspect =
        static_cast<float>(frame.width) /
        static_cast<float>(frame.height);

    const Mat4 projection = perspective(
        frame.fov_y,
        aspect,
        frame.near_plane,
        frame.far_plane,
        ClipSpace::CLIP_SPACE_OPENGL
    );

    const Mat4 view_proj = projection * frame.view;

    // Start GPU timing
    m_gpu_timer.begin();

    // Configure the framebuffer and clear it
    glViewport(0, 0, frame.width, frame.height);
    glClearColor(0.10f, 0.11f, 0.13f, 1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Draw opaque geometry first
    // The demo triangles remain solid; wireframe applies to loaded meshes
    if (frame.draw_list.empty()) {
        m_triangle.draw(view_proj, frame.demo_spin_angle);
    } else if (frame.wireframe) {
        m_meshes.draw_wireframe(frame.draw_list, view_proj);
    } else {
        m_meshes.draw(frame.draw_list, view_proj, frame.view);
    }

    // Build OpenGL-specific grid parameters
    const float tan_half = std::tan(frame.fov_y * 0.5f);

    GridParams gp;
    gp.view_proj = view_proj;
    gp.eye = frame.eye;
    gp.forward = frame.basis.forward;
    gp.right = frame.basis.right * (tan_half * aspect);
    gp.up = frame.basis.up * tan_half;
    gp.fade_end = frame.grid_fade_end;

    // Draw the grid after the opaque objects, as before
    m_grid.draw(gp);

    // Stop GPU timing before presentation and vsync
    m_gpu_timer.end();

    // Present the completed frame
    m_context.swap_buffers();
}

bool GLRenderer::last_gpu_ms(double& ms) {
    return m_gpu_timer.take_latest(ms);
}

MeshId GLRenderer::create_mesh(const Mesh& mesh) {
    return m_meshes.add(mesh);
}

} // namespace gfx
