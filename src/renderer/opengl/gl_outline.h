#pragma once

#include "core/vec3.h"

namespace gfx {

// The selection outline (see kOutlineRadius in renderer.h): the marked
// objects are rendered into an offscreen mask (colour = 0 / 0.5 / 1 for
// unmarked / selected / active, plus a depth buffer that holds the whole
// scene), then a fullscreen pass draws the outline into the window's
// framebuffer. Usage per frame:
//
//     if (outline.begin(w, h)) {
//         meshes.draw_mask(...);             // into the mask
//         outline.finish(near, far, w, h);   // back to the window
//     }
class GLOutline {
public:
    GLOutline() = default;
    ~GLOutline();

    GLOutline(const GLOutline&) = delete;
    GLOutline& operator=(const GLOutline&) = delete;

    bool init();
    void shutdown();

    // Binds the mask framebuffer (creating or resizing it when needed),
    // clears it and sets the state for the mask pass. False: the outline
    // is skipped this frame and the window's framebuffer is still bound
    bool begin(int width, int height);

    // Rebinds the window's framebuffer and draws the outline over it.
    // near/far are the projection's clip distances (the pass compares
    // distances from the camera, not raw depth values)
    void finish(float near_plane, float far_plane, int width, int height);

private:
    bool create_targets(int width, int height);
    void destroy_targets();

    unsigned int m_program = 0;
    unsigned int m_vao = 0;

    unsigned int m_fbo = 0;
    unsigned int m_mask_texture = 0;
    unsigned int m_depth_texture = 0;
    int m_width = 0;
    int m_height = 0;

    int m_mask_location = -1;
    int m_depth_location = -1;
    int m_near_location = -1;
    int m_far_location = -1;
    int m_tolerance_location = -1;
    int m_radius_location = -1;
    int m_selected_location = -1;
    int m_active_location = -1;
};

} // namespace gfx
