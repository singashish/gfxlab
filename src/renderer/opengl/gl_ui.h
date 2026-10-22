#pragma once

#include "ui/ui_draw.h"

namespace gfx {

// Draws a UiDrawList on top of the current framebuffer: one dynamic
// vertex/index buffer pair refilled every frame, the font atlas as a
// single-channel texture, alpha blending, no depth test
class GLUiRenderer {
public:
    GLUiRenderer() = default;
    ~GLUiRenderer();

    GLUiRenderer(const GLUiRenderer&) = delete;
    GLUiRenderer& operator=(const GLUiRenderer&) = delete;

    bool init();

    // width and height are the framebuffer size in pixels
    void draw(const UiDrawList& list, int width, int height);

    void shutdown();

private:
    bool upload_font(const FontAtlas& font);

    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vertex_buffer = 0;
    unsigned int m_index_buffer = 0;
    unsigned int m_texture = 0;

    int m_viewport_location = -1;
    int m_atlas_location = -1;

    // The atlas currently in m_texture. The app keeps one atlas alive
    // for the whole run, so comparing the pointer is enough
    const FontAtlas* m_font = nullptr;
};

} // namespace gfx
