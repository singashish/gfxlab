#pragma once

/*
 * The UI's output: a list of textured, coloured triangles in window
 * pixel coordinates (origin top-left, Y down), the same space as the
 * mouse position in InputState.
 *
 * The app fills one UiDrawList per frame and passes it to the renderer
 * in FrameData::ui. Every backend draws it the same way, on top of the
 * 3D view:
 *
 *   colour = vertex.rgb, alpha = vertex.a * atlas(uv)
 *
 * blended with "source over". Text samples glyphs in the font atlas;
 * rectangles sample the atlas's fully covered texel, so both go through
 * one shader and one draw call.
 */

#include "ui/font_atlas.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

// 20 bytes. color holds 8-bit R, G, B, A in that byte order in memory
// (see ui_rgba), which every backend reads as one normalized RGBA8
// attribute
struct UiVertex {
    float x;
    float y;
    float u;
    float v;
    std::uint32_t color;
};

static_assert(sizeof(UiVertex) == 20, "UiVertex must stay 20 bytes");

// Packs a colour so its bytes are R, G, B, A in memory (little-endian)
constexpr std::uint32_t ui_rgba(int r, int g, int b, int a) {
    return static_cast<std::uint32_t>(r & 0xFF) |
           (static_cast<std::uint32_t>(g & 0xFF) << 8) |
           (static_cast<std::uint32_t>(b & 0xFF) << 16) |
           (static_cast<std::uint32_t>(a & 0xFF) << 24);
}

class UiDrawList {
public:
    // Clears the list for a new frame. Text needs a font; without one
    // only rectangles are recorded and the backends draw nothing,
    // because they need the atlas texture
    void begin(const FontAtlas* font);

    // Filled rectangle from (x0, y0) to (x1, y1)
    void add_rect(float x0, float y0, float x1, float y1,
                  std::uint32_t color);

    // Filled triangle (any winding)
    void add_triangle(float x0, float y0, float x1, float y1,
                      float x2, float y2, std::uint32_t color);

    // Rectangle border of the given thickness, drawn inside the bounds
    void add_rect_outline(float x0, float y0, float x1, float y1,
                          float thickness, std::uint32_t color);

    // Adds a rectangle whose corners are set later with set_rect(); until
    // then it covers nothing. A panel uses it to put its background
    // under its widgets before it knows how tall it will be.
    // Returns a handle for set_rect()
    std::size_t add_rect_placeholder(std::uint32_t color);
    void set_rect(std::size_t handle, float x0, float y0, float x1, float y1);

    // Draws text with its first line's top-left corner at (x, y).
    // '\n' starts a new line; characters outside 32..126 draw as '?'.
    // The start position is rounded to whole pixels so glyphs stay
    // sharp. Returns the width of the widest line
    float add_text(float x, float y, const char* text,
                   std::uint32_t color);

    // Width of the widest line of text, and the height of one line
    float text_width(const char* text) const;
    float line_height() const;

    const FontAtlas* font() const { return m_font; }
    const std::vector<UiVertex>& vertices() const { return m_vertices; }
    const std::vector<std::uint32_t>& indices() const { return m_indices; }
    bool empty() const { return m_indices.empty(); }

private:
    void add_quad(float x0, float y0, float x1, float y1,
                  float u0, float v0, float u1, float v1,
                  std::uint32_t color);

    const FontGlyph* glyph_or_fallback(char c) const;

    const FontAtlas* m_font = nullptr;
    std::vector<UiVertex> m_vertices;
    std::vector<std::uint32_t> m_indices;
};

} // namespace gfx
