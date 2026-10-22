#include "ui/ui_draw.h"

#include <algorithm>
#include <cmath>

namespace gfx {

void UiDrawList::begin(const FontAtlas* font) {
    m_font = (font != nullptr && font->valid()) ? font : nullptr;
    m_vertices.clear();
    m_indices.clear();
}

void UiDrawList::add_quad(float x0, float y0, float x1, float y1,
                          float u0, float v0, float u1, float v1,
                          std::uint32_t color) {
    const std::uint32_t base =
        static_cast<std::uint32_t>(m_vertices.size());

    m_vertices.push_back({x0, y0, u0, v0, color});
    m_vertices.push_back({x1, y0, u1, v0, color});
    m_vertices.push_back({x1, y1, u1, v1, color});
    m_vertices.push_back({x0, y1, u0, v1, color});

    // Two triangles: 0-1-2 and 0-2-3
    m_indices.push_back(base + 0);
    m_indices.push_back(base + 1);
    m_indices.push_back(base + 2);
    m_indices.push_back(base + 0);
    m_indices.push_back(base + 2);
    m_indices.push_back(base + 3);
}

void UiDrawList::add_rect(float x0, float y0, float x1, float y1,
                          std::uint32_t color) {
    if (x1 <= x0 || y1 <= y0) {
        return;
    }

    // Every corner samples the centre of the covered texel, so the
    // whole rectangle reads coverage 1.0
    float u = 0.0f;
    float v = 0.0f;

    if (m_font != nullptr) {
        u = (static_cast<float>(m_font->white_x) + 0.5f) /
            static_cast<float>(m_font->width);
        v = (static_cast<float>(m_font->white_y) + 0.5f) /
            static_cast<float>(m_font->height);
    }

    add_quad(x0, y0, x1, y1, u, v, u, v, color);
}

std::size_t UiDrawList::add_rect_placeholder(std::uint32_t color) {
    const std::size_t handle = m_vertices.size();

    // Same as add_rect, but with all four corners at the origin: a
    // zero-area quad, which no backend draws
    float u = 0.0f;
    float v = 0.0f;

    if (m_font != nullptr) {
        u = (static_cast<float>(m_font->white_x) + 0.5f) /
            static_cast<float>(m_font->width);
        v = (static_cast<float>(m_font->white_y) + 0.5f) /
            static_cast<float>(m_font->height);
    }

    add_quad(0.0f, 0.0f, 0.0f, 0.0f, u, v, u, v, color);
    return handle;
}

void UiDrawList::set_rect(std::size_t handle,
                          float x0, float y0, float x1, float y1) {
    if (handle + 4 > m_vertices.size()) {
        return;
    }

    // Corner order matches add_quad: top-left, top-right,
    // bottom-right, bottom-left
    UiVertex* q = &m_vertices[handle];
    q[0].x = x0; q[0].y = y0;
    q[1].x = x1; q[1].y = y0;
    q[2].x = x1; q[2].y = y1;
    q[3].x = x0; q[3].y = y1;
}

void UiDrawList::add_rect_outline(float x0, float y0, float x1, float y1,
                                  float thickness, std::uint32_t color) {
    const float t = std::min(thickness,
                             0.5f * std::min(x1 - x0, y1 - y0));

    if (t <= 0.0f) {
        return;
    }

    add_rect(x0, y0, x1, y0 + t, color);           // top
    add_rect(x0, y1 - t, x1, y1, color);           // bottom
    add_rect(x0, y0 + t, x0 + t, y1 - t, color);   // left
    add_rect(x1 - t, y0 + t, x1, y1 - t, color);   // right
}

const FontGlyph* UiDrawList::glyph_or_fallback(char c) const {
    const FontGlyph* g = m_font->glyph(c);
    return g != nullptr ? g : m_font->glyph('?');
}

float UiDrawList::add_text(float x, float y, const char* text,
                           std::uint32_t color) {
    if (m_font == nullptr || text == nullptr) {
        return 0.0f;
    }

    const float inv_w = 1.0f / static_cast<float>(m_font->width);
    const float inv_h = 1.0f / static_cast<float>(m_font->height);

    const float start_x = std::round(x);
    float pen_x = start_x;
    float pen_y = std::round(y);
    float widest = 0.0f;

    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == '\n') {
            widest = std::max(widest, pen_x - start_x);
            pen_x = start_x;
            pen_y += static_cast<float>(m_font->line_height);
            continue;
        }

        const FontGlyph* g = glyph_or_fallback(*p);

        if (g->width > 0 && g->height > 0) {
            const float gx = pen_x + static_cast<float>(g->offset_x);
            const float gy = pen_y + static_cast<float>(g->offset_y);

            add_quad(gx, gy,
                     gx + static_cast<float>(g->width),
                     gy + static_cast<float>(g->height),
                     static_cast<float>(g->atlas_x) * inv_w,
                     static_cast<float>(g->atlas_y) * inv_h,
                     static_cast<float>(g->atlas_x + g->width) * inv_w,
                     static_cast<float>(g->atlas_y + g->height) * inv_h,
                     color);
        }

        pen_x += static_cast<float>(g->advance);
    }

    return std::max(widest, pen_x - start_x);
}

float UiDrawList::text_width(const char* text) const {
    if (m_font == nullptr || text == nullptr) {
        return 0.0f;
    }

    int widest = 0;
    int width = 0;

    for (const char* p = text; *p != '\0'; ++p) {
        if (*p == '\n') {
            widest = std::max(widest, width);
            width = 0;
            continue;
        }

        width += glyph_or_fallback(*p)->advance;
    }

    return static_cast<float>(std::max(widest, width));
}

float UiDrawList::line_height() const {
    return m_font != nullptr
               ? static_cast<float>(m_font->line_height)
               : 0.0f;
}

} // namespace gfx
