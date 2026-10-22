#pragma once

/*
 * Software rasterization of a UiDrawList, for backends that have no
 * hardware rasterizer (the CUDA renderer).
 *
 * build_ui_raster_tris() runs on the CPU: it turns each UI triangle into
 * fixed-point edge data (the same 1/16-pixel snapping and top-left fill
 * rule as the CUDA mesh rasterizer, see core/raster_math.h) and computes
 * which pixels the UI can touch at all.
 *
 * ui_blend_triangle() is GFX_HD: the CUDA kernel calls it for every
 * pixel and triangle, and the tests call the same code on the CPU.
 *
 * The result matches the OpenGL and Vulkan UI passes: a pixel is covered
 * when its centre is inside the triangle (top-left rule on shared
 * edges, so the diagonal of a quad is drawn once), attributes are
 * interpolated at the pixel centre, the atlas is sampled with nearest
 * filtering, and "source over" blending is rounded to 8 bits after each
 * triangle, as an RGBA8 framebuffer would.
 */

#include "core/common.h"
#include "core/raster_math.h"
#include "ui/ui_draw.h"

#include <cstdint>
#include <vector>

namespace gfx {

// One UI triangle after setup. Plain 4-byte fields only, so the CPU and
// the GPU agree on the layout
struct UiRasterTri {
    // Vertices in 1/16 pixels, in positive-area order
    int sx[3];
    int sy[3];

    // Edge k is the edge opposite vertex k. Its bias is 0 when it is a
    // top or left edge (pixels exactly on it are inside) and -1
    // otherwise (they are left to the neighbouring triangle)
    int bias[3];

    // Pixel bounds, inclusive, already clamped to the framebuffer
    int min_x;
    int min_y;
    int max_x;
    int max_y;

    // 1 / (twice the triangle's area in 1/16-pixel units), to turn edge
    // values into barycentric weights
    float inv_area;

    // Atlas coordinates and colours (UiVertex::color) in the same
    // vertex order as sx/sy
    float u[3];
    float v[3];
    std::uint32_t color[3];
};

static_assert(sizeof(UiRasterTri) == 23 * 4,
              "UiRasterTri must stay a plain array of 4-byte fields");

// Pixels the UI may touch, inclusive. Empty when x0 > x1
struct UiRasterBounds {
    int x0 = 0;
    int y0 = 0;
    int x1 = -1;
    int y1 = -1;

    bool empty() const { return x0 > x1 || y0 > y1; }
};

// Sets up every triangle of list for a width x height framebuffer.
// Degenerate and off-screen triangles are dropped; the order of the
// others is kept, because blending depends on it
UiRasterBounds build_ui_raster_tris(const UiDrawList& list,
                                    int width,
                                    int height,
                                    std::vector<UiRasterTri>& out);

// Blends triangle t into the pixel (x, y) if the pixel's centre is
// inside it. rgb holds the pixel's current colour as 0..255 values and
// is updated in place. atlas is the font atlas (width aw, height ah)
GFX_HD inline void ui_blend_triangle(const UiRasterTri& t,
                                     int x,
                                     int y,
                                     const std::uint8_t* atlas,
                                     int aw,
                                     int ah,
                                     float rgb[3]) {
    if (x < t.min_x || x > t.max_x || y < t.min_y || y > t.max_y) {
        return;
    }

    const int px = x * kRasterSubpixelScale + kRasterSubpixelHalf;
    const int py = y * kRasterSubpixelScale + kRasterSubpixelHalf;

    const long long e0 =
        edge_function(t.sx[1], t.sy[1], t.sx[2], t.sy[2], px, py);
    const long long e1 =
        edge_function(t.sx[2], t.sy[2], t.sx[0], t.sy[0], px, py);
    const long long e2 =
        edge_function(t.sx[0], t.sy[0], t.sx[1], t.sy[1], px, py);

    if (e0 + t.bias[0] < 0 || e1 + t.bias[1] < 0 || e2 + t.bias[2] < 0) {
        return;
    }

    // Barycentric weights at the pixel centre
    const float w0 = static_cast<float>(e0) * t.inv_area;
    const float w1 = static_cast<float>(e1) * t.inv_area;
    const float w2 = 1.0f - w0 - w1;

    // Nearest texel
    const float u = w0 * t.u[0] + w1 * t.u[1] + w2 * t.u[2];
    const float v = w0 * t.v[0] + w1 * t.v[1] + w2 * t.v[2];

    int tx = static_cast<int>(floorf(u * static_cast<float>(aw)));
    int ty = static_cast<int>(floorf(v * static_cast<float>(ah)));
    tx = tx < 0 ? 0 : (tx >= aw ? aw - 1 : tx);
    ty = ty < 0 ? 0 : (ty >= ah ? ah - 1 : ty);

    const float coverage =
        static_cast<float>(atlas[ty * aw + tx]) * (1.0f / 255.0f);

    // Interpolated vertex colour, channel c in byte c
    float src[4];

    for (int c = 0; c < 4; ++c) {
        const int shift = 8 * c;
        src[c] = (w0 * static_cast<float>((t.color[0] >> shift) & 0xFFu) +
                  w1 * static_cast<float>((t.color[1] >> shift) & 0xFFu) +
                  w2 * static_cast<float>((t.color[2] >> shift) & 0xFFu)) *
                 (1.0f / 255.0f);
    }

    const float alpha = src[3] * coverage;

    if (alpha <= 0.0f) {
        return;
    }

    // Source over, rounded to 8 bits like an RGBA8 framebuffer
    for (int c = 0; c < 3; ++c) {
        const float blended = src[c] * 255.0f * alpha + rgb[c] * (1.0f - alpha);
        const float rounded = floorf(blended + 0.5f);
        rgb[c] = rounded < 0.0f ? 0.0f : (rounded > 255.0f ? 255.0f : rounded);
    }
}

} // namespace gfx
