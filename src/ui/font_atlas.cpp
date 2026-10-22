#include "ui/font_atlas.h"

#include <algorithm>

namespace gfx {

namespace {

// Empty pixels between two glyphs, so a glyph never picks up a
// neighbour's coverage when it is sampled near its edge
constexpr int kPadding = 1;

// Size of the fully covered block in the top-left corner. Its middle
// texel is surrounded by covered texels on every side
constexpr int kWhiteSize = 3;

int next_power_of_two(int v) {
    int p = 1;

    while (p < v) {
        p *= 2;
    }

    return p;
}

} // namespace

bool pack_font_atlas(const std::vector<GlyphBitmap>& glyphs,
                     int line_height,
                     int ascent,
                     float scale,
                     FontAtlas& out) {
    out = FontAtlas{};

    if (glyphs.size() != static_cast<std::size_t>(kFontCharCount) ||
        line_height <= 0) {
        return false;
    }

    for (const GlyphBitmap& g : glyphs) {
        if (g.width < 0 || g.height < 0 ||
            g.alpha.size() != static_cast<std::size_t>(g.width) *
                                  static_cast<std::size_t>(g.height) ||
            g.width > kFontAtlasWidth) {
            return false;
        }
    }

    // Pass 1: place the glyphs in rows ("shelves"). The white block
    // starts the first row
    int x = kWhiteSize + kPadding;
    int y = 0;
    int row_height = kWhiteSize;

    for (int i = 0; i < kFontCharCount; ++i) {
        const GlyphBitmap& g = glyphs[static_cast<std::size_t>(i)];
        FontGlyph& placed = out.glyphs[i];

        placed.offset_x = g.offset_x;
        placed.offset_y = g.offset_y;
        placed.advance = g.advance;

        if (g.width == 0 || g.height == 0) {
            continue;  // blank: nothing to store
        }

        // Start a new row when this glyph does not fit in the current one
        if (x + g.width > kFontAtlasWidth) {
            x = 0;
            y += row_height + kPadding;
            row_height = 0;
        }

        placed.atlas_x = x;
        placed.atlas_y = y;
        placed.width = g.width;
        placed.height = g.height;

        x += g.width + kPadding;
        row_height = std::max(row_height, g.height);
    }

    out.width = kFontAtlasWidth;
    out.height = std::max(16, next_power_of_two(y + row_height));
    out.alpha.assign(static_cast<std::size_t>(out.width) *
                         static_cast<std::size_t>(out.height),
                     0);

    // Pass 2: copy the pixels
    for (int row = 0; row < kWhiteSize; ++row) {
        for (int col = 0; col < kWhiteSize; ++col) {
            out.alpha[static_cast<std::size_t>(row * out.width + col)] = 255;
        }
    }

    out.white_x = kWhiteSize / 2;
    out.white_y = kWhiteSize / 2;

    for (int i = 0; i < kFontCharCount; ++i) {
        const GlyphBitmap& g = glyphs[static_cast<std::size_t>(i)];
        const FontGlyph& placed = out.glyphs[i];

        for (int row = 0; row < placed.height; ++row) {
            const std::uint8_t* src =
                g.alpha.data() + static_cast<std::size_t>(row * g.width);

            std::uint8_t* dst =
                out.alpha.data() +
                static_cast<std::size_t>((placed.atlas_y + row) * out.width +
                                         placed.atlas_x);

            std::copy(src, src + placed.width, dst);
        }
    }

    out.line_height = line_height;
    out.ascent = ascent;
    out.scale = scale;
    return true;
}

} // namespace gfx
