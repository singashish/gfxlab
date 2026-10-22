#pragma once

/*
 * A bitmap font for the UI: every printable ASCII character (32..126)
 * packed into one 8-bit coverage image, the "atlas".
 *
 * The platform layer rasterizes the glyphs (GDI on Windows, see
 * platform/ui_font.h) and hands them to pack_font_atlas(), which is
 * portable and tested. Every backend uploads the atlas once as a
 * single-channel texture and draws text as textured quads.
 *
 * The atlas also holds a small fully covered block. Solid rectangles
 * sample it, so rectangles and text share one shader and one draw call.
 */

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

constexpr int kFontFirstChar = 32;   // ' '
constexpr int kFontLastChar = 126;   // '~'
constexpr int kFontCharCount = kFontLastChar - kFontFirstChar + 1;

// Width of every atlas. A multiple of 4, so each row is 4-byte aligned
// (the default unpack alignment of OpenGL)
constexpr int kFontAtlasWidth = 256;

// Placement of one glyph, in pixels
struct FontGlyph {
    // Rectangle in the atlas. Zero size for blank glyphs such as space
    int atlas_x = 0;
    int atlas_y = 0;
    int width = 0;
    int height = 0;

    // Where the bitmap's top-left corner goes, relative to the pen
    // position at the top of the text line
    int offset_x = 0;
    int offset_y = 0;

    // Horizontal pen movement after this glyph
    int advance = 0;
};

struct FontAtlas {
    int width = 0;
    int height = 0;

    // width * height coverage values, row 0 at the top. 255 = covered
    std::vector<std::uint8_t> alpha;

    FontGlyph glyphs[kFontCharCount] = {};

    // Distance between two lines, and from the top of a line to the
    // baseline, in pixels
    int line_height = 0;
    int ascent = 0;

    // A texel in the middle of the fully covered block
    int white_x = 0;
    int white_y = 0;

    // Display scale the font was made for (1.0 at 96 DPI). UI layout
    // multiplies its paddings and sizes by it
    float scale = 1.0f;

    bool valid() const {
        return width > 0 && height > 0 &&
               alpha.size() == static_cast<std::size_t>(width) *
                                   static_cast<std::size_t>(height);
    }

    // The glyph for c, or nullptr when c is outside 32..126
    const FontGlyph* glyph(char c) const {
        const int code = static_cast<unsigned char>(c);

        if (code < kFontFirstChar || code > kFontLastChar) {
            return nullptr;
        }

        return &glyphs[code - kFontFirstChar];
    }
};

// One rasterized glyph, as produced by the platform layer
struct GlyphBitmap {
    int width = 0;
    int height = 0;
    int offset_x = 0;
    int offset_y = 0;
    int advance = 0;

    // width * height coverage values, row 0 at the top
    std::vector<std::uint8_t> alpha;
};

// Packs kFontCharCount glyphs (for characters 32..126, in order) into
// out. Rows of glyphs are filled left to right, top to bottom, with one
// pixel of space around each glyph; the height is rounded up to a power
// of two. Returns false if the input is malformed or a glyph is wider
// than the atlas
bool pack_font_atlas(const std::vector<GlyphBitmap>& glyphs,
                     int line_height,
                     int ascent,
                     float scale,
                     FontAtlas& out);

} // namespace gfx
