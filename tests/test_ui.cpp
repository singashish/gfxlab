#include "ui/font_atlas.h"
#include "ui/ui_draw.h"
#include "check.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

// Synthetic font: glyph i is (4 + i % 7) x (6 + i % 5) pixels, filled
// with the value i + 1 so every glyph's pixels are recognisable.
// Space is blank. Advance = width + 1
std::vector<gfx::GlyphBitmap> make_glyphs() {
    std::vector<gfx::GlyphBitmap> glyphs(gfx::kFontCharCount);

    for (int i = 0; i < gfx::kFontCharCount; ++i) {
        gfx::GlyphBitmap& g = glyphs[static_cast<std::size_t>(i)];
        const char c = static_cast<char>(gfx::kFontFirstChar + i);

        if (c == ' ') {
            g.advance = 5;
            continue;
        }

        g.width = 4 + i % 7;
        g.height = 6 + i % 5;
        g.offset_x = 1;
        g.offset_y = 2;
        g.advance = g.width + 1;
        g.alpha.assign(static_cast<std::size_t>(g.width * g.height),
                       static_cast<std::uint8_t>(i + 1));
    }

    return glyphs;
}

bool rects_overlap(int ax, int ay, int aw, int ah,
                   int bx, int by, int bw, int bh) {
    return ax < bx + bw && bx < ax + aw && ay < by + bh && by < ay + ah;
}

void test_pack() {
    const std::vector<gfx::GlyphBitmap> glyphs = make_glyphs();

    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(glyphs, 16, 12, 1.5f, atlas));
    CHECK(atlas.valid());
    CHECK(atlas.width == gfx::kFontAtlasWidth);
    CHECK(atlas.line_height == 16);
    CHECK(atlas.ascent == 12);
    CHECK(approx(atlas.scale, 1.5));

    // Height is a power of two
    CHECK(atlas.height >= 16 && (atlas.height & (atlas.height - 1)) == 0);

    // The covered texel and its neighbours are fully covered
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            const int x = atlas.white_x + dx;
            const int y = atlas.white_y + dy;
            CHECK(x >= 0 && y >= 0);
            CHECK(atlas.alpha[static_cast<std::size_t>(y * atlas.width + x)] ==
                  255);
        }
    }

    const gfx::FontGlyph* space = atlas.glyph(' ');
    CHECK(space != nullptr && space->width == 0 && space->advance == 5);

    CHECK(atlas.glyph('\n') == nullptr);
    CHECK(atlas.glyph(static_cast<char>(127)) == nullptr);
    CHECK(atlas.glyph('~') == &atlas.glyphs[gfx::kFontCharCount - 1]);

    for (int i = 0; i < gfx::kFontCharCount; ++i) {
        const gfx::FontGlyph& a = atlas.glyphs[i];
        const gfx::GlyphBitmap& src = glyphs[static_cast<std::size_t>(i)];

        CHECK(a.width == src.width && a.height == src.height);
        CHECK(a.advance == src.advance);
        CHECK(a.offset_x == src.offset_x && a.offset_y == src.offset_y);

        if (a.width == 0) {
            continue;
        }

        // Inside the atlas
        CHECK(a.atlas_x >= 0 && a.atlas_y >= 0);
        CHECK(a.atlas_x + a.width <= atlas.width);
        CHECK(a.atlas_y + a.height <= atlas.height);

        // Clear of the white block (3 x 3 at the corner) ...
        CHECK(!rects_overlap(a.atlas_x, a.atlas_y, a.width, a.height,
                             0, 0, 3, 3));

        // ... and of every other glyph, including the 1-pixel gap
        for (int j = i + 1; j < gfx::kFontCharCount; ++j) {
            const gfx::FontGlyph& b = atlas.glyphs[j];

            if (b.width == 0) {
                continue;
            }

            CHECK(!rects_overlap(a.atlas_x, a.atlas_y,
                                 a.width + 1, a.height + 1,
                                 b.atlas_x, b.atlas_y,
                                 b.width + 1, b.height + 1));
        }

        // Pixels were copied: every texel holds the glyph's fill value
        bool pixels_ok = true;

        for (int y = 0; y < a.height; ++y) {
            for (int x = 0; x < a.width; ++x) {
                const std::size_t k = static_cast<std::size_t>(
                    (a.atlas_y + y) * atlas.width + a.atlas_x + x);

                pixels_ok = pixels_ok &&
                            atlas.alpha[k] == static_cast<std::uint8_t>(i + 1);
            }
        }

        CHECK(pixels_ok);
    }
}

void test_pack_rejects_bad_input() {
    gfx::FontAtlas atlas;

    // Wrong glyph count
    std::vector<gfx::GlyphBitmap> few(10);
    CHECK(!gfx::pack_font_atlas(few, 16, 12, 1.0f, atlas));
    CHECK(!atlas.valid());

    // Pixel data that does not match the size
    std::vector<gfx::GlyphBitmap> glyphs = make_glyphs();
    glyphs[33].alpha.pop_back();
    CHECK(!gfx::pack_font_atlas(glyphs, 16, 12, 1.0f, atlas));

    // A glyph wider than the atlas
    glyphs = make_glyphs();
    glyphs[33].width = gfx::kFontAtlasWidth + 1;
    glyphs[33].height = 1;
    glyphs[33].alpha.assign(gfx::kFontAtlasWidth + 1, 1);
    CHECK(!gfx::pack_font_atlas(glyphs, 16, 12, 1.0f, atlas));
}

void test_rgba() {
    const std::uint32_t c = gfx::ui_rgba(0x11, 0x22, 0x33, 0x44);

    // Bytes in memory are R, G, B, A on a little-endian machine, which
    // is what the backends' RGBA8 vertex attribute reads
    unsigned char bytes[4];
    std::memcpy(bytes, &c, 4);

    CHECK(bytes[0] == 0x11);
    CHECK(bytes[1] == 0x22);
    CHECK(bytes[2] == 0x33);
    CHECK(bytes[3] == 0x44);
}

void test_rects() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);
    CHECK(list.empty());

    const std::uint32_t red = gfx::ui_rgba(255, 0, 0, 255);
    list.add_rect(10, 20, 30, 50, red);

    CHECK(list.vertices().size() == 4);
    CHECK(list.indices().size() == 6);

    const gfx::UiVertex& v0 = list.vertices()[0];
    const gfx::UiVertex& v2 = list.vertices()[2];

    CHECK(approx(v0.x, 10) && approx(v0.y, 20));
    CHECK(approx(v2.x, 30) && approx(v2.y, 50));
    CHECK(v0.color == red);

    // Every corner samples the centre of the covered texel
    const double u = (atlas.white_x + 0.5) / atlas.width;
    const double v = (atlas.white_y + 0.5) / atlas.height;

    for (const gfx::UiVertex& vert : list.vertices()) {
        CHECK(approx(vert.u, u) && approx(vert.v, v));
    }

    // Indices form two triangles of this quad
    const std::uint32_t expected[6] = {0, 1, 2, 0, 2, 3};

    for (int i = 0; i < 6; ++i) {
        CHECK(list.indices()[static_cast<std::size_t>(i)] == expected[i]);
    }

    // Empty and inverted rectangles add nothing
    list.add_rect(5, 5, 5, 10, red);
    list.add_rect(5, 10, 8, 2, red);
    CHECK(list.vertices().size() == 4);

    // An outline is four rectangles, the second quad's indices offset
    list.add_rect_outline(0, 0, 100, 40, 2, red);
    CHECK(list.vertices().size() == 4 + 16);
    CHECK(list.indices()[6] == 4);

    // begin() clears
    list.begin(&atlas);
    CHECK(list.empty() && list.vertices().empty());
}

void test_text() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    const gfx::FontGlyph& a = *atlas.glyph('A');
    const gfx::FontGlyph& b = *atlas.glyph('B');
    const gfx::FontGlyph& q = *atlas.glyph('?');

    gfx::UiDrawList list;
    list.begin(&atlas);

    CHECK(approx(list.line_height(), 16));

    // Width is the sum of the advances; a space advances without a quad
    const float width = list.add_text(100.4f, 50.6f, "A B", 0xFFFFFFFFu);
    CHECK(approx(width, a.advance + 5 + b.advance));
    CHECK(approx(list.text_width("A B"), width));
    CHECK(list.vertices().size() == 8);

    // First glyph: start rounded to (100, 51), then the glyph offset
    const gfx::UiVertex& v0 = list.vertices()[0];
    CHECK(approx(v0.x, 100 + a.offset_x));
    CHECK(approx(v0.y, 51 + a.offset_y));

    const gfx::UiVertex& v2 = list.vertices()[2];
    CHECK(approx(v2.x, 100 + a.offset_x + a.width));
    CHECK(approx(v2.y, 51 + a.offset_y + a.height));

    // Its UVs cover exactly the glyph's atlas rectangle
    CHECK(approx(v0.u, static_cast<double>(a.atlas_x) / atlas.width));
    CHECK(approx(v0.v, static_cast<double>(a.atlas_y) / atlas.height));
    CHECK(approx(v2.u,
                 static_cast<double>(a.atlas_x + a.width) / atlas.width));
    CHECK(approx(v2.v,
                 static_cast<double>(a.atlas_y + a.height) / atlas.height));

    // Second glyph starts after A's advance and the space's
    const gfx::UiVertex& v4 = list.vertices()[4];
    CHECK(approx(v4.x, 100 + a.advance + 5 + b.offset_x));

    // Characters outside 32..126 draw as '?'
    CHECK(approx(list.text_width("\t"), q.advance));

    // Newline: next line starts at the start x, one line lower;
    // the width is the widest line
    list.begin(&atlas);
    const float w2 = list.add_text(0, 0, "AB\nA", 0xFFFFFFFFu);
    CHECK(approx(w2, a.advance + b.advance));
    CHECK(approx(list.text_width("A\nAB"), a.advance + b.advance));

    const gfx::UiVertex& line2 = list.vertices()[8];
    CHECK(approx(line2.x, a.offset_x));
    CHECK(approx(line2.y, 16 + a.offset_y));
}

void test_no_font() {
    gfx::UiDrawList list;
    list.begin(nullptr);

    CHECK(list.font() == nullptr);
    CHECK(approx(list.add_text(0, 0, "abc", 0xFFFFFFFFu), 0));
    CHECK(approx(list.text_width("abc"), 0));
    CHECK(list.empty());

    // An invalid (empty) atlas counts as no font
    gfx::FontAtlas empty;
    list.begin(&empty);
    CHECK(list.font() == nullptr);
}

} // namespace

int main() {
    test_pack();
    test_pack_rejects_bad_input();
    test_rgba();
    test_rects();
    test_text();
    test_no_font();

    if (g_failures == 0) {
        std::printf("All UI tests passed\n");
        return 0;
    }

    std::printf("%d UI test failure(s)\n", g_failures);
    return 1;
}
