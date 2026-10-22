#include "ui/font_atlas.h"
#include "ui/ui.h"
#include "ui/ui_draw.h"
#include "ui/ui_raster.h"
#include "check.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
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

// CPU reference of the CUDA UI pass: every pixel blends every triangle
// in order, exactly like the kernel
struct Image {
    int width;
    int height;
    std::vector<float> rgb;   // 3 values per pixel, 0..255

    Image(int w, int h, float fill)
        : width(w), height(h),
          rgb(static_cast<std::size_t>(w * h * 3), fill) {}

    float at(int x, int y, int c) const {
        return rgb[static_cast<std::size_t>((y * width + x) * 3 + c)];
    }
};

void composite(const gfx::UiDrawList& list, Image& image) {
    std::vector<gfx::UiRasterTri> tris;
    gfx::build_ui_raster_tris(list, image.width, image.height, tris);

    const gfx::FontAtlas& atlas = *list.font();

    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            float* px =
                &image.rgb[static_cast<std::size_t>((y * image.width + x) * 3)];

            for (const gfx::UiRasterTri& t : tris) {
                gfx::ui_blend_triangle(t, x, y, atlas.alpha.data(),
                                       atlas.width, atlas.height, px);
            }
        }
    }
}

void test_raster_rect() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);
    list.add_rect(2, 3, 6, 7, gfx::ui_rgba(200, 100, 50, 128));

    std::vector<gfx::UiRasterTri> tris;
    const gfx::UiRasterBounds bounds =
        gfx::build_ui_raster_tris(list, 10, 10, tris);

    // Two triangles; the pixels whose centres are inside are 2..5, 3..6
    CHECK(tris.size() == 2);
    CHECK(!bounds.empty());
    CHECK(bounds.x0 <= 2 && bounds.y0 <= 3);
    CHECK(bounds.x1 >= 5 && bounds.y1 >= 6);
    CHECK(bounds.x1 <= 6 && bounds.y1 <= 7);

    Image image(10, 10, 20.0f);
    composite(list, image);

    // Expected: source over with alpha 128/255, rounded to 8 bits
    const double a = 128.0 / 255.0;
    const double expected[3] = {
        std::floor(200.0 * a + 20.0 * (1.0 - a) + 0.5),
        std::floor(100.0 * a + 20.0 * (1.0 - a) + 0.5),
        std::floor(50.0 * a + 20.0 * (1.0 - a) + 0.5)
    };

    for (int y = 0; y < 10; ++y) {
        for (int x = 0; x < 10; ++x) {
            const bool inside = x >= 2 && x <= 5 && y >= 3 && y <= 6;

            // Every inside pixel is blended exactly once: the quad's
            // diagonal is shared by both triangles but drawn by one
            for (int c = 0; c < 3; ++c) {
                CHECK(approx(image.at(x, y, c), inside ? expected[c] : 20.0));
            }
        }
    }
}

void test_raster_order_and_clipping() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);

    // Opaque red, then opaque blue overlapping its right half: the later
    // one wins where they overlap
    list.add_rect(0, 0, 8, 4, gfx::ui_rgba(255, 0, 0, 255));
    list.add_rect(4, 0, 8, 4, gfx::ui_rgba(0, 0, 255, 255));

    // Partly off screen: clamped. Entirely off screen: dropped
    list.add_rect(6, 6, 20, 20, gfx::ui_rgba(0, 255, 0, 255));
    list.add_rect(30, 30, 40, 40, gfx::ui_rgba(0, 255, 0, 255));

    std::vector<gfx::UiRasterTri> tris;
    const gfx::UiRasterBounds bounds =
        gfx::build_ui_raster_tris(list, 8, 8, tris);

    CHECK(tris.size() == 6);
    CHECK(bounds.x0 == 0 && bounds.y0 == 0);
    CHECK(bounds.x1 == 7 && bounds.y1 == 7);

    for (const gfx::UiRasterTri& t : tris) {
        CHECK(t.min_x >= 0 && t.max_x <= 7 && t.min_y >= 0 && t.max_y <= 7);
    }

    Image image(8, 8, 0.0f);
    composite(list, image);

    CHECK(approx(image.at(1, 1, 0), 255) && approx(image.at(1, 1, 2), 0));
    CHECK(approx(image.at(6, 1, 0), 0) && approx(image.at(6, 1, 2), 255));
    CHECK(approx(image.at(7, 7, 1), 255));
    CHECK(approx(image.at(5, 5, 1), 0));
}

void test_raster_glyph() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    const gfx::FontGlyph& a = *atlas.glyph('A');
    const int fill = 'A' - gfx::kFontFirstChar + 1;  // see make_glyphs

    gfx::UiDrawList list;
    list.begin(&atlas);
    list.add_text(3, 2, "A", gfx::ui_rgba(255, 255, 255, 255));

    Image image(32, 32, 0.0f);
    composite(list, image);

    // White text over black: each pixel shows the atlas coverage, so the
    // glyph's texels land 1:1 on the pixels at the glyph's offset
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) {
            const bool inside =
                x >= 3 + a.offset_x && x < 3 + a.offset_x + a.width &&
                y >= 2 + a.offset_y && y < 2 + a.offset_y + a.height;

            CHECK(approx(image.at(x, y, 0), inside ? fill : 0));
        }
    }
}

void test_raster_empty() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);

    std::vector<gfx::UiRasterTri> tris(3);
    CHECK(gfx::build_ui_raster_tris(list, 8, 8, tris).empty());
    CHECK(tris.empty());

    // A zero-size framebuffer gives nothing either
    list.add_rect(0, 0, 4, 4, 0xFFFFFFFFu);
    CHECK(gfx::build_ui_raster_tris(list, 0, 0, tris).empty());
}

// ---------------------------------------------------------------------
// Immediate-mode widgets
//
// With the synthetic font (line height 16, scale 1) a panel at (0, 0),
// 200 wide, lays out as:
//   title       y   8 ..  24
//   separator   y  28 ..  33
//   checkbox C  y  37 ..  59
//   slider S    y  63 ..  85, track x 85 .. 192
//   button B    y  89 .. 111
//   (text)      y 115 .. 131
// Panel bottom: last row end + padding 8
// ---------------------------------------------------------------------

struct Widgets {
    bool check = false;
    float value = 0.0f;
    int clicks = 0;
    bool enabled = true;
    const char* extra_text = nullptr;
};

gfx::UiInput mouse(float x, float y, bool down, bool pressed,
                   bool released) {
    gfx::UiInput in;
    in.mouse_x = x;
    in.mouse_y = y;
    in.mouse_down = down;
    in.mouse_pressed = pressed;
    in.mouse_released = released;
    return in;
}

gfx::UiInput press(float x, float y) { return mouse(x, y, true, true, false); }
gfx::UiInput hold(float x, float y) { return mouse(x, y, true, false, false); }
gfx::UiInput release(float x, float y) { return mouse(x, y, false, false, true); }
gfx::UiInput hover(float x, float y) { return mouse(x, y, false, false, false); }

void build(gfx::Ui& ui, gfx::UiDrawList& list, const gfx::FontAtlas& atlas,
           const gfx::UiInput& in, Widgets& w) {
    list.begin(&atlas);
    ui.begin_frame(list, in);
    ui.begin_panel("P", 0.0f, 0.0f, 200.0f);
    ui.set_enabled(w.enabled);
    ui.checkbox("C", &w.check);
    ui.slider("S", &w.value, 0.0f, 100.0f, "%.0f");

    if (ui.button("B")) {
        ++w.clicks;
    }

    if (w.extra_text != nullptr) {
        ui.text(w.extra_text);
    }

    ui.end_panel();
    ui.end_frame();
}

void test_widget_checkbox() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    Widgets w;

    // Hovering the panel claims the mouse; elsewhere it does not
    build(ui, list, atlas, hover(500, 500), w);
    CHECK(!ui.wants_mouse());

    build(ui, list, atlas, hover(20, 48), w);
    CHECK(ui.wants_mouse());

    // A click toggles on release
    build(ui, list, atlas, press(20, 48), w);
    CHECK(!w.check);
    CHECK(ui.wants_mouse());

    build(ui, list, atlas, release(20, 48), w);
    CHECK(w.check);

    build(ui, list, atlas, press(20, 48), w);
    build(ui, list, atlas, release(20, 48), w);
    CHECK(!w.check);

    // Pressed on the box but released elsewhere: no toggle
    build(ui, list, atlas, press(20, 48), w);
    build(ui, list, atlas, release(500, 500), w);
    CHECK(!w.check);
    CHECK(!ui.wants_mouse());

    // Clicks outside the panel do nothing
    build(ui, list, atlas, press(500, 48), w);
    build(ui, list, atlas, release(500, 48), w);
    CHECK(!w.check);

    // Disabled: no reaction
    w.enabled = false;
    build(ui, list, atlas, press(20, 48), w);
    build(ui, list, atlas, release(20, 48), w);
    CHECK(!w.check);
}

void test_widget_slider() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    Widgets w;

    // Press in the middle of the track (x 85 .. 192)
    const float middle = 85.0f + (192.0f - 85.0f) * 0.5f;
    build(ui, list, atlas, press(middle, 70), w);
    CHECK(approx(w.value, 50.0, 1e-3));

    // While held it follows the mouse, even outside the panel, and the
    // UI keeps the mouse
    build(ui, list, atlas, hold(1000, 300), w);
    CHECK(approx(w.value, 100.0));
    CHECK(ui.wants_mouse());

    build(ui, list, atlas, hold(-50, 300), w);
    CHECK(approx(w.value, 0.0));

    // Releasing frees the mouse
    build(ui, list, atlas, release(-50, 300), w);
    CHECK(!ui.wants_mouse());

    // Pressing the label part of the row does not grab the slider
    w.value = 30.0f;
    build(ui, list, atlas, press(20, 70), w);
    build(ui, list, atlas, hold(150, 70), w);
    CHECK(approx(w.value, 30.0));
    build(ui, list, atlas, release(150, 70), w);
}

void test_widget_button() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    Widgets w;

    build(ui, list, atlas, press(100, 100), w);
    CHECK(w.clicks == 0);
    build(ui, list, atlas, hold(100, 100), w);
    CHECK(w.clicks == 0);
    build(ui, list, atlas, release(100, 100), w);
    CHECK(w.clicks == 1);
    build(ui, list, atlas, hover(100, 100), w);
    CHECK(w.clicks == 1);

    // Press and release within one frame still counts as a click
    build(ui, list, atlas, mouse(100, 100, false, true, true), w);
    CHECK(w.clicks == 2);

    // Released away from the button: no click
    build(ui, list, atlas, press(100, 100), w);
    build(ui, list, atlas, release(100, 300), w);
    CHECK(w.clicks == 2);
}

void test_widget_panel_capture() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    Widgets w;

    // A press on the panel's title area holds the mouse: dragging off
    // the panel still belongs to the UI, and releasing over the checkbox
    // does not toggle it
    build(ui, list, atlas, press(100, 12), w);
    CHECK(ui.wants_mouse());
    build(ui, list, atlas, hold(800, 600), w);
    CHECK(ui.wants_mouse());
    build(ui, list, atlas, hold(20, 48), w);
    build(ui, list, atlas, release(20, 48), w);
    CHECK(!w.check);

    // A held widget that disappears (panel not built) is let go
    build(ui, list, atlas, press(20, 48), w);
    CHECK(ui.wants_mouse());
    list.begin(&atlas);
    ui.begin_frame(list, hold(800, 600));
    ui.end_frame();
    CHECK(!ui.wants_mouse());
}

void test_widget_panel_size() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    Widgets w;

    // The background is the first quad of the list and was resized by
    // end_panel to fit the rows: last row (button) ends at 111
    build(ui, list, atlas, hover(500, 500), w);

    const std::vector<gfx::UiVertex>& v = list.vertices();
    CHECK(v.size() >= 4);
    CHECK(approx(v[0].x, 0) && approx(v[0].y, 0));
    CHECK(approx(v[2].x, 200) && approx(v[2].y, 111 + 8));

    // A text row adds 16 + 4; text wider than the panel widens it
    std::string wide(60, 'W');
    w.extra_text = wide.c_str();
    build(ui, list, atlas, hover(500, 500), w);

    const float text_right = 8.0f + list.text_width(wide.c_str());
    CHECK(text_right + 8.0f > 200.0f);
    CHECK(approx(list.vertices()[2].x, text_right + 8.0f));
    CHECK(approx(list.vertices()[2].y, 131 + 8));

    // The mouse over the widened part counts as over the panel
    build(ui, list, atlas, hover(text_right, 120), w);
    CHECK(ui.wants_mouse());
}

void test_widget_header() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    bool open = true;
    bool check = false;

    // Header row at y 37 .. 59; the checkbox under it only exists while
    // the section is open
    const auto build_header = [&](const gfx::UiInput& in) {
        list.begin(&atlas);
        ui.begin_frame(list, in);
        ui.begin_panel("P", 0.0f, 0.0f, 200.0f);

        // Disabled widgets before it do not disable the header
        ui.set_enabled(false);

        if (ui.header("Section", &open)) {
            ui.checkbox("C", &check);
        }

        ui.end_panel();
        ui.end_frame();
    };

    build_header(press(100, 48));
    CHECK(open);
    build_header(release(100, 48));
    CHECK(!open);

    // Folded: the panel ends after the header row (59 + padding 8)
    build_header(hover(500, 500));
    CHECK(approx(list.vertices()[2].y, 59 + 8));

    build_header(press(100, 48));
    build_header(release(100, 48));
    CHECK(open);
}

void test_widget_drag_float3() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::Ui ui;
    gfx::UiDrawList list;
    float values[3] = {1.0f, 2.0f, 3.0f};
    bool changed = false;

    // Row at y 37 .. 59. Label 77 wide, then three 33-wide fields with
    // 4-pixel gaps: x 85 .. 118, 122 .. 155, 159 .. 192
    const auto build_drag = [&](const gfx::UiInput& in) {
        list.begin(&atlas);
        ui.begin_frame(list, in);
        ui.begin_panel("P", 0.0f, 0.0f, 200.0f);
        changed = ui.drag_float3("D", values, 0.5f, "%.1f");
        ui.end_panel();
        ui.end_frame();
    };

    // Pressing alone changes nothing
    build_drag(press(100, 48));
    CHECK(!changed);
    CHECK(approx(values[0], 1.0));

    // Each pixel of movement adds 0.5; only the held field changes,
    // and it keeps following the mouse outside the panel
    build_drag(hold(110, 48));
    CHECK(changed);
    CHECK(approx(values[0], 6.0));
    CHECK(approx(values[1], 2.0) && approx(values[2], 3.0));

    build_drag(hold(90, 300));
    CHECK(approx(values[0], -4.0));
    CHECK(ui.wants_mouse());

    // Vertical movement alone does nothing
    build_drag(hold(90, 10));
    CHECK(!changed);

    build_drag(release(90, 10));

    // The middle field is the y value
    build_drag(press(130, 48));
    build_drag(hold(126, 48));
    CHECK(approx(values[1], 0.0));
    build_drag(release(126, 48));
    CHECK(approx(values[0], -4.0) && approx(values[2], 3.0));
}

void test_triangle() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);
    list.add_triangle(0, 0, 8, 0, 0, 8, gfx::ui_rgba(255, 255, 255, 255));

    CHECK(list.vertices().size() == 3);
    CHECK(list.indices().size() == 3);

    // Rasterized: the corner pixel is inside, the far corner is not
    Image image(8, 8, 0.0f);
    composite(list, image);
    CHECK(approx(image.at(0, 0, 0), 255));
    CHECK(approx(image.at(7, 7, 0), 0));
}

void test_placeholder() {
    gfx::FontAtlas atlas;
    CHECK(gfx::pack_font_atlas(make_glyphs(), 16, 12, 1.0f, atlas));

    gfx::UiDrawList list;
    list.begin(&atlas);

    const std::size_t handle = list.add_rect_placeholder(0xFF0000FFu);
    list.add_rect(1, 1, 2, 2, 0xFFFFFFFFu);

    // Until set, all four corners sit at the origin (zero area)
    CHECK(list.vertices().size() == 8);
    CHECK(approx(list.vertices()[handle + 2].x, 0));

    list.set_rect(handle, 5, 6, 7, 8);
    CHECK(approx(list.vertices()[handle + 0].x, 5));
    CHECK(approx(list.vertices()[handle + 0].y, 6));
    CHECK(approx(list.vertices()[handle + 2].x, 7));
    CHECK(approx(list.vertices()[handle + 2].y, 8));
    CHECK(list.vertices()[handle + 1].color == 0xFF0000FFu);

    // The rectangle added after it is untouched
    CHECK(approx(list.vertices()[4].x, 1));

    // Out-of-range handles are ignored
    list.set_rect(100, 0, 0, 1, 1);
}

} // namespace

int main() {
    test_pack();
    test_pack_rejects_bad_input();
    test_rgba();
    test_rects();
    test_text();
    test_no_font();
    test_raster_rect();
    test_raster_order_and_clipping();
    test_raster_glyph();
    test_raster_empty();
    test_widget_checkbox();
    test_widget_slider();
    test_widget_button();
    test_widget_panel_capture();
    test_widget_panel_size();
    test_placeholder();
    test_widget_header();
    test_widget_drag_float3();
    test_triangle();

    if (g_failures == 0) {
        std::printf("All UI tests passed\n");
        return 0;
    }

    std::printf("%d UI test failure(s)\n", g_failures);
    return 1;
}
