#include "core/image.h"
#include "core/texture_sampling.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

namespace {

using gfx::Image;
using gfx::TextureView;
using gfx::Vec3;

Image make_image(int width, int height) {
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.assign(static_cast<std::size_t>(width) *
                            static_cast<std::size_t>(height) * 4,
                        255);
    return image;
}

void set_pixel(Image& image, int x, int y, int r, int g, int b, int a) {
    std::uint8_t* p = &image.pixels[(static_cast<std::size_t>(y) *
                                         static_cast<std::size_t>(image.width) +
                                     static_cast<std::size_t>(x)) * 4];
    p[0] = static_cast<std::uint8_t>(r);
    p[1] = static_cast<std::uint8_t>(g);
    p[2] = static_cast<std::uint8_t>(b);
    p[3] = static_cast<std::uint8_t>(a);
}

int red(const Image& image, int x, int y) {
    return image.pixels[(static_cast<std::size_t>(y) *
                             static_cast<std::size_t>(image.width) +
                         static_cast<std::size_t>(x)) * 4];
}

// A texture whose texel (x, y) of level 0 has red = 10 * x + 40 * y (a
// ramp), green = 100, blue = 0, so every texel is easy to recognise
Image make_ramp(int width, int height) {
    Image image = make_image(width, height);

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            set_pixel(image, x, y, 10 * x + 40 * y, 100, 0, 255);
        }
    }

    return image;
}

bool close_to(float a, float b, float eps = 1.0e-4f) {
    return std::fabs(a - b) <= eps;
}

bool color_is(Vec3 c, float r, float g, float b, float eps = 1.0e-4f) {
    return close_to(c.x, r, eps) && close_to(c.y, g, eps) && close_to(c.z, b, eps);
}

// ---------------------------------------------------------------------
// Mip chains
// ---------------------------------------------------------------------

void test_chain_sizes() {
    const std::vector<Image> square = gfx::build_mip_chain(make_image(8, 8));
    CHECK(square.size() == 4);
    CHECK(square[0].width == 8 && square[0].height == 8);
    CHECK(square[1].width == 4 && square[1].height == 4);
    CHECK(square[2].width == 2 && square[2].height == 2);
    CHECK(square[3].width == 1 && square[3].height == 1);

    // Odd sizes round down
    const std::vector<Image> odd = gfx::build_mip_chain(make_image(5, 3));
    CHECK(odd.size() == 3);
    CHECK(odd[1].width == 2 && odd[1].height == 1);
    CHECK(odd[2].width == 1 && odd[2].height == 1);

    // One side already at 1 stays at 1 while the other halves
    const std::vector<Image> column = gfx::build_mip_chain(make_image(1, 6));
    CHECK(column.size() == 3);
    CHECK(column[1].width == 1 && column[1].height == 3);
    CHECK(column[2].width == 1 && column[2].height == 1);

    const std::vector<Image> row = gfx::build_mip_chain(make_image(4, 1));
    CHECK(row.size() == 3);
    CHECK(row[1].width == 2 && row[1].height == 1);
    CHECK(row[2].width == 1 && row[2].height == 1);

    const std::vector<Image> single = gfx::build_mip_chain(make_image(1, 1));
    CHECK(single.size() == 1);

    // Invalid input gives nothing
    CHECK(gfx::build_mip_chain(Image{}).empty());

    // The largest accepted image fits in kMaxMipLevels
    CHECK(gfx::mip_size(gfx::kMaxImageSize, 14) == 1);
    CHECK(gfx::mip_size(gfx::kMaxImageSize, 13) == 2);
    CHECK(gfx::mip_size(gfx::kMaxImageSize, 15) == 1);
    CHECK(gfx::mip_size(7, 40) == 1);
}

void test_chain_values() {
    // Level 0 is an exact copy
    const Image source = make_ramp(4, 4);
    const std::vector<Image> chain = gfx::build_mip_chain(source);
    CHECK(chain[0].pixels == source.pixels);

    // 2x2 blocks, rounded to nearest. Block (0, 0) of the ramp holds red
    // 0, 10, 40, 50: average 25. Block (1, 0): 20, 30, 60, 70: 45
    CHECK(red(chain[1], 0, 0) == 25);
    CHECK(red(chain[1], 1, 0) == 45);
    CHECK(red(chain[1], 0, 1) == 105);
    CHECK(red(chain[1], 1, 1) == 125);
    CHECK(chain[1].pixels[1] == 100);  // green
    CHECK(chain[1].pixels[2] == 0);    // blue
    CHECK(chain[1].pixels[3] == 255);  // alpha

    // The top level is the average of everything: (25+45+105+125)/4 = 75
    CHECK(red(chain[2], 0, 0) == 75);

    // Rounding: 0, 0, 0, 1 averages to 0.25 -> 0; 0, 0, 1, 1 -> 0.5 -> 1
    Image tiny = make_image(2, 2);
    set_pixel(tiny, 0, 0, 0, 0, 0, 0);
    set_pixel(tiny, 1, 0, 0, 0, 0, 0);
    set_pixel(tiny, 0, 1, 0, 0, 0, 0);
    set_pixel(tiny, 1, 1, 1, 1, 1, 1);
    CHECK(red(gfx::build_mip_chain(tiny)[1], 0, 0) == 0);

    set_pixel(tiny, 0, 1, 1, 1, 1, 1);
    CHECK(red(gfx::build_mip_chain(tiny)[1], 0, 0) == 1);

    // A single colour stays that colour on every level, odd sizes included
    Image flat = make_image(7, 5);

    for (int y = 0; y < 5; ++y) {
        for (int x = 0; x < 7; ++x) {
            set_pixel(flat, x, y, 200, 17, 3, 128);
        }
    }

    for (const Image& level : gfx::build_mip_chain(flat)) {
        for (std::size_t i = 0; i < level.pixels.size(); i += 4) {
            CHECK(level.pixels[i] == 200);
            CHECK(level.pixels[i + 1] == 17);
            CHECK(level.pixels[i + 2] == 3);
            CHECK(level.pixels[i + 3] == 128);
        }
    }
}

void test_chain_odd_sizes() {
    // 5x1 with reds 0, 100, 200, 100, 0. Level 1 is 2 texels wide;
    // texel 0 covers source texels 0, 1 (full) and half of 2 (weights
    // 2, 2, 1 over 5): (0*2 + 100*2 + 200*1) / 5 = 80. Texel 1 is the mirror
    Image row = make_image(5, 1);
    const int reds[5] = {0, 100, 200, 100, 0};

    for (int x = 0; x < 5; ++x) {
        set_pixel(row, x, 0, reds[x], 0, 0, 255);
    }

    const std::vector<Image> chain = gfx::build_mip_chain(row);
    CHECK(chain.size() == 3);
    CHECK(red(chain[1], 0, 0) == 80);
    CHECK(red(chain[1], 1, 0) == 80);
    CHECK(red(chain[2], 0, 0) == 80);

    // 3x1: one destination texel covering three, equal weights
    Image three = make_image(3, 1);
    set_pixel(three, 0, 0, 30, 0, 0, 255);
    set_pixel(three, 1, 0, 60, 0, 0, 255);
    set_pixel(three, 2, 0, 90, 0, 0, 255);
    CHECK(red(gfx::build_mip_chain(three)[1], 0, 0) == 60);

    // 5x3: a vertical ramp, 40 per row, rows 0, 40, 80, all columns equal.
    // 3 -> 1 rows: (0 + 40 + 80) / 3 = 40 everywhere on level 1
    const Image ramp = make_ramp(5, 3);
    const std::vector<Image> odd = gfx::build_mip_chain(ramp);
    CHECK(odd.size() == 3);
    CHECK(odd[1].width == 2 && odd[1].height == 1);

    // Column weights 2, 2, 1 of reds 0, 10, 20, 30, 40 (+40 per row, whose
    // average over the three rows is 40): texel 0 = (0*2+10*2+20*1)/5 + 40
    // = 8 + 40 = 48; texel 1 = (20*1+30*2+40*2)/5 + 40 = 32 + 40 = 72
    CHECK(red(odd[1], 0, 0) == 48);
    CHECK(red(odd[1], 1, 0) == 72);
}

void test_pack() {
    const std::vector<Image> chain = gfx::build_mip_chain(make_ramp(5, 3));

    TextureView view;
    const std::vector<std::uint8_t> packed = gfx::pack_mip_chain(chain, view);

    CHECK(view.width == 5 && view.height == 3 && view.levels == 3);
    CHECK(view.offsets[0] == 0);
    CHECK(view.offsets[1] == 15);  // 5 * 3 texels
    CHECK(view.offsets[2] == 17);  // + 2 * 1
    CHECK(packed.size() == (15 + 2 + 1) * 4);
    CHECK(view.texels == packed.data());
    CHECK(packed[17 * 4] == red(chain[2], 0, 0));
    CHECK(packed[15 * 4 + 4] == red(chain[1], 1, 0));

    // Bad chains give an empty result and an empty view
    TextureView bad;
    CHECK(gfx::pack_mip_chain({}, bad).empty());
    CHECK(bad.texels == nullptr && bad.levels == 0);

    std::vector<Image> wrong = chain;
    wrong.pop_back();
    wrong.push_back(make_image(2, 2));  // the last level must be 1x1
    CHECK(gfx::pack_mip_chain(wrong, bad).empty());
    CHECK(bad.levels == 0);

    std::vector<Image> too_many(17, make_image(1, 1));
    CHECK(gfx::pack_mip_chain(too_many, bad).empty());
}

// ---------------------------------------------------------------------
// Sampling
// ---------------------------------------------------------------------

// A texture with its mip chain packed, ready for the sampling functions
struct Sampler {
    std::vector<Image> chain;
    std::vector<std::uint8_t> packed;
    TextureView view;

    explicit Sampler(const Image& image) {
        chain = gfx::build_mip_chain(image);
        packed = gfx::pack_mip_chain(chain, view);
    }
};

// 4x2 test texture: red = 60 * x, green = 120 * y, blue = 7
Image make_grid_4x2() {
    Image image = make_image(4, 2);

    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            set_pixel(image, x, y, 60 * x, 120 * y, 7, 255);
        }
    }

    return image;
}

void test_bilinear() {
    const Sampler s(make_grid_4x2());
    const float k = 1.0f / 255.0f;

    // Texel centres are at (i + 0.5) / size: exact values
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 4; ++x) {
            const Vec3 c = gfx::sample_bilinear(
                s.view, 0,
                (static_cast<float>(x) + 0.5f) / 4.0f,
                (static_cast<float>(y) + 0.5f) / 2.0f);

            CHECK(color_is(c, 60.0f * static_cast<float>(x) * k,
                           120.0f * static_cast<float>(y) * k, 7.0f * k));
        }
    }

    // v = 0 is the top row of the image
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.125f, 0.25f),
                   0.0f, 0.0f, 7.0f * k));

    // Halfway between the centres of texels 1 and 2 (u = 0.5): red 90
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.5f, 0.25f),
                   90.0f * k, 0.0f, 7.0f * k));

    // Halfway between the two rows (v = 0.5): green 60
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.125f, 0.5f),
                   0.0f, 60.0f * k, 7.0f * k));

    // A quarter of the way from texel 1 to texel 2: red 75
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.5f - 0.0625f, 0.25f),
                   75.0f * k, 0.0f, 7.0f * k));

    // Both at once, centre of the four middle texels: red 90, green 60
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.5f, 0.5f),
                   90.0f * k, 60.0f * k, 7.0f * k));
}

void test_repeat() {
    const Sampler s(make_grid_4x2());
    const float k = 1.0f / 255.0f;

    // u = 1.375 is the same as u = 0.375, the centre of texel 1 (red 60)
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 1.375f, 0.25f),
                   60.0f * k, 0.0f, 7.0f * k));

    // Negative coordinates wrap upwards: -0.625 is 0.375, and v = -0.25
    // is 0.75, the centre of row 1 (green 120)
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, -0.625f, 0.25f),
                   60.0f * k, 0.0f, 7.0f * k));
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.125f, -0.25f),
                   0.0f, 120.0f * k, 7.0f * k));
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 3.125f, 2.25f),
                   0.0f, 0.0f, 7.0f * k));

    // At u = 0 and u = 1 the filter blends the last and the first texel
    // (red 180 and 0), because the texture repeats: 90
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.0f, 0.25f),
                   90.0f * k, 0.0f, 7.0f * k));
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 1.0f, 0.25f),
                   90.0f * k, 0.0f, 7.0f * k));

    // Same for v: rows 1 and 0 blend (green 120 and 0)
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.125f, 0.0f),
                   0.0f, 60.0f * k, 7.0f * k));

    // Tiny negative numbers must not read outside the level
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, -1.0e-9f, 1.0e-9f),
                   90.0f * k, 60.0f * k, 7.0f * k));

    // Garbage in, no crash out
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    (void)gfx::sample_bilinear(s.view, 0, nan, inf);
    (void)gfx::sample_bilinear(s.view, 0, -inf, 1.0e30f);
    (void)gfx::sample_trilinear(s.view, 0.3f, 0.3f, nan);
    (void)gfx::sample_trilinear(s.view, 0.3f, 0.3f, inf);
}

void test_levels_and_trilinear() {
    // 4x4 where level 0 is red = 200 in the top-left 2x2 and 0 elsewhere,
    // so level 1 holds 200 in its top-left texel, 0 elsewhere, and level 2
    // (1x1) holds the average 50
    Image image = make_image(4, 4);

    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            set_pixel(image, x, y, (x < 2 && y < 2) ? 200 : 0, 0, 0, 255);
        }
    }

    const Sampler s(image);
    CHECK(s.view.levels == 3);

    const float k = 1.0f / 255.0f;

    // Each level on its own
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, 0.125f, 0.125f),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_bilinear(s.view, 1, 0.25f, 0.25f),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_bilinear(s.view, 2, 0.9f, 0.1f),
                   50.0f * k, 0.0f, 0.0f));

    // A level number outside the chain is clamped
    CHECK(color_is(gfx::sample_bilinear(s.view, 9, 0.9f, 0.1f),
                   50.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_bilinear(s.view, -3, 0.125f, 0.125f),
                   200.0f * k, 0.0f, 0.0f));

    // Whole levels are the plain bilinear samples
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.125f, 0.125f, 0.0f),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.25f, 0.25f, 1.0f),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.25f, 0.25f, 2.0f),
                   50.0f * k, 0.0f, 0.0f));

    // At (0.25, 0.25) levels 0 and 1 both give 200, so halfway between
    // them is 200; between level 1 (200) and level 2 (50) it is 125, and
    // a quarter of the way is 162.5
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.25f, 0.25f, 0.5f),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.25f, 0.25f, 1.5f),
                   125.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, 0.25f, 0.25f, 1.25f),
                   162.5f * k, 0.0f, 0.0f));

    // A point where level 0 and level 1 differ: u = 0.375 is the centre
    // of level-0 texel 1 (200), but on the 2-texel-wide level 1 it is a
    // quarter of the way from texel 0 (200) to texel 1 (0): 150.
    // v = 0.25 is the centre of level-1 row 0, between level-0 rows 0 and
    // 1, which are equal here
    const float u = 0.375f;
    const float v = 0.25f;
    CHECK(color_is(gfx::sample_bilinear(s.view, 0, u, v),
                   200.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_bilinear(s.view, 1, u, v),
                   150.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, u, v, 0.5f),
                   175.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, u, v, 0.25f),
                   187.5f * k, 0.0f, 0.0f));

    // lod beyond the chain is clamped to the last level, below 0 to 0
    CHECK(color_is(gfx::sample_trilinear(s.view, u, v, 50.0f),
                   50.0f * k, 0.0f, 0.0f));
    CHECK(color_is(gfx::sample_trilinear(s.view, u, v, -5.0f),
                   200.0f * k, 0.0f, 0.0f));

    // A texture without data is white instead of a crash
    const TextureView none;
    CHECK(color_is(gfx::sample_bilinear(none, 0, 0.5f, 0.5f), 1, 1, 1));
    CHECK(color_is(gfx::sample_trilinear(none, 0.5f, 0.5f, 1.0f), 1, 1, 1));
}

void test_texture_lod() {
    const Sampler s(make_image(256, 64));

    // One screen pixel moves one texel: lod 0. 1/256 of the width, 1/64 of
    // the height per pixel
    CHECK(close_to(gfx::texture_lod(s.view, 1.0f / 256, 0.0f, 0.0f, 1.0f / 64),
               0.0f));

    // Magnification: a pixel covers a fraction of a texel: still 0
    CHECK(close_to(gfx::texture_lod(s.view, 0.25f / 256, 0.0f, 0.0f,
                                0.25f / 64),
               0.0f));
    CHECK(close_to(gfx::texture_lod(s.view, 0.0f, 0.0f, 0.0f, 0.0f), 0.0f));

    // 2x and 4x minification
    CHECK(close_to(gfx::texture_lod(s.view, 2.0f / 256, 0.0f, 0.0f, 2.0f / 64),
               1.0f));
    CHECK(close_to(gfx::texture_lod(s.view, 4.0f / 256, 0.0f, 0.0f, 4.0f / 64),
               2.0f));

    // The longer of the two steps counts, whichever it is
    CHECK(close_to(gfx::texture_lod(s.view, 1.0f / 256, 0.0f, 0.0f,
                                4.0f / 64),
               2.0f));
    CHECK(close_to(gfx::texture_lod(s.view, 8.0f / 256, 0.0f, 0.0f,
                                1.0f / 64),
               3.0f));

    // A diagonal step has length sqrt(a^2 + b^2): 3 and 4 texels -> 5
    CHECK(close_to(gfx::texture_lod(s.view, 3.0f / 256, 4.0f / 64, 0.0f, 0.0f),
               std::log2(5.0f)));

    // Signs do not matter
    CHECK(close_to(gfx::texture_lod(s.view, -2.0f / 256, 0.0f, 0.0f,
                                -2.0f / 64),
               1.0f));

    // NaN and infinity do not produce NaN
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float inf = std::numeric_limits<float>::infinity();
    CHECK(close_to(gfx::texture_lod(s.view, nan, nan, nan, nan), 0.0f));
    CHECK(gfx::texture_lod(s.view, inf, 0.0f, 0.0f, 0.0f) > 100.0f);
}

} // namespace

int main() {
    test_chain_sizes();
    test_chain_values();
    test_chain_odd_sizes();
    test_pack();
    test_bilinear();
    test_repeat();
    test_levels_and_trilinear();
    test_texture_lod();

    if (g_failures == 0) {
        std::printf("All texture tests passed\n");
        return 0;
    }

    std::printf("%d texture test failure(s)\n", g_failures);
    return 1;
}
