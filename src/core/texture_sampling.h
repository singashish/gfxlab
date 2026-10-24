#pragma once

/*
 * Software texture sampling, shared by the CPU (the tests) and the GPU
 * (the CUDA backend, which does not use the hardware texture units).
 * Header-only and GFX_HD, like raster_math.h and ray.h, so the very same
 * code runs in both places. Nothing here calls a host-only std:: function.
 *
 * Two ideas to know:
 *
 * Mip chain. A texture seen from far away covers many texels per screen
 * pixel; sampling one of them would flicker and alias. So the texture is
 * stored several times, each copy half as wide and half as high as the
 * one before (the "levels"), down to 1x1. Level 0 is the original. A
 * sample from level k averages 2^k x 2^k original texels, so picking the
 * level whose texels are about one pixel big gives a clean result.
 *
 * Image coordinates. The functions below take (u, v) with v = 0 at the TOP
 * row of the image (the order the rows are stored in). Meshes use the OBJ
 * convention with v pointing up, so a caller passes (uv.x, 1 - uv.y).
 *
 * Colours are returned as stored, in 0..1, with no sRGB conversion:
 * the viewport shading works directly in display values (see the mesh
 * shader in gl_mesh.cpp).
 */

#include "core/common.h"
#include "core/vec3.h"

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gfx {

// A 16384 x 16384 image (kMaxImageSize) has 15 levels, so 16 is enough
constexpr int kMaxMipLevels = 16;

// A texture with its whole mip chain stored level after level in one
// RGBA8 array (level 0 first, rows top to bottom inside a level). A plain
// struct without pointers to host-only types, so an array of them can be
// copied to the GPU as is
struct TextureView {
    const std::uint8_t* texels = nullptr;  // RGBA8
    int width = 0;   // level 0
    int height = 0;
    int levels = 0;  // <= kMaxMipLevels

    // Where each level starts, counted in texels (4 bytes each), not bytes
    std::uint32_t offsets[kMaxMipLevels] = {};
};

// Side length of a mip level: halved every level (rounding down), never
// below 1. The mip chain builder and the sampler both use this one
// formula, so they always agree on the shape of the chain
GFX_HD inline int mip_size(int base, int level) {
    if (level >= 31) {
        return 1;
    }

    const int size = base >> level;
    return size > 1 ? size : 1;
}

// Maps any float to [0, 1) the way GL_REPEAT wraps coordinates: only the
// fractional part counts. NaN and absurdly large values give 0 instead of
// undefined behaviour in the float-to-int conversions that follow
GFX_HD inline float wrap_repeat(float t) {
    if (!(fabsf(t) < 1.0e30f)) {
        return 0.0f;
    }

    return t - floorf(t);
}

// One texel of one level as RGB in 0..1. x and y must be inside the level
GFX_HD inline Vec3 fetch_texel(const TextureView& texture,
                               int level, int x, int y) {
    const std::size_t width =
        static_cast<std::size_t>(mip_size(texture.width, level));

    const std::size_t index =
        (static_cast<std::size_t>(texture.offsets[level]) +
         static_cast<std::size_t>(y) * width +
         static_cast<std::size_t>(x)) * 4;

    const std::uint8_t* p = texture.texels + index;
    const float scale = 1.0f / 255.0f;

    return {static_cast<float>(p[0]) * scale,
            static_cast<float>(p[1]) * scale,
            static_cast<float>(p[2]) * scale};
}

// Bilinear filtering inside one level: the four texels around the sample
// point are blended by how close the point is to each. Texel i of a level
// that is `size` texels wide has its centre at (i + 0.5) / size, so a
// sample exactly at a centre returns that texel unchanged. Coordinates
// outside 0..1 repeat (GL_REPEAT); a sample between the last and the first
// texel blends those two. A level outside the chain is clamped
GFX_HD inline Vec3 sample_bilinear(const TextureView& texture,
                                   int level, float u, float v) {
    if (texture.texels == nullptr || texture.levels < 1) {
        return {1.0f, 1.0f, 1.0f};  // no texture: white, like a missing one
    }

    level = level < 0 ? 0
          : (level >= texture.levels ? texture.levels - 1 : level);

    const int w = mip_size(texture.width, level);
    const int h = mip_size(texture.height, level);

    // Position in texel units, shifted so that texel centres are integers.
    // The wrapped coordinate is in [0, 1], so x is in [-0.5, w - 0.5] and
    // floor(x) is in [-1, w - 1]: the neighbours below are at most one
    // step outside the level and are wrapped with a compare, not a modulo
    const float x = wrap_repeat(u) * static_cast<float>(w) - 0.5f;
    const float y = wrap_repeat(v) * static_cast<float>(h) - 0.5f;

    const float fx_floor = floorf(x);
    const float fy_floor = floorf(y);

    const float fx = x - fx_floor;  // weight of the right texel
    const float fy = y - fy_floor;  // weight of the lower texel

    const int x0 = static_cast<int>(fx_floor);
    const int y0 = static_cast<int>(fy_floor);

    const int ix0 = x0 < 0 ? w - 1 : x0;
    const int iy0 = y0 < 0 ? h - 1 : y0;
    const int ix1 = x0 + 1 >= w ? 0 : x0 + 1;
    const int iy1 = y0 + 1 >= h ? 0 : y0 + 1;

    const Vec3 t00 = fetch_texel(texture, level, ix0, iy0);
    const Vec3 t10 = fetch_texel(texture, level, ix1, iy0);
    const Vec3 t01 = fetch_texel(texture, level, ix0, iy1);
    const Vec3 t11 = fetch_texel(texture, level, ix1, iy1);

    const Vec3 top = t00 + (t10 - t00) * fx;
    const Vec3 bottom = t01 + (t11 - t01) * fx;

    return top + (bottom - top) * fy;
}

// Trilinear filtering: bilinear samples from the two mip levels on either
// side of lod, blended by the fraction. lod 0 is the original image,
// lod 1 the half-size level, and 1.5 is halfway between those two. The
// result changes smoothly with lod, so there is no visible seam where one
// level takes over from the next. lod is clamped to [0, levels - 1]
// (a NaN counts as 0)
GFX_HD inline Vec3 sample_trilinear(const TextureView& texture,
                                    float u, float v, float lod) {
    if (texture.texels == nullptr || texture.levels < 1) {
        return {1.0f, 1.0f, 1.0f};
    }

    const float max_lod = static_cast<float>(texture.levels - 1);

    lod = !(lod > 0.0f) ? 0.0f : (lod > max_lod ? max_lod : lod);

    const int lower = static_cast<int>(lod);  // floor, since lod >= 0
    const float blend = lod - static_cast<float>(lower);

    if (blend <= 0.0f || lower + 1 >= texture.levels) {
        return sample_bilinear(texture, lower, u, v);
    }

    const Vec3 a = sample_bilinear(texture, lower, u, v);
    const Vec3 b = sample_bilinear(texture, lower + 1, u, v);

    return a + (b - a) * blend;
}

// Which mip level a pixel needs, as a fractional level for trilinear
// filtering. The inputs are how much the texture coordinates change when
// the pixel moves one step right (dudx, dvdx) and one step down (dudy,
// dvdy), in image coordinates (0..1 across the whole texture).
//
// This is OpenGL's formula: scale the changes to level-0 texels, so each
// of the two steps becomes a length in texels, take the longer one (rho)
// and use log2(rho). One pixel covering 1 texel gives lod 0, covering
// 2 texels gives 1, 4 texels gives 2, and so on: every halving of the
// texture is one level. Magnification (rho < 1) gives 0, never negative.
// The footprint is taken as the longer side, not the area, which is what
// "no anisotropic filtering" means: a stretched footprint is blurred
// along its short side too
GFX_HD inline float texture_lod(const TextureView& texture,
                                float dudx, float dvdx,
                                float dudy, float dvdy) {
    const float w = static_cast<float>(texture.width);
    const float h = static_cast<float>(texture.height);

    const float x_u = dudx * w;
    const float x_v = dvdx * h;
    const float y_u = dudy * w;
    const float y_v = dvdy * h;

    const float len_x = sqrtf(x_u * x_u + x_v * x_v);
    const float len_y = sqrtf(y_u * y_u + y_v * y_v);

    // fmaxf ignores a NaN operand; the test below also rejects NaN
    const float rho = fmaxf(len_x, len_y);

    if (!(rho > 1.0f)) {
        return 0.0f;
    }

    return log2f(rho);
}

} // namespace gfx
