#pragma once

/*
 * Image files for textures: PNG and TGA, decoded from scratch.
 *
 * Every decoder produces the same thing: 8-bit RGBA, rows from the top
 * of the image down, no padding. Grey, palette and RGB images are
 * expanded; images without transparency get alpha 255; 16-bit channels
 * keep their high byte.
 */

#include "core/texture_sampling.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

struct Image {
    int width = 0;
    int height = 0;

    // width * height * 4 bytes: R, G, B, A for each pixel, top row first
    std::vector<std::uint8_t> pixels;

    bool valid() const {
        return width > 0 && height > 0 &&
               pixels.size() == static_cast<std::size_t>(width) *
                                    static_cast<std::size_t>(height) * 4;
    }
};

// Largest width or height accepted, to reject corrupt headers before
// allocating
constexpr int kMaxImageSize = 16384;

// PNG: all colour types and bit depths of the standard, with or without
// Adam7 interlacing, with palette and colour-key transparency (tRNS).
// Chunk CRCs and the zlib checksum are verified
bool decode_png(const std::uint8_t* data, std::size_t size, Image& out,
                std::string& error);

// TGA: uncompressed and run-length encoded, true colour (24/32 bit) and
// greyscale (8 bit), either row order
bool decode_tga(const std::uint8_t* data, std::size_t size, Image& out,
                std::string& error);

// Reads a file and decodes it: PNG is recognised by its signature, TGA
// (which has none) by the .tga extension
bool load_image(const std::string& path, Image& out, std::string& error);

// The mip chain of an image (see texture_sampling.h): level 0 is a copy of
// the image, every next level has half the width and height (rounded down,
// at least 1) down to 1x1, so a 5x3 image gives 5x3, 2x1 and 1x1.
//
// Each texel of a level is the average of the source texels it covers,
// weighted by how much of each source texel lies inside it. For even sizes
// that is the plain average of a 2x2 block. For an odd size the blocks
// overlap and the middle source texel is shared between two neighbours
// (5 -> 2: weights 2, 2, 1 and 1, 2, 2 over five texels), which keeps
// every source texel's weight in the result equal. Channels, alpha
// included, are averaged separately and rounded to the nearest value.
// An invalid image gives an empty list
std::vector<Image> build_mip_chain(const Image& level0);

// Stores a mip chain level after level in one RGBA8 array, fills `view`
// with its size, level count and the start of each level (in texels) and
// points view.texels at the returned array. The pointer stays valid while
// the returned vector's buffer lives (moving the vector keeps it). A
// caller that copies the array elsewhere (the CUDA backend: device memory)
// replaces view.texels with its own pointer. Returns an empty vector, and
// leaves `view` empty, for an empty or inconsistent chain: levels must
// follow mip_size() and there must be at most kMaxMipLevels of them
std::vector<std::uint8_t> pack_mip_chain(const std::vector<Image>& chain,
                                         TextureView& view);

} // namespace gfx
