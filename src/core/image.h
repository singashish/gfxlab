#pragma once

/*
 * Image files for textures: PNG and TGA, decoded from scratch.
 *
 * Every decoder produces the same thing: 8-bit RGBA, rows from the top
 * of the image down, no padding. Grey, palette and RGB images are
 * expanded; images without transparency get alpha 255; 16-bit channels
 * keep their high byte.
 */

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

} // namespace gfx
