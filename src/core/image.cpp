#include "core/image.h"
#include "core/inflate.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>

namespace gfx {

namespace {

bool fail(Image& out, std::string& error, const std::string& message) {
    out = Image{};
    error = message;
    return false;
}

std::uint32_t read_be32(const std::uint8_t* p) {
    return (static_cast<std::uint32_t>(p[0]) << 24) |
           (static_cast<std::uint32_t>(p[1]) << 16) |
           (static_cast<std::uint32_t>(p[2]) << 8) |
           static_cast<std::uint32_t>(p[3]);
}

std::uint16_t read_le16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

constexpr std::uint8_t kPngSignature[8] = {
    0x89, 'P', 'N', 'G', '\r', '\n', 0x1A, '\n'};

// ---------------------------------------------------------------------
// PNG
// ---------------------------------------------------------------------

struct PngHeader {
    int width = 0;
    int height = 0;
    int bit_depth = 0;
    int color_type = 0;
    bool interlaced = false;
};

int png_channels(int color_type) {
    switch (color_type) {
    case 0: return 1;   // grey
    case 2: return 3;   // RGB
    case 3: return 1;   // palette index
    case 4: return 2;   // grey + alpha
    case 6: return 4;   // RGBA
    default: return 0;
    }
}

bool png_depth_allowed(int color_type, int bit_depth) {
    switch (color_type) {
    case 0:
        return bit_depth == 1 || bit_depth == 2 || bit_depth == 4 ||
               bit_depth == 8 || bit_depth == 16;
    case 3:
        return bit_depth == 1 || bit_depth == 2 || bit_depth == 4 ||
               bit_depth == 8;
    case 2:
    case 4:
    case 6:
        return bit_depth == 8 || bit_depth == 16;
    default:
        return false;
    }
}

// The predictor of filter type 4
int paeth(int a, int b, int c) {
    const int p = a + b - c;
    const int pa = std::abs(p - a);
    const int pb = std::abs(p - b);
    const int pc = std::abs(p - c);

    if (pa <= pb && pa <= pc) {
        return a;
    }

    return pb <= pc ? b : c;
}

// Undoes a row's filter in place. bpp: bytes per complete pixel (at
// least 1); prior: the previous row of the same pass, already
// unfiltered (all zeros for the first row)
bool unfilter_row(int filter, std::uint8_t* row, const std::uint8_t* prior,
                  std::size_t length, std::size_t bpp) {
    switch (filter) {
    case 0:  // none
        return true;

    case 1:  // sub: plus the byte one pixel to the left
        for (std::size_t i = bpp; i < length; ++i) {
            row[i] = static_cast<std::uint8_t>(row[i] + row[i - bpp]);
        }
        return true;

    case 2:  // up: plus the byte above
        for (std::size_t i = 0; i < length; ++i) {
            row[i] = static_cast<std::uint8_t>(row[i] + prior[i]);
        }
        return true;

    case 3:  // average of left and above
        for (std::size_t i = 0; i < length; ++i) {
            const int left = i >= bpp ? row[i - bpp] : 0;
            row[i] = static_cast<std::uint8_t>(row[i] + ((left + prior[i]) >> 1));
        }
        return true;

    case 4:  // Paeth: whichever of left, above, above-left is closest
        for (std::size_t i = 0; i < length; ++i) {
            const int left = i >= bpp ? row[i - bpp] : 0;
            const int up_left = i >= bpp ? prior[i - bpp] : 0;
            row[i] = static_cast<std::uint8_t>(
                row[i] + paeth(left, prior[i], up_left));
        }
        return true;

    default:
        return false;
    }
}

// Everything needed to turn unfiltered samples into RGBA
struct PngConverter {
    PngHeader header;
    int channels = 0;

    std::uint8_t palette[256][4] = {};
    int palette_size = 0;

    // Colour key from tRNS for grey or RGB images, at full precision
    bool has_key = false;
    std::uint16_t key[3] = {};

    // Sample i (in channel order across the row) at the file's depth
    std::uint16_t sample(const std::uint8_t* row, std::size_t i) const {
        const int depth = header.bit_depth;

        if (depth == 8) {
            return row[i];
        }

        if (depth == 16) {
            return static_cast<std::uint16_t>((row[i * 2] << 8) | row[i * 2 + 1]);
        }

        // 1, 2 or 4 bits, packed from the most significant bit
        const std::size_t bit = i * static_cast<std::size_t>(depth);
        const int shift = 8 - depth - static_cast<int>(bit % 8);
        return static_cast<std::uint16_t>(
            (row[bit / 8] >> shift) & ((1 << depth) - 1));
    }

    // A sample scaled to 8 bits
    std::uint8_t to8(std::uint16_t value) const {
        switch (header.bit_depth) {
        case 16: return static_cast<std::uint8_t>(value >> 8);
        case 8: return static_cast<std::uint8_t>(value);
        default:
            return static_cast<std::uint8_t>(
                value * 255 / ((1 << header.bit_depth) - 1));
        }
    }

    // Writes pixel x of an unfiltered row as RGBA
    bool pixel(const std::uint8_t* row, std::size_t x, std::uint8_t* rgba) const {
        const std::size_t base = x * static_cast<std::size_t>(channels);

        switch (header.color_type) {
        case 0: {
            const std::uint16_t g = sample(row, base);
            const std::uint8_t v = to8(g);
            rgba[0] = rgba[1] = rgba[2] = v;
            rgba[3] = (has_key && g == key[0]) ? 0 : 255;
            return true;
        }
        case 2: {
            const std::uint16_t r = sample(row, base);
            const std::uint16_t g = sample(row, base + 1);
            const std::uint16_t b = sample(row, base + 2);
            rgba[0] = to8(r);
            rgba[1] = to8(g);
            rgba[2] = to8(b);
            rgba[3] = (has_key && r == key[0] && g == key[1] && b == key[2])
                ? 0 : 255;
            return true;
        }
        case 3: {
            const std::uint16_t index = sample(row, base);

            if (index >= palette_size) {
                return false;
            }

            std::copy(palette[index], palette[index] + 4, rgba);
            return true;
        }
        case 4: {
            const std::uint8_t v = to8(sample(row, base));
            rgba[0] = rgba[1] = rgba[2] = v;
            rgba[3] = to8(sample(row, base + 1));
            return true;
        }
        case 6:
            for (int c = 0; c < 4; ++c) {
                rgba[c] = to8(sample(row, base + static_cast<std::size_t>(c)));
            }
            return true;
        default:
            return false;
        }
    }
};

// Adam7: seven passes over sub-grids of the image
constexpr int kAdamX0[7] = {0, 4, 0, 2, 0, 1, 0};
constexpr int kAdamY0[7] = {0, 0, 4, 0, 2, 0, 1};
constexpr int kAdamDx[7] = {8, 8, 4, 4, 2, 2, 1};
constexpr int kAdamDy[7] = {8, 8, 8, 4, 4, 2, 2};

} // namespace

bool decode_png(const std::uint8_t* data, std::size_t size, Image& out,
                std::string& error) {
    out = Image{};
    error.clear();

    if (size < 8 || !std::equal(kPngSignature, kPngSignature + 8, data)) {
        return fail(out, error, "not a PNG file");
    }

    PngConverter conv;
    bool has_header = false;
    bool has_end = false;
    std::vector<std::uint8_t> compressed;

    // 1. Chunks: length, type, data, CRC of type + data
    std::size_t pos = 8;

    while (pos + 12 <= size) {
        const std::uint32_t length = read_be32(data + pos);
        const std::uint8_t* type = data + pos + 4;

        if (length > size - pos - 12) {
            return fail(out, error, "chunk runs past the end of the file");
        }

        const std::uint8_t* body = type + 4;

        if (crc32(type, length + 4) != read_be32(body + length)) {
            return fail(out, error, "chunk CRC mismatch");
        }

        const std::string name(reinterpret_cast<const char*>(type), 4);
        pos += 12 + length;

        if (!has_header && name != "IHDR") {
            return fail(out, error, "IHDR is not the first chunk");
        }

        if (name == "IHDR") {
            if (has_header || length != 13) {
                return fail(out, error, "invalid IHDR");
            }

            PngHeader& h = conv.header;
            const std::uint32_t w = read_be32(body);
            const std::uint32_t ht = read_be32(body + 4);
            h.bit_depth = body[8];
            h.color_type = body[9];
            h.interlaced = body[12] == 1;

            if (w == 0 || ht == 0 ||
                w > static_cast<std::uint32_t>(kMaxImageSize) ||
                ht > static_cast<std::uint32_t>(kMaxImageSize)) {
                return fail(out, error, "unsupported image size");
            }

            if (!png_depth_allowed(h.color_type, h.bit_depth)) {
                return fail(out, error, "invalid colour type / bit depth");
            }

            if (body[10] != 0 || body[11] != 0 || body[12] > 1) {
                return fail(out, error,
                            "unknown compression, filter or interlace method");
            }

            h.width = static_cast<int>(w);
            h.height = static_cast<int>(ht);
            conv.channels = png_channels(h.color_type);
            has_header = true;
        } else if (name == "PLTE") {
            if (length % 3 != 0 || length / 3 > 256 || length == 0) {
                return fail(out, error, "invalid palette");
            }

            conv.palette_size = static_cast<int>(length / 3);

            for (int i = 0; i < conv.palette_size; ++i) {
                conv.palette[i][0] = body[i * 3];
                conv.palette[i][1] = body[i * 3 + 1];
                conv.palette[i][2] = body[i * 3 + 2];
                conv.palette[i][3] = 255;
            }
        } else if (name == "tRNS") {
            const int ct = conv.header.color_type;

            if (ct == 3) {
                // Alpha for the first palette entries
                for (std::uint32_t i = 0; i < length && i < 256; ++i) {
                    conv.palette[i][3] = body[i];
                }
            } else if (ct == 0 && length == 2) {
                conv.has_key = true;
                conv.key[0] = static_cast<std::uint16_t>((body[0] << 8) | body[1]);
            } else if (ct == 2 && length == 6) {
                conv.has_key = true;

                for (int c = 0; c < 3; ++c) {
                    conv.key[c] = static_cast<std::uint16_t>(
                        (body[c * 2] << 8) | body[c * 2 + 1]);
                }
            }
            // tRNS is not allowed with an alpha channel; ignore it then
        } else if (name == "IDAT") {
            compressed.insert(compressed.end(), body, body + length);
        } else if (name == "IEND") {
            has_end = true;
            break;
        } else if ((type[0] & 0x20) == 0) {
            // Upper-case first letter: a "critical" chunk the decoder
            // must understand. Lower-case ones (gAMA, sRGB, tEXt, ...)
            // are optional and skipped
            return fail(out, error, "unknown critical chunk " + name);
        }
    }

    if (!has_header || !has_end) {
        return fail(out, error, "missing IHDR or IEND");
    }

    if (conv.header.color_type == 3 && conv.palette_size == 0) {
        return fail(out, error, "palette image without PLTE");
    }

    if (compressed.empty()) {
        return fail(out, error, "no image data");
    }

    // 2. Decompress all IDAT data as one zlib stream. Its size is known
    // from the header (rows of filter byte + samples), so reserve it
    const PngHeader& hdr = conv.header;
    const std::size_t sample_bits =
        static_cast<std::size_t>(conv.channels * hdr.bit_depth);
    std::size_t expected_size = 0;

    for (int pass = 0; pass < (hdr.interlaced ? 7 : 1); ++pass) {
        const int x0 = hdr.interlaced ? kAdamX0[pass] : 0;
        const int y0 = hdr.interlaced ? kAdamY0[pass] : 0;
        const int dx = hdr.interlaced ? kAdamDx[pass] : 1;
        const int dy = hdr.interlaced ? kAdamDy[pass] : 1;

        if (x0 < hdr.width && y0 < hdr.height) {
            const std::size_t w =
                static_cast<std::size_t>((hdr.width - x0 + dx - 1) / dx);
            const std::size_t rows =
                static_cast<std::size_t>((hdr.height - y0 + dy - 1) / dy);
            expected_size += rows * (1 + (w * sample_bits + 7) / 8);
        }
    }

    std::vector<std::uint8_t> raw;
    raw.reserve(expected_size);
    std::string zlib_error;

    if (!zlib_decompress(compressed.data(), compressed.size(), raw,
                         zlib_error)) {
        return fail(out, error, "image data: " + zlib_error);
    }

    const PngHeader& h = conv.header;
    const std::size_t bits_per_pixel =
        static_cast<std::size_t>(conv.channels * h.bit_depth);
    const std::size_t bpp = std::max<std::size_t>(1, bits_per_pixel / 8);

    out.width = h.width;
    out.height = h.height;
    out.pixels.assign(static_cast<std::size_t>(h.width) *
                          static_cast<std::size_t>(h.height) * 4,
                      0);

    // 3. Unfilter each pass (one pass without interlacing) and write its
    // pixels to their places in the image
    std::size_t offset = 0;
    const int passes = h.interlaced ? 7 : 1;

    for (int pass = 0; pass < passes; ++pass) {
        const int x0 = h.interlaced ? kAdamX0[pass] : 0;
        const int y0 = h.interlaced ? kAdamY0[pass] : 0;
        const int dx = h.interlaced ? kAdamDx[pass] : 1;
        const int dy = h.interlaced ? kAdamDy[pass] : 1;

        if (x0 >= h.width || y0 >= h.height) {
            continue;  // this pass is empty for a small image
        }

        const std::size_t pass_width =
            static_cast<std::size_t>((h.width - x0 + dx - 1) / dx);
        const int pass_height = (h.height - y0 + dy - 1) / dy;
        const std::size_t row_bytes = (pass_width * bits_per_pixel + 7) / 8;

        std::vector<std::uint8_t> prior(row_bytes, 0);

        for (int y = 0; y < pass_height; ++y) {
            if (offset + 1 + row_bytes > raw.size()) {
                return fail(out, error, "image data too short");
            }

            const int filter = raw[offset];
            std::uint8_t* row = raw.data() + offset + 1;
            offset += 1 + row_bytes;

            if (!unfilter_row(filter, row, prior.data(), row_bytes, bpp)) {
                return fail(out, error, "invalid filter type");
            }

            const std::size_t out_y =
                static_cast<std::size_t>(y0 + y * dy);

            for (std::size_t x = 0; x < pass_width; ++x) {
                const std::size_t out_x =
                    static_cast<std::size_t>(x0) + x * static_cast<std::size_t>(dx);
                std::uint8_t* dst =
                    out.pixels.data() +
                    (out_y * static_cast<std::size_t>(h.width) + out_x) * 4;

                if (!conv.pixel(row, x, dst)) {
                    return fail(out, error, "palette index out of range");
                }
            }

            std::copy(row, row + row_bytes, prior.begin());
        }
    }

    return true;
}

// ---------------------------------------------------------------------
// TGA
// ---------------------------------------------------------------------

bool decode_tga(const std::uint8_t* data, std::size_t size, Image& out,
                std::string& error) {
    out = Image{};
    error.clear();

    if (size < 18) {
        return fail(out, error, "TGA file too short");
    }

    const int id_length = data[0];
    const int colormap_type = data[1];
    const int image_type = data[2];
    const int colormap_length = read_le16(data + 5);
    const int colormap_bits = data[7];
    const int width = read_le16(data + 12);
    const int height = read_le16(data + 14);
    const int depth = data[16];
    const int descriptor = data[17];

    const bool rle = image_type == 10 || image_type == 11;
    const bool grey = image_type == 3 || image_type == 11;

    if (image_type != 2 && image_type != 3 && !rle) {
        return fail(out, error,
                    "unsupported TGA type (only true colour and grey)");
    }

    if ((!grey && depth != 24 && depth != 32) || (grey && depth != 8)) {
        return fail(out, error, "unsupported TGA pixel depth");
    }

    if (width == 0 || height == 0 || width > kMaxImageSize ||
        height > kMaxImageSize) {
        return fail(out, error, "unsupported image size");
    }

    // Skip the image ID and any colour map (unused by these types)
    std::size_t pos = 18 + static_cast<std::size_t>(id_length);

    if (colormap_type == 1) {
        pos += static_cast<std::size_t>(colormap_length) *
               static_cast<std::size_t>((colormap_bits + 7) / 8);
    }

    const std::size_t bytes = static_cast<std::size_t>(depth / 8);
    const std::size_t count =
        static_cast<std::size_t>(width) * static_cast<std::size_t>(height);

    // Pixels in file order, still in the file's layout (BGR(A) or grey)
    std::vector<std::uint8_t> file_pixels;
    file_pixels.reserve(count * bytes);

    if (!rle) {
        if (pos + count * bytes > size) {
            return fail(out, error, "TGA pixel data too short");
        }

        file_pixels.assign(data + pos, data + pos + count * bytes);
    } else {
        // Packets: a header byte, then either one pixel repeated
        // (high bit set) or (low 7 bits + 1) literal pixels
        while (file_pixels.size() < count * bytes) {
            if (pos >= size) {
                return fail(out, error, "TGA RLE data too short");
            }

            const int header = data[pos++];
            const std::size_t n = static_cast<std::size_t>(header & 0x7F) + 1;

            if (header & 0x80) {
                if (pos + bytes > size) {
                    return fail(out, error, "TGA RLE data too short");
                }

                for (std::size_t i = 0; i < n; ++i) {
                    file_pixels.insert(file_pixels.end(), data + pos,
                                       data + pos + bytes);
                }

                pos += bytes;
            } else {
                if (pos + n * bytes > size) {
                    return fail(out, error, "TGA RLE data too short");
                }

                file_pixels.insert(file_pixels.end(), data + pos,
                                   data + pos + n * bytes);
                pos += n * bytes;
            }
        }

        // A run may overshoot the last pixel; drop the extra
        file_pixels.resize(count * bytes);
    }

    // Bit 5 of the descriptor: rows top to bottom (else bottom to top).
    // Bit 4: right to left. The low 4 bits count the alpha bits; files
    // that say 0 carry no real alpha even at 32 bits
    const bool top_down = (descriptor & 0x20) != 0;
    const bool right_left = (descriptor & 0x10) != 0;
    const bool use_alpha = depth == 32 && (descriptor & 0x0F) != 0;

    out.width = width;
    out.height = height;
    out.pixels.resize(count * 4);

    for (int y = 0; y < height; ++y) {
        const int src_y = top_down ? y : height - 1 - y;

        for (int x = 0; x < width; ++x) {
            const int src_x = right_left ? width - 1 - x : x;
            const std::uint8_t* src =
                file_pixels.data() +
                (static_cast<std::size_t>(src_y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(src_x)) * bytes;
            std::uint8_t* dst =
                out.pixels.data() +
                (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                 static_cast<std::size_t>(x)) * 4;

            if (grey) {
                dst[0] = dst[1] = dst[2] = src[0];
                dst[3] = 255;
            } else {
                dst[0] = src[2];   // stored as B, G, R(, A)
                dst[1] = src[1];
                dst[2] = src[0];
                dst[3] = use_alpha ? src[3] : 255;
            }
        }
    }

    return true;
}

bool load_image(const std::string& path, Image& out, std::string& error) {
    out = Image{};
    error.clear();

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        error = "could not open '" + path + "'";
        return false;
    }

    const std::vector<std::uint8_t> bytes(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());

    if (bytes.size() >= 8 &&
        std::equal(kPngSignature, kPngSignature + 8, bytes.data())) {
        return decode_png(bytes.data(), bytes.size(), out, error);
    }

    std::string extension = std::filesystem::path(path).extension().string();
    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });

    if (extension == ".tga") {
        return decode_tga(bytes.data(), bytes.size(), out, error);
    }

    error = "unsupported image format (PNG and TGA are supported): " + path;
    return false;
}

} // namespace gfx
