#include "core/image.h"
#include "core/inflate.h"
#include "check.h"
#include "image_test_data.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {

using Bytes = std::vector<std::uint8_t>;

// The text the zlib cases hold (same generator as the Python script)
Bytes make_words(int count) {
    static const char* const words[] = {
        "pixel ", "shader ", "texture ", "vertex ", "ray ",
        "bvh ", "warp ", "cuda\n"};

    std::uint32_t state = 12345;
    Bytes out;

    for (int i = 0; i < count; ++i) {
        state = (state * 1103515245u + 12345u) & 0x7FFFFFFFu;
        const char* w = words[(state >> 16) % 8];
        out.insert(out.end(), w, w + std::strlen(w));
    }

    return out;
}

Bytes bytes_of(const char* s) {
    return Bytes(s, s + std::strlen(s));
}

void put_be32(Bytes& b, std::size_t at, std::uint32_t v) {
    b[at] = static_cast<std::uint8_t>(v >> 24);
    b[at + 1] = static_cast<std::uint8_t>(v >> 16);
    b[at + 2] = static_cast<std::uint8_t>(v >> 8);
    b[at + 3] = static_cast<std::uint8_t>(v);
}

// ---------------------------------------------------------------------
// Checksums and zlib
// ---------------------------------------------------------------------

void test_checksums() {
    const Bytes digits = bytes_of("123456789");
    CHECK(gfx::crc32(digits.data(), digits.size()) == 0xCBF43926u);

    const Bytes wiki = bytes_of("Wikipedia");
    CHECK(gfx::adler32(wiki.data(), wiki.size()) == 0x11E60398u);

    CHECK(gfx::adler32(nullptr, 0) == 1u);
    CHECK(gfx::crc32(nullptr, 0) == 0u);

    // CRC can be computed in pieces
    const std::uint32_t first = gfx::crc32(digits.data(), 4);
    CHECK(gfx::crc32(digits.data() + 4, 5, first) == 0xCBF43926u);

    // Adler-32 over more than one 5552-byte stretch
    const Bytes big(20000, 0xFF);
    std::uint32_t a = 1;
    std::uint32_t b = 0;

    for (std::uint8_t v : big) {
        a = (a + v) % 65521;
        b = (b + a) % 65521;
    }

    CHECK(gfx::adler32(big.data(), big.size()) == ((b << 16) | a));
}

void test_zlib_cases() {
    const Bytes text = make_words(image_test_data::kTextWords);

    for (const auto& c : image_test_data::kZlibCases) {
        Bytes out;
        std::string error;

        const bool ok = gfx::zlib_decompress(c.data, c.size, out, error);

        if (!ok) {
            std::fprintf(stderr, "  %s: %s\n", c.name, error.c_str());
        }

        CHECK(ok);
        CHECK(out.size() == c.text_bytes);
        CHECK(std::equal(out.begin(), out.end(), text.begin()));
    }
}

void test_zlib_errors() {
    const auto& dynamic = image_test_data::kZlibCases[0];
    Bytes data(dynamic.data, dynamic.data + dynamic.size);
    Bytes out;
    std::string error;

    // Wrong checksum
    Bytes bad_sum = data;
    bad_sum.back() ^= 1;
    CHECK(!gfx::zlib_decompress(bad_sum.data(), bad_sum.size(), out, error));
    CHECK(error.find("checksum") != std::string::npos);

    // Cut in the middle of the compressed data
    for (std::size_t keep : {std::size_t{3}, data.size() / 2, data.size() - 5}) {
        out.clear();
        CHECK(!gfx::zlib_decompress(data.data(), keep, out, error));
    }

    // Header problems
    Bytes bad_header = data;
    bad_header[0] = 0x79;  // method 9
    CHECK(!gfx::zlib_decompress(bad_header.data(), bad_header.size(), out,
                                error));

    Bytes bad_check = data;
    bad_check[1] ^= 1;     // header checksum no longer divisible by 31
    CHECK(!gfx::zlib_decompress(bad_check.data(), bad_check.size(), out,
                                error));

    // Raw Deflate: block type 3 does not exist; a stored block whose
    // length check fails
    const std::uint8_t type3[] = {0x07, 0x00};
    out.clear();
    CHECK(!gfx::inflate(type3, sizeof(type3), out, error));

    const std::uint8_t stored_bad[] = {0x01, 0x05, 0x00, 0x00, 0x00, 1, 2, 3, 4, 5};
    out.clear();
    CHECK(!gfx::inflate(stored_bad, sizeof(stored_bad), out, error));

    // A valid stored block: "hello"
    const std::uint8_t stored_ok[] = {0x01, 0x05, 0x00, 0xFA, 0xFF,
                                      'h', 'e', 'l', 'l', 'o'};
    out.clear();
    CHECK(gfx::inflate(stored_ok, sizeof(stored_ok), out, error));
    CHECK(out == bytes_of("hello"));

    // Empty input
    out.clear();
    CHECK(!gfx::inflate(nullptr, 0, out, error));
}

// ---------------------------------------------------------------------
// PNG and TGA
// ---------------------------------------------------------------------

bool matches(const gfx::Image& image, const image_test_data::ImageCase& c) {
    return image.valid() && image.width == c.width &&
           image.height == c.height &&
           std::memcmp(image.pixels.data(), c.rgba,
                       image.pixels.size()) == 0;
}

void test_png_cases() {
    for (const auto& c : image_test_data::kPngCases) {
        gfx::Image image;
        std::string error;

        const bool ok = gfx::decode_png(c.file, c.file_size, image, error);

        if (!ok || !matches(image, c)) {
            std::fprintf(stderr, "  PNG case %s: %s\n", c.name,
                         ok ? "pixels differ" : error.c_str());
        }

        CHECK(ok);
        CHECK(matches(image, c));
    }
}

void test_tga_cases() {
    for (const auto& c : image_test_data::kTgaCases) {
        gfx::Image image;
        std::string error;

        const bool ok = gfx::decode_tga(c.file, c.file_size, image, error);

        if (!ok || !matches(image, c)) {
            std::fprintf(stderr, "  TGA case %s: %s\n", c.name,
                         ok ? "pixels differ" : error.c_str());
        }

        CHECK(ok);
        CHECK(matches(image, c));
    }
}

void test_png_errors() {
    const auto& c = image_test_data::kPngCases[0];
    const Bytes good(c.file, c.file + c.file_size);
    gfx::Image image;
    std::string error;

    const auto decode = [&](const Bytes& b) {
        return gfx::decode_png(b.data(), b.size(), image, error);
    };

    CHECK(decode(good));

    // Not a PNG
    Bytes bad = good;
    bad[1] = 'X';
    CHECK(!decode(bad));
    CHECK(!image.valid());

    // A changed byte inside IHDR breaks its CRC
    bad = good;
    bad[16] ^= 0x01;
    CHECK(!decode(bad));
    CHECK(error.find("CRC") != std::string::npos);

    // Truncated
    for (std::size_t keep : {std::size_t{8}, std::size_t{30}, good.size() / 2,
                             good.size() - 12}) {
        CHECK(!decode(Bytes(good.begin(), good.begin() +
                                              static_cast<std::ptrdiff_t>(keep))));
    }

    // An unknown critical chunk ("ABCD") after IHDR is refused; an
    // unknown ancillary one ("abCD") is skipped
    const auto with_chunk = [&](const char* name) {
        Bytes chunk(12, 0);
        std::memcpy(chunk.data() + 4, name, 4);
        put_be32(chunk, 8, gfx::crc32(chunk.data() + 4, 4));

        Bytes b = good;
        b.insert(b.begin() + 33, chunk.begin(), chunk.end());  // after IHDR
        return b;
    };

    CHECK(!decode(with_chunk("ABCD")));
    CHECK(error.find("critical") != std::string::npos);
    CHECK(decode(with_chunk("abCD")));
    CHECK(matches(image, c));

    // A width of zero
    bad = good;
    put_be32(bad, 16, 0);
    put_be32(bad, 29, gfx::crc32(bad.data() + 12, 17));
    CHECK(!decode(bad));
}

void test_tga_errors() {
    const auto& c = image_test_data::kTgaCases[3];  // RLE
    const Bytes good(c.file, c.file + c.file_size);
    gfx::Image image;
    std::string error;

    CHECK(gfx::decode_tga(good.data(), good.size(), image, error));

    // Cut short: the RLE data runs out
    CHECK(!gfx::decode_tga(good.data(), good.size() - 3, image, error));
    CHECK(!gfx::decode_tga(good.data(), 10, image, error));

    // Colour-mapped and 16-bit images are not supported
    Bytes bad = good;
    bad[2] = 1;
    CHECK(!gfx::decode_tga(bad.data(), bad.size(), image, error));

    bad = good;
    bad[16] = 16;
    CHECK(!gfx::decode_tga(bad.data(), bad.size(), image, error));
}

void test_load_image() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "gfxlab_test_image";
    fs::create_directories(dir);

    const auto& png = image_test_data::kPngCases[0];
    const auto& tga = image_test_data::kTgaCases[1];

    {
        std::ofstream(dir / "a.png", std::ios::binary)
            .write(reinterpret_cast<const char*>(png.file),
                   static_cast<std::streamsize>(png.file_size));

        // PNG is recognised by content, whatever the extension
        std::ofstream(dir / "png_named.tga", std::ios::binary)
            .write(reinterpret_cast<const char*>(png.file),
                   static_cast<std::streamsize>(png.file_size));

        std::ofstream(dir / "b.TGA", std::ios::binary)
            .write(reinterpret_cast<const char*>(tga.file),
                   static_cast<std::streamsize>(tga.file_size));

        std::ofstream(dir / "c.bmp", std::ios::binary) << "BM";
    }

    gfx::Image image;
    std::string error;

    CHECK(gfx::load_image((dir / "a.png").string(), image, error));
    CHECK(matches(image, png));

    CHECK(gfx::load_image((dir / "png_named.tga").string(), image, error));
    CHECK(matches(image, png));

    CHECK(gfx::load_image((dir / "b.TGA").string(), image, error));
    CHECK(matches(image, tga));

    CHECK(!gfx::load_image((dir / "c.bmp").string(), image, error));
    CHECK(error.find("unsupported") != std::string::npos);

    CHECK(!gfx::load_image((dir / "missing.png").string(), image, error));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

} // namespace

int main() {
    test_checksums();
    test_zlib_cases();
    test_zlib_errors();
    test_png_cases();
    test_tga_cases();
    test_png_errors();
    test_tga_errors();
    test_load_image();

    if (g_failures == 0) {
        std::printf("All image tests passed\n");
        return 0;
    }

    std::printf("%d image test failure(s)\n", g_failures);
    return 1;
}
