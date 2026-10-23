"""Generates tests/image_test_data.h for test_image.

Only needed to change or extend the test images; the generated header is
committed. Uses the Python standard library only (zlib for compression).

Every image is described by a formula, so the expected RGBA result is
computed here independently of the C++ decoder:
  - PNG files are written by a small encoder below that supports every
    colour type and bit depth, Adam7 interlacing, all five row filters
    (rows cycle through them) and data split over several IDAT chunks
  - TGA files cover uncompressed and RLE, colour and grey, both row
    orders and right-to-left rows
  - zlib streams cover stored, fixed-Huffman and dynamic-Huffman blocks

Run:  python tests/make_image_test_data.py
"""

import os
import struct
import zlib

OUT = os.path.join(os.path.dirname(__file__), "image_test_data.h")


# ---------------------------------------------------------------------
# Pixel formulas (8 or 16 bit samples)
# ---------------------------------------------------------------------

def value(x, y, c, bits):
    v = (x * 37 + y * 11 + c * 71 + x * y * 5) & 0xFFFF
    return v & ((1 << bits) - 1)


# ---------------------------------------------------------------------
# A tiny PNG encoder
# ---------------------------------------------------------------------

CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
ADAM7 = [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
         (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]


def chunk(name, data):
    body = name + data
    return (struct.pack(">I", len(data)) + body +
            struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF))


def pack_row(samples, depth):
    if depth == 8:
        return bytes(samples)
    if depth == 16:
        return b"".join(struct.pack(">H", s) for s in samples)
    out = bytearray()
    acc = 0
    nbits = 0
    for s in samples:
        acc = (acc << depth) | s
        nbits += depth
        if nbits == 8:
            out.append(acc)
            acc = 0
            nbits = 0
    if nbits:
        out.append(acc << (8 - nbits))
    return bytes(out)


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    return b if pb <= pc else c


def filter_row(kind, row, prior, bpp):
    out = bytearray()
    for i, x in enumerate(row):
        a = row[i - bpp] if i >= bpp else 0
        b = prior[i]
        c = prior[i - bpp] if i >= bpp else 0
        pred = [0, a, b, (a + b) // 2, paeth(a, b, c)][kind]
        out.append((x - pred) & 0xFF)
    return bytes([kind]) + bytes(out)


def encode_png(width, height, color_type, depth, sample_at, interlace=False,
               plte=None, trns=None, compressor=None, idat_parts=1,
               extra_chunks=()):
    """sample_at(x, y) -> list of samples for the pixel"""
    channels = CHANNELS[color_type]
    bpp = max(1, channels * depth // 8)
    passes = ADAM7 if interlace else [(0, 0, 1, 1)]
    raw = bytearray()
    row_index = 0
    for x0, y0, dx, dy in passes:
        if x0 >= width or y0 >= height:
            continue
        xs = list(range(x0, width, dx))
        prior = bytes((len(xs) * channels * depth + 7) // 8)
        for y in range(y0, height, dy):
            samples = []
            for x in xs:
                samples.extend(sample_at(x, y))
            row = pack_row(samples, depth)
            raw += filter_row(row_index % 5, row, prior, bpp)
            prior = row
            row_index += 1
    comp = (compressor or (lambda d: zlib.compress(d, 9)))(bytes(raw))
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth,
                                       color_type, 0, 0, 1 if interlace else 0))
    for name, data in extra_chunks:
        png += chunk(name, data)
    if plte is not None:
        png += chunk(b"PLTE", bytes(c for rgb in plte for c in rgb))
    if trns is not None:
        png += chunk(b"tRNS", trns)
    step = (len(comp) + idat_parts - 1) // idat_parts
    for i in range(0, len(comp), step):
        png += chunk(b"IDAT", comp[i:i + step])
    png += chunk(b"IEND", b"")
    return png


def fixed_huffman(data):
    c = zlib.compressobj(9, zlib.DEFLATED, 15, 9, zlib.Z_FIXED)
    return c.compress(data) + c.flush()


def stored(data, block=1000):
    """zlib stream of stored (uncompressed) blocks of at most block bytes"""
    out = bytearray(b"\x78\x01")
    chunks = [data[i:i + block] for i in range(0, len(data), block)] or [b""]
    for i, part in enumerate(chunks):
        out.append(1 if i == len(chunks) - 1 else 0)
        out += struct.pack("<HH", len(part), len(part) ^ 0xFFFF)
        out += part
    out += struct.pack(">I", zlib.adler32(data) & 0xFFFFFFFF)
    return bytes(out)


def to8(v, depth):
    if depth == 16:
        return v >> 8
    if depth == 8:
        return v
    return v * 255 // ((1 << depth) - 1)


# ---------------------------------------------------------------------
# PNG cases: (name, png bytes, width, height, expected rgba bytes)
# ---------------------------------------------------------------------

def png_case(name, width, height, color_type, depth, interlace=False,
             with_trns=False, palette_size=0, **kw):
    channels = CHANNELS[color_type]
    plte = None
    trns = None
    key = None

    if color_type == 3:
        plte = [((i * 53) & 255, (i * 97 + 31) & 255, (i * 13 + 200) & 255)
                for i in range(palette_size)]
        alphas = [(i * 77) & 255 for i in range(palette_size)]
        if with_trns:
            trns = bytes(alphas[:max(1, palette_size // 2)])

        def sample_at(x, y):
            return [value(x, y, 0, 16) % palette_size]
    else:
        def sample_at(x, y):
            return [value(x, y, c, depth) for c in range(channels)]

        if with_trns:
            key = sample_at(1, 0)
            if color_type == 0:
                trns = struct.pack(">H", key[0])
            else:
                trns = struct.pack(">HHH", *key)

    expected = bytearray()
    for y in range(height):
        for x in range(width):
            s = sample_at(x, y)
            if color_type == 0:
                g = to8(s[0], depth)
                a = 0 if key is not None and s == key else 255
                expected += bytes([g, g, g, a])
            elif color_type == 2:
                a = 0 if key is not None and s == key else 255
                expected += bytes([to8(v, depth) for v in s] + [a])
            elif color_type == 3:
                r, g, b = plte[s[0]]
                a = trns[s[0]] if trns is not None and s[0] < len(trns) else 255
                expected += bytes([r, g, b, a])
            elif color_type == 4:
                g = to8(s[0], depth)
                expected += bytes([g, g, g, to8(s[1], depth)])
            else:
                expected += bytes([to8(v, depth) for v in s])

    png = encode_png(width, height, color_type, depth, sample_at, interlace,
                     plte, trns, **kw)
    return (name, png, width, height, bytes(expected))


PNG_CASES = [
    png_case("rgba8", 9, 7, 6, 8),
    png_case("rgb8_key", 9, 7, 2, 8, with_trns=True),
    png_case("rgb16_key", 5, 4, 2, 16, with_trns=True),
    png_case("grey8", 9, 7, 0, 8),
    png_case("grey1", 13, 5, 0, 1),
    png_case("grey2", 11, 3, 0, 2),
    png_case("grey4_key", 7, 6, 0, 4, with_trns=True),
    png_case("grey16", 6, 5, 0, 16),
    png_case("grey_alpha8", 8, 4, 4, 8),
    png_case("grey_alpha16", 5, 3, 4, 16),
    png_case("rgba16", 4, 5, 6, 16),
    png_case("palette8", 9, 7, 3, 8, palette_size=200, with_trns=True),
    png_case("palette4", 10, 3, 3, 4, palette_size=16, with_trns=True),
    png_case("palette1", 17, 2, 3, 1, palette_size=2),
    png_case("rgba8_adam7", 11, 9, 6, 8, interlace=True),
    png_case("grey1_adam7", 10, 10, 0, 1, interlace=True),
    png_case("palette2_adam7", 9, 7, 3, 2, interlace=True, palette_size=4,
             with_trns=True),
    png_case("rgb8_adam7_tiny", 3, 2, 2, 8, interlace=True),
    png_case("grey8_1x1_adam7", 1, 1, 0, 8, interlace=True),
    png_case("rgba8_fixed_split", 12, 8, 6, 8, compressor=fixed_huffman,
             idat_parts=3, extra_chunks=[(b"gAMA", struct.pack(">I", 45455)),
                                         (b"tEXt", b"Comment\x00gfxlab")]),
    png_case("rgb8_stored", 6, 6, 2, 8, compressor=stored),
]


# ---------------------------------------------------------------------
# TGA cases
# ---------------------------------------------------------------------

def tga_case(name, width, height, image_type, depth, top_down, alpha_bits,
             right_left=False):
    grey = image_type in (3, 11)
    rle = image_type in (10, 11)

    def rgba(x, y):
        # Rows of four equal pixels so RLE finds runs, some crossing rows
        k = (x // 4) + y * 3
        return [(k * 40) & 255, (k * 90 + 7) & 255, (k * 25 + 100) & 255,
                (k * 61 + 13) & 255]

    expected = bytearray()
    for y in range(height):
        for x in range(width):
            r, g, b, a = rgba(x, y)
            if grey:
                expected += bytes([r, r, r, 255])
            else:
                expected += bytes([r, g, b, a if depth == 32 and alpha_bits
                                   else 255])

    # File order of the pixels
    pixels = []
    rows = range(height) if top_down else range(height - 1, -1, -1)
    for y in rows:
        cols = range(width - 1, -1, -1) if right_left else range(width)
        for x in cols:
            r, g, b, a = rgba(x, y)
            if grey:
                pixels.append(bytes([r]))
            elif depth == 24:
                pixels.append(bytes([b, g, r]))
            else:
                pixels.append(bytes([b, g, r, a]))

    body = bytearray()
    if not rle:
        body = b"".join(pixels)
    else:
        i = 0
        while i < len(pixels):
            run = 1
            while i + run < len(pixels) and run < 128 and \
                    pixels[i + run] == pixels[i]:
                run += 1
            if run > 1:
                body += bytes([0x80 | (run - 1)]) + pixels[i]
                i += run
            else:
                j = i
                while j < len(pixels) and j - i < 128 and \
                        (j + 1 >= len(pixels) or pixels[j + 1] != pixels[j]):
                    j += 1
                n = max(1, j - i)
                body += bytes([n - 1]) + b"".join(pixels[i:i + n])
                i += n

    descriptor = alpha_bits | (0x20 if top_down else 0) | \
        (0x10 if right_left else 0)
    image_id = b"gfxlab"
    header = struct.pack("<BBBHHBHHHHBB", len(image_id), 0, image_type, 0, 0,
                         0, 0, 0, width, height, depth, descriptor)
    return (name, header + image_id + bytes(body), width, height,
            bytes(expected))


TGA_CASES = [
    tga_case("tga24_bottom_up", 9, 5, 2, 24, False, 0),
    tga_case("tga32_top_down", 9, 5, 2, 32, True, 8),
    tga_case("tga32_no_alpha_bits", 6, 3, 2, 32, True, 0),
    tga_case("tga32_rle", 13, 6, 10, 32, False, 8),
    tga_case("tga24_rle_right_left", 10, 4, 10, 24, True, 0, right_left=True),
    tga_case("tga_grey", 7, 3, 3, 8, False, 0),
    tga_case("tga_grey_rle", 12, 5, 11, 8, True, 0),
]


# ---------------------------------------------------------------------
# zlib cases: the uncompressed data is rebuilt by the test from the
# same generator (make_words)
# ---------------------------------------------------------------------

def make_words(count):
    """Deterministic text with lots of repetition (same LCG as the test)"""
    words = [b"pixel ", b"shader ", b"texture ", b"vertex ", b"ray ",
             b"bvh ", b"warp ", b"cuda\n"]
    state = 12345
    out = bytearray()
    for _ in range(count):
        state = (state * 1103515245 + 12345) & 0x7FFFFFFF
        out += words[(state >> 16) % len(words)]
    return bytes(out)


TEXT = make_words(3000)       # ~19 KB; stored blocks are 1000 bytes
# (name, compressed, how many leading bytes of TEXT it holds)
ZLIB_CASES = [
    ("zlib_dynamic", zlib.compress(TEXT, 9), len(TEXT)),
    ("zlib_fixed", fixed_huffman(TEXT), len(TEXT)),
    ("zlib_stored", stored(TEXT[:3500]), 3500),
    ("zlib_empty", zlib.compress(b"", 9), 0),
]


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

def c_array(name, data):
    lines = []
    for i in range(0, len(data), 16):
        lines.append("    " + ", ".join("0x%02x" % b for b in data[i:i + 16]))
    return ("static const unsigned char %s[%d] = {\n%s\n};\n"
            % (name, max(1, len(data)), ",\n".join(lines) or "    0"))


def main():
    parts = ["// Generated by tests/make_image_test_data.py; do not edit\n",
             "#pragma once\n", "#include <cstddef>\n",
             "namespace image_test_data {\n",
             "struct ImageCase {\n"
             "    const char* name;\n"
             "    const unsigned char* file;\n"
             "    std::size_t file_size;\n"
             "    int width;\n"
             "    int height;\n"
             "    const unsigned char* rgba;\n"
             "};\n",
             "struct ZlibCase {\n"
             "    const char* name;\n"
             "    const unsigned char* data;\n"
             "    std::size_t size;\n"
             "    std::size_t text_bytes;  // leading bytes of the text\n"
             "};\n",
             "constexpr int kTextWords = 3000;\n"]

    for group, cases in (("png", PNG_CASES), ("tga", TGA_CASES)):
        for name, file, w, h, rgba in cases:
            parts.append(c_array(name + "_file", file))
            parts.append(c_array(name + "_rgba", rgba))
        parts.append("static const ImageCase k%sCases[] = {\n" %
                     group.capitalize())
        for name, file, w, h, rgba in cases:
            parts.append('    {"%s", %s_file, sizeof(%s_file), %d, %d, %s_rgba},\n'
                         % (name, name, name, w, h, name))
        parts.append("};\n")

    for name, data, _ in ZLIB_CASES:
        parts.append(c_array(name, data))
    parts.append("static const ZlibCase kZlibCases[] = {\n")
    for name, data, text_bytes in ZLIB_CASES:
        parts.append('    {"%s", %s, %d, %d},\n'
                     % (name, name, len(data), text_bytes))
    parts.append("};\n")
    parts.append("} // namespace image_test_data\n")

    with open(OUT, "w", newline="\n") as f:
        f.write("\n".join(parts))

    print("wrote", OUT)


if __name__ == "__main__":
    main()
