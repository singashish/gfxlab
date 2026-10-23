#pragma once

/*
 * Deflate decompression (RFC 1951) and its zlib wrapper (RFC 1950),
 * written from scratch for the PNG decoder.
 *
 * Deflate data is a series of blocks. Each block is either stored (raw
 * bytes), or compressed with Huffman codes into a stream of literal
 * bytes and "copy length L from distance D back" instructions (LZ77).
 * The codes are either a fixed set from the standard or described at
 * the start of the block (dynamic).
 */

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

// Decompresses raw Deflate data, appending to out. Stops after the
// final block. On failure returns false with a short reason
bool inflate(const std::uint8_t* data, std::size_t size,
             std::vector<std::uint8_t>& out, std::string& error);

// Decompresses zlib data: a 2-byte header, Deflate data and an Adler-32
// checksum of the result, which is verified
bool zlib_decompress(const std::uint8_t* data, std::size_t size,
                     std::vector<std::uint8_t>& out, std::string& error);

// Checksums used by zlib and PNG
std::uint32_t adler32(const std::uint8_t* data, std::size_t size);
std::uint32_t crc32(const std::uint8_t* data, std::size_t size,
                    std::uint32_t crc = 0);

} // namespace gfx
