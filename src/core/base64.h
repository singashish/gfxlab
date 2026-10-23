#pragma once

/*
 * Base64 (RFC 4648, section 4), for the data: URIs that glTF files use
 * to embed buffers and images in their JSON text.
 *
 * Base64 writes every 3 bytes as 4 characters from a 64-letter alphabet
 * (A-Z a-z 0-9 + /), 6 bits per character. If the byte count is not a
 * multiple of 3, the last group is shorter and may be padded with '='.
 */

#include <cstdint>
#include <string_view>
#include <vector>

namespace gfx {

// Decodes standard-alphabet base64 into out (cleared first). The '='
// padding is optional, but when it is there it must be correct (a total
// length that is a multiple of 4, at most two of them, only at the
// end). Whitespace and any other character outside the alphabet make it
// fail, as does a length that no byte count can produce (4n + 1).
// Returns false on invalid input and leaves out empty
bool decode_base64(std::string_view text, std::vector<std::uint8_t>& out);

} // namespace gfx
