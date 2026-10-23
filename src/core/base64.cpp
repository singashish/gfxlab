#include "core/base64.h"

#include <cstddef>

namespace gfx {

namespace {

// Value 0..63 of a base64 character, or -1 when it is not one
int base64_value(char c) {
    if (c >= 'A' && c <= 'Z') {
        return c - 'A';
    }
    if (c >= 'a' && c <= 'z') {
        return c - 'a' + 26;
    }
    if (c >= '0' && c <= '9') {
        return c - '0' + 52;
    }
    if (c == '+') {
        return 62;
    }
    if (c == '/') {
        return 63;
    }
    return -1;
}

} // namespace

bool decode_base64(std::string_view text, std::vector<std::uint8_t>& out) {
    out.clear();

    // Padding: up to two '=' at the very end. With padding the whole
    // text must be a multiple of 4 characters long
    std::size_t padding = 0;

    while (padding < 2 && padding < text.size() &&
           text[text.size() - 1 - padding] == '=') {
        ++padding;
    }

    if (padding > 0 && text.size() % 4 != 0) {
        return false;
    }

    const std::string_view body = text.substr(0, text.size() - padding);

    // 4n+1 characters would carry 6 bits more than 4n but not enough
    // for another byte: no byte string encodes to that
    if (body.size() % 4 == 1) {
        return false;
    }

    out.reserve(body.size() / 4 * 3 + 2);

    // Collect 6 bits per character; every time 8 or more bits are
    // there, a byte is complete. Leftover bits (at most 4) are the
    // zero padding of the last group and are dropped
    std::uint32_t bits = 0;
    int bit_count = 0;

    for (const char c : body) {
        const int value = base64_value(c);

        if (value < 0) {
            out.clear();
            return false;
        }

        bits = (bits << 6) | static_cast<std::uint32_t>(value);
        bit_count += 6;

        if (bit_count >= 8) {
            bit_count -= 8;
            out.push_back(static_cast<std::uint8_t>((bits >> bit_count) & 0xFFu));

            // Forget the bits already used
            bits &= (1u << bit_count) - 1u;
        }
    }

    return true;
}

} // namespace gfx
