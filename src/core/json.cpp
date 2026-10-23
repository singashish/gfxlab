#include "core/json.h"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace gfx {

// ---------------------------------------------------------------------
// JsonValue
// ---------------------------------------------------------------------

namespace {

const std::string kEmptyString;
const std::vector<JsonValue> kNoElements;
const std::vector<JsonMember> kNoMembers;

} // namespace

const JsonValue& JsonValue::null_value() {
    // A function-local static is created on first use, which avoids
    // problems with the order in which globals are initialised
    static const JsonValue kNull;
    return kNull;
}

bool JsonValue::as_bool(bool fallback) const {
    return m_type == JsonType::JSON_BOOL ? m_bool : fallback;
}

double JsonValue::as_number(double fallback) const {
    return m_type == JsonType::JSON_NUMBER ? m_number : fallback;
}

float JsonValue::as_float(float fallback) const {
    if (m_type != JsonType::JSON_NUMBER) {
        return fallback;
    }

    // Converting a double that is too big for float is undefined
    // behaviour in C++, so clamp first
    const double limit =
        static_cast<double>(std::numeric_limits<float>::max());

    if (m_number > limit) {
        return std::numeric_limits<float>::max();
    }
    if (m_number < -limit) {
        return std::numeric_limits<float>::lowest();
    }

    return static_cast<float>(m_number);
}

int JsonValue::as_int(int fallback) const {
    if (m_type != JsonType::JSON_NUMBER) {
        return fallback;
    }

    // Check the range before converting (an out-of-range conversion is
    // undefined behaviour), then convert back and compare to find out
    // whether the number was whole
    const double low = static_cast<double>(std::numeric_limits<int>::min());
    const double high = static_cast<double>(std::numeric_limits<int>::max());

    if (!(m_number >= low && m_number <= high)) {
        return fallback;
    }

    const int value = static_cast<int>(m_number);

    return static_cast<double>(value) == m_number ? value : fallback;
}

const std::string& JsonValue::as_string() const {
    return m_type == JsonType::JSON_STRING ? m_string : kEmptyString;
}

std::size_t JsonValue::size() const {
    if (m_type == JsonType::JSON_ARRAY) {
        return m_elements.size();
    }
    if (m_type == JsonType::JSON_OBJECT) {
        return m_members.size();
    }

    return 0;
}

const JsonValue& JsonValue::operator[](std::size_t index) const {
    if (m_type != JsonType::JSON_ARRAY || index >= m_elements.size()) {
        return null_value();
    }

    return m_elements[index];
}

const JsonValue& JsonValue::operator[](std::string_view key) const {
    if (m_type == JsonType::JSON_OBJECT) {
        // Linear search: glTF objects have a handful of members, so
        // this is faster than building a hash table for each of them
        for (const JsonMember& member : m_members) {
            if (member.key == key) {
                return member.value;
            }
        }
    }

    return null_value();
}

bool JsonValue::contains(std::string_view key) const {
    if (m_type == JsonType::JSON_OBJECT) {
        for (const JsonMember& member : m_members) {
            if (member.key == key) {
                return true;
            }
        }
    }

    return false;
}

const std::vector<JsonValue>& JsonValue::elements() const {
    return m_type == JsonType::JSON_ARRAY ? m_elements : kNoElements;
}

const std::vector<JsonMember>& JsonValue::members() const {
    return m_type == JsonType::JSON_OBJECT ? m_members : kNoMembers;
}

// ---------------------------------------------------------------------
// Parser
// ---------------------------------------------------------------------

namespace {

bool is_digit(char c) {
    return c >= '0' && c <= '9';
}

// The value of a hex digit (either case), or -1
int hex_value(char c) {
    if (c >= '0' && c <= '9') {
        return c - '0';
    }
    if (c >= 'a' && c <= 'f') {
        return c - 'a' + 10;
    }
    if (c >= 'A' && c <= 'F') {
        return c - 'A' + 10;
    }

    return -1;
}

// Appends a Unicode code point (0 to 0x10FFFF, not a surrogate) as
// UTF-8: 1 to 4 bytes. The first byte tells how many follow, and each
// following byte starts with the bits 10 and carries 6 bits of the value
void append_utf8(std::string& out, std::uint32_t code_point) {
    if (code_point < 0x80) {
        out.push_back(static_cast<char>(code_point));
    } else if (code_point < 0x800) {
        out.push_back(static_cast<char>(0xC0 | (code_point >> 6)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else if (code_point < 0x10000) {
        out.push_back(static_cast<char>(0xE0 | (code_point >> 12)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | (code_point >> 18)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code_point >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code_point & 0x3F)));
    }
}

// Describes a byte for an error message: the character in quotes when
// it is printable, else its value in hex
std::string describe_byte(char c) {
    const unsigned char byte = static_cast<unsigned char>(c);

    if (byte >= 0x20 && byte < 0x7F) {
        return std::string("character '") + c + "'";
    }

    static const char kHex[] = "0123456789ABCDEF";
    std::string text = "byte 0x";
    text.push_back(kHex[byte >> 4]);
    text.push_back(kHex[byte & 0x0F]);

    return text;
}

// Called when std::from_chars reports "out of range" for a number that
// has already been checked against the JSON grammar. That happens when
// the number is too big (1e400) but also when it is too close to zero
// (1e-400). Tells the two apart by the decimal exponent of the first
// non-zero digit: below 0 means the number is tiny.
bool is_tiny_number(std::string_view token) {
    std::size_t i = 0;
    if (!token.empty() && token[0] == '-') {
        ++i;
    }

    // Length of the integer part
    std::size_t int_start = i;
    while (i < token.size() && is_digit(token[i])) {
        ++i;
    }
    const std::size_t int_length = i - int_start;

    // Decimal exponent of the first non-zero digit, as if the number
    // was written d.ddd * 10^order
    long long order = 0;
    bool found = false;

    for (std::size_t k = 0; k < int_length && !found; ++k) {
        if (token[int_start + k] != '0') {
            order = static_cast<long long>(int_length - 1 - k);
            found = true;
        }
    }

    if (i < token.size() && token[i] == '.') {
        ++i;
        long long place = 0;
        while (i < token.size() && is_digit(token[i])) {
            ++place;
            if (!found && token[i] != '0') {
                order = -place;
                found = true;
            }
            ++i;
        }
    }

    if (!found) {
        return false; // all zeros: cannot be out of range
    }

    // The explicit exponent, clamped so that absurdly long exponents
    // cannot overflow
    long long exponent = 0;
    if (i < token.size() && (token[i] == 'e' || token[i] == 'E')) {
        ++i;
        bool negative = false;
        if (i < token.size() && (token[i] == '+' || token[i] == '-')) {
            negative = token[i] == '-';
            ++i;
        }
        while (i < token.size() && is_digit(token[i])) {
            if (exponent < 1000000000LL) {
                exponent = exponent * 10 + (token[i] - '0');
            }
            ++i;
        }
        if (negative) {
            exponent = -exponent;
        }
    }

    return order + exponent < 0;
}

} // namespace

// A recursive descent parser: one function per kind of value, each
// reading from m_pos and moving it past what it consumed. Every
// function returns false after recording an error.
class JsonParser {
public:
    JsonParser(std::string_view text, std::string& error)
        : m_text(text), m_error(error) {}

    bool parse(JsonValue& out) {
        // A byte order mark is not part of the JSON text
        if (m_text.size() >= 3 && m_text.substr(0, 3) == "\xEF\xBB\xBF") {
            m_text.remove_prefix(3);
        }

        skip_whitespace();
        if (m_pos >= m_text.size()) {
            return fail(m_pos, "empty document");
        }

        JsonValue value;
        if (!parse_value(value, 0)) {
            return false;
        }

        skip_whitespace();
        if (m_pos < m_text.size()) {
            return fail(m_pos, "unexpected data after the value");
        }

        out = std::move(value);
        return true;
    }

private:
    std::string_view m_text;
    std::string& m_error;
    std::size_t m_pos = 0;

    // Records an error at a byte offset, converted to line and column
    bool fail(std::size_t offset, const std::string& message) {
        std::size_t line = 1;
        std::size_t line_start = 0;

        // Done once, when the error happens, so the cost does not
        // matter for valid documents
        for (std::size_t i = 0; i < offset && i < m_text.size(); ++i) {
            if (m_text[i] == '\n') {
                ++line;
                line_start = i + 1;
            }
        }

        m_error = "line " + std::to_string(line) + ", column " +
                  std::to_string(offset - line_start + 1) + ": " + message;
        return false;
    }

    void skip_whitespace() {
        while (m_pos < m_text.size()) {
            const char c = m_text[m_pos];
            if (c != ' ' && c != '\t' && c != '\n' && c != '\r') {
                break;
            }
            ++m_pos;
        }
    }

    // depth is the number of arrays/objects this value is inside of
    bool parse_value(JsonValue& out, int depth) {
        if (m_pos >= m_text.size()) {
            return fail(m_pos, "unexpected end of input");
        }

        const char c = m_text[m_pos];

        switch (c) {
        case '{':
            return parse_object(out, depth);
        case '[':
            return parse_array(out, depth);
        case '"':
            out.m_type = JsonType::JSON_STRING;
            return parse_string(out.m_string);
        case 't':
            out.m_type = JsonType::JSON_BOOL;
            out.m_bool = true;
            return parse_literal("true");
        case 'f':
            out.m_type = JsonType::JSON_BOOL;
            out.m_bool = false;
            return parse_literal("false");
        case 'n':
            return parse_literal("null");
        default:
            break;
        }

        if (c == '-' || is_digit(c)) {
            return parse_number(out);
        }

        return fail(m_pos, "unexpected " + describe_byte(c));
    }

    bool parse_literal(std::string_view word) {
        for (std::size_t i = 0; i < word.size(); ++i) {
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in '" +
                                       std::string(word) + "'");
            }
            if (m_text[m_pos] != word[i]) {
                return fail(m_pos, "invalid literal, expected '" +
                                       std::string(word) + "'");
            }
            ++m_pos;
        }

        return true;
    }

    // JSON numbers: -?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?
    // The grammar is checked by hand (std::from_chars is more lenient
    // in some ways and stricter in others), then the checked text is
    // converted
    bool parse_number(JsonValue& out) {
        const std::size_t start = m_pos;
        std::size_t p = m_pos;
        const std::size_t size = m_text.size();

        if (m_text[p] == '-') {
            ++p;
        }

        if (p >= size || !is_digit(m_text[p])) {
            return fail(p, "expected a digit after '-'");
        }

        if (m_text[p] == '0') {
            ++p;
            if (p < size && is_digit(m_text[p])) {
                return fail(p, "leading zeros are not allowed");
            }
        } else {
            while (p < size && is_digit(m_text[p])) {
                ++p;
            }
        }

        if (p < size && m_text[p] == '.') {
            ++p;
            if (p >= size || !is_digit(m_text[p])) {
                return fail(p, "expected a digit after the decimal point");
            }
            while (p < size && is_digit(m_text[p])) {
                ++p;
            }
        }

        if (p < size && (m_text[p] == 'e' || m_text[p] == 'E')) {
            ++p;
            if (p < size && (m_text[p] == '+' || m_text[p] == '-')) {
                ++p;
            }
            if (p >= size || !is_digit(m_text[p])) {
                return fail(p, "expected a digit in the exponent");
            }
            while (p < size && is_digit(m_text[p])) {
                ++p;
            }
        }

        const char* first = m_text.data() + start;
        const char* last = m_text.data() + p;
        double value = 0.0;
        const std::from_chars_result result = std::from_chars(first, last, value);

        if (result.ec == std::errc::result_out_of_range) {
            // Too close to zero to represent (1e-400): becomes 0, with
            // the sign kept. (Numbers that are merely tiny, like
            // 1e-310, convert normally.) Too big (1e400): an error,
            // because JSON has no infinity
            if (!is_tiny_number(std::string_view(first, p - start))) {
                return fail(start, "number out of range");
            }
            value = (*first == '-') ? -0.0 : 0.0;
        } else if (result.ec != std::errc() || result.ptr != last) {
            return fail(start, "invalid number");
        }

        out.m_type = JsonType::JSON_NUMBER;
        out.m_number = value;
        m_pos = p;
        return true;
    }

    // Reads the 4 hex digits of a backslash-u escape. m_pos is at the first one
    bool parse_hex4(std::uint32_t& value) {
        value = 0;

        for (int i = 0; i < 4; ++i) {
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in \\u escape");
            }

            const int digit = hex_value(m_text[m_pos]);
            if (digit < 0) {
                return fail(m_pos, "invalid hex digit in \\u escape");
            }

            value = value * 16 + static_cast<std::uint32_t>(digit);
            ++m_pos;
        }

        return true;
    }

    // m_pos is at the backslash. Appends what the escape stands for
    bool parse_escape(std::string& out) {
        const std::size_t backslash = m_pos;
        ++m_pos;

        if (m_pos >= m_text.size()) {
            return fail(m_pos, "unterminated string");
        }

        const char c = m_text[m_pos];
        ++m_pos;

        switch (c) {
        case '"':  out.push_back('"');  return true;
        case '\\': out.push_back('\\'); return true;
        case '/':  out.push_back('/');  return true;
        case 'b':  out.push_back('\b'); return true;
        case 'f':  out.push_back('\f'); return true;
        case 'n':  out.push_back('\n'); return true;
        case 'r':  out.push_back('\r'); return true;
        case 't':  out.push_back('\t'); return true;
        case 'u':  break;
        default:
            return fail(m_pos - 1, "invalid escape " +
                                       std::string("\\") + c);
        }

        std::uint32_t unit = 0;
        if (!parse_hex4(unit)) {
            return false;
        }

        if (unit >= 0xDC00 && unit <= 0xDFFF) {
            return fail(backslash, "low surrogate without a high surrogate");
        }

        if (unit < 0xD800 || unit > 0xDBFF) {
            append_utf8(out, unit);
            return true;
        }

        // Backslash-u escapes hold only 16 bits, but Unicode has code points up
        // to 0x10FFFF (emoji, for example). Those are written as a
        // "surrogate pair": two escapes, a high surrogate (D800-DBFF)
        // followed by a low surrogate (DC00-DFFF), each carrying 10
        // bits of (code point - 0x10000)
        if (m_pos + 1 >= m_text.size() || m_text[m_pos] != '\\' ||
            m_text[m_pos + 1] != 'u') {
            return fail(m_pos, "high surrogate must be followed by a "
                               "\\u low surrogate");
        }

        const std::size_t second = m_pos;
        m_pos += 2;

        std::uint32_t low = 0;
        if (!parse_hex4(low)) {
            return false;
        }
        if (low < 0xDC00 || low > 0xDFFF) {
            return fail(second, "high surrogate must be followed by a "
                                "low surrogate");
        }

        append_utf8(out, 0x10000 + ((unit - 0xD800) << 10) + (low - 0xDC00));
        return true;
    }

    // m_pos is at the opening quote
    bool parse_string(std::string& out) {
        ++m_pos;

        for (;;) {
            // Copy the plain bytes up to the next special one in one
            // go instead of one at a time
            std::size_t run_end = m_pos;
            while (run_end < m_text.size()) {
                const unsigned char c =
                    static_cast<unsigned char>(m_text[run_end]);
                if (c == '"' || c == '\\' || c < 0x20) {
                    break;
                }
                ++run_end;
            }

            out.append(m_text.data() + m_pos, run_end - m_pos);
            m_pos = run_end;

            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unterminated string");
            }

            const char c = m_text[m_pos];

            if (c == '"') {
                ++m_pos;
                return true;
            }

            if (c == '\\') {
                if (!parse_escape(out)) {
                    return false;
                }
                continue;
            }

            return fail(m_pos, "control character in string "
                               "(must be written as an escape)");
        }
    }

    bool parse_array(JsonValue& out, int depth) {
        if (depth + 1 > kMaxJsonDepth) {
            return fail(m_pos, "too deeply nested");
        }

        ++m_pos;
        out.m_type = JsonType::JSON_ARRAY;
        skip_whitespace();

        if (m_pos < m_text.size() && m_text[m_pos] == ']') {
            ++m_pos;
            return true;
        }

        for (;;) {
            // Parse straight into the new slot instead of moving a
            // finished value in. The reference stays valid because
            // parsing the element only changes the element's own data
            out.m_elements.emplace_back();
            if (!parse_value(out.m_elements.back(), depth + 1)) {
                return false;
            }

            skip_whitespace();
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in array");
            }

            const char c = m_text[m_pos];
            ++m_pos;

            if (c == ']') {
                return true;
            }
            if (c != ',') {
                return fail(m_pos - 1, "expected ',' or ']' in array");
            }

            skip_whitespace();
            if (m_pos < m_text.size() && m_text[m_pos] == ']') {
                return fail(m_pos, "trailing comma is not allowed");
            }
        }
    }

    bool parse_object(JsonValue& out, int depth) {
        if (depth + 1 > kMaxJsonDepth) {
            return fail(m_pos, "too deeply nested");
        }

        ++m_pos;
        out.m_type = JsonType::JSON_OBJECT;
        skip_whitespace();

        if (m_pos < m_text.size() && m_text[m_pos] == '}') {
            ++m_pos;
            return true;
        }

        for (;;) {
            // The key
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in object");
            }
            if (m_text[m_pos] != '"') {
                return fail(m_pos, "expected a string as object key");
            }

            out.m_members.emplace_back();
            JsonMember& member = out.m_members.back();

            if (!parse_string(member.key)) {
                return false;
            }

            // The colon
            skip_whitespace();
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in object");
            }
            if (m_text[m_pos] != ':') {
                return fail(m_pos, "expected ':' after object key");
            }
            ++m_pos;

            // The value
            skip_whitespace();
            if (!parse_value(member.value, depth + 1)) {
                return false;
            }

            // Comma or end
            skip_whitespace();
            if (m_pos >= m_text.size()) {
                return fail(m_pos, "unexpected end of input in object");
            }

            const char c = m_text[m_pos];
            ++m_pos;

            if (c == '}') {
                return true;
            }
            if (c != ',') {
                return fail(m_pos - 1, "expected ',' or '}' in object");
            }

            skip_whitespace();
            if (m_pos < m_text.size() && m_text[m_pos] == '}') {
                return fail(m_pos, "trailing comma is not allowed");
            }
        }
    }
};

bool parse_json(std::string_view text, JsonValue& out, std::string& error) {
    error.clear();

    JsonParser parser(text, error);
    JsonValue value;

    if (!parser.parse(value)) {
        out = JsonValue();
        return false;
    }

    out = std::move(value);
    return true;
}

} // namespace gfx
