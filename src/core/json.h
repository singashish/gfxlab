#pragma once

/*
 * JSON, parsed from scratch (RFC 8259). glTF files are JSON, so the
 * glTF loader is built on this.
 *
 * parse_json() turns text into a tree of JsonValue. The tree is
 * read-only and designed to be pleasant to query: nothing throws, and
 * a lookup that does not fit (missing key, index out of range, value of
 * another type) gives back a "null" value instead of failing. That
 * lets you chain lookups without checking every step:
 *
 *     int mesh = doc["nodes"][2]["mesh"].as_int(-1);
 *
 * If "nodes" is missing, or has fewer than 3 entries, or the node has
 * no "mesh", the result is simply the fallback, -1.
 *
 * Details worth knowing:
 *  - Numbers are stored as double (like JavaScript). Integers up to
 *    2^53 are exact, which is plenty for glTF indices and counts.
 *  - Strings are stored as UTF-8. Unicode (backslash-u) escapes are converted to UTF-8;
 *    all other bytes are copied unchanged and not validated.
 *  - Objects keep their members in file order. Duplicate keys are kept;
 *    lookups by key find the first one.
 *  - Nesting is limited to kMaxJsonDepth levels so that a malicious
 *    file cannot overflow the stack.
 */

#include <cstddef>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace gfx {

enum class JsonType {
    JSON_NULL, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT
};

// Deepest nesting of arrays and objects that parse_json accepts
constexpr int kMaxJsonDepth = 256;

struct JsonMember;

class JsonValue {
public:
    // A default-constructed value is null
    JsonValue() = default;

    JsonType type() const { return m_type; }
    bool is_null() const { return m_type == JsonType::JSON_NULL; }
    bool is_bool() const { return m_type == JsonType::JSON_BOOL; }
    bool is_number() const { return m_type == JsonType::JSON_NUMBER; }
    bool is_string() const { return m_type == JsonType::JSON_STRING; }
    bool is_array() const { return m_type == JsonType::JSON_ARRAY; }
    bool is_object() const { return m_type == JsonType::JSON_OBJECT; }

    // Typed reads. When the value has another type, the fallback is
    // returned
    bool as_bool(bool fallback = false) const;
    double as_number(double fallback = 0.0) const;

    // The number as float. Numbers beyond the range of float are
    // clamped to the largest finite float (the plain conversion would
    // be undefined behaviour)
    float as_float(float fallback = 0.0f) const;

    // The number if it is a whole number that fits in int (so 3 and
    // 3.0 and 3e0 work, 3.5 and 1e10 do not), otherwise the fallback
    int as_int(int fallback = 0) const;

    // The string, or an empty string when the value is not a string
    const std::string& as_string() const;

    // Arrays: number of elements. Objects: number of members.
    // Everything else: 0
    std::size_t size() const;

    // Never throw. An index out of range, a key that is missing, or
    // a value of the wrong type (indexing a number, say) returns the
    // shared null value, so chains like
    // doc["a"][2]["b"].as_int(-1) are safe.
    const JsonValue& operator[](std::size_t index) const;

    // Objects only: the first member with that key
    const JsonValue& operator[](std::string_view key) const;

    // Any other integer type (int, unsigned, long long, ...). Without
    // this, doc[0] would be ambiguous between the size_t and the
    // string_view overloads (0 is also a null pointer), and so would
    // an index of type int or unsigned. Negative indices give null.
    template <typename T,
              typename = std::enable_if_t<std::is_integral_v<T> &&
                                          !std::is_same_v<T, bool> &&
                                          !std::is_same_v<T, std::size_t>>>
    const JsonValue& operator[](T index) const {
        if constexpr (std::is_signed_v<T>) {
            if (index < 0) {
                return null_value();
            }
        }
        return (*this)[static_cast<std::size_t>(index)];
    }

    // True if this is an object with a member of that name
    bool contains(std::string_view key) const;

    // The array elements (empty unless this is an array)
    const std::vector<JsonValue>& elements() const;

    // The object members in file order (empty unless this is an object)
    const std::vector<JsonMember>& members() const;

    // The shared null value that failed lookups return
    static const JsonValue& null_value();

private:
    // The parser fills the private data directly
    friend class JsonParser;

    JsonType m_type = JsonType::JSON_NULL;
    bool m_bool = false;
    double m_number = 0.0;
    std::string m_string;
    std::vector<JsonValue> m_elements;
    std::vector<JsonMember> m_members;
};

struct JsonMember {
    std::string key;
    JsonValue value;
};

// Parses a whole JSON document: exactly one value, with only whitespace
// around it (a UTF-8 byte order mark at the very start is skipped). On
// success returns true and sets out. On failure returns false, sets out
// to null and sets error to "line L, column C: message", where line and
// column are 1-based and point at the character where the problem was
// found (or just past the end of the input when it is cut short).
// Columns count bytes, not characters, and a byte order mark is not
// counted.
bool parse_json(std::string_view text, JsonValue& out, std::string& error);

} // namespace gfx
