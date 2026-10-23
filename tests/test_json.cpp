#include "core/json.h"
#include "check.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <string_view>

namespace {

using gfx::JsonType;
using gfx::JsonValue;

// Parses text that must be valid
JsonValue parse_ok(std::string_view text) {
    JsonValue value;
    std::string error;
    const bool ok = gfx::parse_json(text, value, error);

    if (!ok) {
        std::fprintf(stderr, "unexpected parse error for [%.60s]: %s\n",
                     std::string(text).c_str(), error.c_str());
        ++g_failures;
    }

    return value;
}

// Parses text that must be invalid; returns the error message
std::string parse_fail(std::string_view text) {
    JsonValue value;
    std::string error;
    const bool ok = gfx::parse_json(text, value, error);

    if (ok) {
        std::fprintf(stderr, "parse unexpectedly succeeded for [%.60s]\n",
                     std::string(text).c_str());
        ++g_failures;
    }

    // A failed parse always leaves null behind
    CHECK(value.is_null());

    return error;
}

bool starts_with(const std::string& text, const char* prefix) {
    return text.rfind(prefix, 0) == 0;
}

bool contains_text(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

// ---------------------------------------------------------------------
// Types and queries
// ---------------------------------------------------------------------

void test_types() {
    JsonValue v = parse_ok("null");
    CHECK(v.type() == JsonType::JSON_NULL);
    CHECK(v.is_null());
    CHECK(!v.is_bool() && !v.is_number() && !v.is_string());
    CHECK(!v.is_array() && !v.is_object());

    v = parse_ok("true");
    CHECK(v.type() == JsonType::JSON_BOOL && v.is_bool());
    CHECK(v.as_bool() == true);

    v = parse_ok("false");
    CHECK(v.is_bool() && v.as_bool(true) == false);

    v = parse_ok("12.5");
    CHECK(v.type() == JsonType::JSON_NUMBER && v.is_number());
    CHECK(approx(v.as_number(), 12.5));

    v = parse_ok("\"hi\"");
    CHECK(v.type() == JsonType::JSON_STRING && v.is_string());
    CHECK(v.as_string() == "hi");

    v = parse_ok("[1, 2]");
    CHECK(v.type() == JsonType::JSON_ARRAY && v.is_array());
    CHECK(v.size() == 2);

    v = parse_ok("{\"a\": 1}");
    CHECK(v.type() == JsonType::JSON_OBJECT && v.is_object());
    CHECK(v.size() == 1);

    // Empty containers
    v = parse_ok("[]");
    CHECK(v.is_array() && v.size() == 0 && v.elements().empty());
    v = parse_ok("{}");
    CHECK(v.is_object() && v.size() == 0 && v.members().empty());
    v = parse_ok("[ \n ]");
    CHECK(v.is_array() && v.size() == 0);
    v = parse_ok("{ \t }");
    CHECK(v.is_object() && v.size() == 0);

    // Default-constructed value is null
    JsonValue empty;
    CHECK(empty.is_null());
    CHECK(&JsonValue::null_value() == &JsonValue::null_value());
}

void test_fallbacks() {
    const JsonValue n = parse_ok("3");
    const JsonValue s = parse_ok("\"text\"");
    const JsonValue b = parse_ok("true");
    const JsonValue z = parse_ok("null");

    // Wrong type: fallback comes back
    CHECK(s.as_bool(true) == true);
    CHECK(s.as_bool() == false);
    CHECK(n.as_bool(true) == true);
    CHECK(approx(s.as_number(7.5), 7.5));
    CHECK(approx(b.as_number(), 0.0));
    CHECK(approx(z.as_float(2.5f), 2.5));
    CHECK(s.as_int(-9) == -9);
    CHECK(n.as_string().empty());
    CHECK(z.as_string().empty());
    CHECK(b.as_string().empty());

    // as_string on a non-string always refers to the same empty string
    CHECK(&n.as_string() == &z.as_string());

    // Right type: the value
    CHECK(approx(n.as_number(99.0), 3.0));
    CHECK(approx(n.as_float(99.0f), 3.0));
    CHECK(n.as_int(99) == 3);
    CHECK(b.as_bool(false) == true);

    // size() is 0 for everything that is not a container, even strings
    CHECK(n.size() == 0 && s.size() == 0 && b.size() == 0 && z.size() == 0);
}

void test_as_int_and_float() {
    CHECK(parse_ok("0").as_int(5) == 0);
    CHECK(parse_ok("-0").as_int(5) == 0);
    CHECK(parse_ok("42").as_int(5) == 42);
    CHECK(parse_ok("-42").as_int(5) == -42);
    CHECK(parse_ok("3.0").as_int(5) == 3);
    CHECK(parse_ok("3e0").as_int(5) == 3);
    CHECK(parse_ok("1.5e3").as_int(5) == 1500);
    CHECK(parse_ok("-7E+1").as_int(5) == -70);

    // Not whole: fallback
    CHECK(parse_ok("3.5").as_int(5) == 5);
    CHECK(parse_ok("-0.25").as_int(5) == 5);
    CHECK(parse_ok("1e-5").as_int(5) == 5);

    // The limits of int
    CHECK(parse_ok("2147483647").as_int(5) == 2147483647);
    CHECK(parse_ok("-2147483648").as_int(5) ==
          std::numeric_limits<int>::min());
    CHECK(parse_ok("2147483648").as_int(5) == 5);
    CHECK(parse_ok("-2147483649").as_int(5) == 5);
    CHECK(parse_ok("4294967295").as_int(5) == 5);
    CHECK(parse_ok("1e10").as_int(5) == 5);
    CHECK(parse_ok("-1e300").as_int(5) == 5);
    CHECK(parse_ok("1e300").as_int(-1) == -1);

    // as_float: normal values, and clamping instead of undefined behaviour
    CHECK(approx(parse_ok("0.5").as_float(), 0.5));
    CHECK(approx(parse_ok("-12.75").as_float(), -12.75));
    CHECK(parse_ok("1e300").as_float() ==
          std::numeric_limits<float>::max());
    CHECK(parse_ok("-1e300").as_float() ==
          std::numeric_limits<float>::lowest());
    CHECK(parse_ok("1e-300").as_float() == 0.0f);
}

void test_queries() {
    const JsonValue doc = parse_ok(
        "{\n"
        "        \"name\": \"scene\",\n"
        "        \"count\": 3,\n"
        "        \"flag\": true,\n"
        "        \"list\": [10, 20, {\"deep\": [1, 2, {\"x\": 7}]}],\n"
        "        \"obj\": {\"a\": 1, \"b\": \"two\", \"c\": null}\n"
        "    }");

    CHECK(doc["name"].as_string() == "scene");
    CHECK(doc["count"].as_int() == 3);
    CHECK(doc["flag"].as_bool() == true);
    CHECK(doc["list"].size() == 3);
    CHECK(doc["list"][0].as_int() == 10);
    CHECK(doc["list"][1].as_int() == 20);
    CHECK(doc["list"][2]["deep"][2]["x"].as_int(-1) == 7);

    // doc[0] and doc["k"] both compile; so do other integer types
    const JsonValue& list = doc["list"];
    CHECK(list[0].as_int(-1) == 10);
    CHECK(list[std::size_t(1)].as_int(-1) == 20);
    CHECK(list[1u].as_int(-1) == 20);
    CHECK(list[1L].as_int(-1) == 20);
    CHECK(list[std::uint8_t(0)].as_int(-1) == 10);
    const int signed_index = 1;
    const unsigned unsigned_index = 0;
    CHECK(list[signed_index].as_int(-1) == 20);
    CHECK(list[unsigned_index].as_int(-1) == 10);
    CHECK(doc["name"].as_string() == "scene");
    CHECK(doc[std::string_view("name")].as_string() == "scene");
    CHECK(doc[std::string("name")].as_string() == "scene");

    // Missing key, out-of-range index, negative index
    CHECK(doc["missing"].is_null());
    CHECK(doc["missing"]["still"][5]["fine"].as_int(-1) == -1);
    CHECK(list[3].is_null());
    CHECK(list[1000000].is_null());
    CHECK(list[-1].is_null());
    CHECK(list[std::numeric_limits<std::size_t>::max()].is_null());

    // Wrong type: indexing an object with a number, an array with a
    // key, scalars with either
    CHECK(doc[0].is_null());
    CHECK(list["name"].is_null());
    CHECK(doc["count"][0].is_null());
    CHECK(doc["count"]["x"].is_null());
    CHECK(doc["name"][0].is_null());
    CHECK(doc["flag"]["x"].is_null());

    // Missing results are the shared null, not a copy
    CHECK(&doc["missing"] == &JsonValue::null_value());
    CHECK(&list[99] == &JsonValue::null_value());

    // A key that exists with a null value is different from a missing key
    CHECK(doc["obj"].contains("c"));
    CHECK(doc["obj"]["c"].is_null());
    CHECK(!doc["obj"].contains("zzz"));

    // contains() only for objects
    CHECK(doc.contains("name"));
    CHECK(!doc.contains("Name"));
    CHECK(!list.contains("0"));
    CHECK(!doc["count"].contains("x"));
    CHECK(!doc["name"].contains("name"));

    // size() of containers
    CHECK(doc.size() == 5);
    CHECK(doc["obj"].size() == 3);

    // elements() and members()
    CHECK(list.elements().size() == 3);
    CHECK(list.members().empty());
    CHECK(doc["obj"].elements().empty());
    CHECK(doc["count"].elements().empty());
    CHECK(doc["count"].members().empty());

    int total = 0;
    for (const JsonValue& e : doc["list"].elements()) {
        total += e.as_int(0);
    }
    CHECK(total == 30);

    // Keys are matched exactly (case, length)
    CHECK(doc["nam"].is_null());
    CHECK(doc["names"].is_null());
    CHECK(doc[""].is_null());
}

void test_members_order_and_duplicates() {
    const JsonValue doc = parse_ok(
        "{\"z\": 1, \"a\": 2, \"m\": 3, \"a\": 4, \"\": 5, \"z\": 6}");

    const auto& members = doc.members();
    CHECK(members.size() == 6);
    CHECK(doc.size() == 6);

    if (members.size() == 6) {
        CHECK(members[0].key == "z" && members[0].value.as_int() == 1);
        CHECK(members[1].key == "a" && members[1].value.as_int() == 2);
        CHECK(members[2].key == "m" && members[2].value.as_int() == 3);
        CHECK(members[3].key == "a" && members[3].value.as_int() == 4);
        CHECK(members[4].key.empty() && members[4].value.as_int() == 5);
        CHECK(members[5].key == "z" && members[5].value.as_int() == 6);
    }

    // The first of the duplicates wins
    CHECK(doc["z"].as_int() == 1);
    CHECK(doc["a"].as_int() == 2);
    CHECK(doc[""].as_int() == 5);
    CHECK(doc.contains(""));
}

// ---------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------

void test_numbers() {
    CHECK(approx(parse_ok("0").as_number(-1), 0.0));
    CHECK(approx(parse_ok("1").as_number(-1), 1.0));
    CHECK(approx(parse_ok("-1").as_number(0), -1.0));
    CHECK(approx(parse_ok("1.5e3").as_number(), 1500.0));
    CHECK(approx(parse_ok("1E-2").as_number(), 0.01));
    CHECK(approx(parse_ok("1e+2").as_number(), 100.0));
    CHECK(approx(parse_ok("1E2").as_number(), 100.0));
    CHECK(approx(parse_ok("-12.75").as_number(), -12.75));
    CHECK(approx(parse_ok("0.125").as_number(), 0.125));
    CHECK(approx(parse_ok("0e0").as_number(1), 0.0));
    CHECK(approx(parse_ok("0.0").as_number(1), 0.0));
    CHECK(approx(parse_ok("10").as_number(), 10.0));
    CHECK(approx(parse_ok("100").as_number(), 100.0));
    CHECK(approx(parse_ok("1.000").as_number(), 1.0));
    CHECK(approx(parse_ok("-0.5e-1").as_number(), -0.05));
    CHECK(approx(parse_ok("2.5E+0003").as_number(), 2500.0));

    // Negative zero keeps its sign bit but compares equal to 0
    const double neg_zero = parse_ok("-0").as_number(1.0);
    CHECK(neg_zero == 0.0);
    CHECK(std::signbit(neg_zero));
    CHECK(!std::signbit(parse_ok("0").as_number(1.0)));
    CHECK(std::signbit(parse_ok("-0.0e5").as_number(1.0)));

    // Large integers are exact in a double up to 2^53
    CHECK(parse_ok("4294967295").as_number() == 4294967295.0);
    CHECK(parse_ok("4294967296").as_number() == 4294967296.0);
    CHECK(parse_ok("-4294967295").as_number() == -4294967295.0);
    CHECK(parse_ok("9007199254740992").as_number() == 9007199254740992.0);
    CHECK(parse_ok("123456789012345678901234567890").as_number() ==
          1.2345678901234568e29);

    // Exact rounding of doubles that need all 17 digits
    CHECK(parse_ok("0.1").as_number() == 0.1);
    CHECK(parse_ok("1.7976931348623157e308").as_number() ==
          std::numeric_limits<double>::max());
    CHECK(parse_ok("5e-324").as_number() ==
          std::numeric_limits<double>::denorm_min());
    CHECK(parse_ok("1e-310").as_number() > 0.0);

    // A number followed by more tokens
    const JsonValue a = parse_ok("[1,-2,3.5,4e2,0]");
    CHECK(a.size() == 5);
    CHECK(a[3].as_int() == 400);
    CHECK(a[4].as_int(-1) == 0);
}

void test_number_range() {
    // Too big: an error, since JSON has no infinity
    CHECK(starts_with(parse_fail("1e400"), "line 1, column 1: "));
    CHECK(contains_text(parse_fail("1e400"), "number out of range"));
    CHECK(contains_text(parse_fail("-1e400"), "number out of range"));
    CHECK(contains_text(parse_fail("1e99999999999999999999"),
                        "number out of range"));
    CHECK(contains_text(parse_fail("[1, 1.0E+309]"), "number out of range"));
    CHECK(starts_with(parse_fail("[1, 1.0E+309]"), "line 1, column 5: "));

    // Too small: std::from_chars reports "out of range" for these too,
    // but we turn them into zero, keeping the sign
    JsonValue v = parse_ok("1e-400");
    CHECK(v.is_number() && v.as_number(1.0) == 0.0);
    CHECK(!std::signbit(v.as_number(1.0)));

    v = parse_ok("-1e-400");
    CHECK(v.is_number() && v.as_number(1.0) == 0.0);
    CHECK(std::signbit(v.as_number(1.0)));

    v = parse_ok("0.000000000000000000000000000000000000000001e-400");
    CHECK(v.is_number() && v.as_number(1.0) == 0.0);

    v = parse_ok("123456e-99999999999999999999");
    CHECK(v.is_number() && v.as_number(1.0) == 0.0);

    // Zero with an absurd exponent is just zero
    v = parse_ok("0e99999");
    CHECK(v.is_number() && v.as_number(1.0) == 0.0);

    // The same just inside the range works
    CHECK(parse_ok("1e308").as_number() == 1e308);
    CHECK(parse_ok("1e-300").as_number() == 1e-300);
}

void test_invalid_numbers() {
    const char* const bad[] = {
        "+1", "01", "-01", "00", "1.", ".5", "-.5", "1e", "1E", "1e+",
        "1e-", "-", "--1", "- 1", "NaN", "-NaN", "Infinity", "-Infinity",
        "inf", "0x10", "0X1F", "1.2.3", "1e5.5", "1_000", "1,5", "0b1",
        "1.e5", "1e5e5", "0.5.", "\xEF\xBC\x91",
        "[1.]", "[.5]", "[01]", "[-]", "[1e]", "[+1]", "{\"a\": 01}",
        "{\"a\": -}",
    };

    for (const char* text : bad) {
        const std::string error = parse_fail(text);
        CHECK(starts_with(error, "line 1, column "));
    }

    // Positions and messages of the number errors
    CHECK(parse_fail("01") ==
          "line 1, column 2: leading zeros are not allowed");
    CHECK(parse_fail("-01") ==
          "line 1, column 3: leading zeros are not allowed");
    CHECK(parse_fail("1.") ==
          "line 1, column 3: expected a digit after the decimal point");
    CHECK(parse_fail("1.x") ==
          "line 1, column 3: expected a digit after the decimal point");
    CHECK(parse_fail("-") == "line 1, column 2: expected a digit after '-'");
    CHECK(parse_fail("-x") == "line 1, column 2: expected a digit after '-'");
    CHECK(parse_fail("1e") ==
          "line 1, column 3: expected a digit in the exponent");
    CHECK(parse_fail("1e+") ==
          "line 1, column 4: expected a digit in the exponent");
    CHECK(parse_fail("+1") == "line 1, column 1: unexpected character '+'");
    CHECK(parse_fail(".5") == "line 1, column 1: unexpected character '.'");
    CHECK(parse_fail("NaN") == "line 1, column 1: unexpected character 'N'");
    CHECK(parse_fail("Infinity") ==
          "line 1, column 1: unexpected character 'I'");

    // "0x10" is the number 0 followed by data
    CHECK(parse_fail("0x10") ==
          "line 1, column 2: unexpected data after the value");
}

// ---------------------------------------------------------------------
// Literals
// ---------------------------------------------------------------------

void test_literals() {
    const char* const bad[] = {
        "tru", "fals", "nul", "nulls", "truee", "True", "TRUE", "False",
        "NULL", "nil", "undefined", "t", "n", "f", "nulL", "[tru]",
        "[nulls]", "{\"a\": fals}", "truefalse",
    };

    for (const char* text : bad) {
        parse_fail(text);
    }

    CHECK(parse_fail("tru") ==
          "line 1, column 4: unexpected end of input in 'true'");
    CHECK(parse_fail("trux") ==
          "line 1, column 4: invalid literal, expected 'true'");
    CHECK(parse_fail("nulls") ==
          "line 1, column 5: unexpected data after the value");
    CHECK(parse_fail("[nulls]") ==
          "line 1, column 6: expected ',' or ']' in array");

    const JsonValue v = parse_ok("[true,false,null]");
    CHECK(v.size() == 3);
    CHECK(v[0].is_bool() && v[0].as_bool());
    CHECK(v[1].is_bool() && !v[1].as_bool(true));
    CHECK(v[2].is_null());
}

// ---------------------------------------------------------------------
// Strings
// ---------------------------------------------------------------------

void test_string_escapes() {
    CHECK(parse_ok("\"\"").as_string().empty());
    CHECK(parse_ok("\"\"").is_string());
    CHECK(parse_ok("\"plain text 123\"").as_string() == "plain text 123");

    CHECK(parse_ok("\"\\\"\"").as_string() == "\"");
    CHECK(parse_ok("\"\\\\\"").as_string() == "\\");
    CHECK(parse_ok("\"\\/\"").as_string() == "/");
    CHECK(parse_ok("\"\\b\"").as_string() == "\b");
    CHECK(parse_ok("\"\\f\"").as_string() == "\f");
    CHECK(parse_ok("\"\\n\"").as_string() == "\n");
    CHECK(parse_ok("\"\\r\"").as_string() == "\r");
    CHECK(parse_ok("\"\\t\"").as_string() == "\t");
    CHECK(parse_ok("\"a\\tb\\nc\\\\d\\\"e\"").as_string() == "a\tb\nc\\d\"e");
    CHECK(parse_ok("\"C:\\\\dir\\\\file.png\"").as_string() == "C:\\dir\\file.png");

    // Unicode escapes: ASCII, with either case of hex digits
    CHECK(parse_ok("\"\\u0041\"").as_string() == "A");
    CHECK(parse_ok("\"\\u004a\\u004A\"").as_string() == "JJ");
    CHECK(parse_ok("\"\\u0000\"").as_string() == std::string(1, '\0'));
    CHECK(parse_ok("\"\\u001f\"").as_string() == "\x1F");
    CHECK(parse_ok("\"x\\u0020y\"").as_string() == "x y");
    CHECK(parse_ok("\"\\u007F\"").as_string() == "\x7F");

    // 2-byte UTF-8: U+00E9 (e with acute accent), U+0080, U+07FF
    CHECK(parse_ok("\"\\u00e9\"").as_string() == "\xC3\xA9");
    CHECK(parse_ok("\"\\u00E9\"").as_string() == "\xC3\xA9");
    CHECK(parse_ok("\"\\u0080\"").as_string() == "\xC2\x80");
    CHECK(parse_ok("\"\\u07ff\"").as_string() == "\xDF\xBF");

    // 3-byte UTF-8: U+0800, U+20AC (euro sign), U+FFFF, U+E000
    CHECK(parse_ok("\"\\u0800\"").as_string() == "\xE0\xA0\x80");
    CHECK(parse_ok("\"\\u20AC\"").as_string() == "\xE2\x82\xAC");
    CHECK(parse_ok("\"\\u20ac\"").as_string() == "\xE2\x82\xAC");
    CHECK(parse_ok("\"\\uFFFF\"").as_string() == "\xEF\xBF\xBF");
    CHECK(parse_ok("\"\\uE000\"").as_string() == "\xEE\x80\x80");
    CHECK(parse_ok("\"\\uD7FF\"").as_string() == "\xED\x9F\xBF");

    // 4-byte UTF-8 from a surrogate pair: U+1F600
    CHECK(parse_ok("\"\\uD83D\\uDE00\"").as_string() == "\xF0\x9F\x98\x80");
    CHECK(parse_ok("\"\\ud83d\\ude00\"").as_string() == "\xF0\x9F\x98\x80");
    // The smallest and the largest pair: U+10000 and U+10FFFF
    CHECK(parse_ok("\"\\uD800\\uDC00\"").as_string() == "\xF0\x90\x80\x80");
    CHECK(parse_ok("\"\\uDBFF\\uDFFF\"").as_string() == "\xF4\x8F\xBF\xBF");
    CHECK(parse_ok("\"a\\uD83D\\uDE00b\"").as_string() ==
          "a\xF0\x9F\x98\x80" "b");

    // Raw UTF-8 passes through unchanged
    CHECK(parse_ok("\"caf\xC3\xA9\"").as_string() == "caf\xC3\xA9");
    CHECK(parse_ok("\"\xE2\x82\xAC\"").as_string() == "\xE2\x82\xAC");
    CHECK(parse_ok("\"\xF0\x9F\x98\x80\"").as_string() == "\xF0\x9F\x98\x80");
    CHECK(parse_ok("\"\x7F\"").as_string() == "\x7F");

    // Invalid UTF-8 is not checked, just copied
    CHECK(parse_ok("\"\xFF\xFE\"").as_string() == "\xFF\xFE");

    // Keys are decoded the same way
    const JsonValue obj = parse_ok("{\"caf\\u00e9\": 1, \"a\\nb\": 2}");
    CHECK(obj["caf\xC3\xA9"].as_int() == 1);
    CHECK(obj["a\nb"].as_int() == 2);
}

void test_string_errors() {
    // Lone and reversed surrogates
    CHECK(contains_text(parse_fail("\"\\uD83D\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uD83Dx\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uDE00\""), "low surrogate"));
    CHECK(contains_text(parse_fail("\"\\uDE00\\uD83D\""), "low surrogate"));
    CHECK(contains_text(parse_fail("\"\\uD83D\\u0041\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uD83D\\uD83D\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uD83D\\n\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uDBFF\""), "high surrogate"));
    CHECK(contains_text(parse_fail("\"\\uDC00\""), "low surrogate"));
    CHECK(contains_text(parse_fail("\"\\uDFFF\""), "low surrogate"));
    parse_fail("\"\\uD83D\\uDE0\"");
    parse_fail("\"\\uD83D\\");
    parse_fail("\"\\uD83D");

    // Positions: a lone high surrogate is reported where the low one
    // should be, a lone low or wrong second escape at its backslash
    CHECK(parse_fail("\"\\uD83D\"") ==
          "line 1, column 8: high surrogate must be followed by a "
          "\\u low surrogate");
    CHECK(parse_fail("\"\\uDE00\"") ==
          "line 1, column 2: low surrogate without a high surrogate");
    CHECK(parse_fail("\"\\uD83D\\u0041\"") ==
          "line 1, column 8: high surrogate must be followed by a "
          "low surrogate");

    // Bad hex digits and truncated escapes
    CHECK(parse_fail("\"\\u12G4\"") ==
          "line 1, column 6: invalid hex digit in \\u escape");
    CHECK(parse_fail("\"\\uGGGG\"") ==
          "line 1, column 4: invalid hex digit in \\u escape");
    CHECK(parse_fail("\"\\u 123\"") ==
          "line 1, column 4: invalid hex digit in \\u escape");
    CHECK(parse_fail("\"\\u12\"") ==
          "line 1, column 6: invalid hex digit in \\u escape");
    CHECK(parse_fail("\"\\u12") ==
          "line 1, column 6: unexpected end of input in \\u escape");
    CHECK(parse_fail("\"\\u") ==
          "line 1, column 4: unexpected end of input in \\u escape");
    parse_fail("\"\\u+123\"");
    parse_fail("\"\\u-123\"");
    parse_fail("\"\\U0041\"");
    parse_fail("\"\\x41\"");

    // Unknown escapes
    CHECK(parse_fail("\"\\q\"") == "line 1, column 3: invalid escape \\q");
    CHECK(parse_fail("\"\\a\"") == "line 1, column 3: invalid escape \\a");
    CHECK(parse_fail("\"\\'\"") == "line 1, column 3: invalid escape \\'");
    CHECK(parse_fail("\"\\0\"") == "line 1, column 3: invalid escape \\0");
    CHECK(parse_fail("\"\\\n\"") ==
          "line 1, column 3: invalid escape \\\x0A");
    CHECK(parse_fail("\"ab\\ \"") == "line 1, column 5: invalid escape \\ ");

    // Raw control characters
    CHECK(parse_fail("\"a\nb\"") ==
          "line 1, column 3: control character in string "
          "(must be written as an escape)");
    CHECK(parse_fail("\"a\tb\"") ==
          "line 1, column 3: control character in string "
          "(must be written as an escape)");
    CHECK(contains_text(parse_fail(std::string("\"a\0b\"", 5)),
                        "control character"));
    CHECK(contains_text(parse_fail("\"\x01\""), "control character"));
    CHECK(contains_text(parse_fail("\"\x1F\""), "control character"));
    CHECK(contains_text(parse_fail("\"\r\""), "control character"));
    CHECK(contains_text(parse_fail("{\"a\nb\": 1}"), "control character"));

    // Unterminated strings
    CHECK(parse_fail("\"abc") == "line 1, column 5: unterminated string");
    CHECK(parse_fail("\"") == "line 1, column 2: unterminated string");
    CHECK(parse_fail("\"abc\\") == "line 1, column 6: unterminated string");
    CHECK(parse_fail("[\"abc") == "line 1, column 6: unterminated string");
    CHECK(parse_fail("{\"abc") == "line 1, column 6: unterminated string");
    CHECK(parse_fail("{\"abc\": \"def") ==
          "line 1, column 13: unterminated string");

    // A single quote is not a string delimiter in JSON
    CHECK(parse_fail("'abc'") == "line 1, column 1: unexpected character '''");
}

// ---------------------------------------------------------------------
// Document structure
// ---------------------------------------------------------------------

void test_whitespace_and_bom() {
    // Space, tab, LF and CR are whitespace
    CHECK(parse_ok(" \t\r\n 7 \t\r\n ").as_int() == 7);
    CHECK(parse_ok("\n\n[ 1 ,\r\n 2 ,\t3 ]\n").size() == 3);
    CHECK(parse_ok("{\n\"a\"\t:\r\n1\n}")["a"].as_int() == 1);

    // Other characters are not
    parse_fail("\f1");
    parse_fail("1\f");
    parse_fail("\v1");
    parse_fail("\xC2\xA0" "1");     // no-break space
    parse_fail("[1,\xC2\xA0 2]");
    parse_fail(std::string("1\0", 2));
    parse_fail(std::string("\0" "1", 2));

    // A BOM at the very start is skipped
    CHECK(parse_ok("\xEF\xBB\xBF" "42").as_int() == 42);
    CHECK(parse_ok("\xEF\xBB\xBF  {\"a\": [1]}")["a"][0].as_int() == 1);
    CHECK(parse_ok("\xEF\xBB\xBF\n null").is_null());

    // Only at the start, and only once
    parse_fail(" \xEF\xBB\xBF" "42");
    parse_fail("\xEF\xBB\xBF\xEF\xBB\xBF" "42");
    parse_fail("[\xEF\xBB\xBF" "1]");
    parse_fail("1\xEF\xBB\xBF");
    parse_fail("\xEF\xBB" "1");        // incomplete BOM
    parse_fail("\xEF\xBB\xBF");        // BOM and nothing else
    parse_fail("\xEF\xBB\xBF   ");

    // The BOM does not count for columns
    CHECK(parse_fail("\xEF\xBB\xBF" "x") ==
          "line 1, column 1: unexpected character 'x'");
}

void test_document_errors() {
    // Empty input
    CHECK(parse_fail("") == "line 1, column 1: empty document");
    CHECK(parse_fail("   ") == "line 1, column 4: empty document");
    CHECK(parse_fail("\n\n") == "line 3, column 1: empty document");
    CHECK(parse_fail(" \t\r\n") == "line 2, column 1: empty document");

    // Data after the value
    CHECK(parse_fail("1 2") ==
          "line 1, column 3: unexpected data after the value");
    CHECK(parse_fail("{} {}") ==
          "line 1, column 4: unexpected data after the value");
    CHECK(parse_fail("[1]]") ==
          "line 1, column 4: unexpected data after the value");
    CHECK(parse_fail("{}x") ==
          "line 1, column 3: unexpected data after the value");
    CHECK(parse_fail("null null") ==
          "line 1, column 6: unexpected data after the value");
    CHECK(parse_fail("\"a\" \"b\"") ==
          "line 1, column 5: unexpected data after the value");
    CHECK(parse_fail("[] ,") ==
          "line 1, column 4: unexpected data after the value");
    CHECK(parse_fail("1\n\n  x") ==
          "line 3, column 3: unexpected data after the value");
    CHECK(parse_fail("true,") ==
          "line 1, column 5: unexpected data after the value");

    // Trailing commas
    CHECK(parse_fail("[1,]") ==
          "line 1, column 4: trailing comma is not allowed");
    CHECK(parse_fail("[1, ]") ==
          "line 1, column 5: trailing comma is not allowed");
    CHECK(parse_fail("{\"a\": 1,}") ==
          "line 1, column 9: trailing comma is not allowed");
    CHECK(parse_fail("{\"a\": 1, }") ==
          "line 1, column 10: trailing comma is not allowed");
    CHECK(parse_fail("[[1,],2]") ==
          "line 1, column 5: trailing comma is not allowed");

    // Leading commas, doubled commas, lone commas
    parse_fail("[,1]");
    parse_fail("[1,,2]");
    parse_fail("[,]");
    parse_fail("{,}");
    parse_fail("{\"a\": 1,,\"b\": 2}");
    parse_fail("{,\"a\": 1}");

    // Missing commas
    CHECK(parse_fail("[1 2]") ==
          "line 1, column 4: expected ',' or ']' in array");
    CHECK(parse_fail("[\"a\" \"b\"]") ==
          "line 1, column 6: expected ',' or ']' in array");
    CHECK(parse_fail("[[] []]") ==
          "line 1, column 5: expected ',' or ']' in array");
    CHECK(parse_fail("{\"a\": 1 \"b\": 2}") ==
          "line 1, column 9: expected ',' or '}' in object");
    CHECK(parse_fail("{\"a\": 1 }}") ==
          "line 1, column 10: unexpected data after the value");
    CHECK(parse_fail("[1}") ==
          "line 1, column 3: expected ',' or ']' in array");
    CHECK(parse_fail("{\"a\": 1]") ==
          "line 1, column 8: expected ',' or '}' in object");

    // Missing or wrong colons
    CHECK(parse_fail("{\"a\" 1}") ==
          "line 1, column 6: expected ':' after object key");
    CHECK(parse_fail("{\"a\"}") ==
          "line 1, column 5: expected ':' after object key");
    CHECK(parse_fail("{\"a\", 1}") ==
          "line 1, column 5: expected ':' after object key");
    CHECK(parse_fail("{\"a\":}") ==
          "line 1, column 6: unexpected character '}'");
    CHECK(parse_fail("{\"a\"::1}") ==
          "line 1, column 6: unexpected character ':'");
    CHECK(parse_fail("{\"a\"") ==
          "line 1, column 5: unexpected end of input in object");

    // Keys must be strings
    CHECK(parse_fail("{a: 1}") ==
          "line 1, column 2: expected a string as object key");
    CHECK(parse_fail("{1: 1}") ==
          "line 1, column 2: expected a string as object key");
    CHECK(parse_fail("{'a': 1}") ==
          "line 1, column 2: expected a string as object key");
    CHECK(parse_fail("{null: 1}") ==
          "line 1, column 2: expected a string as object key");
    CHECK(parse_fail("{[1]: 1}") ==
          "line 1, column 2: expected a string as object key");
    CHECK(parse_fail("{\"a\": 1, 2: 3}") ==
          "line 1, column 10: expected a string as object key");

    // Truncated input: the position is just past the end
    CHECK(parse_fail("[") ==
          "line 1, column 2: unexpected end of input");
    CHECK(parse_fail("[1") ==
          "line 1, column 3: unexpected end of input in array");
    CHECK(parse_fail("[1,") ==
          "line 1, column 4: unexpected end of input");
    CHECK(parse_fail("{") ==
          "line 1, column 2: unexpected end of input in object");
    CHECK(parse_fail("{\"a\":") ==
          "line 1, column 6: unexpected end of input");
    CHECK(parse_fail("{\"a\": 1") ==
          "line 1, column 8: unexpected end of input in object");
    CHECK(parse_fail("{\"a\": 1,") ==
          "line 1, column 9: unexpected end of input in object");
    CHECK(parse_fail("[1, 2\n") ==
          "line 2, column 1: unexpected end of input in array");

    // Stray closing brackets and other junk
    parse_fail("]");
    parse_fail("}");
    parse_fail(",");
    parse_fail(":");
    parse_fail("[}");
    parse_fail("{]");
    parse_fail("/* comment */ 1");
    parse_fail("// comment\n1");
    parse_fail("[1, // comment\n 2]");
    parse_fail("#");
    parse_fail("@");
    CHECK(parse_fail("\x01") ==
          "line 1, column 1: unexpected byte 0x01");
    CHECK(parse_fail("\xFF") ==
          "line 1, column 1: unexpected byte 0xFF");
}

void test_error_positions() {
    const std::string multi_line =
        "{\n"
        "  \"a\": [1, 2, 3],\n"
        "  \"b\": {\n"
        "    \"c\": tru\n"
        "  }\n"
        "}\n";
    CHECK(parse_fail(multi_line) ==
          "line 4, column 13: invalid literal, expected 'true'");

    // Missing comma between members, on line 3
    const std::string missing_comma =
        "{\n"
        "  \"a\": 1\n"
        "  \"b\": 2\n"
        "}";
    CHECK(parse_fail(missing_comma) ==
          "line 3, column 3: expected ',' or '}' in object");

    // Trailing comma on its own line
    const std::string trailing =
        "[\n"
        "  1,\n"
        "  2,\n"
        "]";
    CHECK(parse_fail(trailing) ==
          "line 4, column 1: trailing comma is not allowed");

    // Bad number deep in a document, with CRLF line ends: the CR counts
    // as a column, only LF starts a new line
    const std::string crlf =
        "{\r\n"
        "  \"v\": [1, 2,\r\n"
        "    03]\r\n"
        "}";
    CHECK(parse_fail(crlf) ==
          "line 3, column 6: leading zeros are not allowed");

    // Control character inside a string on the third line
    const std::string raw_tab = "[\n\"ok\",\n\"bad\ttab\"\n]";
    CHECK(parse_fail(raw_tab) ==
          "line 3, column 5: control character in string "
          "(must be written as an escape)");

    // Truncated document: the error is where the input ends
    CHECK(parse_fail("{\n  \"a\": [\n    1,\n    2") ==
          "line 4, column 6: unexpected end of input in array");
    CHECK(parse_fail("[\n\n\n") ==
          "line 4, column 1: unexpected end of input");

    // Columns count bytes: two 2-byte characters before the error
    CHECK(parse_fail("[\"\xC3\xA9\xC3\xA9\", x]") ==
          "line 1, column 10: unexpected character 'x'");

    // Unexpected character on a later line
    CHECK(parse_fail("[\n1,\n2,\n  @\n]") ==
          "line 4, column 3: unexpected character '@'");

    // A successful parse clears an old error message
    JsonValue value;
    std::string error = "old error";
    CHECK(gfx::parse_json("[1]", value, error));
    CHECK(error.empty());
    CHECK(!gfx::parse_json("[1", value, error));
    CHECK(!error.empty());
    CHECK(value.is_null());
    CHECK(gfx::parse_json("{\"a\": 1}", value, error));
    CHECK(error.empty() && value["a"].as_int() == 1);

    // A failed parse resets a previously filled value to null
    CHECK(!gfx::parse_json("oops", value, error));
    CHECK(value.is_null());
}

void test_depth() {
    const std::size_t limit = 256;

    // Exactly the limit parses
    std::string text(limit, '[');
    text += std::string(limit, ']');
    JsonValue v = parse_ok(text);

    // Walk down to check the structure is really 256 deep
    std::size_t depth = 0;
    const JsonValue* node = &v;
    while (node->is_array() && node->size() == 1) {
        node = &(*node)[0];
        ++depth;
    }
    CHECK(depth == limit - 1);
    CHECK(node->is_array() && node->size() == 0);

    // One more fails, at the bracket that is too deep
    text = std::string(limit + 1, '[') + std::string(limit + 1, ']');
    CHECK(parse_fail(text) == "line 1, column 257: too deeply nested");

    // Objects count too, and so do mixed nestings
    std::string objects;
    for (std::size_t i = 0; i < limit; ++i) {
        objects += "{\"a\":";
    }
    objects += "1";
    objects += std::string(limit, '}');
    v = parse_ok(objects);
    CHECK(v["a"]["a"].is_object());

    std::string too_many_objects = "{\"a\":" + objects + "}";
    CHECK(contains_text(parse_fail(too_many_objects), "too deeply nested"));

    std::string mixed;
    for (std::size_t i = 0; i < limit; ++i) {
        mixed += (i % 2 == 0) ? "[" : "{\"k\":";
    }
    mixed += "0";
    for (std::size_t i = limit; i > 0; --i) {
        mixed += ((i - 1) % 2 == 0) ? "]" : "}";
    }
    parse_ok(mixed);

    std::string mixed_deeper = "[" + mixed + "]";
    CHECK(contains_text(parse_fail(mixed_deeper), "too deeply nested"));

    // Siblings do not add up: many shallow containers are fine
    std::string wide = "[";
    for (int i = 0; i < 1000; ++i) {
        wide += "[[[]]],";
    }
    wide += "[]]";
    CHECK(parse_ok(wide).size() == 1001);

    // A huge nesting fails fast without running out of stack, even when
    // the input is also unterminated
    CHECK(contains_text(parse_fail(std::string(1000000, '[')),
                        "too deeply nested"));
    std::string deep_objects;
    for (int i = 0; i < 100000; ++i) {
        deep_objects += "{\"a\":[";
    }
    CHECK(contains_text(parse_fail(deep_objects), "too deeply nested"));

    // The limit is exported
    static_assert(gfx::kMaxJsonDepth == 256, "tests assume a limit of 256");
}

void test_copy_and_move() {
    JsonValue original = parse_ok("{\"a\": [1, 2, {\"b\": \"text\"}]}");

    JsonValue copy = original;
    CHECK(copy["a"][2]["b"].as_string() == "text");
    CHECK(original["a"][2]["b"].as_string() == "text");
    CHECK(&copy["a"] != &original["a"]);

    JsonValue moved = std::move(copy);
    CHECK(moved["a"].size() == 3);

    JsonValue assigned;
    assigned = original;
    CHECK(assigned["a"][1].as_int() == 2);

    // Parsing into a value that already holds something replaces it
    CHECK(parse_ok("[1]").size() == 1);
    JsonValue reuse = original;
    std::string error;
    CHECK(gfx::parse_json("\"s\"", reuse, error));
    CHECK(reuse.is_string() && reuse.size() == 0 && reuse.members().empty());
    CHECK(reuse.as_string() == "s");
}

// ---------------------------------------------------------------------
// A glTF-shaped document
// ---------------------------------------------------------------------

const char* const kGltf =
        "\n"
        "{\n"
        "  \"asset\": { \"version\": \"2.0\", \"generator\": \"gfxlab test \\u00e9\" },\n"
        "  \"scene\": 0,\n"
        "  \"scenes\": [ { \"name\": \"Scene\", \"nodes\": [0, 2] } ],\n"
        "  \"nodes\": [\n"
        "    { \"name\": \"Root\", \"children\": [1],\n"
        "      \"matrix\": [1,0,0,0, 0,1,0,0, 0,0,1,0, 5,6,7,1] },\n"
        "    { \"name\": \"Child\", \"mesh\": 0,\n"
        "      \"translation\": [1.5, -2, 0.25],\n"
        "      \"rotation\": [0, 0.7071068, 0, 0.7071068],\n"
        "      \"scale\": [2, 2, 2] },\n"
        "    { \"name\": \"Camera node\", \"camera\": 0 }\n"
        "  ],\n"
        "  \"meshes\": [\n"
        "    { \"name\": \"Triangle\",\n"
        "      \"primitives\": [\n"
        "        { \"attributes\": { \"POSITION\": 0, \"NORMAL\": 1, \"TEXCOORD_0\": 2 },\n"
        "          \"indices\": 3, \"material\": 0, \"mode\": 4 }\n"
        "      ] }\n"
        "  ],\n"
        "  \"accessors\": [\n"
        "    { \"bufferView\": 0, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\",\n"
        "      \"max\": [1.0, 1.0, 0.0], \"min\": [-1.0, -1.0, 0.0] },\n"
        "    { \"bufferView\": 1, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC3\" },\n"
        "    { \"bufferView\": 2, \"componentType\": 5126, \"count\": 3, \"type\": \"VEC2\" },\n"
        "    { \"bufferView\": 3, \"componentType\": 5123, \"count\": 3, \"type\": \"SCALAR\" }\n"
        "  ],\n"
        "  \"bufferViews\": [\n"
        "    { \"buffer\": 0, \"byteOffset\": 0, \"byteLength\": 36, \"target\": 34962 },\n"
        "    { \"buffer\": 0, \"byteOffset\": 36, \"byteLength\": 36, \"target\": 34962 },\n"
        "    { \"buffer\": 0, \"byteOffset\": 72, \"byteLength\": 24, \"target\": 34962 },\n"
        "    { \"buffer\": 0, \"byteOffset\": 96, \"byteLength\": 6, \"target\": 34963 }\n"
        "  ],\n"
        "  \"buffers\": [\n"
        "    { \"byteLength\": 102,\n"
        "      \"uri\": \"data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==\" }\n"
        "  ],\n"
        "  \"materials\": [\n"
        "    { \"name\": \"Red\",\n"
        "      \"pbrMetallicRoughness\": {\n"
        "        \"baseColorFactor\": [0.8, 0.1, 0.1, 1.0],\n"
        "        \"metallicFactor\": 0.0,\n"
        "        \"roughnessFactor\": 0.5,\n"
        "        \"baseColorTexture\": { \"index\": 0, \"texCoord\": 0 } },\n"
        "      \"doubleSided\": true,\n"
        "      \"alphaMode\": \"OPAQUE\" }\n"
        "  ],\n"
        "  \"textures\": [ { \"source\": 0, \"sampler\": 0 } ],\n"
        "  \"images\": [ { \"uri\": \"tex/albedo.png\" } ],\n"
        "  \"samplers\": [ { \"magFilter\": 9729, \"minFilter\": 9987, \"wrapS\": 10497 } ],\n"
        "  \"cameras\": [ { \"type\": \"perspective\",\n"
        "                 \"perspective\": { \"yfov\": 0.7, \"znear\": 0.1 } } ],\n"
        "  \"extensionsUsed\": []\n"
        "}\n";

void test_gltf_document() {
    const JsonValue doc = parse_ok(kGltf);

    CHECK(doc.is_object());
    CHECK(doc["asset"]["version"].as_string() == "2.0");
    CHECK(doc["asset"]["generator"].as_string() == "gfxlab test \xC3\xA9");
    CHECK(doc["scene"].as_int(-1) == 0);
    CHECK(doc["scenes"].size() == 1);
    CHECK(doc["scenes"][0]["nodes"].size() == 2);
    CHECK(doc["scenes"][0]["nodes"][1].as_int(-1) == 2);

    // Nodes
    const JsonValue& nodes = doc["nodes"];
    CHECK(nodes.size() == 3);
    CHECK(nodes[0]["name"].as_string() == "Root");
    CHECK(nodes[0]["children"][0].as_int(-1) == 1);
    CHECK(nodes[0]["matrix"].size() == 16);
    CHECK(approx(nodes[0]["matrix"][12].as_float(), 5.0));
    CHECK(approx(nodes[0]["matrix"][13].as_float(), 6.0));
    CHECK(approx(nodes[0]["matrix"][14].as_float(), 7.0));
    CHECK(approx(nodes[0]["matrix"][15].as_float(), 1.0));
    CHECK(!nodes[0].contains("mesh"));
    CHECK(nodes[0]["mesh"].as_int(-1) == -1);
    CHECK(!nodes[0].contains("translation"));

    CHECK(nodes[1]["mesh"].as_int(-1) == 0);
    CHECK(nodes[1]["translation"].size() == 3);
    CHECK(approx(nodes[1]["translation"][0].as_float(), 1.5));
    CHECK(approx(nodes[1]["translation"][1].as_float(), -2.0));
    CHECK(approx(nodes[1]["translation"][2].as_float(), 0.25));
    CHECK(nodes[1]["rotation"].size() == 4);
    CHECK(approx(nodes[1]["rotation"][1].as_float(), 0.7071068, 1e-6));
    CHECK(approx(nodes[1]["rotation"][3].as_float(), 0.7071068, 1e-6));
    CHECK(nodes[1]["scale"].size() == 3);
    CHECK(approx(nodes[1]["scale"][2].as_float(1.0f), 2.0));
    CHECK(nodes[2]["camera"].as_int(-1) == 0);

    // Reading a vector with defaults, the way a loader would
    float t[3] = {0, 0, 0};
    for (std::size_t i = 0; i < 3; ++i) {
        t[i] = nodes[1]["translation"][i].as_float(0.0f);
    }
    CHECK(approx(t[0], 1.5) && approx(t[1], -2.0) && approx(t[2], 0.25));

    // Meshes
    const JsonValue& prim = doc["meshes"][0]["primitives"][0];
    CHECK(doc["meshes"][0]["name"].as_string() == "Triangle");
    CHECK(doc["meshes"][0]["primitives"].size() == 1);
    CHECK(prim["attributes"]["POSITION"].as_int(-1) == 0);
    CHECK(prim["attributes"]["NORMAL"].as_int(-1) == 1);
    CHECK(prim["attributes"]["TEXCOORD_0"].as_int(-1) == 2);
    CHECK(prim["attributes"]["COLOR_0"].as_int(-1) == -1);
    CHECK(prim["attributes"].size() == 3);
    CHECK(prim["indices"].as_int(-1) == 3);
    CHECK(prim["material"].as_int(-1) == 0);
    CHECK(prim["mode"].as_int(4) == 4);

    // Attribute order is the file order
    CHECK(prim["attributes"].members().size() == 3);
    CHECK(prim["attributes"].members()[0].key == "POSITION");
    CHECK(prim["attributes"].members()[2].key == "TEXCOORD_0");

    // Accessors
    const JsonValue& accessors = doc["accessors"];
    CHECK(accessors.size() == 4);
    CHECK(accessors[0]["componentType"].as_int() == 5126);
    CHECK(accessors[0]["count"].as_int() == 3);
    CHECK(accessors[0]["type"].as_string() == "VEC3");
    CHECK(accessors[0]["max"].size() == 3);
    CHECK(approx(accessors[0]["min"][0].as_float(), -1.0));
    CHECK(accessors[1]["max"].is_null());
    CHECK(accessors[3]["type"].as_string() == "SCALAR");
    CHECK(accessors[3]["componentType"].as_int() == 5123);

    // Buffer views and buffers
    CHECK(doc["bufferViews"].size() == 4);
    CHECK(doc["bufferViews"][3]["byteOffset"].as_int(-1) == 96);
    CHECK(doc["bufferViews"][3]["byteLength"].as_int(-1) == 6);
    CHECK(doc["bufferViews"][3]["target"].as_int(-1) == 34963);
    CHECK(doc["buffers"][0]["byteLength"].as_int() == 102);

    const std::string& uri = doc["buffers"][0]["uri"].as_string();
    CHECK(starts_with(uri, "data:application/octet-stream;base64,AAAA"));
    CHECK(uri.size() > 200);
    CHECK(uri.substr(uri.size() - 2) == "==");

    // Materials
    const JsonValue& material = doc["materials"][0];
    CHECK(material["name"].as_string() == "Red");
    const JsonValue& pbr = material["pbrMetallicRoughness"];
    CHECK(pbr.is_object());
    CHECK(pbr["baseColorFactor"].size() == 4);
    CHECK(approx(pbr["baseColorFactor"][0].as_float(), 0.8));
    CHECK(approx(pbr["baseColorFactor"][3].as_float(), 1.0));
    CHECK(approx(pbr["metallicFactor"].as_float(1.0f), 0.0));
    CHECK(approx(pbr["roughnessFactor"].as_float(1.0f), 0.5));
    CHECK(pbr["baseColorTexture"]["index"].as_int(-1) == 0);
    CHECK(pbr["metallicRoughnessTexture"]["index"].as_int(-1) == -1);
    CHECK(material["doubleSided"].as_bool() == true);
    CHECK(material["alphaMode"].as_string() == "OPAQUE");
    CHECK(approx(material["alphaCutoff"].as_float(0.5f), 0.5));

    // The rest
    CHECK(doc["textures"][0]["source"].as_int(-1) == 0);
    CHECK(doc["images"][0]["uri"].as_string() == "tex/albedo.png");
    CHECK(doc["samplers"][0]["magFilter"].as_int() == 9729);
    CHECK(doc["cameras"][0]["perspective"]["yfov"].as_float() > 0.69f);
    CHECK(doc["extensionsUsed"].is_array() && doc["extensionsUsed"].size() == 0);
    CHECK(doc["extensions"].is_null());
}

// ---------------------------------------------------------------------
// Size and speed
// ---------------------------------------------------------------------

void test_large_document() {
    const int count = 100000;

    std::string text;
    text.reserve(static_cast<std::size_t>(count) * 80);
    text += "[\n";

    for (int i = 0; i < count; ++i) {
        if (i > 0) {
            text += ",\n";
        }
        text += "  {\"id\": " + std::to_string(i) +
                ", \"name\": \"item " + std::to_string(i) +
                "\", \"pos\": [" + std::to_string(i) + ".5, -2, 3e1]" +
                ", \"on\": " + (i % 2 == 0 ? "true" : "false") +
                ", \"none\": null}";
    }
    text += "\n]\n";

    const auto start = std::chrono::steady_clock::now();
    const JsonValue doc = parse_ok(text);
    const auto stop = std::chrono::steady_clock::now();

    const double seconds = std::chrono::duration<double>(stop - start).count();
    std::printf("parsed %.1f MB in %.3f s\n",
                static_cast<double>(text.size()) / (1024.0 * 1024.0), seconds);

    CHECK(doc.is_array());
    CHECK(doc.size() == static_cast<std::size_t>(count));
    CHECK(doc[0]["id"].as_int(-1) == 0);
    CHECK(doc[12345]["id"].as_int(-1) == 12345);
    CHECK(doc[12345]["name"].as_string() == "item 12345");
    CHECK(approx(doc[12345]["pos"][0].as_number(), 12345.5));
    CHECK(approx(doc[12345]["pos"][2].as_number(), 30.0));
    CHECK(doc[99999]["id"].as_int(-1) == 99999);
    CHECK(doc[99999]["on"].as_bool(true) == false);
    CHECK(doc[99998]["on"].as_bool(false) == true);
    CHECK(doc[count - 1]["none"].is_null());
    CHECK(doc[count - 1].contains("none"));
    CHECK(doc[count].is_null());

    // Generous bound; this takes a few tens of milliseconds in Release
    CHECK(seconds < 5.0);

    // One big string and one big flat array
    std::string big = "\"";
    big.append(3 * 1024 * 1024, 'x');
    big += "\\n\\u00e9\"";
    const JsonValue s = parse_ok(big);
    CHECK(s.as_string().size() == 3 * 1024 * 1024 + 3);

    std::string numbers = "[";
    for (int i = 0; i < 200000; ++i) {
        if (i > 0) {
            numbers += ',';
        }
        numbers += std::to_string(i);
    }
    numbers += "]";
    const JsonValue arr = parse_ok(numbers);
    CHECK(arr.size() == 200000);
    CHECK(arr[199999].as_int(-1) == 199999);
}

} // namespace

int main() {
    test_types();
    test_fallbacks();
    test_as_int_and_float();
    test_queries();
    test_members_order_and_duplicates();
    test_numbers();
    test_number_range();
    test_invalid_numbers();
    test_literals();
    test_string_escapes();
    test_string_errors();
    test_whitespace_and_bom();
    test_document_errors();
    test_error_positions();
    test_depth();
    test_copy_and_move();
    test_gltf_document();
    test_large_document();

    if (g_failures == 0) {
        std::printf("All JSON tests passed\n");
        return 0;
    }

    std::printf("%d JSON test failure(s)\n", g_failures);
    return 1;
}
