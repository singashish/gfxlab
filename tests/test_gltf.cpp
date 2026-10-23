#include "core/base64.h"
#include "core/gltf_loader.h"
#include "core/log.h"
#include "core/mat4.h"
#include "core/model.h"
#include "core/transform.h"
#include "check.h"
#include "image_test_data.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

// Most documents below are written as JSON with single quotes, which is
// much easier to read in C++ than escaped double quotes: q() turns the
// single quotes into double quotes. (No JSON text here contains an
// apostrophe.)

namespace {

using Bytes = std::vector<std::uint8_t>;

const std::string kAssets = GFX_ASSETS_DIR;

// Records a failure from code that has already found out that something
// is wrong (and printed why). CHECK(false) would do the same, but a
// condition that is a constant makes some compilers complain
void check_failed(int line) {
    std::fprintf(stderr, "FAIL %s:%d  (see the message above)\n", __FILE__,
                 line);
    ++g_failures;
}

#define CHECK_FAILED() check_failed(__LINE__)

// ---------------------------------------------------------------------
// Helpers: text, binary data and documents
// ---------------------------------------------------------------------

std::string q(std::string text) {
    for (char& c : text) {
        if (c == '\'') {
            c = '"';
        }
    }

    return text;
}

bool has(const std::string& text, const char* part) {
    return text.find(part) != std::string::npos;
}

bool approx_vec(const gfx::Vec3& a, const gfx::Vec3& b, double eps = 1e-5) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

// Base64 encoder for building data: URIs (the real code only decodes)
std::string base64_encode(const Bytes& data, bool padding = true) {
    static const char kAlphabet[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string out;

    for (std::size_t i = 0; i < data.size(); i += 3) {
        const std::size_t left = data.size() - i;
        const std::uint32_t b0 = data[i];
        const std::uint32_t b1 = left > 1 ? data[i + 1] : 0u;
        const std::uint32_t b2 = left > 2 ? data[i + 2] : 0u;
        const std::uint32_t triple = (b0 << 16) | (b1 << 8) | b2;

        out.push_back(kAlphabet[(triple >> 18) & 63u]);
        out.push_back(kAlphabet[(triple >> 12) & 63u]);

        if (left > 1) {
            out.push_back(kAlphabet[(triple >> 6) & 63u]);
        } else if (padding) {
            out.push_back('=');
        }

        if (left > 2) {
            out.push_back(kAlphabet[triple & 63u]);
        } else if (padding) {
            out.push_back('=');
        }
    }

    return out;
}

std::string data_uri(const Bytes& data) {
    return "data:application/octet-stream;base64," + base64_encode(data);
}

// Little-endian binary writer
struct Bin {
    Bytes data;

    void u8(std::uint32_t v) {
        data.push_back(static_cast<std::uint8_t>(v & 0xFFu));
    }

    void u16(std::uint32_t v) {
        u8(v);
        u8(v >> 8);
    }

    void u32(std::uint32_t v) {
        u16(v & 0xFFFFu);
        u16(v >> 16);
    }

    void f32(float f) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &f, sizeof(bits));
        u32(bits);
    }

    void vec3(float x, float y, float z) {
        f32(x);
        f32(y);
        f32(z);
    }

    // Filler bytes that a correct reader must skip over
    void junk(std::size_t count) {
        for (std::size_t i = 0; i < count; ++i) {
            u8(0xEEu);
        }
    }

    void append(const Bytes& more) {
        data.insert(data.end(), more.begin(), more.end());
    }
};

// The triangle (0,0,0) (1,0,0) (0,1,0), the default geometry
Bytes triangle_bytes() {
    Bin b;
    b.vec3(0.0f, 0.0f, 0.0f);
    b.vec3(1.0f, 0.0f, 0.0f);
    b.vec3(0.0f, 1.0f, 0.0f);
    return b.data;
}

// A whole glTF document. The defaults describe one triangle: a buffer
// with 3 positions, one view over all of it, accessor 0 for the
// positions, a mesh with one primitive, a node using the mesh and a
// scene with that node. Tests replace the parts they are about.
// Strings use single quotes (see q)
struct Doc {
    Bytes bin = triangle_bytes();
    std::string buffer_uri;     // empty: embed bin as a data URI
    bool glb_bin = false;       // buffer 0 without uri (the BIN chunk)
    std::string views;          // empty: one view over the whole buffer
    std::string accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'}";
    std::string meshes = "{'primitives':[{'attributes':{'POSITION':0}}]}";
    std::string nodes = "{'mesh':0}";
    std::string scenes = "{'nodes':[0]}";   // empty: no scenes
    std::string scene_member = ",'scene':0";
    std::string extra;          // more top-level members, with leading comma

    std::string json() const {
        const std::string length = std::to_string(bin.size());

        std::string buffer = "{'byteLength':" + length;

        if (!buffer_uri.empty()) {
            buffer += ",'uri':'" + buffer_uri + "'";
        } else if (!glb_bin) {
            buffer += ",'uri':'" + data_uri(bin) + "'";
        }

        buffer += "}";

        std::string out = "{'asset':{'version':'2.0'},'buffers':[" + buffer +
                          "],'bufferViews':[" +
                          (views.empty() ? "{'buffer':0,'byteLength':" +
                                               length + "}"
                                         : views) +
                          "],'accessors':[" + accessors + "],'meshes':[" +
                          meshes + "],'nodes':[" + nodes + "]";

        if (!scenes.empty()) {
            out += ",'scenes':[" + scenes + "]" + scene_member;
        }

        out += extra + "}";
        return q(out);
    }
};

bool parse_text(const std::string& text, gfx::Model& model,
                std::string& error, const std::string& base_dir = "") {
    return gfx::parse_gltf(
        reinterpret_cast<const std::uint8_t*>(text.data()), text.size(),
        base_dir, model, error);
}

bool parse_bytes(const Bytes& data, gfx::Model& model, std::string& error,
                 const std::string& base_dir = "") {
    return gfx::parse_gltf(data.data(), data.size(), base_dir, model, error);
}

bool parse_doc(const Doc& doc, gfx::Model& model) {
    std::string error;
    const bool ok = parse_text(doc.json(), model, error);

    if (!ok) {
        std::fprintf(stderr, "  unexpected error: %s\n", error.c_str());
    }

    return ok;
}

// True if the document is rejected with a message containing `part`
bool fails_with(const Doc& doc, const char* part) {
    gfx::Model model;
    std::string error;

    if (parse_text(doc.json(), model, error)) {
        std::fprintf(stderr, "  expected an error containing '%s', got "
                     "success\n", part);
        return false;
    }

    if (!has(error, part)) {
        std::fprintf(stderr, "  expected an error containing '%s', got "
                     "'%s'\n", part, error.c_str());
        return false;
    }

    // A failed parse leaves nothing behind
    return model.meshes.empty() && model.instances.empty();
}

// The change of axes the loader applies: (x, y, z) -> (x, -z, y)
gfx::Vec3 to_z_up(gfx::Vec3 p) {
    return {p.x, -p.z, p.y};
}

// ---------------------------------------------------------------------
// GLB containers
// ---------------------------------------------------------------------

struct Chunk {
    std::uint32_t type;
    Bytes data;
};

constexpr std::uint32_t kJsonChunk = 0x4E4F534Au;
constexpr std::uint32_t kBinChunk = 0x004E4942u;

Bytes text_bytes(const std::string& text) {
    return Bytes(text.begin(), text.end());
}

// Pads to a multiple of 4 with the given byte
Bytes padded(Bytes data, std::uint8_t fill) {
    while (data.size() % 4 != 0) {
        data.push_back(fill);
    }

    return data;
}

void put_u32(Bytes& b, std::size_t at, std::uint32_t v) {
    b[at] = static_cast<std::uint8_t>(v & 0xFFu);
    b[at + 1] = static_cast<std::uint8_t>((v >> 8) & 0xFFu);
    b[at + 2] = static_cast<std::uint8_t>((v >> 16) & 0xFFu);
    b[at + 3] = static_cast<std::uint8_t>((v >> 24) & 0xFFu);
}

Bytes make_glb_from_chunks(const std::vector<Chunk>& chunks) {
    Bin out;
    out.u32(0x46546C67u);   // "glTF"
    out.u32(2);
    out.u32(0);             // total length, patched below

    for (const Chunk& chunk : chunks) {
        out.u32(static_cast<std::uint32_t>(chunk.data.size()));
        out.u32(chunk.type);
        out.append(chunk.data);
    }

    put_u32(out.data, 8, static_cast<std::uint32_t>(out.data.size()));
    return out.data;
}

// A .glb whose buffer 0 is the BIN chunk
Bytes make_glb(const Doc& source) {
    Doc doc = source;
    doc.glb_bin = true;

    return make_glb_from_chunks({
        {kJsonChunk, padded(text_bytes(doc.json()), ' ')},
        {kBinChunk, padded(doc.bin, 0)}
    });
}

// ---------------------------------------------------------------------
// base64
// ---------------------------------------------------------------------

std::string decoded_text(const char* base64, bool& ok) {
    Bytes out;
    ok = gfx::decode_base64(base64, out);
    return std::string(out.begin(), out.end());
}

void test_base64() {
    bool ok = false;

    // The test vectors of RFC 4648, section 10, with and without padding
    CHECK(decoded_text("", ok) == "" && ok);
    CHECK(decoded_text("Zg==", ok) == "f" && ok);
    CHECK(decoded_text("Zg", ok) == "f" && ok);
    CHECK(decoded_text("Zm8=", ok) == "fo" && ok);
    CHECK(decoded_text("Zm8", ok) == "fo" && ok);
    CHECK(decoded_text("Zm9v", ok) == "foo" && ok);
    CHECK(decoded_text("Zm9vYg==", ok) == "foob" && ok);
    CHECK(decoded_text("Zm9vYg", ok) == "foob" && ok);
    CHECK(decoded_text("Zm9vYmE=", ok) == "fooba" && ok);
    CHECK(decoded_text("Zm9vYmE", ok) == "fooba" && ok);
    CHECK(decoded_text("Zm9vYmFy", ok) == "foobar" && ok);

    // The two characters that are not letters or digits, and all bits
    Bytes bytes;
    CHECK(gfx::decode_base64("/+8A", bytes));
    CHECK(bytes.size() == 3 && bytes[0] == 0xFF && bytes[1] == 0xEF &&
          bytes[2] == 0x00);

    // Invalid input is refused and leaves nothing in the output
    const char* const bad[] = {
        "Zm9v!",      // character outside the alphabet
        "Zm 9v",      // whitespace inside
        "Zm9v\\n",    // a backslash is not in the alphabet either
        "Zm9v\n",     // newline at the end
        "Z",          // 6 bits cannot make a byte
        "Zm9vY",      // 4n + 1 characters
        "Zg=",        // padding, but the length is not a multiple of 4
        "Z===",       // three paddings
        "=Zg=",       // padding at the start
        "Zg=A",       // padding in the middle
        "Zm=v",
        "Zm9v====",   // more than two paddings
        "Zm-_",       // URL-safe alphabet is not accepted
    };

    for (const char* text : bad) {
        bytes.assign(5, 1);

        if (gfx::decode_base64(text, bytes) || !bytes.empty()) {
            std::fprintf(stderr, "  base64 should be refused: %s\n", text);
            CHECK_FAILED();
        }
    }

    // Round trip of every length from 0 to 40 with arbitrary bytes, with
    // and without padding
    std::uint32_t state = 7;

    for (std::size_t length = 0; length <= 40; ++length) {
        Bytes original;

        for (std::size_t i = 0; i < length; ++i) {
            state = state * 1664525u + 1013904223u;
            original.push_back(static_cast<std::uint8_t>(state >> 24));
        }

        for (const bool padding : {true, false}) {
            const std::string text = base64_encode(original, padding);

            if (!gfx::decode_base64(text, bytes) || bytes != original) {
                std::fprintf(stderr, "  base64 round trip failed at %zu\n",
                             length);
                CHECK_FAILED();
            }
        }
    }
}

// ---------------------------------------------------------------------
// Basics: version, JSON errors, the simplest model, Y-up to Z-up
// ---------------------------------------------------------------------

void test_basic_model() {
    gfx::Model model;
    CHECK(parse_doc(Doc{}, model));

    CHECK(model.meshes.size() == 1);
    CHECK(model.instances.size() == 1);
    CHECK(model.images.empty());
    CHECK(model.instances[0].mesh == 0);

    const gfx::Mesh& mesh = model.meshes[0];

    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.indices.size() == 3);
    CHECK(mesh.indices[0] == 0 && mesh.indices[1] == 1 &&
          mesh.indices[2] == 2);
    CHECK(!mesh.has_normals());
    CHECK(!mesh.has_texcoords());
    CHECK(mesh.edges.empty());
    CHECK(mesh.triangle_materials.size() == 1);
    CHECK(mesh.triangle_materials[0] == 0);

    // No material in the file: one default material
    CHECK(mesh.materials.size() == 1);
    CHECK(mesh.materials[0].name == "default");

    // Y-up to Z-up: (0, 1, 0) ends at (0, 0, 1)
    CHECK(approx_vec(mesh.positions[0], {0.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(mesh.positions[1], {1.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(mesh.positions[2], {0.0f, 0.0f, 1.0f}));

    // An identity node stays an identity
    const gfx::Mat4& t = model.instances[0].transform;
    CHECK(approx_vec(gfx::transform_point(t, {1.0f, 2.0f, 3.0f}),
                     {1.0f, 2.0f, 3.0f}));
}

void test_version_and_json_errors() {
    gfx::Model model;
    std::string error;

    const std::string good = Doc{}.json();

    // Version
    CHECK(parse_text(good, model, error));

    std::string text = good;
    const std::size_t at = text.find("2.0");
    CHECK(at != std::string::npos);

    text.replace(at, 3, "2.1");
    CHECK(parse_text(text, model, error));

    text.replace(at, 3, "1.0");
    CHECK(!parse_text(text, model, error) && has(error, "version"));

    text.replace(at, 3, "3.0");
    CHECK(!parse_text(text, model, error) && has(error, "version"));

    CHECK(!parse_text(q("{'nodes':[]}"), model, error) &&
          has(error, "asset.version"));
    CHECK(!parse_text(q("{'asset':{}}"), model, error) &&
          has(error, "asset.version"));

    // Syntax errors come from the JSON parser, with line and column
    CHECK(!parse_text("", model, error) && has(error, "JSON: line 1"));
    CHECK(!parse_text("{", model, error) && has(error, "JSON: line 1, column"));
    CHECK(!parse_text("{\n  \"asset\":\n    oops}", model, error) &&
          has(error, "JSON: line 3, column 5"));
    CHECK(!parse_text(good + " trailing", model, error) &&
          has(error, "JSON: "));

    // Valid JSON that is not a glTF document
    CHECK(!parse_text("[1, 2]", model, error) && has(error, "JSON object"));
    CHECK(!parse_text("42", model, error));

    // A document with nothing to show
    CHECK(!parse_text(q("{'asset':{'version':'2.0'}}"), model, error) &&
          has(error, "no meshes in the scene"));

    // A byte order mark in front of the JSON is fine
    CHECK(parse_text("\xEF\xBB\xBF" + good, model, error));
}

void test_y_up_to_z_up() {
    // Normals turn like positions: (0,1,0) -> (0,0,1), (0,0,1) -> (0,-1,0)
    Bin b;
    b.vec3(0.0f, 0.0f, 0.0f);
    b.vec3(0.0f, 1.0f, 0.0f);
    b.vec3(0.0f, 0.0f, 1.0f);
    b.vec3(0.0f, 1.0f, 0.0f);   // normals
    b.vec3(0.0f, 0.0f, 1.0f);
    b.vec3(1.0f, 0.0f, 0.0f);

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':36},"
                "{'buffer':0,'byteOffset':36,'byteLength':36}";
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
        "{'bufferView':1,'componentType':5126,'count':3,'type':'VEC3'}";
    doc.meshes = "{'primitives':[{'attributes':"
                 "{'POSITION':0,'NORMAL':1}}]}";

    gfx::Model model;
    CHECK(parse_doc(doc, model));
    CHECK(model.meshes.size() == 1);

    const gfx::Mesh& mesh = model.meshes[0];
    CHECK(mesh.positions.size() == 3 && mesh.normals.size() == 3);
    CHECK(approx_vec(mesh.positions[1], {0.0f, 0.0f, 1.0f}));
    CHECK(approx_vec(mesh.positions[2], {0.0f, -1.0f, 0.0f}));
    CHECK(approx_vec(mesh.normals[0], {0.0f, 0.0f, 1.0f}));
    CHECK(approx_vec(mesh.normals[1], {0.0f, -1.0f, 0.0f}));
    CHECK(approx_vec(mesh.normals[2], {1.0f, 0.0f, 0.0f}));
}

// ---------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------

void test_accessor_offset_and_stride() {
    // 8 junk bytes, then 3 vertices of 16 bytes: 4 junk bytes and a
    // position. The view starts at byte 8, the accessor 4 bytes into it
    Bin b;
    b.junk(8);
    const float values[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};

    for (const auto& v : values) {
        b.junk(4);
        b.vec3(v[0], v[1], v[2]);
    }

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteOffset':8,'byteLength':48,"
                "'byteStride':16}";
    doc.accessors = "{'bufferView':0,'byteOffset':4,'componentType':5126,"
                    "'count':3,'type':'VEC3'}";

    gfx::Model model;
    CHECK(parse_doc(doc, model));
    CHECK(model.meshes.size() == 1);

    if (!model.meshes.empty()) {
        const gfx::Mesh& mesh = model.meshes[0];
        CHECK(mesh.positions.size() == 3);

        for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
            CHECK(approx_vec(mesh.positions[i],
                             to_z_up({values[i][0], values[i][1],
                                      values[i][2]})));
        }
    }

    // The same data one byte short: the last element no longer fits in
    // the view
    doc.views = "{'buffer':0,'byteOffset':8,'byteLength':47,"
                "'byteStride':16}";
    CHECK(fails_with(doc, "accessor 0 runs past the end of buffer view 0"));

    // The view itself past the end of the buffer
    doc.views = "{'buffer':0,'byteOffset':9,'byteLength':48,"
                "'byteStride':16}";
    CHECK(fails_with(doc, "buffer view 0 runs past the end of buffer 0"));

    // A stride smaller than the element
    doc.views = "{'buffer':0,'byteOffset':8,'byteLength':48,"
                "'byteStride':8}";
    CHECK(fails_with(doc, "byteStride"));
}

// Three vertices with texture coordinates given as the raw bytes `uv`
// and the accessor properties `component` (JSON members)
Doc uv_doc(const Bytes& uv, const std::string& component) {
    Bin b;
    b.append(triangle_bytes());
    b.append(uv);

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':36},"
                "{'buffer':0,'byteOffset':36,'byteLength':" +
                std::to_string(uv.size()) + "}";
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
        "{'bufferView':1," + component + ",'count':3,'type':'VEC2'}";
    doc.meshes = "{'primitives':[{'attributes':"
                 "{'POSITION':0,'TEXCOORD_0':1}}]}";
    return doc;
}

void check_uvs(const Doc& doc, const float (&expected)[3][2]) {
    gfx::Model model;
    CHECK(parse_doc(doc, model));

    if (model.meshes.size() != 1 || model.meshes[0].texcoords.size() != 3) {
        CHECK_FAILED();
        return;
    }

    for (std::size_t i = 0; i < 3; ++i) {
        const gfx::Vec2 uv = model.meshes[0].texcoords[i];

        if (!approx(uv.x, expected[i][0], 1e-5) ||
            !approx(uv.y, expected[i][1], 1e-5)) {
            std::fprintf(stderr, "  uv %zu is (%g, %g), expected (%g, %g)\n",
                         i, static_cast<double>(uv.x),
                         static_cast<double>(uv.y),
                         static_cast<double>(expected[i][0]),
                         static_cast<double>(expected[i][1]));
            CHECK_FAILED();
        }
    }
}

void test_texcoords() {
    // Floats. glTF has (0, 0) at the top-left of the image, Mesh at the
    // bottom-left: v is stored as 1 - v
    {
        Bin b;
        b.f32(0.0f); b.f32(0.0f);
        b.f32(0.25f); b.f32(0.75f);
        b.f32(1.0f); b.f32(1.0f);

        const float expected[3][2] = {{0.0f, 1.0f}, {0.25f, 0.25f},
                                      {1.0f, 0.0f}};
        check_uvs(uv_doc(b.data, "'componentType':5126"), expected);
    }

    // Normalized unsigned byte: value / 255
    {
        Bin b;
        b.u8(0); b.u8(255);
        b.u8(51); b.u8(102);
        b.u8(255); b.u8(0);

        const float expected[3][2] = {{0.0f, 0.0f}, {0.2f, 0.6f},
                                      {1.0f, 1.0f}};
        check_uvs(uv_doc(b.data, "'componentType':5121,'normalized':true"),
                  expected);
    }

    // Normalized unsigned short: value / 65535
    {
        Bin b;
        b.u16(0); b.u16(65535);
        b.u16(32768); b.u16(0);
        b.u16(65535); b.u16(65535);

        const float expected[3][2] = {{0.0f, 0.0f},
                                      {32768.0f / 65535.0f, 1.0f},
                                      {1.0f, 0.0f}};
        check_uvs(uv_doc(b.data, "'componentType':5123,'normalized':true"),
                  expected);
    }

    // Normalized signed byte: max(c / 127, -1), so both -128 and -127
    // give -1
    {
        Bin b;
        b.u8(0x80); b.u8(127);
        b.u8(0); b.u8(0x81);
        b.u8(0x81); b.u8(0);

        const float expected[3][2] = {{-1.0f, 0.0f}, {0.0f, 2.0f},
                                      {-1.0f, 1.0f}};
        check_uvs(uv_doc(b.data, "'componentType':5120,'normalized':true"),
                  expected);
    }

    // Normalized signed short: max(c / 32767, -1)
    {
        Bin b;
        b.u16(0x8000); b.u16(32767);
        b.u16(0); b.u16(0x8001);
        b.u16(16384); b.u16(0);

        const float expected[3][2] = {{-1.0f, 0.0f}, {0.0f, 2.0f},
                                      {16384.0f / 32767.0f, 1.0f}};
        check_uvs(uv_doc(b.data, "'componentType':5122,'normalized':true"),
                  expected);
    }

    // Integers that are not normalized are not texture coordinates
    {
        Bin b;
        b.u8(0); b.u8(1); b.u8(2); b.u8(3); b.u8(4); b.u8(5);
        CHECK(fails_with(uv_doc(b.data, "'componentType':5121"),
                         "TEXCOORD_0"));
    }

    // The wrong number of components
    {
        Doc doc = uv_doc(Bytes(12, 0), "'componentType':5126");
        doc.accessors =
            "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
            "{'bufferView':1,'componentType':5126,'count':3,'type':'SCALAR'}";
        CHECK(fails_with(doc, "TEXCOORD_0"));
    }

    // Not one per vertex
    {
        Doc doc = uv_doc(Bytes(24, 0), "'componentType':5126");
        doc.accessors =
            "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
            "{'bufferView':1,'componentType':5126,'count':2,'type':'VEC2'}";
        CHECK(fails_with(doc, "TEXCOORD_0"));
    }
}

void test_accessor_without_buffer_view() {
    // No bufferView means "all zeros", which the specification allows
    Doc doc;
    doc.accessors = "{'componentType':5126,'count':3,'type':'VEC3'}";

    gfx::Model model;
    CHECK(parse_doc(doc, model));
    CHECK(model.meshes.size() == 1 && model.meshes[0].positions.size() == 3);

    if (!model.meshes.empty()) {
        for (const gfx::Vec3& p : model.meshes[0].positions) {
            CHECK(approx_vec(p, {0.0f, 0.0f, 0.0f}));
        }
    }

    // ...but not with a count that would take gigabytes
    doc.accessors = "{'componentType':5126,'count':4000000000,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "count"));
}

void test_accessor_errors() {
    Doc doc;

    doc.accessors = "{'bufferView':0,'componentType':5126,'count':4,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "accessor 0 runs past the end of buffer view 0"));

    doc.accessors = "{'bufferView':0,'componentType':5126,'count':0,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "count"));

    doc.accessors = "{'bufferView':0,'componentType':5126,'count':2.5,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "count"));

    doc.accessors = "{'bufferView':0,'componentType':5126,'count':-3,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "count"));

    doc.accessors = "{'bufferView':0,'componentType':5124,'count':3,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "componentType"));

    doc.accessors = "{'bufferView':0,'componentType':5126,'count':3,"
                    "'type':'VEC5'}";
    CHECK(fails_with(doc, "type"));

    // POSITION must be 3 floats
    doc.accessors = "{'bufferView':0,'componentType':5126,'count':3,"
                    "'type':'VEC2'}";
    CHECK(fails_with(doc, "POSITION"));

    doc.accessors = "{'bufferView':0,'componentType':5123,'count':3,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "POSITION"));

    doc.accessors = "{'bufferView':5,'componentType':5126,'count':3,"
                    "'type':'VEC3'}";
    CHECK(fails_with(doc, "buffer view 5 does not exist"));

    // Huge numbers must not overflow any arithmetic
    doc.accessors = "{'bufferView':0,'byteOffset':9000000000000000,"
                    "'componentType':5126,'count':3,'type':'VEC3'}";
    CHECK(fails_with(doc, "runs past"));

    doc.accessors = "{'bufferView':0,'componentType':5126,"
                    "'count':900000000000000,'type':'VEC3'}";
    CHECK(fails_with(doc, "runs past"));

    doc = Doc{};
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':7}}]}";
    CHECK(fails_with(doc, "accessor 7 does not exist"));

    // Views and buffers
    doc = Doc{};
    doc.views = "{'buffer':0,'byteLength':9000000000000000}";
    CHECK(fails_with(doc, "runs past the end of buffer 0"));

    doc.views = "{'buffer':3,'byteLength':36}";
    CHECK(fails_with(doc, "buffer 3 does not exist"));

    doc.views = "{'buffer':0,'byteLength':-1}";
    CHECK(fails_with(doc, "byteLength"));

    doc.views = "{'buffer':0,'byteOffset':-4,'byteLength':8}";
    CHECK(fails_with(doc, "byteOffset"));

    // A buffer that claims more bytes than its data has
    doc = Doc{};
    std::string text = doc.json();
    const std::size_t at = text.find("\"byteLength\":36");
    CHECK(at != std::string::npos);
    text.replace(at, std::strlen("\"byteLength\":36"),
                 "\"byteLength\":1000000000000");

    gfx::Model model;
    std::string error;
    CHECK(!parse_text(text, model, error) && has(error, "byteLength"));

    // ...and one that claims fewer: the bytes beyond are off limits
    doc = Doc{};
    doc.bin.resize(48, 0);
    doc.views = "{'buffer':0,'byteLength':36}";
    text = doc.json();
    const std::size_t at48 = text.find("\"byteLength\":48");
    CHECK(at48 != std::string::npos);
    text.replace(at48, std::strlen("\"byteLength\":48"), "\"byteLength\":24");
    CHECK(!parse_text(text, model, error) &&
          has(error, "runs past the end of buffer 0"));
}

void test_sparse_accessors() {
    // A sparse accessor skips its primitive, not the file
    Bin b;
    b.append(triangle_bytes());
    b.append(triangle_bytes());

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':72}";
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3',"
        "'sparse':{'count':1}},"
        "{'bufferView':0,'byteOffset':36,'componentType':5126,'count':3,"
        "'type':'VEC3'}";
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0}},"
                 "{'attributes':{'POSITION':1}}]}";

    gfx::Model model;
    CHECK(parse_doc(doc, model));
    CHECK(model.meshes.size() == 1);

    if (!model.meshes.empty()) {
        CHECK(model.meshes[0].positions.size() == 3);
        CHECK(model.meshes[0].triangle_count() == 1);
    }

    // Sparse normals skip it as well
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
        "{'bufferView':0,'byteOffset':36,'componentType':5126,'count':3,"
        "'type':'VEC3','sparse':{'count':1}}";
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0,'NORMAL':1}}]}";
    CHECK(fails_with(doc, "no meshes in the scene"));
}

// ---------------------------------------------------------------------
// Indices and primitive modes
// ---------------------------------------------------------------------

// Six positions, then `indices` as a second view. `component` is the
// componentType of the index accessor, `count` the number of indices
Doc index_doc(const Bin& indices, int component, std::size_t count,
              const std::string& primitive_extra) {
    Bin b;

    for (int i = 0; i < 6; ++i) {
        b.vec3(static_cast<float>(i), 0.0f, 0.0f);
    }

    b.append(indices.data);

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':72},"
                "{'buffer':0,'byteOffset':72,'byteLength':" +
                std::to_string(indices.data.size()) + "}";
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':6,'type':'VEC3'},"
        "{'bufferView':1,'componentType':" + std::to_string(component) +
        ",'count':" + std::to_string(count) + ",'type':'SCALAR'}";
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},"
                 "'indices':1" + primitive_extra + "}]}";
    return doc;
}

std::vector<std::uint32_t> loaded_indices(const Doc& doc) {
    gfx::Model model;

    if (!parse_doc(doc, model) || model.meshes.size() != 1) {
        return {};
    }

    return model.meshes[0].indices;
}

using Indices = std::vector<std::uint32_t>;

void test_index_types() {
    {
        Bin b;
        for (const std::uint32_t v : {0u, 1u, 2u, 2u, 1u, 3u}) {
            b.u8(v);
        }
        CHECK(loaded_indices(index_doc(b, 5121, 6, "")) ==
              (Indices{0, 1, 2, 2, 1, 3}));
    }

    {
        Bin b;
        for (const std::uint32_t v : {5u, 4u, 3u}) {
            b.u16(v);
        }
        CHECK(loaded_indices(index_doc(b, 5123, 3, "")) ==
              (Indices{5, 4, 3}));
    }

    {
        Bin b;
        for (const std::uint32_t v : {0u, 5u, 2u}) {
            b.u32(v);
        }
        CHECK(loaded_indices(index_doc(b, 5125, 3, "")) ==
              (Indices{0, 5, 2}));
    }

    // Indices in a view with a stride (a u16 every 4 bytes)
    {
        Bin b;
        for (const std::uint32_t v : {2u, 1u, 0u}) {
            b.u16(v);
            b.u16(0xFFFFu);
        }

        Doc doc = index_doc(b, 5123, 3, "");
        doc.views = "{'buffer':0,'byteLength':72},"
                    "{'buffer':0,'byteOffset':72,'byteLength':12,"
                    "'byteStride':4}";
        CHECK(loaded_indices(doc) == (Indices{2, 1, 0}));
    }

    // Without indices the vertices are used in order
    {
        Doc doc;
        Bin b;
        for (int i = 0; i < 6; ++i) {
            b.vec3(static_cast<float>(i), 1.0f, 0.0f);
        }
        doc.bin = b.data;
        doc.accessors =
            "{'bufferView':0,'componentType':5126,'count':6,'type':'VEC3'}";
        CHECK(loaded_indices(doc) == (Indices{0, 1, 2, 3, 4, 5}));
    }

    // 4 indices make one triangle; the one left over is ignored
    {
        Bin b;
        for (const std::uint32_t v : {0u, 1u, 2u, 3u}) {
            b.u8(v);
        }
        CHECK(loaded_indices(index_doc(b, 5121, 4, "")) ==
              (Indices{0, 1, 2}));
    }
}

void test_index_errors() {
    Bin b;
    for (const std::uint32_t v : {0u, 1u, 6u}) {
        b.u8(v);
    }
    CHECK(fails_with(index_doc(b, 5121, 3, ""), "out of range"));

    Bin big;
    big.u16(0xFFFFu);
    big.u16(0u);
    big.u16(1u);
    CHECK(fails_with(index_doc(big, 5123, 3, ""), "out of range"));

    Bin bytes;
    bytes.u8(0);
    bytes.u8(1);
    bytes.u8(2);

    // Floats cannot be indices
    Bin one_float;
    one_float.f32(0.0f);
    CHECK(fails_with(index_doc(one_float, 5126, 1, ""), "indices"));

    // More indices than the view holds
    CHECK(fails_with(index_doc(bytes, 5121, 4, ""), "runs past the end"));

    // Not an index accessor at all
    Doc doc = index_doc(bytes, 5121, 3, "");
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},"
                 "'indices':9}]}";
    CHECK(fails_with(doc, "accessor 9 does not exist"));
}

void test_primitive_modes() {
    Bin five;
    for (const std::uint32_t v : {0u, 1u, 2u, 3u, 4u}) {
        five.u8(v);
    }

    // Strip: each index after the first two makes a triangle with the
    // two before it; every second one is flipped to keep the winding
    CHECK(loaded_indices(index_doc(five, 5121, 5, ",'mode':5")) ==
          (Indices{0, 1, 2, 2, 1, 3, 2, 3, 4}));

    // Fan: all triangles share the first vertex
    CHECK(loaded_indices(index_doc(five, 5121, 5, ",'mode':6")) ==
          (Indices{0, 1, 2, 0, 2, 3, 0, 3, 4}));

    // Mode 4 spelled out
    CHECK(loaded_indices(index_doc(five, 5121, 3, ",'mode':4")) ==
          (Indices{0, 1, 2}));

    // Without indices
    Doc strip;
    Bin b;
    for (int i = 0; i < 6; ++i) {
        b.vec3(static_cast<float>(i), 0.0f, 1.0f);
    }
    strip.bin = b.data;
    strip.accessors =
        "{'bufferView':0,'componentType':5126,'count':6,'type':'VEC3'}";

    strip.meshes = "{'primitives':[{'attributes':{'POSITION':0},'mode':5}]}";
    CHECK(loaded_indices(strip) ==
          (Indices{0, 1, 2, 2, 1, 3, 2, 3, 4, 4, 3, 5}));

    strip.meshes = "{'primitives':[{'attributes':{'POSITION':0},'mode':6}]}";
    CHECK(loaded_indices(strip) ==
          (Indices{0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5}));

    // Fewer than 3 vertices make no triangle
    Bin two;
    two.u8(0);
    two.u8(1);
    CHECK(fails_with(index_doc(two, 5121, 2, ",'mode':5"),
                     "no meshes in the scene"));
    CHECK(fails_with(index_doc(two, 5121, 2, ",'mode':6"),
                     "no meshes in the scene"));

    // Points and lines are skipped (and then there is nothing left)
    for (int mode = 0; mode <= 3; ++mode) {
        CHECK(fails_with(index_doc(five, 5121, 5,
                                   ",'mode':" + std::to_string(mode)),
                         "no meshes in the scene"));
    }

    CHECK(fails_with(index_doc(five, 5121, 5, ",'mode':7"), "mode"));
    CHECK(fails_with(index_doc(five, 5121, 5, ",'mode':-1"), "mode"));

    // A skipped primitive does not take the others down with it
    Doc mixed = index_doc(five, 5121, 3, "");
    mixed.meshes =
        "{'primitives':[{'attributes':{'POSITION':0},'indices':1,'mode':1},"
        "{'attributes':{'POSITION':0},'indices':1}]}";
    gfx::Model model;
    CHECK(parse_doc(mixed, model));
    CHECK(model.meshes.size() == 1 && model.meshes[0].triangle_count() == 1);
}

// ---------------------------------------------------------------------
// Meshes with several primitives
// ---------------------------------------------------------------------

void test_primitives_in_one_mesh() {
    // Views: positions A (3), positions B (3), normals (3)
    Bin b;
    b.append(triangle_bytes());
    b.vec3(0.0f, 0.0f, 5.0f);
    b.vec3(1.0f, 0.0f, 5.0f);
    b.vec3(0.0f, 1.0f, 5.0f);
    for (int i = 0; i < 3; ++i) {
        b.vec3(0.0f, 1.0f, 0.0f);
    }

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':108}";
    doc.accessors =
        "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
        "{'bufferView':0,'byteOffset':36,'componentType':5126,'count':3,"
        "'type':'VEC3'},"
        "{'bufferView':0,'byteOffset':72,'componentType':5126,'count':3,"
        "'type':'VEC3'}";

    // Both primitives with normals: they are kept, and the second one's
    // indices are shifted by the first one's vertex count
    doc.meshes = "{'primitives':["
                 "{'attributes':{'POSITION':0,'NORMAL':2}},"
                 "{'attributes':{'POSITION':1,'NORMAL':2}}]}";

    gfx::Model model;
    CHECK(parse_doc(doc, model));
    CHECK(model.meshes.size() == 1);

    if (!model.meshes.empty()) {
        const gfx::Mesh& mesh = model.meshes[0];
        CHECK(mesh.positions.size() == 6);
        CHECK(mesh.normals.size() == 6);
        CHECK(mesh.indices == (Indices{0, 1, 2, 3, 4, 5}));
        CHECK(mesh.triangle_materials.size() == 2);
    }

    // One without normals: the whole mesh has none
    doc.meshes = "{'primitives':["
                 "{'attributes':{'POSITION':0,'NORMAL':2}},"
                 "{'attributes':{'POSITION':1}}]}";
    CHECK(parse_doc(doc, model));

    if (!model.meshes.empty()) {
        CHECK(model.meshes[0].positions.size() == 6);
        CHECK(model.meshes[0].normals.empty());
    }

    // The other way round
    doc.meshes = "{'primitives':["
                 "{'attributes':{'POSITION':0}},"
                 "{'attributes':{'POSITION':1,'NORMAL':2}}]}";
    CHECK(parse_doc(doc, model));

    if (!model.meshes.empty()) {
        CHECK(model.meshes[0].normals.empty());
    }

    // Primitives that read the same accessors share their vertices
    doc.meshes = "{'primitives':["
                 "{'attributes':{'POSITION':0,'NORMAL':2}},"
                 "{'attributes':{'POSITION':0,'NORMAL':2}},"
                 "{'attributes':{'POSITION':0}}]}";
    CHECK(parse_doc(doc, model));

    if (!model.meshes.empty()) {
        const gfx::Mesh& mesh = model.meshes[0];
        CHECK(mesh.positions.size() == 6);   // 3 shared + 3 without normals
        CHECK(mesh.triangle_count() == 3);
        CHECK(mesh.indices[0] == 0 && mesh.indices[3] == 0 &&
              mesh.indices[6] == 3);
    }

    // Same rule for texture coordinates
    {
        Bin t;
        t.append(triangle_bytes());
        for (int i = 0; i < 3; ++i) {
            t.f32(0.5f);
            t.f32(0.5f);
        }

        Doc d = uv_doc(Bytes(t.data.begin() + 36, t.data.end()),
                       "'componentType':5126");
        d.meshes = "{'primitives':["
                   "{'attributes':{'POSITION':0,'TEXCOORD_0':1}},"
                   "{'attributes':{'POSITION':0}}]}";
        CHECK(parse_doc(d, model));

        if (!model.meshes.empty()) {
            CHECK(model.meshes[0].positions.size() == 6);
            CHECK(model.meshes[0].texcoords.empty());
        }
    }

    // A mesh without primitives is a broken file
    doc.meshes = "{'primitives':[]}";
    CHECK(fails_with(doc, "no primitives"));
    doc.meshes = "{'name':'x'}";
    CHECK(fails_with(doc, "no primitives"));

    doc.meshes = "{'primitives':[{'mode':4}]}";
    CHECK(fails_with(doc, "POSITION"));
}

// ---------------------------------------------------------------------
// Nodes and scenes
// ---------------------------------------------------------------------

gfx::Model parse_ok(const Doc& doc) {
    gfx::Model model;

    if (!parse_doc(doc, model)) {
        model = gfx::Model{};
    }

    return model;
}

void test_node_transforms() {
    // translation, rotation (a quaternion x y z w) and scale
    // Rotating (2, 0, 0) by 90 degrees about +Y gives (0, 0, -2)
    {
        Doc doc;
        doc.nodes = "{'mesh':0,'translation':[1,2,3],"
                    "'rotation':[0,0.70710678,0,0.70710678],"
                    "'scale':[2,2,2]}";

        const gfx::Model model = parse_ok(doc);
        CHECK(model.instances.size() == 1);

        if (!model.instances.empty()) {
            // Mesh space (Z-up) (1,0,0) is Y-up (1,0,0), which ends at
            // (1,2,1) in Y-up, so (1,-1,2) in Z-up
            const gfx::Vec3 p = gfx::transform_point(
                model.instances[0].transform, {1.0f, 0.0f, 0.0f});
            CHECK(approx_vec(p, {1.0f, -1.0f, 2.0f}, 1e-4));
        }
    }

    // matrix: 16 numbers, column by column. The last four are the
    // translation (5, 6, 7), and the first column scales x by 3
    {
        Doc doc;
        doc.nodes = "{'mesh':0,'matrix':[3,0,0,0, 0,1,0,0, 0,0,1,0, "
                    "5,6,7,1]}";

        const gfx::Model model = parse_ok(doc);
        CHECK(model.instances.size() == 1);

        if (!model.instances.empty()) {
            const gfx::Mat4& t = model.instances[0].transform;

            // The origin goes to the translation, (1,0,0) 3 further in x
            CHECK(approx_vec(gfx::transform_point(t, {0.0f, 0.0f, 0.0f}),
                             to_z_up({5.0f, 6.0f, 7.0f})));
            CHECK(approx_vec(gfx::transform_point(t, {1.0f, 0.0f, 0.0f}),
                             to_z_up({8.0f, 6.0f, 7.0f})));
        }
    }

    // Defaults: only a scale, only a translation, rotation only
    {
        Doc doc;
        doc.nodes = "{'mesh':0,'scale':[1,2,3]}";
        const gfx::Model model = parse_ok(doc);

        if (!model.instances.empty()) {
            // Y-up scale (1,2,3) in Z-up: x stays, Y-up y is Z-up z,
            // Y-up z is Z-up -y
            const gfx::Vec3 p = gfx::transform_point(
                model.instances[0].transform, {1.0f, 1.0f, 1.0f});
            CHECK(approx_vec(p, {1.0f, 3.0f, 2.0f}));
        }
    }

    // A quaternion that is not exactly unit length is normalised, and
    // an all-zero one means no rotation
    {
        Doc doc;
        doc.nodes = "{'mesh':0,'rotation':[0,1.4142136,0,1.4142136]}";
        gfx::Model model = parse_ok(doc);

        if (!model.instances.empty()) {
            // 90 degrees about Y: Y-up (1,0,0) -> (0,0,-1): Z-up (0,1,0)
            const gfx::Vec3 p = gfx::transform_point(
                model.instances[0].transform, {1.0f, 0.0f, 0.0f});
            CHECK(approx_vec(p, {0.0f, 1.0f, 0.0f}, 1e-4));
        }

        doc.nodes = "{'mesh':0,'rotation':[0,0,0,0]}";
        model = parse_ok(doc);

        if (!model.instances.empty()) {
            const gfx::Vec3 p = gfx::transform_point(
                model.instances[0].transform, {1.0f, 2.0f, 3.0f});
            CHECK(approx_vec(p, {1.0f, 2.0f, 3.0f}));
        }
    }

    // Malformed transforms
    {
        Doc doc;
        doc.nodes = "{'mesh':0,'matrix':[1,0,0,0,0,1,0,0,0,0,1,0,0,0,1]}";
        CHECK(fails_with(doc, "matrix"));

        doc.nodes = "{'mesh':0,'translation':[1,2]}";
        CHECK(fails_with(doc, "translation"));

        doc.nodes = "{'mesh':0,'rotation':[0,0,0]}";
        CHECK(fails_with(doc, "rotation"));

        doc.nodes = "{'mesh':0,'scale':[1,'a',1]}";
        CHECK(fails_with(doc, "scale"));

        doc.nodes = "{'mesh':4}";
        CHECK(fails_with(doc, "mesh"));

        doc.nodes = "{'mesh':0,'children':[3]}";
        CHECK(fails_with(doc, "child"));

        doc.nodes = "{'mesh':0,'children':5}";
        CHECK(fails_with(doc, "children"));
    }
}

void test_node_hierarchy() {
    // The parent moves by (10,0,0) and scales by 2; the child, 1 further
    // along x, is therefore at 10 + 2 * 1 = 12. The grandchild is 1 more
    // in its parent's scaled space
    Doc doc;
    doc.nodes = "{'translation':[10,0,0],'scale':[2,2,2],'children':[1]},"
                "{'mesh':0,'translation':[1,0,0],'children':[2]},"
                "{'mesh':0,'translation':[1,0,0]}";

    const gfx::Model model = parse_ok(doc);

    // The parent has no mesh: two instances, from the child and grandchild
    CHECK(model.instances.size() == 2);
    CHECK(model.meshes.size() == 1);

    if (model.instances.size() == 2) {
        CHECK(approx_vec(gfx::transform_point(model.instances[0].transform,
                                              {0.0f, 0.0f, 0.0f}),
                         {12.0f, 0.0f, 0.0f}));
        CHECK(approx_vec(gfx::transform_point(model.instances[1].transform,
                                              {0.0f, 0.0f, 0.0f}),
                         {14.0f, 0.0f, 0.0f}));
        CHECK(model.instances[0].mesh == model.instances[1].mesh);
    }
}

void test_instancing_and_unused_meshes() {
    // Two nodes using one mesh: one mesh, two instances
    {
        Doc doc;
        doc.nodes = "{'children':[1,2]},"
                    "{'mesh':0,'translation':[1,0,0]},"
                    "{'mesh':0,'translation':[-1,0,0]}";

        const gfx::Model model = parse_ok(doc);
        CHECK(model.meshes.size() == 1);
        CHECK(model.instances.size() == 2);

        if (model.instances.size() == 2) {
            CHECK(model.instances[0].mesh == 0 &&
                  model.instances[1].mesh == 0);
            CHECK(approx(model.instances[0].transform.col[3].x, 1.0));
            CHECK(approx(model.instances[1].transform.col[3].x, -1.0));
        }

        CHECK(gfx::model_triangle_count(model) == 2);
    }

    // A mesh no node in the scene uses is not loaded, and the indices
    // are renumbered: the scene uses glTF mesh 1, which becomes mesh 0
    {
        Bin b;
        b.append(triangle_bytes());   // mesh 0: y = 0
        b.vec3(0.0f, 0.0f, 0.0f);     // mesh 1: positions at x = 7
        b.vec3(7.0f, 0.0f, 0.0f);
        b.vec3(7.0f, 1.0f, 0.0f);

        Doc doc;
        doc.bin = b.data;
        doc.views = "{'buffer':0,'byteLength':72}";
        doc.accessors =
            "{'bufferView':0,'componentType':5126,'count':3,'type':'VEC3'},"
            "{'bufferView':0,'byteOffset':36,'componentType':5126,"
            "'count':3,'type':'VEC3'}";
        doc.meshes = "{'primitives':[{'attributes':{'POSITION':0}}]},"
                     "{'primitives':[{'attributes':{'POSITION':1}}]}";
        doc.nodes = "{'mesh':0},{'mesh':1}";
        doc.scenes = "{'nodes':[1]}";

        const gfx::Model model = parse_ok(doc);
        CHECK(model.meshes.size() == 1);
        CHECK(model.instances.size() == 1);

        if (!model.meshes.empty() && !model.instances.empty()) {
            CHECK(model.instances[0].mesh == 0);
            CHECK(approx(model.meshes[0].positions[1].x, 7.0));
        }
    }

    // A mesh that is broken but unused does not matter either
    {
        Doc doc;
        doc.meshes = "{'primitives':[{'attributes':{'POSITION':0}}]},"
                     "{'primitives':[{'attributes':{'POSITION':99}}]}";
        const gfx::Model model = parse_ok(doc);
        CHECK(model.meshes.size() == 1);
    }
}

void test_scene_selection() {
    Doc doc;
    doc.nodes = "{'mesh':0,'translation':[1,0,0]},"
                "{'mesh':0,'translation':[5,0,0]}";
    doc.scenes = "{'nodes':[0]},{'nodes':[1]}";

    // "scene" picks one
    doc.scene_member = ",'scene':1";
    gfx::Model model = parse_ok(doc);
    CHECK(model.instances.size() == 1);

    if (!model.instances.empty()) {
        CHECK(approx(model.instances[0].transform.col[3].x, 5.0));
    }

    // Without "scene", scene 0
    doc.scene_member = "";
    model = parse_ok(doc);
    CHECK(model.instances.size() == 1);

    if (!model.instances.empty()) {
        CHECK(approx(model.instances[0].transform.col[3].x, 1.0));
    }

    // A scene that does not exist
    doc.scene_member = ",'scene':5";
    CHECK(fails_with(doc, "scene 5 does not exist"));

    doc.scene_member = ",'scene':-1";
    CHECK(fails_with(doc, "scene"));

    // A scene listing a node that does not exist
    doc.scene_member = "";
    doc.scenes = "{'nodes':[7]}";
    CHECK(fails_with(doc, "node that does not exist"));

    // No scenes at all: every node that is nobody's child is a root.
    // Node 1 is a child of node 0 and counts once, through it
    doc = Doc{};
    doc.nodes = "{'mesh':0,'children':[1]},{'mesh':0},{'mesh':0}";
    doc.scenes = "";
    model = parse_ok(doc);
    CHECK(model.instances.size() == 3);

    // "scene" without scenes
    doc.scenes = "";
    doc.scene_member = "";
    doc.extra = ",'scene':0";
    CHECK(fails_with(doc, "scene 0 does not exist"));
}

void test_node_cycles_and_depth() {
    Doc doc;

    doc.nodes = "{'children':[1]},{'mesh':0,'children':[0]}";
    CHECK(fails_with(doc, "cycle"));

    doc.nodes = "{'mesh':0,'children':[0]}";
    CHECK(fails_with(doc, "cycle"));

    doc.nodes = "{'children':[1]},{'children':[2]},{'children':[3]},"
                "{'mesh':0,'children':[1]}";
    CHECK(fails_with(doc, "cycle"));

    // Nesting: node k is the child of node k - 1; the root is at depth
    // 0 and 64 levels below it are allowed
    for (const std::size_t count : {std::size_t{65}, std::size_t{66}}) {
        Doc chain;
        chain.nodes.clear();

        for (std::size_t i = 0; i < count; ++i) {
            if (i != 0) {
                chain.nodes += ",";
            }

            if (i + 1 < count) {
                chain.nodes += "{'children':[" + std::to_string(i + 1) + "]}";
            } else {
                chain.nodes += "{'mesh':0}";
            }
        }

        if (count == 65) {
            gfx::Model model;
            CHECK(parse_doc(chain, model));
            CHECK(model.instances.size() == 1);
        } else {
            CHECK(fails_with(chain, "deeper than 64"));
        }
    }

    // Nodes that two parents share are tolerated, but the number of
    // visits is limited so a chain of diamonds cannot take forever
    {
        Doc diamonds;
        diamonds.nodes.clear();

        for (int i = 0; i < 24; ++i) {
            diamonds.nodes += "{'children':[" + std::to_string(i + 1) + "," +
                              std::to_string(i + 1) + "]},";
        }

        diamonds.nodes += "{'mesh':0}";
        CHECK(fails_with(diamonds, "too many nodes"));

        Doc shared;
        shared.nodes = "{'children':[1,1]},{'mesh':0}";
        gfx::Model model;
        CHECK(parse_doc(shared, model));
        CHECK(model.instances.size() == 2);
    }
}

// ---------------------------------------------------------------------
// Materials
// ---------------------------------------------------------------------

Doc material_doc(const std::string& materials) {
    Doc doc;
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},"
                 "'material':0}]}";
    doc.extra = ",'materials':[" + materials + "]";
    return doc;
}

void test_material_values() {
    {
        gfx::Model model = parse_ok(material_doc(
            "{'name':'gold','pbrMetallicRoughness':{"
            "'baseColorFactor':[1.0,0.766,0.336,0.5],"
            "'metallicFactor':0.25,'roughnessFactor':0.3}}"));

        CHECK(model.meshes.size() == 1);

        if (!model.meshes.empty()) {
            CHECK(model.meshes[0].materials.size() == 1);

            if (!model.meshes[0].materials.empty()) {
                const gfx::Material& m = model.meshes[0].materials[0];
                CHECK(m.name == "gold");
                CHECK(approx_vec(m.base_color, {1.0f, 0.766f, 0.336f}));
                CHECK(approx(m.metallic, 0.25));
                CHECK(approx(m.roughness, 0.3));
                CHECK(approx_vec(m.emission, {0.0f, 0.0f, 0.0f}));
                CHECK(m.base_color_image == -1);
                CHECK(m.base_color_texture.empty());
            }
        }
    }

    // glTF's defaults for an empty material: white, metallic 1, rough 1.
    // An unnamed material is called after its number
    {
        gfx::Model model = parse_ok(material_doc("{}"));

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(m.name == "material 0");
            CHECK(approx_vec(m.base_color, {1.0f, 1.0f, 1.0f}));
            CHECK(approx(m.metallic, 1.0));
            CHECK(approx(m.roughness, 1.0));
            CHECK(approx_vec(m.emission, {0.0f, 0.0f, 0.0f}));
            CHECK(m.metallic_roughness_image == -1);
            CHECK(m.normal_image == -1);
            CHECK(m.emissive_image == -1);
        }
    }

    // Only one of the two factors given: the other keeps its default
    {
        gfx::Model model = parse_ok(material_doc(
            "{'pbrMetallicRoughness':{'metallicFactor':0}}"));

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(approx(m.metallic, 0.0));
            CHECK(approx(m.roughness, 1.0));
        }
    }

    // Factors outside 0..1 are clamped
    {
        gfx::Model model = parse_ok(material_doc(
            "{'pbrMetallicRoughness':{'metallicFactor':2,"
            "'roughnessFactor':-1}}"));

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(approx(m.metallic, 1.0));
            CHECK(approx(m.roughness, 0.0));
        }
    }

    // Emission, plain and with KHR_materials_emissive_strength
    {
        gfx::Model model = parse_ok(material_doc(
            "{'emissiveFactor':[0.5,0.25,0]}"));

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            CHECK(approx_vec(model.meshes[0].materials[0].emission,
                             {0.5f, 0.25f, 0.0f}));
        }

        model = parse_ok(material_doc(
            "{'emissiveFactor':[0.5,0.25,0],'extensions':{"
            "'KHR_materials_emissive_strength':{'emissiveStrength':4}}}"));

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            CHECK(approx_vec(model.meshes[0].materials[0].emission,
                             {2.0f, 1.0f, 0.0f}));
        }

        // Other material extensions are ignored
        model = parse_ok(material_doc(
            "{'extensions':{'KHR_materials_clearcoat':"
            "{'clearcoatFactor':1}}}"));
        CHECK(model.meshes.size() == 1);
    }

    // Malformed values
    CHECK(fails_with(material_doc(
        "{'pbrMetallicRoughness':{'baseColorFactor':[1,1,1]}}"),
        "baseColorFactor"));
    CHECK(fails_with(material_doc(
        "{'pbrMetallicRoughness':{'metallicFactor':'a'}}"),
        "metallicFactor"));
    CHECK(fails_with(material_doc("{'emissiveFactor':[1,1]}"),
                     "emissiveFactor"));
}

void test_material_assignment() {
    // Primitive without material: the "default" material, put after the
    // file's own. Every mesh gets the same list
    Doc doc;
    doc.meshes =
        "{'primitives':[{'attributes':{'POSITION':0},'material':0},"
        "{'attributes':{'POSITION':0}}]},"
        "{'primitives':[{'attributes':{'POSITION':0},'material':1}]}";
    doc.nodes = "{'mesh':0},{'mesh':1}";
    doc.scenes = "{'nodes':[0,1]}";
    doc.extra = ",'materials':[{'name':'a'},{'name':'b'}]";

    const gfx::Model model = parse_ok(doc);
    CHECK(model.meshes.size() == 2);

    if (model.meshes.size() == 2) {
        for (const gfx::Mesh& mesh : model.meshes) {
            CHECK(mesh.materials.size() == 3);
            CHECK(mesh.materials[0].name == "a");
            CHECK(mesh.materials[1].name == "b");
            CHECK(mesh.materials[2].name == "default");
        }

        CHECK(model.meshes[0].triangle_materials ==
              (Indices{0, 2}));
        CHECK(model.meshes[1].triangle_materials == (Indices{1}));
    }

    // No primitive needs the default material: it is not added
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},'material':1}]}";
    doc.nodes = "{'mesh':0}";
    doc.scenes = "{'nodes':[0]}";
    const gfx::Model without_default = parse_ok(doc);

    if (!without_default.meshes.empty()) {
        CHECK(without_default.meshes[0].materials.size() == 2);
        CHECK(without_default.meshes[0].triangle_materials == (Indices{1}));
    }

    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},'material':2}]}";
    CHECK(fails_with(doc, "material"));
}

// ---------------------------------------------------------------------
// Textures and images
// ---------------------------------------------------------------------

const image_test_data::ImageCase& png_case() {
    return image_test_data::kPngCases[0];
}

Bytes png_bytes() {
    const image_test_data::ImageCase& c = png_case();
    return Bytes(c.file, c.file + c.file_size);
}

// A triangle followed by `image` in the buffer, as buffer view 1
Doc image_doc(const Bytes& image, const std::string& images,
              const std::string& textures, const std::string& material) {
    Bin b;
    b.append(triangle_bytes());
    b.append(image);

    Doc doc;
    doc.bin = b.data;
    doc.views = "{'buffer':0,'byteLength':36},"
                "{'buffer':0,'byteOffset':36,'byteLength':" +
                std::to_string(image.size()) + "}";
    doc.meshes = "{'primitives':[{'attributes':{'POSITION':0},"
                 "'material':0}]}";
    doc.extra = ",'images':[" + images + "],'textures':[" + textures +
                "],'materials':[" + material + "]";
    return doc;
}

bool same_pixels(const gfx::Image& image) {
    const image_test_data::ImageCase& c = png_case();

    if (image.width != c.width || image.height != c.height) {
        return false;
    }

    const std::size_t size = static_cast<std::size_t>(c.width) *
                             static_cast<std::size_t>(c.height) * 4;

    return image.pixels.size() == size &&
           std::memcmp(image.pixels.data(), c.rgba, size) == 0;
}

void test_embedded_images() {
    // One image, used by two textures, and by all four material slots:
    // decoded once
    {
        Doc doc = image_doc(
            png_bytes(),
            "{'bufferView':1,'mimeType':'image/png'}",
            "{'source':0},{'source':0}",
            "{'pbrMetallicRoughness':{'baseColorTexture':{'index':0},"
            "'metallicRoughnessTexture':{'index':1}},"
            "'normalTexture':{'index':1},'emissiveTexture':{'index':0}}");

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.size() == 1);

        if (model.images.size() == 1) {
            CHECK(model.images[0].valid());
            CHECK(same_pixels(model.images[0]));
        }

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(m.base_color_image == 0);
            CHECK(m.metallic_roughness_image == 0);
            CHECK(m.normal_image == 0);
            CHECK(m.emissive_image == 0);
            CHECK(m.base_color_texture == "embedded image 0");
        }
    }

    // Two glTF images (the same bytes twice): two decoded images, and
    // each slot finds its own
    {
        Doc doc = image_doc(
            png_bytes(),
            "{'bufferView':1},{'bufferView':1}",
            "{'source':1},{'source':0}",
            "{'pbrMetallicRoughness':{'baseColorTexture':{'index':0},"
            "'metallicRoughnessTexture':{'index':1}}}");

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.size() == 2);

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(m.base_color_image == 0);          // glTF image 1, first used
            CHECK(m.metallic_roughness_image == 1);  // glTF image 0
            CHECK(m.base_color_texture == "embedded image 1");
        }
    }

    // An image no material uses is not decoded at all
    {
        Doc doc = image_doc(png_bytes(), "{'bufferView':1}", "{'source':0}",
                            "{}");
        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.empty());
    }

    // A data URI
    {
        const std::string uri =
            "data:image/png;base64," + base64_encode(png_bytes());

        Doc doc = image_doc(png_bytes(), "{'uri':'" + uri + "'}",
                            "{'source':0}",
                            "{'pbrMetallicRoughness':{'baseColorTexture':"
                            "{'index':0}}}");

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.size() == 1);

        if (model.images.size() == 1) {
            CHECK(same_pixels(model.images[0]));
        }
    }
}

void test_unusable_textures() {
    const std::string base_color =
        "{'pbrMetallicRoughness':{'baseColorTexture':{'index':0}}}";

    // JPEG: a warning, no image, the index stays -1 but the description
    // is kept for the logs
    {
        Bytes jpeg = {0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
                      0x49, 0x46, 0x00, 0x01};
        Doc doc = image_doc(jpeg, "{'bufferView':1,'mimeType':'image/jpeg'}",
                            "{'source':0}", base_color);

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.empty());

        if (!model.meshes.empty() && !model.meshes[0].materials.empty()) {
            const gfx::Material& m = model.meshes[0].materials[0];
            CHECK(m.base_color_image == -1);
            CHECK(m.base_color_texture == "embedded image 0");
        }
    }

    // Garbage, and a PNG cut short
    {
        const Bytes junk(40, 0x55);
        const gfx::Model model = parse_ok(image_doc(
            junk, "{'bufferView':1}", "{'source':0}", base_color));
        CHECK(model.images.empty());
        CHECK(!model.meshes.empty());

        Bytes cut = png_bytes();
        cut.resize(cut.size() / 2);
        const gfx::Model model2 = parse_ok(image_doc(
            cut, "{'bufferView':1}", "{'source':0}", base_color));
        CHECK(model2.images.empty());
        CHECK(!model2.meshes.empty());
        CHECK(model2.meshes[0].materials[0].base_color_image == -1);
    }

    // Texture coordinates other than TEXCOORD_0: texture ignored, and
    // the image is not even decoded
    {
        Doc doc = image_doc(
            png_bytes(), "{'bufferView':1}", "{'source':0}",
            "{'pbrMetallicRoughness':{'baseColorTexture':"
            "{'index':0,'texCoord':1}}}");

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.empty());

        if (!model.meshes.empty()) {
            CHECK(model.meshes[0].materials[0].base_color_image == -1);
            CHECK(model.meshes[0].materials[0].base_color_texture.empty());
        }
    }

    // A texture transform extension is ignored (the texture is used)
    {
        Doc doc = image_doc(
            png_bytes(), "{'bufferView':1}", "{'source':0}",
            "{'pbrMetallicRoughness':{'baseColorTexture':{'index':0,"
            "'extensions':{'KHR_texture_transform':{'scale':[2,2]}}}}}");

        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.size() == 1);
    }

    // External files cannot be read when there is no base folder: the
    // texture is dropped, the model loads
    {
        Doc doc = image_doc(png_bytes(), "{'uri':'textures/some.png'}",
                            "{'source':0}", base_color);
        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.empty());

        if (!model.meshes.empty()) {
            CHECK(model.meshes[0].materials[0].base_color_texture ==
                  "some.png");
        }
    }

    // A texture without a source (supplied only by an extension)
    {
        Doc doc = image_doc(png_bytes(), "{'bufferView':1}", "{}",
                            base_color);
        const gfx::Model model = parse_ok(doc);
        CHECK(model.images.empty());
        CHECK(model.meshes.size() == 1);
    }

    // But references to things that do not exist are errors
    CHECK(fails_with(image_doc(png_bytes(), "{'bufferView':1}",
                               "{'source':0}",
                               "{'pbrMetallicRoughness':{'baseColorTexture':"
                               "{'index':3}}}"),
                     "texture 3 does not exist"));
    CHECK(fails_with(image_doc(png_bytes(), "{'bufferView':1}",
                               "{'source':4}", base_color),
                     "image that does not exist"));
    CHECK(fails_with(image_doc(png_bytes(), "{'bufferView':1}",
                               "{'source':0}",
                               "{'pbrMetallicRoughness':{'baseColorTexture':"
                               "{}}}"),
                     "texture index"));
}

// ---------------------------------------------------------------------
// Extensions
// ---------------------------------------------------------------------

void test_extensions() {
    Doc doc;

    doc.extra = ",'extensionsRequired':['KHR_draco_mesh_compression']";
    CHECK(fails_with(doc,
                     "requires unsupported extension "
                     "KHR_draco_mesh_compression"));

    doc.extra = ",'extensionsRequired':['EXT_meshopt_compression']";
    CHECK(fails_with(doc, "requires unsupported extension "
                          "EXT_meshopt_compression"));

    doc.extra = ",'extensionsRequired':['KHR_texture_basisu']";
    CHECK(fails_with(doc, "KHR_texture_basisu"));

    doc.extra = ",'extensionsRequired':['KHR_materials_emissive_strength',"
                "'KHR_draco_mesh_compression']";
    CHECK(fails_with(doc, "KHR_draco_mesh_compression"));

    // Fine: handled, or harmless to ignore
    for (const char* name : {"KHR_materials_emissive_strength",
                             "KHR_texture_transform",
                             "KHR_materials_clearcoat"}) {
        doc.extra = std::string(",'extensionsRequired':['") + name + "']";
        gfx::Model model;
        CHECK(parse_doc(doc, model));
    }

    // Merely "used" extensions are not a problem, whatever they are
    doc.extra = ",'extensionsUsed':['KHR_draco_mesh_compression']";
    gfx::Model model;
    CHECK(parse_doc(doc, model));

    // Animations and skins are ignored (with a warning)
    doc.extra = ",'animations':[{}],'skins':[{}]";
    CHECK(parse_doc(doc, model));
}

// ---------------------------------------------------------------------
// GLB containers
// ---------------------------------------------------------------------

void test_glb() {
    const Doc doc{};
    const Bytes glb = make_glb(doc);

    gfx::Model model;
    std::string error;

    CHECK(parse_bytes(glb, model, error));
    CHECK(model.meshes.size() == 1 && model.meshes[0].positions.size() == 3);

    // The JSON text can be padded with zero bytes instead of spaces, and
    // bytes after the length in the header are not part of the file
    {
        const Doc d{};
        Doc with_bin = d;
        with_bin.glb_bin = true;

        Bytes g = make_glb_from_chunks({
            {kJsonChunk, padded(text_bytes(with_bin.json()), 0)},
            {kBinChunk, padded(d.bin, 0)}});
        CHECK(parse_bytes(g, model, error));

        Bytes extra = glb;
        extra.insert(extra.end(), 16, 0x77);
        CHECK(parse_bytes(extra, model, error));
    }

    // Chunks of other types are skipped
    {
        Doc d;
        d.glb_bin = true;

        const Bytes g = make_glb_from_chunks({
            {kJsonChunk, padded(text_bytes(d.json()), ' ')},
            {0x12345678u, Bytes(8, 1)},
            {kBinChunk, padded(d.bin, 0)}});
        CHECK(parse_bytes(g, model, error));
        CHECK(model.meshes.size() == 1);
    }

    // The header
    {
        Bytes g = glb;
        g.resize(11);
        CHECK(!parse_bytes(g, model, error) && has(error, "header"));

        g = glb;
        put_u32(g, 4, 1);
        CHECK(!parse_bytes(g, model, error) && has(error, "version 1"));

        g = glb;
        put_u32(g, 4, 3);
        CHECK(!parse_bytes(g, model, error) && has(error, "version 3"));

        g = glb;
        put_u32(g, 8, static_cast<std::uint32_t>(g.size() + 1));
        CHECK(!parse_bytes(g, model, error) &&
              has(error, "header says"));

        // Truncated file
        g = glb;
        g.resize(g.size() - 5);
        CHECK(!parse_bytes(g, model, error) && has(error, "header says"));

        g = glb;
        put_u32(g, 8, 12);
        CHECK(!parse_bytes(g, model, error) && has(error, "no chunks"));
    }

    // The chunks
    {
        Bytes g = glb;
        const std::uint32_t json_length =
            static_cast<std::uint32_t>(g[12]) |
            (static_cast<std::uint32_t>(g[13]) << 8) |
            (static_cast<std::uint32_t>(g[14]) << 16) |
            (static_cast<std::uint32_t>(g[15]) << 24);

        // A chunk longer than the file
        put_u32(g, 12, json_length + 1000);
        CHECK(!parse_bytes(g, model, error) &&
              has(error, "runs past the end"));

        // A length that is not a multiple of 4
        g = glb;
        put_u32(g, 12, json_length - 1);
        CHECK(!parse_bytes(g, model, error) &&
              has(error, "multiple of 4"));

        // The first chunk must be JSON
        g = glb;
        put_u32(g, 16, kBinChunk);
        CHECK(!parse_bytes(g, model, error) && has(error, "not JSON"));

        // Cut off in the middle of the second chunk header
        g = glb;
        put_u32(g, 8, 12 + 8 + json_length + 3);
        CHECK(!parse_bytes(g, model, error) &&
              has(error, "truncated chunk header"));

        // The BIN chunk claims more than the file has
        g = glb;
        put_u32(g, 20 + json_length, 100000);
        CHECK(!parse_bytes(g, model, error) &&
              has(error, "runs past the end"));
    }

    // Buffer 0 without uri, but no BIN chunk
    {
        Doc d;
        d.glb_bin = true;

        const Bytes g = make_glb_from_chunks({
            {kJsonChunk, padded(text_bytes(d.json()), ' ')}});
        CHECK(!parse_bytes(g, model, error) && has(error, "BIN"));
    }

    // Garbage JSON inside a GLB
    {
        const Bytes g = make_glb_from_chunks({
            {kJsonChunk, padded(text_bytes("{\"asset\": oops}"), ' ')}});
        CHECK(!parse_bytes(g, model, error) && has(error, "JSON: line 1"));
    }

    // A .glb can also use a data: URI buffer (other buffers than the
    // first have to)
    {
        const Doc d{};
        const Bytes g = make_glb_from_chunks({
            {kJsonChunk, padded(text_bytes(d.json()), ' ')}});
        CHECK(parse_bytes(g, model, error));
    }
}

// ---------------------------------------------------------------------
// Files: load_gltf, load_model, assets
// ---------------------------------------------------------------------

void write_file(const std::filesystem::path& path, const Bytes& data) {
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
}

void write_text(const std::filesystem::path& path, const std::string& text) {
    write_file(path, text_bytes(text));
}

void test_external_files() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gfxlab_test_gltf";
    fs::remove_all(dir);
    fs::create_directories(dir / "sub");

    // The buffer file name has a space and a non-ASCII letter (UTF-8
    // bytes for e with an acute accent); the URI percent-encodes both
    const std::string accent = "\xC3\xA9";
    const std::string file_name = "buf fer " + accent + ".bin";

    Doc doc;
    doc.buffer_uri = "sub/buf%20fer%20%C3%A9.bin";

    write_file(dir / "sub" / fs::u8path(file_name), doc.bin);
    write_text(dir / "model.gltf", doc.json());

    gfx::Model model;
    std::string error;

    CHECK(gfx::load_gltf((dir / "model.gltf").string(), model, error));
    CHECK(model.meshes.size() == 1 && model.meshes[0].positions.size() == 3);

    // load_model adds what glTF does not give: flat normals and edges
    CHECK(gfx::load_model((dir / "model.gltf").string(), model, error));
    CHECK(model.meshes.size() == 1);

    if (!model.meshes.empty()) {
        const gfx::Mesh& mesh = model.meshes[0];
        CHECK(mesh.normals.size() == mesh.positions.size());
        CHECK(mesh.edges.size() == 6);

        // The triangle faces +Z in Y-up, so -Y in Z-up
        if (!mesh.normals.empty()) {
            CHECK(approx_vec(mesh.normals[0], {0.0f, -1.0f, 0.0f}));
        }
    }

    // parse_gltf with a base folder, and with none: the file cannot be
    // reached then
    {
        const Bytes data = text_bytes(doc.json());
        CHECK(parse_bytes(data, model, error, (dir).u8string()));
        CHECK(!parse_bytes(data, model, error, "") &&
              has(error, "external file"));
    }

    // A bare file name is relative to the current folder
    {
        const fs::path old = fs::current_path();
        fs::current_path(dir);

        const bool ok = gfx::load_gltf("model.gltf", model, error);

        fs::current_path(old);
        CHECK(ok);
    }

    // A missing buffer file, and a missing model file
    {
        Doc broken;
        broken.buffer_uri = "nothing_here.bin";
        write_text(dir / "broken.gltf", broken.json());

        CHECK(!gfx::load_gltf((dir / "broken.gltf").string(), model, error));
        CHECK(has(error, "buffer 0") && has(error, "nothing_here.bin"));

        CHECK(!gfx::load_gltf((dir / "absent.gltf").string(), model, error));
        CHECK(!gfx::load_model((dir / "absent.glb").string(), model, error));
        CHECK(!gfx::load_model((dir / "absent.obj").string(), model, error));
    }

    // Remote and malformed URIs
    {
        Doc d;
        d.buffer_uri = "https://example.com/data.bin";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "remote"));

        d.buffer_uri = "data.bin%zz";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "%"));

        d.buffer_uri = "data.bin%4";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "%"));

        d.buffer_uri = "data:application/octet-stream,AAAA";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "base64"));

        d.buffer_uri = "data:application/octet-stream;base64,AA*A";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "base64"));

        d.buffer_uri = "data:nocomma";
        CHECK(!parse_bytes(text_bytes(d.json()), model, error,
                           dir.u8string()) && has(error, "comma"));
    }

    // A .glb loaded from a file
    {
        write_file(dir / "model.glb", make_glb(Doc{}));
        CHECK(gfx::load_model((dir / "model.glb").string(), model, error));
        CHECK(model.instances.size() == 1);
    }

    fs::remove_all(dir);
}

void test_load_model_obj() {
    namespace fs = std::filesystem;

    // The textured cube: one mesh, one instance, both images decoded
    gfx::Model model;
    std::string error;

    CHECK(gfx::load_model(kAssets + "/textured_cube.obj", model, error));
    CHECK(model.meshes.size() == 1);
    CHECK(model.instances.size() == 1);
    CHECK(model.images.size() == 2);

    if (model.meshes.size() == 1 && model.images.size() == 2) {
        const gfx::Mesh& mesh = model.meshes[0];
        CHECK(mesh.has_normals());
        CHECK(mesh.has_texcoords());
        CHECK(!mesh.edges.empty());
        CHECK(mesh.materials.size() == 2);

        // Each material knows its image
        CHECK(mesh.materials[0].name == "grid");
        CHECK(mesh.materials[0].base_color_image == 0);
        CHECK(mesh.materials[1].name == "label");
        CHECK(mesh.materials[1].base_color_image == 1);
        CHECK(model.images[0].valid() && model.images[1].valid());

        // Centred already (the cube spans -1..1), and its triangles
        CHECK(gfx::model_triangle_count(model) == 12);

        const gfx::Aabb b = gfx::model_bounds(model);
        CHECK(approx_vec(b.min, {-1.0f, -1.0f, -1.0f}));
        CHECK(approx_vec(b.max, {1.0f, 1.0f, 1.0f}));
    }

    // A model without textures
    CHECK(gfx::load_model(kAssets + "/cube.obj", model, error));
    CHECK(model.images.empty() && model.meshes.size() == 1);

    // An off-centre OBJ (Y-up): centring moves the instance, not the
    // vertices
    const fs::path dir = fs::temp_directory_path() / "gfxlab_test_gltf_obj";
    fs::remove_all(dir);
    fs::create_directories(dir);

    const std::string obj = "v 0 0 0\nv 2 0 0\nv 0 4 0\nf 1 2 3\n";
    write_text(dir / "off.obj", obj);
    write_text(dir / "UPPER.OBJ", obj);

    for (const char* name : {"off.obj", "UPPER.OBJ"}) {
        CHECK(gfx::load_model((dir / name).string(), model, error));
        CHECK(model.meshes.size() == 1 && model.instances.size() == 1);

        if (model.meshes.size() == 1 && model.instances.size() == 1) {
            const gfx::Mesh& mesh = model.meshes[0];

            // Z-up: (x, y, z) -> (x, -z, y), nothing else
            CHECK(mesh.positions.size() == 3);
            CHECK(approx_vec(mesh.positions[1], {2.0f, 0.0f, 0.0f}));
            CHECK(approx_vec(mesh.positions[2], {0.0f, 0.0f, 4.0f}));

            // Their box is (0,0,0)-(2,0,4): the centre (1,0,2) is
            // subtracted by the instance
            const gfx::Mat4& t = model.instances[0].transform;
            CHECK(approx_vec(gfx::transform_point(t, {0.0f, 0.0f, 0.0f}),
                             {-1.0f, 0.0f, -2.0f}));

            const gfx::Aabb b = gfx::model_bounds(model);
            CHECK(approx_vec(b.min, {-1.0f, 0.0f, -2.0f}));
            CHECK(approx_vec(b.max, {1.0f, 0.0f, 2.0f}));

            // Normals were generated, and the edge list built
            CHECK(mesh.has_normals());
            CHECK(mesh.edges.size() == 6);
        }
    }

    // Unsupported formats and missing files
    write_text(dir / "thing.stl", "solid x");
    CHECK(!gfx::load_model((dir / "thing.stl").string(), model, error));
    CHECK(has(error, "unsupported model format"));
    CHECK(model.meshes.empty());

    CHECK(!gfx::load_model("", model, error));
    CHECK(!gfx::load_model((dir / "missing.obj").string(), model, error));

    fs::remove_all(dir);
}

// Every model the loader returns must be safe to hand to a renderer
bool model_is_consistent(const gfx::Model& model) {
    for (const gfx::ModelInstance& instance : model.instances) {
        if (instance.mesh >= model.meshes.size()) {
            return false;
        }
    }

    for (const gfx::Mesh& mesh : model.meshes) {
        if (mesh.indices.size() % 3 != 0 ||
            mesh.triangle_materials.size() != mesh.triangle_count() ||
            mesh.materials.empty()) {
            return false;
        }

        if (!mesh.normals.empty() &&
            mesh.normals.size() != mesh.positions.size()) {
            return false;
        }

        if (!mesh.texcoords.empty() &&
            mesh.texcoords.size() != mesh.positions.size()) {
            return false;
        }

        for (const std::uint32_t index : mesh.indices) {
            if (index >= mesh.positions.size()) {
                return false;
            }
        }

        for (const std::uint32_t material : mesh.triangle_materials) {
            if (material >= mesh.materials.size()) {
                return false;
            }
        }

        for (const gfx::Material& m : mesh.materials) {
            const int slots[] = {m.base_color_image,
                                 m.metallic_roughness_image, m.normal_image,
                                 m.emissive_image};

            for (const int slot : slots) {
                if (slot < -1 ||
                    slot >= static_cast<int>(model.images.size())) {
                    return false;
                }
            }
        }
    }

    for (const gfx::Image& image : model.images) {
        if (!image.valid()) {
            return false;
        }
    }

    return true;
}

bool same_mesh(const gfx::Mesh& a, const gfx::Mesh& b) {
    if (a.positions.size() != b.positions.size() ||
        a.normals.size() != b.normals.size() ||
        a.texcoords.size() != b.texcoords.size() ||
        a.indices != b.indices || a.edges != b.edges ||
        a.triangle_materials != b.triangle_materials ||
        a.materials.size() != b.materials.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.positions.size(); ++i) {
        if (!approx_vec(a.positions[i], b.positions[i], 1e-6)) {
            return false;
        }
    }

    for (std::size_t i = 0; i < a.normals.size(); ++i) {
        if (!approx_vec(a.normals[i], b.normals[i], 1e-6)) {
            return false;
        }
    }

    for (std::size_t i = 0; i < a.texcoords.size(); ++i) {
        if (!approx(a.texcoords[i].x, b.texcoords[i].x, 1e-6) ||
            !approx(a.texcoords[i].y, b.texcoords[i].y, 1e-6)) {
            return false;
        }
    }

    // The texture description differs between the two files ("embedded
    // image 0" or the file name), everything else must match
    for (std::size_t i = 0; i < a.materials.size(); ++i) {
        const gfx::Material& x = a.materials[i];
        const gfx::Material& y = b.materials[i];

        if (x.name != y.name || !approx_vec(x.base_color, y.base_color) ||
            !approx(x.roughness, y.roughness) ||
            !approx(x.metallic, y.metallic) ||
            !approx_vec(x.emission, y.emission) ||
            x.base_color_image != y.base_color_image ||
            x.metallic_roughness_image != y.metallic_roughness_image ||
            x.normal_image != y.normal_image ||
            x.emissive_image != y.emissive_image) {
            return false;
        }
    }

    return true;
}

bool same_model(const gfx::Model& a, const gfx::Model& b) {
    if (a.meshes.size() != b.meshes.size() ||
        a.images.size() != b.images.size() ||
        a.instances.size() != b.instances.size()) {
        return false;
    }

    for (std::size_t i = 0; i < a.meshes.size(); ++i) {
        if (!same_mesh(a.meshes[i], b.meshes[i])) {
            return false;
        }
    }

    for (std::size_t i = 0; i < a.images.size(); ++i) {
        if (a.images[i].width != b.images[i].width ||
            a.images[i].height != b.images[i].height ||
            a.images[i].pixels != b.images[i].pixels) {
            return false;
        }
    }

    for (std::size_t i = 0; i < a.instances.size(); ++i) {
        if (a.instances[i].mesh != b.instances[i].mesh) {
            return false;
        }

        for (int c = 0; c < 4; ++c) {
            const gfx::Vec4& u = a.instances[i].transform.col[c];
            const gfx::Vec4& v = b.instances[i].transform.col[c];

            if (!approx(u.x, v.x, 1e-6) || !approx(u.y, v.y, 1e-6) ||
                !approx(u.z, v.z, 1e-6) || !approx(u.w, v.w, 1e-6)) {
                return false;
            }
        }
    }

    return true;
}

float column_length(const gfx::Mat4& m, int c) {
    return gfx::length({m.col[c].x, m.col[c].y, m.col[c].z});
}

void test_cubes_assets() {
    gfx::Model glb;
    gfx::Model gltf;
    std::string error;

    CHECK(gfx::load_model(kAssets + "/gltf/cubes.glb", glb, error));
    CHECK(gfx::load_model(kAssets + "/gltf/cubes.gltf", gltf, error));

    // The two files describe the same scene
    CHECK(same_model(glb, gltf));
    CHECK(model_is_consistent(glb));
    CHECK(model_is_consistent(gltf));

    // Two meshes (the cube, shared, and the pyramid), three instances
    CHECK(glb.meshes.size() == 2);
    CHECK(glb.instances.size() == 3);
    CHECK(glb.images.size() == 1);

    if (glb.meshes.size() != 2 || glb.instances.size() != 3) {
        return;
    }

    CHECK(glb.instances[0].mesh == 0);
    CHECK(glb.instances[1].mesh == 0);
    CHECK(glb.instances[2].mesh == 1);

    // 12 triangles twice and 6 once
    CHECK(gfx::model_triangle_count(glb) == 30);

    // The cube: 24 vertices shared by both primitives, normals and
    // texture coordinates, 10 triangles of "grid" and 2 of "gold"
    const gfx::Mesh& cube = glb.meshes[0];
    CHECK(cube.positions.size() == 24);
    CHECK(cube.has_normals() && cube.has_texcoords());
    CHECK(cube.triangle_count() == 12);
    CHECK(cube.materials.size() == 3);

    std::size_t grid = 0;
    std::size_t gold = 0;

    for (const std::uint32_t m : cube.triangle_materials) {
        grid += m == 0 ? 1 : 0;
        gold += m == 1 ? 1 : 0;
    }

    CHECK(grid == 10 && gold == 2);

    if (cube.materials.size() == 3) {
        CHECK(cube.materials[0].name == "grid");
        CHECK(cube.materials[0].base_color_image == 0);
        CHECK(approx(cube.materials[0].roughness, 0.5));
        CHECK(approx(cube.materials[0].metallic, 0.0));
        CHECK(cube.materials[1].name == "gold");
        CHECK(approx_vec(cube.materials[1].base_color,
                         {1.0f, 0.766f, 0.336f}));
        CHECK(approx(cube.materials[1].metallic, 1.0));
        CHECK(approx(cube.materials[1].roughness, 0.3));
        CHECK(cube.materials[2].name == "glow");
        CHECK(approx_vec(cube.materials[2].emission, {4.0f, 1.6f, 0.4f}));
    }

    CHECK(glb.images[0].width > 0 && glb.images[0].valid());

    // Its faces are those of textured_cube.obj: the same vertices in the
    // same order, with the same normals and texture coordinates (the
    // OBJ has them with v up, the glTF stores 1 - v and the loader
    // flips back)
    gfx::Model obj;
    CHECK(gfx::load_model(kAssets + "/textured_cube.obj", obj, error));

    if (obj.meshes.size() == 1) {
        const gfx::Mesh& reference = obj.meshes[0];
        CHECK(reference.positions.size() == 24);

        if (reference.positions.size() == 24) {
            for (std::size_t i = 0; i < 24; ++i) {
                CHECK(approx_vec(cube.positions[i], reference.positions[i]));
                CHECK(approx_vec(cube.normals[i], reference.normals[i]));
                CHECK(approx(cube.texcoords[i].x, reference.texcoords[i].x));
                CHECK(approx(cube.texcoords[i].y, reference.texcoords[i].y));
            }
        }
    }

    // The pyramid: 18 vertices, no texture coordinates, and flat normals
    // that load_model made. Its first triangle is the +Z side, tilted
    // up: Y-up normal (0, 0.447, 0.894) is (0, -0.894, 0.447) in Z-up
    const gfx::Mesh& pyramid = glb.meshes[1];
    CHECK(pyramid.positions.size() == 18);
    CHECK(pyramid.triangle_count() == 6);
    CHECK(!pyramid.has_texcoords());
    CHECK(pyramid.has_normals());

    if (pyramid.normals.size() == 18) {
        CHECK(approx_vec(pyramid.normals[0], {0.0f, -0.894427f, 0.447214f},
                         1e-4));

        for (const gfx::Vec3& n : pyramid.normals) {
            CHECK(approx(gfx::length(n), 1.0, 1e-4));
        }
    }

    // Instances: cube_right has scale 0.5, cube_left (a matrix) 0.75
    // and the pyramid 1; rotations and translations do not change the
    // length of the columns
    CHECK(approx(column_length(glb.instances[0].transform, 0), 0.5, 1e-5));
    CHECK(approx(column_length(glb.instances[1].transform, 0), 0.75, 1e-5));
    CHECK(approx(column_length(glb.instances[2].transform, 0), 1.0, 1e-5));

    // load_model centred the whole thing
    const gfx::Aabb bounds = gfx::model_bounds(glb);
    CHECK(approx_vec((bounds.min + bounds.max) * 0.5f, {0.0f, 0.0f, 0.0f},
                     1e-5));
    CHECK(bounds.max.x - bounds.min.x > 3.0f);
}

void test_gltf_loader_matches_hand_math() {
    // cubes.glb through parse_gltf, not centred: the instances as the
    // file describes them. cube_right is the cube scaled by 0.5, moved
    // 1.5 along x in the root's frame; the root is lifted 0.5 and
    // turned 30 degrees about Y. In Y-up the cube's centre ends at
    //   root * (1.5, 0, 0) = (1.5 cos 30, 0.5, -1.5 sin 30)
    // which is (1.299, 0.5, -0.75), and (1.299, 0.75, 0.5) in Z-up
    std::ifstream file(kAssets + "/gltf/cubes.glb", std::ios::binary);
    const Bytes bytes((std::istreambuf_iterator<char>(file)),
                      std::istreambuf_iterator<char>());
    CHECK(!bytes.empty());

    gfx::Model model;
    std::string error;
    CHECK(parse_bytes(bytes, model, error));

    if (model.instances.size() != 3) {
        CHECK_FAILED();
        return;
    }

    const gfx::Mat4& t = model.instances[0].transform;
    const gfx::Vec3 centre = gfx::transform_point(t, {0.0f, 0.0f, 0.0f});

    const double c30 = std::cos(30.0 * 3.14159265358979323846 / 180.0);
    const double s30 = std::sin(30.0 * 3.14159265358979323846 / 180.0);

    CHECK(approx(centre.x, 1.5 * c30, 1e-5));
    CHECK(approx(centre.y, 1.5 * s30, 1e-5));
    CHECK(approx(centre.z, 0.5, 1e-5));
}

// ---------------------------------------------------------------------
// Model helpers and normal_matrix_of
// ---------------------------------------------------------------------

// A mesh whose vertices are the 8 corners of a box, with 12 triangles
gfx::Mesh box_mesh(gfx::Vec3 lo, gfx::Vec3 hi) {
    gfx::Mesh mesh;

    for (int corner = 0; corner < 8; ++corner) {
        mesh.positions.push_back({(corner & 1) ? hi.x : lo.x,
                                  (corner & 2) ? hi.y : lo.y,
                                  (corner & 4) ? hi.z : lo.z});
    }

    for (std::uint32_t t = 0; t < 12; ++t) {
        mesh.indices.push_back(t % 8);
        mesh.indices.push_back((t + 1) % 8);
        mesh.indices.push_back((t + 2) % 8);
        mesh.triangle_materials.push_back(0);
    }

    return mesh;
}

void test_model_bounds_and_centering() {
    gfx::Model model;

    // Empty
    {
        const gfx::Aabb b = gfx::model_bounds(model);
        CHECK(approx_vec(b.min, {0.0f, 0.0f, 0.0f}));
        CHECK(approx_vec(b.max, {0.0f, 0.0f, 0.0f}));
        CHECK(gfx::model_triangle_count(model) == 0);
        gfx::center_model_at_origin(model);   // does nothing, no crash
    }

    // One box [-1,1]^3 placed twice, at x = 4 and x = -2
    model.meshes.push_back(box_mesh({-1.0f, -1.0f, -1.0f},
                                    {1.0f, 1.0f, 1.0f}));

    gfx::ModelInstance a;
    a.mesh = 0;
    a.transform = gfx::mat4_translate({4.0f, 0.0f, 0.0f});
    gfx::ModelInstance b;
    b.mesh = 0;
    b.transform = gfx::mat4_translate({-2.0f, 0.0f, 0.0f});
    model.instances = {a, b};

    {
        const gfx::Aabb box = gfx::model_bounds(model);
        CHECK(approx_vec(box.min, {-3.0f, -1.0f, -1.0f}));
        CHECK(approx_vec(box.max, {5.0f, 1.0f, 1.0f}));
        CHECK(gfx::model_triangle_count(model) == 24);
    }

    gfx::center_model_at_origin(model);

    {
        const gfx::Aabb box = gfx::model_bounds(model);
        CHECK(approx_vec(box.min, {-4.0f, -1.0f, -1.0f}));
        CHECK(approx_vec(box.max, {4.0f, 1.0f, 1.0f}));

        // The vertices did not move
        CHECK(approx_vec(model.meshes[0].positions[0],
                         {-1.0f, -1.0f, -1.0f}));
        CHECK(approx(model.instances[0].transform.col[3].x, 3.0));
        CHECK(approx(model.instances[1].transform.col[3].x, -3.0));
    }

    // A rotated instance: the box turned 45 degrees about Z has a box of
    // +-sqrt(2) in x and y
    {
        gfx::Model rotated;
        rotated.meshes.push_back(box_mesh({-1.0f, -1.0f, -1.0f},
                                          {1.0f, 1.0f, 1.0f}));

        gfx::ModelInstance r;
        r.mesh = 0;
        r.transform = gfx::mat4_rotate({0.0f, 0.0f, 1.0f},
                                       0.78539816339744831f);
        rotated.instances = {r};

        const gfx::Aabb box = gfx::model_bounds(rotated);
        const float root2 = std::sqrt(2.0f);
        CHECK(approx_vec(box.min, {-root2, -root2, -1.0f}, 1e-5));
        CHECK(approx_vec(box.max, {root2, root2, 1.0f}, 1e-5));
    }

    // An instance of a mesh that does not exist is skipped
    {
        gfx::ModelInstance bad;
        bad.mesh = 9;
        model.instances.push_back(bad);
        CHECK(gfx::model_triangle_count(model) == 24);

        const gfx::Aabb box = gfx::model_bounds(model);
        CHECK(approx_vec(box.min, {-4.0f, -1.0f, -1.0f}));
    }
}

void test_normal_matrix_of() {
    // Identity and rotation: the matrix itself
    {
        const gfx::Mat4 n = gfx::normal_matrix_of(gfx::mat4_identity());
        CHECK(approx_vec(gfx::transform_direction(n, {1.0f, 2.0f, 3.0f}),
                         {1.0f, 2.0f, 3.0f}));

        const gfx::Mat4 r = gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, 0.5f);
        const gfx::Mat4 nr = gfx::normal_matrix_of(r);
        CHECK(approx_vec(gfx::transform_direction(nr, {1.0f, 0.0f, 0.0f}),
                         gfx::transform_direction(r, {1.0f, 0.0f, 0.0f})));
    }

    // Non-uniform scale: the normals use the inverse scale
    {
        const gfx::Mat4 s = gfx::mat4_scale({1.0f, 2.0f, 4.0f});
        const gfx::Mat4 n = gfx::normal_matrix_of(s);
        CHECK(approx_vec(gfx::transform_direction(n, {1.0f, 1.0f, 1.0f}),
                         {1.0f, 0.5f, 0.25f}));

        // The point of it: a normal stays perpendicular to the surface.
        // The plane with normal (1,1,0) contains the direction (1,-1,0)
        const gfx::Vec3 tangent =
            gfx::transform_direction(s, {1.0f, -1.0f, 0.0f});
        const gfx::Vec3 normal =
            gfx::transform_direction(n, {1.0f, 1.0f, 0.0f});
        CHECK(approx(gfx::dot(tangent, normal), 0.0));

        // Transforming the normal like a position would not
        const gfx::Vec3 wrong =
            gfx::transform_direction(s, {1.0f, 1.0f, 0.0f});
        CHECK(std::fabs(gfx::dot(tangent, wrong)) > 1.0);
    }

    // Rotation and scale together, with a translation that must not matter
    {
        gfx::Transform t;
        t.position = {5.0f, -3.0f, 2.0f};
        t.rotation = {0.3f, -0.8f, 1.1f};
        t.scale = {2.0f, 0.5f, 3.0f};

        const gfx::Mat4 mine = gfx::normal_matrix_of(t.model_matrix());
        const gfx::Mat4 reference = t.normal_matrix();

        for (int c = 0; c < 3; ++c) {
            CHECK(approx(mine.col[c].x, reference.col[c].x, 1e-5));
            CHECK(approx(mine.col[c].y, reference.col[c].y, 1e-5));
            CHECK(approx(mine.col[c].z, reference.col[c].z, 1e-5));
            CHECK(approx(mine.col[c].w, 0.0));
        }

        CHECK(approx(mine.col[3].x, 0.0) && approx(mine.col[3].y, 0.0) &&
              approx(mine.col[3].z, 0.0) && approx(mine.col[3].w, 1.0));
    }

    // Negative scale (a mirror)
    {
        const gfx::Mat4 n =
            gfx::normal_matrix_of(gfx::mat4_scale({-1.0f, 1.0f, 1.0f}));
        CHECK(approx_vec(gfx::transform_direction(n, {1.0f, 1.0f, 1.0f}),
                         {-1.0f, 1.0f, 1.0f}));
    }

    // Singular: the identity instead of NaN
    {
        const gfx::Mat4 n =
            gfx::normal_matrix_of(gfx::mat4_scale({0.0f, 1.0f, 1.0f}));
        CHECK(approx_vec(gfx::transform_direction(n, {1.0f, 2.0f, 3.0f}),
                         {1.0f, 2.0f, 3.0f}));
    }

    // Transpose
    {
        gfx::Mat4 m = gfx::mat4_identity();
        m.col[0] = {1.0f, 2.0f, 3.0f, 4.0f};
        m.col[1] = {5.0f, 6.0f, 7.0f, 8.0f};
        m.col[2] = {9.0f, 10.0f, 11.0f, 12.0f};
        m.col[3] = {13.0f, 14.0f, 15.0f, 16.0f};

        const gfx::Mat4 t = gfx::mat4_transpose(m);
        CHECK(t.col[0].x == 1.0f && t.col[0].y == 5.0f &&
              t.col[0].z == 9.0f && t.col[0].w == 13.0f);
        CHECK(t.col[3].x == 4.0f && t.col[3].y == 8.0f &&
              t.col[3].z == 12.0f && t.col[3].w == 16.0f);

        const gfx::Mat4 back = gfx::mat4_transpose(t);
        CHECK(back.col[1].z == 7.0f && back.col[2].y == 10.0f);
    }
}

// ---------------------------------------------------------------------
// Robustness: damaged files never crash and never give a broken model
// ---------------------------------------------------------------------

void test_damaged_files() {
    std::ifstream file(kAssets + "/gltf/cubes.glb", std::ios::binary);
    const Bytes original((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
    CHECK(original.size() > 1000);

    gfx::Model model;
    std::string error;

    // Cut off anywhere: always an error (the header knows the size)
    for (std::size_t length = 0; length < original.size(); length += 61) {
        const Bytes cut(original.begin(),
                        original.begin() + static_cast<std::ptrdiff_t>(length));

        if (parse_bytes(cut, model, error)) {
            std::fprintf(stderr, "  truncated file (%zu bytes) loaded\n",
                         length);
            CHECK_FAILED();
        }
    }

    // Random bytes changed. Most of the file is image data, so half of
    // the damage goes to the first 3000 bytes (the JSON). The result
    // may be an error or a model, but a model must be consistent
    std::uint32_t state = 12345;

    const auto next = [&state]() {
        state = state * 1664525u + 1013904223u;
        return state >> 8;
    };

    std::size_t loaded = 0;

    for (int trial = 0; trial < 1500; ++trial) {
        Bytes damaged = original;
        const std::uint32_t changes = 1 + next() % 4;

        for (std::uint32_t i = 0; i < changes; ++i) {
            const std::size_t limit =
                (trial % 2 == 0) ? std::size_t{3000} : damaged.size();
            const std::size_t at = next() % std::min(limit, damaged.size());
            damaged[at] = static_cast<std::uint8_t>(next());
        }

        if (parse_bytes(damaged, model, error)) {
            ++loaded;

            if (!model_is_consistent(model)) {
                std::fprintf(stderr, "  damaged file (trial %d) gave an "
                             "inconsistent model\n", trial);
                CHECK_FAILED();
            }
        }
    }

    // Some damage is harmless (a changed name, image data...)
    CHECK(loaded > 0);

    // The same for the text form, cut at every 7th byte
    const std::string json = Doc{}.json();

    for (std::size_t length = 0; length < json.size(); length += 7) {
        if (parse_text(json.substr(0, length), model, error)) {
            CHECK_FAILED();
        }
    }
}

} // namespace

int main() {
    // Many tests feed broken files on purpose: keep the warnings out
    // of the test output
    gfx::log_set_level(gfx::LogLevel::LOG_LEVEL_ERROR);

    test_base64();
    test_basic_model();
    test_version_and_json_errors();
    test_y_up_to_z_up();
    test_accessor_offset_and_stride();
    test_texcoords();
    test_accessor_without_buffer_view();
    test_accessor_errors();
    test_sparse_accessors();
    test_index_types();
    test_index_errors();
    test_primitive_modes();
    test_primitives_in_one_mesh();
    test_node_transforms();
    test_node_hierarchy();
    test_instancing_and_unused_meshes();
    test_scene_selection();
    test_node_cycles_and_depth();
    test_material_values();
    test_material_assignment();
    test_embedded_images();
    test_unusable_textures();
    test_extensions();
    test_glb();
    test_external_files();
    test_load_model_obj();
    test_cubes_assets();
    test_gltf_loader_matches_hand_math();
    test_model_bounds_and_centering();
    test_normal_matrix_of();
    test_damaged_files();

    if (g_failures == 0) {
        std::printf("All glTF tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d glTF test check(s) failed\n", g_failures);
    return 1;
}
