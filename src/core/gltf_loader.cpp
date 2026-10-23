#include "core/gltf_loader.h"

#include "core/base64.h"
#include "core/image.h"
#include "core/json.h"
#include "core/log.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

namespace gfx {

namespace {

// ---------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------

constexpr std::uint32_t kGlbMagic = 0x46546C67u;   // "glTF"
constexpr std::uint32_t kGlbVersion = 2;
constexpr std::uint32_t kChunkJson = 0x4E4F534Au;  // "JSON"
constexpr std::uint32_t kChunkBin = 0x004E4942u;   // "BIN"

// glTF component types (accessor.componentType)
constexpr std::uint64_t kByte = 5120;
constexpr std::uint64_t kUnsignedByte = 5121;
constexpr std::uint64_t kShort = 5122;
constexpr std::uint64_t kUnsignedShort = 5123;
constexpr std::uint64_t kUnsignedInt = 5125;
constexpr std::uint64_t kFloat = 5126;

// Primitive modes
constexpr std::uint64_t kModeTriangles = 4;
constexpr std::uint64_t kModeTriangleStrip = 5;
constexpr std::uint64_t kModeTriangleFan = 6;

// Limits that keep a hostile file from using absurd amounts of time or
// memory. Real accessors are bounded by the size of their buffer; these
// cover what a buffer cannot (accessors without data, a node tree whose
// nodes are shared, many primitives reading the same big accessor)
constexpr int kMaxNodeDepth = 64;
constexpr std::size_t kMaxNodeVisits = std::size_t{1} << 18;
constexpr std::uint64_t kMaxZeroAccessorCount = std::uint64_t{1} << 22;
constexpr std::uint64_t kMaxTotalVertices = std::uint64_t{1} << 25;
constexpr std::uint64_t kMaxTotalIndices = std::uint64_t{1} << 27;

const std::uint8_t kPngSignature[8] = {
    0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A
};

// ---------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------

// Little-endian reads. glTF is always little-endian; assembling the
// value from bytes works on any machine and avoids unaligned loads
std::uint32_t load_u32(const std::uint8_t* p) {
    return static_cast<std::uint32_t>(p[0])
         | (static_cast<std::uint32_t>(p[1]) << 8)
         | (static_cast<std::uint32_t>(p[2]) << 16)
         | (static_cast<std::uint32_t>(p[3]) << 24);
}

std::uint16_t load_u16(const std::uint8_t* p) {
    return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

float load_f32(const std::uint8_t* p) {
    const std::uint32_t bits = load_u32(p);
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

// A JSON number that is a whole number from 0 up (JSON numbers are
// doubles; 9e15 is below 2^53, so every such integer is exact)
bool to_uint(const JsonValue& value, std::uint64_t& out) {
    if (!value.is_number()) {
        return false;
    }

    const double d = value.as_number();

    if (!(d >= 0.0 && d <= 9.0e15) || d != std::floor(d)) {
        return false;
    }

    out = static_cast<std::uint64_t>(d);
    return true;
}

std::string number_text(std::uint64_t value) {
    return std::to_string(value);
}

bool starts_with_icase(const std::string& text, const char* prefix) {
    const std::size_t length = std::strlen(prefix);

    if (text.size() < length) {
        return false;
    }

    for (std::size_t i = 0; i < length; ++i) {
        const int a = std::tolower(static_cast<unsigned char>(text[i]));
        const int b = std::tolower(static_cast<unsigned char>(prefix[i]));

        if (a != b) {
            return false;
        }
    }

    return true;
}

bool ends_with_icase(const std::string& text, const char* suffix) {
    const std::size_t length = std::strlen(suffix);

    if (text.size() < length) {
        return false;
    }

    return starts_with_icase(text.substr(text.size() - length), suffix);
}

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

// URIs write special characters as %XX (a space is %20). File names are
// UTF-8 bytes, so the result is a UTF-8 string
bool percent_decode(const std::string& text, std::string& out) {
    out.clear();
    out.reserve(text.size());

    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] != '%') {
            out.push_back(text[i]);
            continue;
        }

        // '%' must be followed by two more characters
        if (i + 2 >= text.size()) {
            return false;
        }

        const int high = hex_value(text[i + 1]);
        const int low = hex_value(text[i + 2]);

        if (high < 0 || low < 0) {
            return false;
        }

        out.push_back(static_cast<char>(high * 16 + low));
        i += 2;
    }

    return true;
}

bool read_file_bytes(const std::filesystem::path& path,
                     std::vector<std::uint8_t>& out) {
    out.clear();

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        return false;
    }

    file.seekg(0, std::ios::end);
    const std::streamoff size = file.tellg();

    if (size < 0) {
        return false;
    }

    file.seekg(0, std::ios::beg);
    out.resize(static_cast<std::size_t>(size));

    if (size > 0) {
        file.read(reinterpret_cast<char*>(out.data()), size);

        if (!file) {
            out.clear();
            return false;
        }
    }

    return true;
}

// Number of components of an accessor "type": SCALAR is 1, VEC3 is 3,
// MAT4 is 16. rows is the length of a column (matrices are stored
// column by column), columns is 1 for everything but matrices
bool accessor_shape(const std::string& type, int& rows, int& columns) {
    columns = 1;

    if (type == "SCALAR") { rows = 1; return true; }
    if (type == "VEC2")   { rows = 2; return true; }
    if (type == "VEC3")   { rows = 3; return true; }
    if (type == "VEC4")   { rows = 4; return true; }
    if (type == "MAT2")   { rows = 2; columns = 2; return true; }
    if (type == "MAT3")   { rows = 3; columns = 3; return true; }
    if (type == "MAT4")   { rows = 4; columns = 4; return true; }

    return false;
}

std::size_t component_size(std::uint64_t component_type) {
    switch (component_type) {
        case kByte:
        case kUnsignedByte:
            return 1;
        case kShort:
        case kUnsignedShort:
            return 2;
        case kUnsignedInt:
        case kFloat:
            return 4;
        default:
            return 0;
    }
}

// One number of an accessor as float. Integer types that are
// "normalized" mean a value between 0 and 1 (or -1 and 1 when signed):
// the full range of the integer maps onto it. These are the formulas of
// the glTF specification; the signed ones clamp because -128 / 127
// would be slightly below -1
float read_component(const std::uint8_t* p, std::uint64_t type,
                     bool normalized) {
    switch (type) {
        case kByte: {
            const float v = static_cast<float>(static_cast<std::int8_t>(p[0]));
            return normalized ? std::max(v / 127.0f, -1.0f) : v;
        }
        case kUnsignedByte: {
            const float v = static_cast<float>(p[0]);
            return normalized ? v / 255.0f : v;
        }
        case kShort: {
            const float v = static_cast<float>(
                static_cast<std::int16_t>(load_u16(p)));
            return normalized ? std::max(v / 32767.0f, -1.0f) : v;
        }
        case kUnsignedShort: {
            const float v = static_cast<float>(load_u16(p));
            return normalized ? v / 65535.0f : v;
        }
        case kUnsignedInt:
            return static_cast<float>(load_u32(p));
        case kFloat:
            return load_f32(p);
        default:
            return 0.0f;
    }
}

// ---------------------------------------------------------------------
// Matrices from node transforms
// ---------------------------------------------------------------------

// Rotation from a quaternion (x, y, z, w). The spec wants a unit
// quaternion; it is normalised here so a slightly-off one does not skew
// the model, and an all-zero one becomes "no rotation"
Mat4 rotation_from_quaternion(float qx, float qy, float qz, float qw) {
    const float len = std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);

    if (!(len > 1.0e-12f)) {
        return mat4_identity();
    }

    const float x = qx / len;
    const float y = qy / len;
    const float z = qz / len;
    const float w = qw / len;

    return {{
        {1.0f - 2.0f * (y * y + z * z),
         2.0f * (x * y + z * w),
         2.0f * (x * z - y * w),
         0.0f},
        {2.0f * (x * y - z * w),
         1.0f - 2.0f * (x * x + z * z),
         2.0f * (y * z + x * w),
         0.0f},
        {2.0f * (x * z + y * w),
         2.0f * (y * z - x * w),
         1.0f - 2.0f * (x * x + y * y),
         0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};
}

// The change of axes from Y-up to Z-up as a matrix: (x, y, z) ->
// (x, -z, y). Its columns are where the old axes end up
Mat4 y_up_to_z_up_matrix() {
    return {{
        {1.0f, 0.0f, 0.0f, 0.0f},   // old +X stays +X
        {0.0f, 0.0f, 1.0f, 0.0f},   // old +Y (up) becomes +Z
        {0.0f, -1.0f, 0.0f, 0.0f},  // old +Z (towards the viewer) becomes -Y
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};
}

// ---------------------------------------------------------------------
// The reader
// ---------------------------------------------------------------------

// A range of bytes somewhere in memory
struct ByteRange {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
};

// A buffer view, bounds-checked against its buffer
struct BufferViewData {
    const std::uint8_t* data = nullptr;
    std::size_t size = 0;
    std::size_t stride = 0;  // byteStride, or 0 when not given
};

// An accessor, validated against its buffer view: element e, component
// k starts at
//   data + e * stride + (k / rows) * column_bytes + (k % rows) * component_size
// (vectors have one column, so that reduces to k * component_size)
struct Accessor {
    std::uint64_t index = 0;
    std::uint64_t component_type = 0;
    std::size_t component_size = 0;
    int rows = 0;
    int columns = 1;
    bool normalized = false;
    std::uint64_t count = 0;

    // false = the accessor has no buffer view, which means "all zeros"
    bool has_data = false;
    const std::uint8_t* data = nullptr;
    std::size_t stride = 0;
    std::size_t column_bytes = 0;

    int components() const { return rows * columns; }
};

enum class AccessorStatus {
    ACCESSOR_OK,
    ACCESSOR_ERROR,
    ACCESSOR_UNSUPPORTED  // valid, but uses a feature we skip (sparse)
};

enum class FetchResult {
    FETCH_OK,
    FETCH_ABSENT,
    FETCH_SKIP,   // primitive must be skipped (already warned)
    FETCH_ERROR
};

enum class PrimitiveResult {
    PRIMITIVE_OK,
    PRIMITIVE_SKIPPED,
    PRIMITIVE_ERROR
};

// A glTF mesh being assembled from its primitives
struct MeshBuild {
    Mesh mesh;
    bool all_normals = true;
    bool all_texcoords = true;

    // Primitives that read the same position/normal/texcoord accessors
    // (the usual case when one mesh has several materials) share one
    // copy of the vertices. Key: those three accessor numbers (or
    // kNoAccessor); value: where the vertices start in mesh.positions
    std::map<std::array<std::uint64_t, 3>, std::uint32_t> vertex_base;
};

constexpr std::uint64_t kNoAccessor = ~std::uint64_t{0};

class GltfReader {
public:
    GltfReader(const JsonValue& doc, ByteRange glb_bin, bool has_glb_bin,
               const std::string& base_dir)
        : m_doc(doc),
          m_glb_bin(glb_bin),
          m_has_glb_bin(has_glb_bin),
          m_has_base_dir(!base_dir.empty()),
          m_base_dir(std::filesystem::u8path(base_dir)) {}

    bool run(Model& out, std::string& error) {
        const bool ok = read_all();

        if (ok) {
            out = std::move(m_model);
        } else {
            error = m_error;
        }

        return ok;
    }

private:
    // ---- Errors and warnings --------------------------------------

    bool fail(const std::string& message) {
        m_error = message;
        return false;
    }

    void warn(const std::string& message) {
        LOG_WARN("glTF: %s", message.c_str());
    }

    // For warnings that would otherwise repeat for every use
    void warn_once(const std::string& key, const std::string& message) {
        if (m_warned.insert(key).second) {
            warn(message);
        }
    }

    // ---- Reading JSON members -------------------------------------

    // Optional whole number member. Missing gives the fallback; present
    // but not a whole number >= 0 is an error
    bool opt_uint(const JsonValue& object, const char* key,
                  std::uint64_t fallback, std::uint64_t& out,
                  const std::string& where) {
        out = fallback;

        if (!object.contains(key)) {
            return true;
        }

        if (!to_uint(object[key], out)) {
            return fail(where + ": '" + key +
                        "' must be a whole number of 0 or more");
        }

        return true;
    }

    // Optional number member
    bool opt_number(const JsonValue& object, const char* key,
                    float fallback, float& out, const std::string& where) {
        out = fallback;

        if (!object.contains(key)) {
            return true;
        }

        if (!object[key].is_number()) {
            return fail(where + ": '" + key + "' must be a number");
        }

        out = object[key].as_float();
        return true;
    }

    // Optional array of exactly count numbers; out is left alone when
    // the member is missing
    bool opt_float_array(const JsonValue& object, const char* key,
                         std::size_t count, float* out, bool& present,
                         const std::string& where) {
        present = object.contains(key);

        if (!present) {
            return true;
        }

        const JsonValue& array = object[key];

        if (!array.is_array() || array.size() != count) {
            return fail(where + ": '" + key + "' must be an array of " +
                        std::to_string(count) + " numbers");
        }

        for (std::size_t i = 0; i < count; ++i) {
            if (!array[i].is_number()) {
                return fail(where + ": '" + key + "' must be an array of " +
                            std::to_string(count) + " numbers");
            }

            out[i] = array[i].as_float();
        }

        return true;
    }

    // ---- Buffers --------------------------------------------------

    // Data behind a URI: a data: URI (base64) or a file next to the
    // glTF. The error describes the problem without saying where
    bool read_uri(const std::string& uri, std::vector<std::uint8_t>& out,
                  std::string& err) {
        if (starts_with_icase(uri, "data:")) {
            // data:<media type>;base64,<data>
            const std::size_t comma = uri.find(',');

            if (comma == std::string::npos) {
                err = "malformed data URI (no comma)";
                return false;
            }

            if (!ends_with_icase(uri.substr(0, comma), ";base64")) {
                err = "only base64 data URIs are supported";
                return false;
            }

            if (!decode_base64(std::string_view(uri).substr(comma + 1),
                               out)) {
                err = "invalid base64 data in data URI";
                return false;
            }

            return true;
        }

        if (uri.empty()) {
            err = "empty uri";
            return false;
        }

        if (uri.find("://") != std::string::npos) {
            err = "'" + uri + "' is a remote URI (not supported)";
            return false;
        }

        if (!m_has_base_dir) {
            err = "'" + uri + "' is an external file, but the glTF was "
                  "loaded from memory without a base folder";
            return false;
        }

        std::string decoded;

        if (!percent_decode(uri, decoded)) {
            err = "invalid %-escape in uri '" + uri + "'";
            return false;
        }

        // u8path: the name is UTF-8 whatever the platform's own
        // encoding is. If the URI is an absolute path, the folder is
        // simply replaced by it
        const std::filesystem::path path =
            m_base_dir / std::filesystem::u8path(decoded);

        if (!read_file_bytes(path, out)) {
            err = "could not read file '" + path.u8string() + "'";
            return false;
        }

        return true;
    }

    // Makes sure buffer `index` is loaded and returns its bytes
    bool get_buffer(std::uint64_t index, ByteRange& out, std::string& err) {
        if (index >= m_buffers.size()) {
            err = "buffer " + number_text(index) + " does not exist (the "
                  "file has " + std::to_string(m_buffers.size()) + ")";
            return false;
        }

        LoadedBuffer& buffer = m_buffers[static_cast<std::size_t>(index)];

        if (!buffer.loaded) {
            const JsonValue& js = m_doc["buffers"][index];
            const std::string where = "buffer " + number_text(index);

            std::uint64_t byte_length = 0;

            if (!to_uint(js["byteLength"], byte_length)) {
                err = where + " has no valid byteLength";
                return false;
            }

            ByteRange range;

            if (!js.contains("uri")) {
                // In a .glb, the first buffer without a uri is the BIN
                // chunk
                if (index != 0 || !m_has_glb_bin) {
                    err = where + " has no uri";

                    if (index == 0) {
                        err += " and the file has no BIN chunk";
                    }

                    return false;
                }

                range = m_glb_bin;
            } else {
                std::string uri_error;

                if (!read_uri(js["uri"].as_string(), buffer.owned,
                              uri_error)) {
                    err = where + ": " + uri_error;
                    return false;
                }

                range.data = buffer.owned.data();
                range.size = buffer.owned.size();
            }

            if (byte_length > range.size) {
                err = where + " says byteLength " + number_text(byte_length) +
                      " but has only " + std::to_string(range.size) +
                      " bytes of data";
                return false;
            }

            // Only the declared length may be used
            buffer.data = range.data;
            buffer.size = static_cast<std::size_t>(byte_length);
            buffer.loaded = true;
        }

        out.data = buffer.data;
        out.size = buffer.size;
        return true;
    }

    // ---- Buffer views and accessors ---------------------------------

    bool get_view(std::uint64_t index, BufferViewData& out,
                  std::string& err) {
        const JsonValue& views = m_doc["bufferViews"];

        if (index >= views.size()) {
            err = "buffer view " + number_text(index) + " does not exist "
                  "(the file has " + std::to_string(views.size()) + ")";
            return false;
        }

        const JsonValue& js = views[index];
        const std::string where = "buffer view " + number_text(index);

        std::uint64_t buffer_index = 0;
        std::uint64_t offset = 0;
        std::uint64_t length = 0;
        std::uint64_t stride = 0;

        if (!to_uint(js["buffer"], buffer_index)) {
            err = where + " has no valid buffer";
            return false;
        }

        if (!to_uint(js["byteLength"], length)) {
            err = where + " has no valid byteLength";
            return false;
        }

        if (js.contains("byteOffset") && !to_uint(js["byteOffset"], offset)) {
            err = where + " has an invalid byteOffset";
            return false;
        }

        if (js.contains("byteStride") && !to_uint(js["byteStride"], stride)) {
            err = where + " has an invalid byteStride";
            return false;
        }

        ByteRange buffer;

        if (!get_buffer(buffer_index, buffer, err)) {
            return false;
        }

        if (offset > buffer.size || length > buffer.size - offset) {
            err = where + " runs past the end of buffer " +
                  number_text(buffer_index);
            return false;
        }

        out.data = buffer.data + offset;
        out.size = static_cast<std::size_t>(length);
        out.stride = static_cast<std::size_t>(stride);
        return true;
    }

    AccessorStatus get_accessor(std::uint64_t index, Accessor& out,
                                std::string& err) {
        const JsonValue& accessors = m_doc["accessors"];

        if (index >= accessors.size()) {
            err = "accessor " + number_text(index) + " does not exist (the "
                  "file has " + std::to_string(accessors.size()) + ")";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        const JsonValue& js = accessors[index];
        const std::string where = "accessor " + number_text(index);

        // Sparse accessors patch a few elements of a base array. Rare
        // in practice (mostly morph targets); not supported
        if (js.contains("sparse")) {
            err = where + " uses sparse storage, which is not supported";
            return AccessorStatus::ACCESSOR_UNSUPPORTED;
        }

        out = Accessor{};
        out.index = index;

        if (!to_uint(js["componentType"], out.component_type) ||
            component_size(out.component_type) == 0) {
            err = where + " has an invalid componentType";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        out.component_size = component_size(out.component_type);

        if (!accessor_shape(js["type"].as_string(), out.rows, out.columns)) {
            err = where + " has an invalid type '" +
                  js["type"].as_string() + "'";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        if (!to_uint(js["count"], out.count) || out.count == 0) {
            err = where + " has an invalid count (must be 1 or more)";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        out.normalized = js["normalized"].as_bool(false);

        // A matrix column is padded to a multiple of 4 bytes (only
        // matters for 1 and 2 byte components)
        out.column_bytes = static_cast<std::size_t>(out.rows) *
                           out.component_size;

        if (out.columns > 1 && out.column_bytes % 4 != 0) {
            out.column_bytes += 4 - out.column_bytes % 4;
        }

        const std::uint64_t element_bytes =
            static_cast<std::uint64_t>(out.column_bytes) *
            static_cast<std::uint64_t>(out.columns);

        // No buffer view: the accessor is all zeros. Nothing limits the
        // count then, so cap it
        if (!js.contains("bufferView")) {
            if (out.count > kMaxZeroAccessorCount) {
                err = where + " has no buffer view and an unreasonably "
                      "large count";
                return AccessorStatus::ACCESSOR_ERROR;
            }

            out.has_data = false;
            return AccessorStatus::ACCESSOR_OK;
        }

        std::uint64_t view_index = 0;

        if (!to_uint(js["bufferView"], view_index)) {
            err = where + " has an invalid bufferView";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        std::uint64_t offset = 0;

        if (js.contains("byteOffset") && !to_uint(js["byteOffset"], offset)) {
            err = where + " has an invalid byteOffset";
            return AccessorStatus::ACCESSOR_ERROR;
        }

        BufferViewData view;

        if (!get_view(view_index, view, err)) {
            return AccessorStatus::ACCESSOR_ERROR;
        }

        const std::uint64_t stride =
            view.stride != 0 ? view.stride : element_bytes;

        if (stride < element_bytes) {
            err = where + ": its elements (" + number_text(element_bytes) +
                  " bytes) are larger than the byteStride of buffer view " +
                  number_text(view_index);
            return AccessorStatus::ACCESSOR_ERROR;
        }

        // The last element must lie completely inside the view: the
        // first element, then count - 1 steps of one stride. Compared
        // with a division so that a hostile count cannot overflow the
        // multiplication
        const bool fits =
            offset <= view.size &&
            element_bytes <= view.size - offset &&
            out.count - 1 <= (view.size - offset - element_bytes) / stride;

        if (!fits) {
            err = where + " runs past the end of buffer view " +
                  number_text(view_index);
            return AccessorStatus::ACCESSOR_ERROR;
        }

        out.has_data = true;
        out.data = view.data + offset;
        out.stride = static_cast<std::size_t>(stride);
        return AccessorStatus::ACCESSOR_OK;
    }

    // Every component of every element as float, element after element
    void read_floats(const Accessor& accessor, std::vector<float>& out) {
        const std::size_t count = static_cast<std::size_t>(accessor.count);
        const std::size_t components =
            static_cast<std::size_t>(accessor.components());

        out.assign(count * components, 0.0f);

        if (!accessor.has_data) {
            return;
        }

        const std::size_t rows = static_cast<std::size_t>(accessor.rows);

        for (std::size_t e = 0; e < count; ++e) {
            const std::uint8_t* element = accessor.data + e * accessor.stride;

            for (std::size_t k = 0; k < components; ++k) {
                const std::uint8_t* p = element +
                    (k / rows) * accessor.column_bytes +
                    (k % rows) * accessor.component_size;

                out[e * components + k] = read_component(
                    p, accessor.component_type, accessor.normalized);
            }
        }
    }

    // Index values. Only SCALAR unsigned byte/short/int are valid
    bool read_indices(const Accessor& accessor,
                      std::vector<std::uint32_t>& out, std::string& err) {
        const bool type_ok =
            accessor.components() == 1 &&
            (accessor.component_type == kUnsignedByte ||
             accessor.component_type == kUnsignedShort ||
             accessor.component_type == kUnsignedInt);

        if (!type_ok) {
            err = "indices accessor " + number_text(accessor.index) +
                  " must be SCALAR with unsigned byte, short or int "
                  "components";
            return false;
        }

        const std::size_t count = static_cast<std::size_t>(accessor.count);
        out.assign(count, 0u);

        if (!accessor.has_data) {
            return true;
        }

        for (std::size_t e = 0; e < count; ++e) {
            const std::uint8_t* p = accessor.data + e * accessor.stride;

            switch (accessor.component_type) {
                case kUnsignedByte:
                    out[e] = p[0];
                    break;
                case kUnsignedShort:
                    out[e] = load_u16(p);
                    break;
                default:
                    out[e] = load_u32(p);
                    break;
            }
        }

        return true;
    }

    // ---- Images and materials ---------------------------------------

    // What to call a glTF image in logs: its file name, or a description
    // for images inside the file
    std::string image_description(std::size_t source) {
        const JsonValue& js = m_doc["images"][source];

        if (js.contains("uri")) {
            const std::string& uri = js["uri"].as_string();
            std::string decoded;

            if (!starts_with_icase(uri, "data:") &&
                percent_decode(uri, decoded)) {
                const std::string name =
                    std::filesystem::u8path(decoded).filename().u8string();

                if (!name.empty()) {
                    return name;
                }
            }
        }

        return "embedded image " + std::to_string(source);
    }

    // Decodes one glTF image. Any problem is only a warning: the
    // material just goes without its texture
    bool decode_gltf_image(std::size_t source, Image& out) {
        const JsonValue& js = m_doc["images"][source];
        const std::string name = image_description(source);
        const std::string prefix = "image " + std::to_string(source) +
                                   " (" + name + "): ";

        std::vector<std::uint8_t> owned;
        const std::uint8_t* data = nullptr;
        std::size_t size = 0;
        std::string err;

        if (js.contains("uri")) {
            if (!read_uri(js["uri"].as_string(), owned, err)) {
                warn(prefix + err + "; texture ignored");
                return false;
            }

            data = owned.data();
            size = owned.size();
        } else if (js.contains("bufferView")) {
            std::uint64_t view_index = 0;
            BufferViewData view;

            if (!to_uint(js["bufferView"], view_index) ||
                !get_view(view_index, view, err)) {
                if (err.empty()) {
                    err = "invalid bufferView";
                }

                warn(prefix + err + "; texture ignored");
                return false;
            }

            data = view.data;
            size = view.size;
        } else {
            warn(prefix + "has neither uri nor bufferView; texture ignored");
            return false;
        }

        if (size >= 8 && std::memcmp(data, kPngSignature, 8) == 0) {
            std::string png_error;

            if (!decode_png(data, size, out, png_error)) {
                warn(prefix + png_error + "; texture ignored");
                return false;
            }

            return true;
        }

        if (size >= 3 && data[0] == 0xFF && data[1] == 0xD8 &&
            data[2] == 0xFF) {
            warn(prefix + "JPEG textures are not supported yet; texture "
                 "ignored");
            return false;
        }

        warn(prefix + "unsupported image format (only PNG is supported); "
             "texture ignored");
        return false;
    }

    // Index into Model::images of glTF image `source`, decoding it the
    // first time it is asked for; -1 if it cannot be used. A failure is
    // remembered too, so the warning appears once
    int resolve_image(std::size_t source) {
        int& result = m_image_result[source];

        if (result != kImageUntried) {
            return result;
        }

        result = -1;

        Image image;

        if (decode_gltf_image(source, image)) {
            m_model.images.push_back(std::move(image));
            result = static_cast<int>(m_model.images.size()) - 1;
        }

        return result;
    }

    // Reads a texture reference like
    //   "baseColorTexture": { "index": 0, "texCoord": 0 }
    // from `parent`. image_index becomes -1 for no texture; description
    // is set when there is a texture (even one that failed to decode)
    bool read_texture_info(const JsonValue& parent, const char* key,
                           const std::string& where, int& image_index,
                           std::string& description) {
        image_index = -1;
        description.clear();

        const JsonValue& info = parent[key];

        if (!info.is_object()) {
            return true;
        }

        const std::string slot = where + " " + key;

        std::uint64_t tex_coord = 0;

        if (!opt_uint(info, "texCoord", 0, tex_coord, slot)) {
            return false;
        }

        if (tex_coord != 0) {
            warn(slot + " uses TEXCOORD_" + number_text(tex_coord) +
                 ", but only TEXCOORD_0 is supported; texture ignored");
            return true;
        }

        if (info["extensions"].contains("KHR_texture_transform")) {
            warn_once("KHR_texture_transform",
                      "KHR_texture_transform is not supported; textures "
                      "are used untransformed");
        }

        std::uint64_t texture_index = 0;

        if (!to_uint(info["index"], texture_index)) {
            return fail(slot + " has no valid texture index");
        }

        const JsonValue& textures = m_doc["textures"];

        if (texture_index >= textures.size()) {
            return fail(slot + ": texture " + number_text(texture_index) +
                        " does not exist (the file has " +
                        std::to_string(textures.size()) + ")");
        }

        // The sampler (filtering and wrapping) of the texture is not
        // used; the renderers choose their own for now.
        // A texture without "source" is one that only an extension
        // (Basis, WebP, ...) can supply
        const JsonValue& texture = textures[texture_index];

        if (!texture.contains("source")) {
            warn(slot + ": texture " + number_text(texture_index) +
                 " has no source image; texture ignored");
            return true;
        }

        std::uint64_t source = 0;
        const JsonValue& images = m_doc["images"];

        if (!to_uint(texture["source"], source) || source >= images.size()) {
            return fail(slot + ": texture " + number_text(texture_index) +
                        " refers to an image that does not exist");
        }

        description = image_description(static_cast<std::size_t>(source));
        image_index = resolve_image(static_cast<std::size_t>(source));
        return true;
    }

    bool parse_material(std::size_t index, Material& material) {
        const JsonValue& js = m_doc["materials"][index];
        const std::string where = "material " + std::to_string(index);

        material = Material{};
        material.name = js["name"].is_string() && !js["name"].as_string().empty()
                            ? js["name"].as_string()
                            : where;

        const JsonValue& pbr = js["pbrMetallicRoughness"];

        // glTF's own defaults, which differ from our Material struct's:
        // a glTF material with no values is white, fully metallic and
        // fully rough
        material.base_color = {1.0f, 1.0f, 1.0f};
        material.metallic = 1.0f;
        material.roughness = 1.0f;

        float color[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        bool present = false;

        if (!opt_float_array(pbr, "baseColorFactor", 4, color, present,
                             where)) {
            return false;
        }

        // The alpha (color[3]) is ignored for now
        material.base_color = {color[0], color[1], color[2]};

        float factor = 1.0f;

        if (!opt_number(pbr, "metallicFactor", 1.0f, factor, where)) {
            return false;
        }
        material.metallic = std::min(std::max(factor, 0.0f), 1.0f);

        if (!opt_number(pbr, "roughnessFactor", 1.0f, factor, where)) {
            return false;
        }
        material.roughness = std::min(std::max(factor, 0.0f), 1.0f);

        // Emission: the colour times the optional strength extension
        // (glTF colours are limited to 0..1, the extension allows
        // brighter lights)
        float emissive[3] = {0.0f, 0.0f, 0.0f};

        if (!opt_float_array(js, "emissiveFactor", 3, emissive, present,
                             where)) {
            return false;
        }

        float strength = 1.0f;
        const JsonValue& extensions = js["extensions"];
        const JsonValue& strength_ext =
            extensions["KHR_materials_emissive_strength"];

        if (strength_ext.is_object() &&
            !opt_number(strength_ext, "emissiveStrength", 1.0f, strength,
                        where)) {
            return false;
        }

        material.emission = {emissive[0] * strength,
                             emissive[1] * strength,
                             emissive[2] * strength};

        // Other material extensions (clearcoat, transmission, ...) are
        // ignored; say so once
        for (const JsonMember& member : extensions.members()) {
            if (member.key != "KHR_materials_emissive_strength") {
                warn_once("ext:" + member.key,
                          "material extension " + member.key +
                          " is not supported and is ignored");
            }
        }

        // The four textures
        std::string description;

        if (!read_texture_info(pbr, "baseColorTexture", where,
                               material.base_color_image, description)) {
            return false;
        }
        material.base_color_texture = description;

        if (!read_texture_info(pbr, "metallicRoughnessTexture", where,
                               material.metallic_roughness_image,
                               description)) {
            return false;
        }

        if (!read_texture_info(js, "normalTexture", where,
                               material.normal_image, description)) {
            return false;
        }

        if (!read_texture_info(js, "emissiveTexture", where,
                               material.emissive_image, description)) {
            return false;
        }

        return true;
    }

    bool parse_materials() {
        const std::size_t count = m_doc["materials"].size();

        m_materials.resize(count);

        for (std::size_t i = 0; i < count; ++i) {
            if (!parse_material(i, m_materials[i])) {
                return false;
            }
        }

        return true;
    }

    // ---- Meshes -----------------------------------------------------

    // Looks up the accessor an attribute names
    FetchResult fetch_attribute(const JsonValue& attributes, const char* name,
                                Accessor& accessor, const std::string& where) {
        if (!attributes.contains(name)) {
            return FetchResult::FETCH_ABSENT;
        }

        std::uint64_t index = 0;

        if (!to_uint(attributes[name], index)) {
            fail(where + ": attribute " + name +
                 " must be an accessor index");
            return FetchResult::FETCH_ERROR;
        }

        std::string err;

        switch (get_accessor(index, accessor, err)) {
            case AccessorStatus::ACCESSOR_OK:
                return FetchResult::FETCH_OK;
            case AccessorStatus::ACCESSOR_UNSUPPORTED:
                warn(where + " skipped: " + err);
                return FetchResult::FETCH_SKIP;
            default:
                fail(err);
                return FetchResult::FETCH_ERROR;
        }
    }

    // Turns the vertex order of a primitive into triangles
    //   TRIANGLES:      every 3 entries are one triangle
    //   TRIANGLE_STRIP: each entry after the first two adds a triangle
    //                   with the two before it; every second triangle is
    //                   flipped so all keep the same winding
    //   TRIANGLE_FAN:   every triangle shares the first entry
    void make_triangles(std::uint64_t mode, const std::string& where,
                        const std::vector<std::uint32_t>& order,
                        std::vector<std::uint32_t>& triangles) {
        triangles.clear();

        const std::size_t n = order.size();

        if (mode == kModeTriangles) {
            std::size_t usable = n - n % 3;

            if (usable != n) {
                warn(where + ": " + std::to_string(n - usable) +
                     " trailing index(es) do not form a triangle and "
                     "are ignored");
            }

            triangles.assign(order.begin(),
                             order.begin() + static_cast<std::ptrdiff_t>(usable));
            return;
        }

        if (n < 3) {
            return;
        }

        triangles.reserve((n - 2) * 3);

        for (std::size_t i = 0; i + 2 < n; ++i) {
            if (mode == kModeTriangleStrip) {
                if (i % 2 == 0) {
                    triangles.push_back(order[i]);
                    triangles.push_back(order[i + 1]);
                } else {
                    triangles.push_back(order[i + 1]);
                    triangles.push_back(order[i]);
                }
                triangles.push_back(order[i + 2]);
            } else {
                triangles.push_back(order[0]);
                triangles.push_back(order[i + 1]);
                triangles.push_back(order[i + 2]);
            }
        }
    }

    PrimitiveResult add_primitive(std::size_t mesh_index,
                                  std::size_t primitive_index,
                                  const JsonValue& primitive,
                                  MeshBuild& build) {
        const std::string where = "mesh " + std::to_string(mesh_index) +
                                  " primitive " +
                                  std::to_string(primitive_index);

        if (!primitive.is_object()) {
            fail(where + " is not an object");
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        std::uint64_t mode = kModeTriangles;

        if (!opt_uint(primitive, "mode", kModeTriangles, mode, where)) {
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        if (mode <= 3) {
            warn(where + " skipped: points and lines are not supported");
            return PrimitiveResult::PRIMITIVE_SKIPPED;
        }

        if (mode > kModeTriangleFan) {
            fail(where + " has an invalid mode " + number_text(mode));
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        const JsonValue& attributes = primitive["attributes"];

        if (!attributes.is_object() || !attributes.contains("POSITION")) {
            fail(where + " has no POSITION attribute");
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        // ---- Look up and check the accessors ---------------------
        Accessor position;
        Accessor normal;
        Accessor texcoord;
        Accessor index_accessor;

        FetchResult result =
            fetch_attribute(attributes, "POSITION", position, where);

        if (result == FetchResult::FETCH_SKIP) {
            return PrimitiveResult::PRIMITIVE_SKIPPED;
        }
        if (result != FetchResult::FETCH_OK) {
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        if (position.components() != 3 || position.component_type != kFloat) {
            fail(where + ": POSITION accessor " + number_text(position.index) +
                 " must be VEC3 of floats");
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        result = fetch_attribute(attributes, "NORMAL", normal, where);

        if (result == FetchResult::FETCH_SKIP) {
            return PrimitiveResult::PRIMITIVE_SKIPPED;
        }
        if (result == FetchResult::FETCH_ERROR) {
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        const bool has_normal = result == FetchResult::FETCH_OK;

        if (has_normal &&
            (normal.components() != 3 || normal.component_type != kFloat ||
             normal.count != position.count)) {
            fail(where + ": NORMAL accessor " + number_text(normal.index) +
                 " must be VEC3 of floats with one entry per vertex");
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        result = fetch_attribute(attributes, "TEXCOORD_0", texcoord, where);

        if (result == FetchResult::FETCH_SKIP) {
            return PrimitiveResult::PRIMITIVE_SKIPPED;
        }
        if (result == FetchResult::FETCH_ERROR) {
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        const bool has_texcoord = result == FetchResult::FETCH_OK;

        if (has_texcoord) {
            const bool type_ok =
                texcoord.components() == 2 &&
                (texcoord.component_type == kFloat ||
                 (texcoord.normalized &&
                  texcoord.component_type != kUnsignedInt));

            if (!type_ok || texcoord.count != position.count) {
                fail(where + ": TEXCOORD_0 accessor " +
                     number_text(texcoord.index) + " must be VEC2 of "
                     "floats or normalized bytes/shorts, with one entry "
                     "per vertex");
                return PrimitiveResult::PRIMITIVE_ERROR;
            }
        }

        // Indices (optional) and material
        std::vector<std::uint32_t> order;
        bool has_indices = false;

        if (primitive.contains("indices")) {
            std::uint64_t accessor_index = 0;

            if (!to_uint(primitive["indices"], accessor_index)) {
                fail(where + ": 'indices' must be an accessor index");
                return PrimitiveResult::PRIMITIVE_ERROR;
            }

            std::string err;

            switch (get_accessor(accessor_index, index_accessor, err)) {
                case AccessorStatus::ACCESSOR_OK:
                    break;
                case AccessorStatus::ACCESSOR_UNSUPPORTED:
                    warn(where + " skipped: " + err);
                    return PrimitiveResult::PRIMITIVE_SKIPPED;
                default:
                    fail(err);
                    return PrimitiveResult::PRIMITIVE_ERROR;
            }

            has_indices = true;
        }

        std::uint32_t material_index = 0;
        bool uses_default_material = false;

        if (primitive.contains("material")) {
            std::uint64_t value = 0;

            if (!to_uint(primitive["material"], value) ||
                value >= m_materials.size()) {
                fail(where + ": 'material' is not a valid material index "
                     "(the file has " + std::to_string(m_materials.size()) +
                     " materials)");
                return PrimitiveResult::PRIMITIVE_ERROR;
            }

            material_index = static_cast<std::uint32_t>(value);
        } else {
            // The default material goes after the file's own materials
            material_index = static_cast<std::uint32_t>(m_materials.size());
            uses_default_material = true;
        }

        // ---- Vertices: reuse an earlier primitive's copy, or read ------
        const std::array<std::uint64_t, 3> key = {
            position.index,
            has_normal ? normal.index : kNoAccessor,
            has_texcoord ? texcoord.index : kNoAccessor
        };

        const auto shared = build.vertex_base.find(key);
        const bool reuse_vertices = shared != build.vertex_base.end();

        // Budget, so that hostile files cannot ask for gigabytes
        const std::uint64_t new_vertices = reuse_vertices ? 0 : position.count;
        const std::uint64_t index_count =
            has_indices ? index_accessor.count : position.count;

        if (m_total_vertices + new_vertices > kMaxTotalVertices ||
            m_total_indices + index_count > kMaxTotalIndices) {
            fail(where + ": the model is too large");
            return PrimitiveResult::PRIMITIVE_ERROR;
        }

        const std::size_t vertex_count =
            static_cast<std::size_t>(position.count);

        std::vector<Vec3> positions;
        std::vector<Vec3> normals;
        std::vector<Vec2> texcoords;

        if (!reuse_vertices) {
            std::vector<float> floats;
            read_floats(position, floats);

            positions.resize(vertex_count);

            for (std::size_t i = 0; i < vertex_count; ++i) {
                positions[i] = {floats[i * 3], floats[i * 3 + 1],
                                floats[i * 3 + 2]};

                if (!std::isfinite(positions[i].x) ||
                    !std::isfinite(positions[i].y) ||
                    !std::isfinite(positions[i].z)) {
                    fail(where + ": POSITION contains a value that is not "
                         "a finite number");
                    return PrimitiveResult::PRIMITIVE_ERROR;
                }
            }

            if (has_normal) {
                read_floats(normal, floats);
                normals.resize(vertex_count);

                for (std::size_t i = 0; i < vertex_count; ++i) {
                    normals[i] = {floats[i * 3], floats[i * 3 + 1],
                                  floats[i * 3 + 2]};
                }
            }

            if (has_texcoord) {
                read_floats(texcoord, floats);
                texcoords.resize(vertex_count);

                for (std::size_t i = 0; i < vertex_count; ++i) {
                    // glTF has (0, 0) at the TOP-left of the image, the
                    // convention of Mesh::texcoords (that of OBJ) at the
                    // bottom-left, so v is flipped here and nowhere else
                    texcoords[i] = {floats[i * 2], 1.0f - floats[i * 2 + 1]};
                }
            }
        }

        // ---- Triangles ----------------------------------------------
        if (has_indices) {
            std::string err;

            if (!read_indices(index_accessor, order, err)) {
                fail(where + ": " + err);
                return PrimitiveResult::PRIMITIVE_ERROR;
            }

            for (const std::uint32_t value : order) {
                if (value >= position.count) {
                    fail(where + ": index " + std::to_string(value) +
                         " is out of range (the primitive has " +
                         number_text(position.count) + " vertices)");
                    return PrimitiveResult::PRIMITIVE_ERROR;
                }
            }
        } else {
            // Without indices the vertices are used in order
            order.resize(vertex_count);

            for (std::size_t i = 0; i < vertex_count; ++i) {
                order[i] = static_cast<std::uint32_t>(i);
            }
        }

        std::vector<std::uint32_t> triangles;
        make_triangles(mode, where, order, triangles);

        if (triangles.empty()) {
            warn(where + " skipped: it has no complete triangle");
            return PrimitiveResult::PRIMITIVE_SKIPPED;
        }

        // ---- Append to the mesh ----------------------------------------
        Mesh& mesh = build.mesh;
        std::uint32_t base = 0;

        if (reuse_vertices) {
            base = shared->second;
        } else {
            base = static_cast<std::uint32_t>(mesh.positions.size());
            build.vertex_base[key] = base;

            mesh.positions.insert(mesh.positions.end(), positions.begin(),
                                  positions.end());

            // If any primitive lacks normals (or texture coordinates),
            // the mesh gets none: the arrays must be parallel to the
            // positions
            if (!has_normal) {
                build.all_normals = false;
                mesh.normals.clear();
            } else if (build.all_normals) {
                mesh.normals.insert(mesh.normals.end(), normals.begin(),
                                    normals.end());
            }

            if (!has_texcoord) {
                build.all_texcoords = false;
                mesh.texcoords.clear();
            } else if (build.all_texcoords) {
                mesh.texcoords.insert(mesh.texcoords.end(),
                                      texcoords.begin(), texcoords.end());
            }

            m_total_vertices += position.count;
        }

        for (const std::uint32_t value : triangles) {
            mesh.indices.push_back(value + base);
        }

        mesh.triangle_materials.insert(mesh.triangle_materials.end(),
                                       triangles.size() / 3, material_index);

        m_total_indices += index_count;

        if (uses_default_material) {
            m_default_material_needed = true;
        }

        return PrimitiveResult::PRIMITIVE_OK;
    }

    // Loads glTF mesh `index` the first time it is asked for. result is
    // its index in Model::meshes, or -1 when nothing usable is in it
    bool load_mesh(std::size_t index, int& result) {
        result = m_mesh_map[index];

        if (result != kMeshUntried) {
            return true;
        }

        const JsonValue& js = m_doc["meshes"][index];
        const JsonValue& primitives = js["primitives"];

        if (!primitives.is_array() || primitives.size() == 0) {
            return fail("mesh " + std::to_string(index) +
                        " has no primitives");
        }

        MeshBuild build;
        std::size_t added = 0;

        for (std::size_t p = 0; p < primitives.size(); ++p) {
            const PrimitiveResult r =
                add_primitive(index, p, primitives[p], build);

            if (r == PrimitiveResult::PRIMITIVE_ERROR) {
                return false;
            }

            if (r == PrimitiveResult::PRIMITIVE_OK) {
                ++added;
            }
        }

        if (added == 0) {
            warn("mesh " + std::to_string(index) +
                 " has no usable triangles and is left out");
            m_mesh_map[index] = -1;
            result = -1;
            return true;
        }

        // glTF is Y-up, the model Z-up
        mesh_y_up_to_z_up(build.mesh);

        m_model.meshes.push_back(std::move(build.mesh));
        result = static_cast<int>(m_model.meshes.size()) - 1;
        m_mesh_map[index] = result;
        return true;
    }

    // ---- Nodes ------------------------------------------------------

    // Local transform of a node: either "matrix" (16 numbers, column by
    // column) or translation * rotation * scale
    bool node_local_matrix(std::size_t index, Mat4& out) {
        const JsonValue& node = m_doc["nodes"][index];
        const std::string where = "node " + std::to_string(index);

        out = mat4_identity();

        float values[16];
        bool present = false;

        if (!opt_float_array(node, "matrix", 16, values, present, where)) {
            return false;
        }

        if (present) {
            for (int c = 0; c < 4; ++c) {
                out.col[c] = {values[c * 4], values[c * 4 + 1],
                              values[c * 4 + 2], values[c * 4 + 3]};
            }

            return true;
        }

        float translation[3] = {0.0f, 0.0f, 0.0f};
        float rotation[4] = {0.0f, 0.0f, 0.0f, 1.0f};
        float scale[3] = {1.0f, 1.0f, 1.0f};

        if (!opt_float_array(node, "translation", 3, translation, present,
                             where) ||
            !opt_float_array(node, "rotation", 4, rotation, present, where) ||
            !opt_float_array(node, "scale", 3, scale, present, where)) {
            return false;
        }

        // T * R * S: scale the mesh first, then rotate, then move.
        // Scaling the columns of R by (sx, sy, sz) is R * S
        Mat4 m = rotation_from_quaternion(rotation[0], rotation[1],
                                          rotation[2], rotation[3]);

        m.col[0] = m.col[0] * scale[0];
        m.col[1] = m.col[1] * scale[1];
        m.col[2] = m.col[2] * scale[2];
        m.col[3] = {translation[0], translation[1], translation[2], 1.0f};

        out = m;
        return true;
    }

    // Walks the tree below `index`, multiplying the transforms down, and
    // adds an instance for every node with a mesh
    bool visit_node(std::size_t index, const Mat4& parent_world, int depth) {
        if (depth > kMaxNodeDepth) {
            return fail("the node hierarchy is deeper than " +
                        std::to_string(kMaxNodeDepth) + " levels");
        }

        if (++m_node_visits > kMaxNodeVisits) {
            return fail("the scene visits too many nodes (shared nodes?)");
        }

        if (m_on_stack[index]) {
            return fail("node " + std::to_string(index) +
                        " is its own ancestor (the node hierarchy has a "
                        "cycle)");
        }

        const JsonValue& node = m_doc["nodes"][index];
        const std::string where = "node " + std::to_string(index);

        Mat4 local;

        if (!node_local_matrix(index, local)) {
            return false;
        }

        const Mat4 world = parent_world * local;

        if (node.contains("mesh")) {
            std::uint64_t mesh_index = 0;
            const JsonValue& meshes = m_doc["meshes"];

            if (!to_uint(node["mesh"], mesh_index) ||
                mesh_index >= meshes.size()) {
                return fail(where + ": 'mesh' is not a valid mesh index "
                            "(the file has " +
                            std::to_string(meshes.size()) + " meshes)");
            }

            int mesh = -1;

            if (!load_mesh(static_cast<std::size_t>(mesh_index), mesh)) {
                return false;
            }

            if (mesh >= 0) {
                ModelInstance instance;
                instance.mesh = static_cast<std::uint32_t>(mesh);
                instance.transform = world;
                m_model.instances.push_back(instance);
            }
        }

        const JsonValue& children = node["children"];

        if (node.contains("children") && !children.is_array()) {
            return fail(where + ": 'children' must be an array");
        }

        m_on_stack[index] = true;

        for (std::size_t i = 0; i < children.size(); ++i) {
            std::uint64_t child = 0;

            if (!to_uint(children[i], child) || child >= m_on_stack.size()) {
                m_on_stack[index] = false;
                return fail(where + ": child " + std::to_string(i) +
                            " is not a valid node index");
            }

            if (!visit_node(static_cast<std::size_t>(child), world,
                            depth + 1)) {
                m_on_stack[index] = false;
                return false;
            }
        }

        m_on_stack[index] = false;
        return true;
    }

    // The nodes the scene starts from: the scene named by "scene", else
    // scene 0, else (no scenes) every node that is nobody's child
    bool find_root_nodes(std::vector<std::size_t>& roots) {
        const JsonValue& nodes = m_doc["nodes"];
        const JsonValue& scenes = m_doc["scenes"];

        roots.clear();

        if (m_doc.contains("scene") || scenes.size() > 0) {
            std::uint64_t scene_index = 0;

            if (!opt_uint(m_doc, "scene", 0, scene_index, "glTF")) {
                return false;
            }

            if (scene_index >= scenes.size()) {
                return fail("scene " + number_text(scene_index) +
                            " does not exist (the file has " +
                            std::to_string(scenes.size()) + ")");
            }

            const JsonValue& list = scenes[scene_index]["nodes"];

            for (std::size_t i = 0; i < list.size(); ++i) {
                std::uint64_t node = 0;

                if (!to_uint(list[i], node) || node >= nodes.size()) {
                    return fail("scene " + number_text(scene_index) +
                                " lists a node that does not exist");
                }

                roots.push_back(static_cast<std::size_t>(node));
            }

            return true;
        }

        std::vector<bool> has_parent(nodes.size(), false);

        for (std::size_t n = 0; n < nodes.size(); ++n) {
            const JsonValue& children = nodes[n]["children"];

            for (std::size_t i = 0; i < children.size(); ++i) {
                std::uint64_t child = 0;

                if (!to_uint(children[i], child) || child >= nodes.size()) {
                    return fail("node " + std::to_string(n) + ": child " +
                                std::to_string(i) +
                                " is not a valid node index");
                }

                has_parent[static_cast<std::size_t>(child)] = true;
            }
        }

        for (std::size_t n = 0; n < nodes.size(); ++n) {
            if (!has_parent[n]) {
                roots.push_back(n);
            }
        }

        return true;
    }

    // ---- Whole file -------------------------------------------------

    bool check_header() {
        const JsonValue& version = m_doc["asset"]["version"];

        if (!version.is_string()) {
            return fail("the file has no asset.version");
        }

        if (version.as_string().compare(0, 2, "2.") != 0) {
            return fail("unsupported glTF version '" + version.as_string() +
                        "' (only 2.x is supported)");
        }

        // Extensions the file cannot be shown correctly without
        const JsonValue& required = m_doc["extensionsRequired"];

        for (std::size_t i = 0; i < required.size(); ++i) {
            const std::string& name = required[i].as_string();

            if (name == "KHR_materials_emissive_strength") {
                continue;
            }

            if (name == "KHR_texture_transform") {
                warn_once("KHR_texture_transform",
                          "KHR_texture_transform is not supported; "
                          "textures are used untransformed");
                continue;
            }

            if (name.compare(0, 14, "KHR_materials_") == 0) {
                warn_once("ext:" + name,
                          "material extension " + name +
                          " is not supported and is ignored");
                continue;
            }

            return fail("the file requires unsupported extension " + name);
        }

        if (m_doc["animations"].size() > 0 || m_doc["skins"].size() > 0) {
            warn("animations and skins are not supported and are ignored");
        }

        return true;
    }

    bool read_all() {
        if (!check_header()) {
            return false;
        }

        m_buffers.resize(m_doc["buffers"].size());
        m_image_result.assign(m_doc["images"].size(), kImageUntried);
        m_mesh_map.assign(m_doc["meshes"].size(), kMeshUntried);
        m_on_stack.assign(m_doc["nodes"].size(), false);

        if (!parse_materials()) {
            return false;
        }

        std::vector<std::size_t> roots;

        if (!find_root_nodes(roots)) {
            return false;
        }

        for (const std::size_t root : roots) {
            if (!visit_node(root, mat4_identity(), 0)) {
                return false;
            }
        }

        if (m_model.instances.empty()) {
            return fail("no meshes in the scene");
        }

        // Every mesh gets the whole material list (see gltf_loader.h)
        std::vector<Material> materials = m_materials;

        if (m_default_material_needed) {
            Material fallback;   // our own defaults, named "default"
            materials.push_back(fallback);
        }

        for (Mesh& mesh : m_model.meshes) {
            mesh.materials = materials;
        }

        // Y-up -> Z-up for the placement as well. The meshes were
        // rotated by C; a point p in mesh space was placed at M * p, so
        // it must now be placed at C * M * p. With the stored vertices
        // equal to C * p that is (C * M * C^-1) * (C * p). C is a
        // rotation, so its inverse is its transpose
        const Mat4 c = y_up_to_z_up_matrix();
        const Mat4 c_inverse = mat4_transpose(c);

        for (ModelInstance& instance : m_model.instances) {
            instance.transform = c * instance.transform * c_inverse;
        }

        return true;
    }

    // ---- Data -------------------------------------------------------

    static constexpr int kImageUntried = -2;
    static constexpr int kMeshUntried = -2;

    struct LoadedBuffer {
        bool loaded = false;
        std::vector<std::uint8_t> owned;   // for buffers read from uri
        const std::uint8_t* data = nullptr;
        std::size_t size = 0;
    };

    const JsonValue& m_doc;
    ByteRange m_glb_bin;
    bool m_has_glb_bin;
    bool m_has_base_dir;
    std::filesystem::path m_base_dir;

    std::string m_error;
    std::set<std::string> m_warned;

    std::vector<LoadedBuffer> m_buffers;
    std::vector<Material> m_materials;
    std::vector<int> m_image_result;  // per glTF image
    std::vector<int> m_mesh_map;      // per glTF mesh
    std::vector<bool> m_on_stack;     // per node, during the walk

    bool m_default_material_needed = false;
    std::size_t m_node_visits = 0;
    std::uint64_t m_total_vertices = 0;
    std::uint64_t m_total_indices = 0;

    Model m_model;
};

// ---------------------------------------------------------------------
// .glb container
// ---------------------------------------------------------------------

// Splits a .glb into its JSON text and optional binary chunk
//
//   header: magic "glTF", version 2, total length          (12 bytes)
//   chunk:  length, type, data (length bytes, 4-aligned)   (repeated)
//
// The first chunk is JSON; the second, if any, is the binary buffer
bool split_glb(const std::uint8_t* data, std::size_t size,
               std::string_view& json, ByteRange& bin, bool& has_bin,
               std::string& error) {
    has_bin = false;
    bin = ByteRange{};

    if (size < 12) {
        error = "GLB file is too short for its header";
        return false;
    }

    if (load_u32(data) != kGlbMagic) {
        error = "not a GLB file (bad magic)";
        return false;
    }

    const std::uint32_t version = load_u32(data + 4);

    if (version != kGlbVersion) {
        error = "unsupported GLB version " + std::to_string(version) +
                " (only 2 is supported)";
        return false;
    }

    const std::uint64_t total = load_u32(data + 8);

    if (total > size) {
        error = "GLB header says " + std::to_string(total) +
                " bytes, but the file has only " + std::to_string(size);
        return false;
    }

    if (total < 12 + 8) {
        error = "GLB has no chunks";
        return false;
    }

    std::uint64_t offset = 12;
    int chunk_number = 0;
    bool have_json = false;

    while (offset < total) {
        if (total - offset < 8) {
            error = "GLB has a truncated chunk header at byte " +
                    std::to_string(offset);
            return false;
        }

        const std::uint64_t length = load_u32(data + offset);
        const std::uint32_t type = load_u32(data + offset + 4);
        offset += 8;

        if (length > total - offset) {
            error = "GLB chunk " + std::to_string(chunk_number) +
                    " (" + std::to_string(length) +
                    " bytes) runs past the end of the file";
            return false;
        }

        if (length % 4 != 0) {
            error = "GLB chunk " + std::to_string(chunk_number) +
                    " has a length that is not a multiple of 4";
            return false;
        }

        const std::uint8_t* chunk = data + offset;

        if (chunk_number == 0) {
            if (type != kChunkJson) {
                error = "the first GLB chunk is not JSON";
                return false;
            }

            std::size_t json_size = static_cast<std::size_t>(length);

            // Writers pad the JSON with spaces; some pad with zeros
            while (json_size > 0 && chunk[json_size - 1] == 0) {
                --json_size;
            }

            json = std::string_view(reinterpret_cast<const char*>(chunk),
                                    json_size);
            have_json = true;
        } else if (type == kChunkBin && !has_bin) {
            bin.data = chunk;
            bin.size = static_cast<std::size_t>(length);
            has_bin = true;
        }
        // Other chunk types are skipped, as the specification asks

        offset += length;
        ++chunk_number;
    }

    if (!have_json) {
        error = "GLB has no JSON chunk";
        return false;
    }

    return true;
}

} // namespace

bool parse_gltf(const std::uint8_t* data, std::size_t size,
                const std::string& base_dir, Model& out,
                std::string& error) {
    out = Model{};
    error.clear();

    std::string_view json_text;
    ByteRange bin;
    bool has_bin = false;

    if (size >= 4 && std::memcmp(data, "glTF", 4) == 0) {
        if (!split_glb(data, size, json_text, bin, has_bin, error)) {
            return false;
        }
    } else {
        json_text = std::string_view(reinterpret_cast<const char*>(data),
                                     size);
    }

    JsonValue doc;
    std::string json_error;

    if (!parse_json(json_text, doc, json_error)) {
        error = "JSON: " + json_error;
        return false;
    }

    if (!doc.is_object()) {
        error = "the glTF document must be a JSON object";
        return false;
    }

    GltfReader reader(doc, bin, has_bin, base_dir);

    return reader.run(out, error);
}

bool load_gltf(const std::string& path, Model& out, std::string& error) {
    out = Model{};
    error.clear();

    std::vector<std::uint8_t> bytes;

    if (!read_file_bytes(std::filesystem::path(path), bytes)) {
        error = "could not read file '" + path + "'";
        return false;
    }

    // Relative URIs are relative to the folder of the file. A bare file
    // name has an empty folder, which parse_gltf would read as "no
    // external files allowed": use the current folder then
    std::string base_dir =
        std::filesystem::path(path).parent_path().u8string();

    if (base_dir.empty()) {
        base_dir = ".";
    }

    return parse_gltf(bytes.data(), bytes.size(), base_dir, out, error);
}

} // namespace gfx
