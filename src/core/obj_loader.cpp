#include "core/obj_loader.h"
#include "core/log.h"

#include <algorithm>
#include <array>
#include <filesystem>
#include <charconv>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <iterator>
#include <set>

namespace gfx {
namespace {

struct ObjCorner {
    int position = -1;
    int texcoord = -1;
    int normal = -1;
};

void clear_mesh(Mesh& mesh) {
    mesh.positions.clear();
    mesh.normals.clear();
    mesh.indices.clear();
    mesh.edges.clear();
    mesh.texcoords.clear();
    mesh.materials.clear();
    mesh.triangle_materials.clear();
}

std::string_view trim_right(std::string_view text) {
    while (!text.empty() &&
           (text.back() == ' ' || text.back() == '\t')) {
        text.remove_suffix(1);
    }

    return text;
}

bool fail(Mesh& out, std::string& error, std::size_t line_number, 
                     const std::string& message) {
    clear_mesh(out);
    error = "line " + std::to_string(line_number) + ": " + message;
    return false;
}

bool is_space(char c) {
    return c == ' ' || c == '\t';
}

std::string_view trim_left(std::string_view text) {
    while (!text.empty() && is_space(text.front())) {
        text.remove_prefix(1);
    }

    return text;
}

bool next_token(std::string_view& text, std::string_view& token) {
    text = trim_left(text);

    if (text.empty()) {
        return false;
    }

    const std::size_t end = text.find_first_of(" \t");

    if (end == std::string_view::npos) {
        token = text;
        text = {};
    } else {
        token = text.substr(0, end);
        text.remove_prefix(end);
    }

    return true;
}

bool parse_int(std::string_view token, int& value) {
    if (token.empty()) {
        return false;
    }

    const char* first = token.data();
    const char* last = first + token.size();

    const auto result = std::from_chars(first, last, value);

    return result.ec == std::errc{} && result.ptr == last;
}

bool parse_float(std::string_view token, float& value) {
    if (token.empty()) {
        return false;
    }

    const char* first = token.data();
    const char* last = first + token.size();

    const auto result = std::from_chars(
        first,
        last,
        value,
        std::chars_format::general
    );

    return result.ec == std::errc{} && result.ptr == last;
}

bool read_float(std::string_view& text, float& value) {
    std::string_view token;

    if (!next_token(text, token)) {
        return false;
    }

    return parse_float(token, value);
}

bool read_vec3(std::string_view& text, Vec3& value) {
    return read_float(text, value.x)
        && read_float(text, value.y)
        && read_float(text, value.z);
}

bool has_extra_token(std::string_view text) {
    std::string_view token;
    return next_token(text, token);
}

bool resolve_index(int obj_index, std::size_t count, int& resolved) {
    if (obj_index == 0) {
        return false;
    }

    const long long index =
        obj_index > 0
            ? static_cast<long long>(obj_index) - 1
            : static_cast<long long>(count) + obj_index;

    if (index < 0 ||
        index >= static_cast<long long>(count) ||
        index > std::numeric_limits<int>::max()) {
        return false;
    }

    resolved = static_cast<int>(index);
    return true;
}

// Splits a face corner ("v", "v/vt", "v//vn" or "v/vt/vn") and resolves
// each index against the number of elements read so far
bool parse_corner(std::string_view token, std::size_t position_count,
                  std::size_t texcoord_count, std::size_t normal_count,
                  ObjCorner& corner, std::string& error) {
    std::string_view parts[3];
    int part_count = 0;

    while (part_count < 3) {
        const std::size_t slash = token.find('/');

        if (slash == std::string_view::npos) {
            parts[part_count++] = token;
            break;
        }

        parts[part_count++] = token.substr(0, slash);
        token.remove_prefix(slash + 1);

        if (part_count == 3) {
            error = "too many '/' in face corner";
            return false;
        }
    }

    const std::string_view position_token = parts[0];
    const std::string_view texcoord_token =
        part_count >= 2 ? parts[1] : std::string_view{};
    const std::string_view normal_token =
        part_count >= 3 ? parts[2] : std::string_view{};

    if (part_count == 3 && normal_token.empty()) {
        error = "missing normal index";
        return false;
    }

    int position_index = 0;

    if (!parse_int(position_token, position_index)) {
        error = "invalid face position index";
        return false;
    }

    if (!resolve_index(
            position_index,
            position_count,
            corner.position)) {
        error = "face index " + std::to_string(position_index)
              + " out of range";
        return false;
    }

    corner.texcoord = -1;

    if (!texcoord_token.empty()) {
        int texcoord_index = 0;

        if (!parse_int(texcoord_token, texcoord_index)) {
            error = "invalid texture coordinate index";
            return false;
        }

        if (texcoord_index == 0) {
            error = "texture coordinate index 0 is invalid";
            return false;
        }

        if (!resolve_index(
                texcoord_index,
                texcoord_count,
                corner.texcoord)) {
            error = "texture coordinate index " +
                    std::to_string(texcoord_index) + " out of range";
            return false;
        }
    } else if (part_count == 2) {
        error = "invalid texture coordinate index";
        return false;
    }

    corner.normal = -1;

    if (!normal_token.empty()) {
        int normal_index = 0;

        if (!parse_int(normal_token, normal_index)) {
            error = "invalid normal index";
            return false;
        }

        if (!resolve_index(
                normal_index,
                normal_count,
                corner.normal)) {
            error = "normal index " + std::to_string(normal_index)
                  + " out of range";
            return false;
        }
    }

    return true;
}

// Index of the material called name in materials, appending a new one
// with default values when it is not there yet
std::uint32_t find_or_add_material(std::vector<Material>& materials,
                                   std::string_view name) {
    for (std::size_t i = 0; i < materials.size(); ++i) {
        if (materials[i].name == name) {
            return static_cast<std::uint32_t>(i);
        }
    }

    Material material;
    material.name = std::string(name);
    materials.push_back(material);
    return static_cast<std::uint32_t>(materials.size() - 1);
}

} // namespace

bool parse_obj(std::string_view text, Mesh& out, std::string& error,
               std::vector<std::string>* material_libraries) {
    clear_mesh(out);
    error.clear();

    std::vector<Vec3> source_positions;
    std::vector<Vec2> source_texcoords;
    std::vector<Vec3> source_normals;

    // One output vertex per distinct (position, texcoord, normal) triple
    std::map<std::array<int, 3>, std::uint32_t> vertex_map;
    std::set<std::uint64_t> edge_keys;

    bool every_corner_has_normal = true;
    bool every_corner_has_texcoord = true;
    bool has_face = false;

    // Material of the faces being read; none until the first usemtl or
    // the first face (which then gets "default")
    bool has_current_material = false;
    std::uint32_t current_material = 0;

    std::size_t line_number = 0;

    while (!text.empty()) {
        ++line_number;

        const std::size_t newline = text.find('\n');

        std::string_view line;

        if (newline == std::string_view::npos) {
            line = text;
            text = {};
        } else {
            line = text.substr(0, newline);
            text.remove_prefix(newline + 1);
        }

        // Accept Windows CRLF line endings
        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        const std::size_t comment = line.find('#');

        if (comment != std::string_view::npos) {
            line = line.substr(0, comment);
        }

        line = trim_left(line);

        if (line.empty() || line.front() == '#') {
            continue;
        }

        std::string_view keyword;

        if (!next_token(line, keyword)) {
            continue;
        }

        if (keyword == "v") {
            Vec3 position{};

            if (!read_vec3(line, position)) {
                return fail(
                    out, error, line_number,
                    "invalid vertex position"
                );
            }

            // Anything after z is ignored: OBJ allows a 4th number (w), and many
            // exporters append vertex colours (r g b) here

            source_positions.push_back(position);
        } else if (keyword == "vn") {
            Vec3 normal{};

            if (!read_vec3(line, normal)) {
                return fail(
                    out, error, line_number,
                    "invalid vertex normal"
                );
            }

            if (has_extra_token(line)) {
                return fail(
                    out, error, line_number,
                    "too many normal components"
                );
            }

            source_normals.push_back(normal);
        } else if (keyword == "vt") {
            Vec2 texcoord{};

            // An optional third value (w) is allowed and ignored
            if (!read_float(line, texcoord.x) ||
                !read_float(line, texcoord.y)) {
                return fail(
                    out, error, line_number,
                    "invalid texture coordinate"
                );
            }

            source_texcoords.push_back(texcoord);
        } else if (keyword == "usemtl") {
            const std::string_view name = trim_right(trim_left(line));

            current_material = find_or_add_material(
                out.materials, name.empty() ? "default" : name);
            has_current_material = true;
        } else if (keyword == "mtllib") {
            // Several names may follow; each is one library file
            std::string_view library;

            while (next_token(line, library)) {
                if (material_libraries != nullptr) {
                    material_libraries->emplace_back(library);
                }
            }
        } else if (keyword == "f") {
            has_face = true;

            if (!has_current_material) {
                current_material =
                    find_or_add_material(out.materials, "default");
                has_current_material = true;
            }

            std::vector<ObjCorner> face;
            std::string_view token;

            while (next_token(line, token)) {
                ObjCorner corner;
                std::string corner_error;

                if (!parse_corner(
                        token,
                        source_positions.size(),
                        source_texcoords.size(),
                        source_normals.size(),
                        corner,
                        corner_error)) {
                    return fail(
                        out, error, line_number, corner_error
                    );
                }

                if (corner.normal < 0) {
                    every_corner_has_normal = false;
                }

                if (corner.texcoord < 0) {
                    every_corner_has_texcoord = false;
                }

                face.push_back(corner);
            }

            if (face.size() < 3) {
                return fail(
                    out, error, line_number,
                    "face needs at least 3 corners"
                );
            }

            std::vector<std::uint32_t> face_indices;
            face_indices.reserve(face.size());

            for (const ObjCorner& corner : face) {
                // A position used with two different texture
                // coordinates (a UV seam) becomes two vertices
                const std::array<int, 3> key{
                    corner.position,
                    corner.texcoord,
                    corner.normal
                };

                const auto found = vertex_map.find(key);

                if (found != vertex_map.end()) {
                    face_indices.push_back(found->second);
                    continue;
                }

                if (out.positions.size() >=
                    std::numeric_limits<std::uint32_t>::max()) {
                    return fail(
                        out, error, line_number,
                        "too many output vertices"
                    );
                }

                const std::uint32_t output_index =
                    static_cast<std::uint32_t>(out.positions.size());

                out.positions.push_back(
                    source_positions[
                        static_cast<std::size_t>(corner.position)
                    ]
                );

                if (corner.normal >= 0) {
                    out.normals.push_back(
                        source_normals[
                            static_cast<std::size_t>(corner.normal)
                        ]
                    );
                }

                if (corner.texcoord >= 0) {
                    out.texcoords.push_back(
                        source_texcoords[
                            static_cast<std::size_t>(corner.texcoord)
                        ]
                    );
                }

                vertex_map.emplace(key, output_index);
                face_indices.push_back(output_index);
            }

            // Preserve the original polygon boundary before triangulation.
            // Deduplicate using OBJ position indices, not output vertex indices,
            // because vertices may be duplicated when normals differ
            for (std::size_t i = 0; i < face.size(); ++i) {
                const std::size_t next = (i + 1) % face.size();

                const std::uint32_t va = face_indices[i];
                const std::uint32_t vb = face_indices[next];

                const std::uint32_t pa =
                    static_cast<std::uint32_t>(face[i].position);
                const std::uint32_t pb =
                    static_cast<std::uint32_t>(face[next].position);

                // Skip zero-length edges when a face repeats the same position
                if (pa == pb) {
                    continue;
                }

                const std::uint32_t lo = std::min(pa, pb);
                const std::uint32_t hi = std::max(pa, pb);

                const std::uint64_t key =
                    (static_cast<std::uint64_t>(lo) << 32) |
                    static_cast<std::uint64_t>(hi);

                if (edge_keys.insert(key).second) {
                    out.edges.push_back(va);
                    out.edges.push_back(vb);
                }
            }

            // Fan triangulation:
            // (0,1,2), (0,2,3), (0,3,4), ...
            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
                out.indices.push_back(face_indices[0]);
                out.indices.push_back(face_indices[i]);
                out.indices.push_back(face_indices[i + 1]);
                out.triangle_materials.push_back(current_material);
            }
        }

        // All other OBJ statements are intentionally ignored
    }

    if (!has_face) {
        return fail(out, error, 1, "no faces");
    }

    if (!every_corner_has_normal) {
        out.normals.clear();
    }

    if (!every_corner_has_texcoord) {
        out.texcoords.clear();
    }

    // A material named by usemtl but never used by a face is kept, so
    // indices stay in order of first mention
    return true;
}

bool parse_mtl(std::string_view text, const std::string& base_dir,
               std::vector<Material>& out, std::string& error) {
    error.clear();

    const std::size_t first_new = out.size();
    Material* current = nullptr;

    // Pr (roughness) wins over Ns, whichever comes first
    bool current_has_pr = false;

    std::size_t line_number = 0;

    const auto mtl_fail = [&](const std::string& message) {
        out.resize(first_new);
        error = "line " + std::to_string(line_number) + ": " + message;
        return false;
    };

    while (!text.empty()) {
        ++line_number;

        const std::size_t newline = text.find('\n');
        std::string_view line;

        if (newline == std::string_view::npos) {
            line = text;
            text = {};
        } else {
            line = text.substr(0, newline);
            text.remove_prefix(newline + 1);
        }

        if (!line.empty() && line.back() == '\r') {
            line.remove_suffix(1);
        }

        const std::size_t comment = line.find('#');

        if (comment != std::string_view::npos) {
            line = line.substr(0, comment);
        }

        std::string_view keyword;

        if (!next_token(line, keyword)) {
            continue;
        }

        if (keyword == "newmtl") {
            const std::string_view name = trim_right(trim_left(line));

            if (name.empty()) {
                return mtl_fail("newmtl without a name");
            }

            Material material;
            material.name = std::string(name);
            out.push_back(material);
            current = &out.back();
            current_has_pr = false;
            continue;
        }

        // Statements before the first newmtl have nothing to describe
        if (current == nullptr) {
            continue;
        }

        if (keyword == "Kd" || keyword == "Ke") {
            Vec3 color{};

            if (!read_vec3(line, color)) {
                return mtl_fail("invalid colour after " +
                                std::string(keyword));
            }

            (keyword == "Kd" ? current->base_color : current->emission) =
                color;
        } else if (keyword == "Ns") {
            float ns = 0.0f;

            if (!read_float(line, ns)) {
                return mtl_fail("invalid Ns");
            }

            if (!current_has_pr) {
                current->roughness = roughness_from_phong_exponent(ns);
            }
        } else if (keyword == "Pr" || keyword == "Pm") {
            float value = 0.0f;

            if (!read_float(line, value)) {
                return mtl_fail("invalid " + std::string(keyword));
            }

            value = std::clamp(value, 0.0f, 1.0f);

            if (keyword == "Pr") {
                current->roughness = value;
                current_has_pr = true;
            } else {
                current->metallic = value;
            }
        } else if (keyword == "map_Kd") {
            // Options come first ("-s 1 1 1", "-bm 0.5", ...); the rest
            // of the line is the file name, which may contain spaces
            line = trim_left(line);

            while (!line.empty() && line.front() == '-') {
                std::string_view option;
                next_token(line, option);

                int arguments = 1;

                if (option == "-s" || option == "-o" || option == "-t") {
                    arguments = 3;
                } else if (option == "-mm") {
                    arguments = 2;
                }

                for (int i = 0; i < arguments; ++i) {
                    std::string_view ignored;

                    if (!next_token(line, ignored)) {
                        return mtl_fail("incomplete map_Kd option");
                    }
                }

                line = trim_left(line);
            }

            const std::string_view file = trim_right(line);

            if (file.empty()) {
                return mtl_fail("map_Kd without a file name");
            }

            const std::filesystem::path texture_path(std::string{file});

            current->base_color_texture =
                texture_path.is_absolute() || base_dir.empty()
                    ? texture_path.string()
                    : (std::filesystem::path(base_dir) / texture_path)
                          .string();
        }

        // Everything else (Ka, Ks, Ni, d, illum, other maps) is ignored
    }

    return true;
}

bool load_obj(const std::string& path, Mesh& out, std::string& error) {
    clear_mesh(out);
    error.clear();

    std::ifstream file(path, std::ios::binary);

    if (!file) {
        error = "line 0: could not open file '" + path + "'";
        return false;
    }

    std::string contents(
        (std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>()
    );

    if (file.bad()) {
        error = "line 0: failed while reading file '" + path + "'";
        return false;
    }

    std::vector<std::string> libraries;

    if (!parse_obj(contents, out, error, &libraries)) {
        return false;
    }

    // Material libraries live next to the OBJ file
    const std::string obj_dir =
        std::filesystem::path(path).parent_path().string();

    std::vector<Material> library_materials;

    for (const std::string& library : libraries) {
        const std::filesystem::path library_path =
            std::filesystem::path(obj_dir) / library;

        std::ifstream mtl_file(library_path, std::ios::binary);

        if (!mtl_file) {
            LOG_WARN("material library not found: %s",
                     library_path.string().c_str());
            continue;
        }

        const std::string mtl_text(
            (std::istreambuf_iterator<char>(mtl_file)),
            std::istreambuf_iterator<char>()
        );

        std::string mtl_error;

        if (!parse_mtl(mtl_text, library_path.parent_path().string(),
                       library_materials, mtl_error)) {
            LOG_WARN("material library %s ignored: %s",
                     library_path.string().c_str(), mtl_error.c_str());
        }
    }

    // Fill in the materials the faces use. A name defined twice: the
    // first definition wins
    for (Material& material : out.materials) {
        const auto found = std::find_if(
            library_materials.begin(), library_materials.end(),
            [&](const Material& m) { return m.name == material.name; });

        if (found != library_materials.end()) {
            material = *found;
        } else if (!libraries.empty() || material.name != "default") {
            LOG_WARN("material '%s' not found; using defaults",
                     material.name.c_str());
        }
    }

    return true;
}

} // namespace gfx
