#include "core/obj_loader.h"

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

namespace gfx {
namespace {

struct ObjCorner {
    int position = -1;
    int normal = -1;
};

void clear_mesh(Mesh& mesh) {
    mesh.positions.clear();
    mesh.normals.clear();
    mesh.indices.clear();
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

bool parse_corner(std::string_view token, std::size_t position_count,
                  std::size_t normal_count, ObjCorner& corner,
                  std::string& error) {
    const std::size_t first_slash = token.find('/');

    std::string_view position_token;
    std::string_view normal_token;

    if (first_slash == std::string_view::npos) {
        position_token = token;
    } else {
        position_token = token.substr(0, first_slash);

        const std::size_t second_slash =
            token.find('/', first_slash + 1);

        if (second_slash == std::string_view::npos) {
            // v/vt
            const std::string_view texcoord_token =
                token.substr(first_slash + 1);

            int ignored_texcoord = 0;

            if (!parse_int(texcoord_token, ignored_texcoord)) {
                error = "invalid texture coordinate index";
                return false;
            }

            if (ignored_texcoord == 0) {
                error = "texture coordinate index 0 is invalid";
                return false;
            }
        } else {
            // v//vn or v/vt/vn
            const std::string_view texcoord_token =
                token.substr(
                    first_slash + 1,
                    second_slash - first_slash - 1
                );

            normal_token = token.substr(second_slash + 1);

            if (second_slash != first_slash + 1) {
                int ignored_texcoord = 0;

                if (!parse_int(texcoord_token, ignored_texcoord)) {
                    error = "invalid texture coordinate index";
                    return false;
                }

                if (ignored_texcoord == 0) {
                    error = "texture coordinate index 0 is invalid";
                    return false;
                }
            }

            if (normal_token.empty()) {
                error = "missing normal index";
                return false;
            }
        }
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

} // namespace

bool parse_obj(std::string_view text, Mesh& out, std::string& error) {
    clear_mesh(out);
    error.clear();

    std::vector<Vec3> source_positions;
    std::vector<Vec3> source_normals;

    std::map<std::pair<int, int>, std::uint32_t> vertex_map;

    bool every_corner_has_normal = true;
    bool has_face = false;

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
        } else if (keyword == "f") {
            has_face = true;

            std::vector<ObjCorner> face;
            std::string_view token;

            while (next_token(line, token)) {
                ObjCorner corner;
                std::string corner_error;

                if (!parse_corner(
                        token,
                        source_positions.size(),
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
                const std::pair<int, int> key{
                    corner.position,
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

                vertex_map.emplace(key, output_index);
                face_indices.push_back(output_index);
            }

            // Fan triangulation:
            // (0,1,2), (0,2,3), (0,3,4), ...
            for (std::size_t i = 1; i + 1 < face_indices.size(); ++i) {
                out.indices.push_back(face_indices[0]);
                out.indices.push_back(face_indices[i]);
                out.indices.push_back(face_indices[i + 1]);
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

    return parse_obj(contents, out, error);
}

} // namespace gfx
