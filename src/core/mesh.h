#pragma once

#include "core/material.h"
#include "core/vec2.h"
#include "core/vec3.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

struct Mesh {
    // Parallel arrays: vertex i uses positions[i] and normals[i]
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;          // empty if the file had no (complete) normals
    std::vector<std::uint32_t> indices; // 3 per triangle
    std::vector<std::uint32_t> edges; // 2 indices per unique edge

    // Texture coordinates, parallel to positions. Empty unless every
    // face corner in the file had one. Kept as written in the OBJ:
    // (0, 0) is the bottom-left corner of the image, v pointing up
    std::vector<Vec2> texcoords;

    // Surfaces. A loaded mesh always has at least one (a "default"
    // material when the file names none), and every triangle has an
    // index into this list
    std::vector<Material> materials;
    std::vector<std::uint32_t> triangle_materials; // 1 per triangle

    bool has_normals() const {
        return !normals.empty();
    }

    bool has_texcoords() const {
        return !texcoords.empty();
    }

    std::size_t triangle_count() const {
        return indices.size() / 3;
    }
};

struct Aabb {
    Vec3 min = {0.0f, 0.0f, 0.0f};
    Vec3 max = {0.0f, 0.0f, 0.0f};
};

// Smallest box containing every position. An empty mesh gives a zero box
Aabb compute_bounds(const Mesh& mesh);

// Move the mesh so the centre of its bounding box is at the origin
void mesh_center_at_origin(Mesh& mesh);

// Rotate a Y-up mesh into the Z-up world: (x, y, z) -> (x, -z, y)
void mesh_y_up_to_z_up(Mesh& mesh);

// Makes sure there is one unit-length normal per vertex
// Returns true if normals had to be generated
bool ensure_normals(Mesh& mesh);

// Builds deduplicated triangle edges if the mesh has no edge list
void ensure_edges(Mesh& mesh);

} // namespace gfx
