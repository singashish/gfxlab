#pragma once

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

    bool has_normals() const {
        return !normals.empty();
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

} // namespace gfx
