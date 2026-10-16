#include "core/mesh.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace gfx {

namespace {

constexpr float kNormalEpsilon = 1.0e-30f;

Vec3 normalize_or_default(const Vec3& v) {
    const float len_squared =
        v.x * v.x + v.y * v.y + v.z * v.z;

    // The threshold applies to squared length. The negated comparison
    // also catches NaN, which would otherwise propagate into the normal
    if (!(len_squared >= kNormalEpsilon)) {
        return Vec3{0.0f, 0.0f, 1.0f};
    }

    const float inv_len = 1.0f / std::sqrt(len_squared);

    return Vec3{
        v.x * inv_len,
        v.y * inv_len,
        v.z * inv_len
    };
}

} // namespace

Aabb compute_bounds(const Mesh& mesh) {
    Aabb bounds{};

    if (mesh.positions.empty()) {
        return bounds;
    }

    bounds.min = mesh.positions[0];
    bounds.max = mesh.positions[0];

    for (const Vec3& p : mesh.positions) {
        if (p.x < bounds.min.x) bounds.min.x = p.x;
        if (p.y < bounds.min.y) bounds.min.y = p.y;
        if (p.z < bounds.min.z) bounds.min.z = p.z;

        if (p.x > bounds.max.x) bounds.max.x = p.x;
        if (p.y > bounds.max.y) bounds.max.y = p.y;
        if (p.z > bounds.max.z) bounds.max.z = p.z;
    }

    return bounds;
}

void mesh_y_up_to_z_up(Mesh& mesh) {
    for (Vec3& p : mesh.positions) {
        const float old_y = p.y;
        p.y = -p.z;
        p.z = old_y;
    }

    for (Vec3& n : mesh.normals) {
        const float old_y = n.y;
        n.y = -n.z;
        n.z = old_y;
    }
}

bool ensure_normals(Mesh& mesh) {
    // Existing normals are valid only when there is one per vertex
    // Normalize them and replace zero-length normals with the fallback
    if (!mesh.normals.empty() &&
        mesh.normals.size() == mesh.positions.size()) {
        for (Vec3& normal : mesh.normals) {
            normal = normalize_or_default(normal);
        }

        return false;
    }

    // Without valid per-vertex normals, unweld the mesh so each triangle
    // can have its own flat normal
    std::vector<Vec3> new_positions;
    std::vector<Vec3> new_normals;
    std::vector<std::uint32_t> new_indices;

    const std::size_t triangle_count = mesh.indices.size() / 3;

    new_positions.reserve(triangle_count * 3);
    new_normals.reserve(triangle_count * 3);
    new_indices.reserve(triangle_count * 3);

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const std::uint32_t ia = mesh.indices[i];
        const std::uint32_t ib = mesh.indices[i + 1];
        const std::uint32_t ic = mesh.indices[i + 2];

        // Defensive check: the OBJ loader already validates indices
        // This prevents out-of-bounds access if another caller supplies
        // an invalid mesh. False here does not mean normals were present
        if (ia >= mesh.positions.size() ||
            ib >= mesh.positions.size() ||
            ic >= mesh.positions.size()) {
            return false;
        }

        const Vec3 pa = mesh.positions[ia];
        const Vec3 pb = mesh.positions[ib];
        const Vec3 pc = mesh.positions[ic];

        const Vec3 normal =
            normalize_or_default(cross(pb - pa, pc - pa));

        const std::uint32_t base =
            static_cast<std::uint32_t>(new_positions.size());

        new_positions.push_back(pa);
        new_positions.push_back(pb);
        new_positions.push_back(pc);

        new_normals.push_back(normal);
        new_normals.push_back(normal);
        new_normals.push_back(normal);

        new_indices.push_back(base);
        new_indices.push_back(base + 1);
        new_indices.push_back(base + 2);
    }

    mesh.positions = std::move(new_positions);
    mesh.normals = std::move(new_normals);
    mesh.indices = std::move(new_indices);

    return true;
}

} // namespace gfx
