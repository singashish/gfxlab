#include "core/mesh.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
#include <limits>
#include <algorithm>
#include <set>

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

void mesh_center_at_origin(Mesh& mesh) {
    if (mesh.positions.empty()) {
        return;
    }

    const Aabb bounds = compute_bounds(mesh);
    const Vec3 center = (bounds.min + bounds.max) * 0.5f;

    for (Vec3& position : mesh.positions) {
        position = position - center;
    }
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

    // Without valid per-vertex normals, unweld the mesh so each
    // triangle can have its own flat normal
    const std::size_t old_vertex_count = mesh.positions.size();

    std::vector<Vec3> new_positions;
    std::vector<Vec3> new_normals;
    std::vector<std::uint32_t> new_indices;

    // first_copy[old_index] stores the first new vertex corresponding
    // to that old vertex. Multiple triangles may copy the same vertex
    const std::uint32_t kUnmapped =
        std::numeric_limits<std::uint32_t>::max();

    std::vector<std::uint32_t> first_copy(
        old_vertex_count,
        kUnmapped
    );

    const std::size_t triangle_count = mesh.indices.size() / 3;

    new_positions.reserve(triangle_count * 3);
    new_normals.reserve(triangle_count * 3);
    new_indices.reserve(triangle_count * 3);

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const std::uint32_t ia = mesh.indices[i];
        const std::uint32_t ib = mesh.indices[i + 1];
        const std::uint32_t ic = mesh.indices[i + 2];

        // Defensive validation before indexing the old positions
        if (ia >= old_vertex_count ||
            ib >= old_vertex_count ||
            ic >= old_vertex_count) {
            return false;
        }

        const Vec3 pa = mesh.positions[ia];
        const Vec3 pb = mesh.positions[ib];
        const Vec3 pc = mesh.positions[ic];

        const Vec3 normal =
            normalize_or_default(cross(pb - pa, pc - pa));

        const std::uint32_t base =
            static_cast<std::uint32_t>(new_positions.size());

        // Record the first copy of each original vertex
        if (first_copy[ia] == kUnmapped) {
            first_copy[ia] = base;
        }
        if (first_copy[ib] == kUnmapped) {
            first_copy[ib] = base + 1;
        }
        if (first_copy[ic] == kUnmapped) {
            first_copy[ic] = base + 2;
        }

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

    // Validate and remap the edge list before replacing the old arrays
    std::vector<std::uint32_t> new_edges;
    new_edges.reserve(mesh.edges.size());

    if (mesh.edges.size() % 2 != 0) {
        return false;
    }

    for (const std::uint32_t old_index : mesh.edges) {
        if (old_index >= old_vertex_count ||
            first_copy[old_index] == kUnmapped) {
            return false;
        }

        new_edges.push_back(first_copy[old_index]);
    }

    mesh.positions = std::move(new_positions);
    mesh.normals = std::move(new_normals);
    mesh.indices = std::move(new_indices);
    mesh.edges = std::move(new_edges);

    return true;
}

void ensure_edges(Mesh& mesh) {
    // An existing edge list came from the original polygon topology
    // Do not replace it with triangle edges, which would add diagonals
    if (!mesh.edges.empty()) {
        return;
    }

    std::set<std::uint64_t> edge_keys;

    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const std::uint32_t triangle[] = {
            mesh.indices[i],
            mesh.indices[i + 1],
            mesh.indices[i + 2]
        };

        for (std::size_t j = 0; j < 3; ++j) {
            const std::uint32_t a = triangle[j];
            const std::uint32_t b = triangle[(j + 1) % 3];

            if (a >= mesh.positions.size() ||
                b >= mesh.positions.size() ||
                a == b) {
                continue;
            }

            const std::uint32_t lo = std::min(a, b);
            const std::uint32_t hi = std::max(a, b);

            const std::uint64_t key =
                (static_cast<std::uint64_t>(lo) << 32) |
                static_cast<std::uint64_t>(hi);

            if (edge_keys.insert(key).second) {
                mesh.edges.push_back(a);
                mesh.edges.push_back(b);
            }
        }
    }
}

} // namespace gfx
