#include "core/mesh.h"

namespace gfx {

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

} // namespace gfx
