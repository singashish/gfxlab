#include "core/pick.h"

#include "core/math_util.h"

#include <cmath>

namespace gfx {

namespace {

// "No hit yet" distance for the BVH query
constexpr float kFarAway = 1.0e30f;

} // namespace

bool make_pick_ray(const EditView& view, float px, float py, PickRay& out) {
    // The same guards as project_to_screen
    if (view.width <= 0.0f ||
        view.height <= 0.0f ||
        view.fov_y <= 0.0f ||
        view.fov_y >= kPi) {
        return false;
    }

    const float tan_half = std::tan(view.fov_y * 0.5f);
    const float aspect = view.width / view.height;

    if (tan_half <= 0.0f || aspect <= 0.0f) {
        return false;
    }

    // Pixel -> normalized device coordinates: x from -1 (left) to +1
    // (right), y from -1 (bottom) to +1 (top); pixel rows count downwards
    const float ndc_x = (px / view.width) * 2.0f - 1.0f;
    const float ndc_y = 1.0f - (py / view.height) * 2.0f;

    // At distance 1 in front of the camera the view plane is
    // tan_half * aspect wide (half-width) and tan_half high (half-height)
    const Vec3 dir = view.basis.forward
                   + view.basis.right * (ndc_x * tan_half * aspect)
                   + view.basis.up * (ndc_y * tan_half);

    out.origin = view.eye;
    out.dir = normalize(dir);
    return true;
}

const Bvh& PickIndex::bvh(const Model& model, std::uint32_t mesh) {
    static const Bvh kEmpty;

    if (m_bvhs.size() < model.meshes.size()) {
        m_bvhs.resize(model.meshes.size());
        m_built.resize(model.meshes.size(), false);
    }

    if (mesh >= model.meshes.size()) {
        return kEmpty;
    }

    if (!m_built[mesh]) {
        m_bvhs[mesh] = build_bvh(model.meshes[mesh]);
        m_built[mesh] = true;
    }

    return m_bvhs[mesh];
}

void PickIndex::clear() {
    m_bvhs.clear();
    m_built.clear();
}

bool pick_object(const Scene& scene, const Model& model, PickIndex& index,
                 const PickRay& ray, PickHit& hit) {
    bool found = false;
    float best_t = kFarAway;

    for (std::size_t i = 0; i < scene.objects.size(); ++i) {
        const SceneObject& object = scene.objects[i];

        if (object.hidden || object.mesh >= model.meshes.size()) {
            continue;
        }

        const Mesh& mesh = model.meshes[object.mesh];

        if (mesh.positions.empty()) {
            continue;
        }

        // World -> object space. A zero scale has no inverse: the object
        // is flat as a point or a line and cannot be clicked
        Mat4 to_object;

        if (!mat4_inverse(object.world_matrix(), to_object)) {
            continue;
        }

        const Vec3 origin = transform_point(to_object, ray.origin);

        // Not normalized on purpose: t then means the same in both spaces
        const Vec3 dir = transform_direction(to_object, ray.dir);

        BvhHit local;

        // Only hits nearer than the best one so far matter
        if (bvh_trace(index.bvh(model, object.mesh), mesh, origin, dir,
                      0.0f, best_t, local)) {
            best_t = local.t;
            hit.object = static_cast<int>(i);
            hit.triangle = local.triangle;
            hit.t = local.t;
            found = true;
        }
    }

    if (found) {
        hit.point = ray.origin + ray.dir * best_t;
    }

    return found;
}

} // namespace gfx
