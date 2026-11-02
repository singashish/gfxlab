#include "core/scene.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <utility>

namespace gfx {

namespace {

// Space between two collections, as a fraction of the larger one's
// longest side
constexpr float kCollectionGap = 0.25f;

Vec3 column_xyz(const Mat4& m, int column) {
    return {m.col[column].x, m.col[column].y, m.col[column].z};
}

// True when two matrices agree everywhere, relative to the size of the
// entries (the decomposition goes through sin/cos/atan2)
bool matrices_match(const Mat4& a, const Mat4& b) {
    for (int c = 0; c < 4; ++c) {
        const float av[4] = {a.col[c].x, a.col[c].y, a.col[c].z, a.col[c].w};
        const float bv[4] = {b.col[c].x, b.col[c].y, b.col[c].z, b.col[c].w};

        for (int i = 0; i < 4; ++i) {
            if (std::fabs(av[i] - bv[i]) > 1.0e-4f * (1.0f + std::fabs(av[i]))) {
                return false;
            }
        }
    }

    return true;
}

} // namespace

bool decompose_trs(const Mat4& m, Transform& out) {
    // A position + rotation + scale matrix has 0 0 0 1 as its bottom row
    for (int c = 0; c < 3; ++c) {
        if (m.col[c].w != 0.0f) {
            return false;
        }
    }

    if (m.col[3].w != 1.0f) {
        return false;
    }

    Vec3 axis[3] = {column_xyz(m, 0), column_xyz(m, 1), column_xyz(m, 2)};
    Vec3 scale = {length(axis[0]), length(axis[1]), length(axis[2])};

    if (!(scale.x > 1.0e-8f) || !(scale.y > 1.0e-8f) ||
        !(scale.z > 1.0e-8f)) {
        return false;
    }

    axis[0] = axis[0] * (1.0f / scale.x);
    axis[1] = axis[1] * (1.0f / scale.y);
    axis[2] = axis[2] * (1.0f / scale.z);

    // A mirror (left-handed axes) cannot be a rotation: flip the x axis
    // and remember it as a negative x scale
    if (dot(cross(axis[0], axis[1]), axis[2]) < 0.0f) {
        axis[0] = axis[0] * -1.0f;
        scale.x = -scale.x;
    }

    const Mat4 rotation = {{
        {axis[0].x, axis[0].y, axis[0].z, 0.0f},
        {axis[1].x, axis[1].y, axis[1].z, 0.0f},
        {axis[2].x, axis[2].y, axis[2].z, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};

    Transform result;
    result.position = column_xyz(m, 3);
    result.rotation = euler_from_matrix(rotation);
    result.scale = scale;

    // Shear shows up here: the rebuilt matrix then differs from m
    if (!matrices_match(result.model_matrix(), m)) {
        return false;
    }

    out = result;
    return true;
}

Aabb scene_bounds(const Scene& scene, const Model& model) {
    std::vector<Aabb> mesh_boxes;
    mesh_boxes.reserve(model.meshes.size());

    for (const Mesh& mesh : model.meshes) {
        mesh_boxes.push_back(compute_bounds(mesh));
    }

    Aabb bounds{};
    bool first = true;

    for (const SceneObject& object : scene.objects) {
        if (object.mesh >= model.meshes.size() ||
            model.meshes[object.mesh].positions.empty()) {
            continue;
        }

        const Aabb& box = mesh_boxes[object.mesh];
        const Mat4 world = object.world_matrix();

        for (int corner = 0; corner < 8; ++corner) {
            const Vec3 local = {
                (corner & 1) ? box.max.x : box.min.x,
                (corner & 2) ? box.max.y : box.min.y,
                (corner & 4) ? box.max.z : box.min.z
            };

            const Vec3 p = transform_point(world, local);

            if (first) {
                bounds.min = p;
                bounds.max = p;
                first = false;
                continue;
            }

            bounds.min.x = std::min(bounds.min.x, p.x);
            bounds.min.y = std::min(bounds.min.y, p.y);
            bounds.min.z = std::min(bounds.min.z, p.z);
            bounds.max.x = std::max(bounds.max.x, p.x);
            bounds.max.y = std::max(bounds.max.y, p.y);
            bounds.max.z = std::max(bounds.max.z, p.z);
        }
    }

    return bounds;
}

std::size_t scene_add_model(Scene& scene, Model& model, Model&& src,
                            const std::string& file_name) {
    const std::size_t first_object = scene.objects.size();

    // Measured before src is moved away
    const Aabb box = model_bounds(src);
    const Vec3 size = box.max - box.min;
    const Vec3 center = (box.min + box.max) * 0.5f;

    // Where the collection goes: standing on the floor, the first one
    // centred on the origin, the next ones to the right of the scene
    Vec3 offset = {-center.x, -center.y, -box.min.z};

    if (!scene.objects.empty()) {
        const Aabb existing = scene_bounds(scene, model);
        const float gap =
            kCollectionGap * std::max(size.x, std::max(size.y, size.z));

        offset.x = existing.max.x + gap - box.min.x;
    }

    const std::uint32_t collection =
        static_cast<std::uint32_t>(scene.collections.size());

    scene.collections.push_back({file_name, box});

    const std::string stem =
        std::filesystem::path(file_name).stem().string();

    const std::size_t first_instance = append_model(model, std::move(src));

    // How many objects of this name the collection has so far
    std::map<std::string, int> name_count;

    for (std::size_t i = first_instance; i < model.instances.size(); ++i) {
        const ModelInstance& instance = model.instances[i];

        SceneObject object;
        object.collection = collection;
        object.mesh = instance.mesh;

        const std::string base = instance.name.empty() ? stem : instance.name;
        const int seen = name_count[base]++;

        if (seen == 0) {
            object.name = base;
        } else {
            char suffix[16];
            std::snprintf(suffix, sizeof(suffix), ".%03d", seen);
            object.name = base + suffix;
        }

        const Mat4 placed = mat4_translate(offset) * instance.transform;

        if (!decompose_trs(placed, object.transform)) {
            // Sheared placement: keep the matrix, move by the offset only
            object.local = instance.transform;
            object.transform = Transform{};
            object.transform.position = offset;
        }

        object.initial = object.transform;
        scene.objects.push_back(std::move(object));
    }

    return first_object;
}

void scene_select_only(Scene& scene, int index) {
    for (SceneObject& object : scene.objects) {
        object.selected = false;
    }

    scene.active = -1;

    if (index >= 0 && static_cast<std::size_t>(index) < scene.objects.size()) {
        scene.objects[static_cast<std::size_t>(index)].selected = true;
        scene.active = index;
    }
}

void scene_deselect_all(Scene& scene) {
    for (SceneObject& object : scene.objects) {
        object.selected = false;
    }
}

void scene_select_all(Scene& scene) {
    for (SceneObject& object : scene.objects) {
        if (!object.hidden) {
            object.selected = true;
        }
    }
}

int scene_selected_count(const Scene& scene) {
    int count = 0;

    for (const SceneObject& object : scene.objects) {
        if (object.selected) {
            ++count;
        }
    }

    return count;
}

void scene_click_select(Scene& scene, int object, bool extend) {
    const bool on_object =
        object >= 0 && static_cast<std::size_t>(object) < scene.objects.size();

    if (!extend) {
        if (on_object) {
            scene_select_only(scene, object);
        } else {
            scene_deselect_all(scene);
        }

        return;
    }

    if (!on_object) {
        return;
    }

    SceneObject& clicked = scene.objects[static_cast<std::size_t>(object)];

    if (!clicked.selected) {
        clicked.selected = true;
        scene.active = object;
    } else if (scene.active != object) {
        scene.active = object;
    } else {
        clicked.selected = false;
    }
}

} // namespace gfx
