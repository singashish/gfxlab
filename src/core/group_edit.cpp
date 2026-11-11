#include "core/group_edit.h"

namespace gfx {

bool group_edit_begin(const Scene& scene, GroupEdit& edit) {
    edit = GroupEdit{};

    Vec3 sum = {0.0f, 0.0f, 0.0f};

    for (std::size_t i = 0; i < scene.objects.size(); ++i) {
        const SceneObject& object = scene.objects[i];

        if (!object.selected || object.hidden) {
            continue;
        }

        edit.objects.push_back(i);
        edit.starts.push_back(object.transform);
        sum = sum + object.transform.position;
    }

    if (edit.objects.empty()) {
        return false;
    }

    edit.pivot = sum * (1.0f / static_cast<float>(edit.objects.size()));
    return true;
}

Transform group_edit_pivot_transform(const GroupEdit& edit) {
    Transform pivot;
    pivot.position = edit.pivot;
    return pivot;
}

void group_edit_apply(Scene& scene, const GroupEdit& edit,
                      const Transform& edited_pivot) {
    const Mat4 rotation = edited_pivot.rotation_matrix();
    const Vec3 s = edited_pivot.scale;

    for (std::size_t k = 0; k < edit.objects.size(); ++k) {
        if (edit.objects[k] >= scene.objects.size()) {
            continue;
        }

        const Transform& start = edit.starts[k];
        Transform& out = scene.objects[edit.objects[k]].transform;

        // Where the object's origin goes: its offset from the pivot is
        // scaled along the world axes, then turned, then moved with the
        // pivot
        const Vec3 offset = start.position - edit.pivot;
        const Vec3 scaled = {offset.x * s.x, offset.y * s.y, offset.z * s.z};

        out.position = edited_pivot.position + transform_direction(rotation, scaled);
        out.rotation = euler_from_matrix(rotation * start.rotation_matrix());
        out.scale = {start.scale.x * s.x, start.scale.y * s.y,
                     start.scale.z * s.z};
    }
}

void group_edit_restore(Scene& scene, const GroupEdit& edit) {
    for (std::size_t k = 0; k < edit.objects.size(); ++k) {
        if (edit.objects[k] < scene.objects.size()) {
            scene.objects[edit.objects[k]].transform = edit.starts[k];
        }
    }
}

} // namespace gfx
