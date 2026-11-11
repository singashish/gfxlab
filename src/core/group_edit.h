#pragma once

/*
 * G / R / S on the whole selection.
 *
 * TransformEditor (transform_edit.h) turns mouse movement into a change of
 * ONE Transform. To move several objects at once, the editor is given a
 * made-up "pivot" transform: its position is the middle of the selected
 * objects (the median point, Blender's default pivot), its rotation is
 * zero and its scale is one. Whatever the editor does to that transform
 * (move it, turn it, scale it) is then done to every selected object as a
 * rigid group:
 *
 *   - grab: every object moves by the same distance;
 *   - rotate: the objects turn around the pivot, so their positions go
 *     around it and each object's own orientation turns by the same
 *     rotation;
 *   - scale: the distances to the pivot are scaled (along the world axes
 *     the editor scales), and every object's own scale is multiplied by
 *     the same factors, like the editor does for a single object.
 *
 * With one selected object the pivot is the object's own position, so
 * the result is exactly what the editor gives for that object alone.
 *
 * Every update starts from the transforms saved at the start, never from
 * the result of the previous frame (like the editor), so no error
 * accumulates and cancelling restores everything exactly.
 */

#include "core/scene.h"
#include "core/transform.h"
#include "core/vec3.h"

#include <cstddef>
#include <vector>

namespace gfx {

struct GroupEdit {
    std::vector<std::size_t> objects; // indices into Scene::objects
    std::vector<Transform> starts;    // their transforms when the edit began
    Vec3 pivot = {0.0f, 0.0f, 0.0f};  // the median of their positions
};

// Collects the selected, visible objects, saves their transforms and
// computes the pivot. Returns false (and leaves `edit` empty) when
// nothing is selected
bool group_edit_begin(const Scene& scene, GroupEdit& edit);

// The made-up transform the TransformEditor works on: position = pivot,
// no rotation, scale 1
Transform group_edit_pivot_transform(const GroupEdit& edit);

// Sets every object of the group from its start transform and the
// transform the editor produced for the pivot
void group_edit_apply(Scene& scene, const GroupEdit& edit,
                      const Transform& edited_pivot);

// Puts every object of the group back as it was at the start
void group_edit_restore(Scene& scene, const GroupEdit& edit);

} // namespace gfx
