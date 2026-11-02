#pragma once

/*
 * Scene: the objects you can select and move.
 *
 * A Model (model.h) is geometry: meshes plus "draw mesh M with this
 * transform" instances. It has no idea which instances belong together
 * or how to move one of them. The Scene adds that:
 *
 *   - a Collection is one loaded file (Blender's "collection");
 *   - an Object is one instance of a mesh, with its own name, its own
 *     editable Transform, and flags for selected / hidden.
 *
 * Several files can be loaded: their meshes and images are appended to
 * ONE Model (so the renderers still get one set of meshes and
 * textures), and every file becomes a collection of objects.
 *
 * World matrix of an object = transform.model_matrix() * local.
 * Normally `local` is the identity and the whole placement of the
 * instance is in `transform` (position / rotation / scale), so G/R/S
 * turn and scale the object around its own origin. Only when an
 * instance's matrix cannot be written as position + rotation + scale
 * (it has shear, which glTF allows through non-uniformly scaled parent
 * nodes) the matrix is kept in `local` and `transform` starts as a plain
 * placement; such an object then turns around the collection's origin
 * instead of its own.
 */

#include "core/mat4.h"
#include "core/mesh.h"
#include "core/model.h"
#include "core/transform.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

struct SceneObject {
    std::string name;
    std::uint32_t collection = 0; // index into Scene::collections
    std::uint32_t mesh = 0;       // index into Model::meshes

    Mat4 local = mat4_identity(); // mesh space -> object space (see above)
    Transform transform;          // object space -> world, edited by G/R/S
    Transform initial;            // transform when loaded (Reset transform)

    bool selected = false;
    bool hidden = false;          // not drawn, not pickable

    Mat4 world_matrix() const {
        return transform.model_matrix() * local;
    }
};

struct SceneCollection {
    std::string name; // the file name
    Aabb bounds;      // of the file's model space, as loaded
};

struct Scene {
    std::vector<SceneCollection> collections;
    std::vector<SceneObject> objects;

    // The active object: the one the panel shows, and the one G/R/S work
    // on while it is selected. Like in Blender it is the last object that
    // was clicked, and it stays active (but unselected) after it is
    // deselected, until another object is clicked. -1 = none
    int active = -1;

    SceneObject* active_object() {
        return active >= 0 &&
               static_cast<std::size_t>(active) < objects.size()
                   ? &objects[static_cast<std::size_t>(active)]
                   : nullptr;
    }
};

// Writes m as position + rotation + scale (Transform). Returns false when
// that is not possible exactly (shear, a zero scale, a bottom row other
// than 0 0 0 1); out is then unspecified. A mirroring matrix (negative
// determinant) is accepted: the sign goes into scale.x
bool decompose_trs(const Mat4& m, Transform& out);

// Appends the model src (as load_model returns it: Z-up, centred on the
// origin) to `model` and adds a collection with one object per instance.
// The new collection stands on the floor (lowest point at z = 0). The
// first one is centred on x = 0, y = 0; each later one is placed to the
// right (+x) of everything already in the scene, with a gap. Objects
// are named after the glTF node, or after the file; equal names within a
// collection get ".001", ".002" ... like Blender.
// New objects start unselected, and the active object does not change.
// Returns the index of the first new object (equal to the number of
// objects before the call, also when src has no instances)
std::size_t scene_add_model(Scene& scene, Model& model, Model&& src,
                            const std::string& file_name);

// Box around every object in world space (hidden ones too, so the camera
// framing does not change when something is hidden). An empty scene
// gives a zero box
Aabb scene_bounds(const Scene& scene, const Model& model);

// Deselects everything, then selects object `index` and makes it
// active. An index outside the list only deselects and clears the
// active object
void scene_select_only(Scene& scene, int index);

// ---- Selection, the way Blender does it ----------------------------------

// A mouse click on `object` (an index into Scene::objects), or on empty
// space when the index is -1 or outside the list.
//
// Plain click (extend = false):
//   - on an object: only that object is selected, and it becomes active
//     (also when it was one of several selected objects);
//   - on empty space: everything is deselected; `active` is kept.
// Shift click (extend = true):
//   - on an unselected object: it is added to the selection and becomes
//     active;
//   - on a selected object that is not active: it stays selected and
//     becomes active;
//   - on the active selected object: it is deselected (it stays the
//     active object, now unselected);
//   - on empty space: nothing changes.
void scene_click_select(Scene& scene, int object, bool extend);

// Selects every visible object (A). Hidden objects are left alone, and
// `active` is kept
void scene_select_all(Scene& scene);

// Deselects everything (Alt+A). `active` is kept
void scene_deselect_all(Scene& scene);

// How many objects are selected
int scene_selected_count(const Scene& scene);

} // namespace gfx
