#pragma once

/*
 * Picking: which object is under the mouse cursor?
 *
 * Two parts:
 *
 *   1. make_pick_ray() turns a pixel into a ray in the world: it starts
 *      at the camera and goes through that pixel. It is the exact inverse
 *      of project_to_screen() (transform_edit.h), so a point that is
 *      drawn at pixel (x, y) lies on the ray made for (x, y).
 *
 *   2. pick_object() shoots such a ray at every visible object of the
 *      scene and returns the nearest hit. Each object has its own
 *      transform, but its triangles stay where they are in mesh space.
 *      So instead of moving all triangles into the world, the ray is
 *      moved into each object's own space (by the inverse of the object's
 *      world matrix) and tested there against that mesh's BVH
 *      (core/bvh.h). One BVH per mesh is enough for any number of
 *      objects, and moving, rotating or scaling an object never needs a
 *      rebuild.
 *
 * The distance t is the same number in both spaces: the world point
 * origin + t * dir becomes (inverse * origin) + t * (inverse * dir) in
 * object space, because the inverse is an affine map. That is why the
 * hits of different objects can be compared directly, even for scaled
 * objects, as long as dir is not normalized again in object space.
 *
 * Triangles are two-sided, so the back of a plane or an open mesh can be
 * picked as well.
 */

#include "core/bvh.h"
#include "core/mat4.h"
#include "core/model.h"
#include "core/scene.h"
#include "core/transform_edit.h"
#include "core/vec3.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

struct PickRay {
    Vec3 origin;
    Vec3 dir; // unit length, so a hit's t is a distance in world units
};

// The ray from the camera through pixel (px, py) of the viewport (pixels
// from the top-left corner, like the mouse). Returns false when the view
// is unusable (zero size, field of view outside (0, pi))
bool make_pick_ray(const EditView& view, float px, float py, PickRay& out);

// One BVH per mesh of a Model, built the first time the mesh is picked
// (so a model nobody clicks costs nothing) and kept. Meshes added to the
// Model later are handled by the same call
class PickIndex {
public:
    // The BVH of mesh `mesh` of `model`. A mesh number outside the model
    // gives an empty BVH. The reference stays valid until the next call
    const Bvh& bvh(const Model& model, std::uint32_t mesh);

    // Forget everything (use it when the meshes themselves change)
    void clear();

private:
    std::vector<Bvh> m_bvhs;
    std::vector<bool> m_built;
};

struct PickHit {
    int object = -1;             // index into Scene::objects, -1 = nothing
    std::uint32_t triangle = 0;  // triangle number in the object's mesh
    float t = 0.0f;              // distance from the ray origin (world)
    Vec3 point = {0.0f, 0.0f, 0.0f}; // the hit point (world)
};

// The nearest visible object along the ray. Hidden objects, objects with
// an empty or missing mesh and objects whose matrix cannot be inverted
// (a zero scale) are skipped. On a hit fills `hit` and returns true; on a
// miss returns false and leaves `hit` unchanged
bool pick_object(const Scene& scene, const Model& model, PickIndex& index,
                 const PickRay& ray, PickHit& hit);

} // namespace gfx
