#pragma once

/*
 * Ray tests shared by the CPU (BVH builder tests) and the GPU (the CUDA
 * ray tracer). Everything is GFX_HD and header-only, like raster_math.h.
 */

#include "core/common.h"
#include "core/vec3.h"

#include <cmath>

namespace gfx {

// Component-wise reciprocal of a ray direction. A zero component gives
// +/-infinity, which the slab test below handles correctly
GFX_HD inline Vec3 ray_inverse_direction(Vec3 dir) {
    return {1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z};
}

// Slab test: does the ray hit the box within [0, t_max]?
// t_entry receives the distance at which the ray enters the box
// (0 when the origin is inside). inv_dir is ray_inverse_direction(dir)
GFX_HD inline bool ray_hits_box(Vec3 origin,
                                Vec3 inv_dir,
                                Vec3 box_min,
                                Vec3 box_max,
                                float t_max,
                                float& t_entry) {
    const float tx0 = (box_min.x - origin.x) * inv_dir.x;
    const float tx1 = (box_max.x - origin.x) * inv_dir.x;
    const float ty0 = (box_min.y - origin.y) * inv_dir.y;
    const float ty1 = (box_max.y - origin.y) * inv_dir.y;
    const float tz0 = (box_min.z - origin.z) * inv_dir.z;
    const float tz1 = (box_max.z - origin.z) * inv_dir.z;

    const float t_near = fmaxf(
        fmaxf(fminf(tx0, tx1), fminf(ty0, ty1)),
        fmaxf(fminf(tz0, tz1), 0.0f));

    const float t_far = fminf(
        fminf(fmaxf(tx0, tx1), fmaxf(ty0, ty1)),
        fminf(fmaxf(tz0, tz1), t_max));

    t_entry = t_near;
    return t_near <= t_far;
}

// Moller-Trumbore ray/triangle intersection, two-sided.
// On a hit inside (t_min, t_max) it returns true with the distance t and
// the barycentric weights of b and c (u, v); the weight of a is 1 - u - v
GFX_HD inline bool ray_hits_triangle(Vec3 origin,
                                     Vec3 dir,
                                     Vec3 a,
                                     Vec3 b,
                                     Vec3 c,
                                     float t_min,
                                     float t_max,
                                     float& t,
                                     float& u,
                                     float& v) {
    const Vec3 e1 = b - a;
    const Vec3 e2 = c - a;

    const Vec3 p = cross(dir, e2);
    const float det = dot(e1, p);

    // The ray is parallel to the triangle's plane (or the triangle is
    // degenerate)
    if (fabsf(det) < 1.0e-20f) {
        return false;
    }

    const float inv_det = 1.0f / det;
    const Vec3 s = origin - a;

    u = dot(s, p) * inv_det;

    if (u < 0.0f || u > 1.0f) {
        return false;
    }

    const Vec3 q = cross(s, e1);
    v = dot(dir, q) * inv_det;

    if (v < 0.0f || u + v > 1.0f) {
        return false;
    }

    t = dot(e2, q) * inv_det;
    return t > t_min && t < t_max;
}

} // namespace gfx
