#pragma once

/*
 * Sampling the emissive triangles of the scene ("area lights") for the
 * path tracer. Header-only and GFX_HD like path_math.h and brdf.h, so
 * the CPU tests run the very same code as the GPU.
 *
 * Why. Until now a lamp was found only by luck: a bounce ray that happened
 * to hit it picked up its emission. A small lamp is hit by very few
 * rays, so the picture was full of speckle. The cure is next-event
 * estimation, as for the sun: at every surface point also pick a random
 * point on a lamp and ask whether it is visible. Combining that with the
 * bounce rays by multiple importance sampling (MIS, explained in
 * brdf.h) gives the best of both: the aimed ray wins for small lamps, the
 * bounce rays win for large lamps close to the surface.
 *
 * The densities. A random point on a triangle of area A, picked with
 * probability P among all the lamp triangles, has the density
 *     pdf_area = P / A       (per unit of area)
 * but the path tracer compares it with the bounce density, which is per
 * unit of SOLID ANGLE seen from the surface point. The two are related by
 * how much area a small cone of directions cuts out of the lamp:
 *     pdf_solid_angle = pdf_area * dist^2 / cos_light,
 * where dist is the distance to the lamp point and cos_light the cosine
 * of the angle between the lamp's normal and the direction back to the
 * surface (a lamp seen at a grazing angle or from far away covers little
 * solid angle for its area, so the same area sample is "denser").
 *
 * Choosing a lamp. Every lamp triangle gets a weight, its area times the
 * luminance of its emission factor (bigger and brighter lamps are
 * picked more often, the sensible guess for how much light each one
 * contributes). The triangles of one mesh keep a cumulative sum of
 * those weights (a CDF), and a random number in [0, 1) picks a triangle
 * by binary search: the probability of a triangle is its weight over the
 * total. Each mesh instance additionally carries one number, its total
 * weight scaled by the instance's transform, to pick the instance first.
 */

#include "core/mat4.h"
#include "core/vec3.h"

#include <cmath>
#include <cstdint>

namespace gfx {

// A light direction that grazes the lamp's plane, or a lamp point that is
// practically at the surface point, is not sampled (both would give huge
// or undefined densities). The same limits are used when a bounce ray
// hits a lamp, so the two strategies keep covering the same cases
constexpr float kMinLightCosine = 1.0e-5f;
constexpr float kMinLightDistance = 1.0e-4f;

// The shadow ray towards a lamp point stops this fraction of the distance
// short of it, so it does not hit the lamp triangle itself
constexpr float kLightShadowShortening = 1.0e-3f;

// Rec. 709 luminance of a linear colour
GFX_HD inline float emission_luminance(Vec3 e) {
    return 0.2126f * e.x + 0.7152f * e.y + 0.0722f * e.z;
}

// How much a model matrix scales areas: |det of its 3x3 part| ^ (2/3).
// Exact for a uniform scale (and unaffected by rotation and
// translation); for a non-uniform scale it is the area change of an
// "average" triangle. It only decides how often an instance is picked
// (a probability guess); the density used in the estimates is always
// computed from the real triangle in world space, so a wrong guess costs
// noise, never correctness
GFX_HD inline float area_scale(const Mat4& m) {
    const Vec3 c0{m.col[0].x, m.col[0].y, m.col[0].z};
    const Vec3 c1{m.col[1].x, m.col[1].y, m.col[1].z};
    const Vec3 c2{m.col[2].x, m.col[2].y, m.col[2].z};
    const float s = cbrtf(fabsf(dot(c0, cross(c1, c2))));

    return s * s;
}

// Area and unit normal of the triangle (p0, p1, p2). A degenerate triangle
// gets area 0 and a zero normal
GFX_HD inline void triangle_area_normal(
                Vec3 p0,
                Vec3 p1,
                Vec3 p2,
                float& area,
                Vec3& normal) {

    const Vec3 c = cross(p1 - p0, p2 - p0);
    const float len = length(c);

    area = 0.5f * len;
    normal = len > 0.0f ? c * (1.0f / len) : Vec3{0.0f, 0.0f, 0.0f};
}

// A point chosen uniformly by area inside a triangle, as barycentric
// weights (b0, b1, b2) of its three corners (they add up to 1). The
// point is b0 p0 + b1 p1 + b2 p2. u1 and u2 are uniform in [0, 1).
// Why the square root: the share of the area within a distance
// fraction r of corner 0 grows like r^2, so r = sqrt(u1) gives a
// uniform spread; u2 then slides the point along the line at that
// fraction
GFX_HD inline void sample_triangle_barycentric(
                float u1,
                float u2,
                float& b0,
                float& b1,
                float& b2) {

    const float r = sqrtf(u1);

    b0 = 1.0f - r;
    b1 = u2 * r;
    b2 = 1.0f - b0 - b1;
}

// The solid-angle density from the area density (see the notes at the
// top). 0 when the lamp is seen edge-on or behind (cos_light <= 0)
GFX_HD inline float area_to_solid_angle_pdf(
                float pdf_area,
                float dist,
                float cos_light) {

    if (!(cos_light > 0.0f)) {
        return 0.0f;
    }

    return pdf_area * dist * dist / cos_light;
}

// Picks an entry from an inclusive cumulative sum (cdf[i] = weight of
// entries 0..i, so cdf[count - 1] is the total). u in [0, 1): the result
// is the first index whose cdf is at least u * total, found by binary
// search, so entry i is picked with probability weight_i / total.
// probability receives that probability. An empty list (or a total of 0)
// returns 0 with probability 0
GFX_HD inline unsigned int pick_from_cdf(
                const float* cdf,
                unsigned int count,
                float u,
                float& probability) {

    probability = 0.0f;

    if (count == 0u || !(cdf[count - 1u] > 0.0f)) {
        return 0u;
    }

    const float total = cdf[count - 1u];
    const float target = u * total;

    unsigned int lo = 0u;
    unsigned int hi = count - 1u;  // cdf[hi] >= target always holds

    while (lo < hi) {
        const unsigned int mid = lo + (hi - lo) / 2u;

        if (cdf[mid] >= target) {
            hi = mid;
        } else {
            lo = mid + 1u;
        }
    }

    const float before = lo > 0u ? cdf[lo - 1u] : 0.0f;

    probability = (cdf[lo] - before) / total;

    return lo;
}

// The probability pick_from_cdf gives entry i
GFX_HD inline float cdf_probability(
                const float* cdf,
                unsigned int count,
                unsigned int i) {

    const float before = i > 0u ? cdf[i - 1u] : 0.0f;

    return (cdf[i] - before) / cdf[count - 1u];
}

// The position of `value` in an ascending list of triangle numbers, or
// count when it is not in the list
GFX_HD inline unsigned int find_in_sorted(
                const std::uint32_t* sorted,
                unsigned int count,
                std::uint32_t value) {

    unsigned int lo = 0u;
    unsigned int hi = count;

    while (lo < hi) {
        const unsigned int mid = lo + (hi - lo) / 2u;

        if (sorted[mid] < value) {
            lo = mid + 1u;
        } else {
            hi = mid;
        }
    }

    return (lo < count && sorted[lo] == value) ? lo : count;
}

} // namespace gfx
