#pragma once

/*
 * The host side of the area-light sampling (see light_sampling.h): from a
 * mesh and the emission of each triangle, the list of lamp triangles and
 * their cumulative weights, which CudaMeshStore uploads with the mesh.
 * Plain C++ (std::vector), used by the CUDA mesh store and by the tests.
 */

#include "core/light_sampling.h"
#include "core/vec3.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

struct LightList {
    std::vector<std::uint32_t> triangles;  // ascending triangle numbers
    std::vector<float> cdf;                // inclusive cumulative weights
};

// Object-space area of every triangle. CudaMeshStore keeps these so a
// material edit can rebuild the light list without the mesh
inline std::vector<float> triangle_areas(
                const std::vector<Vec3>& positions,
                const std::vector<std::uint32_t>& indices) {

    std::vector<float> areas(indices.size() / 3, 0.0f);

    for (std::size_t t = 0; t < areas.size(); ++t) {
        float area = 0.0f;
        Vec3 normal;

        triangle_area_normal(positions[indices[t * 3 + 0]],
                             positions[indices[t * 3 + 1]],
                             positions[indices[t * 3 + 2]], area, normal);
        areas[t] = area;
    }

    return areas;
}

// emission_luminance_per_triangle[t] is the luminance of the emission of
// triangle t's material (0 for no emission). A triangle is a lamp when
// that is above 0 and its object-space area is above 0; its weight is
// area * luminance
inline LightList build_light_list_from_areas(
                const std::vector<float>& areas,
                const std::vector<float>& emission_luminance_per_triangle) {

    LightList list;
    float running = 0.0f;

    for (std::size_t t = 0; t < areas.size() &&
                            t < emission_luminance_per_triangle.size(); ++t) {
        const float luminance = emission_luminance_per_triangle[t];

        if (!(luminance > 0.0f) || !(areas[t] > 0.0f)) {
            continue;
        }

        running += areas[t] * luminance;
        list.triangles.push_back(static_cast<std::uint32_t>(t));
        list.cdf.push_back(running);
    }

    return list;
}

// The same from the mesh itself
inline LightList build_light_list(
                const std::vector<Vec3>& positions,
                const std::vector<std::uint32_t>& indices,
                const std::vector<float>& emission_luminance_per_triangle) {

    return build_light_list_from_areas(
        triangle_areas(positions, indices), emission_luminance_per_triangle);
}

} // namespace gfx
