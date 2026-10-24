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

// emission_luminance_per_triangle[t] is the luminance of the emission of
// triangle t's material (0 for no emission). A triangle is a lamp when
// that is above 0 and its object-space area is above 0; its weight is
// area * luminance
inline LightList build_light_list(
                const std::vector<Vec3>& positions,
                const std::vector<std::uint32_t>& indices,
                const std::vector<float>& emission_luminance_per_triangle) {

    LightList list;
    const std::size_t triangle_count = indices.size() / 3;
    float running = 0.0f;

    for (std::size_t t = 0; t < triangle_count &&
                            t < emission_luminance_per_triangle.size(); ++t) {
        const float luminance = emission_luminance_per_triangle[t];

        if (!(luminance > 0.0f)) {
            continue;
        }

        float area = 0.0f;
        Vec3 normal;

        triangle_area_normal(positions[indices[t * 3 + 0]],
                             positions[indices[t * 3 + 1]],
                             positions[indices[t * 3 + 2]], area, normal);

        if (!(area > 0.0f)) {
            continue;
        }

        running += area * luminance;
        list.triangles.push_back(static_cast<std::uint32_t>(t));
        list.cdf.push_back(running);
    }

    return list;
}

} // namespace gfx
