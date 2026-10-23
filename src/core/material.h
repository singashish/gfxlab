#pragma once

/*
 * Surface description shared by every renderer: a small PBR set modelled
 * on Blender's Principled BSDF, filled from an OBJ's .mtl file.
 *
 * The rasterizers use base_color (and its texture); the path tracer uses
 * all of it: a diffuse and a glossy (GGX) layer mixed by metallic, with
 * roughness controlling the glossy highlight, plus emission.
 */

#include "core/vec3.h"

#include <cmath>
#include <string>

namespace gfx {

struct Material {
    std::string name = "default";

    // Linear RGB, 0..1. Multiplied with the texture when there is one
    Vec3 base_color = {0.8f, 0.8f, 0.8f};

    // 0 = mirror-like, 1 = fully rough
    float roughness = 0.5f;

    // 0 = dielectric (plastic, wood, ...), 1 = metal
    float metallic = 0.0f;

    // Light given off by the surface, linear RGB (0 = none)
    Vec3 emission = {0.0f, 0.0f, 0.0f};

    // Image whose colour replaces base_color (multiplied), or empty.
    // A full path: the loader resolves it relative to the .mtl file
    std::string base_color_texture;
};

// Roughness from an OBJ/MTL Phong exponent (Ns, 0..1000), the same
// mapping Blender's OBJ importer uses: Ns 1000 -> 0, Ns 0 -> 1
inline float roughness_from_phong_exponent(float ns) {
    const float clamped = ns < 0.0f ? 0.0f : (ns > 1000.0f ? 1000.0f : ns);
    return 1.0f - std::sqrt(clamped / 1000.0f);
}

} // namespace gfx
