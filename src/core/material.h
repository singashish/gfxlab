#pragma once

/*
 * Surface description shared by every renderer: a small PBR set modelled
 * on Blender's Principled BSDF, filled from an OBJ's .mtl file or a
 * glTF material.
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
    // For OBJ a full path (the loader resolves it relative to the .mtl
    // file); for glTF the image's file name or "embedded image N". Only
    // meant for logging: renderers use base_color_image
    std::string base_color_texture;

    // Textures, as indices into Model::images (-1 = none, or the image
    // could not be loaded). The decoded images live in the Model, not in
    // the material, so materials stay cheap to copy and many materials
    // can share one image.
    //
    // Only base_color_image is used by the rasterizers; the other three
    // are stored for the path tracer.
    int base_color_image = -1;

    // glTF channel meaning: G = roughness, B = metallic (R and A are
    // unused). Multiplied with the roughness and metallic factors
    int metallic_roughness_image = -1;

    // Tangent-space normal map (RGB = XYZ remapped from 0..1 to -1..1)
    int normal_image = -1;

    // Light given off, multiplied with the emission colour
    int emissive_image = -1;
};

// Roughness from an OBJ/MTL Phong exponent (Ns, 0..1000), the same
// mapping Blender's OBJ importer uses: Ns 1000 -> 0, Ns 0 -> 1
inline float roughness_from_phong_exponent(float ns) {
    const float clamped = ns < 0.0f ? 0.0f : (ns > 1000.0f ? 1000.0f : ns);
    return 1.0f - std::sqrt(clamped / 1000.0f);
}

} // namespace gfx
