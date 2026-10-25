#pragma once

/*
 * What the panel's material editor works with: the few material values
 * that can be changed live (no textures), their clamping rules, and the
 * list of materials a model offers for editing.
 *
 * sanitized() is the one place that decides which values are legal; the
 * CUDA mesh store uses it for the materials it uploads, so a value from a
 * file and a value from the editor are treated the same.
 */

#include "core/material.h"
#include "core/mesh.h"
#include "core/vec3.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

struct MaterialParams {
    Vec3 base_color = {0.8f, 0.8f, 0.8f};  // multiplied with the texture
    float roughness = 0.5f;
    float metallic = 0.0f;
    Vec3 emission = {0.0f, 0.0f, 0.0f};    // linear radiance, may exceed 1
};

inline MaterialParams params_of(const Material& material) {
    MaterialParams params;
    params.base_color = material.base_color;
    params.roughness = material.roughness;
    params.metallic = material.metallic;
    params.emission = material.emission;
    return params;
}

namespace material_edit_detail {

// A value that must lie in 0..1. NaN and infinity (a damaged file) give
// `fallback`; other values are clamped
inline float unit_value(float value, float fallback) {
    if (!std::isfinite(value)) {
        return fallback;
    }

    return value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
}

// One component of an emission colour: finite and not negative, with no
// upper limit. A bad value gives 0 (no light) rather than a lamp of
// unknown strength
inline float emission_value(float value) {
    return std::isfinite(value) && value > 0.0f ? value : 0.0f;
}

} // namespace material_edit_detail

// The legal version of params: base colour 0..1 per channel (a bad value
// gives the default grey 0.8), roughness 0..1 (bad: 0.5), metallic 0..1
// (bad: 0), emission finite and >= 0
inline MaterialParams sanitized(const MaterialParams& params) {
    using namespace material_edit_detail;

    MaterialParams out;
    out.base_color = Vec3{unit_value(params.base_color.x, 0.8f),
                          unit_value(params.base_color.y, 0.8f),
                          unit_value(params.base_color.z, 0.8f)};
    out.roughness = unit_value(params.roughness, 0.5f);
    out.metallic = unit_value(params.metallic, 0.0f);
    out.emission = Vec3{emission_value(params.emission.x),
                        emission_value(params.emission.y),
                        emission_value(params.emission.z)};
    return out;
}

// An emission as a colour and a strength, so the strength can go to 0
// without losing the colour: strength = the largest component, colour =
// emission / strength (white for no emission)
inline void split_emission(Vec3 emission, Vec3& colour, float& strength) {
    using material_edit_detail::emission_value;

    const Vec3 e{emission_value(emission.x), emission_value(emission.y),
                 emission_value(emission.z)};
    const float largest = std::max(e.x, std::max(e.y, e.z));

    if (!(largest > 0.0f)) {
        colour = Vec3{1.0f, 1.0f, 1.0f};
        strength = 0.0f;
        return;
    }

    colour = Vec3{e.x / largest, e.y / largest, e.z / largest};
    strength = largest;
}

inline Vec3 combine_emission(Vec3 colour, float strength) {
    using material_edit_detail::emission_value;

    const float k = emission_value(strength);

    return Vec3{emission_value(colour.x) * k, emission_value(colour.y) * k,
                emission_value(colour.z) * k};
}

// What the editor changes for one material: the values as sliders see
// them (emission as colour and strength)
struct MaterialEditState {
    Vec3 base_color = {0.8f, 0.8f, 0.8f};
    float roughness = 0.5f;
    float metallic = 0.0f;
    Vec3 emission_color = {1.0f, 1.0f, 1.0f};
    float emission_strength = 0.0f;
};

inline MaterialEditState edit_state_of(const MaterialParams& params) {
    MaterialEditState state;
    state.base_color = params.base_color;
    state.roughness = params.roughness;
    state.metallic = params.metallic;
    split_emission(params.emission, state.emission_color,
                   state.emission_strength);
    return state;
}

inline MaterialParams params_of(const MaterialEditState& state) {
    MaterialParams params;
    params.base_color = state.base_color;
    params.roughness = state.roughness;
    params.metallic = state.metallic;
    params.emission =
        combine_emission(state.emission_color, state.emission_strength);
    return params;
}

// One material of the model as the editor lists it. A glTF file gives
// every mesh a copy of its whole material list, so an entry stands for
// "material k with this name" and keeps every (mesh, k) it must update
struct MaterialTarget {
    std::size_t mesh = 0;  // index into the model's meshes
    int index = 0;         // index into that mesh's materials
};

struct MaterialEntry {
    int index = 0;
    std::string name;
    bool has_texture = false;  // any of its textures (still applied)
    MaterialParams original;   // as loaded, for the reset buttons
    std::vector<MaterialTarget> targets;
};

// The materials used by at least one triangle of some mesh, ordered by
// index (then by first use). A mesh without triangle_materials, or with
// the wrong number of them, counts as all material 0 (the rule of
// group_triangles_by_material); a number outside the material list is
// the renderer's fallback material and is not editable
inline std::vector<MaterialEntry> build_material_entries(
                const std::vector<Mesh>& meshes) {

    std::vector<MaterialEntry> entries;

    for (std::size_t m = 0; m < meshes.size(); ++m) {
        const Mesh& mesh = meshes[m];
        const std::size_t material_count = mesh.materials.size();
        const std::size_t triangle_count = mesh.indices.size() / 3;
        std::vector<bool> used(material_count, false);

        if (mesh.triangle_materials.size() == triangle_count) {
            for (std::uint32_t number : mesh.triangle_materials) {
                if (static_cast<std::size_t>(number) < material_count) {
                    used[number] = true;
                }
            }
        } else if (material_count > 0 && triangle_count > 0) {
            used[0] = true;
        }

        for (std::size_t k = 0; k < material_count; ++k) {
            if (!used[k]) {
                continue;
            }

            const Material& material = mesh.materials[k];
            const int index = static_cast<int>(k);
            MaterialEntry* found = nullptr;

            for (MaterialEntry& entry : entries) {
                if (entry.index == index && entry.name == material.name) {
                    found = &entry;
                    break;
                }
            }

            if (found == nullptr) {
                entries.emplace_back();
                found = &entries.back();
                found->index = index;
                found->name = material.name;
                found->has_texture = material.base_color_image >= 0 ||
                                     material.metallic_roughness_image >= 0 ||
                                     material.emissive_image >= 0;
                found->original = params_of(material);
            }

            found->targets.push_back(MaterialTarget{m, index});
        }
    }

    std::stable_sort(entries.begin(), entries.end(),
                     [](const MaterialEntry& a, const MaterialEntry& b) {
                         return a.index < b.index;
                     });
    return entries;
}

} // namespace gfx
