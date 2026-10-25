#include "core/light_list.h"
#include "core/material_edit.h"
#include "core/mesh.h"
#include "check.h"

#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

using gfx::Material;
using gfx::MaterialParams;
using gfx::Vec3;

const float kNaN = std::numeric_limits<float>::quiet_NaN();
const float kInf = std::numeric_limits<float>::infinity();

bool same(Vec3 a, Vec3 b) {
    return a.x == b.x && a.y == b.y && a.z == b.z;
}

void test_params_of() {
    Material material;
    material.base_color = {0.1f, 0.2f, 0.3f};
    material.roughness = 0.25f;
    material.metallic = 0.75f;
    material.emission = {4.0f, 0.5f, 0.0f};

    const MaterialParams params = gfx::params_of(material);
    CHECK(same(params.base_color, material.base_color));
    CHECK(params.roughness == 0.25f);
    CHECK(params.metallic == 0.75f);
    CHECK(same(params.emission, material.emission));

    // Legal values pass sanitized() unchanged (a lamp may exceed 1)
    const MaterialParams clean = gfx::sanitized(params);
    CHECK(same(clean.base_color, params.base_color));
    CHECK(clean.roughness == 0.25f && clean.metallic == 0.75f);
    CHECK(same(clean.emission, params.emission));
}

void test_sanitized() {
    // The cases of the Step 7a material table test
    Material bad;
    bad.roughness = 1.7f;
    bad.metallic = -0.2f;
    bad.emission = {-1.0f, kNaN, kInf};

    MaterialParams clean = gfx::sanitized(gfx::params_of(bad));
    CHECK(clean.roughness == 1.0f);
    CHECK(clean.metallic == 0.0f);
    CHECK(same(clean.emission, Vec3{0.0f, 0.0f, 0.0f}));

    // NaN roughness -> 0.5, infinite metallic -> 0
    Material nan_material;
    nan_material.roughness = kNaN;
    nan_material.metallic = kInf;
    clean = gfx::sanitized(gfx::params_of(nan_material));
    CHECK(clean.roughness == 0.5f);
    CHECK(clean.metallic == 0.0f);

    // Base colour: clamped to 0..1, a bad channel gives the default grey
    MaterialParams params;
    params.base_color = {-0.5f, 1.5f, kNaN};
    clean = gfx::sanitized(params);
    CHECK(clean.base_color.x == 0.0f);
    CHECK(clean.base_color.y == 1.0f);
    CHECK(clean.base_color.z == 0.8f);

    // Emission has no upper limit
    params = MaterialParams();
    params.emission = {1000.0f, 0.5f, -0.0f};
    clean = gfx::sanitized(params);
    CHECK(clean.emission.x == 1000.0f && clean.emission.y == 0.5f);
    CHECK(clean.emission.z == 0.0f);
}

void test_split_combine() {
    Vec3 colour;
    float strength = -1.0f;

    // No emission: white and 0, so the colour survives strength 0
    gfx::split_emission(Vec3{0.0f, 0.0f, 0.0f}, colour, strength);
    CHECK(same(colour, Vec3{1.0f, 1.0f, 1.0f}) && strength == 0.0f);
    CHECK(same(gfx::combine_emission(colour, strength),
               Vec3{0.0f, 0.0f, 0.0f}));

    // Strength 8 with a coloured light
    gfx::split_emission(Vec3{8.0f, 4.0f, 2.0f}, colour, strength);
    CHECK(strength == 8.0f);
    CHECK(same(colour, Vec3{1.0f, 0.5f, 0.25f}));
    CHECK(same(gfx::combine_emission(colour, strength),
               Vec3{8.0f, 4.0f, 2.0f}));

    // Round trip through the editor state
    MaterialParams params;
    params.emission = {0.3f, 0.9f, 0.6f};
    const gfx::MaterialEditState state = gfx::edit_state_of(params);
    CHECK(approx(state.emission_strength, 0.9));
    CHECK(approx(state.emission_color.y, 1.0));

    const MaterialParams back = gfx::params_of(state);
    CHECK(approx(back.emission.x, 0.3, 1e-6));
    CHECK(approx(back.emission.y, 0.9, 1e-6));
    CHECK(approx(back.emission.z, 0.6, 1e-6));
    CHECK(same(back.base_color, params.base_color));

    // Bad values: no light, never a NaN
    gfx::split_emission(Vec3{-1.0f, kNaN, kInf}, colour, strength);
    CHECK(strength == 0.0f);
    CHECK(gfx::combine_emission(Vec3{1.0f, 1.0f, 1.0f}, kNaN).x == 0.0f);
    CHECK(gfx::combine_emission(Vec3{1.0f, 1.0f, 1.0f}, -3.0f).y == 0.0f);

    // Strength 0 keeps the colour in the state
    gfx::MaterialEditState edit = gfx::edit_state_of(params);
    edit.emission_strength = 0.0f;
    CHECK(same(gfx::params_of(edit).emission, Vec3{0.0f, 0.0f, 0.0f}));
    CHECK(approx(edit.emission_color.x, 0.3333333, 1e-5));
}

gfx::Mesh make_mesh(std::size_t triangles) {
    gfx::Mesh mesh;

    for (std::size_t t = 0; t < triangles; ++t) {
        const float z = static_cast<float>(t);
        const std::uint32_t base = static_cast<std::uint32_t>(t * 3);
        mesh.positions.push_back(Vec3{0.0f, 0.0f, z});
        mesh.positions.push_back(Vec3{1.0f, 0.0f, z});
        mesh.positions.push_back(Vec3{0.0f, 1.0f, z});
        mesh.indices.push_back(base);
        mesh.indices.push_back(base + 1);
        mesh.indices.push_back(base + 2);
    }

    return mesh;
}

Material named(const char* name, float roughness) {
    Material material;
    material.name = name;
    material.roughness = roughness;
    return material;
}

void test_entries() {
    // Two meshes that each carry the whole file's list a, b, c, d, but
    // use different parts of it
    std::vector<gfx::Mesh> meshes;
    const std::vector<Material> list = {
        named("a", 0.1f), named("b", 0.2f), named("c", 0.3f),
        named("d", 0.4f)};

    gfx::Mesh first = make_mesh(3);
    first.materials = list;
    first.triangle_materials = {2, 0, 2};  // c and a
    meshes.push_back(first);

    gfx::Mesh second = make_mesh(2);
    second.materials = list;
    second.triangle_materials = {2, 3};    // c and d
    meshes.push_back(second);

    // No list at all: counts as all material 0
    gfx::Mesh third = make_mesh(1);
    third.materials = list;
    meshes.push_back(third);

    // A number outside the list is the fallback: not editable
    gfx::Mesh fourth = make_mesh(1);
    fourth.materials = list;
    fourth.triangle_materials = {9};
    meshes.push_back(fourth);

    // A wrong-length list counts as material 0 too; a mesh without
    // materials adds nothing
    gfx::Mesh fifth = make_mesh(2);
    fifth.materials = list;
    fifth.triangle_materials = {3};
    meshes.push_back(fifth);
    meshes.push_back(make_mesh(1));

    const std::vector<gfx::MaterialEntry> entries =
        gfx::build_material_entries(meshes);

    // a (0), c (2), d (3); b is used by nothing
    CHECK(entries.size() == 3);

    if (entries.size() == 3) {
        CHECK(entries[0].index == 0 && entries[0].name == "a");
        CHECK(entries[1].index == 2 && entries[1].name == "c");
        CHECK(entries[2].index == 3 && entries[2].name == "d");

        // a: mesh 0, mesh 2 (no list), mesh 4 (wrong length)
        CHECK(entries[0].targets.size() == 3);
        CHECK(entries[0].targets[0].mesh == 0 &&
              entries[0].targets[1].mesh == 2 &&
              entries[0].targets[2].mesh == 4);
        CHECK(entries[0].targets[0].index == 0);

        // c: meshes 0 and 1 with the same index
        CHECK(entries[1].targets.size() == 2);
        CHECK(entries[1].targets[1].mesh == 1 &&
              entries[1].targets[1].index == 2);

        // d: mesh 1 only
        CHECK(entries[2].targets.size() == 1);
        CHECK(approx(entries[2].original.roughness, 0.4, 1e-6));
        CHECK(!entries[2].has_texture);
    }

    // The same index with a different name is a different material
    gfx::Mesh other = make_mesh(1);
    other.materials = {named("x", 0.9f)};
    other.triangle_materials = {0};
    meshes.push_back(other);
    CHECK(gfx::build_material_entries(meshes).size() == 4);

    // A texture is noted
    gfx::Mesh textured = make_mesh(1);
    textured.materials = {named("t", 0.5f)};
    textured.materials[0].base_color_image = 0;
    const std::vector<gfx::Mesh> only_textured = {textured};
    const std::vector<gfx::MaterialEntry> t_entries =
        gfx::build_material_entries(only_textured);
    CHECK(t_entries.size() == 1 && t_entries[0].has_texture);

    CHECK(gfx::build_material_entries({}).empty());
}

void test_light_list_from_areas() {
    // The list built from stored areas is the list built from the mesh
    const gfx::Mesh mesh = make_mesh(4);
    const std::vector<float> luminance = {2.0f, 0.0f, 0.5f, 1.0f};

    const gfx::LightList direct =
        gfx::build_light_list(mesh.positions, mesh.indices, luminance);
    const gfx::LightList stored = gfx::build_light_list_from_areas(
        gfx::triangle_areas(mesh.positions, mesh.indices), luminance);

    CHECK(direct.triangles == stored.triangles);
    CHECK(direct.cdf == stored.cdf);
    CHECK(direct.triangles.size() == 3);
}

} // namespace

int main() {
    test_params_of();
    test_sanitized();
    test_split_combine();
    test_entries();
    test_light_list_from_areas();

    if (g_failures == 0) {
        std::printf("All material edit tests passed\n");
        return 0;
    }

    std::printf("%d material edit test failure(s)\n", g_failures);
    return 1;
}
