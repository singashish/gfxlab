#include "core/obj_loader.h"
#include "check.h"

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool approx_vec(const gfx::Vec3& a, const gfx::Vec3& b, double eps = 1e-5) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

const char* kTriangle = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
)OBJ";

const char* kCubeWithNormals = R"OBJ(
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
vn  0  0 -1
vn  0  0  1
vn  0 -1  0
vn  1  0  0
vn  0  1  0
vn -1  0  0
f 1//1 4//1 3//1 2//1
f 5//2 6//2 7//2 8//2
f 1//3 2//3 6//3 5//3
f 2//4 3//4 7//4 6//4
f 3//5 4//5 8//5 7//5
f 4//6 1//6 5//6 8//6
)OBJ";

const char* kCubeWithoutNormals = R"OBJ(
v -1 -1 -1
v  1 -1 -1
v  1  1 -1
v -1  1 -1
v -1 -1  1
v  1 -1  1
v  1  1  1
v -1  1  1
f 1 4 3 2
f 5 6 7 8
f 1 2 6 5
f 2 3 7 6
f 3 4 8 7
f 4 1 5 8
)OBJ";

const std::string cube_path =
    std::string(GFX_ASSETS_DIR) + "/cube.obj";

const std::string cube_no_normals_path =
    std::string(GFX_ASSETS_DIR) + "/cube_no_normals.obj";

bool parse(std::string_view text, gfx::Mesh& mesh, std::string& error) {
    return gfx::parse_obj(text, mesh, error);
}

void test_triangle() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(kTriangle, mesh, error));
    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.indices.size() == 3);
    CHECK(mesh.triangle_count() == 1);
    CHECK(mesh.edges.size() == 6);
    CHECK(!mesh.has_normals());

    CHECK(mesh.indices[0] == 0);
    CHECK(mesh.indices[1] == 1);
    CHECK(mesh.indices[2] == 2);
}

void test_quad_triangulation() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
f 1 2 3 4
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.positions.size() == 4);
    CHECK(mesh.indices.size() == 6);
    CHECK(mesh.edges.size() == 8);

    const std::uint32_t expected[] = {0, 1, 2, 0, 2, 3};

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(mesh.indices[i] == expected[i]);
    }
}

void test_two_quads_share_edge() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 2 0 0
v 0 1 0
v 1 1 0
v 2 1 0
f 1 2 5 4
f 2 3 6 5
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));

    // Two quads have 8 boundary-edge occurrences,
    // but their shared edge must only be stored once
    CHECK(mesh.edges.size() == 14);
}

void test_negative_indices() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
f -3 -2 -1
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.indices.size() == 3);

    CHECK(mesh.indices[0] == 0);
    CHECK(mesh.indices[1] == 1);
    CHECK(mesh.indices[2] == 2);
}

void test_position_texcoord_normal() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
vt 1 0
vt 0 1
vn 0 0 1
f 1/1/1 2/2/1 3/3/1
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.has_normals());
    CHECK(mesh.normals.size() == mesh.positions.size());

    for (const gfx::Vec3& normal : mesh.normals) {
        CHECK(approx_vec(normal, {0, 0, 1}));
    }
}

void test_position_double_slash_normal() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
vn 0 0 1
f 1//1 2//1 3//1
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.has_normals());
    CHECK(mesh.normals.size() == 3);

    for (const gfx::Vec3& normal : mesh.normals) {
        CHECK(approx_vec(normal, {0, 0, 1}));
    }
}

void test_mixed_normals() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vn 0 0 1
f 1//1 2//1 3//1
f 2 4 3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(!mesh.has_normals());
    CHECK(mesh.normals.empty());
}

void test_ignored_keywords_and_crlf() {
    const char* source = R"OBJ(
# comment
o x
g y
s off
usemtl m
mtllib a.mtl

v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 3
)OBJ";

    std::string crlf;

    for (const char* p = source; *p != '\0'; ++p) {
        if (*p == '\n') {
            crlf += "\r\n";
        } else {
            crlf += *p;
        }
    }

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(crlf, mesh, error));
    CHECK(mesh.triangle_count() == 1);
    CHECK(mesh.positions.size() == 3);
}

void test_float_formats_and_w() {
    const char* obj = R"OBJ(
v 1e-3 .5 -2 1.0
v 1 0 0
v 0 1 0
f 1 2 3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.positions.size() == 3);
    CHECK(approx_vec(mesh.positions[0], {0.001f, 0.5f, -2.0f}));
}

void test_out_of_range_face_index() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
f 1 2 9
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(!parse(obj, mesh, error));
    CHECK(error.find("line") != std::string::npos);
    CHECK(error.find("face index 9 out of range") != std::string::npos);
    CHECK(mesh.positions.empty());
    CHECK(mesh.normals.empty());
    CHECK(mesh.indices.empty());
}

void expect_failure(std::string_view obj) {
    gfx::Mesh mesh;
    mesh.positions.push_back({99, 99, 99});
    mesh.indices.push_back(0);

    std::string error;

    CHECK(!parse(obj, mesh, error));
    CHECK(!error.empty());
    CHECK(error.find("line ") == 0);
    CHECK(mesh.positions.empty());
    CHECK(mesh.normals.empty());
    CHECK(mesh.indices.empty());
    CHECK(mesh.edges.empty());
}

void test_invalid_inputs() {
    expect_failure(R"OBJ(
v 0 0 0
v 1 0 0
f 1 2
)OBJ");

    expect_failure(R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
f 0 1 2
)OBJ");

    expect_failure(R"OBJ(
v 1 abc 3
v 1 0 0
v 0 1 0
f 1 2 3
)OBJ");

    expect_failure(R"OBJ(
# no faces
v 0 0 0
v 1 0 0
v 0 1 0
)OBJ");
}

void test_missing_file() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(!gfx::load_obj(
        "this/does/not/exist.obj",
        mesh,
        error
    ));

    CHECK(!error.empty());
    CHECK(mesh.positions.empty());
    CHECK(mesh.normals.empty());
    CHECK(mesh.indices.empty());
    CHECK(mesh.edges.empty());
}

void test_cube_bounds() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(kCubeWithoutNormals, mesh, error));

    const gfx::Aabb bounds = gfx::compute_bounds(mesh);

    CHECK(approx_vec(bounds.min, {-1, -1, -1}));
    CHECK(approx_vec(bounds.max, {1, 1, 1}));
}

void test_vertex_colors_ignored() {
    const char* obj = R"OBJ(
v 1 2 3 0.5 0.5 0.5
v 4 2 3 1.0 0.0 0.0
v 1 5 3 0.0 1.0 0.0
f 1 2 3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.triangle_count() == 1);
    CHECK(approx_vec(mesh.positions[0], {1, 2, 3}));
    CHECK(approx_vec(mesh.positions[1], {4, 2, 3}));
    CHECK(approx_vec(mesh.positions[2], {1, 5, 3}));
}

void test_y_up_to_z_up() {
    gfx::Mesh mesh;
    mesh.positions.push_back({1.0f, 2.0f, 3.0f});
    mesh.normals.push_back({0.0f, 1.0f, 0.0f});

    gfx::mesh_y_up_to_z_up(mesh);

    CHECK(approx_vec(mesh.positions[0], {1.0f, -3.0f, 2.0f}));
    CHECK(approx_vec(mesh.normals[0], {0.0f, 0.0f, 1.0f}));
}

void test_cube_bounds_after_axis_conversion() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(kCubeWithoutNormals, mesh, error));

    gfx::mesh_y_up_to_z_up(mesh);

    const gfx::Aabb bounds = gfx::compute_bounds(mesh);

    CHECK(approx_vec(bounds.min, {-1.0f, -1.0f, -1.0f}));
    CHECK(approx_vec(bounds.max, {1.0f, 1.0f, 1.0f}));
}

void test_flat_normals_for_triangle() {
    gfx::Mesh mesh;
    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };
    mesh.indices = {0, 1, 2};

    CHECK(ensure_normals(mesh));

    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.indices.size() == 3);
    CHECK(mesh.normals.size() == 3);

    for (const gfx::Vec3& n : mesh.normals) {
        CHECK(approx(n.x, 0.0));
        CHECK(approx(n.y, 0.0));
        CHECK(approx(n.z, 1.0));
    }
}

void test_degenerate_triangle_normals() {
    gfx::Mesh mesh;
    mesh.positions = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };
    mesh.indices = {0, 1, 2};

    CHECK(ensure_normals(mesh));

    for (const gfx::Vec3& n : mesh.normals) {
        CHECK(approx(n.x, 0.0));
        CHECK(approx(n.y, 0.0));
        CHECK(approx(n.z, 1.0));
        CHECK(std::isfinite(n.x));
        CHECK(std::isfinite(n.y));
        CHECK(std::isfinite(n.z));
    }
}

void test_existing_normals_are_normalized() {
    gfx::Mesh mesh;
    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f}
    };
    mesh.normals = {
        {0.0f, 0.0f, 2.0f},
        {0.0f, 0.0f, 0.0f}
    };

    CHECK(!ensure_normals(mesh));
    CHECK(mesh.positions.size() == 2);

    for (const gfx::Vec3& n : mesh.normals) {
        CHECK(approx(n.x, 0.0));
        CHECK(approx(n.y, 0.0));
        CHECK(approx(n.z, 1.0));
    }
}

void test_cube_without_normals() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(load_obj(cube_no_normals_path, mesh, error));

    CHECK(ensure_normals(mesh));
    CHECK(mesh.edges.size() == 24);

    CHECK(mesh.positions.size() == 36);
    CHECK(mesh.indices.size() == 36);
    CHECK(mesh.normals.size() == 36);

    for (const std::uint32_t index : mesh.edges) {
        CHECK(index < mesh.positions.size());

        if (index >= mesh.positions.size()) {
            continue;
        }

        const gfx::Vec3& p = mesh.positions[index];

        CHECK(approx(std::abs(p.x), 1.0));
        CHECK(approx(std::abs(p.y), 1.0));
        CHECK(approx(std::abs(p.z), 1.0));
    }

    for (std::size_t i = 0; i < mesh.indices.size(); ++i) {
        CHECK(mesh.indices[i] == i);

        const gfx::Vec3& n = mesh.normals[i];

        const float len = std::sqrt(
            n.x * n.x + n.y * n.y + n.z * n.z
        );

        CHECK(approx(len, 1.0));
    }

    // The cube is centered at the origin. Each triangle's normal
    // should point away from the origin, not into the cube.
    for (std::size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
        const gfx::Vec3& a = mesh.positions[mesh.indices[i]];
        const gfx::Vec3& b = mesh.positions[mesh.indices[i + 1]];
        const gfx::Vec3& c = mesh.positions[mesh.indices[i + 2]];
        const gfx::Vec3& n = mesh.normals[mesh.indices[i]];

        const gfx::Vec3 centroid = (a + b + c) * (1.0f / 3.0f);

        const float outward_dot =
            n.x * centroid.x +
            n.y * centroid.y +
            n.z * centroid.z;

        CHECK(outward_dot > 0.0f);
    }
}

void test_cube_with_normals() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(load_obj(cube_path, mesh, error));

    const std::size_t original_vertex_count = mesh.positions.size();

    CHECK(original_vertex_count == 24);
    CHECK(!ensure_normals(mesh));
    CHECK(mesh.positions.size() == original_vertex_count);
    CHECK(mesh.normals.size() == original_vertex_count);
}

void test_small_triangle_normal() {
    gfx::Mesh mesh;

    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {5.0e-4f, 0.0f, 0.0f},
        {0.0f, 0.0f, 5.0e-4f}
    };

    mesh.indices = {0, 1, 2};

    CHECK(ensure_normals(mesh));

    CHECK(mesh.positions.size() == 3);
    CHECK(mesh.normals.size() == 3);

    for (const gfx::Vec3& n : mesh.normals) {
        CHECK(approx(n.x, 0.0));
        CHECK(approx(n.y, -1.0));
        CHECK(approx(n.z, 0.0));
    }
}

void test_cube_has_twelve_edges() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(kCubeWithNormals, mesh, error));

    // Six quad faces, each with four edges
    // Twelve unique edges after removing shared duplicates
    CHECK(mesh.edges.size() == 24);
}

void test_ensure_edges_fallback() {
    gfx::Mesh mesh;

    mesh.positions = {
        {0.0f, 0.0f, 0.0f},
        {1.0f, 0.0f, 0.0f},
        {1.0f, 1.0f, 0.0f},
        {0.0f, 1.0f, 0.0f}
    };

    // Two triangles form a quad. The shared diagonal is deduplicated,
    // leaving five unique edges, including that diagonal
    mesh.indices = {0, 1, 2, 0, 2, 3};

    gfx::ensure_edges(mesh);

    CHECK(mesh.edges.size() == 10);

    // Calling it again must not duplicate existing edges
    gfx::ensure_edges(mesh);

    CHECK(mesh.edges.size() == 10);
}

// ---------------------------------------------------------------------
// Texture coordinates and materials (M9)
// ---------------------------------------------------------------------

bool approx_vec2(const gfx::Vec2& a, float x, float y) {
    return approx(a.x, x) && approx(a.y, y);
}

void test_texcoords() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
vt 0.25 0.5
vt 1 0 0.0
vt 0 1
f 1/1 2/2 3/3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.has_texcoords());
    CHECK(!mesh.has_normals());
    CHECK(mesh.texcoords.size() == mesh.positions.size());
    CHECK(approx_vec2(mesh.texcoords[0], 0.25f, 0.5f));
    CHECK(approx_vec2(mesh.texcoords[1], 1.0f, 0.0f));
    CHECK(approx_vec2(mesh.texcoords[2], 0.0f, 1.0f));
}

void test_uv_seam_splits_vertices() {
    // Position 1 is used with two different texture coordinates
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vt 0 0
vt 1 0
vt 0 1
vt 0.5 0.5
f 1/1 2/2 3/3
f 2/2 4/4 1/4
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.triangle_count() == 2);

    // 1/1, 2/2, 3/3, 4/4, 1/4: five vertices, not four
    CHECK(mesh.positions.size() == 5);
    CHECK(mesh.texcoords.size() == 5);

    // The shared corner 2/2 is still shared
    CHECK(mesh.indices[1] == mesh.indices[3]);
}

void test_partial_texcoords_dropped() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
vt 0 0
f 1/1 2/1 3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(!mesh.has_texcoords());
}

void test_texcoord_errors() {
    expect_failure("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0\nf 1 2 3\n");
    expect_failure("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/2 2/1 3/1\n");
    expect_failure("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/0 2/1 3/1\n");
    expect_failure("v 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nf 1/1/1/1 2 3\n");
}

void test_usemtl_assigns_triangles() {
    const char* obj = R"OBJ(
mtllib first.mtl second.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
f 1 2 3
usemtl red
f 1 2 3
usemtl Blue Paint
f 1 2 4 3
usemtl red
f 2 4 3
)OBJ";

    gfx::Mesh mesh;
    std::string error;
    std::vector<std::string> libraries;

    CHECK(gfx::parse_obj(obj, mesh, error, &libraries));

    // In order of first use; the first face had no usemtl
    CHECK(mesh.materials.size() == 3);
    CHECK(mesh.materials[0].name == "default");
    CHECK(mesh.materials[1].name == "red");
    CHECK(mesh.materials[2].name == "Blue Paint");

    // One entry per triangle: the quad gives two
    const std::vector<std::uint32_t> expected = {0, 1, 2, 2, 1};
    CHECK(mesh.triangle_materials == expected);

    CHECK(libraries.size() == 2);
    CHECK(libraries[0] == "first.mtl");
    CHECK(libraries[1] == "second.mtl");
}

void test_mesh_always_has_a_material() {
    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(kTriangle, mesh, error));
    CHECK(mesh.materials.size() == 1);
    CHECK(mesh.materials[0].name == "default");
    CHECK(mesh.triangle_materials.size() == 1);
    CHECK(mesh.triangle_materials[0] == 0);
}

void test_parse_mtl() {
    const char* mtl = R"MTL(
# Blender MTL
Kd 9 9 9
newmtl Wood
Ns 250.0
Kd 0.8 0.5 0.2
Ke 0 0 0
map_Kd -s 1 1 1 -bm 0.5 textures/wood grain.png
newmtl Lamp
Pr 0.3
Ns 1000
Pm 1.5
Ke 4 3 2
illum 2
)MTL";

    std::vector<gfx::Material> materials;
    std::string error;

    CHECK(gfx::parse_mtl(mtl, "models", materials, error));
    CHECK(materials.size() == 2);

    const gfx::Material& wood = materials[0];
    CHECK(wood.name == "Wood");
    CHECK(approx_vec(wood.base_color, {0.8f, 0.5f, 0.2f}));

    // Ns 250 -> 1 - sqrt(0.25) = 0.5
    CHECK(approx(wood.roughness, 0.5));
    CHECK(approx(wood.metallic, 0.0));
    CHECK(wood.base_color_texture ==
          (std::filesystem::path("models") / "textures/wood grain.png")
              .string());

    const gfx::Material& lamp = materials[1];
    CHECK(lamp.name == "Lamp");

    // Pr wins over a later Ns; Pm is clamped to 1
    CHECK(approx(lamp.roughness, 0.3));
    CHECK(approx(lamp.metallic, 1.0));
    CHECK(approx_vec(lamp.emission, {4, 3, 2}));
    CHECK(lamp.base_color_texture.empty());
    CHECK(approx_vec(lamp.base_color, {0.8f, 0.8f, 0.8f}));
}

void test_parse_mtl_errors() {
    std::vector<gfx::Material> materials(1);
    std::string error;

    // A bad line fails, names the line, and adds nothing
    CHECK(!gfx::parse_mtl("newmtl a\nKd 1 x 0\n", "", materials, error));
    CHECK(error.rfind("line 2:", 0) == 0);
    CHECK(materials.size() == 1);

    CHECK(!gfx::parse_mtl("newmtl\n", "", materials, error));
    CHECK(!gfx::parse_mtl("newmtl a\nmap_Kd -s 1 1\n", "", materials, error));
    CHECK(!gfx::parse_mtl("newmtl a\nmap_Kd\n", "", materials, error));
    CHECK(materials.size() == 1);
}

void test_roughness_from_phong() {
    CHECK(approx(gfx::roughness_from_phong_exponent(1000.0f), 0.0));
    CHECK(approx(gfx::roughness_from_phong_exponent(0.0f), 1.0));
    CHECK(approx(gfx::roughness_from_phong_exponent(5000.0f), 0.0));
    CHECK(approx(gfx::roughness_from_phong_exponent(-3.0f), 1.0));
}

void test_load_obj_with_mtl() {
    namespace fs = std::filesystem;

    const fs::path dir = fs::temp_directory_path() / "gfxlab_test_obj_mtl";
    fs::create_directories(dir);

    {
        std::ofstream obj(dir / "model.obj", std::ios::binary);
        obj << "mtllib model.mtl missing.mtl\n"
               "v 0 0 0\nv 1 0 0\nv 0 1 0\n"
               "vt 0 0\nvt 1 0\nvt 0 1\n"
               "usemtl Red\nf 1/1 2/2 3/3\n"
               "usemtl Unknown\nf 1/1 3/3 2/2\n";

        std::ofstream mtl(dir / "model.mtl", std::ios::binary);
        mtl << "newmtl Red\nKd 1 0 0\nPr 0.25\nmap_Kd red.png\n";
    }

    gfx::Mesh mesh;
    std::string error;

    // The missing library and the unknown name only log warnings
    CHECK(gfx::load_obj((dir / "model.obj").string(), mesh, error));
    CHECK(mesh.materials.size() == 2);
    CHECK(mesh.materials[0].name == "Red");
    CHECK(approx_vec(mesh.materials[0].base_color, {1, 0, 0}));
    CHECK(approx(mesh.materials[0].roughness, 0.25));
    CHECK(mesh.materials[0].base_color_texture ==
          (dir / "red.png").string());
    CHECK(mesh.materials[1].name == "Unknown");
    CHECK(approx(mesh.materials[1].roughness, 0.5));

    std::error_code ec;
    fs::remove_all(dir, ec);
}

void test_flat_normals_keep_texcoords() {
    const char* obj = R"OBJ(
v 0 0 0
v 1 0 0
v 0 1 0
v 1 1 0
vt 0 0
vt 1 0
vt 0 1
vt 1 1
usemtl a
f 1/1 2/2 3/3
usemtl b
f 2/2 4/4 3/3
)OBJ";

    gfx::Mesh mesh;
    std::string error;

    CHECK(parse(obj, mesh, error));
    CHECK(mesh.positions.size() == 4);
    CHECK(gfx::ensure_normals(mesh));

    // Unwelded: one vertex per triangle corner, each keeping its UV,
    // and the triangles keep their materials
    CHECK(mesh.positions.size() == 6);
    CHECK(mesh.texcoords.size() == 6);
    CHECK(approx_vec2(mesh.texcoords[mesh.indices[4]], 1.0f, 1.0f));
    CHECK(mesh.triangle_materials.size() == 2);
    CHECK(mesh.triangle_materials[1] == 1);
}

} // namespace


int main() {
    test_triangle();
    test_quad_triangulation();
    test_two_quads_share_edge();
    test_negative_indices();
    test_position_texcoord_normal();
    test_position_double_slash_normal();
    test_mixed_normals();
    test_ignored_keywords_and_crlf();
    test_float_formats_and_w();
    test_out_of_range_face_index();
    test_invalid_inputs();
    test_missing_file();
    test_cube_bounds();
    test_vertex_colors_ignored();
    test_y_up_to_z_up();
    test_cube_bounds_after_axis_conversion();
    test_flat_normals_for_triangle();
    test_degenerate_triangle_normals();
    test_existing_normals_are_normalized();
    test_cube_without_normals();
    test_cube_with_normals();
    test_small_triangle_normal();
    test_cube_has_twelve_edges();
    test_ensure_edges_fallback();
    test_texcoords();
    test_uv_seam_splits_vertices();
    test_partial_texcoords_dropped();
    test_texcoord_errors();
    test_usemtl_assigns_triangles();
    test_mesh_always_has_a_material();
    test_parse_mtl();
    test_parse_mtl_errors();
    test_roughness_from_phong();
    test_load_obj_with_mtl();
    test_flat_normals_keep_texcoords();

    if (g_failures == 0) {
        std::printf("All OBJ tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d OBJ test(s) failed\n", g_failures);
    return 1;
}
