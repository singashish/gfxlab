#include "core/obj_loader.h"
#include "check.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

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

    const std::uint32_t expected[] = {0, 1, 2, 0, 2, 3};

    for (std::size_t i = 0; i < 6; ++i) {
        CHECK(mesh.indices[i] == expected[i]);
    }
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

    CHECK(load_obj("../../../assets/cube_no_normals.obj", mesh, error));

    CHECK(ensure_normals(mesh));

    CHECK(mesh.positions.size() == 36);
    CHECK(mesh.indices.size() == 36);
    CHECK(mesh.normals.size() == 36);

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

    CHECK(load_obj("../../../assets/cube.obj", mesh, error));

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

} // namespace


int main() {
    test_triangle();
    test_quad_triangulation();
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

    if (g_failures == 0) {
        std::printf("All OBJ tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d OBJ test(s) failed\n", g_failures);
    return 1;
}
