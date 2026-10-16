#include "core/transform.h"
#include "core/mesh.h"
#include "check.h"

#include <cmath>
#include <cstdio>

namespace {

bool approx_vec(gfx::Vec3 a, gfx::Vec3 b, double eps = 1e-5) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

bool approx_mat(const gfx::Mat4& a, const gfx::Mat4& b,
                double eps = 1e-5) {
    for (int col = 0; col < 4; ++col) {
        if (!approx(a.col[col].x, b.col[col].x, eps)
            || !approx(a.col[col].y, b.col[col].y, eps)
            || !approx(a.col[col].z, b.col[col].z, eps)
            || !approx(a.col[col].w, b.col[col].w, eps)) {
            return false;
        }
    }
    return true;
}

void test_default_transform() {
    const gfx::Transform t;

    CHECK(approx_mat(t.model_matrix(), gfx::mat4_identity()));
    CHECK(approx_mat(t.normal_matrix(), gfx::mat4_identity()));
}

void test_translation() {
    gfx::Transform t;
    t.position = {1.0f, 2.0f, 3.0f};

    CHECK(approx_vec(
        gfx::transform_point(t.model_matrix(), {0.0f, 0.0f, 0.0f}),
        {1.0f, 2.0f, 3.0f}
    ));
}

void test_scale() {
    gfx::Transform t;
    t.scale = {2.0f, 3.0f, 4.0f};

    CHECK(approx_vec(
        gfx::transform_point(t.model_matrix(), {1.0f, 1.0f, 1.0f}),
        {2.0f, 3.0f, 4.0f}
    ));
}

void test_rotation_z() {
    gfx::Transform t;
    constexpr float kHalfPi = 1.57079632679489661923f;
    t.rotation.z = kHalfPi;

    CHECK(approx_vec(
        gfx::transform_direction(t.model_matrix(), {1.0f, 0.0f, 0.0f}),
        {0.0f, 1.0f, 0.0f}
    ));
}

void test_transform_order() {
    gfx::Transform t;
    t.position = {10.0f, 0.0f, 0.0f};
    t.scale = {2.0f, 2.0f, 2.0f};
    t.rotation.z = 1.57079632679489661923f;

    // (1,0,0) -> scale (2,0,0)
    // -> rotate (0,2,0) -> translate (10,2,0)
    CHECK(approx_vec(
        gfx::transform_point(t.model_matrix(), {1.0f, 0.0f, 0.0f}),
        {10.0f, 2.0f, 0.0f}
    ));
}

void test_normal_matrix() {
    gfx::Transform t;
    t.scale = {1.0f, 2.0f, 3.0f};
    t.rotation = {0.3f, 0.5f, 0.7f};

    const gfx::Vec3 tangent = {1.0f, 0.0f, 1.0f};
    const gfx::Vec3 normal = {1.0f, 0.0f, -1.0f};

    CHECK(approx(gfx::dot(tangent, normal), 0.0));

    const gfx::Vec3 world_tangent =
        gfx::transform_direction(t.model_matrix(), tangent);

    const gfx::Vec3 corrected_normal =
        gfx::transform_direction(t.normal_matrix(), normal);

    const gfx::Vec3 incorrect_normal =
        gfx::transform_direction(t.model_matrix(), normal);

    // A correctly transformed normal remains perpendicular to the tangent
    CHECK(approx(gfx::dot(world_tangent, corrected_normal), 0.0, 1e-4));

    // The model matrix does not correctly transform normals under
    // non-uniform scale
    CHECK(std::fabs(gfx::dot(world_tangent, incorrect_normal)) > 1e-3);
}

void test_zero_scale_is_safe() {
    gfx::Transform t;
    t.scale = {0.0f, 1.0f, 0.0f};

    const gfx::Mat4 n = t.normal_matrix();

    for (int col = 0; col < 4; ++col) {
        CHECK(std::isfinite(n.col[col].x));
        CHECK(std::isfinite(n.col[col].y));
        CHECK(std::isfinite(n.col[col].z));
        CHECK(std::isfinite(n.col[col].w));
    }
}

void test_mesh_center_at_origin() {
    gfx::Mesh mesh;

    // A cube whose bounds are [4,6] on every axis
    mesh.positions = {
        {4,4,4}, {6,4,4}, {6,6,4}, {4,6,4},
        {4,4,6}, {6,4,6}, {6,6,6}, {4,6,6}
    };

    gfx::mesh_center_at_origin(mesh);
    const gfx::Aabb bounds = gfx::compute_bounds(mesh);

    CHECK(approx_vec(bounds.min, {-1.0f, -1.0f, -1.0f}));
    CHECK(approx_vec(bounds.max, { 1.0f,  1.0f,  1.0f}));
}

void test_xyz_euler_order() {
    gfx::Transform t;

    constexpr float kHalfPi = 1.57079632679489661923f;
    t.rotation = {kHalfPi, 0.0f, kHalfPi};

    const gfx::Vec3 result = gfx::transform_direction(
        t.model_matrix(),
        {0.0f, 1.0f, 0.0f}
    );

    // X rotation first: +Y becomes +Z.
    // Z rotation then leaves +Z unchanged.
    CHECK(approx_vec(result, {0.0f, 0.0f, 1.0f}));
}

} // namespace


int main() {
    test_default_transform();
    test_translation();
    test_scale();
    test_rotation_z();
    test_transform_order();
    test_normal_matrix();
    test_zero_scale_is_safe();
    test_mesh_center_at_origin();
    test_xyz_euler_order();

    if (g_failures == 0) {
        std::printf("All transform tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d transform test(s) failed\n", g_failures);
    return 1;
}
