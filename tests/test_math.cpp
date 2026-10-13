#include "core/vec3.h"
#include "core/mat4.h"
#include "core/math_util.h"
#include "core/camera_math.h"
#include "check.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <type_traits>

static bool approx_vec(gfx::Vec3 a, gfx::Vec3 b, float eps = 1e-5f) {
    return approx(a.x, b.x, eps) &&
           approx(a.y, b.y, eps) &&
           approx(a.z, b.z, eps);
}

static gfx::Vec3 project(const gfx::Mat4& m, gfx::Vec3 p) {
    const gfx::Vec4 clip = m * gfx::make_vec4(p, 1.0f);

    return {
        clip.x / clip.w,
        clip.y / clip.w,
        clip.z / clip.w
    };
}


int main() {
    using gfx::Vec3;

    const Vec3 X{1.0f, 0.0f, 0.0f};
    const Vec3 Y{0.0f, 1.0f, 0.0f};
    const Vec3 Z{0.0f, 0.0f, 1.0f};
    const Vec3 zero{0.0f, 0.0f, 0.0f};

    // Vec3 must remain trivially copyable
    CHECK(std::is_trivially_copyable<Vec3>::value);

    // Addition and subtraction
    CHECK(approx_vec(X + Y, {1.0f, 1.0f, 0.0f}));
    CHECK(approx_vec(X - Y, {1.0f, -1.0f, 0.0f}));

    // Scalar multiplication
    CHECK(approx_vec(X * 3.0f, {3.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(2.0f * Y, {0.0f, 2.0f, 0.0f}));

    // Unary negation
    CHECK(approx_vec(-Z, {0.0f, 0.0f, -1.0f}));

    // Dot product
    CHECK(approx(gfx::dot(X, Y), 0.0f));
    CHECK(approx(gfx::dot({3.0f, 4.0f, 0.0f},
                          {3.0f, 4.0f, 0.0f}), 25.0f));

    // Cross product: right-handed coordinate system
    CHECK(approx_vec(gfx::cross(X, Y), Z));
    CHECK(approx_vec(gfx::cross(Y, Z), X));
    CHECK(approx_vec(gfx::cross(Y, X), -Z));

    // Cross product of parallel vectors is zero
    CHECK(approx_vec(gfx::cross(X, X), zero));

    // Vector length
    CHECK(approx(gfx::length({3.0f, 4.0f, 0.0f}), 5.0f));
    CHECK(approx(gfx::length(zero), 0.0f));

    // Normalize a nonzero vector
    const Vec3 n = gfx::normalize({3.0f, 4.0f, 0.0f});

    CHECK(approx(gfx::length(n), 1.0f));
    CHECK(approx(n.x, 0.6f));
    CHECK(approx(n.y, 0.8f));
    CHECK(approx(n.z, 0.0f));

    // ---------------------------------------------------------------
    // Normalize the zero vector without producing NaNs
    const Vec3 nz = gfx::normalize(zero);

    CHECK(approx_vec(nz, zero));
    CHECK(!std::isnan(nz.x));
    CHECK(!std::isnan(nz.y));
    CHECK(!std::isnan(nz.z));

    // Mat4: identity
    const gfx::Mat4 identity = gfx::mat4_identity();
    const gfx::Vec4 v4{2.0f, 3.0f, 4.0f, 1.0f};

    const gfx::Vec4 identity_result = identity * v4;

    CHECK(approx(identity_result.x, v4.x));
    CHECK(approx(identity_result.y, v4.y));
    CHECK(approx(identity_result.z, v4.z));
    CHECK(approx(identity_result.w, v4.w));

    // Mat4: translation
    const gfx::Mat4 translation =
        gfx::mat4_translate({1.0f, 2.0f, 3.0f});

    const gfx::Vec3 point{4.0f, 5.0f, 6.0f};
    const gfx::Vec3 direction{4.0f, 5.0f, 6.0f};

    const gfx::Vec3 moved =
        gfx::transform_point(translation, point);

    const gfx::Vec3 unchanged =
        gfx::transform_direction(translation, direction);

    const gfx::Vec3 expected_moved{5.0f, 7.0f, 9.0f};

    CHECK(approx_vec(moved, expected_moved));
    CHECK(approx_vec(unchanged, direction));

    // Mat4: scaling
    const gfx::Mat4 scaling =
        gfx::mat4_scale({2.0f, 3.0f, 4.0f});

    const gfx::Vec3 scaled =
        gfx::transform_point(scaling, {1.0f, 1.0f, 1.0f});

    const gfx::Vec3 expected_scaled{2.0f, 3.0f, 4.0f};

    CHECK(approx_vec(scaled, expected_scaled));

    // Mat4: right-hand rule
    constexpr float HALF_PI = 1.57079632679f;

    const gfx::Mat4 rotate_z =
        gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, HALF_PI);

    const gfx::Vec3 rotated_x =
        gfx::transform_direction(rotate_z, {1.0f, 0.0f, 0.0f});

    const gfx::Vec3 expected_y{0.0f, 1.0f, 0.0f};

    CHECK(approx_vec(rotated_x, expected_y));

    const gfx::Mat4 rotate_x =
        gfx::mat4_rotate({1.0f, 0.0f, 0.0f}, HALF_PI);

    const gfx::Vec3 rotated_y =
        gfx::transform_direction(rotate_x, {0.0f, 1.0f, 0.0f});

    const gfx::Vec3 expected_z{0.0f, 0.0f, 1.0f};

    CHECK(approx_vec(rotated_y, expected_z));

    // Mat4: matrix multiplication and composition order
    const gfx::Mat4 A =
        gfx::mat4_translate({1.0f, 2.0f, 3.0f});

    const gfx::Mat4 B =
        gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, HALF_PI);

    const gfx::Vec4 test_vector{2.0f, 3.0f, 4.0f, 1.0f};

    // (A * B) * v must equal A * (B * v).
    const gfx::Vec4 lhs = (A * B) * test_vector;
    const gfx::Vec4 rhs = A * (B * test_vector);

    CHECK(approx(lhs.x, rhs.x));
    CHECK(approx(lhs.y, rhs.y));
    CHECK(approx(lhs.z, rhs.z));
    CHECK(approx(lhs.w, rhs.w));

    // Translation * rotation versus rotation * translation.
    const gfx::Vec3 test_point{1.0f, 0.0f, 0.0f};

    const gfx::Vec3 translate_then_rotate_matrix =
        gfx::transform_point(A * B, test_point);

    const gfx::Vec3 rotate_then_translate_matrix =
        gfx::transform_point(B * A, test_point);

    CHECK(!approx_vec(
        translate_then_rotate_matrix,
        rotate_then_translate_matrix
    ));

    // Mat4: GPU memory layout
    static_assert(sizeof(gfx::Mat4) == 64,
                "Mat4 must occupy exactly 64 bytes");

    const gfx::Mat4 upload_matrix =
        gfx::mat4_translate({1.0f, 2.0f, 3.0f});

    float gpu_data[16] = {};

    std::memcpy(gpu_data, &upload_matrix, sizeof(upload_matrix));

    CHECK(approx(gpu_data[12], 1.0f));
    CHECK(approx(gpu_data[13], 2.0f));
    CHECK(approx(gpu_data[14], 3.0f));
    CHECK(approx(gpu_data[15], 1.0f));

    // Mat4: trivially copyable
    CHECK((std::is_trivially_copyable<gfx::Mat4>::value));
    CHECK((std::is_trivially_copyable<gfx::Vec4>::value));

    // ---------------------------------------------------------------
    // Camera math: look_at
    const gfx::Vec3 eye{0.0f, -5.0f, 0.0f};
    const gfx::Vec3 target{0.0f, 0.0f, 0.0f};
    const gfx::Vec3 world_up{0.0f, 0.0f, 1.0f};

    const gfx::Mat4 view = gfx::look_at(eye, target, world_up);

    // The camera's eye must map to the origin
    CHECK(approx_vec(
        gfx::transform_point(view, eye),
        {0.0f, 0.0f, 0.0f}
    ));

    // The target is five units in front of the camera (-Z)
    CHECK(approx_vec(
        gfx::transform_point(view, target),
        {0.0f, 0.0f, -5.0f}
    ));

    // World +X must remain camera +X.
    CHECK(approx_vec(
        gfx::transform_direction(view, {1.0f, 0.0f, 0.0f}),
        {1.0f, 0.0f, 0.0f}
    ));

    // World +Z must map to camera +Y
    CHECK(approx_vec(
        gfx::transform_direction(view, {0.0f, 0.0f, 1.0f}),
        {0.0f, 1.0f, 0.0f}
    ));

    // ---------------------------------------------------------------
    // Camera math: perspective projection
    constexpr float PI = 3.14159265358979323846f;

    const float fov_y = PI / 2.0f;  // 90 degrees
    const float aspect = 1.0f;
    const float near_plane = 0.1f;
    const float far_plane = 100.0f;

    const gfx::Mat4 proj_gl = gfx::perspective(
        fov_y, aspect, near_plane, far_plane,
        gfx::ClipSpace::CLIP_SPACE_OPENGL
    );

    const gfx::Mat4 proj_vk = gfx::perspective(
        fov_y, aspect, near_plane, far_plane,
        gfx::ClipSpace::CLIP_SPACE_VULKAN
    );

    // Near plane: OpenGL -> -1, Vulkan -> 0
    CHECK(approx(
        project(proj_gl, {0.0f, 0.0f, -near_plane}).z,
        -1.0f
    ));

    CHECK(approx(
        project(proj_vk, {0.0f, 0.0f, -near_plane}).z,
        0.0f
    ));

    // Far plane: both conventions -> +1
    CHECK(approx(
        project(proj_gl, {0.0f, 0.0f, -far_plane}).z,
        1.0f
    ));

    CHECK(approx(
        project(proj_vk, {0.0f, 0.0f, -far_plane}).z,
        1.0f
    ));

    // Positive camera-space Y maps to positive GL NDC Y
    // and negative Vulkan NDC Y
    const gfx::Vec3 point_above{0.0f, 1.0f, -5.0f};

    const gfx::Vec3 gl_above = project(proj_gl, point_above);
    const gfx::Vec3 vk_above = project(proj_vk, point_above);

    CHECK(gl_above.y > 0.0f);
    CHECK(vk_above.y < 0.0f);

    // At the near plane's vertical FOV edge, GL NDC Y = +1
    const float half_height =
        near_plane * std::tan(fov_y * 0.5f);

    CHECK(approx(
        project(proj_gl, {0.0f, half_height, -near_plane}).y,
        1.0f
    ));

    // A point behind the camera produces negative clip W
    const gfx::Vec4 behind_clip =
        proj_gl * gfx::make_vec4({0.0f, 0.0f, 1.0f}, 1.0f);

    CHECK(behind_clip.w < 0.0f);

    // Conversion to Radian
    CHECK(approx(gfx::to_radians(180.0f), gfx::kPi));
    CHECK(approx(gfx::to_radians(90.0f), gfx::kPi / 2.0f));


    if (g_failures == 0) {
        std::printf("All math tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d math test(s) failed\n", g_failures);
    return 1;
}
