#include "core/orbit_camera.h"
#include "check.h"

#include <cmath>

namespace {

constexpr float kEpsilon = 1e-4f;

bool approx_vec(const gfx::Vec3& a, const gfx::Vec3& b, float eps = kEpsilon) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

void test_eye_distance() {
    gfx::OrbitCamera camera;

    CHECK(approx(
        gfx::length(camera.eye() - camera.target),
        camera.distance
    ));
}

void test_orbit_preserves_distance() {
    gfx::OrbitCamera camera;

    camera.orbit(120.0f, -35.0f);
    camera.orbit(-64.0f, 80.0f);
    camera.orbit(12.0f, 15.0f);

    CHECK(approx(
        gfx::length(camera.eye() - camera.target),
        camera.distance
    ));
}

void test_pitch_clamp_and_finite_view() {
    gfx::OrbitCamera camera;
    const float half_pi = gfx::kPi * 0.5f;

    camera.orbit(0.0f, 1e6f);
    CHECK(camera.pitch < half_pi);
    CHECK(camera.pitch > -half_pi);

    gfx::Mat4 view = camera.view();
    CHECK(view.col[0].x == view.col[0].x); // not NaN

    camera.orbit(0.0f, -1e6f);
    CHECK(camera.pitch < half_pi);
    CHECK(camera.pitch > -half_pi);

    view = camera.view();
    CHECK(view.col[0].x == view.col[0].x);
}

void test_target_is_straight_ahead() {
    gfx::OrbitCamera camera;

    const gfx::Vec3 p = gfx::transform_point(
        camera.view(), camera.target
    );

    CHECK(approx_vec(p, {0.0f, 0.0f, -camera.distance}));
}

void test_zoom() {
    gfx::OrbitCamera camera;
    const float initial_distance = camera.distance;

    camera.zoom(1.0f);
    CHECK(approx(camera.distance, initial_distance * 0.9f));

    for (int i = 0; i < 100; ++i) {
        camera.zoom(100.0f);
    }

    CHECK(approx(camera.distance, 0.2f));
    CHECK(camera.distance > 0.0f);
}

void test_pan_perpendicular_to_view() {
    gfx::OrbitCamera camera;

    const gfx::Vec3 old_target = camera.target;
    const gfx::Vec3 old_eye = camera.eye();
    const float old_distance = camera.distance;
    const gfx::Vec3 forward = gfx::normalize(old_target - old_eye);

    camera.pan(37.0f, -12.0f, 0.01f);

    const gfx::Vec3 delta = camera.target - old_target;

    CHECK(approx(gfx::dot(delta, forward), 0.0f));
    CHECK(approx(
        gfx::length(camera.eye() - camera.target),
        old_distance
    ));
}

void test_pan_direction() {
    gfx::OrbitCamera camera;

    const gfx::Vec3 old_target = camera.target;
    const gfx::Vec3 forward = gfx::normalize(
        camera.target - camera.eye()
    );
    const gfx::Vec3 right = gfx::normalize(
        gfx::cross(forward, {0.0f, 0.0f, 1.0f})
    );

    camera.pan(100.0f, 0.0f, 0.01f);

    const gfx::Vec3 delta = camera.target - old_target;
    CHECK(gfx::dot(delta, right) < 0.0f);
}

void test_camera_basis() {
    const gfx::OrbitCamera camera;
    const gfx::CameraBasis b = camera.basis();

    CHECK(approx(gfx::length(b.right), 1.0f));
    CHECK(approx(gfx::length(b.up), 1.0f));
    CHECK(approx(gfx::length(b.forward), 1.0f));

    CHECK(approx(gfx::dot(b.right, b.up), 0.0f));
    CHECK(approx(gfx::dot(b.right, b.forward), 0.0f));
    CHECK(approx(gfx::dot(b.up, b.forward), 0.0f));

    CHECK(approx(b.right.z, 0.0f));

    const gfx::Vec3 expected_forward =
        gfx::normalize(camera.target - camera.eye());

    CHECK(approx(b.forward.x, expected_forward.x));
    CHECK(approx(b.forward.y, expected_forward.y));
    CHECK(approx(b.forward.z, expected_forward.z));

    const gfx::Vec3 cross_result = gfx::cross(b.right, b.up);

    CHECK(approx(cross_result.x, -b.forward.x));
    CHECK(approx(cross_result.y, -b.forward.y));
    CHECK(approx(cross_result.z, -b.forward.z));
}

void test_ray_reconstruction() {
    const gfx::OrbitCamera camera;

    constexpr float aspect = 16.0f / 9.0f;
    constexpr float fov = gfx::to_radians(60.0f);

    const gfx::Mat4 view = camera.view();
    const gfx::Mat4 proj = gfx::perspective(
        fov, aspect, 0.1f, 1000.0f,
        gfx::ClipSpace::CLIP_SPACE_OPENGL
    );

    const gfx::Vec3 p = {1.0f, 2.0f, 0.0f};

    const gfx::Vec4 clip =
        (proj * view) * gfx::make_vec4(p, 1.0f);

    CHECK(std::fabs(clip.w) > 1e-6f);

    const float ndc_x = clip.x / clip.w;
    const float ndc_y = clip.y / clip.w;

    const gfx::CameraBasis b = camera.basis();
    const float tan_half = std::tan(fov * 0.5f);

    const gfx::Vec3 right_scaled =
        b.right * (tan_half * aspect);

    const gfx::Vec3 up_scaled = b.up * tan_half;

    const gfx::Vec3 dir =
        b.forward
        + right_scaled * ndc_x
        + up_scaled * ndc_y;

    const gfx::Vec3 ray_to_point = p - camera.eye();
    const gfx::Vec3 error = gfx::cross(dir, ray_to_point);

    CHECK(approx(error.x, 0.0f));
    CHECK(approx(error.y, 0.0f));
    CHECK(approx(error.z, 0.0f));
}

} // namespace


int main() {
    test_eye_distance();
    test_orbit_preserves_distance();
    test_pitch_clamp_and_finite_view();
    test_target_is_straight_ahead();
    test_zoom();
    test_pan_perpendicular_to_view();
    test_pan_direction();
    test_camera_basis();
    test_ray_reconstruction();
    
    if (g_failures == 0) {
        std::printf("All camera tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d camera test(s) failed\n", g_failures);
    return 1;
}
