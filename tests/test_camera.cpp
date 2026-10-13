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

} // namespace


int main() {
    test_eye_distance();
    test_orbit_preserves_distance();
    test_pitch_clamp_and_finite_view();
    test_target_is_straight_ahead();
    test_zoom();
    test_pan_perpendicular_to_view();
    test_pan_direction();
    
    if (g_failures == 0) {
        std::printf("All camera tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d camera test(s) failed\n", g_failures);
    return 1;
}
