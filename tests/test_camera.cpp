#include "core/orbit_camera.h"

#include <cassert>
#include <cmath>

namespace {

constexpr float kEpsilon = 1e-4f;

bool near(float a, float b, float eps = kEpsilon) {
    return std::fabs(a - b) <= eps;
}

bool near_vec(const gfx::Vec3& a, const gfx::Vec3& b, float eps = kEpsilon) {
    return near(a.x, b.x, eps)
        && near(a.y, b.y, eps)
        && near(a.z, b.z, eps);
}

void test_eye_distance() {
    gfx::OrbitCamera camera;

    assert(near(
        gfx::length(camera.eye() - camera.target),
        camera.distance
    ));
}

void test_orbit_preserves_distance() {
    gfx::OrbitCamera camera;

    camera.orbit(120.0f, -35.0f);
    camera.orbit(-64.0f, 80.0f);
    camera.orbit(12.0f, 15.0f);

    assert(near(
        gfx::length(camera.eye() - camera.target),
        camera.distance
    ));
}

void test_pitch_clamp_and_finite_view() {
    gfx::OrbitCamera camera;
    const float half_pi = gfx::kPi * 0.5f;

    camera.orbit(0.0f, 1e6f);
    assert(camera.pitch < half_pi);
    assert(camera.pitch > -half_pi);

    gfx::Mat4 view = camera.view();
    assert(view.col[0].x == view.col[0].x); // not NaN

    camera.orbit(0.0f, -1e6f);
    assert(camera.pitch < half_pi);
    assert(camera.pitch > -half_pi);

    view = camera.view();
    assert(view.col[0].x == view.col[0].x);
}

void test_target_is_straight_ahead() {
    gfx::OrbitCamera camera;

    const gfx::Vec3 p = gfx::transform_point(
        camera.view(), camera.target
    );

    assert(near_vec(p, {0.0f, 0.0f, -camera.distance}));
}

void test_zoom() {
    gfx::OrbitCamera camera;
    const float initial_distance = camera.distance;

    camera.zoom(1.0f);
    assert(near(camera.distance, initial_distance * 0.9f));

    for (int i = 0; i < 100; ++i) {
        camera.zoom(100.0f);
    }

    assert(near(camera.distance, 0.2f));
    assert(camera.distance > 0.0f);
}

void test_pan_perpendicular_to_view() {
    gfx::OrbitCamera camera;

    const gfx::Vec3 old_target = camera.target;
    const gfx::Vec3 old_eye = camera.eye();
    const float old_distance = camera.distance;
    const gfx::Vec3 forward = gfx::normalize(old_target - old_eye);

    camera.pan(37.0f, -12.0f, 0.01f);

    const gfx::Vec3 delta = camera.target - old_target;

    assert(near(gfx::dot(delta, forward), 0.0f));
    assert(near(
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
    assert(gfx::dot(delta, right) < 0.0f);
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
    return 0;
}
