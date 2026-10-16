#include "core/transform_edit.h"
#include "core/math_util.h"
#include "check.h"

#include <cmath>
#include <cstdio>

namespace {

constexpr float kEps = 1.0e-4f;
constexpr float kHalfPi = gfx::kPi * 0.5f;

bool approx_vec(gfx::Vec3 a, gfx::Vec3 b, float eps = kEps) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

gfx::EditView make_view(const gfx::OrbitCamera& camera) {
    return {
        camera.eye(),
        camera.basis(),
        gfx::to_radians(60.0f),
        1280.0f,
        720.0f
    };
}

void test_projection() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);
    const gfx::Vec3 ahead = camera.eye() + view.basis.forward * 10.0f;

    float x = 0.0f;
    float y = 0.0f;

    CHECK(gfx::project_to_screen(view, ahead, x, y));
    CHECK(approx(x, 640.0f));
    CHECK(approx(y, 360.0f));

    const float depth = 10.0f;
    const float aspect = 1280.0f / 720.0f;
    const float tan_half = std::tan(view.fov_y * 0.5f);

    const gfx::Vec3 right_point =
        ahead + view.basis.right * (depth * tan_half * aspect);

    CHECK(gfx::project_to_screen(view, right_point, x, y));
    CHECK(approx(x, 1280.0f, 1e-3));

    const gfx::Vec3 behind = camera.eye() - view.basis.forward;
    CHECK(!gfx::project_to_screen(view, behind, x, y));
}

void test_grab_free() {
    gfx::OrbitCamera camera;
    gfx::EditView view = make_view(camera);
    gfx::Transform start;
    start.position = camera.target;

    gfx::Transform out;
    gfx::TransformEditor editor;
    editor.begin(gfx::EditMode::EDIT_GRAB, start, 640, 360, view);

    editor.update(740, 360, view, out);

    const gfx::Vec3 delta = out.position - start.position;
    const float depth = gfx::dot(
        start.position - view.eye, view.basis.forward
    );
    const float s = 2.0f * depth *
        std::tan(view.fov_y * 0.5f) / view.height;

    CHECK(approx_vec(delta, view.basis.right * (100.0f * s)));

    editor.update(640, 460, view, out);
    CHECK(approx_vec(
        out.position - start.position,
        -view.basis.up * (100.0f * s)
    ));
}

void test_grab_x_constraint() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = camera.target;

    gfx::TransformEditor editor;
    gfx::Transform out;

    editor.begin(gfx::EditMode::EDIT_GRAB, start, 640, 360, view);
    editor.set_axis(gfx::EditAxis::EDIT_AXIS_X);
    editor.update(740, 410, view, out);

    const float depth = gfx::dot(
        start.position - view.eye, view.basis.forward
    );
    const float s = 2.0f * depth *
        std::tan(view.fov_y * 0.5f) / view.height;

    const gfx::Vec3 free_delta =
        view.basis.right * (100.0f * s)
        - view.basis.up * (50.0f * s);

    const gfx::Vec3 expected = {
        start.position.x + free_delta.x,
        start.position.y,
        start.position.z
    };

    CHECK(approx_vec(out.position, expected));
}

void test_cancel_restores_start() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = {1.0f, 2.0f, 3.0f};
    start.rotation = {0.2f, 0.3f, 0.4f};
    start.scale = {2.0f, 3.0f, 4.0f};

    gfx::TransformEditor editor;
    gfx::Transform out = start;

    editor.begin(gfx::EditMode::EDIT_GRAB, start, 640, 360, view);
    editor.update(900, 500, view, out);

    const gfx::Transform restored = editor.cancel();

    CHECK(!editor.active());
    CHECK(approx_vec(restored.position, start.position, 0.0f));
    CHECK(approx_vec(restored.rotation, start.rotation, 0.0f));
    CHECK(approx_vec(restored.scale, start.scale, 0.0f));
}

void test_axis_toggle() {
    gfx::TransformEditor editor;
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);
    gfx::Transform t;

    editor.begin(gfx::EditMode::EDIT_GRAB, t, 640, 360, view);

    editor.set_axis(gfx::EditAxis::EDIT_AXIS_X);
    CHECK(editor.axis() == gfx::EditAxis::EDIT_AXIS_X);

    editor.set_axis(gfx::EditAxis::EDIT_AXIS_X);
    CHECK(editor.axis() == gfx::EditAxis::EDIT_AXIS_FREE);
}

void test_scale_free_and_y() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = camera.target;
    start.scale = {2.0f, 3.0f, 4.0f};

    float px = 0.0f;
    float py = 0.0f;
    CHECK(gfx::project_to_screen(view, start.position, px, py));

    gfx::TransformEditor editor;
    gfx::Transform out;

    editor.begin(gfx::EditMode::EDIT_SCALE, start, px + 20, py, view);
    editor.update(px + 40, py, view, out);

    CHECK(approx_vec(out.scale, {4.0f, 6.0f, 8.0f}));

    editor.begin(gfx::EditMode::EDIT_SCALE, start, px + 20, py, view);
    editor.set_axis(gfx::EditAxis::EDIT_AXIS_Y);
    editor.update(px + 40, py, view, out);

    CHECK(approx(out.scale.x, start.scale.x));
    CHECK(approx(out.scale.y, 6.0f));
    CHECK(approx(out.scale.z, start.scale.z));
}

void test_scale_clamp_and_pivot_start() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = camera.target;

    float px = 0.0f;
    float py = 0.0f;
    CHECK(gfx::project_to_screen(view, start.position, px, py));

    gfx::TransformEditor editor;
    gfx::Transform out;

    // Starting exactly at the pivot must not divide by zero.
    editor.begin(gfx::EditMode::EDIT_SCALE, start, px, py, view);
    editor.update(px, py, view, out);

    CHECK(std::isfinite(out.scale.x));
    CHECK(std::isfinite(out.scale.y));
    CHECK(std::isfinite(out.scale.z));
    CHECK(out.scale.x > 0.0f);
    CHECK(out.scale.y > 0.0f);
    CHECK(out.scale.z > 0.0f);
}

void test_rotate_z_above_and_below() {
    for (float pitch : {0.4f, -0.4f}) {
        gfx::OrbitCamera camera;
        camera.pitch = pitch;

        const gfx::EditView view = make_view(camera);

        gfx::Transform start;
        start.position = camera.target;

        float px = 0.0f;
        float py = 0.0f;
        CHECK(gfx::project_to_screen(view, start.position, px, py));

        gfx::TransformEditor editor;
        gfx::Transform out;

        editor.begin(
            gfx::EditMode::EDIT_ROTATE,
            start,
            px + 100.0f,
            py,
            view
        );

        editor.set_axis(gfx::EditAxis::EDIT_AXIS_Z);
        editor.update(px, py - 100.0f, view, out);

        const gfx::Vec3 result = gfx::transform_direction(
            out.rotation_matrix(), {1.0f, 0.0f, 0.0f}
        );

        const float facing =
            gfx::dot(gfx::Vec3{0.0f, 0.0f, 1.0f},
                     -view.basis.forward);

        if (facing >= 0.0f) {
            CHECK(approx_vec(result, {0.0f, 1.0f, 0.0f}));
        } else {
            CHECK(approx_vec(result, {0.0f, -1.0f, 0.0f}));
        }
    }
}

void test_rotate_free() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = camera.target;

    float px = 0.0f;
    float py = 0.0f;
    CHECK(gfx::project_to_screen(view, start.position, px, py));

    gfx::TransformEditor editor;
    gfx::Transform out;

    editor.begin(gfx::EditMode::EDIT_ROTATE, start, px + 100, py, view);
    editor.update(px, py - 100, view, out);

    const gfx::Vec3 result = gfx::transform_direction(
        out.rotation_matrix(), view.basis.right
    );

    CHECK(approx_vec(result, view.basis.up));
}

void test_rotate_past_180_degrees() {
    gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera);

    gfx::Transform start;
    start.position = camera.target;

    float px = 0.0f;
    float py = 0.0f;
    CHECK(gfx::project_to_screen(view, start.position, px, py));

    gfx::TransformEditor editor;
    gfx::Transform out;

    editor.begin(gfx::EditMode::EDIT_ROTATE, start, px + 100, py, view);

    for (int degrees = 30; degrees <= 270; degrees += 30) {
        const float angle = gfx::to_radians(
            static_cast<float>(degrees)
        );
        editor.update(
            px + 100.0f * std::cos(angle),
            py - 100.0f * std::sin(angle),
            view,
            out
        );
    }

    const gfx::Vec3 result = gfx::transform_direction(
        out.rotation_matrix(), {1.0f, 0.0f, 0.0f}
    );

    CHECK(approx_vec(result, {0.0f, -1.0f, 0.0f}));
}

void test_euler_round_trip() {
    const gfx::Vec3 angles[] = {
        {0.1f, 0.2f, 0.3f},
        {-0.5f, 0.7f, 1.2f},
        {1.1f, -0.9f, 2.0f},
        {-2.0f, 0.4f, -1.3f}
    };

    for (const gfx::Vec3 angles_in : angles) {
        gfx::Transform t;
        t.rotation = angles_in;

        const gfx::Vec3 recovered =
            gfx::euler_from_matrix(t.rotation_matrix());

        gfx::Transform check_transform;
        check_transform.rotation = recovered;

        CHECK(approx_vec(
            gfx::transform_direction(t.rotation_matrix(), {1, 0, 0}),
            gfx::transform_direction(
                check_transform.rotation_matrix(), {1, 0, 0}
            )
        ));

        CHECK(approx_vec(
            gfx::transform_direction(t.rotation_matrix(), {0, 1, 0}),
            gfx::transform_direction(
                check_transform.rotation_matrix(), {0, 1, 0}
            )
        ));

        CHECK(approx_vec(
            gfx::transform_direction(t.rotation_matrix(), {0, 0, 1}),
            gfx::transform_direction(
                check_transform.rotation_matrix(), {0, 0, 1}
            )
        ));
    }

    gfx::Transform gimbal;
    gimbal.rotation = {0.7f, kHalfPi, -0.4f};

    const gfx::Vec3 recovered =
        gfx::euler_from_matrix(gimbal.rotation_matrix());

    CHECK(std::isfinite(recovered.x));
    CHECK(std::isfinite(recovered.y));
    CHECK(std::isfinite(recovered.z));

    gfx::Transform check_transform;
    check_transform.rotation = recovered;

    CHECK(approx_vec(
        gfx::transform_direction(gimbal.rotation_matrix(), {1, 0, 0}),
        gfx::transform_direction(
            check_transform.rotation_matrix(), {1, 0, 0}
        )
    ));
}

} // namespace

int main() {
    test_projection();
    test_grab_free();
    test_grab_x_constraint();
    test_cancel_restores_start();
    test_axis_toggle();
    test_scale_free_and_y();
    test_scale_clamp_and_pivot_start();
    test_rotate_z_above_and_below();
    test_rotate_free();
    test_rotate_past_180_degrees();
    test_euler_round_trip();

    if (g_failures == 0) {
        std::printf("All transform-edit tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d transform-edit test(s) failed\n", g_failures);
    return 1;
}
