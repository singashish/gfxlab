// G / R / S on the whole selection (core/group_edit.h): the pivot, the
// result for a group, and that one selected object gives exactly what the
// TransformEditor gives for that object alone.

#include "core/group_edit.h"
#include "core/math_util.h"
#include "core/model.h"
#include "core/orbit_camera.h"
#include "core/scene.h"
#include "core/transform_edit.h"
#include "check.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <utility>

namespace {

using gfx::Vec3;

bool approx_vec(Vec3 a, Vec3 b, double eps = 1e-4) {
    return approx(a.x, b.x, eps) && approx(a.y, b.y, eps) &&
           approx(a.z, b.z, eps);
}

bool approx_matrix(const gfx::Mat4& a, const gfx::Mat4& b, double eps = 1e-4) {
    for (int c = 0; c < 4; ++c) {
        if (!approx(a.col[c].x, b.col[c].x, eps) ||
            !approx(a.col[c].y, b.col[c].y, eps) ||
            !approx(a.col[c].z, b.col[c].z, eps) ||
            !approx(a.col[c].w, b.col[c].w, eps)) {
            return false;
        }
    }

    return true;
}

// Same transform (rotations compared as matrices, because two sets of
// Euler angles can describe one rotation)
bool same_transform(const gfx::Transform& a, const gfx::Transform& b) {
    return approx_vec(a.position, b.position) &&
           approx_vec(a.scale, b.scale) &&
           approx_matrix(a.rotation_matrix(), b.rotation_matrix());
}

// A scene with `count` objects (they all share a tiny triangle mesh; the
// group edit only looks at the transforms)
gfx::Scene make_scene(std::size_t count) {
    gfx::Model src;
    gfx::Mesh mesh;
    mesh.positions = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}};
    mesh.indices = {0, 1, 2};
    src.meshes.push_back(mesh);

    for (std::size_t i = 0; i < count; ++i) {
        src.instances.push_back({0, gfx::mat4_identity(), ""});
    }

    gfx::Scene scene;
    gfx::Model model;
    gfx::scene_add_model(scene, model, std::move(src), "t.obj");
    return scene;
}

void put(gfx::Scene& scene, std::size_t i, Vec3 position,
         Vec3 rotation = {0.0f, 0.0f, 0.0f},
         Vec3 scale = {1.0f, 1.0f, 1.0f}) {
    scene.objects[i].transform.position = position;
    scene.objects[i].transform.rotation = rotation;
    scene.objects[i].transform.scale = scale;
}

gfx::EditView make_view() {
    gfx::OrbitCamera camera;
    camera.yaw = gfx::to_radians(-60.0f);
    camera.pitch = gfx::to_radians(20.0f);
    camera.distance = 12.0f;

    return {camera.eye(), camera.basis(), gfx::to_radians(60.0f),
            1280.0f, 720.0f};
}

// ---- group_edit_begin --------------------------------------------------

void test_begin() {
    gfx::Scene scene = make_scene(4);
    put(scene, 0, {0.0f, 0.0f, 0.0f});
    put(scene, 1, {2.0f, 0.0f, 0.0f});
    put(scene, 2, {4.0f, 6.0f, 0.0f});
    put(scene, 3, {100.0f, 100.0f, 100.0f});

    gfx::GroupEdit edit;

    // Nothing selected
    CHECK(!gfx::group_edit_begin(scene, edit));
    CHECK(edit.objects.empty());

    scene.objects[0].selected = true;
    scene.objects[1].selected = true;
    scene.objects[2].selected = true;

    CHECK(gfx::group_edit_begin(scene, edit));
    CHECK(edit.objects.size() == 3);
    CHECK(edit.starts.size() == 3);
    CHECK(approx_vec(edit.pivot, {2.0f, 2.0f, 0.0f}));

    const gfx::Transform pivot = gfx::group_edit_pivot_transform(edit);

    CHECK(approx_vec(pivot.position, edit.pivot));
    CHECK(approx_vec(pivot.rotation, {0.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(pivot.scale, {1.0f, 1.0f, 1.0f}));

    // A hidden object is never part of the group
    scene.objects[2].hidden = true;

    CHECK(gfx::group_edit_begin(scene, edit));
    CHECK(edit.objects.size() == 2);
    CHECK(approx_vec(edit.pivot, {1.0f, 0.0f, 0.0f}));
}

// ---- group_edit_apply --------------------------------------------------

void test_grab() {
    gfx::Scene scene = make_scene(3);
    put(scene, 0, {-2.0f, 0.0f, 0.0f}, {0.1f, 0.2f, 0.3f}, {1.0f, 2.0f, 3.0f});
    put(scene, 1, {2.0f, 0.0f, 0.0f});
    put(scene, 2, {50.0f, 0.0f, 0.0f});
    scene.objects[0].selected = true;
    scene.objects[1].selected = true;

    gfx::GroupEdit edit;
    CHECK(gfx::group_edit_begin(scene, edit));

    gfx::Transform moved = gfx::group_edit_pivot_transform(edit);
    moved.position = moved.position + Vec3{1.0f, 2.0f, 3.0f};

    gfx::group_edit_apply(scene, edit, moved);

    CHECK(approx_vec(scene.objects[0].transform.position, {-1.0f, 2.0f, 3.0f}));
    CHECK(approx_vec(scene.objects[1].transform.position, {3.0f, 2.0f, 3.0f}));

    // Rotation and scale of the objects are untouched
    CHECK(approx_matrix(scene.objects[0].transform.rotation_matrix(),
                        gfx::Transform{{}, {0.1f, 0.2f, 0.3f}, {}}
                            .rotation_matrix()));
    CHECK(approx_vec(scene.objects[0].transform.scale, {1.0f, 2.0f, 3.0f}));

    // The unselected object did not move
    CHECK(approx_vec(scene.objects[2].transform.position, {50.0f, 0.0f, 0.0f}));
}

void test_rotate() {
    gfx::Scene scene = make_scene(2);
    put(scene, 0, {-2.0f, 0.0f, 0.0f});
    put(scene, 1, {2.0f, 0.0f, 0.0f});
    scene.objects[0].selected = true;
    scene.objects[1].selected = true;

    gfx::GroupEdit edit;
    CHECK(gfx::group_edit_begin(scene, edit));

    // A quarter turn around Z, counter-clockwise seen from above
    gfx::Transform turned = gfx::group_edit_pivot_transform(edit);
    turned.rotation = {0.0f, 0.0f, gfx::kPi * 0.5f};

    gfx::group_edit_apply(scene, edit, turned);

    CHECK(approx_vec(scene.objects[0].transform.position, {0.0f, -2.0f, 0.0f}));
    CHECK(approx_vec(scene.objects[1].transform.position, {0.0f, 2.0f, 0.0f}));
    CHECK(approx(scene.objects[0].transform.rotation.z, gfx::kPi * 0.5f));
    CHECK(approx(scene.objects[1].transform.rotation.z, gfx::kPi * 0.5f));

    // The turn composes with an object's own rotation (the group's
    // rotation is applied last, around the world axes)
    put(scene, 0, {-2.0f, 0.0f, 0.0f}, {0.4f, 0.0f, 0.0f});
    put(scene, 1, {2.0f, 0.0f, 0.0f});
    CHECK(gfx::group_edit_begin(scene, edit));
    gfx::group_edit_apply(scene, edit, turned);

    const gfx::Mat4 expected =
        turned.rotation_matrix() *
        gfx::Transform{{}, {0.4f, 0.0f, 0.0f}, {}}.rotation_matrix();

    CHECK(approx_matrix(scene.objects[0].transform.rotation_matrix(), expected));
}

void test_scale() {
    gfx::Scene scene = make_scene(2);
    put(scene, 0, {-2.0f, 1.0f, 0.0f}, {}, {1.0f, 2.0f, 1.0f});
    put(scene, 1, {2.0f, 1.0f, 0.0f});
    scene.objects[0].selected = true;
    scene.objects[1].selected = true;

    gfx::GroupEdit edit;
    CHECK(gfx::group_edit_begin(scene, edit));
    CHECK(approx_vec(edit.pivot, {0.0f, 1.0f, 0.0f}));

    // Twice as big in every direction
    gfx::Transform big = gfx::group_edit_pivot_transform(edit);
    big.scale = {2.0f, 2.0f, 2.0f};

    gfx::group_edit_apply(scene, edit, big);

    CHECK(approx_vec(scene.objects[0].transform.position, {-4.0f, 1.0f, 0.0f}));
    CHECK(approx_vec(scene.objects[1].transform.position, {4.0f, 1.0f, 0.0f}));
    CHECK(approx_vec(scene.objects[0].transform.scale, {2.0f, 4.0f, 2.0f}));
    CHECK(approx_vec(scene.objects[1].transform.scale, {2.0f, 2.0f, 2.0f}));

    // Only along X (the X key): the distances grow along X only
    gfx::Transform along_x = gfx::group_edit_pivot_transform(edit);
    along_x.scale = {3.0f, 1.0f, 1.0f};

    gfx::group_edit_apply(scene, edit, along_x);

    CHECK(approx_vec(scene.objects[0].transform.position, {-6.0f, 1.0f, 0.0f}));
    CHECK(approx_vec(scene.objects[1].transform.position, {6.0f, 1.0f, 0.0f}));
    CHECK(approx_vec(scene.objects[0].transform.scale, {3.0f, 2.0f, 1.0f}));
}

// Applying twice from the same start gives the same result (no
// accumulation), and restore brings everything back
void test_apply_is_repeatable_and_restore_works() {
    gfx::Scene scene = make_scene(3);
    put(scene, 0, {-2.0f, 0.5f, 0.0f}, {0.2f, -0.4f, 0.9f}, {1.0f, 2.0f, 3.0f});
    put(scene, 1, {2.0f, 1.0f, 3.0f}, {}, {2.0f, 2.0f, 2.0f});
    put(scene, 2, {7.0f, 7.0f, 7.0f}, {0.1f, 0.1f, 0.1f});
    scene.objects[0].selected = true;
    scene.objects[1].selected = true;

    const gfx::Transform before0 = scene.objects[0].transform;
    const gfx::Transform before1 = scene.objects[1].transform;
    const gfx::Transform before2 = scene.objects[2].transform;

    gfx::GroupEdit edit;
    CHECK(gfx::group_edit_begin(scene, edit));

    gfx::Transform change = gfx::group_edit_pivot_transform(edit);
    change.position = change.position + Vec3{1.0f, 0.0f, -1.0f};
    change.rotation = {0.3f, 0.2f, 0.1f};
    change.scale = {1.5f, 1.5f, 1.5f};

    gfx::group_edit_apply(scene, edit, change);
    const gfx::Transform first = scene.objects[0].transform;

    gfx::group_edit_apply(scene, edit, change);

    CHECK(same_transform(scene.objects[0].transform, first));

    gfx::group_edit_restore(scene, edit);

    CHECK(same_transform(scene.objects[0].transform, before0));
    CHECK(same_transform(scene.objects[1].transform, before1));
    CHECK(same_transform(scene.objects[2].transform, before2));
}

// ---- one object: the same as the editor on the object itself -----------

void test_single_object_matches_the_editor() {
    const gfx::EditView view = make_view();

    const gfx::EditMode modes[3] = {gfx::EditMode::EDIT_GRAB,
                                    gfx::EditMode::EDIT_ROTATE,
                                    gfx::EditMode::EDIT_SCALE};
    const gfx::EditAxis axes[4] = {gfx::EditAxis::EDIT_AXIS_FREE,
                                   gfx::EditAxis::EDIT_AXIS_X,
                                   gfx::EditAxis::EDIT_AXIS_Y,
                                   gfx::EditAxis::EDIT_AXIS_Z};

    for (const gfx::EditMode mode : modes) {
        for (const gfx::EditAxis axis : axes) {
            gfx::Scene scene = make_scene(1);
            put(scene, 0, {1.0f, 0.5f, 2.0f}, {0.3f, 0.2f, 0.1f},
                {1.0f, 2.0f, 1.5f});
            scene.objects[0].selected = true;

            const gfx::Transform start = scene.objects[0].transform;

            // The editor on the object itself
            gfx::TransformEditor alone;
            alone.begin(mode, start, 400.0f, 300.0f, view);
            alone.set_axis(axis);

            gfx::Transform expected = start;
            alone.update(520.0f, 250.0f, view, expected);

            // The editor on the pivot, applied to the group
            gfx::GroupEdit edit;
            CHECK(gfx::group_edit_begin(scene, edit));

            gfx::TransformEditor grouped;
            grouped.begin(mode, gfx::group_edit_pivot_transform(edit),
                          400.0f, 300.0f, view);
            grouped.set_axis(axis);

            gfx::Transform edited = gfx::group_edit_pivot_transform(edit);
            grouped.update(520.0f, 250.0f, view, edited);

            gfx::group_edit_apply(scene, edit, edited);

            CHECK(same_transform(scene.objects[0].transform, expected));
        }
    }
}

// Cancelling an edit that ran through the editor restores everything
void test_cancel_through_the_editor() {
    const gfx::EditView view = make_view();

    gfx::Scene scene = make_scene(2);
    put(scene, 0, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.3f});
    put(scene, 1, {3.0f, 1.0f, 0.0f});
    scene.objects[0].selected = true;
    scene.objects[1].selected = true;

    const gfx::Transform before0 = scene.objects[0].transform;
    const gfx::Transform before1 = scene.objects[1].transform;

    gfx::GroupEdit edit;
    CHECK(gfx::group_edit_begin(scene, edit));

    gfx::TransformEditor editor;
    editor.begin(gfx::EditMode::EDIT_ROTATE,
                 gfx::group_edit_pivot_transform(edit), 400.0f, 300.0f, view);

    gfx::Transform edited = gfx::group_edit_pivot_transform(edit);
    editor.update(450.0f, 200.0f, view, edited);
    gfx::group_edit_apply(scene, edit, edited);

    CHECK(!same_transform(scene.objects[0].transform, before0));

    (void)editor.cancel();
    gfx::group_edit_restore(scene, edit);

    CHECK(same_transform(scene.objects[0].transform, before0));
    CHECK(same_transform(scene.objects[1].transform, before1));
}

} // namespace

int main() {
    test_begin();
    test_grab();
    test_rotate();
    test_scale();
    test_apply_is_repeatable_and_restore_works();
    test_single_object_matches_the_editor();
    test_cancel_through_the_editor();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }

    std::printf("group edit tests passed\n");
    return 0;
}
