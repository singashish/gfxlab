// Box select (core/box_select.h): the grid of sample points, the objects
// found through a rectangle (occluded and hidden ones are not), and how
// scene_box_select (core/scene.h) changes the selection.
//
// The rays are not parallel to an axis here, but test_pick.cpp has the
// same MSVC remark: a zero direction component can trip C4723 in Release
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

#include "core/box_select.h"
#include "core/math_util.h"
#include "core/model.h"
#include "core/orbit_camera.h"
#include "core/scene.h"
#include "check.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

namespace {

using gfx::Vec3;

constexpr float kWidth = 1280.0f;
constexpr float kHeight = 720.0f;

gfx::EditView make_view(const gfx::OrbitCamera& camera) {
    return {
        camera.eye(),
        camera.basis(),
        gfx::to_radians(60.0f),
        kWidth,
        kHeight
    };
}

// A cube from -half to +half on every axis
gfx::Mesh make_cube(float half) {
    gfx::Mesh mesh;

    for (int corner = 0; corner < 8; ++corner) {
        mesh.positions.push_back({
            (corner & 1) ? half : -half,
            (corner & 2) ? half : -half,
            (corner & 4) ? half : -half
        });
    }

    const std::uint32_t quads[6][4] = {
        {0, 1, 3, 2}, {4, 5, 7, 6},
        {0, 1, 5, 4}, {2, 3, 7, 6},
        {0, 2, 6, 4}, {1, 3, 7, 5}
    };

    for (const auto& quad : quads) {
        mesh.indices.insert(mesh.indices.end(),
                            {quad[0], quad[1], quad[2],
                             quad[0], quad[2], quad[3]});
    }

    return mesh;
}

struct CubeScene {
    gfx::Model model;
    gfx::Scene scene;
};

CubeScene make_cube_scene(std::size_t count) {
    gfx::Model src;
    src.meshes.push_back(make_cube(1.0f));

    for (std::size_t i = 0; i < count; ++i) {
        src.instances.push_back({0, gfx::mat4_identity(), ""});
    }

    CubeScene result;
    gfx::scene_add_model(result.scene, result.model, std::move(src),
                         "cube.obj");
    return result;
}

void place(CubeScene& s, std::size_t i, Vec3 position,
           float scale = 1.0f) {
    gfx::SceneObject& object = s.scene.objects[i];

    object.local = gfx::mat4_identity();
    object.transform.position = position;
    object.transform.rotation = {0.0f, 0.0f, 0.0f};
    object.transform.scale = {scale, scale, scale};
}

// Camera at (0, -15, 0) looking along +y: +x is to the right
gfx::OrbitCamera front_camera() {
    gfx::OrbitCamera camera;
    camera.yaw = gfx::to_radians(-90.0f);
    camera.pitch = gfx::to_radians(0.0f);
    camera.distance = 15.0f;
    return camera;
}

std::vector<int> box_around(const CubeScene& s, const gfx::EditView& view,
                            gfx::PickIndex& index, Vec3 world,
                            float half_pixels) {
    float px = 0.0f;
    float py = 0.0f;

    CHECK(gfx::project_to_screen(view, world, px, py));

    return gfx::box_select_objects(s.scene, s.model, index, view,
                                   px - half_pixels, py - half_pixels,
                                   px + half_pixels, py + half_pixels);
}

// ---- make_box_grid -----------------------------------------------------

void test_grid_basic() {
    const gfx::BoxGrid grid =
        gfx::make_box_grid(10.0f, 20.0f, 50.0f, 60.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(grid.nx == 10);
    CHECK(grid.ny == 10);
    CHECK(approx(grid.step_x, 4.0));
    CHECK(approx(grid.step_y, 4.0));
    CHECK(approx(grid.x0, 10.0));
    CHECK(approx(grid.y0, 20.0));

    // Not a multiple of the spacing: the spacing shrinks, never grows
    const gfx::BoxGrid odd =
        gfx::make_box_grid(0.0f, 0.0f, 30.0f, 10.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(odd.nx == 8);
    CHECK(odd.ny == 3);
    CHECK(odd.step_x <= 4.0f);
    CHECK(odd.step_y <= 4.0f);
    CHECK(approx(odd.step_x * static_cast<float>(odd.nx), 30.0, 1e-4));
}

void test_grid_corner_order() {
    const gfx::BoxGrid a =
        gfx::make_box_grid(10.0f, 20.0f, 50.0f, 60.0f, kWidth, kHeight,
                           4.0f, 16384);
    const gfx::BoxGrid b =
        gfx::make_box_grid(50.0f, 60.0f, 10.0f, 20.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(a.nx == b.nx);
    CHECK(a.ny == b.ny);
    CHECK(approx(a.x0, b.x0));
    CHECK(approx(a.y0, b.y0));
}

void test_grid_cap() {
    // The whole window at 4 px would be 320 x 180 = 57600 rays
    const gfx::BoxGrid grid =
        gfx::make_box_grid(0.0f, 0.0f, kWidth, kHeight, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(grid.nx > 0 && grid.ny > 0);
    CHECK(static_cast<std::size_t>(grid.nx) *
              static_cast<std::size_t>(grid.ny) <= 16384u);

    // Still covers the whole box
    CHECK(approx(grid.step_x * static_cast<float>(grid.nx), kWidth, 1e-2));
    CHECK(approx(grid.step_y * static_cast<float>(grid.ny), kHeight, 1e-2));

    // A tiny cap still gives at least one ray
    const gfx::BoxGrid one =
        gfx::make_box_grid(0.0f, 0.0f, kWidth, kHeight, kWidth, kHeight,
                           4.0f, 0);

    CHECK(one.nx == 1 && one.ny == 1);

    const gfx::BoxGrid four =
        gfx::make_box_grid(0.0f, 0.0f, kWidth, kHeight, kWidth, kHeight,
                           4.0f, 4);

    CHECK(four.nx >= 1 && four.ny >= 1);
    CHECK(static_cast<std::size_t>(four.nx) *
              static_cast<std::size_t>(four.ny) <= 4u);
}

void test_grid_clipping() {
    // Partly outside: cut to the window
    const gfx::BoxGrid part =
        gfx::make_box_grid(-100.0f, -100.0f, 20.0f, 20.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(approx(part.x0, 0.0));
    CHECK(approx(part.y0, 0.0));
    CHECK(part.nx == 5 && part.ny == 5);

    // Completely outside, on each side
    CHECK(gfx::make_box_grid(-50.0f, 10.0f, -5.0f, 40.0f, kWidth, kHeight,
                             4.0f, 16384).nx == 0);
    CHECK(gfx::make_box_grid(kWidth + 5.0f, 10.0f, kWidth + 50.0f, 40.0f,
                             kWidth, kHeight, 4.0f, 16384).nx == 0);
    CHECK(gfx::make_box_grid(10.0f, -50.0f, 40.0f, -5.0f, kWidth, kHeight,
                             4.0f, 16384).ny == 0);
    CHECK(gfx::make_box_grid(10.0f, kHeight + 5.0f, 40.0f, kHeight + 50.0f,
                             kWidth, kHeight, 4.0f, 16384).ny == 0);

    // Unusable window or NaN
    CHECK(gfx::make_box_grid(0.0f, 0.0f, 10.0f, 10.0f, 0.0f, kHeight,
                             4.0f, 16384).nx == 0);
    const float nan = std::numeric_limits<float>::quiet_NaN();
    CHECK(gfx::make_box_grid(nan, 0.0f, 10.0f, 10.0f, kWidth, kHeight,
                             4.0f, 16384).nx == 0);
}

void test_grid_degenerate_box() {
    // No height: one row of rays along the line
    const gfx::BoxGrid line =
        gfx::make_box_grid(10.0f, 30.0f, 50.0f, 30.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(line.nx == 10);
    CHECK(line.ny == 1);
    CHECK(approx(line.y0, 30.0));
    CHECK(approx(line.step_y, 0.0));

    // A single point: one ray
    const gfx::BoxGrid point =
        gfx::make_box_grid(10.0f, 30.0f, 10.0f, 30.0f, kWidth, kHeight,
                           4.0f, 16384);

    CHECK(point.nx == 1 && point.ny == 1);
}

// ---- box_select_objects ------------------------------------------------

void test_box_finds_objects() {
    CubeScene s = make_cube_scene(2);
    place(s, 0, {-3.0f, 0.0f, 0.0f});
    place(s, 1, {3.0f, 0.0f, 0.0f});

    const gfx::OrbitCamera camera = front_camera();
    const gfx::EditView view = make_view(camera);
    gfx::PickIndex index;

    // A box inside the left cube, inside the right cube
    const std::vector<int> left =
        box_around(s, view, index, {-3.0f, 0.0f, 0.0f}, 20.0f);
    const std::vector<int> right =
        box_around(s, view, index, {3.0f, 0.0f, 0.0f}, 20.0f);

    CHECK(left.size() == 1 && left[0] == 0);
    CHECK(right.size() == 1 && right[0] == 1);

    // Empty space between the cubes
    CHECK(box_around(s, view, index, {0.0f, 0.0f, 0.0f}, 20.0f).empty());

    // A box over both, given with the corners swapped
    float lx = 0.0f, ly = 0.0f, rx = 0.0f, ry = 0.0f;
    CHECK(gfx::project_to_screen(view, {-3.0f, 0.0f, 0.0f}, lx, ly));
    CHECK(gfx::project_to_screen(view, {3.0f, 0.0f, 0.0f}, rx, ry));

    const std::vector<int> both = gfx::box_select_objects(
        s.scene, s.model, index, view, rx + 10.0f, ry + 10.0f,
        lx - 10.0f, ly - 10.0f);

    CHECK(both.size() == 2 && both[0] == 0 && both[1] == 1);

    // The whole window
    const std::vector<int> all = gfx::box_select_objects(
        s.scene, s.model, index, view, 0.0f, 0.0f, kWidth, kHeight);

    CHECK(all.size() == 2);

    // A box far outside the window
    CHECK(gfx::box_select_objects(s.scene, s.model, index, view,
                                  -500.0f, -500.0f, -100.0f, -100.0f)
              .empty());
}

// A box that overlaps only the edge of an object still finds it
void test_box_partly_over_an_object() {
    CubeScene s = make_cube_scene(1);
    place(s, 0, {0.0f, 0.0f, 0.0f});

    const gfx::EditView view = make_view(front_camera());
    gfx::PickIndex index;

    float cx = 0.0f, cy = 0.0f;
    CHECK(gfx::project_to_screen(view, {0.0f, 0.0f, 0.0f}, cx, cy));

    // The cube is about 120 px wide here: a box from far to the left to
    // a point inside it
    const std::vector<int> hit = gfx::box_select_objects(
        s.scene, s.model, index, view, cx - 400.0f, cy - 10.0f,
        cx - 20.0f, cy + 10.0f);

    CHECK(hit.size() == 1);

    // A box that ends before the cube begins
    const std::vector<int> miss = gfx::box_select_objects(
        s.scene, s.model, index, view, cx - 400.0f, cy - 10.0f,
        cx - 300.0f, cy + 10.0f);

    CHECK(miss.empty());
}

// An object that is completely hidden behind another is not found; the
// big one behind a small one is found where it sticks out
void test_box_ignores_occluded_objects() {
    CubeScene s = make_cube_scene(2);
    place(s, 0, {0.0f, -4.0f, 0.0f}, 1.0f); // near the camera, small
    place(s, 1, {0.0f, 4.0f, 0.0f}, 3.0f);  // behind it, big

    const gfx::EditView view = make_view(front_camera());
    gfx::PickIndex index;

    float cx = 0.0f, cy = 0.0f;
    CHECK(gfx::project_to_screen(view, {0.0f, -4.0f, 0.0f}, cx, cy));

    // A small box in the middle sees only the near cube
    const std::vector<int> middle = gfx::box_select_objects(
        s.scene, s.model, index, view, cx - 5.0f, cy - 5.0f,
        cx + 5.0f, cy + 5.0f);

    CHECK(middle.size() == 1 && middle[0] == 0);

    // The whole window sees both
    const std::vector<int> all = gfx::box_select_objects(
        s.scene, s.model, index, view, 0.0f, 0.0f, kWidth, kHeight);

    CHECK(all.size() == 2);

    // A cube exactly behind a bigger one is never found
    place(s, 1, {0.0f, 4.0f, 0.0f}, 0.5f);
    place(s, 0, {0.0f, -4.0f, 0.0f}, 2.0f);

    const std::vector<int> hidden = gfx::box_select_objects(
        s.scene, s.model, index, view, 0.0f, 0.0f, kWidth, kHeight);

    CHECK(hidden.size() == 1 && hidden[0] == 0);
}

void test_box_skips_hidden_objects() {
    CubeScene s = make_cube_scene(2);
    place(s, 0, {-3.0f, 0.0f, 0.0f});
    place(s, 1, {3.0f, 0.0f, 0.0f});
    s.scene.objects[1].hidden = true;

    const gfx::EditView view = make_view(front_camera());
    gfx::PickIndex index;

    const std::vector<int> all = gfx::box_select_objects(
        s.scene, s.model, index, view, 0.0f, 0.0f, kWidth, kHeight);

    CHECK(all.size() == 1 && all[0] == 0);
}

// With very few rays the result is still valid (it may just miss things)
void test_box_with_few_rays() {
    CubeScene s = make_cube_scene(1);
    place(s, 0, {0.0f, 0.0f, 0.0f});

    const gfx::EditView view = make_view(front_camera());
    gfx::PickIndex index;

    // One ray in the middle of the window goes through the cube
    const std::vector<int> one = gfx::box_select_objects(
        s.scene, s.model, index, view, 0.0f, 0.0f, kWidth, kHeight,
        4.0f, 1);

    CHECK(one.size() == 1);

    CubeScene empty;
    CHECK(gfx::box_select_objects(empty.scene, empty.model, index, view,
                                  0.0f, 0.0f, kWidth, kHeight).empty());
}

// ---- scene_box_select --------------------------------------------------

CubeScene three_objects() {
    CubeScene s = make_cube_scene(3);
    place(s, 0, {-4.0f, 0.0f, 0.0f});
    place(s, 1, {0.0f, 0.0f, 0.0f});
    place(s, 2, {4.0f, 0.0f, 0.0f});
    return s;
}

void test_select_replaces() {
    CubeScene s = three_objects();

    gfx::scene_click_select(s.scene, 2, false); // 2 selected and active

    gfx::scene_box_select(s.scene, {0, 1}, false);

    CHECK(s.scene.objects[0].selected);
    CHECK(s.scene.objects[1].selected);
    CHECK(!s.scene.objects[2].selected);
    CHECK(gfx::scene_selected_count(s.scene) == 2);

    // Like Blender, a box select does not change the active object
    CHECK(s.scene.active == 2);
}

void test_select_extends() {
    CubeScene s = three_objects();

    gfx::scene_click_select(s.scene, 0, false);

    gfx::scene_box_select(s.scene, {1}, true);

    CHECK(s.scene.objects[0].selected);
    CHECK(s.scene.objects[1].selected);
    CHECK(!s.scene.objects[2].selected);
    CHECK(s.scene.active == 0);

    // Extending with an object that is already selected changes nothing,
    // and Shift never deselects
    gfx::scene_box_select(s.scene, {0, 1}, true);

    CHECK(gfx::scene_selected_count(s.scene) == 2);

    // An empty box with Shift keeps the selection
    gfx::scene_box_select(s.scene, {}, true);

    CHECK(gfx::scene_selected_count(s.scene) == 2);
}

void test_select_empty_box_deselects() {
    CubeScene s = three_objects();

    gfx::scene_select_all(s.scene);
    s.scene.active = 1;

    gfx::scene_box_select(s.scene, {}, false);

    CHECK(gfx::scene_selected_count(s.scene) == 0);
    CHECK(s.scene.active == 1); // kept, now unselected
}

void test_select_ignores_bad_indices() {
    CubeScene s = three_objects();

    s.scene.objects[1].hidden = true;

    gfx::scene_box_select(s.scene, {-1, 1, 7, 2}, false);

    CHECK(!s.scene.objects[0].selected);
    CHECK(!s.scene.objects[1].selected); // hidden
    CHECK(s.scene.objects[2].selected);
    CHECK(gfx::scene_selected_count(s.scene) == 1);
}

// The whole chain: a box in the view, then the selection
void test_box_then_select() {
    CubeScene s = three_objects();

    const gfx::EditView view = make_view(front_camera());
    gfx::PickIndex index;

    float ax = 0.0f, ay = 0.0f, bx = 0.0f, by = 0.0f;
    CHECK(gfx::project_to_screen(view, {-4.0f, 0.0f, 0.0f}, ax, ay));
    CHECK(gfx::project_to_screen(view, {0.0f, 0.0f, 0.0f}, bx, by));

    const std::vector<int> found = gfx::box_select_objects(
        s.scene, s.model, index, view, ax - 15.0f, ay - 15.0f,
        bx + 15.0f, by + 15.0f);

    gfx::scene_box_select(s.scene, found, false);

    CHECK(s.scene.objects[0].selected);
    CHECK(s.scene.objects[1].selected);
    CHECK(!s.scene.objects[2].selected);
}

} // namespace

int main() {
    test_grid_basic();
    test_grid_corner_order();
    test_grid_cap();
    test_grid_clipping();
    test_grid_degenerate_box();
    test_box_finds_objects();
    test_box_partly_over_an_object();
    test_box_ignores_occluded_objects();
    test_box_skips_hidden_objects();
    test_box_with_few_rays();
    test_select_replaces();
    test_select_extends();
    test_select_empty_box_deselects();
    test_select_ignores_bad_indices();
    test_box_then_select();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }

    std::printf("box select tests passed\n");
    return 0;
}
