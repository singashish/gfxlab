// The tests use rays parallel to an axis (a zero direction component makes
// ray_inverse_direction return infinity, which the slab test relies on).
// After inlining, MSVC's optimizer cannot see that guard and reports C4723
// "potential divide by 0" in Release builds (the same as in test_bvh.cpp)
#ifdef _MSC_VER
#pragma warning(disable : 4723)
#endif

#include "core/pick.h"
#include "core/math_util.h"
#include "core/model.h"
#include "core/orbit_camera.h"
#include "core/ray.h"
#include "core/scene.h"
#include "check.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

using gfx::Vec3;

// Small deterministic random generator, so failures are repeatable
struct Lcg {
    std::uint32_t state = 777u;

    float next01() {
        state = state * 1664525u + 1013904223u;
        return static_cast<float>(state >> 8) / 16777216.0f;
    }

    float range(float lo, float hi) {
        return lo + (hi - lo) * next01();
    }
};

bool approx_vec(Vec3 a, Vec3 b, double eps = 1e-4) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

gfx::EditView make_view(const gfx::OrbitCamera& camera, float fov_degrees,
                        float width, float height) {
    return {
        camera.eye(),
        camera.basis(),
        gfx::to_radians(fov_degrees),
        width,
        height
    };
}

// A cube from -half to +half on every axis: 8 vertices, 12 triangles
gfx::Mesh make_cube(float half) {
    gfx::Mesh mesh;

    for (int corner = 0; corner < 8; ++corner) {
        mesh.positions.push_back({
            (corner & 1) ? half : -half,
            (corner & 2) ? half : -half,
            (corner & 4) ? half : -half
        });
    }

    // Two triangles per face; the corner number bits are x = 1, y = 2, z = 4
    const std::uint32_t quads[6][4] = {
        {0, 1, 3, 2}, {4, 5, 7, 6}, // z = -half, z = +half
        {0, 1, 5, 4}, {2, 3, 7, 6}, // y = -half, y = +half
        {0, 2, 6, 4}, {1, 3, 7, 5}  // x = -half, x = +half
    };

    for (const auto& quad : quads) {
        mesh.indices.insert(mesh.indices.end(),
                            {quad[0], quad[1], quad[2],
                             quad[0], quad[2], quad[3]});
    }

    return mesh;
}

// `count` objects that all use one cube mesh (-1..1). scene_add_model lays
// them out; the tests then place them with place()
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
           Vec3 rotation = {0.0f, 0.0f, 0.0f},
           Vec3 scale = {1.0f, 1.0f, 1.0f}) {
    gfx::SceneObject& object = s.scene.objects[i];

    object.local = gfx::mat4_identity();
    object.transform.position = position;
    object.transform.rotation = rotation;
    object.transform.scale = scale;
}

// A ray straight down from (x, y, 10)
gfx::PickRay down_ray(float x, float y, float z = 10.0f) {
    return {{x, y, z}, {0.0f, 0.0f, -1.0f}};
}

// ---- make_pick_ray -----------------------------------------------------

void test_ray_through_the_centre() {
    const gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera, 60.0f, 1280.0f, 720.0f);

    gfx::PickRay ray;

    CHECK(gfx::make_pick_ray(view, 640.0f, 360.0f, ray));
    CHECK(approx_vec(ray.origin, camera.eye()));
    CHECK(approx_vec(ray.dir, view.basis.forward));
    CHECK(approx(gfx::length(ray.dir), 1.0));
}

void test_ray_corners() {
    const gfx::OrbitCamera camera;
    const gfx::EditView view = make_view(camera, 60.0f, 1280.0f, 720.0f);

    const float tan_half = std::tan(gfx::to_radians(60.0f) * 0.5f);
    const float aspect = 1280.0f / 720.0f;

    gfx::PickRay ray;

    // Top-left pixel: left of and above the view direction
    CHECK(gfx::make_pick_ray(view, 0.0f, 0.0f, ray));

    const float along = gfx::dot(ray.dir, view.basis.forward);

    CHECK(approx(gfx::dot(ray.dir, view.basis.right) / along,
                 -tan_half * aspect, 1e-4));
    CHECK(approx(gfx::dot(ray.dir, view.basis.up) / along,
                 tan_half, 1e-4));

    // Bottom-right pixel: the mirror image
    CHECK(gfx::make_pick_ray(view, 1280.0f, 720.0f, ray));

    const float along2 = gfx::dot(ray.dir, view.basis.forward);

    CHECK(approx(gfx::dot(ray.dir, view.basis.right) / along2,
                 tan_half * aspect, 1e-4));
    CHECK(approx(gfx::dot(ray.dir, view.basis.up) / along2,
                 -tan_half, 1e-4));
}

// The ray for the pixel where a point is drawn passes through the point
void test_ray_inverts_the_projection() {
    gfx::OrbitCamera camera;
    camera.yaw = gfx::to_radians(30.0f);
    camera.pitch = gfx::to_radians(-10.0f);
    camera.distance = 12.0f;

    const float sizes[3][2] = {{1280.0f, 720.0f}, {800.0f, 600.0f},
                               {400.0f, 900.0f}};
    const float fovs[3] = {60.0f, 45.0f, 90.0f};

    Lcg rng;
    int tested = 0;

    for (int k = 0; k < 3; ++k) {
        const gfx::EditView view =
            make_view(camera, fovs[k], sizes[k][0], sizes[k][1]);

        for (int i = 0; i < 100; ++i) {
            const Vec3 point = camera.eye()
                + view.basis.forward * rng.range(1.0f, 30.0f)
                + view.basis.right * rng.range(-8.0f, 8.0f)
                + view.basis.up * rng.range(-8.0f, 8.0f);

            float px = 0.0f;
            float py = 0.0f;

            if (!gfx::project_to_screen(view, point, px, py)) {
                continue;
            }

            gfx::PickRay ray;

            CHECK(gfx::make_pick_ray(view, px, py, ray));

            // Distance from the point to the ray
            const Vec3 to_point = point - ray.origin;
            const float along = gfx::dot(to_point, ray.dir);
            const Vec3 off = to_point - ray.dir * along;

            CHECK(along > 0.0f);
            CHECK(gfx::length(off) < 1.0e-3f * (1.0f + along));
            ++tested;
        }
    }

    CHECK(tested > 200);
}

void test_ray_rejects_bad_views() {
    const gfx::OrbitCamera camera;
    gfx::PickRay ray;

    CHECK(!gfx::make_pick_ray(make_view(camera, 60.0f, 0.0f, 720.0f),
                              1.0f, 1.0f, ray));
    CHECK(!gfx::make_pick_ray(make_view(camera, 60.0f, 1280.0f, 0.0f),
                              1.0f, 1.0f, ray));
    CHECK(!gfx::make_pick_ray(make_view(camera, 0.0f, 1280.0f, 720.0f),
                              1.0f, 1.0f, ray));
    CHECK(!gfx::make_pick_ray(make_view(camera, 180.0f, 1280.0f, 720.0f),
                              1.0f, 1.0f, ray));
}

// ---- pick_object: cases with known answers ----------------------------

void test_hit_miss_and_sides() {
    CubeScene s = make_cube_scene(1);
    place(s, 0, {0.0f, 0.0f, 0.0f});

    gfx::PickIndex index;
    gfx::PickHit hit;

    // From above: the top face at z = 1, nine units away
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(0.0f, 0.0f),
                           hit));
    CHECK(hit.object == 0);
    CHECK(approx(hit.t, 9.0));
    CHECK(approx_vec(hit.point, {0.0f, 0.0f, 1.0f}));
    CHECK(hit.triangle < 12);

    // Beside the cube: a miss leaves the hit as it was
    hit.object = 42;
    hit.t = 5.0f;
    CHECK(!gfx::pick_object(s.scene, s.model, index, down_ray(1.5f, 0.0f),
                            hit));
    CHECK(hit.object == 42);
    CHECK(approx(hit.t, 5.0));

    // Pointing away from the cube: nothing behind the origin counts
    const gfx::PickRay away = {{0.0f, 0.0f, 10.0f}, {0.0f, 0.0f, 1.0f}};
    CHECK(!gfx::pick_object(s.scene, s.model, index, away, hit));

    // From below: the back side of the cube is picked as well
    const gfx::PickRay up = {{0.0f, 0.0f, -10.0f}, {0.0f, 0.0f, 1.0f}};
    CHECK(gfx::pick_object(s.scene, s.model, index, up, hit));
    CHECK(approx(hit.t, 9.0));
    CHECK(approx_vec(hit.point, {0.0f, 0.0f, -1.0f}));
}

void test_object_transforms() {
    CubeScene s = make_cube_scene(1);
    gfx::PickIndex index;
    gfx::PickHit hit;

    // Moved: the old place is empty, the new place is hit
    place(s, 0, {5.0f, 0.0f, 0.0f});
    CHECK(!gfx::pick_object(s.scene, s.model, index, down_ray(0.0f, 0.0f),
                            hit));
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(5.0f, 0.0f),
                           hit));
    CHECK(approx(hit.t, 9.0));

    // Scaled by 3: the top is at z = 3, and the box is 3 wide
    place(s, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {3.0f, 3.0f, 3.0f});
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(2.5f, 2.5f),
                           hit));
    CHECK(approx(hit.t, 7.0));
    CHECK(approx_vec(hit.point, {2.5f, 2.5f, 3.0f}));
    CHECK(!gfx::pick_object(s.scene, s.model, index, down_ray(3.5f, 0.0f),
                            hit));

    // An oblique, normalized ray against the scaled cube: t is still a
    // world distance, so the hit point lies on the world-space top face
    const Vec3 oblique = gfx::normalize({0.1f, 0.0f, -1.0f});
    CHECK(gfx::pick_object(s.scene, s.model, index,
                           {{0.0f, 0.0f, 10.0f}, oblique}, hit));
    CHECK(approx(hit.point.z, 3.0, 1e-3));
    CHECK(approx_vec(hit.point,
                     Vec3{0.0f, 0.0f, 10.0f} + oblique * hit.t, 1e-3));

    // Stretched along x by 2 and turned 45 degrees about z. In its own
    // space it is 4 long (x) and 2 wide (y)
    place(s, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, gfx::kPi * 0.25f},
          {2.0f, 1.0f, 1.0f});
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(1.0f, 0.0f),
                           hit));
    CHECK(approx(hit.t, 9.0));

    // Along the long diagonal (1.9 / sqrt 2 = 1.34 from the middle, inside
    // the length 2) it is hit; at the same distance sideways it is not
    CHECK(gfx::pick_object(s.scene, s.model, index,
                           down_ray(1.3f, 1.3f), hit));
    CHECK(!gfx::pick_object(s.scene, s.model, index,
                            down_ray(1.3f, -1.3f), hit));
    CHECK(!gfx::pick_object(s.scene, s.model, index,
                            down_ray(1.9f, 0.0f), hit));

    // Mirrored (negative scale): same shape, still picked
    place(s, 0, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},
          {-1.0f, 1.0f, 1.0f});
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(0.5f, 0.5f),
                           hit));
    CHECK(approx(hit.t, 9.0));
}

void test_nearest_object_wins() {
    CubeScene s = make_cube_scene(2);
    gfx::PickIndex index;
    gfx::PickHit hit;

    // Object 1 is above object 0: seen from above it is nearer
    place(s, 0, {0.0f, 0.0f, 0.0f});
    place(s, 1, {0.0f, 0.0f, 5.0f});

    CHECK(gfx::pick_object(s.scene, s.model, index,
                           down_ray(0.0f, 0.0f, 20.0f), hit));
    CHECK(hit.object == 1);
    CHECK(approx(hit.t, 14.0));

    // Swap them: the winner is the other index (the list order does not
    // matter)
    place(s, 0, {0.0f, 0.0f, 5.0f});
    place(s, 1, {0.0f, 0.0f, 0.0f});

    CHECK(gfx::pick_object(s.scene, s.model, index,
                           down_ray(0.0f, 0.0f, 20.0f), hit));
    CHECK(hit.object == 0);
    CHECK(approx(hit.t, 14.0));

    // Hidden objects cannot be picked: the one behind it is
    s.scene.objects[0].hidden = true;
    CHECK(gfx::pick_object(s.scene, s.model, index,
                           down_ray(0.0f, 0.0f, 20.0f), hit));
    CHECK(hit.object == 1);
    CHECK(approx(hit.t, 19.0));

    s.scene.objects[1].hidden = true;
    CHECK(!gfx::pick_object(s.scene, s.model, index,
                            down_ray(0.0f, 0.0f, 20.0f), hit));
}

void test_unpickable_objects_are_skipped() {
    CubeScene s = make_cube_scene(3);
    gfx::PickIndex index;
    gfx::PickHit hit;

    place(s, 0, {0.0f, 0.0f, 0.0f});
    place(s, 1, {0.0f, 0.0f, 3.0f});
    place(s, 2, {0.0f, 0.0f, 6.0f});

    // A flat object (zero scale) has no inverse: skipped, no crash
    s.scene.objects[2].transform.scale = {0.0f, 1.0f, 1.0f};

    // An object whose mesh does not exist
    s.scene.objects[1].mesh = 99;

    CHECK(gfx::pick_object(s.scene, s.model, index,
                           down_ray(0.0f, 0.0f, 20.0f), hit));
    CHECK(hit.object == 0);
    CHECK(approx(hit.t, 19.0));

    // No objects at all
    gfx::Scene empty;
    CHECK(!gfx::pick_object(empty, s.model, index,
                            down_ray(0.0f, 0.0f), hit));
}

// A ray from the camera through a pixel finds the object drawn there
void test_pick_through_a_pixel() {
    CubeScene s = make_cube_scene(2);
    place(s, 0, {-3.0f, 0.0f, 0.0f});
    place(s, 1, {3.0f, 0.0f, 0.0f});

    gfx::OrbitCamera camera;
    camera.yaw = gfx::to_radians(-90.0f); // looking along +y
    camera.pitch = gfx::to_radians(0.0f);
    camera.distance = 15.0f;

    const gfx::EditView view = make_view(camera, 60.0f, 1280.0f, 720.0f);
    gfx::PickIndex index;

    for (std::size_t i = 0; i < 2; ++i) {
        float px = 0.0f;
        float py = 0.0f;

        CHECK(gfx::project_to_screen(
            view, s.scene.objects[i].transform.position, px, py));

        gfx::PickRay ray;
        gfx::PickHit hit;

        CHECK(gfx::make_pick_ray(view, px, py, ray));
        CHECK(gfx::pick_object(s.scene, s.model, index, ray, hit));
        CHECK(hit.object == static_cast<int>(i));
    }

    // The middle between the two cubes is empty
    float mx = 0.0f;
    float my = 0.0f;
    gfx::PickRay ray;
    gfx::PickHit hit;

    CHECK(gfx::project_to_screen(view, {0.0f, 0.0f, 0.0f}, mx, my));
    CHECK(gfx::make_pick_ray(view, mx, my, ray));
    CHECK(!gfx::pick_object(s.scene, s.model, index, ray, hit));
}

// ---- PickIndex ---------------------------------------------------------

void test_pick_index() {
    CubeScene s = make_cube_scene(1);
    place(s, 0, {0.0f, 0.0f, 0.0f});

    gfx::PickIndex index;

    const gfx::Bvh& first = index.bvh(s.model, 0);

    CHECK(!first.nodes.empty());

    // Built once, then kept
    CHECK(&index.bvh(s.model, 0) == &first);

    // A mesh number outside the model: an empty BVH
    CHECK(index.bvh(s.model, 5).nodes.empty());

    // A second file is appended to the model: its mesh gets its own BVH
    // and picking finds both
    gfx::Model more;
    more.meshes.push_back(make_cube(1.0f));
    more.instances.push_back({0, gfx::mat4_identity(), ""});
    gfx::scene_add_model(s.scene, s.model, std::move(more), "more.obj");
    place(s, 1, {10.0f, 0.0f, 0.0f});

    CHECK(s.scene.objects[1].mesh == 1);
    CHECK(!index.bvh(s.model, 1).nodes.empty());

    gfx::PickHit hit;

    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(10.0f, 0.0f),
                           hit));
    CHECK(hit.object == 1);
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(0.0f, 0.0f),
                           hit));
    CHECK(hit.object == 0);

    // clear() forgets; the next query rebuilds
    index.clear();
    CHECK(gfx::pick_object(s.scene, s.model, index, down_ray(10.0f, 0.0f),
                           hit));
    CHECK(hit.object == 1);
}

// ---- Against the brute-force answer ------------------------------------

// Reference: move every triangle of every visible object into the world and
// test them all
bool brute_force_pick(const gfx::Scene& scene, const gfx::Model& model,
                      const gfx::PickRay& ray, int& object, float& best_t) {
    bool found = false;
    best_t = 1.0e30f;

    for (std::size_t i = 0; i < scene.objects.size(); ++i) {
        const gfx::SceneObject& o = scene.objects[i];

        if (o.hidden) {
            continue;
        }

        const gfx::Mesh& mesh = model.meshes[o.mesh];
        const gfx::Mat4 world = o.world_matrix();

        for (std::size_t t = 0; t + 2 < mesh.indices.size(); t += 3) {
            const Vec3 a =
                gfx::transform_point(world, mesh.positions[mesh.indices[t]]);
            const Vec3 b = gfx::transform_point(
                world, mesh.positions[mesh.indices[t + 1]]);
            const Vec3 c = gfx::transform_point(
                world, mesh.positions[mesh.indices[t + 2]]);

            float hit_t, u, v;

            if (gfx::ray_hits_triangle(ray.origin, ray.dir, a, b, c, 0.0f,
                                       best_t, hit_t, u, v)) {
                best_t = hit_t;
                object = static_cast<int>(i);
                found = true;
            }
        }
    }

    return found;
}

void test_against_brute_force() {
    const std::string dir = std::string(GFX_ASSETS_DIR);

    gfx::Model first;
    gfx::Model second;
    std::string error;

    CHECK(gfx::load_model(dir + "/suzanne_s.obj", first, error));
    CHECK(gfx::load_model(dir + "/gltf/materials.glb", second, error));

    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(scene, model, std::move(first), "suzanne_s.obj");
    gfx::scene_add_model(scene, model, std::move(second), "materials.glb");

    CHECK(scene.objects.size() == 4);

    // Awkward transforms: turned, stretched and mirrored
    scene.objects[0].transform.rotation = {0.4f, -0.7f, 1.1f};
    scene.objects[0].transform.scale = {1.5f, 0.8f, 1.2f};
    scene.objects[1].transform.rotation = {0.0f, 0.5f, 0.0f};
    scene.objects[1].transform.position.z += 1.0f;
    scene.objects[2].transform.scale = {-1.0f, 1.0f, 1.0f};
    scene.objects[3].transform.rotation = {0.0f, 0.0f, 0.8f};

    const gfx::Aabb box = gfx::scene_bounds(scene, model);
    const Vec3 center = (box.min + box.max) * 0.5f;
    const float radius = 0.5f * gfx::length(box.max - box.min);

    gfx::PickIndex index;
    Lcg rng;

    int hits = 0;
    int misses = 0;

    for (int i = 0; i < 300; ++i) {
        // From a random direction on a sphere around the scene, towards a
        // random point near its middle
        const float z = rng.range(-1.0f, 1.0f);
        const float angle = rng.range(0.0f, 2.0f * gfx::kPi);
        const float ring = std::sqrt(1.0f - z * z);
        const Vec3 from_dir = {ring * std::cos(angle), ring * std::sin(angle),
                               z};

        const Vec3 target = center + Vec3{rng.range(-0.6f, 0.6f) * radius,
                                          rng.range(-0.6f, 0.6f) * radius,
                                          rng.range(-0.6f, 0.6f) * radius};

        gfx::PickRay ray;
        ray.origin = center + from_dir * (3.0f * radius);
        ray.dir = gfx::normalize(target - ray.origin);

        int expected_object = -1;
        float expected_t = 0.0f;
        const bool expected = brute_force_pick(scene, model, ray,
                                               expected_object, expected_t);

        gfx::PickHit hit;
        const bool found = gfx::pick_object(scene, model, index, ray, hit);

        CHECK(found == expected);

        if (found && expected) {
            CHECK(hit.object == expected_object);
            CHECK(approx(hit.t, expected_t, 1.0e-3 * (1.0 + expected_t)));
            ++hits;
        } else {
            ++misses;
        }
    }

    // The test must have exercised both outcomes
    CHECK(hits > 60);
    CHECK(misses > 20);

    // The same with one object hidden
    scene.objects[0].hidden = true;

    for (int i = 0; i < 100; ++i) {
        const float angle = rng.range(0.0f, 2.0f * gfx::kPi);

        gfx::PickRay ray;
        ray.origin = center + Vec3{std::cos(angle), std::sin(angle), 0.5f}
                                  * (3.0f * radius);
        ray.dir = gfx::normalize(
            center + Vec3{rng.range(-0.4f, 0.4f) * radius,
                          rng.range(-0.4f, 0.4f) * radius, 0.0f}
            - ray.origin);

        int expected_object = -1;
        float expected_t = 0.0f;
        const bool expected = brute_force_pick(scene, model, ray,
                                               expected_object, expected_t);

        gfx::PickHit hit;
        const bool found = gfx::pick_object(scene, model, index, ray, hit);

        CHECK(found == expected);

        if (found && expected) {
            CHECK(hit.object == expected_object);
            CHECK(hit.object != 0);
        }
    }
}

// The material test bench: straight down onto the floor
void test_real_scene() {
    const std::string dir = std::string(GFX_ASSETS_DIR);

    gfx::Model src;
    std::string error;

    CHECK(gfx::load_model(dir + "/gltf/materials.glb", src, error));

    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(scene, model, std::move(src), "materials.glb");

    const gfx::Aabb box = gfx::scene_bounds(scene, model);

    gfx::PickIndex index;
    gfx::PickHit hit;

    // A corner of the floor, seen from high above
    const float x = box.max.x - 0.2f;
    const float y = box.max.y - 0.2f;

    CHECK(gfx::pick_object(scene, model, index,
                           down_ray(x, y, box.max.z + 50.0f), hit));
    CHECK(hit.object >= 0);
    CHECK(scene.objects[static_cast<std::size_t>(hit.object)].name == "floor");
    CHECK(hit.point.z < 0.5f);

    // Past the end of the scene: nothing
    CHECK(!gfx::pick_object(scene, model, index,
                            down_ray(box.max.x + 5.0f, y, box.max.z + 50.0f),
                            hit));
}

} // namespace


int main() {
    test_ray_through_the_centre();
    test_ray_corners();
    test_ray_inverts_the_projection();
    test_ray_rejects_bad_views();
    test_hit_miss_and_sides();
    test_object_transforms();
    test_nearest_object_wins();
    test_unpickable_objects_are_skipped();
    test_pick_through_a_pixel();
    test_pick_index();
    test_against_brute_force();
    test_real_scene();

    if (g_failures == 0) {
        std::printf("All pick tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d pick test(s) failed\n", g_failures);
    return 1;
}
