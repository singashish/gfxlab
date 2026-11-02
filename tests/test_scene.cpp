#include "core/scene.h"
#include "core/model.h"
#include "check.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>

namespace {

bool approx_vec(gfx::Vec3 a, gfx::Vec3 b, double eps = 1e-4) {
    return approx(a.x, b.x, eps)
        && approx(a.y, b.y, eps)
        && approx(a.z, b.z, eps);
}

bool approx_mat(const gfx::Mat4& a, const gfx::Mat4& b, double eps = 1e-4) {
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

// A mesh that is only 8 corner positions: enough for bounds and layout
gfx::Mesh make_box_mesh(gfx::Vec3 lo, gfx::Vec3 hi) {
    gfx::Mesh mesh;

    for (int corner = 0; corner < 8; ++corner) {
        mesh.positions.push_back({
            (corner & 1) ? hi.x : lo.x,
            (corner & 2) ? hi.y : lo.y,
            (corner & 4) ? hi.z : lo.z
        });
    }

    return mesh;
}

// One mesh (the box lo..hi) and one instance per given matrix
gfx::Model make_model(gfx::Vec3 lo, gfx::Vec3 hi,
                      const std::vector<gfx::Mat4>& placements,
                      const std::vector<std::string>& names = {}) {
    gfx::Model model;
    model.meshes.push_back(make_box_mesh(lo, hi));

    for (std::size_t i = 0; i < placements.size(); ++i) {
        gfx::ModelInstance instance;
        instance.mesh = 0;
        instance.transform = placements[i];

        if (i < names.size()) {
            instance.name = names[i];
        }

        model.instances.push_back(instance);
    }

    return model;
}

// World box of the objects of one collection
gfx::Aabb collection_bounds(const gfx::Scene& scene, const gfx::Model& model,
                            std::uint32_t collection) {
    gfx::Scene part;

    for (const gfx::SceneObject& object : scene.objects) {
        if (object.collection == collection) {
            part.objects.push_back(object);
        }
    }

    return gfx::scene_bounds(part, model);
}

void test_decompose_roundtrip() {
    gfx::Transform t;
    t.position = {1.0f, -2.0f, 3.5f};
    t.rotation = {0.3f, 0.5f, -1.0f};
    t.scale = {2.0f, 0.5f, 3.0f};

    gfx::Transform back;

    CHECK(gfx::decompose_trs(t.model_matrix(), back));
    CHECK(approx_vec(back.position, t.position));
    CHECK(approx_vec(back.rotation, t.rotation));
    CHECK(approx_vec(back.scale, t.scale));
    CHECK(approx_mat(back.model_matrix(), t.model_matrix()));

    // The identity decomposes to the default transform
    CHECK(gfx::decompose_trs(gfx::mat4_identity(), back));
    CHECK(approx_vec(back.position, {0.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(back.rotation, {0.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(back.scale, {1.0f, 1.0f, 1.0f}));
}

void test_decompose_mirror() {
    gfx::Transform t;
    t.position = {0.0f, 1.0f, 0.0f};
    t.rotation = {0.2f, -0.4f, 0.9f};
    t.scale = {-2.0f, 1.0f, 1.5f};

    gfx::Transform back;

    // The mirror is written as a negative x scale; the matrix is the same
    CHECK(gfx::decompose_trs(t.model_matrix(), back));
    CHECK(back.scale.x < 0.0f);
    CHECK(approx_mat(back.model_matrix(), t.model_matrix()));
}

void test_decompose_rejects() {
    gfx::Transform back;

    // Shear: scale after a rotation (columns not at right angles)
    const gfx::Mat4 sheared =
        gfx::mat4_scale({3.0f, 1.0f, 1.0f}) *
        gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, 0.6f);
    CHECK(!gfx::decompose_trs(sheared, back));

    // A perspective-like bottom row
    gfx::Mat4 projective = gfx::mat4_identity();
    projective.col[0].w = 0.5f;
    CHECK(!gfx::decompose_trs(projective, back));

    // A flattened axis
    CHECK(!gfx::decompose_trs(gfx::mat4_scale({1.0f, 0.0f, 1.0f}), back));
}

void test_append_model() {
    gfx::Model a;
    a.meshes.push_back(make_box_mesh({0, 0, 0}, {1, 1, 1}));
    a.meshes.push_back(make_box_mesh({0, 0, 0}, {2, 2, 2}));
    a.images.resize(2);

    gfx::Material material_a;
    material_a.base_color_image = 0;
    material_a.normal_image = 1;
    a.meshes[0].materials.push_back(material_a);

    a.instances.push_back({0, gfx::mat4_identity(), ""});
    a.instances.push_back({1, gfx::mat4_identity(), ""});

    gfx::Model b;
    b.meshes.push_back(make_box_mesh({0, 0, 0}, {3, 3, 3}));
    b.images.resize(1);

    gfx::Material material_b;
    material_b.base_color_image = 0;
    // metallic_roughness_image and emissive_image stay -1
    b.meshes[0].materials.push_back(material_b);

    b.instances.push_back({0, gfx::mat4_identity(), "b0"});

    const std::size_t first = gfx::append_model(a, std::move(b));

    CHECK(first == 2);
    CHECK(a.meshes.size() == 3);
    CHECK(a.images.size() == 3);
    CHECK(a.instances.size() == 3);

    // The appended instance now points at the appended mesh
    CHECK(a.instances[2].mesh == 2);
    CHECK(a.instances[2].name == "b0");

    // Its material points at the appended image; "none" stays none
    CHECK(a.meshes[2].materials[0].base_color_image == 2);
    CHECK(a.meshes[2].materials[0].metallic_roughness_image == -1);
    CHECK(a.meshes[2].materials[0].emissive_image == -1);

    // The first model is untouched
    CHECK(a.meshes[0].materials[0].base_color_image == 0);
    CHECK(a.meshes[0].materials[0].normal_image == 1);
    CHECK(a.instances[1].mesh == 1);

    // Appending nothing changes nothing
    gfx::Model empty;
    CHECK(gfx::append_model(a, std::move(empty)) == 3);
    CHECK(a.meshes.size() == 3);
}

void test_single_file_layout() {
    // 2 x 4 x 6, centred on the origin, as load_model returns it
    gfx::Model src = make_model({-1, -2, -3}, {1, 2, 3},
                                {gfx::mat4_identity()});
    gfx::Model model;
    gfx::Scene scene;

    const std::size_t first =
        gfx::scene_add_model(scene, model, std::move(src), "cube.obj");

    CHECK(first == 0);
    CHECK(scene.objects.size() == 1);
    CHECK(scene.collections.size() == 1);
    CHECK(scene.collections[0].name == "cube.obj");

    const gfx::SceneObject& object = scene.objects[0];

    CHECK(object.name == "cube");
    CHECK(object.collection == 0);

    // Stands on the floor: half the height up, centred in x and y. This is
    // what the single model transform used to do
    CHECK(approx_vec(object.transform.position, {0.0f, 0.0f, 3.0f}));
    CHECK(approx_mat(object.world_matrix(),
                     gfx::mat4_translate({0.0f, 0.0f, 3.0f})));
    CHECK(approx_mat(object.local, gfx::mat4_identity()));

    // Nothing is selected or active after loading: the user clicks
    CHECK(scene.active == -1);
    CHECK(!object.selected);
    CHECK(scene.active_object() == nullptr);
    CHECK(gfx::scene_selected_count(scene) == 0);

    // initial is the loaded placement
    CHECK(approx_vec(object.initial.position, object.transform.position));
}

void test_off_centre_model_is_moved_to_the_origin() {
    gfx::Model src = make_model({10, 10, 10}, {12, 14, 16},
                                {gfx::mat4_identity()});
    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(scene, model, std::move(src), "far.obj");

    const gfx::Aabb box = gfx::scene_bounds(scene, model);

    CHECK(approx(box.min.z, 0.0));
    CHECK(approx(box.max.z, 6.0));
    CHECK(approx(0.5 * (box.min.x + box.max.x), 0.0));
    CHECK(approx(0.5 * (box.min.y + box.max.y), 0.0));
}

void test_two_files_side_by_side() {
    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(
        scene, model,
        make_model({-1, -1, -1}, {1, 1, 1}, {gfx::mat4_identity()}),
        "first.obj");
    gfx::scene_add_model(
        scene, model,
        make_model({-2, -1, -1}, {2, 1, 1}, {gfx::mat4_identity()}),
        "second.obj");

    CHECK(scene.collections.size() == 2);
    CHECK(scene.objects.size() == 2);
    CHECK(scene.objects[0].collection == 0);
    CHECK(scene.objects[1].collection == 1);
    CHECK(scene.objects[1].mesh == 1);
    CHECK(model.meshes.size() == 2);

    const gfx::Aabb first = collection_bounds(scene, model, 0);
    const gfx::Aabb second = collection_bounds(scene, model, 1);

    // The first stays centred on the origin
    CHECK(approx(0.5 * (first.min.x + first.max.x), 0.0));

    // The second starts after the first plus a quarter of its own
    // longest side (4 -> gap 1)
    CHECK(approx(second.min.x, first.max.x + 1.0));

    // Both stand on the floor, centred in y
    CHECK(approx(first.min.z, 0.0));
    CHECK(approx(second.min.z, 0.0));
    CHECK(approx(0.5 * (second.min.y + second.max.y), 0.0));

    // Loading a file selects nothing, and keeps the active object
    CHECK(scene.active == -1);
    CHECK(gfx::scene_selected_count(scene) == 0);

    // The camera framing box covers both
    const gfx::Aabb all = gfx::scene_bounds(scene, model);
    CHECK(approx(all.min.x, first.min.x));
    CHECK(approx(all.max.x, second.max.x));
}

void test_object_names() {
    gfx::Model model;
    gfx::Scene scene;

    // No names: the file's name, then .001, .002
    gfx::scene_add_model(
        scene, model,
        make_model({0, 0, 0}, {1, 1, 1},
                   {gfx::mat4_identity(), gfx::mat4_identity(),
                    gfx::mat4_identity()}),
        "dir/table.glb");

    CHECK(scene.objects[0].name == "table");
    CHECK(scene.objects[1].name == "table.001");
    CHECK(scene.objects[2].name == "table.002");

    // Node names win; repeats within the collection get a suffix
    gfx::scene_add_model(
        scene, model,
        make_model({0, 0, 0}, {1, 1, 1},
                   {gfx::mat4_identity(), gfx::mat4_identity(),
                    gfx::mat4_identity()},
                   {"Leg", "Leg", "Top"}),
        "chair.glb");

    CHECK(scene.objects[3].name == "Leg");
    CHECK(scene.objects[4].name == "Leg.001");
    CHECK(scene.objects[5].name == "Top");
}

void test_rotated_instance_keeps_its_own_origin() {
    gfx::Transform placement;
    placement.position = {1.0f, 2.0f, 3.0f};
    placement.rotation = {0.0f, 0.0f, 0.5f};
    placement.scale = {2.0f, 2.0f, 2.0f};

    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(
        scene, model,
        make_model({-1, -1, -1}, {1, 1, 1}, {placement.model_matrix()}),
        "r.obj");

    const gfx::SceneObject& object = scene.objects[0];

    // The rotation and scale are in the object's own transform, so G/R/S
    // work around the object, not around the collection's origin
    CHECK(approx(object.transform.rotation.z, 0.5));
    CHECK(approx_vec(object.transform.scale, {2.0f, 2.0f, 2.0f}));
    CHECK(approx_mat(object.local, gfx::mat4_identity()));

    // Standing on the floor: the lowest corner of the scaled box is z = 0
    CHECK(approx(gfx::scene_bounds(scene, model).min.z, 0.0));
}

void test_sheared_instance_is_kept_whole() {
    const gfx::Mat4 sheared =
        gfx::mat4_scale({3.0f, 1.0f, 1.0f}) *
        gfx::mat4_rotate({0.0f, 0.0f, 1.0f}, 0.6f);

    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(
        scene, model, make_model({-1, -1, -1}, {1, 1, 1}, {sheared}),
        "s.obj");

    const gfx::SceneObject& object = scene.objects[0];

    // The matrix is in `local`; the transform is only the placement, and
    // the world matrix is still placement * original instance matrix
    CHECK(approx_mat(object.local, sheared));
    CHECK(approx_vec(object.transform.rotation, {0.0f, 0.0f, 0.0f}));
    CHECK(approx_vec(object.transform.scale, {1.0f, 1.0f, 1.0f}));
    CHECK(approx_mat(object.world_matrix(),
                     gfx::mat4_translate(object.transform.position) *
                         sheared));
    CHECK(approx(gfx::scene_bounds(scene, model).min.z, 0.0));
}

void test_select_only() {
    gfx::Model model;
    gfx::Scene scene;

    CHECK(scene.active_object() == nullptr);

    gfx::scene_add_model(
        scene, model,
        make_model({0, 0, 0}, {1, 1, 1},
                   {gfx::mat4_identity(), gfx::mat4_identity(),
                    gfx::mat4_identity()}),
        "x.obj");

    gfx::scene_select_only(scene, 2);

    CHECK(scene.active == 2);
    CHECK(!scene.objects[0].selected);
    CHECK(!scene.objects[1].selected);
    CHECK(scene.objects[2].selected);
    CHECK(scene.active_object() == &scene.objects[2]);

    gfx::scene_select_only(scene, -1);

    CHECK(scene.active == -1);
    CHECK(!scene.objects[2].selected);
    CHECK(scene.active_object() == nullptr);

    gfx::scene_select_only(scene, 99);

    CHECK(scene.active == -1);
}

// A scene with `count` objects (one cube each), nothing selected
gfx::Scene make_selection_scene(std::size_t count) {
    gfx::Model model;
    gfx::Scene scene;
    std::vector<gfx::Mat4> placements(count, gfx::mat4_identity());

    gfx::scene_add_model(scene, model,
                         make_model({0, 0, 0}, {1, 1, 1}, placements),
                         "s.obj");
    return scene;
}

// The selected objects as a string like "0,2", and the active one
std::string selection_text(const gfx::Scene& scene) {
    std::string text;

    for (std::size_t i = 0; i < scene.objects.size(); ++i) {
        if (scene.objects[i].selected) {
            if (!text.empty()) {
                text += ",";
            }

            text += std::to_string(i);
        }
    }

    return text + " active " + std::to_string(scene.active);
}

void test_click_select() {
    gfx::Scene scene = make_selection_scene(4);

    CHECK(selection_text(scene) == " active -1");

    // Plain clicks select one object and make it active
    gfx::scene_click_select(scene, 1, false);
    CHECK(selection_text(scene) == "1 active 1");

    gfx::scene_click_select(scene, 2, false);
    CHECK(selection_text(scene) == "2 active 2");

    // Shift click adds, and the added object becomes active
    gfx::scene_click_select(scene, 0, true);
    CHECK(selection_text(scene) == "0,2 active 0");

    gfx::scene_click_select(scene, 1, true);
    CHECK(selection_text(scene) == "0,1,2 active 1");

    // Shift click on a selected object that is not active only makes it
    // active
    gfx::scene_click_select(scene, 2, true);
    CHECK(selection_text(scene) == "0,1,2 active 2");

    // Shift click on the active selected object deselects it; it stays the
    // active object (unselected)
    gfx::scene_click_select(scene, 2, true);
    CHECK(selection_text(scene) == "0,1 active 2");
    CHECK(scene.active_object() != nullptr);
    CHECK(!scene.active_object()->selected);

    // Shift click on empty space changes nothing
    gfx::scene_click_select(scene, -1, true);
    CHECK(selection_text(scene) == "0,1 active 2");
    gfx::scene_click_select(scene, 99, true);
    CHECK(selection_text(scene) == "0,1 active 2");

    // A plain click on an object that is one of several selected objects
    // selects only that one
    gfx::scene_click_select(scene, 1, false);
    CHECK(selection_text(scene) == "1 active 1");

    // A plain click on empty space deselects everything and keeps the
    // active object
    gfx::scene_click_select(scene, -1, false);
    CHECK(selection_text(scene) == " active 1");
    gfx::scene_click_select(scene, 99, false);
    CHECK(selection_text(scene) == " active 1");
    gfx::scene_click_select(scene, -7, false);
    CHECK(selection_text(scene) == " active 1");
}

// Loading another file later (the Load models button) leaves the selection
// and the active object as they are; the new objects start unselected
void test_loading_keeps_the_selection() {
    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(
        scene, model,
        make_model({0, 0, 0}, {1, 1, 1},
                   {gfx::mat4_identity(), gfx::mat4_identity()}),
        "a.obj");
    gfx::scene_click_select(scene, 1, false);
    gfx::scene_click_select(scene, 0, true);

    gfx::scene_add_model(
        scene, model,
        make_model({0, 0, 0}, {1, 1, 1}, {gfx::mat4_identity()}), "b.obj");

    CHECK(selection_text(scene) == "0,1 active 0");
    CHECK(scene.objects.size() == 3);
    CHECK(!scene.objects[2].selected);
}

void test_select_all_and_deselect_all() {
    gfx::Scene scene = make_selection_scene(4);

    // Select all keeps "no active object" ...
    gfx::scene_select_all(scene);
    CHECK(selection_text(scene) == "0,1,2,3 active -1");
    CHECK(gfx::scene_selected_count(scene) == 4);

    // ... and Alt+A keeps the active object
    gfx::scene_click_select(scene, 2, false);
    gfx::scene_select_all(scene);
    CHECK(selection_text(scene) == "0,1,2,3 active 2");

    gfx::scene_deselect_all(scene);
    CHECK(selection_text(scene) == " active 2");
    CHECK(gfx::scene_selected_count(scene) == 0);

    // Hidden objects are not selected by "select all"
    scene.objects[1].hidden = true;
    gfx::scene_select_all(scene);
    CHECK(selection_text(scene) == "0,2,3 active 2");
    CHECK(gfx::scene_selected_count(scene) == 3);

    // Nothing to do on an empty scene
    gfx::Scene empty;
    gfx::scene_select_all(empty);
    gfx::scene_deselect_all(empty);
    gfx::scene_click_select(empty, 0, false);
    gfx::scene_click_select(empty, 0, true);
    CHECK(empty.active == -1);
}

// The real files: one file must place its instances exactly as the old
// single model transform did, and a second file must keep its textures
void test_real_files() {
    const std::string dir = std::string(GFX_ASSETS_DIR);
    const char* files[] = {"gltf/materials.glb", "gltf/cubes.glb"};

    for (const char* file : files) {
        gfx::Model loaded;
        std::string error;

        CHECK(gfx::load_model(dir + "/" + file, loaded, error));

        if (loaded.instances.empty()) {
            continue;
        }

        // What main.cpp did before: lift by half the height, then draw each
        // instance with its own matrix
        const gfx::Aabb box = gfx::model_bounds(loaded);
        const gfx::Mat4 lift = gfx::mat4_translate(
            {0.0f, 0.0f, (box.max.z - box.min.z) * 0.5f});

        const std::vector<gfx::ModelInstance> expected = loaded.instances;

        gfx::Model model;
        gfx::Scene scene;

        gfx::scene_add_model(scene, model, std::move(loaded), file);

        CHECK(scene.objects.size() == expected.size());

        for (std::size_t i = 0; i < expected.size() &&
                                i < scene.objects.size(); ++i) {
            CHECK(approx_mat(scene.objects[i].world_matrix(),
                             lift * expected[i].transform, 2e-4));
        }
    }

    // materials.glb, then the textured cube: the cube's image indices must
    // be shifted past the images already in the model
    gfx::Model first;
    gfx::Model second;
    std::string error;

    CHECK(gfx::load_model(dir + "/gltf/materials.glb", first, error));
    CHECK(gfx::load_model(dir + "/textured_cube.obj", second, error));

    const std::size_t first_images = first.images.size();
    const std::size_t first_objects = first.instances.size();
    const std::size_t second_images = second.images.size();

    gfx::Model model;
    gfx::Scene scene;

    gfx::scene_add_model(scene, model, std::move(first), "materials.glb");
    gfx::scene_add_model(scene, model, std::move(second),
                         "textured_cube.obj");

    CHECK(scene.collections.size() == 2);
    CHECK(scene.objects.size() == first_objects + 1);
    CHECK(model.images.size() == first_images + second_images);

    const gfx::SceneObject& cube = scene.objects.back();

    CHECK(cube.collection == 1);
    CHECK(cube.mesh < model.meshes.size());

    bool found_image = false;

    for (const gfx::Material& material : model.meshes[cube.mesh].materials) {
        if (material.base_color_image >= 0) {
            found_image = true;

            CHECK(static_cast<std::size_t>(material.base_color_image) >=
                  first_images);
            CHECK(static_cast<std::size_t>(material.base_color_image) <
                  model.images.size());
        }
    }

    CHECK(found_image);
}

} // namespace


int main() {
    test_decompose_roundtrip();
    test_decompose_mirror();
    test_decompose_rejects();
    test_append_model();
    test_single_file_layout();
    test_off_centre_model_is_moved_to_the_origin();
    test_two_files_side_by_side();
    test_object_names();
    test_rotated_instance_keeps_its_own_origin();
    test_sheared_instance_is_kept_whole();
    test_select_only();
    test_click_select();
    test_loading_keeps_the_selection();
    test_select_all_and_deselect_all();
    test_real_files();

    if (g_failures == 0) {
        std::printf("All scene tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d scene test(s) failed\n", g_failures);
    return 1;
}
