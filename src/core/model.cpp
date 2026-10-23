#include "core/model.h"

#include "core/gltf_loader.h"
#include "core/log.h"
#include "core/obj_loader.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <map>
#include <utility>

namespace gfx {

namespace {

std::string lower_extension(const std::string& path) {
    std::string extension = std::filesystem::path(path).extension().string();

    std::transform(extension.begin(), extension.end(), extension.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });

    return extension;
}

// OBJ: one mesh, textures are files named by the materials. Decodes
// every distinct file once, however many materials use it
bool load_obj_model(const std::string& path, Model& out,
                    std::string& error) {
    Mesh mesh;

    if (!load_obj(path, mesh, error)) {
        return false;
    }

    // OBJ files are Y-up by convention (Blender's exporter can write
    // either, but Y-up is what most files use)
    mesh_y_up_to_z_up(mesh);

    // Texture file name -> index into out.images, or -1 when it could
    // not be decoded (remembered so the warning appears only once)
    std::map<std::string, int> image_of_path;

    for (Material& material : mesh.materials) {
        const std::string& texture = material.base_color_texture;

        if (texture.empty()) {
            continue;
        }

        const auto known = image_of_path.find(texture);

        if (known != image_of_path.end()) {
            material.base_color_image = known->second;
            continue;
        }

        Image image;
        std::string image_error;
        int index = -1;

        if (load_image(texture, image, image_error)) {
            out.images.push_back(std::move(image));
            index = static_cast<int>(out.images.size()) - 1;
        } else {
            LOG_WARN("texture not loaded: %s", image_error.c_str());
        }

        image_of_path[texture] = index;
        material.base_color_image = index;
    }

    out.meshes.push_back(std::move(mesh));

    ModelInstance instance;
    instance.mesh = 0;
    instance.transform = mat4_identity();
    out.instances.push_back(instance);

    return true;
}

} // namespace

bool load_model(const std::string& path, Model& out, std::string& error) {
    out = Model{};
    error.clear();

    const std::string extension = lower_extension(path);

    bool ok = false;

    if (extension == ".obj") {
        ok = load_obj_model(path, out, error);
    } else if (extension == ".gltf" || extension == ".glb") {
        // The glTF loader already converts to Z-up and decodes textures
        ok = load_gltf(path, out, error);
    } else {
        error = "unsupported model format '" + extension +
                "' (expected .obj, .gltf or .glb): " + path;
    }

    if (!ok) {
        out = Model{};
        return false;
    }

    // Normals first: when a mesh has none they are generated per
    // triangle, which unwelds the vertices; the edge list is built
    // afterwards from the final vertex numbering
    std::size_t generated = 0;

    for (Mesh& mesh : out.meshes) {
        if (ensure_normals(mesh)) {
            ++generated;
        }

        ensure_edges(mesh);
    }

    if (generated == 0) {
        LOG_INFO("normals in file: yes");
    } else if (generated == out.meshes.size()) {
        LOG_INFO("normals generated (flat)");
    } else {
        LOG_INFO("normals generated (flat) for %zu of %zu meshes",
                 generated, out.meshes.size());
    }

    center_model_at_origin(out);

    return true;
}

Aabb model_bounds(const Model& model) {
    // A mesh's own box is the same for all its instances: compute it
    // once per mesh, not once per instance
    std::vector<Aabb> mesh_boxes;
    mesh_boxes.reserve(model.meshes.size());

    for (const Mesh& mesh : model.meshes) {
        mesh_boxes.push_back(compute_bounds(mesh));
    }

    Aabb bounds{};
    bool first = true;

    for (const ModelInstance& instance : model.instances) {
        if (instance.mesh >= model.meshes.size() ||
            model.meshes[instance.mesh].positions.empty()) {
            continue;
        }

        const Aabb& box = mesh_boxes[instance.mesh];

        // Bit 0 of the corner number picks min/max for x, bit 1 for y,
        // bit 2 for z
        for (int corner = 0; corner < 8; ++corner) {
            const Vec3 local = {
                (corner & 1) ? box.max.x : box.min.x,
                (corner & 2) ? box.max.y : box.min.y,
                (corner & 4) ? box.max.z : box.min.z
            };

            const Vec3 p = transform_point(instance.transform, local);

            if (first) {
                bounds.min = p;
                bounds.max = p;
                first = false;
                continue;
            }

            bounds.min.x = std::min(bounds.min.x, p.x);
            bounds.min.y = std::min(bounds.min.y, p.y);
            bounds.min.z = std::min(bounds.min.z, p.z);
            bounds.max.x = std::max(bounds.max.x, p.x);
            bounds.max.y = std::max(bounds.max.y, p.y);
            bounds.max.z = std::max(bounds.max.z, p.z);
        }
    }

    return bounds;
}

void center_model_at_origin(Model& model) {
    const Aabb bounds = model_bounds(model);
    const Vec3 center = (bounds.min + bounds.max) * 0.5f;

    // Translating after the instance transform moves the result in
    // model space, which is what centring means
    const Mat4 shift = mat4_translate(-center);

    for (ModelInstance& instance : model.instances) {
        instance.transform = shift * instance.transform;
    }
}

std::size_t model_triangle_count(const Model& model) {
    std::size_t total = 0;

    for (const ModelInstance& instance : model.instances) {
        if (instance.mesh < model.meshes.size()) {
            total += model.meshes[instance.mesh].triangle_count();
        }
    }

    return total;
}

} // namespace gfx
