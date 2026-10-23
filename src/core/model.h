#pragma once

/*
 * Model: what any model file loads into.
 *
 * A Mesh is one block of triangles with its own materials. A model can
 * need several of them, placed in space: a glTF scene is a tree of
 * nodes, and two nodes may use the same mesh (a table with four
 * identical legs stores the leg once). The Model keeps exactly that:
 * the meshes, and a flat list of instances saying "draw mesh M with
 * this transform". The tree itself is gone after loading; every
 * instance carries the product of all transforms above it.
 *
 * Everything in a Model is Z-up (the world convention of gfxlab, see
 * vec3.h), whatever the file used.
 */

#include "core/image.h"
#include "core/mat4.h"
#include "core/mesh.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gfx {

// One placement of a mesh: the node hierarchy flattened to model space
struct ModelInstance {
    std::uint32_t mesh = 0;           // index into Model::meshes
    Mat4 transform = mat4_identity(); // mesh space -> model space (Z-up)
};

// What any model file loads into
struct Model {
    std::vector<Mesh> meshes;            // shared by instances (glTF instancing)
    std::vector<Image> images;           // decoded textures, see Material::*_image
    std::vector<ModelInstance> instances;
};

// Loads .obj, .gltf or .glb (chosen by extension, case-insensitive) and
// prepares the result for rendering: Z-up, every mesh with normals
// (ensure_normals) and an edge list (ensure_edges), textures decoded,
// and centred so the bounding box of all instances is centred on the
// origin. Missing/unsupported textures only log warnings (LOG_WARN).
//
// Centring moves the instances (their transforms), never the vertices:
// with instancing, the vertices belong to all instances at once.
// On failure returns false, sets error and leaves out empty
bool load_model(const std::string& path, Model& out, std::string& error);

// Bounding box of all instances in model space: the 8 corners of each
// mesh's box, transformed. A rotated instance therefore gets a box that
// is a little larger than the tightest one around its vertices.
// A model without instances gives a zero box
Aabb model_bounds(const Model& model);

// Moves every instance so model_bounds is centred on the origin
void center_model_at_origin(Model& model);

// Triangles drawn by the whole model: every instance counts, so a mesh
// used twice counts twice
std::size_t model_triangle_count(const Model& model);

} // namespace gfx
