#pragma once

/*
 * glTF 2.0 loader (.gltf with external or embedded data, and the
 * single-file binary form .glb), written from scratch on top of the
 * JSON parser. glTF is the "JPEG of 3D": the format most tools export
 * to and most model sites offer.
 *
 * A glTF file describes a scene as a tree of nodes. A node can place a
 * mesh and carries a transform (a matrix, or translation + rotation +
 * scale) relative to its parent. A mesh is a list of primitives, each a
 * set of triangles with one material. The vertex data sits in binary
 * buffers; "accessors" describe how to read numbers out of them
 * ("3 floats per element, starting at byte 24, every 32 bytes").
 *
 * What is loaded: triangle primitives (also strips and fans, converted)
 * with POSITION, NORMAL and TEXCOORD_0; the node hierarchy, flattened
 * to ModelInstance transforms; PBR metallic-roughness materials with
 * their PNG textures. What is ignored, with a warning where it matters:
 * cameras, lights, skins, animations, morph targets, sparse accessors
 * (that primitive is skipped), texture samplers and JPEG textures.
 * Files that require an extension we cannot honour (Draco compression,
 * meshopt, Basis textures, ...) are refused.
 *
 * Conventions of the result (see model.h):
 *  - Z-up. glTF is Y-up; parse_gltf converts at the end, rotating the
 *    mesh vertices and normals and re-expressing every instance
 *    transform in the new axes, so the model looks the same.
 *  - Texture coordinates follow the OBJ convention of Mesh::texcoords
 *    ((0, 0) = bottom-left of the image). glTF puts (0, 0) at the
 *    top-left, so v is stored as 1 - v.
 *  - Materials are per Mesh, as in the OBJ loader: every Mesh gets a
 *    copy of the file's whole material list (plus a "default" material
 *    at the end when some primitive names no material), so a
 *    triangle_materials entry indexes straight into any mesh's list.
 *    Image indices in the materials refer to Model::images.
 *  - Not centred and normals exactly as in the file (a mesh where some
 *    primitive lacks NORMAL gets none at all, and load_model generates
 *    flat ones). The edge list is left empty for load_model to build.
 */

#include "core/model.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace gfx {

// From memory: data is either a .glb (starts with "glTF") or JSON text.
// base_dir resolves relative URIs (buffers, images); empty = only
// embedded data allowed.
//
// On failure returns false with a specific message and leaves out empty
bool parse_gltf(const std::uint8_t* data, std::size_t size,
                const std::string& base_dir, Model& out, std::string& error);

// Reads a .gltf or .glb file and parses it; relative URIs are resolved
// against the file's folder
bool load_gltf(const std::string& path, Model& out, std::string& error);

} // namespace gfx
