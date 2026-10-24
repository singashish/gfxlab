#pragma once

#include "core/bvh.h"
#include "core/bvh4.h"
#include "core/vec2.h"
#include "core/vec3.h"
#include "renderer/renderer.h"

#include <cstdint>
#include <vector>

namespace gfx {

class CudaTextureStore;

// One entry of the renderer-wide material table in device memory. Every
// mesh appends its own materials to it (see CudaMeshStore::add), so a
// triangle's material is found with one global index
struct CudaMaterial {
    Vec3 base_color;  // multiplied with the texture, as in the GL shader

    // Index into the texture table (CudaTextureStore). Always valid: a
    // material without a texture, or with an unknown one, holds 0, the
    // 1x1 white texture, so the kernels never branch on "no texture"
    int texture;
};

static_assert(sizeof(CudaMaterial) == 16,
              "CudaMaterial must be tightly packed");

struct DeviceMesh {
    Vec3* positions = nullptr;
    Vec3* normals = nullptr;
    std::uint32_t* indices = nullptr;
    std::uint32_t* edges = nullptr;

    // Texture coordinates, one per vertex; null when the mesh has none
    // (the shaders then use uv (0, 0), like the OpenGL path)
    Vec2* texcoords = nullptr;

    // One entry per triangle: which material it uses, counted from this
    // mesh's first material. Always present and always in range. The
    // renderer-wide table index is material_base + triangle_materials[t]
    std::uint32_t* triangle_materials = nullptr;
    std::uint32_t material_base = 0;

    std::uint32_t vertex_count = 0;
    std::uint32_t index_count = 0;
    std::uint32_t edge_index_count = 0;

    // BVH built on the CPU when the mesh is added, used by the
    // ray tracer. Node 0 is the root
    BvhNode* bvh_nodes = nullptr;
    std::uint32_t* bvh_tri_indices = nullptr;
    std::uint32_t bvh_node_count = 0;

    // The same tree collapsed to 4-wide nodes; shares bvh_tri_indices
    Bvh4Node* bvh4_nodes = nullptr;

    // Object-space bounding box (the BVH root), kept on the host side of
    // this struct so the renderer can size effects to the model
    Vec3 bounds_min = {0.0f, 0.0f, 0.0f};
    Vec3 bounds_max = {0.0f, 0.0f, 0.0f};
};

class CudaMeshStore {
public:
    CudaMeshStore() = default;
    ~CudaMeshStore();

    CudaMeshStore(const CudaMeshStore&) = delete;
    CudaMeshStore& operator=(const CudaMeshStore&) = delete;

    // textures[i] is the TextureId of Model::images[i] (0 = none) and
    // store resolves those ids to texture table entries. The mesh's
    // materials are appended to the global material table, which is
    // uploaded again (it is tiny)
    MeshId add(const Mesh& mesh,
               const std::vector<TextureId>& textures,
               const CudaTextureStore& store);
    const DeviceMesh* find(MeshId id) const;

    // The global material table in device memory; null while no mesh has
    // been added. The pointer changes when a mesh is added, so read it
    // again each frame
    const CudaMaterial* device_materials() const {
        return m_device_materials;
    }

    void shutdown();

private:
    std::vector<DeviceMesh> m_meshes;

    // Host copy of the material table and the device copy
    std::vector<CudaMaterial> m_materials;
    CudaMaterial* m_device_materials = nullptr;
};

} // namespace gfx
