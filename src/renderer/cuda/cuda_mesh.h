#pragma once

#include "core/bvh.h"
#include "core/bvh4.h"
#include "renderer/renderer.h"

#include <cstdint>
#include <vector>

namespace gfx {

struct DeviceMesh {
    Vec3* positions = nullptr;
    Vec3* normals = nullptr;
    std::uint32_t* indices = nullptr;
    std::uint32_t* edges = nullptr;

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

    MeshId add(const Mesh& mesh);
    const DeviceMesh* find(MeshId id) const;
    void shutdown();

private:
    std::vector<DeviceMesh> m_meshes;
};

} // namespace gfx
