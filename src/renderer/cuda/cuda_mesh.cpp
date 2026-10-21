#include "renderer/cuda/cuda_mesh.h"

#include <cuda_runtime.h>

#include <chrono>
#include <limits>

#include "core/log.h"
#include "renderer/cuda/cuda_api.h"

namespace gfx {

namespace {

static_assert(sizeof(Vec3) == 12,
              "CUDA mesh upload requires tightly packed Vec3");

void free_device_mesh(DeviceMesh& mesh) {
    if (mesh.positions) {
        (void)cuda_ok(cudaFree(mesh.positions),
                      "cudaFree mesh positions");
    }

    if (mesh.normals) {
        (void)cuda_ok(cudaFree(mesh.normals),
                      "cudaFree mesh normals");
    }

    if (mesh.indices) {
        (void)cuda_ok(cudaFree(mesh.indices),
                      "cudaFree mesh indices");
    }

    if (mesh.edges) {
        (void)cuda_ok(
            cudaFree(mesh.edges),
            "cudaFree mesh edges");
    }

    if (mesh.bvh_nodes) {
        (void)cuda_ok(
            cudaFree(mesh.bvh_nodes),
            "cudaFree mesh BVH nodes");
    }

    if (mesh.bvh_tri_indices) {
        (void)cuda_ok(
            cudaFree(mesh.bvh_tri_indices),
            "cudaFree mesh BVH triangle indices");
    }

    mesh = {};
}

template<typename T>
bool upload_array(T*& destination,
                  const std::vector<T>& source,
                  const char* description) {
    const std::size_t bytes = source.size() * sizeof(T);

    if (bytes == 0) {
        destination = nullptr;
        return true;
    }

    if (!cuda_ok(
            cudaMalloc(reinterpret_cast<void**>(&destination), bytes),
            description)) {
        destination = nullptr;
        return false;
    }

    if (!cuda_ok(
            cudaMemcpy(destination,
                       source.data(),
                       bytes,
                       cudaMemcpyHostToDevice),
            description)) {
        return false;
    }

    return true;
}

} // namespace

CudaMeshStore::~CudaMeshStore() {
    shutdown();
}

MeshId CudaMeshStore::add(const Mesh& mesh) {
    if (mesh.positions.empty()) {
        LOG_ERROR("CUDA mesh: positions array is empty");
        return 0;
    }

    if (mesh.normals.size() != mesh.positions.size()) {
        LOG_ERROR(
            "CUDA mesh: normals count (%zu) differs from positions (%zu)",
            mesh.normals.size(),
            mesh.positions.size()
        );
        return 0;
    }

    if (mesh.indices.empty() || mesh.indices.size() % 3 != 0) {
        LOG_ERROR(
            "CUDA mesh: index count must be nonzero and divisible by 3"
        );
        return 0;
    }

    if (mesh.positions.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        mesh.indices.size() >
            std::numeric_limits<std::uint32_t>::max() ||
        m_meshes.size() >=
            std::numeric_limits<MeshId>::max()) {
        LOG_ERROR("CUDA mesh: count exceeds supported range");
        return 0;
    }

    const auto vertex_count =
        static_cast<std::uint32_t>(mesh.positions.size());

    for (std::uint32_t index : mesh.indices) {
        if (index >= vertex_count) {
            LOG_ERROR(
                "CUDA mesh: index %u exceeds vertex count %u",
                index, vertex_count
            );
            return 0;
        }
    }

    if (!mesh.edges.empty()) {
        if (mesh.edges.size() % 2 != 0) {
            LOG_ERROR(
                "CUDA mesh: edge index count must be even");
            return 0;
        }

        if (mesh.edges.size() >
            std::numeric_limits<std::uint32_t>::max()) {
            LOG_ERROR(
                "CUDA mesh: edge index count exceeds supported range");
            return 0;
        }

        for (std::uint32_t index : mesh.edges) {
            if (index >= vertex_count) {
                LOG_ERROR(
                    "CUDA mesh: edge index %u exceeds vertex count %u",
                    index, vertex_count);
                return 0;
            }
        }
    }

    // Build the ray-tracing BVH once, on the CPU, then upload it as two
    // flat arrays. The GPU walks the same structure with bvh_trace()
    const auto build_start = std::chrono::steady_clock::now();
    const Bvh bvh = build_bvh(mesh);
    const auto build_end = std::chrono::steady_clock::now();

    const BvhStats stats = bvh_stats(bvh);

    LOG_INFO(
        "CUDA mesh BVH: %u nodes, %u leaves, max depth %u, "
        "largest leaf %u, built in %.1f ms",
        stats.node_count,
        stats.leaf_count,
        stats.max_depth,
        stats.max_leaf_size,
        std::chrono::duration<double, std::milli>(
            build_end - build_start).count());

    DeviceMesh device{};
    device.vertex_count = vertex_count;
    device.bvh_node_count = stats.node_count;
    device.index_count =
        static_cast<std::uint32_t>(mesh.indices.size());
    device.edge_index_count =
        static_cast<std::uint32_t>(mesh.edges.size());

    if (!upload_array(device.positions, mesh.positions,
                      "CUDA mesh positions upload") ||
        !upload_array(device.normals, mesh.normals,
                      "CUDA mesh normals upload") ||
        !upload_array(device.indices, mesh.indices,
                      "CUDA mesh indices upload") ||
        !upload_array(device.edges, mesh.edges,
                      "CUDA mesh edges upload") ||
        !upload_array(device.bvh_nodes, bvh.nodes,
                      "CUDA mesh BVH nodes upload") ||
        !upload_array(device.bvh_tri_indices, bvh.tri_indices,
                      "CUDA mesh BVH triangle indices upload")) {
        free_device_mesh(device);
        return 0;
    }

    m_meshes.push_back(device);

    const MeshId id = static_cast<MeshId>(m_meshes.size());

    LOG_INFO(
        "CUDA mesh uploaded: id=%u, vertices=%u, triangles=%u",
        id,
        device.vertex_count,
        device.index_count / 3
    );

    return id;
}

const DeviceMesh* CudaMeshStore::find(MeshId id) const {
    if (id == 0 || id > m_meshes.size()) {
        return nullptr;
    }

    return &m_meshes[id - 1];
}

void CudaMeshStore::shutdown() {
    for (DeviceMesh& mesh : m_meshes) {
        free_device_mesh(mesh);
    }

    m_meshes.clear();
}

} // namespace gfx
