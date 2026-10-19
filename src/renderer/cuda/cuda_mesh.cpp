#include "renderer/cuda/cuda_mesh.h"

#include <cuda_runtime.h>

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

    DeviceMesh device{};
    device.vertex_count = vertex_count;
    device.index_count =
        static_cast<std::uint32_t>(mesh.indices.size());

    if (!upload_array(device.positions, mesh.positions,
                      "CUDA mesh positions upload") ||
        !upload_array(device.normals, mesh.normals,
                      "CUDA mesh normals upload") ||
        !upload_array(device.indices, mesh.indices,
                      "CUDA mesh indices upload")) {
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
