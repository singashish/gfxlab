#include "renderer/cuda/cuda_mesh.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <limits>
#include <utility>

#include "core/light_list.h"
#include "core/log.h"
#include "renderer/cuda/cuda_api.h"
#include "renderer/cuda/cuda_texture.h"

namespace gfx {

namespace {

static_assert(sizeof(Vec3) == 12,
              "CUDA mesh upload requires tightly packed Vec3");
static_assert(sizeof(Vec2) == 8,
              "CUDA mesh upload requires tightly packed Vec2");

// The texture table index for a Model image number: -1 (no texture), a
// number outside the list, or an image that failed to upload (TextureId
// 0) all give 0, the 1x1 white texture. Every material texture slot is
// resolved this way, so the kernels never need a "has texture" branch
int resolve_texture(int image,
                    const std::vector<TextureId>& textures,
                    const CudaTextureStore& store) {
    if (image < 0 ||
        static_cast<std::size_t>(image) >= textures.size()) {
        return 0;
    }

    return store.table_index(textures[static_cast<std::size_t>(image)]);
}

// The device entry of a Material. `textures` and `store` resolve the
// material's images as described at resolve_texture
CudaMaterial make_cuda_material(const Material& material,
                                const std::vector<TextureId>& textures,
                                const CudaTextureStore& store) {
    // The values the editor can change are cleaned by sanitized(), the
    // same function update_material uses
    const MaterialParams params = sanitized(params_of(material));

    CudaMaterial entry{};
    entry.base_color = params.base_color;
    entry.texture =
        resolve_texture(material.base_color_image, textures, store);

    entry.emission = params.emission;
    entry.emissive_texture =
        resolve_texture(material.emissive_image, textures, store);

    entry.roughness = params.roughness;
    entry.metallic = params.metallic;
    entry.metallic_roughness_texture =
        resolve_texture(material.metallic_roughness_image, textures, store);

    return entry;
}

bool is_emissive(const CudaMaterial& material) {
    return material.emission.x > 0.0f || material.emission.y > 0.0f ||
           material.emission.z > 0.0f;
}

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

    if (mesh.texcoords) {
        (void)cuda_ok(
            cudaFree(mesh.texcoords),
            "cudaFree mesh texcoords");
    }

    if (mesh.triangle_materials) {
        (void)cuda_ok(
            cudaFree(mesh.triangle_materials),
            "cudaFree mesh triangle materials");
    }

    if (mesh.bvh_nodes) {
        (void)cuda_ok(
            cudaFree(mesh.bvh_nodes),
            "cudaFree mesh BVH nodes");
    }

    if (mesh.bvh4_nodes) {
        (void)cuda_ok(
            cudaFree(mesh.bvh4_nodes),
            "cudaFree mesh 4-wide BVH nodes");
    }

    if (mesh.bvh_tri_indices) {
        (void)cuda_ok(
            cudaFree(mesh.bvh_tri_indices),
            "cudaFree mesh BVH triangle indices");
    }

    if (mesh.light_tris) {
        (void)cuda_ok(
            cudaFree(mesh.light_tris),
            "cudaFree mesh light triangles");
    }

    if (mesh.light_cdf) {
        (void)cuda_ok(
            cudaFree(mesh.light_cdf),
            "cudaFree mesh light cdf");
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

MeshId CudaMeshStore::add(const Mesh& mesh,
                          const std::vector<TextureId>& textures,
                          const CudaTextureStore& store) {
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

    // Texture coordinates are optional. Without a matching count the mesh
    // gets none and every vertex uses uv (0, 0), like the OpenGL path
    const bool has_uv = mesh.texcoords.size() == mesh.positions.size();

    if (!mesh.texcoords.empty() && !has_uv) {
        LOG_WARN("CUDA mesh: texcoords (%zu) and positions (%zu) differ in "
                 "count; ignoring the texcoords",
                 mesh.texcoords.size(), mesh.positions.size());
    }

    // This mesh's part of the material table: its materials in order,
    // then one extra "fallback" entry (the default Material with no
    // texture) at index material_count. A triangle whose material number
    // is out of range uses the fallback, as in the OpenGL path
    const std::size_t triangle_count = mesh.indices.size() / 3;
    const std::size_t material_count = mesh.materials.size();

    std::vector<CudaMaterial> new_materials;
    new_materials.reserve(material_count + 1);

    const std::vector<TextureId> no_textures;
    std::size_t emissive_count = 0;

    for (const Material& material : mesh.materials) {
        const CudaMaterial entry =
            make_cuda_material(material, textures, store);

        emissive_count += is_emissive(entry) ? 1 : 0;
        new_materials.push_back(entry);
    }

    // The fallback has no texture and no emission, and it is not
    // counted as a material of the mesh
    new_materials.push_back(
        make_cuda_material(Material{}, no_textures, store));

    if (m_materials.size() + new_materials.size() >
        static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        LOG_ERROR("CUDA mesh: material count exceeds supported range");
        return 0;
    }

    // One material number per triangle, made safe on the CPU so the GPU
    // can index without checks. A mesh with no list (or one of the wrong
    // length) uses material 0, the rule of group_triangles_by_material
    std::vector<std::uint32_t> triangle_materials(triangle_count, 0u);

    if (mesh.triangle_materials.size() == triangle_count) {
        for (std::size_t t = 0; t < triangle_count; ++t) {
            const std::uint32_t number = mesh.triangle_materials[t];

            triangle_materials[t] =
                static_cast<std::size_t>(number) < material_count
                    ? number
                    : static_cast<std::uint32_t>(material_count);
        }
    } else if (!mesh.triangle_materials.empty()) {
        LOG_WARN("CUDA mesh: %zu triangle materials for %zu triangles; "
                 "using material 0 for all",
                 mesh.triangle_materials.size(), triangle_count);
    }

    // The lamp triangles for light sampling: the luminance of each
    // triangle's emission factor (the fallback material has none), then
    // the list and cumulative weights (core/light_list.h)
    std::vector<float> triangle_luminance(triangle_count, 0.0f);

    for (std::size_t t = 0; t < triangle_count; ++t) {
        triangle_luminance[t] = emission_luminance(
            new_materials[triangle_materials[t]].emission);
    }

    std::vector<float> areas = triangle_areas(mesh.positions, mesh.indices);
    const LightList lights =
        build_light_list_from_areas(areas, triangle_luminance);

    // Build the ray-tracing BVH once, on the CPU, then upload it as two
    // flat arrays. The GPU walks the same structure with bvh_trace()
    const auto build_start = std::chrono::steady_clock::now();
    const Bvh bvh = build_bvh(mesh);
    const auto build_end = std::chrono::steady_clock::now();

    const BvhStats stats = bvh_stats(bvh);

    // The 4-wide tree is made from the binary one and keeps its
    // triangle order, so the two share one triangle index array
    const Bvh4 bvh4 = build_bvh4(bvh);

    LOG_INFO(
        "CUDA mesh BVH: %u nodes, %u leaves, max depth %u, "
        "largest leaf %u, %zu 4-wide nodes, built in %.1f ms",
        stats.node_count,
        stats.leaf_count,
        stats.max_depth,
        stats.max_leaf_size,
        bvh4.nodes.size(),
        std::chrono::duration<double, std::milli>(
            build_end - build_start).count());

    // An empty list uploads as a null pointer
    const std::vector<Vec2> no_texcoords;

    DeviceMesh device{};
    device.vertex_count = vertex_count;
    device.bvh_node_count = stats.node_count;
    device.material_base = static_cast<std::uint32_t>(m_materials.size());

    if (!bvh.nodes.empty()) {
        device.bounds_min = bvh.nodes[0].box_min;
        device.bounds_max = bvh.nodes[0].box_max;
    }
    device.index_count =
        static_cast<std::uint32_t>(mesh.indices.size());
    device.edge_index_count =
        static_cast<std::uint32_t>(mesh.edges.size());
    device.light_count = static_cast<std::uint32_t>(lights.cdf.size());
    device.light_weight =
        lights.cdf.empty() ? 0.0f : lights.cdf.back();

    if (!upload_array(device.positions, mesh.positions,
                      "CUDA mesh positions upload") ||
        !upload_array(device.normals, mesh.normals,
                      "CUDA mesh normals upload") ||
        !upload_array(device.indices, mesh.indices,
                      "CUDA mesh indices upload") ||
        !upload_array(device.edges, mesh.edges,
                      "CUDA mesh edges upload") ||
        !upload_array(device.texcoords,
                      has_uv ? mesh.texcoords : no_texcoords,
                      "CUDA mesh texcoords upload") ||
        !upload_array(device.triangle_materials, triangle_materials,
                      "CUDA mesh triangle materials upload") ||
        !upload_array(device.bvh_nodes, bvh.nodes,
                      "CUDA mesh BVH nodes upload") ||
        !upload_array(device.bvh_tri_indices, bvh.tri_indices,
                      "CUDA mesh BVH triangle indices upload") ||
        !upload_array(device.bvh4_nodes, bvh4.nodes,
                      "CUDA mesh 4-wide BVH nodes upload") ||
        !upload_array(device.light_tris, lights.triangles,
                      "CUDA mesh light triangles upload") ||
        !upload_array(device.light_cdf, lights.cdf,
                      "CUDA mesh light cdf upload")) {
        free_device_mesh(device);
        return 0;
    }

    // Append this mesh's materials and upload the whole table again. The
    // new table is made first, so a failure leaves the old one in place
    std::vector<CudaMaterial> combined = m_materials;
    combined.insert(combined.end(), new_materials.begin(),
                    new_materials.end());

    CudaMaterial* table = nullptr;

    if (!upload_array(table, combined, "CUDA material table upload")) {
        if (table != nullptr) {
            (void)cuda_ok(cudaFree(table), "cudaFree material table");
        }

        free_device_mesh(device);
        return 0;
    }

    if (m_device_materials != nullptr) {
        (void)cuda_ok(cudaFree(m_device_materials),
                      "cudaFree old material table");
    }

    m_device_materials = table;
    m_materials = std::move(combined);

    m_meshes.push_back(device);

    MeshHost host;
    host.triangle_materials = std::move(triangle_materials);
    host.triangle_areas = std::move(areas);
    host.material_count = static_cast<std::uint32_t>(material_count);
    m_host.push_back(std::move(host));

    const MeshId id = static_cast<MeshId>(m_meshes.size());

    LOG_INFO(
        "CUDA mesh uploaded: id=%u, vertices=%u, triangles=%u, "
        "materials=%zu (%zu emissive)%s",
        id,
        device.vertex_count,
        device.index_count / 3,
        material_count,
        emissive_count,
        has_uv ? ", with texcoords" : ""
    );

    return id;
}

const DeviceMesh* CudaMeshStore::find(MeshId id) const {
    if (id == 0 || id > m_meshes.size()) {
        return nullptr;
    }

    return &m_meshes[id - 1];
}

bool CudaMeshStore::update_material(MeshId id, int index,
                                    const MaterialParams& params) {
    if (id == 0 || id > m_meshes.size() || m_device_materials == nullptr) {
        return false;
    }

    DeviceMesh& mesh = m_meshes[id - 1];
    const MeshHost& host = m_host[id - 1];

    if (index < 0 || static_cast<std::uint32_t>(index) >= host.material_count) {
        return false;
    }

    // The new entry: the four values change, the texture slots stay
    const MaterialParams clean = sanitized(params);
    const std::size_t global = mesh.material_base +
                               static_cast<std::size_t>(index);
    CudaMaterial entry = m_materials[global];

    entry.base_color = clean.base_color;
    entry.roughness = clean.roughness;
    entry.metallic = clean.metallic;
    entry.emission = clean.emission;

    // The lamp weights depend on the emission's luminance only, so the
    // light list is rebuilt only when that changed. The new arrays are
    // uploaded first and the old ones freed after the entry is in, so a
    // failure leaves everything as it was
    std::uint32_t* new_tris = nullptr;
    float* new_cdf = nullptr;
    std::uint32_t new_count = 0;
    float new_weight = 0.0f;
    bool lights_changed = false;

    if (emission_luminance(entry.emission) !=
        emission_luminance(m_materials[global].emission)) {
        std::vector<float> luminance(host.triangle_materials.size(), 0.0f);

        for (std::size_t t = 0; t < luminance.size(); ++t) {
            const std::size_t number =
                mesh.material_base + host.triangle_materials[t];

            luminance[t] = emission_luminance(
                number == global ? entry.emission
                                 : m_materials[number].emission);
        }

        const LightList lights =
            build_light_list_from_areas(host.triangle_areas, luminance);

        if (!upload_array(new_tris, lights.triangles,
                          "CUDA mesh light triangles upload") ||
            !upload_array(new_cdf, lights.cdf,
                          "CUDA mesh light cdf upload")) {
            if (new_tris != nullptr) {
                (void)cuda_ok(cudaFree(new_tris),
                              "cudaFree new light triangles");
            }

            if (new_cdf != nullptr) {
                (void)cuda_ok(cudaFree(new_cdf), "cudaFree new light cdf");
            }

            return false;
        }

        new_count = static_cast<std::uint32_t>(lights.cdf.size());
        new_weight = lights.cdf.empty() ? 0.0f : lights.cdf.back();
        lights_changed = true;
    }

    // One 48-byte entry, in place: the table's address does not change
    if (!cuda_ok(cudaMemcpy(m_device_materials + global, &entry,
                            sizeof(CudaMaterial), cudaMemcpyHostToDevice),
                 "CUDA material update")) {
        if (new_tris != nullptr) {
            (void)cuda_ok(cudaFree(new_tris), "cudaFree new light triangles");
        }

        if (new_cdf != nullptr) {
            (void)cuda_ok(cudaFree(new_cdf), "cudaFree new light cdf");
        }

        return false;
    }

    m_materials[global] = entry;

    if (lights_changed) {
        if (mesh.light_tris != nullptr) {
            (void)cuda_ok(cudaFree(mesh.light_tris),
                          "cudaFree old light triangles");
        }

        if (mesh.light_cdf != nullptr) {
            (void)cuda_ok(cudaFree(mesh.light_cdf),
                          "cudaFree old light cdf");
        }

        mesh.light_tris = new_tris;
        mesh.light_cdf = new_cdf;
        mesh.light_count = new_count;
        mesh.light_weight = new_weight;
    }

    ++m_revision;
    return true;
}

void CudaMeshStore::shutdown() {
    for (DeviceMesh& mesh : m_meshes) {
        free_device_mesh(mesh);
    }

    m_meshes.clear();
    m_host.clear();
    m_revision = 0;

    if (m_device_materials != nullptr) {
        (void)cuda_ok(cudaFree(m_device_materials),
                      "cudaFree material table");
        m_device_materials = nullptr;
    }

    m_materials.clear();
}

} // namespace gfx
