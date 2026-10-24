#include "renderer/cuda/cuda_texture.h"

#include <cuda_runtime.h>

#include <limits>

#include "core/log.h"
#include "renderer/cuda/cuda_api.h"

namespace gfx {

CudaTextureStore::~CudaTextureStore() {
    shutdown();
}

bool CudaTextureStore::init() {
    shutdown();

    // Entry 0: one white texel, a mip chain with a single level
    Image white;
    white.width = 1;
    white.height = 1;
    white.pixels = {255, 255, 255, 255};

    TextureView view;
    const std::vector<std::uint8_t> packed =
        pack_mip_chain(build_mip_chain(white), view);

    if (packed.empty() || !add_chain(packed, view)) {
        LOG_ERROR("CUDA textures: cannot create the white texture");
        shutdown();
        return false;
    }

    return true;
}

TextureId CudaTextureStore::create(const Image& image) {
    if (m_views.empty()) {
        LOG_ERROR("CUDA textures: the store is not initialized");
        return 0;
    }

    if (!image.valid()) {
        LOG_ERROR("CUDA textures: cannot upload an invalid image");
        return 0;
    }

    if (m_views.size() >=
        static_cast<std::size_t>(std::numeric_limits<TextureId>::max())) {
        LOG_ERROR("CUDA textures: too many textures");
        return 0;
    }

    TextureView view;
    const std::vector<std::uint8_t> packed =
        pack_mip_chain(build_mip_chain(image), view);

    if (packed.empty()) {
        LOG_ERROR("CUDA textures: cannot build the mip chain of a %dx%d image",
                  image.width, image.height);
        return 0;
    }

    if (!add_chain(packed, view)) {
        return 0;
    }

    const TextureId id = static_cast<TextureId>(m_views.size() - 1);

    LOG_INFO("CUDA texture uploaded: id=%u, %dx%d, %d mip levels, %.2f MiB",
             id, view.width, view.height, view.levels,
             static_cast<double>(packed.size()) / (1024.0 * 1024.0));

    return id;
}

int CudaTextureStore::table_index(TextureId id) const {
    if (id == 0 || static_cast<std::size_t>(id) >= m_views.size()) {
        return 0;
    }

    return static_cast<int>(id);
}

bool CudaTextureStore::add_chain(const std::vector<std::uint8_t>& packed,
                                 TextureView view) {
    std::uint8_t* device = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&device),
                            packed.size()),
                 "cudaMalloc texture texels")) {
        return false;
    }

    if (!cuda_ok(cudaMemcpy(device, packed.data(), packed.size(),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy texture texels")) {
        (void)cuda_ok(cudaFree(device), "cudaFree texture texels");
        return false;
    }

    // From here on the view describes the device copy
    view.texels = device;

    m_views.push_back(view);
    m_buffers.push_back(device);

    if (!upload_table()) {
        m_views.pop_back();
        m_buffers.pop_back();
        (void)cuda_ok(cudaFree(device), "cudaFree texture texels");
        return false;
    }

    m_texel_bytes += packed.size();
    return true;
}

bool CudaTextureStore::upload_table() {
    // The table is a few dozen bytes per texture, so it is simply made
    // again each time: allocate the new one first, so the old one is
    // still there if that fails
    TextureView* table = nullptr;
    const std::size_t bytes = m_views.size() * sizeof(TextureView);

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&table), bytes),
                 "cudaMalloc texture table")) {
        return false;
    }

    if (!cuda_ok(cudaMemcpy(table, m_views.data(), bytes,
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy texture table")) {
        (void)cuda_ok(cudaFree(table), "cudaFree texture table");
        return false;
    }

    if (m_device_table != nullptr) {
        (void)cuda_ok(cudaFree(m_device_table), "cudaFree old texture table");
    }

    m_device_table = table;
    return true;
}

void CudaTextureStore::shutdown() {
    for (std::uint8_t* buffer : m_buffers) {
        (void)cuda_ok(cudaFree(buffer), "cudaFree texture texels");
    }

    m_buffers.clear();
    m_views.clear();
    m_texel_bytes = 0;

    if (m_device_table != nullptr) {
        (void)cuda_ok(cudaFree(m_device_table), "cudaFree texture table");
        m_device_table = nullptr;
    }
}

} // namespace gfx
