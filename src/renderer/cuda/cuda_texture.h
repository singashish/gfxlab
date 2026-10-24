#pragma once

#include "core/image.h"
#include "core/texture_sampling.h"
#include "renderer/renderer.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

// The textures of the CUDA backend. This backend does not use CUDA's
// texture objects: the kernels read the texels themselves with the
// software sampler in core/texture_sampling.h, so what is stored here is
// plain device memory.
//
// Every texture is one device buffer holding its whole mip chain (RGBA8,
// level after level) and one TextureView describing it. The views of all
// textures sit in one small device array, the "texture table", which the
// kernels index.
//
// Table layout: entry 0 is a 1x1 white texture made by init(), entry i
// (i >= 1) is the texture with TextureId i. So a TextureId is also its
// table index, and "no texture" is simply entry 0: white leaves the base
// colour unchanged, and the kernels never need a branch for a missing
// texture. (White is also the neutral value of the other texture roles:
// it leaves an emission factor and the roughness and metallic factors
// unchanged.)
//
// The store keeps the texels exactly as the image has them (RGBA8) and
// knows nothing about what a texture means. Whether its values are
// colours (base colour, emission: stored as sRGB) or plain data
// (metallic-roughness: used as stored) is decided where it is sampled,
// by the material that uses it
class CudaTextureStore {
public:
    CudaTextureStore() = default;
    ~CudaTextureStore();

    CudaTextureStore(const CudaTextureStore&) = delete;
    CudaTextureStore& operator=(const CudaTextureStore&) = delete;

    // Creates the white texture. Needs the CUDA device to be selected
    // already (the presenter does that). Returns false on a CUDA error
    bool init();

    // Builds the mip chain of the image and uploads it. Returns the
    // TextureId (1, 2, ...) or 0 if the image is invalid or the upload
    // fails; the table is unchanged then
    TextureId create(const Image& image);

    // Table index for a TextureId: the id itself, or 0 (white) for 0 and
    // for ids this store does not know
    int table_index(TextureId id) const;

    // The texture table in device memory; null before init(). The pointer
    // changes whenever a texture is added, so read it again each frame
    const TextureView* device_table() const {
        return m_device_table;
    }

    // Textures made with create() so far (not counting the white one)
    std::size_t count() const {
        return m_views.empty() ? 0 : m_views.size() - 1;
    }

    // Device memory used by the texels of all textures, in bytes
    std::size_t texel_bytes() const {
        return m_texel_bytes;
    }

    void shutdown();

private:
    // Uploads one packed mip chain and appends its view to m_views;
    // view.texels already points at the device copy. Returns false on a
    // CUDA error (nothing is added then)
    bool add_chain(const std::vector<std::uint8_t>& packed,
                   TextureView view);

    // Replaces the device texture table with a copy of m_views
    bool upload_table();

    // Host copy of the table (the views point into device memory), and
    // the device buffers behind them, in the same order
    std::vector<TextureView> m_views;
    std::vector<std::uint8_t*> m_buffers;

    TextureView* m_device_table = nullptr;
    std::size_t m_texel_bytes = 0;
};

} // namespace gfx
