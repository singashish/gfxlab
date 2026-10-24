#pragma once

#include "core/image.h"
#include "renderer/renderer.h"
#include "renderer/vulkan/vk_api.h"

#include <cstdint>
#include <vector>

namespace gfx {

class VulkanContext;
class VulkanUploader;

// The Vulkan counterpart of GLTextureStore: the GPU images of a model's
// textures, plus a 1x1 white texture for surfaces without one.
//
// In Vulkan a shader cannot "bind a texture"; it reads a descriptor set
// that points at an image view and a sampler. This store makes one
// descriptor set per texture (all written once, at creation) and the
// mesh renderer binds the right set as set number 1 before each draw.
// All textures share one sampler and one set layout
//
// Owned by the Vulkan renderer. Call shutdown (or let the destructor
// run) only when the GPU is idle
class VulkanTextureStore {
public:
    VulkanTextureStore() = default;
    ~VulkanTextureStore();

    VulkanTextureStore(const VulkanTextureStore&) = delete;
    VulkanTextureStore& operator=(const VulkanTextureStore&) = delete;

    // Creates the sampler, the set layout and the white texture
    bool init(const VulkanContext& ctx, VulkanUploader& uploader);

    // Uploads an RGBA8 image (top row first) with a full mip chain.
    // Returns 0 on failure; otherwise a 1-based TextureId
    TextureId create(const VulkanContext& ctx,
                     VulkanUploader& uploader,
                     const Image& image);

    // Descriptor set (set 1 of the mesh pipeline) of an id from create.
    // 0 and unknown ids give the white texture's set
    VkDescriptorSet descriptor_set(TextureId id) const;

    VkDescriptorSet white_set() const {
        return m_white.set;
    }

    // Layout of those sets: binding 0 = combined image sampler,
    // fragment stage. The mesh pipeline layout needs it
    VkDescriptorSetLayout set_layout() const {
        return m_set_layout;
    }

    void shutdown();

private:
    struct GpuTexture {
        VkImage image = VK_NULL_HANDLE;
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        VkDescriptorSet set = VK_NULL_HANDLE;
    };

    // Image + memory + view + filled descriptor set for tightly packed
    // RGBA8 pixels. On failure out is left empty and nothing leaks
    bool create_gpu_texture(const VulkanContext& ctx,
                            VulkanUploader& uploader,
                            const std::uint8_t* pixels,
                            std::uint32_t width,
                            std::uint32_t height,
                            GpuTexture& out);

    void destroy_gpu_texture(GpuTexture& texture);

    // Hands out one set from the newest pool, adding a pool when the
    // newest is full
    bool allocate_set(VkDescriptorSet& out);
    bool add_pool();

    VkDevice m_device = VK_NULL_HANDLE;
    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;

    // Descriptor sets come from pools of a fixed size. A pool cannot
    // grow, so a full one is kept and a new one is added; sets are
    // never freed individually, only by destroying their pool
    std::vector<VkDescriptorPool> m_pools;
    std::uint32_t m_sets_in_newest_pool = 0;

    // Texture limits and capabilities, read once in init
    bool m_mipmaps_supported = false;
    std::uint32_t m_max_dimension = 0;

    GpuTexture m_white;

    // m_textures[id - 1] is the texture of TextureId id
    std::vector<GpuTexture> m_textures;
};

} // namespace gfx
