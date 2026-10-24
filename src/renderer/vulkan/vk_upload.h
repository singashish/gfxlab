#pragma once

#include "renderer/vulkan/vk_api.h"

#include <cstdint>
#include <functional>

namespace gfx {

class VulkanContext;
class VulkanBuffer;

class VulkanUploader {
public:
    VulkanUploader() = default;
    ~VulkanUploader();

    VulkanUploader(const VulkanUploader&) = delete;
    VulkanUploader& operator=(const VulkanUploader&) = delete;

    bool init(const VulkanContext& ctx);
    void shutdown();

    bool upload_to_device_buffer(const VulkanContext& ctx,
                        VulkanBuffer& dst, VkBufferUsageFlags usage,
                        const void* data, VkDeviceSize size);

    // Copies tightly packed pixels into a whole 2D image (one mip level,
    // one layer). The image must have been created with
    // VK_IMAGE_USAGE_TRANSFER_DST_BIT and be in VK_IMAGE_LAYOUT_UNDEFINED;
    // afterwards it is in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    // ready for fragment shaders to sample
    bool upload_to_image(const VulkanContext& ctx,
                         VkImage image,
                         std::uint32_t width,
                         std::uint32_t height,
                         const void* data,
                         VkDeviceSize size);

    // Uploads RGBA8 pixels (tightly packed, width * height * 4 bytes, top
    // row first) into mip level 0 of a 2D image and fills the other
    // mip levels by repeatedly blitting the level above at half size
    // with a linear filter. The image must have mip_levels levels, one
    // layer, format R8G8B8A8_UNORM, optimal tiling, usage SAMPLED |
    // TRANSFER_SRC | TRANSFER_DST, and be in VK_IMAGE_LAYOUT_UNDEFINED.
    // Afterwards every level is in VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
    // With mip_levels == 1 no blit happens, so the format needs no blit
    // support in that case
    bool upload_texture(const VulkanContext& ctx,
                        VkImage image,
                        std::uint32_t width,
                        std::uint32_t height,
                        std::uint32_t mip_levels,
                        const void* data,
                        VkDeviceSize size);

private:
    // Records commands with record(), submits them and waits until the
    // GPU has finished. what names the operation in error messages
    bool submit_and_wait(const VulkanContext& ctx,
                         const std::function<void(VkCommandBuffer)>& record,
                         const char* what);

    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
};

} // namespace gfx
