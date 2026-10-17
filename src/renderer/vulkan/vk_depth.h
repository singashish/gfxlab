#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

class VulkanContext;

class VulkanDepthBuffer {
public:
    VulkanDepthBuffer() = default;
    ~VulkanDepthBuffer();

    VulkanDepthBuffer(const VulkanDepthBuffer&) = delete;
    VulkanDepthBuffer& operator=(const VulkanDepthBuffer&) = delete;

    bool create(const VulkanContext& ctx, VkExtent2D extent);
    void destroy();

    VkFormat format() const { return m_format; }
    VkImage image() const { return m_image; }
    VkImageView view() const { return m_view; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkImage m_image = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkImageView m_view = VK_NULL_HANDLE;
};

} // namespace gfx
