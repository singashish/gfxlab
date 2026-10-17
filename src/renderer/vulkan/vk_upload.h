#pragma once

#include "renderer/vulkan/vk_api.h"

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

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;
    VkFence m_fence = VK_NULL_HANDLE;
};

} // namespace gfx
