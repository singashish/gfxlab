#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

class VulkanContext;

class VulkanBuffer {
public:
    VulkanBuffer() = default;
    ~VulkanBuffer();

    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    bool create(const VulkanContext& ctx, VkDeviceSize size,
                VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags);

    void destroy();

    bool write(const void* data, VkDeviceSize size);

    VkBuffer buffer() const { return m_buffer; }
    VkDeviceSize size() const { return m_size; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VkDeviceMemory m_memory = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    VkMemoryPropertyFlags m_memory_flags = 0;
};

} // namespace gfx
