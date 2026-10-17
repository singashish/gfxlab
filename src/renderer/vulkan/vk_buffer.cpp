#include "renderer/vulkan/vk_buffer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_memory.h"

#include <cstring>

namespace gfx {

VulkanBuffer::~VulkanBuffer() {
    destroy();
}

void VulkanBuffer::destroy() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_buffer != VK_NULL_HANDLE) {
            vkDestroyBuffer(m_device, m_buffer, nullptr);
        }

        if (m_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    m_buffer = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_size = 0;
    m_memory_flags = 0;
}

bool VulkanBuffer::create(const VulkanContext& ctx, VkDeviceSize size,
                VkBufferUsageFlags usage, VkMemoryPropertyFlags memory_flags) {
    destroy();

    if (ctx.device() == VK_NULL_HANDLE ||
        ctx.physical_device() == VK_NULL_HANDLE ||
        size == 0) {
        LOG_ERROR("invalid arguments for Vulkan buffer creation");
        return false;
    }

    m_device = ctx.device();

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (!vk_ok(
            vkCreateBuffer(
                m_device, &buffer_info, nullptr, &m_buffer),
            "create Vulkan buffer")) {
        destroy();
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(
        m_device, m_buffer, &requirements);

    std::uint32_t memory_type_index = 0;

    if (!find_memory_type(
            ctx.physical_device(),
            requirements.memoryTypeBits,
            memory_flags,
            memory_type_index)) {
        destroy();
        return false;
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;

    if (!vk_ok(
            vkAllocateMemory(
                m_device,
                &allocation_info,
                nullptr,
                &m_memory),
            "allocate Vulkan buffer memory")) {
        destroy();
        return false;
    }

    if (!vk_ok(
            vkBindBufferMemory(
                m_device, m_buffer, m_memory, 0),
            "bind Vulkan buffer memory")) {
        destroy();
        return false;
    }

    m_size = size;
    m_memory_flags = memory_flags;

    return true;
}

bool VulkanBuffer::write(const void* data, VkDeviceSize size) {
    if (m_device == VK_NULL_HANDLE ||
        m_buffer == VK_NULL_HANDLE ||
        m_memory == VK_NULL_HANDLE ||
        (m_memory_flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) == 0 ||
        data == nullptr ||
        size == 0 ||
        size > m_size) {
        LOG_ERROR("invalid arguments for Vulkan buffer write");
        return false;
    }

    void* mapped = nullptr;

    if (!vk_ok(
            vkMapMemory(
                m_device,
                m_memory,
                0,
                VK_WHOLE_SIZE,
                0,
                &mapped),
            "map Vulkan buffer memory")) {
        return false;
    }

    std::memcpy(mapped, data, static_cast<std::size_t>(size));

    vkUnmapMemory(m_device, m_memory);

    return true;
}

} // namespace gfx
