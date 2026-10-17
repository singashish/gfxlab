#include "renderer/vulkan/vk_frames.h"

namespace gfx {

VulkanFrames::~VulkanFrames() {
    destroy();
}

void VulkanFrames::destroy() {
    // The caller must ensure the device is idle
    if (m_device == VK_NULL_HANDLE) {
        m_command_pool = VK_NULL_HANDLE;
        m_slots = {};
        m_render_finished.clear();
        m_current_slot = 0;
        return;
    }

    for (Slot& slot : m_slots) {
        if (slot.in_flight != VK_NULL_HANDLE) {
            vkDestroyFence(m_device, slot.in_flight, nullptr);
        }

        if (slot.image_available != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, slot.image_available, nullptr);
        }

        slot = {};
    }

    for (VkSemaphore semaphore : m_render_finished) {
        if (semaphore != VK_NULL_HANDLE) {
            vkDestroySemaphore(m_device, semaphore, nullptr);
        }
    }

    m_render_finished.clear();

    // Destroying the pool also frees all command buffers allocated from it
    if (m_command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(m_device, m_command_pool, nullptr);
        m_command_pool = VK_NULL_HANDLE;
    }

    m_device = VK_NULL_HANDLE;
    m_current_slot = 0;
}

bool VulkanFrames::create(VkDevice device, std::uint32_t queue_family,
                          std::uint32_t swapchain_image_count) {

    destroy();

    if (device == VK_NULL_HANDLE) {
        LOG_ERROR("cannot create Vulkan frames with a null device");
        return false;
    }

    if (swapchain_image_count == 0) {
        LOG_ERROR("cannot create Vulkan frames with zero swapchain images");
        return false;
    }

    m_device = device;

    // 1. Create the command pool
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags =
        VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = queue_family;

    if (!vk_ok(
            vkCreateCommandPool(
                m_device, &pool_info, nullptr, &m_command_pool),
            "create command pool")) {
        destroy();
        return false;
    }

    // 2. Allocate one primary command buffer per frame slot
    VkCommandBufferAllocateInfo allocation_info{};
    allocation_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation_info.commandPool = m_command_pool;
    allocation_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation_info.commandBufferCount = kFramesInFlight;

    VkCommandBuffer commands[kFramesInFlight]{};

    if (!vk_ok(
            vkAllocateCommandBuffers(
                m_device, &allocation_info, commands),
            "allocate command buffers")) {
        destroy();
        return false;
    }

    for (std::uint32_t i = 0; i < kFramesInFlight; ++i) {
        m_slots[i].cmd = commands[i];
    }

    // 3. Create frame-slot synchronization objects
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType =
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (Slot& slot : m_slots) {
        if (!vk_ok(
                vkCreateFence(
                    m_device, &fence_info, nullptr, &slot.in_flight),
                "create frame fence")) {
            destroy();
            return false;
        }

        if (!vk_ok(
                vkCreateSemaphore(
                    m_device, &semaphore_info, nullptr,
                    &slot.image_available),
                "create image-available semaphore")) {
            destroy();
            return false;
        }
    }

    // 4. Create one render-finished semaphore per swapchain image
    m_render_finished.resize(swapchain_image_count, VK_NULL_HANDLE);

    for (VkSemaphore& semaphore : m_render_finished) {
        if (!vk_ok(
                vkCreateSemaphore(
                    m_device, &semaphore_info, nullptr, &semaphore),
                "create render-finished semaphore")) {
            destroy();
            return false;
        }
    }

    m_current_slot = 0;

    LOG_INFO(
        "Vulkan frame resources created: %u frame slots, "
        "%u swapchain images",
        kFramesInFlight, swapchain_image_count);

    return true;
}

} // namespace gfx
