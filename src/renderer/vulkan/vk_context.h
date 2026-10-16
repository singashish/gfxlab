#pragma once

#include "renderer/vulkan/vk_api.h"

#include <cstdint>

namespace gfx {

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();

    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    bool create(void* native_window);
    void destroy();

    VkInstance instance() const { return m_instance; }
    VkSurfaceKHR surface() const { return m_surface; }
    VkPhysicalDevice physical_device() const {
        return m_physical_device;
    }
    VkDevice device() const { return m_device; }
    VkQueue queue() const { return m_queue; }
    std::uint32_t queue_family() const {
        return m_queue_family;
    }

private:
    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debug_messenger =
        VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkPhysicalDevice m_physical_device = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;

    std::uint32_t m_queue_family = 0;
    bool m_validation_enabled = false;

    PFN_vkDestroyDebugUtilsMessengerEXT m_destroy_debug_messenger =
        nullptr;
};

} // namespace gfx
