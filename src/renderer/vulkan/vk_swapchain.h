#pragma once

#include "renderer/vulkan/vk_api.h"

#include <cstdint>
#include <vector>

namespace gfx {

class VulkanContext;

class VulkanSwapchain {
public:
    VulkanSwapchain() = default;
    ~VulkanSwapchain();

    VulkanSwapchain(const VulkanSwapchain&) = delete;
    VulkanSwapchain& operator=(const VulkanSwapchain&) = delete;

    bool create(
        const VulkanContext& ctx,
        std::uint32_t desired_width,
        std::uint32_t desired_height,
        bool vsync);

    void destroy();

    VkSwapchainKHR swapchain() const { return m_swapchain; }
    VkFormat format() const { return m_format; }
    VkExtent2D extent() const { return m_extent; }

    std::uint32_t image_count() const {
        return static_cast<std::uint32_t>(m_images.size());
    }

    VkImage image(std::uint32_t i) const {
        return i < m_images.size()
            ? m_images[i]
            : VK_NULL_HANDLE;
    }

    VkImageView image_view(std::uint32_t i) const {
        return i < m_image_views.size()
            ? m_image_views[i]
            : VK_NULL_HANDLE;
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;

    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkColorSpaceKHR m_color_space =
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D m_extent{0, 0};

    // Images are owned by the swapchain. We only store their handles
    std::vector<VkImage> m_images;

    // Image views are owned by this class and must be destroyed
    std::vector<VkImageView> m_image_views;
};

} // namespace gfx
