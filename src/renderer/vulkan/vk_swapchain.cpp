#include "renderer/vulkan/vk_swapchain.h"
#include "renderer/vulkan/vk_context.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace gfx {
namespace {

const char* format_name(VkFormat format) {
    switch (format) {
    case VK_FORMAT_B8G8R8A8_UNORM:
        return "VK_FORMAT_B8G8R8A8_UNORM";
    case VK_FORMAT_R8G8B8A8_UNORM:
        return "VK_FORMAT_R8G8B8A8_UNORM";
    case VK_FORMAT_B8G8R8A8_SRGB:
        return "VK_FORMAT_B8G8R8A8_SRGB";
    case VK_FORMAT_R8G8B8A8_SRGB:
        return "VK_FORMAT_R8G8B8A8_SRGB";
    default:
        return "other";
    }
}

const char* color_space_name(VkColorSpaceKHR color_space) {
    switch (color_space) {
    case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:
        return "VK_COLOR_SPACE_SRGB_NONLINEAR_KHR";
    default:
        return "other";
    }
}

const char* present_mode_name(VkPresentModeKHR mode) {
    switch (mode) {
    case VK_PRESENT_MODE_FIFO_KHR:
        return "FIFO";
    case VK_PRESENT_MODE_IMMEDIATE_KHR:
        return "IMMEDIATE";
    case VK_PRESENT_MODE_MAILBOX_KHR:
        return "MAILBOX";
    default:
        return "other";
    }
}

bool has_present_mode(const std::vector<VkPresentModeKHR>& modes,
                      VkPresentModeKHR requested) {

    return std::find(modes.begin(), modes.end(), requested)
        != modes.end();
}

bool choose_surface_format(const std::vector<VkSurfaceFormatKHR>& formats,
                           VkSurfaceFormatKHR& selected) {

    if (formats.empty()) {
        LOG_ERROR("surface reports no supported formats");
        return false;
    }

    constexpr VkColorSpaceKHR preferred_color_space =
        VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_B8G8R8A8_UNORM &&
            format.colorSpace == preferred_color_space) {
            selected = format;
            return true;
        }
    }

    for (const auto& format : formats) {
        if (format.format == VK_FORMAT_R8G8B8A8_UNORM &&
            format.colorSpace == preferred_color_space) {
            selected = format;
            return true;
        }
    }

    selected = formats[0];

    LOG_WARN(
        "preferred UNORM surface formats unavailable; "
        "using the first supported format");

    return true;
}

bool choose_present_mode(const std::vector<VkPresentModeKHR>& modes,
                         bool vsync, VkPresentModeKHR& selected) {

    if (modes.empty()) {
        LOG_ERROR("surface reports no present modes");
        return false;
    }

    if (vsync) {
        if (!has_present_mode(modes, VK_PRESENT_MODE_FIFO_KHR)) {
            LOG_ERROR("FIFO present mode is unavailable");
            return false;
        }

        selected = VK_PRESENT_MODE_FIFO_KHR;
        return true;
    }

    // Prefer an uncapped mode when vsync is disabled
    if (has_present_mode(modes, VK_PRESENT_MODE_IMMEDIATE_KHR)) {
        selected = VK_PRESENT_MODE_IMMEDIATE_KHR;
        return true;
    }

    if (has_present_mode(modes, VK_PRESENT_MODE_MAILBOX_KHR)) {
        selected = VK_PRESENT_MODE_MAILBOX_KHR;
        return true;
    }

    // FIFO is guaranteed by Vulkan for a valid surface
    if (has_present_mode(modes, VK_PRESENT_MODE_FIFO_KHR)) {
        selected = VK_PRESENT_MODE_FIFO_KHR;
        return true;
    }

    LOG_ERROR("no supported present mode could be selected");
    return false;
}

} // namespace

VulkanSwapchain::~VulkanSwapchain() {
    destroy();
}

void VulkanSwapchain::destroy() {
    // Image views must be destroyed before their swapchain
    if (m_device != VK_NULL_HANDLE) {
        for (VkImageView view : m_image_views) {
            if (view != VK_NULL_HANDLE) {
                vkDestroyImageView(m_device, view, nullptr);
            }
        }
    }

    m_image_views.clear();
    m_images.clear();

    if (m_swapchain != VK_NULL_HANDLE &&
        m_device != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    }

    m_swapchain = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
    m_format = VK_FORMAT_UNDEFINED;
    m_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    m_extent = {0, 0};
}

bool VulkanSwapchain::create(const VulkanContext& ctx,
                             std::uint32_t desired_width,
                             std::uint32_t desired_height,
                             bool vsync) {

    // Safe when called repeatedly to recreate the swapchain
    destroy();

    if (ctx.device() == VK_NULL_HANDLE ||
        ctx.physical_device() == VK_NULL_HANDLE ||
        ctx.surface() == VK_NULL_HANDLE) {
        LOG_ERROR("cannot create swapchain: Vulkan context is incomplete");
        return false;
    }

    // Keep the device handle needed for cleanup
    m_device = ctx.device();

    // Step 1: Query surface capabilities
    VkSurfaceCapabilitiesKHR capabilities{};

    if (!vk_ok(
            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
                ctx.physical_device(),
                ctx.surface(),
                &capabilities),
            "query surface capabilities")) {
        destroy();
        return false;
    }

    std::uint32_t format_count = 0;

    if (!vk_ok(
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                ctx.physical_device(),
                ctx.surface(),
                &format_count,
                nullptr),
            "query surface format count")) {
        destroy();
        return false;
    }

    if (format_count == 0) {
        LOG_ERROR("surface has no supported formats");
        destroy();
        return false;
    }

    std::vector<VkSurfaceFormatKHR> formats(format_count);

    if (!vk_ok(
            vkGetPhysicalDeviceSurfaceFormatsKHR(
                ctx.physical_device(),
                ctx.surface(),
                &format_count,
                formats.data()),
            "query surface formats")) {
        destroy();
        return false;
    }

    formats.resize(format_count);

    std::uint32_t present_mode_count = 0;

    if (!vk_ok(
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                ctx.physical_device(),
                ctx.surface(),
                &present_mode_count,
                nullptr),
            "query present mode count")) {
        destroy();
        return false;
    }

    if (present_mode_count == 0) {
        LOG_ERROR("surface has no supported present modes");
        destroy();
        return false;
    }

    std::vector<VkPresentModeKHR> present_modes(present_mode_count);

    if (!vk_ok(
            vkGetPhysicalDeviceSurfacePresentModesKHR(
                ctx.physical_device(),
                ctx.surface(),
                &present_mode_count,
                present_modes.data()),
            "query present modes")) {
        destroy();
        return false;
    }

    present_modes.resize(present_mode_count);

    // Step 2: Select the surface format
    VkSurfaceFormatKHR surface_format{};

    if (!choose_surface_format(formats, surface_format)) {
        destroy();
        return false;
    }

    m_format = surface_format.format;
    m_color_space = surface_format.colorSpace;

    // Step 3: Select the present mode
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

    if (!choose_present_mode(present_modes, vsync, present_mode)) {
        destroy();
        return false;
    }

    // Step 4: Select the swapchain extent
    constexpr std::uint32_t variable_extent =
        std::numeric_limits<std::uint32_t>::max();

    if (capabilities.currentExtent.width != variable_extent) {
        m_extent = capabilities.currentExtent;
    } else {
        m_extent.width = std::clamp(
            desired_width,
            capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);

        m_extent.height = std::clamp(
            desired_height,
            capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
    }

    if (m_extent.width == 0 || m_extent.height == 0) {
        LOG_WARN(
            "cannot create swapchain for zero-sized window: %ux%u",
            m_extent.width, m_extent.height);
        destroy();
        return false;
    }

    // Step 5: Request one more image than the surface minimum
    std::uint32_t requested_image_count =
        capabilities.minImageCount + 1;

    if (capabilities.maxImageCount != 0 &&
        requested_image_count > capabilities.maxImageCount) {
        requested_image_count = capabilities.maxImageCount;
    }

    // Step 6: Validate image usage and composite alpha.
    if ((capabilities.supportedUsageFlags &
         VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0) {
        LOG_ERROR("surface does not support color attachment images");
        destroy();
        return false;
    }

    constexpr VkCompositeAlphaFlagBitsKHR composite_alpha =
        VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

    if ((capabilities.supportedCompositeAlpha & composite_alpha) == 0) {
        LOG_ERROR("surface does not support opaque composite alpha");
        destroy();
        return false;
    }

    VkSwapchainCreateInfoKHR swapchain_info{};
    swapchain_info.sType =
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    swapchain_info.surface = ctx.surface();
    swapchain_info.minImageCount = requested_image_count;
    swapchain_info.imageFormat = m_format;
    swapchain_info.imageColorSpace = m_color_space;
    swapchain_info.imageExtent = m_extent;
    swapchain_info.imageArrayLayers = 1;
    swapchain_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    swapchain_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    swapchain_info.queueFamilyIndexCount = 0;
    swapchain_info.pQueueFamilyIndices = nullptr;
    swapchain_info.preTransform = capabilities.currentTransform;
    swapchain_info.compositeAlpha = composite_alpha;
    swapchain_info.presentMode = present_mode;
    swapchain_info.clipped = VK_TRUE;
    swapchain_info.oldSwapchain = VK_NULL_HANDLE;

    if (!vk_ok(
            vkCreateSwapchainKHR(
                m_device, &swapchain_info, nullptr, &m_swapchain),
            "create swapchain")) {
        destroy();
        return false;
    }

    // Step 7: Retrieve the actual swapchain images
    std::uint32_t actual_image_count = 0;

    if (!vk_ok(
            vkGetSwapchainImagesKHR(
                m_device, m_swapchain, &actual_image_count, nullptr),
            "query swapchain image count")) {
        destroy();
        return false;
    }

    if (actual_image_count == 0) {
        LOG_ERROR("swapchain returned zero images");
        destroy();
        return false;
    }

    m_images.resize(actual_image_count);

    if (!vk_ok(
            vkGetSwapchainImagesKHR(
                m_device, m_swapchain,
                &actual_image_count, m_images.data()),
            "retrieve swapchain images")) {
        destroy();
        return false;
    }

    m_images.resize(actual_image_count);

    // Step 8: Create one 2D color image view per swapchain image
    m_image_views.reserve(m_images.size());

    for (VkImage image : m_images) {
        VkImageViewCreateInfo view_info{};
        view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = m_format;

        view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

        view_info.subresourceRange.aspectMask =
            VK_IMAGE_ASPECT_COLOR_BIT;
        view_info.subresourceRange.baseMipLevel = 0;
        view_info.subresourceRange.levelCount = 1;
        view_info.subresourceRange.baseArrayLayer = 0;
        view_info.subresourceRange.layerCount = 1;

        VkImageView view = VK_NULL_HANDLE;

        if (!vk_ok(
                vkCreateImageView(
                    m_device, &view_info, nullptr, &view),
                "create swapchain image view")) {
            destroy();
            return false;
        }

        m_image_views.push_back(view);
    }

    // Step 9: Report the choices
    LOG_INFO(
        "swapchain format: %s (%d), color space: %s (%d)",
        format_name(m_format),
        static_cast<int>(m_format),
        color_space_name(m_color_space),
        static_cast<int>(m_color_space));

    LOG_INFO("swapchain present mode: %s",
             present_mode_name(present_mode));

    LOG_INFO("swapchain extent: %ux%u",
             m_extent.width, m_extent.height);

    LOG_INFO("swapchain images: %u", image_count());

    return true;
}

} // namespace gfx
