#include "renderer/vulkan/vk_depth.h"

#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_memory.h"

namespace gfx {
namespace {

const char* depth_format_name(VkFormat format) {
    switch (format) {
    case VK_FORMAT_D32_SFLOAT:
        return "VK_FORMAT_D32_SFLOAT";
    default:
        return "unknown";
    }
}

} // namespace

VulkanDepthBuffer::~VulkanDepthBuffer() {
    destroy();
}

void VulkanDepthBuffer::destroy() {
    // Destroy dependent resources before their backing image/memory.
    if (m_device != VK_NULL_HANDLE) {
        if (m_view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_view, nullptr);
        }

        if (m_image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, m_image, nullptr);
        }

        if (m_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_memory, nullptr);
        }
    }

    m_view = VK_NULL_HANDLE;
    m_image = VK_NULL_HANDLE;
    m_memory = VK_NULL_HANDLE;
    m_format = VK_FORMAT_UNDEFINED;
    m_device = VK_NULL_HANDLE;
}

bool VulkanDepthBuffer::create(const VulkanContext& ctx, VkExtent2D extent) {

    // Also makes create() safe to call on an existing object
    destroy();

    if (ctx.device() == VK_NULL_HANDLE ||
        ctx.physical_device() == VK_NULL_HANDLE) {
        LOG_ERROR(
            "cannot create depth buffer: "
            "Vulkan context is incomplete");
        return false;
    }

    if (extent.width == 0 || extent.height == 0) {
        LOG_ERROR(
            "cannot create depth buffer for zero extent: %ux%u",
            extent.width,
            extent.height);
        return false;
    }

    m_device = ctx.device();
    m_format = VK_FORMAT_D32_SFLOAT;

    // Step 1: Verify depth-attachment support
    VkFormatProperties format_properties{};

    vkGetPhysicalDeviceFormatProperties(
        ctx.physical_device(),
        m_format,
        &format_properties);

    if ((format_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) == 0) {
        LOG_ERROR(
            "%s does not support optimal depth attachments",
            depth_format_name(m_format));
        destroy();
        return false;
    }

    LOG_INFO(
        "depth buffer format: %s (%d)",
        depth_format_name(m_format),
        static_cast<int>(m_format));

    // Step 2: Create the image
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = m_format;
    image_info.extent = {
        extent.width,
        extent.height,
        1
    };
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!vk_ok(
            vkCreateImage(
                m_device,
                &image_info,
                nullptr,
                &m_image),
            "create depth image")) {
        destroy();
        return false;
    }

    // Step 3: Query requirements, allocate device-local memory,
    // and bind that memory to the image
    VkMemoryRequirements requirements{};

    vkGetImageMemoryRequirements(
        m_device,
        m_image,
        &requirements);

    std::uint32_t memory_type_index = 0;

    if (!find_memory_type(
            ctx.physical_device(),
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
            memory_type_index)) {
        destroy();
        return false;
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType =
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;

    if (!vk_ok(
            vkAllocateMemory(
                m_device,
                &allocation_info,
                nullptr,
                &m_memory),
            "allocate depth image memory")) {
        destroy();
        return false;
    }

    if (!vk_ok(
            vkBindImageMemory(
                m_device,
                m_image,
                m_memory,
                0),
            "bind depth image memory")) {
        destroy();
        return false;
    }

    // Step 4: Create a depth-only image view
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = m_format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask =
        VK_IMAGE_ASPECT_DEPTH_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (!vk_ok(
            vkCreateImageView(
                m_device,
                &view_info,
                nullptr,
                &m_view),
            "create depth image view")) {
        destroy();
        return false;
    }

    LOG_INFO(
        "depth buffer created: %ux%u",
        extent.width,
        extent.height);

    return true;
}

} // namespace gfx
