#include "renderer/vulkan/vk_outline.h"

#include "renderer/renderer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_memory.h"
#include "renderer/vulkan/vk_pipeline.h"
#include "renderer/vulkan/vk_shader.h"

namespace gfx {
namespace {

// Read by outline.frag; 3 x vec4 = 48 bytes, fragment stage only
struct OutlinePush {
    float selected_color[4];
    float active_color[4];
    float params[4];  // near, far, depth tolerance, radius
};

static_assert(sizeof(OutlinePush) == 48,
              "OutlinePush must match the push block of outline.frag");

// One image with its memory and a 2D view
bool make_image(const VulkanContext& ctx, VkExtent2D extent,
                VkFormat format, VkImageUsageFlags usage,
                VkImageAspectFlags aspect, const char* what,
                VkImage& image, VkDeviceMemory& memory,
                VkImageView& view) {

    const VkDevice device = ctx.device();

    // The image is both rendered to and sampled
    VkFormatProperties properties{};
    vkGetPhysicalDeviceFormatProperties(
        ctx.physical_device(), format, &properties);

    const VkFormatFeatureFlags needed =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
        ((aspect & VK_IMAGE_ASPECT_DEPTH_BIT) != 0
             ? VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT
             : VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT);

    if ((properties.optimalTilingFeatures & needed) != needed) {
        LOG_ERROR("outline %s: format %d cannot be rendered to and "
                  "sampled", what, static_cast<int>(format));
        return false;
    }

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent = {extent.width, extent.height, 1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!vk_ok(vkCreateImage(device, &image_info, nullptr, &image),
               "create outline image")) {
        return false;
    }

    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device, image, &requirements);

    std::uint32_t memory_type_index = 0;

    if (!find_memory_type(ctx.physical_device(),
                          requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          memory_type_index)) {
        return false;
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;

    if (!vk_ok(vkAllocateMemory(device, &allocation_info, nullptr, &memory),
               "allocate outline image memory") ||
        !vk_ok(vkBindImageMemory(device, image, memory, 0),
               "bind outline image memory")) {
        return false;
    }

    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.subresourceRange.aspectMask = aspect;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    return vk_ok(vkCreateImageView(device, &view_info, nullptr, &view),
                 "create outline image view");
}

void destroy_image(VkDevice device, VkImage& image, VkDeviceMemory& memory,
                   VkImageView& view) {
    if (device != VK_NULL_HANDLE) {
        if (view != VK_NULL_HANDLE) {
            vkDestroyImageView(device, view, nullptr);
        }

        if (image != VK_NULL_HANDLE) {
            vkDestroyImage(device, image, nullptr);
        }

        if (memory != VK_NULL_HANDLE) {
            vkFreeMemory(device, memory, nullptr);
        }
    }

    view = VK_NULL_HANDLE;
    image = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
}

} // namespace

VulkanOutline::~VulkanOutline() {
    shutdown();
}

bool VulkanOutline::init(VkDevice device) {
    shutdown();

    if (device == VK_NULL_HANDLE) {
        LOG_ERROR("invalid device for the Vulkan outline");
        return false;
    }

    m_device = device;

    // Binding 0: the mask, binding 1: the depth. Read with texelFetch, so
    // plain sampled images (no sampler)
    VkDescriptorSetLayoutBinding bindings[2]{};

    for (std::uint32_t i = 0; i < 2; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 2;
    layout_info.pBindings = bindings;

    if (!vk_ok(vkCreateDescriptorSetLayout(
                   m_device, &layout_info, nullptr, &m_set_layout),
               "create outline descriptor set layout")) {
        shutdown();
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    pool_size.descriptorCount = 2;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    if (!vk_ok(vkCreateDescriptorPool(
                   m_device, &pool_info, nullptr, &m_pool),
               "create outline descriptor pool")) {
        shutdown();
        return false;
    }

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = m_pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &m_set_layout;

    if (!vk_ok(vkAllocateDescriptorSets(
                   m_device, &allocate_info, &m_set),
               "allocate outline descriptor set")) {
        shutdown();
        return false;
    }

    return true;
}

void VulkanOutline::destroy_targets() {
    destroy_image(m_device, m_mask_image, m_mask_memory, m_mask_view);
    destroy_image(m_device, m_depth_image, m_depth_memory, m_depth_view);
}

bool VulkanOutline::create_targets(const VulkanContext& ctx,
                                   VkExtent2D extent,
                                   VkFormat depth_format) {
    destroy_targets();

    if (m_device == VK_NULL_HANDLE || m_set == VK_NULL_HANDLE ||
        extent.width == 0 || extent.height == 0) {
        LOG_ERROR("invalid arguments for the outline targets");
        return false;
    }

    if (!make_image(ctx, extent, kOutlineMaskFormat,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_COLOR_BIT, "mask",
                    m_mask_image, m_mask_memory, m_mask_view) ||
        !make_image(ctx, extent, depth_format,
                    VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                        VK_IMAGE_USAGE_SAMPLED_BIT,
                    VK_IMAGE_ASPECT_DEPTH_BIT, "depth",
                    m_depth_image, m_depth_memory, m_depth_view)) {
        destroy_targets();
        return false;
    }

    // The layouts here are the ones the images are in when the outline
    // pass reads them (the renderer's barriers put them there)
    VkDescriptorImageInfo mask_info{};
    mask_info.imageView = m_mask_view;
    mask_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkDescriptorImageInfo depth_info{};
    depth_info.imageView = m_depth_view;
    depth_info.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet writes[2]{};

    for (std::uint32_t i = 0; i < 2; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    }

    writes[0].pImageInfo = &mask_info;
    writes[1].pImageInfo = &depth_info;

    vkUpdateDescriptorSets(m_device, 2, writes, 0, nullptr);

    LOG_INFO("outline targets created: %ux%u", extent.width, extent.height);
    return true;
}

void VulkanOutline::destroy_pipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }

        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        }
    }

    m_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
}

bool VulkanOutline::create_pipeline(VkFormat color_format,
                                    VkFormat depth_format) {
    destroy_pipeline();

    if (m_device == VK_NULL_HANDLE || m_set_layout == VK_NULL_HANDLE ||
        color_format == VK_FORMAT_UNDEFINED ||
        depth_format == VK_FORMAT_UNDEFINED) {
        LOG_ERROR("invalid arguments for the Vulkan outline pipeline");
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(OutlinePush);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &m_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    if (!vk_ok(vkCreatePipelineLayout(
                   m_device, &layout_info, nullptr, &m_pipeline_layout),
               "create outline pipeline layout")) {
        destroy_pipeline();
        return false;
    }

    VkShaderModule vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/outline.vert.spv");

    if (vertex_shader == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    VkShaderModule fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/outline.frag.spv");

    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    // Drawn inside the main rendering, which has a depth attachment, so
    // the pipeline must name its format even though it neither tests nor
    // writes depth
    VulkanPipelineDesc desc{};
    desc.vert = vertex_shader;
    desc.frag = fragment_shader;
    desc.layout = m_pipeline_layout;
    desc.color_format = color_format;
    desc.depth_format = depth_format;
    desc.depth_test = false;
    desc.depth_write = false;
    desc.blend = false;

    m_pipeline = create_graphics_pipeline(m_device, desc);

    vkDestroyShaderModule(m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(m_device, vertex_shader, nullptr);

    if (m_pipeline == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    LOG_INFO("Vulkan outline pipeline created");
    return true;
}

bool VulkanOutline::ready() const {
    return m_pipeline != VK_NULL_HANDLE &&
           m_mask_view != VK_NULL_HANDLE &&
           m_depth_view != VK_NULL_HANDLE;
}

void VulkanOutline::draw(VkCommandBuffer cmd, VkExtent2D extent,
                         float near_plane, float far_plane) const {
    if (!ready() || cmd == VK_NULL_HANDLE ||
        extent.width == 0 || extent.height == 0) {
        return;
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);
    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline_layout,
        0, 1, &m_set, 0, nullptr);

    const Vec3 selected = edge_color(Highlight::HIGHLIGHT_SELECTED);
    const Vec3 active = edge_color(Highlight::HIGHLIGHT_ACTIVE);

    OutlinePush push{};
    push.selected_color[0] = selected.x;
    push.selected_color[1] = selected.y;
    push.selected_color[2] = selected.z;
    push.active_color[0] = active.x;
    push.active_color[1] = active.y;
    push.active_color[2] = active.z;
    push.params[0] = near_plane;
    push.params[1] = far_plane;
    push.params[2] = kOutlineDepthTolerance;
    push.params[3] = static_cast<float>(kOutlineRadius);

    vkCmdPushConstants(cmd, m_pipeline_layout,
                       VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                       sizeof(OutlinePush), &push);

    vkCmdDraw(cmd, 3, 1, 0, 0);
}

void VulkanOutline::shutdown() {
    destroy_pipeline();
    destroy_targets();

    if (m_device != VK_NULL_HANDLE) {
        // Destroying the pool frees its set
        if (m_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        }

        if (m_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_set_layout, nullptr);
        }
    }

    m_pool = VK_NULL_HANDLE;
    m_set_layout = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
