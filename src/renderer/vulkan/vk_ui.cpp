#include "renderer/vulkan/vk_ui.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_memory.h"
#include "renderer/vulkan/vk_pipeline.h"
#include "renderer/vulkan/vk_shader.h"
#include "renderer/vulkan/vk_upload.h"

#include <algorithm>
#include <cstddef>

namespace gfx {

namespace {

// Smallest size of a per-frame UI buffer. A typical panel needs a few
// kilobytes; growing doubles the size
constexpr VkDeviceSize kMinBufferBytes = 64 * 1024;

// Push constant: the framebuffer size in pixels
struct UiPush {
    float viewport[2];
};

static_assert(sizeof(UiPush) == 8, "UiPush must be two floats");

// Makes sure buffer can hold bytes, replacing it with a larger one if
// needed. Safe only while the GPU is not using this buffer
bool ensure_capacity(const VulkanContext& ctx,
                     std::unique_ptr<VulkanBuffer>& buffer,
                     VkDeviceSize bytes,
                     VkBufferUsageFlags usage) {
    if (buffer && buffer->buffer() != VK_NULL_HANDLE &&
        buffer->size() >= bytes) {
        return true;
    }

    const VkDeviceSize old_size = buffer ? buffer->size() : 0;
    const VkDeviceSize new_size =
        std::max({bytes, old_size * 2, kMinBufferBytes});

    if (!buffer) {
        buffer = std::make_unique<VulkanBuffer>();
    }

    return buffer->create(
        ctx,
        new_size,
        usage,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

} // namespace

VulkanUiRenderer::~VulkanUiRenderer() {
    shutdown();
}

bool VulkanUiRenderer::init(const VulkanContext& ctx,
                            std::uint32_t frame_count) {
    shutdown();

    if (ctx.device() == VK_NULL_HANDLE || frame_count == 0) {
        LOG_ERROR("invalid arguments for the Vulkan UI renderer");
        return false;
    }

    m_device = ctx.device();
    m_frames.resize(frame_count);

    // Nearest filtering: glyphs are drawn at their exact size on whole
    // pixels, so every pixel samples one texel centre
    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = VK_FILTER_NEAREST;
    sampler_info.minFilter = VK_FILTER_NEAREST;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sampler_info.maxLod = 0.0f;

    if (!vk_ok(
            vkCreateSampler(m_device, &sampler_info, nullptr, &m_sampler),
            "create UI sampler")) {
        shutdown();
        return false;
    }

    // Set 0, binding 0: the font atlas, read by the fragment shader
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo set_layout_info{};
    set_layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set_layout_info.bindingCount = 1;
    set_layout_info.pBindings = &binding;

    if (!vk_ok(
            vkCreateDescriptorSetLayout(
                m_device, &set_layout_info, nullptr, &m_set_layout),
            "create UI descriptor set layout")) {
        shutdown();
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = 1;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 1;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    if (!vk_ok(
            vkCreateDescriptorPool(m_device, &pool_info, nullptr, &m_pool),
            "create UI descriptor pool")) {
        shutdown();
        return false;
    }

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = m_pool;
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &m_set_layout;

    if (!vk_ok(
            vkAllocateDescriptorSets(m_device, &allocate_info, &m_set),
            "allocate UI descriptor set")) {
        shutdown();
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(UiPush);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &m_set_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    if (!vk_ok(
            vkCreatePipelineLayout(
                m_device, &layout_info, nullptr, &m_pipeline_layout),
            "create UI pipeline layout")) {
        shutdown();
        return false;
    }

    return true;
}

void VulkanUiRenderer::destroy_pipeline() {
    if (m_device != VK_NULL_HANDLE && m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
    }

    m_pipeline = VK_NULL_HANDLE;
}

bool VulkanUiRenderer::create_pipeline(VkFormat color_format,
                                       VkFormat depth_format) {
    destroy_pipeline();

    if (m_device == VK_NULL_HANDLE ||
        m_pipeline_layout == VK_NULL_HANDLE) {
        LOG_ERROR("Vulkan UI renderer is not initialized");
        return false;
    }

    VkShaderModule vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/ui.vert.spv");

    if (vertex_shader == VK_NULL_HANDLE) {
        return false;
    }

    VkShaderModule fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/ui.frag.spv");

    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, vertex_shader, nullptr);
        return false;
    }

    // One interleaved binding: UiVertex, 20 bytes
    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = sizeof(UiVertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position{};
    position.location = 0;
    position.binding = 0;
    position.format = VK_FORMAT_R32G32_SFLOAT;
    position.offset = static_cast<std::uint32_t>(offsetof(UiVertex, x));

    VkVertexInputAttributeDescription uv{};
    uv.location = 1;
    uv.binding = 0;
    uv.format = VK_FORMAT_R32G32_SFLOAT;
    uv.offset = static_cast<std::uint32_t>(offsetof(UiVertex, u));

    // Four bytes R, G, B, A, read as 0..1
    VkVertexInputAttributeDescription color{};
    color.location = 2;
    color.binding = 0;
    color.format = VK_FORMAT_R8G8B8A8_UNORM;
    color.offset = static_cast<std::uint32_t>(offsetof(UiVertex, color));

    VulkanPipelineDesc desc{};
    desc.vert = vertex_shader;
    desc.frag = fragment_shader;
    desc.layout = m_pipeline_layout;
    desc.color_format = color_format;

    // The UI draws inside the same rendering pass as the scene, which
    // has a depth attachment, so the pipeline must name its format even
    // though it never touches depth
    desc.depth_format = depth_format;
    desc.bindings = {binding};
    desc.attributes = {position, uv, color};
    desc.depth_test = false;
    desc.depth_write = false;
    desc.blend = true;

    m_pipeline = create_graphics_pipeline(m_device, desc);

    vkDestroyShaderModule(m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(m_device, vertex_shader, nullptr);

    if (m_pipeline == VK_NULL_HANDLE) {
        return false;
    }

    LOG_INFO("Vulkan UI pipeline created");
    return true;
}

void VulkanUiRenderer::destroy_font() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_font_view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, m_font_view, nullptr);
        }

        if (m_font_image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, m_font_image, nullptr);
        }

        if (m_font_memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, m_font_memory, nullptr);
        }
    }

    m_font_view = VK_NULL_HANDLE;
    m_font_image = VK_NULL_HANDLE;
    m_font_memory = VK_NULL_HANDLE;
    m_font = nullptr;
}

bool VulkanUiRenderer::upload_font(const VulkanContext& ctx,
                                   VulkanUploader& uploader,
                                   const FontAtlas& font) {
    // Replacing the atlas: earlier frames may still sample the old
    // image and the descriptor set points at it. The atlas changes at
    // most a few times per run, so simply wait for the GPU
    if (m_font_image != VK_NULL_HANDLE) {
        (void)vk_ok(vkDeviceWaitIdle(m_device),
                    "wait for device idle before replacing the UI font");
    }

    destroy_font();

    // 1. A single-channel image the shader samples, filled by a copy
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = VK_FORMAT_R8_UNORM;
    image_info.extent = {static_cast<std::uint32_t>(font.width),
                         static_cast<std::uint32_t>(font.height),
                         1};
    image_info.mipLevels = 1;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage =
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    if (!vk_ok(
            vkCreateImage(m_device, &image_info, nullptr, &m_font_image),
            "create UI font image")) {
        destroy_font();
        return false;
    }

    // 2. Device-local memory for it
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(m_device, m_font_image, &requirements);

    std::uint32_t memory_type_index = 0;

    if (!find_memory_type(ctx.physical_device(),
                          requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          memory_type_index)) {
        destroy_font();
        return false;
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;

    if (!vk_ok(
            vkAllocateMemory(
                m_device, &allocation_info, nullptr, &m_font_memory),
            "allocate UI font image memory") ||
        !vk_ok(
            vkBindImageMemory(m_device, m_font_image, m_font_memory, 0),
            "bind UI font image memory")) {
        destroy_font();
        return false;
    }

    // 3. Copy the pixels; the image ends up ready for sampling
    if (!uploader.upload_to_image(
            ctx,
            m_font_image,
            static_cast<std::uint32_t>(font.width),
            static_cast<std::uint32_t>(font.height),
            font.alpha.data(),
            static_cast<VkDeviceSize>(font.alpha.size()))) {
        destroy_font();
        return false;
    }

    // 4. View and descriptor
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = m_font_image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = VK_FORMAT_R8_UNORM;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = 1;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (!vk_ok(
            vkCreateImageView(m_device, &view_info, nullptr, &m_font_view),
            "create UI font image view")) {
        destroy_font();
        return false;
    }

    VkDescriptorImageInfo image_descriptor{};
    image_descriptor.sampler = m_sampler;
    image_descriptor.imageView = m_font_view;
    image_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = m_set;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &image_descriptor;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    m_font = &font;

    LOG_INFO("Vulkan UI font uploaded: %dx%d", font.width, font.height);
    return true;
}

bool VulkanUiRenderer::prepare(const VulkanContext& ctx,
                               VulkanUploader& uploader,
                               std::uint32_t frame_index,
                               const UiDrawList& list) {
    if (frame_index >= m_frames.size()) {
        return false;
    }

    FrameBuffers& frame = m_frames[frame_index];
    frame.index_count = 0;

    const FontAtlas* font = list.font();

    if (list.empty() || font == nullptr || m_pipeline == VK_NULL_HANDLE) {
        return false;
    }

    if (font != m_font && !upload_font(ctx, uploader, *font)) {
        return false;
    }

    const std::vector<UiVertex>& vertices = list.vertices();
    const std::vector<std::uint32_t>& indices = list.indices();

    const VkDeviceSize vertex_bytes =
        static_cast<VkDeviceSize>(vertices.size() * sizeof(UiVertex));
    const VkDeviceSize index_bytes =
        static_cast<VkDeviceSize>(indices.size() * sizeof(std::uint32_t));

    // The caller waited for this slot's fence, so the GPU is done with
    // these buffers and they can be rewritten (or replaced)
    if (!ensure_capacity(ctx, frame.vertices, vertex_bytes,
                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT) ||
        !ensure_capacity(ctx, frame.indices, index_bytes,
                         VK_BUFFER_USAGE_INDEX_BUFFER_BIT)) {
        LOG_ERROR("failed to create Vulkan UI buffers");
        return false;
    }

    if (!frame.vertices->write(vertices.data(), vertex_bytes) ||
        !frame.indices->write(indices.data(), index_bytes)) {
        return false;
    }

    frame.index_count = static_cast<std::uint32_t>(indices.size());
    return true;
}

void VulkanUiRenderer::draw(VkCommandBuffer cmd,
                            VkExtent2D extent,
                            std::uint32_t frame_index) const {
    if (frame_index >= m_frames.size() ||
        m_pipeline == VK_NULL_HANDLE ||
        m_font_view == VK_NULL_HANDLE ||
        extent.width == 0 ||
        extent.height == 0) {
        return;
    }

    const FrameBuffers& frame = m_frames[frame_index];

    if (frame.index_count == 0) {
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
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipeline_layout,
        0,
        1,
        &m_set,
        0,
        nullptr);

    const UiPush push{{static_cast<float>(extent.width),
                       static_cast<float>(extent.height)}};

    vkCmdPushConstants(cmd, m_pipeline_layout, VK_SHADER_STAGE_VERTEX_BIT,
                       0, sizeof(push), &push);

    const VkBuffer vertex_buffer = frame.vertices->buffer();
    const VkDeviceSize offset = 0;

    vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, &offset);
    vkCmdBindIndexBuffer(cmd, frame.indices->buffer(), 0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdDrawIndexed(cmd, frame.index_count, 1, 0, 0, 0);
}

void VulkanUiRenderer::shutdown() {
    destroy_pipeline();
    destroy_font();

    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipeline_layout, nullptr);
        }

        // Destroying the pool frees the descriptor set
        if (m_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        }

        if (m_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_set_layout, nullptr);
        }

        if (m_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_sampler, nullptr);
        }
    }

    m_pipeline_layout = VK_NULL_HANDLE;
    m_pool = VK_NULL_HANDLE;
    m_set = VK_NULL_HANDLE;
    m_set_layout = VK_NULL_HANDLE;
    m_sampler = VK_NULL_HANDLE;

    // Buffers are destroyed while the device is still alive
    m_frames.clear();

    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
