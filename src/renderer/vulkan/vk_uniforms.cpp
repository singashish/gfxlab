#include "renderer/vulkan/vk_uniforms.h"
#include "renderer/vulkan/vk_context.h"

namespace gfx {

VulkanFrameUniforms::~VulkanFrameUniforms() {
    shutdown();
}

bool VulkanFrameUniforms::init(const VulkanContext& ctx,
                               std::uint32_t frame_count) {
    shutdown();

    if (ctx.device() == VK_NULL_HANDLE || frame_count == 0) {
        LOG_ERROR("invalid arguments for Vulkan frame uniforms");
        return false;
    }

    m_device = ctx.device();

    // Create one host-visible uniform buffer per frame in flight
    m_buffers.reserve(frame_count);

    for (std::uint32_t i = 0; i < frame_count; ++i) {
        auto buffer = std::make_unique<VulkanBuffer>();

        if (!buffer->create(
                ctx,
                sizeof(FrameUniforms),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            LOG_ERROR("failed to create frame uniform buffer");
            shutdown();
            return false;
        }

        m_buffers.push_back(std::move(buffer));
    }

    // Binding 0: FrameUniforms, visible to both shader stages
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    binding.descriptorCount = 1;
    binding.stageFlags =
        VK_SHADER_STAGE_VERTEX_BIT |
        VK_SHADER_STAGE_FRAGMENT_BIT;
    binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;

    if (!vk_ok(
            vkCreateDescriptorSetLayout(
                m_device, &layout_info, nullptr, &m_layout),
            "create frame uniform descriptor set layout")) {
        shutdown();
        return false;
    }

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_size.descriptorCount = frame_count;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = frame_count;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    if (!vk_ok(
            vkCreateDescriptorPool(
                m_device, &pool_info, nullptr, &m_pool),
            "create frame uniform descriptor pool")) {
        shutdown();
        return false;
    }

    // Allocate all descriptor sets using the same layout
    std::vector<VkDescriptorSetLayout> layouts(
        frame_count, m_layout);

    m_sets.resize(frame_count, VK_NULL_HANDLE);

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = m_pool;
    allocate_info.descriptorSetCount = frame_count;
    allocate_info.pSetLayouts = layouts.data();

    if (!vk_ok(
            vkAllocateDescriptorSets(
                m_device, &allocate_info, m_sets.data()),
            "allocate frame uniform descriptor sets")) {
        shutdown();
        return false;
    }

    // Each descriptor set permanently references its own buffer
    std::vector<VkDescriptorBufferInfo> buffer_infos(frame_count);
    std::vector<VkWriteDescriptorSet> writes(frame_count);

    for (std::uint32_t i = 0; i < frame_count; ++i) {
        buffer_infos[i].buffer = m_buffers[i]->buffer();
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = sizeof(FrameUniforms);

        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = m_sets[i];
        writes[i].dstBinding = 0;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[i].descriptorCount = 1;
        writes[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(
        m_device,
        frame_count,
        writes.data(),
        0,
        nullptr);

    LOG_INFO("Vulkan frame uniforms initialized");
    return true;
}

bool VulkanFrameUniforms::update(std::uint32_t frame_index,
                                 const FrameUniforms& uniforms) {
    if (m_device == VK_NULL_HANDLE ||
        frame_index >= m_buffers.size() ||
        m_buffers[frame_index] == nullptr) {
        LOG_ERROR("invalid frame index for uniform update");
        return false;
    }

    return m_buffers[frame_index]->write(
        &uniforms, sizeof(uniforms));
}

VkDescriptorSet VulkanFrameUniforms::set(
    std::uint32_t frame_index) const {

    if (frame_index >= m_sets.size()) {
        return VK_NULL_HANDLE;
    }

    return m_sets[frame_index];
}

void VulkanFrameUniforms::shutdown() {
    // Destroy descriptor sets by destroying their pool first
    if (m_device != VK_NULL_HANDLE) {
        if (m_pool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_pool, nullptr);
        }

        if (m_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_layout, nullptr);
        }
    }

    m_pool = VK_NULL_HANDLE;
    m_layout = VK_NULL_HANDLE;

    // Buffers are destroyed while the logical device is still alive
    m_buffers.clear();
    m_sets.clear();

    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
