#include "renderer/vulkan/vk_buffer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_mesh.h"
#include "renderer/vulkan/vk_pipeline.h"
#include "renderer/vulkan/vk_shader.h"
#include "renderer/vulkan/vk_upload.h"

#include <cstddef>
#include <limits>

namespace gfx {

static_assert(sizeof(Vec3) == 12,
              "Vec3 must contain exactly three floats");

static_assert(sizeof(Mat4) == 64,
              "Mat4 must be 64 bytes for push constants");

struct MeshPush {
    Mat4 model;
    Mat4 normal_matrix;
};

static_assert(sizeof(MeshPush) == 128,
              "MeshPush must contain two 64-byte matrices");

VulkanMeshRenderer::~VulkanMeshRenderer() {
    shutdown();
}

void VulkanMeshRenderer::destroy_pipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }

        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                m_device, m_pipeline_layout, nullptr);
        }
    }

    m_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
}

bool VulkanMeshRenderer::init(VkDevice device, VkFormat color_format,
            VkFormat depth_format, VkDescriptorSetLayout descriptor_layout) {

    // Rebuild the pipeline without deleting uploaded mesh buffers
    destroy_pipeline();

    if (device == VK_NULL_HANDLE ||
        color_format == VK_FORMAT_UNDEFINED ||
        depth_format == VK_FORMAT_UNDEFINED ||
        descriptor_layout == VK_NULL_HANDLE) {
        LOG_ERROR("invalid arguments for Vulkan mesh pipeline");
        return false;
    }

    m_device = device;

    VkShaderModule vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/mesh.vert.spv");

    if (vertex_shader == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    VkShaderModule fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/mesh.frag.spv");

    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            m_device, vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    VkPushConstantRange push_range{};
    push_range.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    push_range.offset = 0;
    push_range.size = sizeof(MeshPush);

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 1;
    layout_info.pSetLayouts = &descriptor_layout;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    if (!vk_ok(
            vkCreatePipelineLayout(
                m_device,
                &layout_info,
                nullptr,
                &m_pipeline_layout),
            "create mesh pipeline layout")) {
        vkDestroyShaderModule(
            m_device, fragment_shader, nullptr);
        vkDestroyShaderModule(
            m_device, vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = 6 * sizeof(float);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position{};
    position.location = 0;
    position.binding = 0;
    position.format = VK_FORMAT_R32G32B32_SFLOAT;
    position.offset = 0;

    VkVertexInputAttributeDescription normal{};
    normal.location = 1;
    normal.binding = 0;
    normal.format = VK_FORMAT_R32G32B32_SFLOAT;
    normal.offset = 3 * sizeof(float);

    VulkanPipelineDesc desc{};
    desc.vert = vertex_shader;
    desc.frag = fragment_shader;
    desc.layout = m_pipeline_layout;
    desc.color_format = color_format;
    desc.depth_format = depth_format;
    desc.bindings = {binding};
    desc.attributes = {position, normal};

    m_pipeline = create_graphics_pipeline(m_device, desc);

    vkDestroyShaderModule(
        m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(
        m_device, vertex_shader, nullptr);

    if (m_pipeline == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    LOG_INFO("Vulkan mesh pipeline created");
    return true;
}

MeshId VulkanMeshRenderer::create(const VulkanContext& ctx,
                                  VulkanUploader& uploader,
                                  const Mesh& mesh) {

    if (m_device == VK_NULL_HANDLE ||
        ctx.device() != m_device ||
        m_pipeline == VK_NULL_HANDLE) {
        LOG_ERROR("Vulkan mesh renderer is not initialized");
        return 0;
    }

    if (mesh.positions.empty() ||
        mesh.indices.empty() ||
        mesh.indices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        LOG_ERROR("invalid mesh data or index count");
        return 0;
    }

    if (mesh.normals.size() != mesh.positions.size()) {
        LOG_ERROR(
            "mesh normals and positions have different sizes");
        return 0;
    }

    for (const std::uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size()) {
            LOG_ERROR("mesh contains an out-of-range index");
            return 0;
        }
    }

    constexpr std::size_t kFloatsPerVertex = 6;

    if (mesh.positions.size() >
        std::numeric_limits<std::size_t>::max() /
            (kFloatsPerVertex * sizeof(float))) {
        LOG_ERROR("mesh vertex buffer size overflow");
        return 0;
    }

    if (mesh.indices.size() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint32_t)) {
        LOG_ERROR("mesh index buffer size overflow");
        return 0;
    }

    std::vector<float> vertex_data;
    vertex_data.reserve(
        mesh.positions.size() * kFloatsPerVertex);

    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
        const Vec3& p = mesh.positions[i];
        const Vec3& n = mesh.normals[i];

        vertex_data.push_back(p.x);
        vertex_data.push_back(p.y);
        vertex_data.push_back(p.z);

        vertex_data.push_back(n.x);
        vertex_data.push_back(n.y);
        vertex_data.push_back(n.z);
    }

    const VkDeviceSize vertex_bytes =
        static_cast<VkDeviceSize>(
            vertex_data.size() * sizeof(float));

    const VkDeviceSize index_bytes =
        static_cast<VkDeviceSize>(
            mesh.indices.size() * sizeof(std::uint32_t));

    auto gpu_mesh = std::make_unique<GpuMesh>();

    gpu_mesh->vertex_buffer =
        std::make_unique<VulkanBuffer>();

    if (!uploader.upload_to_device_buffer(
            ctx,
            *gpu_mesh->vertex_buffer,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_data.data(),
            vertex_bytes)) {
        LOG_ERROR("failed to upload mesh vertex buffer");
        return 0;
    }

    gpu_mesh->index_buffer =
        std::make_unique<VulkanBuffer>();

    if (!uploader.upload_to_device_buffer(
            ctx,
            *gpu_mesh->index_buffer,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            mesh.indices.data(),
            index_bytes)) {
        LOG_ERROR("failed to upload mesh index buffer");
        return 0;
    }

    gpu_mesh->index_count =
        static_cast<std::uint32_t>(mesh.indices.size());

    if (m_meshes.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<MeshId>::max())) {
        LOG_ERROR("maximum Vulkan mesh count exceeded");
        return 0;
    }

    m_meshes.push_back(std::move(gpu_mesh));

    return static_cast<MeshId>(m_meshes.size());
}

void VulkanMeshRenderer::draw(VkCommandBuffer cmd, VkExtent2D extent,
        VkDescriptorSet descriptor_set, const std::vector<DrawItem>& items) {

    if (m_pipeline == VK_NULL_HANDLE ||
        m_pipeline_layout == VK_NULL_HANDLE ||
        cmd == VK_NULL_HANDLE ||
        descriptor_set == VK_NULL_HANDLE ||
        extent.width == 0 ||
        extent.height == 0) {
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

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);

    const VkDeviceSize offset = 0;

    for (const DrawItem& item : items) {
        if (item.mesh == 0 ||
            static_cast<std::size_t>(item.mesh) > m_meshes.size()) {
            continue;
        }

        const GpuMesh& mesh =
            *m_meshes[static_cast<std::size_t>(item.mesh) - 1];

        if (mesh.vertex_buffer == nullptr ||
            mesh.index_buffer == nullptr ||
            mesh.index_count == 0) {
            continue;
        }

        const MeshPush push{
            item.model,
            item.normal_matrix
        };

        vkCmdPushConstants(
            cmd,
            m_pipeline_layout,
            VK_SHADER_STAGE_VERTEX_BIT,
            0,
            sizeof(MeshPush),
            &push);

        const VkBuffer vertex_buffer =
            mesh.vertex_buffer->buffer();

        vkCmdBindVertexBuffers(
            cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdBindIndexBuffer(
            cmd,
            mesh.index_buffer->buffer(),
            0,
            VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(
            cmd, mesh.index_count, 1, 0, 0, 0);
    }
}

void VulkanMeshRenderer::shutdown() {
    // The renderer must be idle before this is called
    m_meshes.clear();
    destroy_pipeline();
    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
