#include "renderer/vulkan/vk_triangle_demo.h"
#include "renderer/vulkan/vk_pipeline.h"
#include "renderer/vulkan/vk_shader.h"

namespace gfx {

VulkanTriangleDemo::~VulkanTriangleDemo() {
    shutdown();
}

bool VulkanTriangleDemo::init(VkDevice device, VkFormat color_format, VkFormat depth_format) {
    shutdown();

    if (device == VK_NULL_HANDLE ||
        color_format == VK_FORMAT_UNDEFINED ||
        depth_format == VK_FORMAT_UNDEFINED) {
        LOG_ERROR("invalid arguments for Vulkan triangle pipeline");
        return false;
    }

    m_device = device;

    VkShaderModule vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/triangle.vert.spv");

    if (vertex_shader == VK_NULL_HANDLE) {
        shutdown();
        return false;
    }

    VkShaderModule fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/triangle.frag.spv");

    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(m_device, vertex_shader, nullptr);
        shutdown();
        return false;
    }

    VkPipelineLayoutCreateInfo layout_info{
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO
    };

    if (!vk_ok(
            vkCreatePipelineLayout(
                m_device, &layout_info, nullptr, &m_pipeline_layout),
            "vkCreatePipelineLayout")) {
        vkDestroyShaderModule(m_device, fragment_shader, nullptr);
        vkDestroyShaderModule(m_device, vertex_shader, nullptr);
        shutdown();
        return false;
    }

    VulkanPipelineDesc desc{};
    desc.vert = vertex_shader;
    desc.frag = fragment_shader;
    desc.layout = m_pipeline_layout;
    desc.color_format = color_format;
    desc.depth_format = depth_format;

    // No vertex bindings or attributes: gl_VertexIndex
    // Default depth testing/writing matches the original triangle pipeline
    m_pipeline = create_graphics_pipeline(m_device, desc);

    // Shader modules are no longer needed after pipeline creation
    vkDestroyShaderModule(m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(m_device, vertex_shader, nullptr);

    if (m_pipeline == VK_NULL_HANDLE) {
        shutdown();
        return false;
    }

    LOG_INFO("Vulkan triangle pipeline created");
    return true;
}

void VulkanTriangleDemo::record(VkCommandBuffer cmd, VkExtent2D extent) {
    if (m_pipeline == VK_NULL_HANDLE ||
        cmd == VK_NULL_HANDLE ||
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

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdBindPipeline(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipeline);

    vkCmdDraw(cmd, 6, 1, 0, 0);
}

void VulkanTriangleDemo::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }

        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                m_device, m_pipeline_layout, nullptr);
            m_pipeline_layout = VK_NULL_HANDLE;
        }
    }

    m_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
