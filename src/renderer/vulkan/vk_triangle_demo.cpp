#include "renderer/vulkan/vk_triangle_demo.h"
#include "renderer/vulkan/vk_shader.h"

namespace gfx {

VulkanTriangleDemo::~VulkanTriangleDemo() {
    shutdown();
}

bool VulkanTriangleDemo::init(VkDevice device, VkFormat color_format, VkFormat depth_format) {
    shutdown();

    if (device == VK_NULL_HANDLE || color_format == VK_FORMAT_UNDEFINED) {
        LOG_ERROR("Cannot initialize triangle demo with invalid device or format");
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

    VkPipelineShaderStageCreateInfo shader_stages[2]{};

    shader_stages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = vertex_shader;
    shader_stages[0].pName = "main";

    shader_stages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = fragment_shader;
    shader_stages[1].pName = "main";

    // Dynamic rendering: declare the colour format used by this pipeline
    VkPipelineRenderingCreateInfo rendering_info{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO
    };
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &color_format;
    rendering_info.depthAttachmentFormat = depth_format;

    VkPipelineVertexInputStateCreateInfo vertex_input{
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO
    };
    // No vertex bindings or attributes: the shader uses gl_VertexIndex

    VkPipelineInputAssemblyStateCreateInfo input_assembly{
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO
    };
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state{
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO
    };
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;
    // Both are dynamic, so no viewport or scissor pointers are supplied

    VkPipelineRasterizationStateCreateInfo rasterization{
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO
    };
    rasterization.depthClampEnable = VK_FALSE;
    rasterization.rasterizerDiscardEnable = VK_FALSE;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.depthBiasEnable = VK_FALSE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO
    };
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.sampleShadingEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable = VK_FALSE;
    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blending{
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO
    };
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &blend_attachment;

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO
    };
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;

    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;

    VkGraphicsPipelineCreateInfo pipeline_info{
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO
    };
    pipeline_info.pNext = &rendering_info;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterization;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = m_pipeline_layout;
    pipeline_info.renderPass = VK_NULL_HANDLE;
    pipeline_info.subpass = 0;

    const VkResult result = vkCreateGraphicsPipelines(
        m_device,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        nullptr,
        &m_pipeline);

    // Shader modules are needed to create the pipeline, not to use it
    vkDestroyShaderModule(m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(m_device, vertex_shader, nullptr);

    if (!vk_ok(result, "vkCreateGraphicsPipelines")) {
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
