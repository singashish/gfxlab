#include "renderer/vulkan/vk_pipeline.h"

namespace gfx {

VkPipeline create_graphics_pipeline(VkDevice device,
                            const VulkanPipelineDesc& desc) {

    if (device == VK_NULL_HANDLE ||
        desc.vert == VK_NULL_HANDLE ||
        desc.frag == VK_NULL_HANDLE ||
        desc.layout == VK_NULL_HANDLE ||
        desc.color_format == VK_FORMAT_UNDEFINED ||
        desc.depth_format == VK_FORMAT_UNDEFINED) {
        LOG_ERROR("invalid Vulkan graphics pipeline description");
        return VK_NULL_HANDLE;
    }

    VkPipelineShaderStageCreateInfo shader_stages[2]{};

    shader_stages[0].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
    shader_stages[0].module = desc.vert;
    shader_stages[0].pName = "main";

    shader_stages[1].sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    shader_stages[1].module = desc.frag;
    shader_stages[1].pName = "main";

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    vertex_input.vertexBindingDescriptionCount =
        static_cast<std::uint32_t>(desc.bindings.size());
    vertex_input.pVertexBindingDescriptions =
        desc.bindings.empty() ? nullptr : desc.bindings.data();

    vertex_input.vertexAttributeDescriptionCount =
        static_cast<std::uint32_t>(desc.attributes.size());
    vertex_input.pVertexAttributeDescriptions =
        desc.attributes.empty() ? nullptr : desc.attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology = desc.topology;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport_state.viewportCount = 1;
    viewport_state.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterization{};
    rasterization.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterization.depthClampEnable = VK_FALSE;
    rasterization.rasterizerDiscardEnable = VK_FALSE;
    rasterization.polygonMode = VK_POLYGON_MODE_FILL;
    rasterization.cullMode = VK_CULL_MODE_NONE;
    rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterization.depthBiasEnable = VK_FALSE;
    rasterization.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    multisampling.sampleShadingEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth_stencil.depthTestEnable =
        desc.depth_test ? VK_TRUE : VK_FALSE;
    depth_stencil.depthWriteEnable =
        desc.depth_write ? VK_TRUE : VK_FALSE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState blend_attachment{};
    blend_attachment.blendEnable =
        desc.blend ? VK_TRUE : VK_FALSE;

    blend_attachment.srcColorBlendFactor =
        VK_BLEND_FACTOR_SRC_ALPHA;
    blend_attachment.dstColorBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;

    blend_attachment.srcAlphaBlendFactor =
        VK_BLEND_FACTOR_SRC_ALPHA;
    blend_attachment.dstAlphaBlendFactor =
        VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

    blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT |
        VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT |
        VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &blend_attachment;

    const VkDynamicState dynamic_states[] = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };

    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic_state.dynamicStateCount = 2;
    dynamic_state.pDynamicStates = dynamic_states;

    VkPipelineRenderingCreateInfo rendering_info{};
    rendering_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachmentFormats = &desc.color_format;
    rendering_info.depthAttachmentFormat = desc.depth_format;

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType =
        VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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
    pipeline_info.layout = desc.layout;
    pipeline_info.renderPass = VK_NULL_HANDLE;
    pipeline_info.subpass = 0;

    VkPipeline pipeline = VK_NULL_HANDLE;

    const VkResult result = vkCreateGraphicsPipelines(
        device,
        VK_NULL_HANDLE,
        1,
        &pipeline_info,
        nullptr,
        &pipeline);

    if (!vk_ok(result, "create Vulkan graphics pipeline")) {
        if (pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, pipeline, nullptr);
        }
        return VK_NULL_HANDLE;
    }

    return pipeline;
}

} // namespace gfx
