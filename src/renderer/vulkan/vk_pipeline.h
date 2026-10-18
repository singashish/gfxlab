#pragma once

#include "renderer/vulkan/vk_api.h"

#include <vector>

namespace gfx {

struct VulkanPipelineDesc {
    VkShaderModule vert = VK_NULL_HANDLE;
    VkShaderModule frag = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;

    VkFormat color_format = VK_FORMAT_UNDEFINED;
    VkFormat depth_format = VK_FORMAT_UNDEFINED;

    std::vector<VkVertexInputBindingDescription> bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;

    bool depth_test = true;
    bool depth_write = true;
    bool blend = false;
};

// Returns VK_NULL_HANDLE on failure.
// The caller owns the returned pipeline and must destroy it
VkPipeline create_graphics_pipeline(VkDevice device,
                            const VulkanPipelineDesc& desc);

} // namespace gfx
