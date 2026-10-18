#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

class VulkanGridRenderer {
public:
    VulkanGridRenderer() = default;
    ~VulkanGridRenderer();

    VulkanGridRenderer(const VulkanGridRenderer&) = delete;
    VulkanGridRenderer& operator=(const VulkanGridRenderer&) = delete;

    bool init(VkDevice device, VkFormat color_format,
              VkFormat depth_format, VkDescriptorSetLayout descriptor_layout);

    void draw(VkCommandBuffer cmd, VkExtent2D extent,
              VkDescriptorSet descriptor_set) const;

    void shutdown();

private:
    void destroy_pipeline();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

} // namespace gfx
