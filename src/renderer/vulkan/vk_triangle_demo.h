#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

class VulkanTriangleDemo {
public:
    VulkanTriangleDemo() = default;
    ~VulkanTriangleDemo();

    VulkanTriangleDemo(const VulkanTriangleDemo&) = delete;
    VulkanTriangleDemo& operator=(const VulkanTriangleDemo&) = delete;

    bool init(VkDevice device, VkFormat color_format, VkFormat depth_format);
    void record(VkCommandBuffer cmd, VkExtent2D extent);
    void shutdown();

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

} // namespace gfx
