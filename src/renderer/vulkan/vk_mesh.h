#pragma once

#include "core/mesh.h"
#include "renderer/vulkan/vk_api.h"
#include "renderer/vulkan/vk_buffer.h"
#include "renderer/renderer.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

class VulkanContext;
class VulkanUploader;

class VulkanMeshRenderer {
public:
    VulkanMeshRenderer() = default;
    ~VulkanMeshRenderer();

    VulkanMeshRenderer(const VulkanMeshRenderer&) = delete;
    VulkanMeshRenderer& operator=(const VulkanMeshRenderer&) = delete;

    bool init(VkDevice device, VkFormat color_format, VkFormat depth_format);

    void shutdown();

    MeshId create(const VulkanContext& ctx, VulkanUploader& uploader,
                  const Mesh& mesh);

    void draw(VkCommandBuffer cmd, VkExtent2D extent,
              const std::vector<DrawItem>& items, const Mat4& view_proj);

private:
    struct GpuMesh {
        std::unique_ptr<VulkanBuffer> vertex_buffer;
        std::unique_ptr<VulkanBuffer> index_buffer;
        std::uint32_t index_count = 0;
    };

    void destroy_pipeline();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;

    std::vector<std::unique_ptr<GpuMesh>> m_meshes;
};

} // namespace gfx
