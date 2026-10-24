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
class VulkanTextureStore;

class VulkanMeshRenderer {
public:
    VulkanMeshRenderer() = default;
    ~VulkanMeshRenderer();

    VulkanMeshRenderer(const VulkanMeshRenderer&) = delete;
    VulkanMeshRenderer& operator=(const VulkanMeshRenderer&) = delete;

    // frame_layout is set 0 (the frame uniforms), texture_layout is set 1
    // (one base-colour texture per draw call, from VulkanTextureStore)
    bool init(VkDevice device, VkFormat color_format,
              VkFormat depth_format, VkDescriptorSetLayout frame_layout,
              VkDescriptorSetLayout texture_layout);

    void shutdown();

    // textures[i] is the TextureId of Model::images[i] (0 = none); a
    // material's base_color_image indexes this list. The meshes keep the
    // textures' descriptor sets, so the store must outlive them
    MeshId create(const VulkanContext& ctx, VulkanUploader& uploader,
                  const Mesh& mesh,
                  const std::vector<TextureId>& textures,
                  const VulkanTextureStore& store);

    void draw(VkCommandBuffer cmd, VkExtent2D extent,
              VkDescriptorSet descriptor_set,
              const std::vector<DrawItem>& items);

    void draw_wireframe(VkCommandBuffer cmd,
                        VkExtent2D extent,
                        VkDescriptorSet descriptor_set,
                        const std::vector<DrawItem>& items);

private:
    // One draw call: a run of triangles that share a material
    struct GpuRange {
        std::uint32_t first_index = 0;
        std::uint32_t index_count = 0;
        Vec3 base_color = {0.8f, 0.8f, 0.8f};

        // Set 1: the material's texture (the white one if it has none).
        // Owned by the texture store
        VkDescriptorSet texture_set = VK_NULL_HANDLE;
    };

    struct GpuMesh {
        std::unique_ptr<VulkanBuffer> vertex_buffer;
        std::unique_ptr<VulkanBuffer> index_buffer;
        std::unique_ptr<VulkanBuffer> edge_buffer;

        // Triangles grouped by material, one entry per material
        std::vector<GpuRange> ranges;
        std::uint32_t edge_index_count = 0;
    };

    void destroy_pipeline();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
    VkPipeline m_wire_pipeline = VK_NULL_HANDLE;

    // Both pipelines use this same layout
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;

    std::vector<std::unique_ptr<GpuMesh>> m_meshes;
};

} // namespace gfx
