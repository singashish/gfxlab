#pragma once

#include "renderer/vulkan/vk_api.h"
#include "renderer/vulkan/vk_buffer.h"
#include "ui/ui_draw.h"

#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

class VulkanContext;
class VulkanUploader;

// Draws a UiDrawList on top of the frame, the Vulkan counterpart of
// GLUiRenderer:
//  - the font atlas lives in an R8 image, sampled through one
//    descriptor set (image + nearest sampler)
//  - every frame in flight has its own host-visible vertex and index
//    buffers, so the CPU can fill one slot while the GPU still reads the
//    other
//  - the pipeline blends, and neither tests nor writes depth
class VulkanUiRenderer {
public:
    VulkanUiRenderer() = default;
    ~VulkanUiRenderer();

    VulkanUiRenderer(const VulkanUiRenderer&) = delete;
    VulkanUiRenderer& operator=(const VulkanUiRenderer&) = delete;

    // Sampler, descriptor set and pipeline layout. Call once
    bool init(const VulkanContext& ctx, std::uint32_t frame_count);

    // (Re)creates the pipeline for these attachment formats. Called at
    // start-up and whenever the swapchain format changes
    bool create_pipeline(VkFormat color_format, VkFormat depth_format);

    // Before recording the frame: uploads the font atlas when it changed
    // and copies the list into frame_index's buffers. The caller must
    // have waited for that frame slot's fence. Returns false when there
    // is nothing to draw
    bool prepare(const VulkanContext& ctx,
                 VulkanUploader& uploader,
                 std::uint32_t frame_index,
                 const UiDrawList& list);

    // Records the draw prepared for frame_index. Call inside the
    // frame's dynamic rendering, after the 3D passes
    void draw(VkCommandBuffer cmd,
              VkExtent2D extent,
              std::uint32_t frame_index) const;

    void shutdown();

private:
    struct FrameBuffers {
        std::unique_ptr<VulkanBuffer> vertices;
        std::unique_ptr<VulkanBuffer> indices;
        std::uint32_t index_count = 0;
    };

    bool upload_font(const VulkanContext& ctx,
                     VulkanUploader& uploader,
                     const FontAtlas& font);
    void destroy_font();
    void destroy_pipeline();

    VkDevice m_device = VK_NULL_HANDLE;

    VkSampler m_sampler = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;
    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    // Font atlas image. m_font is the atlas it holds; the app keeps one
    // atlas alive for the whole run, so comparing pointers is enough
    VkImage m_font_image = VK_NULL_HANDLE;
    VkDeviceMemory m_font_memory = VK_NULL_HANDLE;
    VkImageView m_font_view = VK_NULL_HANDLE;
    const FontAtlas* m_font = nullptr;

    std::vector<FrameBuffers> m_frames;
};

} // namespace gfx
