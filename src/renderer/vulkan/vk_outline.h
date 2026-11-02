#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

class VulkanContext;

// The format of the outline's mask: 0 / 0.5 / 1 for unmarked / selected /
// active objects
constexpr VkFormat kOutlineMaskFormat = VK_FORMAT_R8_UNORM;

// The selection outline (see kOutlineRadius in renderer.h). Per frame the
// renderer
//   1. renders every object into the mask image (colour = mask value,
//      plus a depth image that holds the whole scene), using the mask
//      pipeline of VulkanMeshRenderer;
//   2. turns both images into shader-readable ones (barriers);
//   3. calls draw() inside the main rendering, after the grid: a
//      fullscreen triangle that colours the pixels just outside the marked
//      objects.
// This class owns the two images, the descriptor set that lets the
// fragment shader read them, and the pipeline of step 3. The images are
// shared by the frames in flight, like the main depth buffer; the barriers
// at the start of each frame order them against the previous frame's reads
class VulkanOutline {
public:
    VulkanOutline() = default;
    ~VulkanOutline();

    VulkanOutline(const VulkanOutline&) = delete;
    VulkanOutline& operator=(const VulkanOutline&) = delete;

    // Descriptor layout, pool and set (independent of the window size)
    bool init(VkDevice device);

    // (Re)creates the mask and depth images for a window size and points
    // the descriptor set at them. The device must be idle
    bool create_targets(const VulkanContext& ctx, VkExtent2D extent,
                        VkFormat depth_format);
    void destroy_targets();

    // (Re)creates the outline pipeline for the swapchain / depth formats
    bool create_pipeline(VkFormat color_format, VkFormat depth_format);

    void shutdown();

    // True when the images and the pipeline exist
    bool ready() const;

    VkImage mask_image() const { return m_mask_image; }
    VkImageView mask_view() const { return m_mask_view; }
    VkImage depth_image() const { return m_depth_image; }
    VkImageView depth_view() const { return m_depth_view; }

    // The fullscreen pass (step 3). near/far are the projection's clip
    // distances
    void draw(VkCommandBuffer cmd, VkExtent2D extent,
              float near_plane, float far_plane) const;

private:
    void destroy_pipeline();

    VkDevice m_device = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_set_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;
    VkDescriptorSet m_set = VK_NULL_HANDLE;

    VkPipelineLayout m_pipeline_layout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    VkImage m_mask_image = VK_NULL_HANDLE;
    VkDeviceMemory m_mask_memory = VK_NULL_HANDLE;
    VkImageView m_mask_view = VK_NULL_HANDLE;

    VkImage m_depth_image = VK_NULL_HANDLE;
    VkDeviceMemory m_depth_memory = VK_NULL_HANDLE;
    VkImageView m_depth_view = VK_NULL_HANDLE;
};

} // namespace gfx
