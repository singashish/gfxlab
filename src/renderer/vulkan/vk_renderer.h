#pragma once

#include "renderer/renderer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_depth.h"
#include "renderer/vulkan/vk_frames.h"
#include "renderer/vulkan/vk_swapchain.h"
#include "renderer/vulkan/vk_triangle_demo.h"

namespace gfx {

class VulkanRenderer final : public IRenderer {
public:
    VulkanRenderer() = default;
    ~VulkanRenderer() override;

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    const char* name() const override;
    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    void set_vsync(bool enabled) override;
    MeshId create_mesh(const Mesh& mesh) override;

private:
    bool recreate_swapchain();

    // Order is important
    VulkanContext m_context;
    VulkanSwapchain m_swapchain;
    VulkanDepthBuffer m_depth;
    VulkanFrames m_frames;
    VulkanTriangleDemo m_triangle;

    VkFormat m_color_format = VK_FORMAT_UNDEFINED;
    VkFormat m_depth_format = VK_FORMAT_UNDEFINED;

    HWND m_native_window = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;

    bool m_vsync = true;
};

} // namespace gfx
