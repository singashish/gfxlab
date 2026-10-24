#pragma once

#include "renderer/renderer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_depth.h"
#include "renderer/vulkan/vk_frames.h"
#include "renderer/vulkan/vk_grid.h"
#include "renderer/vulkan/vk_mesh.h"
#include "renderer/vulkan/vk_swapchain.h"
#include "renderer/vulkan/vk_timer.h"
#include "renderer/vulkan/vk_triangle_demo.h"
#include "renderer/vulkan/vk_ui.h"
#include "renderer/vulkan/vk_uniforms.h"
#include "renderer/vulkan/vk_upload.h"

namespace gfx {

class VulkanRenderer final : public IRenderer {
public:
    VulkanRenderer() = default;
    ~VulkanRenderer() override;

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    const char* name() const override;
    bool last_gpu_ms(double& ms) override;
    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    void set_vsync(bool enabled) override;
    MeshId create_mesh(const Mesh& mesh,
                       const std::vector<TextureId>& textures) override;

private:
    bool recreate_swapchain();
    bool init_pipelines(VkFormat color_format, VkFormat depth_format);

    // Order is important: dependent GPU resources are destroyed
    // before the context
    VulkanContext m_context;
    VulkanGpuTimer m_gpu_timer;
    VulkanSwapchain m_swapchain;
    VulkanDepthBuffer m_depth;
    VulkanFrames m_frames;
    VulkanUploader m_uploader;
    VulkanFrameUniforms m_uniforms;
    VulkanTriangleDemo m_triangle;
    VulkanMeshRenderer m_meshes;
    VulkanGridRenderer m_grid;
    VulkanUiRenderer m_ui;

    VkFormat m_color_format = VK_FORMAT_UNDEFINED;
    VkFormat m_depth_format = VK_FORMAT_UNDEFINED;

    HWND m_native_window = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;

    bool m_vsync = true;
};

} // namespace gfx
