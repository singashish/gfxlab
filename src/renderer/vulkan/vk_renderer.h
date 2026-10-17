#pragma once

#include "renderer/renderer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_swapchain.h"
#include "renderer/vulkan/vk_frames.h"

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

    VulkanContext m_context;
    VulkanSwapchain m_swapchain;
    VulkanFrames m_frames;

    HWND m_native_window = nullptr;
    std::uint32_t m_width = 0;
    std::uint32_t m_height = 0;

    bool m_vsync = true;
};

} // namespace gfx
