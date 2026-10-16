#pragma once

#include "renderer/renderer.h"
#include "renderer/vulkan/vk_context.h"

namespace gfx {

class VulkanRenderer final : public IRenderer {
public:
    VulkanRenderer() = default;
    ~VulkanRenderer() override = default;

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    const char* name() const override;
    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    void set_vsync(bool enabled) override;
    MeshId create_mesh(const Mesh& mesh) override;

private:
    VulkanContext m_context;
    bool m_vsync = true;
};

} // namespace gfx
