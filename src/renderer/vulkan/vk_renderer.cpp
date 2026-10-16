#include "renderer/vulkan/vk_renderer.h"

namespace gfx {

const char* VulkanRenderer::name() const {
    return "Vulkan 1.4";
}

bool VulkanRenderer::init(void* native_window) {
    return m_context.create(native_window);
}

void VulkanRenderer::render(const FrameData&) {
    // Rendering starts in a later milestone.
    // Step 2 has no swapchain or command buffers.
}

void VulkanRenderer::set_vsync(bool enabled) {
    m_vsync = enabled;
}

MeshId VulkanRenderer::create_mesh(const Mesh&) {
    LOG_WARN("Vulkan mesh creation is not implemented yet");
    return 0;
}

} // namespace gfx
