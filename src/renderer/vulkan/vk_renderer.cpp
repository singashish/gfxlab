#include "renderer/vulkan/vk_renderer.h"

namespace gfx {

const char* VulkanRenderer::name() const {
    return "Vulkan 1.3";
}

bool VulkanRenderer::init(void* native_window) {
    if (native_window == nullptr) {
        LOG_ERROR("Vulkan renderer received a null window");
        return false;
    }

    m_native_window = static_cast<HWND>(native_window);

    if (!m_context.create(native_window)) {
        return false;
    }

    RECT rect{};

    if (!GetClientRect(m_native_window, &rect)) {
        LOG_ERROR("GetClientRect failed for Vulkan window");
        return false;
    }

    m_width = static_cast<std::uint32_t>(
        rect.right - rect.left);

    m_height = static_cast<std::uint32_t>(
        rect.bottom - rect.top);

    if (!m_swapchain.create(
            m_context, m_width, m_height, m_vsync)) {
        LOG_ERROR("failed to create Vulkan swapchain");
        return false;
    }

    return true;
}

void VulkanRenderer::render(const FrameData&) {
    // Step 3 creates the swapchain, but does not render yet
    // Command buffers and rendering begin in Step 4
}

void VulkanRenderer::set_vsync(bool enabled) {
    if (enabled == m_vsync) {
        return;
    }

    m_vsync = enabled;

    if (m_swapchain.swapchain() != VK_NULL_HANDLE) {
        if (!recreate_swapchain()) {
            LOG_ERROR("failed to recreate swapchain after vsync change");
        }
    }
}

bool VulkanRenderer::recreate_swapchain() {
    if (m_context.device() == VK_NULL_HANDLE) {
        LOG_ERROR("cannot recreate swapchain without a Vulkan device");
        return false;
    }

    LOG_INFO("recreating Vulkan swapchain");

    if (!vk_ok(
            vkDeviceWaitIdle(m_context.device()),
            "wait for device idle before swapchain recreation")) {
        return false;
    }

    // Release all resources belonging to the old swapchain
    m_swapchain.destroy();

    RECT rect{};

    if (m_native_window == nullptr ||
        !GetClientRect(m_native_window, &rect)) {
        LOG_ERROR("failed to query client size for swapchain recreation");
        return false;
    }

    m_width = static_cast<std::uint32_t>(
        rect.right - rect.left);

    m_height = static_cast<std::uint32_t>(
        rect.bottom - rect.top);

    if (!m_swapchain.create(
            m_context, m_width, m_height, m_vsync)) {
        LOG_ERROR("failed to create replacement Vulkan swapchain");
        return false;
    }

    LOG_INFO("Vulkan swapchain recreated successfully");
    return true;
}

MeshId VulkanRenderer::create_mesh(const Mesh&) {
    LOG_WARN("Vulkan mesh creation is not implemented yet");
    return 0;
}

} // namespace gfx
