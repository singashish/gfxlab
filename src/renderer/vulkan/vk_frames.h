#pragma once

#include "renderer/vulkan/vk_api.h"

#include <array>
#include <cstdint>
#include <vector>

namespace gfx {

class VulkanFrames {
public:
    static constexpr std::uint32_t kFramesInFlight = 2;

    struct Slot {
        VkCommandBuffer cmd = VK_NULL_HANDLE;
        VkSemaphore image_available = VK_NULL_HANDLE;
        VkFence in_flight = VK_NULL_HANDLE;
    };

    VulkanFrames() = default;
    ~VulkanFrames();

    VulkanFrames(const VulkanFrames&) = delete;
    VulkanFrames& operator=(const VulkanFrames&) = delete;

    bool create(
        VkDevice device,
        std::uint32_t queue_family,
        std::uint32_t swapchain_image_count);

    void destroy();

    Slot& current() {
        return m_slots[m_current_slot];
    }

    void advance() {
        m_current_slot = (m_current_slot + 1) % kFramesInFlight;
    }

    VkSemaphore render_finished(std::uint32_t image_index) const {
        return image_index < m_render_finished.size()
             ? m_render_finished[image_index]
             : VK_NULL_HANDLE;
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_command_pool = VK_NULL_HANDLE;

    std::array<Slot, kFramesInFlight> m_slots{};
    std::vector<VkSemaphore> m_render_finished;

    std::uint32_t m_current_slot = 0;
};

} // namespace gfx
