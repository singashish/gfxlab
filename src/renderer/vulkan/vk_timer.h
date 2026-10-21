#pragma once

#include "renderer/vulkan/vk_api.h"

#include <array>
#include <cstdint>

namespace gfx {

class VulkanGpuTimer {
public:
    static constexpr std::uint32_t kSlots = 2;

    VulkanGpuTimer() = default;
    ~VulkanGpuTimer();

    VulkanGpuTimer(const VulkanGpuTimer&) = delete;
    VulkanGpuTimer& operator=(const VulkanGpuTimer&) = delete;

    // Unsupported timestamp timing disables this feature only
    bool init(
            VkPhysicalDevice physical_device,
            VkDevice device,
            std::uint32_t queue_family);

    // Called while recording the slot's command buffer,
    // before vkCmdBeginRendering
    void begin(VkCommandBuffer cmd, std::uint32_t slot);

    // Called after vkCmdEndRendering.
    void end(VkCommandBuffer cmd, std::uint32_t slot);

    // Called only after this slot's in-flight fence has signaled
    void collect(std::uint32_t slot);

    bool take_latest(double& ms);

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueryPool m_pool = VK_NULL_HANDLE;

    float m_timestamp_period = 0.0f;
    std::uint32_t m_valid_bits = 0;
    std::uint64_t m_mask = 0;

    std::array<bool, kSlots> m_written{};

    bool m_enabled = false;
    bool m_has_latest = false;
    double m_latest_ms = 0.0;
};

} // namespace gfx
