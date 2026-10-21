#include "renderer/vulkan/vk_timer.h"

#include <vector>

namespace gfx {

VulkanGpuTimer::~VulkanGpuTimer() {
    if (m_pool != VK_NULL_HANDLE &&
        m_device != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_pool, nullptr);
    }
}

bool VulkanGpuTimer::init(
                VkPhysicalDevice physical_device,
                VkDevice device,
                std::uint32_t queue_family) {

    if (m_enabled) {
        return true;
    }

    if (physical_device == VK_NULL_HANDLE ||
        device == VK_NULL_HANDLE) {
        LOG_WARN("Vulkan GPU timing unavailable: invalid device");
        return false;
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(
        physical_device, &properties);

    m_timestamp_period = properties.limits.timestampPeriod;

    std::uint32_t family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, nullptr);

    std::vector<VkQueueFamilyProperties> families(family_count);

    vkGetPhysicalDeviceQueueFamilyProperties(
        physical_device, &family_count, families.data());

    if (queue_family >= family_count) {
        LOG_WARN(
            "Vulkan GPU timing unavailable: invalid queue family");
        return false;
    }

    m_valid_bits = families[queue_family].timestampValidBits;

    if (m_valid_bits == 0 || m_timestamp_period <= 0.0f) {
        LOG_WARN(
            "Vulkan GPU timestamps unsupported on graphics queue");
        return false;
    }

    m_mask = m_valid_bits >= 64
        ? ~std::uint64_t{0}
        : ((std::uint64_t{1} << m_valid_bits) - 1);

    VkQueryPoolCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    info.queryType = VK_QUERY_TYPE_TIMESTAMP;
    info.queryCount = kSlots * 2;

    const VkResult result = vkCreateQueryPool(
        device, &info, nullptr, &m_pool);

    if (result != VK_SUCCESS) {
        LOG_WARN(
            "Vulkan GPU timing unavailable: "
            "vkCreateQueryPool failed (%d)",
            static_cast<int>(result));
        m_pool = VK_NULL_HANDLE;
        return false;
    }

    m_device = device;
    m_written.fill(false);
    m_enabled = true;

    LOG_INFO(
        "Vulkan GPU timestamps enabled: "
        "period %.3f ns/tick, valid bits %u",
        m_timestamp_period, m_valid_bits);

    return true;
}

void VulkanGpuTimer::begin(
                VkCommandBuffer cmd,
                std::uint32_t slot) {

    if (!m_enabled || cmd == VK_NULL_HANDLE ||
        slot >= kSlots) {
        return;
    }

    const std::uint32_t first = slot * 2;

    // Reset this slot's previous queries before using them
    // This command must be outside a rendering instance
    vkCmdResetQueryPool(cmd, m_pool, first, 2);
    m_written[slot] = false;

    vkCmdWriteTimestamp2(
        cmd,
        VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT,
        m_pool,
        first);
}

void VulkanGpuTimer::end(
                VkCommandBuffer cmd,
                std::uint32_t slot) {

    if (!m_enabled || cmd == VK_NULL_HANDLE ||
        slot >= kSlots) {
        return;
    }

    vkCmdWriteTimestamp2(
        cmd,
        VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT,
        m_pool,
        slot * 2 + 1);

    m_written[slot] = true;
}

void VulkanGpuTimer::collect(std::uint32_t slot) {
    if (!m_enabled || slot >= kSlots ||
        !m_written[slot]) {
        return;
    }

    std::uint64_t timestamps[2]{};

    const VkResult result = vkGetQueryPoolResults(
        m_device,
        m_pool,
        slot * 2,
        2,
        sizeof(timestamps),
        timestamps,
        sizeof(std::uint64_t),
        VK_QUERY_RESULT_64_BIT);

    // No WAIT_BIT: a result must never stall the CPU
    // The frame fence should already guarantee completion
    if (result == VK_NOT_READY) {
        return;
    }

    if (result != VK_SUCCESS) {
        LOG_WARN(
            "Vulkan GPU timestamp read failed (%d)",
            static_cast<int>(result));
        return;
    }

    const std::uint64_t start = timestamps[0] & m_mask;
    const std::uint64_t end = timestamps[1] & m_mask;

    // Unsigned subtraction plus masking handles timestamp wraparound
    const std::uint64_t ticks = (end - start) & m_mask;

    m_latest_ms =
        static_cast<double>(ticks) *
        static_cast<double>(m_timestamp_period) * 1.0e-6;

    m_has_latest = true;
    m_written[slot] = false;
}

bool VulkanGpuTimer::take_latest(double& ms) {
    if (!m_has_latest) {
        return false;
    }

    ms = m_latest_ms;
    m_has_latest = false;
    return true;
}

} // namespace gfx
