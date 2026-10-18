#include "renderer/vulkan/vk_upload.h"
#include "renderer/vulkan/vk_buffer.h"
#include "renderer/vulkan/vk_context.h"

namespace gfx {

VulkanUploader::~VulkanUploader() {
    shutdown();
}

bool VulkanUploader::init(const VulkanContext& ctx) {
    shutdown();

    if (ctx.device() == VK_NULL_HANDLE) {
        LOG_ERROR("cannot initialize uploader without a device");
        return false;
    }

    m_device = ctx.device();

    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    pool_info.queueFamilyIndex = ctx.queue_family();

    if (!vk_ok(
            vkCreateCommandPool(
                m_device,
                &pool_info,
                nullptr,
                &m_command_pool),
            "create upload command pool")) {
        shutdown();
        return false;
    }

    // Start signaled so the first upload can wait on the fence
    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    if (!vk_ok(
            vkCreateFence(
                m_device, &fence_info, nullptr, &m_fence),
            "create upload fence")) {
        shutdown();
        return false;
    }

    return true;
}

void VulkanUploader::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        // This uploader is used only during loading, between frames
        if (m_command_pool != VK_NULL_HANDLE ||
            m_fence != VK_NULL_HANDLE) {
            (void)vkDeviceWaitIdle(m_device);
        }

        if (m_fence != VK_NULL_HANDLE) {
            vkDestroyFence(m_device, m_fence, nullptr);
        }

        if (m_command_pool != VK_NULL_HANDLE) {
            vkDestroyCommandPool(
                m_device, m_command_pool, nullptr);
        }
    }

    m_fence = VK_NULL_HANDLE;
    m_command_pool = VK_NULL_HANDLE;
    m_device = VK_NULL_HANDLE;
}

bool VulkanUploader::upload_to_device_buffer(const VulkanContext& ctx,
                        VulkanBuffer& dst, VkBufferUsageFlags usage,
                        const void* data, VkDeviceSize size) {

    if (m_device == VK_NULL_HANDLE ||
        m_command_pool == VK_NULL_HANDLE ||
        m_fence == VK_NULL_HANDLE ||
        ctx.device() != m_device ||
        ctx.queue() == VK_NULL_HANDLE ||
        data == nullptr ||
        size == 0) {
        LOG_ERROR("invalid arguments for Vulkan buffer upload");
        return false;
    }

    VulkanBuffer staging;

    if (!staging.create(
            ctx,
            size,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        return false;
    }

    if (!staging.write(data, size)) {
        return false;
    }

    if (!dst.create(
            ctx,
            size,
            usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
        return false;
    }

    VkCommandBufferAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocate_info.commandPool = m_command_pool;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;

    if (!vk_ok(
            vkAllocateCommandBuffers(
                m_device, &allocate_info, &cmd),
            "allocate upload command buffer")) {
        dst.destroy();
        return false;
    }

    const auto free_command_buffer = [&]() {
        vkFreeCommandBuffers(
            m_device, m_command_pool, 1, &cmd);
        cmd = VK_NULL_HANDLE;
    };

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (!vk_ok(
            vkBeginCommandBuffer(cmd, &begin_info),
            "begin upload command buffer")) {
        free_command_buffer();
        dst.destroy();
        return false;
    }

    VkBufferCopy copy{};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size = size;

    vkCmdCopyBuffer(
        cmd, staging.buffer(), dst.buffer(), 1, &copy);

    if (!vk_ok(
            vkEndCommandBuffer(cmd),
            "end upload command buffer")) {
        free_command_buffer();
        dst.destroy();
        return false;
    }

    VkCommandBufferSubmitInfo command_info{};
    command_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    command_info.commandBuffer = cmd;
    command_info.deviceMask = 0;

    VkSubmitInfo2 submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;
    submit_info.commandBufferInfoCount = 1;
    submit_info.pCommandBufferInfos = &command_info;

    const VkResult submit_result = vkQueueSubmit2(
        ctx.queue(), 1, &submit_info, m_fence);

    if (!vk_ok(submit_result, "submit upload command buffer")) {
        free_command_buffer();
        dst.destroy();
        return false;
    }

    const VkResult wait_result = vkWaitForFences(
        m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    if (!vk_ok(wait_result, "wait for buffer upload")) {
        // Ensure the command buffer is no longer in use before freeing it
        (void)vkDeviceWaitIdle(m_device);
        free_command_buffer();
        dst.destroy();
        return false;
    }

    free_command_buffer();

    // The upload completed successfully. Prepare the fence for the next upload
    if (!vk_ok(
            vkResetFences(m_device, 1, &m_fence),
            "reset upload fence after completion")) {
        dst.destroy();
        return false;
    }

    return true;
}

} // namespace gfx
