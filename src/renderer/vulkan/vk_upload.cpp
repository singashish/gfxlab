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

    // Start unsignaled; each upload submits work before waiting on the fence
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

bool VulkanUploader::submit_and_wait(
    const VulkanContext& ctx,
    const std::function<void(VkCommandBuffer)>& record,
    const char* what) {

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
        return false;
    }

    record(cmd);

    if (!vk_ok(
            vkEndCommandBuffer(cmd),
            "end upload command buffer")) {
        free_command_buffer();
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
        return false;
    }

    const VkResult wait_result = vkWaitForFences(
        m_device, 1, &m_fence, VK_TRUE, UINT64_MAX);

    if (!vk_ok(wait_result, what)) {
        // Ensure the command buffer is no longer in use before freeing it
        (void)vkDeviceWaitIdle(m_device);
        free_command_buffer();
        return false;
    }

    free_command_buffer();

    // The upload completed successfully. Prepare the fence for the next upload
    return vk_ok(
        vkResetFences(m_device, 1, &m_fence),
        "reset upload fence after completion");
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

    const auto record = [&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.srcOffset = 0;
        copy.dstOffset = 0;
        copy.size = size;

        vkCmdCopyBuffer(
            cmd, staging.buffer(), dst.buffer(), 1, &copy);
    };

    if (!submit_and_wait(ctx, record, "wait for buffer upload")) {
        dst.destroy();
        return false;
    }

    return true;
}

bool VulkanUploader::upload_to_image(const VulkanContext& ctx,
                                     VkImage image,
                                     std::uint32_t width,
                                     std::uint32_t height,
                                     const void* data,
                                     VkDeviceSize size) {
    if (m_device == VK_NULL_HANDLE ||
        m_command_pool == VK_NULL_HANDLE ||
        m_fence == VK_NULL_HANDLE ||
        ctx.device() != m_device ||
        ctx.queue() == VK_NULL_HANDLE ||
        image == VK_NULL_HANDLE ||
        width == 0 ||
        height == 0 ||
        data == nullptr ||
        size == 0) {
        LOG_ERROR("invalid arguments for Vulkan image upload");
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

    const auto record = [&](VkCommandBuffer cmd) {
        VkImageMemoryBarrier2 barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        barrier.subresourceRange.baseMipLevel = 0;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.baseArrayLayer = 0;
        barrier.subresourceRange.layerCount = 1;

        VkDependencyInfo dependency{};
        dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        dependency.imageMemoryBarrierCount = 1;
        dependency.pImageMemoryBarriers = &barrier;

        // 1. UNDEFINED -> TRANSFER_DST before the copy writes the image
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        barrier.srcAccessMask = 0;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;

        vkCmdPipelineBarrier2(cmd, &dependency);

        // 2. Copy the tightly packed rows (bufferRowLength 0 = width)
        VkBufferImageCopy region{};
        region.bufferOffset = 0;
        region.bufferRowLength = 0;
        region.bufferImageHeight = 0;
        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        region.imageSubresource.mipLevel = 0;
        region.imageSubresource.baseArrayLayer = 0;
        region.imageSubresource.layerCount = 1;
        region.imageOffset = {0, 0, 0};
        region.imageExtent = {width, height, 1};

        vkCmdCopyBufferToImage(
            cmd,
            staging.buffer(),
            image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1,
            &region);

        // 3. TRANSFER_DST -> SHADER_READ_ONLY for sampling. The fence
        // wait below makes later frames see the finished image
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        vkCmdPipelineBarrier2(cmd, &dependency);
    };

    return submit_and_wait(ctx, record, "wait for image upload");
}

} // namespace gfx
