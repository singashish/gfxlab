#include "renderer/vulkan/vk_renderer.h"

namespace gfx {
namespace {

void transition_image(
    VkCommandBuffer cmd,
    VkImage image,
    VkImageAspectFlags aspect,
    VkImageLayout old_layout,
    VkImageLayout new_layout,
    VkPipelineStageFlags2 src_stage,
    VkAccessFlags2 src_access,
    VkPipelineStageFlags2 dst_stage,
    VkAccessFlags2 dst_access) {

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_stage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_stage;
    barrier.dstAccessMask = dst_access;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;

    barrier.subresourceRange.aspectMask = aspect;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependency{};
    dependency.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependency.imageMemoryBarrierCount = 1;
    dependency.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &dependency);
}

} // namespace

VulkanRenderer::~VulkanRenderer() {
    if (m_context.device() != VK_NULL_HANDLE) {
        (void)vk_ok(
            vkDeviceWaitIdle(m_context.device()),
            "wait for device idle before renderer destruction");
    }

    // Members are destroyed after this body, in reverse order:
    // frames -> swapchain -> context
}

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

    m_width = static_cast<std::uint32_t>(rect.right - rect.left);
    m_height = static_cast<std::uint32_t>(rect.bottom - rect.top);

    if (!m_swapchain.create(m_context, m_width, m_height, m_vsync)) {
        LOG_ERROR("failed to create Vulkan swapchain");
        return false;
    }

    if (!m_depth.create(m_context, m_swapchain.extent())) {
        LOG_ERROR("failed to create Vulkan depth buffer");
        return false;
    }

    m_color_format = m_swapchain.format();
    m_depth_format = m_depth.format();

    if (!m_frames.create(m_context.device(),
                         m_context.queue_family(),
                         m_swapchain.image_count())) {
        LOG_ERROR("failed to create Vulkan frame resources");
        return false;
    }

    if (!m_uploader.init(m_context)) {
        LOG_ERROR("failed to initialize Vulkan uploader");
        return false;
    }

    if (!m_uniforms.init(m_context, VulkanFrames::kFramesInFlight)) {
        LOG_ERROR("failed to initialize Vulkan frame uniforms");
        return false;
    }

    if (!m_triangle.init(m_context.device(),
                         m_color_format,
                         m_depth_format)) {
        LOG_ERROR("failed to initialize Vulkan triangle demo pipeline");
        return false;
    }

    if (!m_meshes.init(m_context.device(),
                       m_color_format,
                       m_depth_format,
                       m_uniforms.layout())) {
        LOG_ERROR("failed to initialize Vulkan mesh renderer");
        return false;
    }

    return true;
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
    if (m_context.device() == VK_NULL_HANDLE ||
        m_native_window == nullptr) {
        LOG_ERROR("cannot recreate swapchain without a device and window");
        return false;
    }

    // Query the client rectangle before destroying working resources
    RECT rect{};

    if (!GetClientRect(m_native_window, &rect)) {
        LOG_ERROR("GetClientRect failed during swapchain recreation");
        return false;
    }

    const std::uint32_t width =
        static_cast<std::uint32_t>(rect.right - rect.left);
    const std::uint32_t height =
        static_cast<std::uint32_t>(rect.bottom - rect.top);

    // A minimized window has no drawable area. Defer recreation until
    // the window has a non-zero extent
    if (width == 0 || height == 0) {
        LOG_INFO("swapchain recreation deferred: window has zero extent");
        return false;
    }

    LOG_INFO("recreating Vulkan swapchain");

    if (!vk_ok(
            vkDeviceWaitIdle(m_context.device()),
            "wait for device idle before swapchain recreation")) {
        return false;
    }

    // Destroy resources associated with the old swapchain
    m_frames.destroy();
    m_depth.destroy();
    m_swapchain.destroy();

    m_width = width;
    m_height = height;

    // Recreate the swapchain
    if (!m_swapchain.create(m_context, m_width, m_height, m_vsync)) {
        LOG_ERROR("failed to recreate Vulkan swapchain");
        return false;
    }

    // Recreate the depth buffer using the new swapchain extent
    if (!m_depth.create(m_context, m_swapchain.extent())) {
        LOG_ERROR("failed to recreate Vulkan depth buffer");
        m_swapchain.destroy();
        return false;
    }

    // Frame synchronization resources depend on the image count
    if (!m_frames.create(m_context.device(),
                         m_context.queue_family(),
                         m_swapchain.image_count())) {
        LOG_ERROR("failed to recreate Vulkan frame resources");
        m_frames.destroy();
        m_depth.destroy();
        m_swapchain.destroy();
        return false;
    }

    // Attachment formats determine graphics pipeline compatibility
    // A resize alone does not require rebuilding either pipeline
    const VkFormat new_color_format = m_swapchain.format();
    const VkFormat new_depth_format = m_depth.format();

    if (new_color_format != m_color_format ||
        new_depth_format != m_depth_format) {

        if (!m_triangle.init(m_context.device(),
                             new_color_format,
                             new_depth_format)) {
            LOG_ERROR("failed to recreate Vulkan triangle pipeline");

            m_frames.destroy();
            m_depth.destroy();
            m_swapchain.destroy();
            return false;
        }

        if (!m_meshes.init(m_context.device(),
                           new_color_format,
                           new_depth_format,
                           m_uniforms.layout())) {
            LOG_ERROR("failed to recreate Vulkan mesh pipeline");

            m_frames.destroy();
            m_depth.destroy();
            m_swapchain.destroy();
            return false;
        }

        // Update only after both pipelines have been recreated successfully
        m_color_format = new_color_format;
        m_depth_format = new_depth_format;
    }

    LOG_INFO("Vulkan swapchain recreated successfully");
    return true;
}


void VulkanRenderer::render(const FrameData& frame) {
    // Minimal handling for a missing swapchain
    if (m_swapchain.swapchain() == VK_NULL_HANDLE) {
        if (frame.width > 0 && frame.height > 0) {
            (void)recreate_swapchain();
        }

        if (m_swapchain.swapchain() == VK_NULL_HANDLE) {
            return;
        }
    }

    const std::uint32_t frame_index = m_frames.current_index();
    VulkanFrames::Slot& slot = m_frames.current();

    // 1. Wait for this frame slot's previous GPU work
    if (!vk_ok(
            vkWaitForFences(
                m_context.device(), 1, &slot.in_flight,
                VK_TRUE, UINT64_MAX),
            "wait for frame fence")) {
        return;
    }

    const VkExtent2D extent = m_swapchain.extent();

    if (extent.width == 0 || extent.height == 0) {
        return;
    }

    const float aspect =
        static_cast<float>(extent.width) /
        static_cast<float>(extent.height);

    const Mat4 projection = perspective(
        frame.fov_y,
        aspect,
        frame.near_plane,
        frame.far_plane,
        ClipSpace::CLIP_SPACE_VULKAN);

    const Mat4 view_proj = projection * frame.view;

    if (!m_uniforms.update(frame_index,
                        FrameUniforms{frame.view, view_proj})) {
        LOG_ERROR("failed to update Vulkan frame uniforms");
        return;
    }

    // 2. Acquire a swapchain image
    std::uint32_t image_index = 0;

    const VkResult acquire_result = vkAcquireNextImageKHR(
        m_context.device(),
        m_swapchain.swapchain(),
        UINT64_MAX,
        slot.image_available,
        VK_NULL_HANDLE,
        &image_index);

    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR) {
        (void)recreate_swapchain();
        return;
    }

    if (acquire_result != VK_SUCCESS &&
        acquire_result != VK_SUBOPTIMAL_KHR) {
        (void)vk_ok(acquire_result, "acquire swapchain image");
        return;
    }

    if (image_index >= m_swapchain.image_count()) {
        LOG_ERROR("acquired swapchain image index is out of range");
        return;
    }

    const VkImage image = m_swapchain.image(image_index);
    const VkImageView view = m_swapchain.image_view(image_index);
    const VkSemaphore finished = m_frames.render_finished(image_index);

    if (image == VK_NULL_HANDLE ||
        view == VK_NULL_HANDLE ||
        finished == VK_NULL_HANDLE) {
        LOG_ERROR(
            "Invalid swapchain image, image view, or "
            "render-finished semaphore");
        return;
    }

    // Reset the fence only after the non-recording checks pass
    if (!vk_ok(
            vkResetFences(
                m_context.device(), 1, &slot.in_flight),
            "reset frame fence")) {
        return;
    }

    // 3. Reset and begin recording
    if (!vk_ok(
            vkResetCommandBuffer(slot.cmd, 0),
            "reset frame command buffer")) {
        return;
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (!vk_ok(
            vkBeginCommandBuffer(slot.cmd, &begin_info),
            "begin frame command buffer")) {
        return;
    }

    // 4. Transition the swapchain image into a layout suitable for
    // rendering. The destination stage matches the stage the submit
    // waits on for image_available, so the layout change happens only
    // after the image has been acquired.
    transition_image(
        slot.cmd,
        image,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        0,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);

    // 5. Transition the shared depth image. The depth image is shared
    // by frames in flight. These stages ensure earlier depth writes,
    // including late fragment tests, are ordered before this frame
    // reads or writes depth.
    transition_image(
        slot.cmd,
        m_depth.image(),
        VK_IMAGE_ASPECT_DEPTH_BIT,
        VK_IMAGE_LAYOUT_UNDEFINED,
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
            VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT);

    // 6. Configure the color attachment
    VkRenderingAttachmentInfo color_attachment{};
    color_attachment.sType =
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

    color_attachment.imageView = view;
    color_attachment.imageLayout =
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    color_attachment.clearValue.color.float32[0] = 0.10f;
    color_attachment.clearValue.color.float32[1] = 0.11f;
    color_attachment.clearValue.color.float32[2] = 0.13f;
    color_attachment.clearValue.color.float32[3] = 1.0f;

    // 7. Configure the depth attachment
    VkRenderingAttachmentInfo depth_attachment{};
    depth_attachment.sType =
        VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;

    depth_attachment.imageView = m_depth.view();
    depth_attachment.imageLayout =
        VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;

    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;

    depth_attachment.clearValue.depthStencil = {1.0f, 0};

    // 8. Begin dynamic rendering with both attachments
    VkRenderingInfo rendering_info{};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;

    rendering_info.renderArea.offset = {0, 0};
    rendering_info.renderArea.extent = m_swapchain.extent();

    rendering_info.layerCount = 1;

    rendering_info.colorAttachmentCount = 1;
    rendering_info.pColorAttachments = &color_attachment;
    rendering_info.pDepthAttachment = &depth_attachment;

    vkCmdBeginRendering(slot.cmd, &rendering_info);

    if (frame.draw_list.empty()) {
        // Preserve the Step 6 depth-test demonstration when no mesh is loaded
        m_triangle.record(slot.cmd, extent);
    } else {
        m_meshes.draw(
            slot.cmd,
            extent,
            m_uniforms.set(frame_index),
            frame.draw_list);
    }

    vkCmdEndRendering(slot.cmd);

    // 9. Transition the color image for presentation
    transition_image(
        slot.cmd,
        image,
        VK_IMAGE_ASPECT_COLOR_BIT,
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
        VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        VK_PIPELINE_STAGE_2_NONE,
        0);

    if (!vk_ok(
            vkEndCommandBuffer(slot.cmd),
            "end frame command buffer")) {
        return;
    }

    // 10. Submit, waiting for image acquisition
    VkSemaphoreSubmitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    wait_info.semaphore = slot.image_available;
    wait_info.stageMask =
        VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    wait_info.deviceIndex = 0;

    VkCommandBufferSubmitInfo command_info{};
    command_info.sType =
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO;
    command_info.commandBuffer = slot.cmd;
    command_info.deviceMask = 0;

    VkSemaphoreSubmitInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO;
    signal_info.semaphore = finished;
    signal_info.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    signal_info.deviceIndex = 0;

    VkSubmitInfo2 submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2;

    submit_info.waitSemaphoreInfoCount = 1;
    submit_info.pWaitSemaphoreInfos = &wait_info;

    submit_info.commandBufferInfoCount = 1;
    submit_info.pCommandBufferInfos = &command_info;

    submit_info.signalSemaphoreInfoCount = 1;
    submit_info.pSignalSemaphoreInfos = &signal_info;

    if (!vk_ok(
            vkQueueSubmit2(
                m_context.queue(),
                1,
                &submit_info,
                slot.in_flight),
            "submit frame")) {
        return;
    }

    // 11. Present the image
    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = &finished;

    const VkSwapchainKHR swapchain = m_swapchain.swapchain();

    present_info.swapchainCount = 1;
    present_info.pSwapchains = &swapchain;
    present_info.pImageIndices = &image_index;

    const VkResult present_result =
        vkQueuePresentKHR(m_context.queue(), &present_info);

    // The submitted frame has consumed its frame slot
    m_frames.advance();

    if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
        present_result == VK_SUBOPTIMAL_KHR) {
        (void)recreate_swapchain();
        return;
    }

    if (present_result != VK_SUCCESS) {
        (void)vk_ok(present_result, "present swapchain image");
    }
}

MeshId VulkanRenderer::create_mesh(const Mesh& mesh) {
    return m_meshes.create(m_context, m_uploader, mesh);
}

} // namespace gfx
