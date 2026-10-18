#pragma once

#include "core/mat4.h"
#include "core/vec4.h"
#include "renderer/vulkan/vk_api.h"
#include "renderer/vulkan/vk_buffer.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

class VulkanContext;

struct FrameUniforms {
    Mat4 view;
    Mat4 view_proj;

    Vec4 eye;
    Vec4 forward;
    Vec4 right;
    Vec4 up;
    Vec4 params;
};

static_assert(sizeof(FrameUniforms) == 208,
              "FrameUniforms must be 208 bytes");

static_assert(offsetof(FrameUniforms, eye) == 128,
              "FrameUniforms::eye must start at byte 128");

static_assert(offsetof(FrameUniforms, params) == 192,
              "FrameUniforms::params must start at byte 192");

class VulkanFrameUniforms {
public:
    VulkanFrameUniforms() = default;
    ~VulkanFrameUniforms();

    VulkanFrameUniforms(const VulkanFrameUniforms&) = delete;
    VulkanFrameUniforms& operator=(const VulkanFrameUniforms&) = delete;

    bool init(const VulkanContext& ctx, std::uint32_t frame_count);
    void shutdown();

    bool update(std::uint32_t frame_index, const FrameUniforms& uniforms);

    VkDescriptorSetLayout layout() const {
        return m_layout;
    }

    VkDescriptorSet set(std::uint32_t frame_index) const;

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_layout = VK_NULL_HANDLE;
    VkDescriptorPool m_pool = VK_NULL_HANDLE;

    std::vector<std::unique_ptr<VulkanBuffer>> m_buffers;
    std::vector<VkDescriptorSet> m_sets;
};

} // namespace gfx
