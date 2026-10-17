#pragma once

#include <vulkan/vulkan.h>

namespace gfx {

VkShaderModule load_shader_module(VkDevice device, const char* path);

} // namespace gfx
