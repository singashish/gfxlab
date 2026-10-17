#pragma once

#include "renderer/vulkan/vk_api.h"

namespace gfx {

VkShaderModule load_shader_module(VkDevice device, const char* path);

} // namespace gfx
