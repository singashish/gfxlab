#pragma once

#include "renderer/vulkan/vk_api.h"

#include <cstdint>

namespace gfx {

bool find_memory_type(VkPhysicalDevice physical_device, std::uint32_t type_bits,
                      VkMemoryPropertyFlags required, std::uint32_t& out_index);

} // namespace gfx
