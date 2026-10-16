#pragma once

#include <vulkan/vulkan.h>

#include "core/log.h"

namespace gfx {

inline bool vk_ok(VkResult result, const char* operation) {
    if (result == VK_SUCCESS) {
        return true;
    }

    LOG_ERROR("%s failed with VkResult %d",
              operation, static_cast<int>(result));
    return false;
}

} // namespace gfx
