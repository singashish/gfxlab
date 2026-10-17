#include "renderer/vulkan/vk_memory.h"

namespace gfx {

bool find_memory_type(VkPhysicalDevice physical_device, std::uint32_t type_bits,
                      VkMemoryPropertyFlags required, std::uint32_t& out_index) {

    if (physical_device == VK_NULL_HANDLE) {
        LOG_ERROR("cannot find memory type: null physical device");
        return false;
    }

    VkPhysicalDeviceMemoryProperties memory_properties{};

    vkGetPhysicalDeviceMemoryProperties(
        physical_device,
        &memory_properties);

    for (std::uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        const bool type_supported = (type_bits & (std::uint32_t{1} << i)) != 0;

        const bool properties_supported =
            (memory_properties.memoryTypes[i].propertyFlags
             & required) == required;

        if (type_supported && properties_supported) {
            out_index = i;
            return true;
        }
    }

    LOG_ERROR(
        "no compatible Vulkan memory type found "
        "(type_bits=0x%08x, required=0x%08x)",
        static_cast<unsigned int>(type_bits),
        static_cast<unsigned int>(required));

    return false;
}

} // namespace gfx
