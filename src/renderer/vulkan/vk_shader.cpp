#include "renderer/vulkan/vk_shader.h"
#include "renderer/vulkan/vk_api.h"

#include <cstdint>
#include <fstream>
#include <limits>
#include <vector>

namespace gfx {

VkShaderModule load_shader_module(VkDevice device, const char* path) {
    if (device == VK_NULL_HANDLE || path == nullptr) {
        LOG_ERROR("Cannot load shader: invalid device or path");
        return VK_NULL_HANDLE;
    }

    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open shader file: %s", path);
        return VK_NULL_HANDLE;
    }

    const std::streampos end = file.tellg();
    if (end <= std::streampos(0)) {
        LOG_ERROR("Shader file is empty or unreadable: %s", path);
        return VK_NULL_HANDLE;
    }

    const auto byte_count = static_cast<std::uint64_t>(
        static_cast<std::streamoff>(end));

    if (byte_count % sizeof(std::uint32_t) != 0 ||
        byte_count > std::numeric_limits<std::size_t>::max()) {
        LOG_ERROR("Shader file has an invalid SPIR-V size: %s", path);
        return VK_NULL_HANDLE;
    }

    const auto word_count = static_cast<std::size_t>(
        byte_count / sizeof(std::uint32_t));

    std::vector<std::uint32_t> words(word_count);

    file.seekg(0, std::ios::beg);
    file.read(
        reinterpret_cast<char*>(words.data()),
        static_cast<std::streamsize>(byte_count));

    if (!file || static_cast<std::uint64_t>(file.gcount()) != byte_count) {
        LOG_ERROR("Failed to read shader file completely: %s", path);
        return VK_NULL_HANDLE;
    }

    VkShaderModuleCreateInfo create_info{
        VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO
    };
    create_info.codeSize = static_cast<std::size_t>(byte_count);
    create_info.pCode = words.data();

    VkShaderModule module = VK_NULL_HANDLE;
    if (!vk_ok(
            vkCreateShaderModule(
                device, &create_info, nullptr, &module),
            "vkCreateShaderModule")) {
        LOG_ERROR("Failed to create shader module for: %s", path);
        return VK_NULL_HANDLE;
    }

    return module;
}

} // namespace gfx
