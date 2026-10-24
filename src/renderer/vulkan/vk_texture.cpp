#include "renderer/vulkan/vk_texture.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_memory.h"
#include "renderer/vulkan/vk_upload.h"

#include <algorithm>
#include <limits>

namespace gfx {

namespace {

// Same wish as the OpenGL backend: 8x is plenty, the hardware limit
// may be lower
constexpr float kWantedAnisotropy = 8.0f;

// How many descriptor sets one pool holds
constexpr std::uint32_t kSetsPerPool = 64;

// Texture images are plain RGBA8, not sRGB: the viewport shading works
// in display values on purpose (see gl_texture.cpp), so the bytes are
// used as they are
constexpr VkFormat kTextureFormat = VK_FORMAT_R8G8B8A8_UNORM;

// Number of mip levels of a full chain: halve the larger side until 1,
// so 1 + floor(log2(max(width, height)))
std::uint32_t full_mip_count(std::uint32_t width, std::uint32_t height) {
    std::uint32_t levels = 1;

    for (std::uint32_t side = std::max(width, height); side > 1;
         side >>= 1) {
        ++levels;
    }

    return levels;
}

} // namespace

VulkanTextureStore::~VulkanTextureStore() {
    shutdown();
}

bool VulkanTextureStore::init(const VulkanContext& ctx,
                              VulkanUploader& uploader) {
    shutdown();

    if (ctx.device() == VK_NULL_HANDLE ||
        ctx.physical_device() == VK_NULL_HANDLE) {
        LOG_ERROR("invalid arguments for the Vulkan texture store");
        return false;
    }

    m_device = ctx.device();

    // Mipmaps are generated on the GPU by blitting (scaling) each level
    // into the next with a linear filter, so the format has to support
    // being a blit source and destination and linear filtering. Every
    // desktop GPU does this for RGBA8; if one did not we would fall back
    // to a single level instead of failing
    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(
        ctx.physical_device(), kTextureFormat, &format_properties);

    constexpr VkFormatFeatureFlags kSampledFeatures =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    constexpr VkFormatFeatureFlags kLinearFeature =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    constexpr VkFormatFeatureFlags kBlitFeatures =
        VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;

    const VkFormatFeatureFlags features =
        format_properties.optimalTilingFeatures;

    if ((features & kSampledFeatures) != kSampledFeatures) {
        LOG_ERROR("the GPU cannot sample RGBA8 textures");
        shutdown();
        return false;
    }

    const bool linear_filter = (features & kLinearFeature) != 0;

    m_mipmaps_supported =
        linear_filter && (features & kBlitFeatures) == kBlitFeatures;

    if (!m_mipmaps_supported) {
        LOG_WARN("the GPU cannot blit RGBA8 images with linear "
                 "filtering; textures get no mipmaps");
    }

    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(ctx.physical_device(), &properties);
    m_max_dimension = properties.limits.maxImageDimension2D;

    // One sampler for every texture: trilinear filtering (linear inside
    // a level and between two levels), repeat outside 0..1 (glTF's
    // default), and anisotropic filtering where the GPU has it, which
    // keeps textures sharp on surfaces seen at a grazing angle
    const VkFilter filter =
        linear_filter ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    sampler_info.magFilter = filter;
    sampler_info.minFilter = filter;
    sampler_info.mipmapMode = linear_filter
        ? VK_SAMPLER_MIPMAP_MODE_LINEAR
        : VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.mipLodBias = 0.0f;

    // Anisotropy may only be requested when the feature was enabled at
    // device creation, and not at all for a nearest filter
    const bool anisotropy = ctx.sampler_anisotropy() && linear_filter;
    sampler_info.anisotropyEnable = anisotropy ? VK_TRUE : VK_FALSE;
    sampler_info.maxAnisotropy = anisotropy
        ? std::max(1.0f,
                   std::min(ctx.max_sampler_anisotropy(),
                            kWantedAnisotropy))
        : 1.0f;

    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = VK_LOD_CLAMP_NONE;
    sampler_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    if (!vk_ok(
            vkCreateSampler(m_device, &sampler_info, nullptr, &m_sampler),
            "create texture sampler")) {
        shutdown();
        return false;
    }

    // Binding 0 of set 1: the base-colour texture, read by the fragment
    // shader
    VkDescriptorSetLayoutBinding binding{};
    binding.binding = 0;
    binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    binding.descriptorCount = 1;
    binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layout_info.bindingCount = 1;
    layout_info.pBindings = &binding;

    if (!vk_ok(
            vkCreateDescriptorSetLayout(
                m_device, &layout_info, nullptr, &m_set_layout),
            "create texture descriptor set layout")) {
        shutdown();
        return false;
    }

    // base colour * white = base colour, so a surface without a texture
    // needs no special case in the shader
    const std::uint8_t white_pixel[4] = {255, 255, 255, 255};

    if (!create_gpu_texture(ctx, uploader, white_pixel, 1, 1, m_white)) {
        LOG_ERROR("failed to create the white fallback texture");
        shutdown();
        return false;
    }

    return true;
}

bool VulkanTextureStore::add_pool() {
    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_size.descriptorCount = kSetsPerPool;

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = kSetsPerPool;
    pool_info.poolSizeCount = 1;
    pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool pool = VK_NULL_HANDLE;

    if (!vk_ok(
            vkCreateDescriptorPool(m_device, &pool_info, nullptr, &pool),
            "create texture descriptor pool")) {
        return false;
    }

    m_pools.push_back(pool);
    m_sets_in_newest_pool = 0;
    return true;
}

bool VulkanTextureStore::allocate_set(VkDescriptorSet& out) {
    if (m_pools.empty() || m_sets_in_newest_pool >= kSetsPerPool) {
        if (!add_pool()) {
            return false;
        }
    }

    VkDescriptorSetAllocateInfo allocate_info{};
    allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate_info.descriptorPool = m_pools.back();
    allocate_info.descriptorSetCount = 1;
    allocate_info.pSetLayouts = &m_set_layout;

    if (!vk_ok(
            vkAllocateDescriptorSets(m_device, &allocate_info, &out),
            "allocate texture descriptor set")) {
        out = VK_NULL_HANDLE;
        return false;
    }

    ++m_sets_in_newest_pool;
    return true;
}

bool VulkanTextureStore::create_gpu_texture(const VulkanContext& ctx,
                                            VulkanUploader& uploader,
                                            const std::uint8_t* pixels,
                                            std::uint32_t width,
                                            std::uint32_t height,
                                            GpuTexture& out) {
    out = GpuTexture{};

    if (m_device == VK_NULL_HANDLE ||
        m_sampler == VK_NULL_HANDLE ||
        m_set_layout == VK_NULL_HANDLE ||
        pixels == nullptr ||
        width == 0 ||
        height == 0 ||
        width > m_max_dimension ||
        height > m_max_dimension) {
        LOG_ERROR("invalid %ux%u texture for the GPU (limit %u)",
                  width, height, m_max_dimension);
        return false;
    }

    const std::uint32_t mip_levels =
        m_mipmaps_supported ? full_mip_count(width, height) : 1;

    // 1. The image. TRANSFER_DST receives the pixels, TRANSFER_SRC lets
    // each mip level be blitted into the next, SAMPLED is for the shader
    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = kTextureFormat;
    image_info.extent = {width, height, 1};
    image_info.mipLevels = mip_levels;
    image_info.arrayLayers = 1;
    image_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_info.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                       VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                       VK_IMAGE_USAGE_SAMPLED_BIT;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    GpuTexture texture;

    if (!vk_ok(
            vkCreateImage(m_device, &image_info, nullptr, &texture.image),
            "create texture image")) {
        destroy_gpu_texture(texture);
        return false;
    }

    // 2. Device-local memory for it
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(m_device, texture.image, &requirements);

    std::uint32_t memory_type_index = 0;

    if (!find_memory_type(ctx.physical_device(),
                          requirements.memoryTypeBits,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                          memory_type_index)) {
        destroy_gpu_texture(texture);
        return false;
    }

    VkMemoryAllocateInfo allocation_info{};
    allocation_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;

    if (!vk_ok(
            vkAllocateMemory(
                m_device, &allocation_info, nullptr, &texture.memory),
            "allocate texture image memory") ||
        !vk_ok(
            vkBindImageMemory(m_device, texture.image, texture.memory, 0),
            "bind texture image memory")) {
        destroy_gpu_texture(texture);
        return false;
    }

    // 3. Copy the pixels into level 0 and build the other levels. Smaller
    // copies of the image keep far-away surfaces from shimmering: the
    // sampler picks the level that matches the on-screen size. Afterwards
    // all levels are ready for sampling
    if (!uploader.upload_texture(
            ctx,
            texture.image,
            width,
            height,
            mip_levels,
            pixels,
            static_cast<VkDeviceSize>(width) * height * 4)) {
        destroy_gpu_texture(texture);
        return false;
    }

    // 4. View of the whole mip chain
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_info.image = texture.image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = kTextureFormat;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    if (!vk_ok(
            vkCreateImageView(
                m_device, &view_info, nullptr, &texture.view),
            "create texture image view")) {
        destroy_gpu_texture(texture);
        return false;
    }

    // 5. The descriptor set: "image view + sampler" as one binding. It is
    // written once, here, before any frame can use it, and never again.
    // This is the last step that can fail, so a set is never leaked
    if (!allocate_set(texture.set)) {
        destroy_gpu_texture(texture);
        return false;
    }

    VkDescriptorImageInfo image_descriptor{};
    image_descriptor.sampler = m_sampler;
    image_descriptor.imageView = texture.view;
    image_descriptor.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = texture.set;
    write.dstBinding = 0;
    write.dstArrayElement = 0;
    write.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    write.descriptorCount = 1;
    write.pImageInfo = &image_descriptor;

    vkUpdateDescriptorSets(m_device, 1, &write, 0, nullptr);

    out = texture;
    return true;
}

void VulkanTextureStore::destroy_gpu_texture(GpuTexture& texture) {
    // The set belongs to a pool and goes away with it
    if (m_device != VK_NULL_HANDLE) {
        if (texture.view != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, texture.view, nullptr);
        }

        if (texture.image != VK_NULL_HANDLE) {
            vkDestroyImage(m_device, texture.image, nullptr);
        }

        if (texture.memory != VK_NULL_HANDLE) {
            vkFreeMemory(m_device, texture.memory, nullptr);
        }
    }

    texture = GpuTexture{};
}

TextureId VulkanTextureStore::create(const VulkanContext& ctx,
                                     VulkanUploader& uploader,
                                     const Image& image) {
    if (m_white.set == VK_NULL_HANDLE) {
        LOG_ERROR("the Vulkan texture store is not initialized");
        return 0;
    }

    if (!image.valid()) {
        LOG_ERROR("cannot upload an invalid image");
        return 0;
    }

    if (m_textures.size() >=
        static_cast<std::size_t>(std::numeric_limits<TextureId>::max())) {
        LOG_ERROR("maximum Vulkan texture count exceeded");
        return 0;
    }

    GpuTexture texture;

    if (!create_gpu_texture(ctx,
                            uploader,
                            image.pixels.data(),
                            static_cast<std::uint32_t>(image.width),
                            static_cast<std::uint32_t>(image.height),
                            texture)) {
        return 0;
    }

    m_textures.push_back(texture);

    return static_cast<TextureId>(m_textures.size());
}

VkDescriptorSet VulkanTextureStore::descriptor_set(TextureId id) const {
    if (id == 0 || static_cast<std::size_t>(id) > m_textures.size()) {
        return m_white.set;
    }

    return m_textures[static_cast<std::size_t>(id) - 1].set;
}

void VulkanTextureStore::shutdown() {
    if (m_device != VK_NULL_HANDLE) {
        // Normally the renderer is already idle; this makes it safe to
        // call from anywhere. Nothing is in flight afterwards
        (void)vkDeviceWaitIdle(m_device);

        // Pools first (this frees all their sets), then the sampler and
        // the set layout, and last the images the sets pointed at
        for (const VkDescriptorPool pool : m_pools) {
            vkDestroyDescriptorPool(m_device, pool, nullptr);
        }

        if (m_sampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_sampler, nullptr);
        }

        if (m_set_layout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_set_layout, nullptr);
        }

        for (GpuTexture& texture : m_textures) {
            destroy_gpu_texture(texture);
        }

        destroy_gpu_texture(m_white);
    }

    m_pools.clear();
    m_textures.clear();
    m_white = GpuTexture{};
    m_sampler = VK_NULL_HANDLE;
    m_set_layout = VK_NULL_HANDLE;
    m_sets_in_newest_pool = 0;
    m_mipmaps_supported = false;
    m_max_dimension = 0;
    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
