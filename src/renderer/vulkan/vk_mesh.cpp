#include "renderer/vulkan/vk_buffer.h"
#include "renderer/vulkan/vk_context.h"
#include "renderer/vulkan/vk_mesh.h"
#include "renderer/vulkan/vk_pipeline.h"
#include "renderer/vulkan/vk_shader.h"
#include "renderer/vulkan/vk_texture.h"
#include "renderer/vulkan/vk_upload.h"

#include <cstddef>
#include <limits>

namespace gfx {

static_assert(sizeof(Vec3) == 12,
              "Vec3 must contain exactly three floats");

static_assert(sizeof(Mat4) == 64,
              "Mat4 must be 64 bytes for push constants");

// Push constants are the fastest way to give a draw call a few values,
// but the guaranteed budget is only 128 bytes. Two mat4s (model and
// normal matrix) would use all of it, leaving no room for the material
// colour. The normal matrix only needs its upper 3x3 part, so it is
// sent as three columns, each padded to a vec4 (a mat3 in a GLSL push
// block would waste the same space anyway): 64 + 3 * 16 + 16 = 128.
// The shaders declare the same block (mesh.vert, mesh.frag, wire.vert)
struct MeshPush {
    Mat4 model;         // offset 0
    Vec4 normal_col0;   // offset 64: normal matrix, column 0 (w unused)
    Vec4 normal_col1;   // offset 80
    Vec4 normal_col2;   // offset 96
    Vec4 base_color;    // offset 112: material colour (rgb), w unused
};

static_assert(sizeof(MeshPush) == 128,
              "MeshPush must fit the 128 byte push constant budget");
static_assert(offsetof(MeshPush, model) == 0,
              "MeshPush::model must start at byte 0");
static_assert(offsetof(MeshPush, normal_col0) == 64,
              "MeshPush::normal_col0 must start at byte 64");
static_assert(offsetof(MeshPush, normal_col1) == 80,
              "MeshPush::normal_col1 must start at byte 80");
static_assert(offsetof(MeshPush, normal_col2) == 96,
              "MeshPush::normal_col2 must start at byte 96");
static_assert(offsetof(MeshPush, base_color) == 112,
              "MeshPush::base_color must start at byte 112");

// Both pipelines use this one range, and every vkCmdPushConstants call
// must pass exactly the same stage flags as the range it writes into
constexpr VkShaderStageFlags kPushStages =
    VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

// Vertex layout: position (3 floats), normal (3 floats), uv (2 floats)
constexpr std::uint32_t kVertexStride =
    static_cast<std::uint32_t>(8 * sizeof(float));

static MeshPush make_push(const DrawItem& item, Vec3 base_color) {
    MeshPush push{};
    push.model = item.model;

    // Only the upper 3x3 of the normal matrix matters
    const Mat4& n = item.normal_matrix;
    push.normal_col0 = Vec4{n.col[0].x, n.col[0].y, n.col[0].z, 0.0f};
    push.normal_col1 = Vec4{n.col[1].x, n.col[1].y, n.col[1].z, 0.0f};
    push.normal_col2 = Vec4{n.col[2].x, n.col[2].y, n.col[2].z, 0.0f};
    push.base_color = make_vec4(base_color, 0.0f);
    return push;
}

VulkanMeshRenderer::~VulkanMeshRenderer() {
    shutdown();
}

void VulkanMeshRenderer::destroy_pipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
        }

        if (m_wire_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_wire_pipeline, nullptr);
        }

        if (m_pipeline_layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(
                m_device, m_pipeline_layout, nullptr);
        }
    }

    m_pipeline = VK_NULL_HANDLE;
    m_wire_pipeline = VK_NULL_HANDLE;
    m_pipeline_layout = VK_NULL_HANDLE;
}

bool VulkanMeshRenderer::init(VkDevice device, VkFormat color_format,
            VkFormat depth_format, VkDescriptorSetLayout frame_layout,
            VkDescriptorSetLayout texture_layout) {

    // Rebuild the pipeline without deleting uploaded mesh buffers
    destroy_pipeline();

    if (device == VK_NULL_HANDLE ||
        color_format == VK_FORMAT_UNDEFINED ||
        depth_format == VK_FORMAT_UNDEFINED ||
        frame_layout == VK_NULL_HANDLE ||
        texture_layout == VK_NULL_HANDLE) {
        LOG_ERROR("invalid arguments for Vulkan mesh pipeline");
        return false;
    }

    m_device = device;

    VkShaderModule vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/mesh.vert.spv");

    if (vertex_shader == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    VkShaderModule fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/mesh.frag.spv");

    if (fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            m_device, vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    // One range for the whole push block, readable by both stages: the
    // vertex shader needs the matrices, the fragment shader the colour
    VkPushConstantRange push_range{};
    push_range.stageFlags = kPushStages;
    push_range.offset = 0;
    push_range.size = sizeof(MeshPush);

    // Set 0 = per-frame uniforms, set 1 = the texture of the draw call.
    // The wireframe pipeline shares this layout but never binds set 1
    const VkDescriptorSetLayout set_layouts[2] = {
        frame_layout, texture_layout
    };

    VkPipelineLayoutCreateInfo layout_info{};
    layout_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout_info.setLayoutCount = 2;
    layout_info.pSetLayouts = set_layouts;
    layout_info.pushConstantRangeCount = 1;
    layout_info.pPushConstantRanges = &push_range;

    if (!vk_ok(
            vkCreatePipelineLayout(
                m_device,
                &layout_info,
                nullptr,
                &m_pipeline_layout),
            "create mesh pipeline layout")) {
        vkDestroyShaderModule(
            m_device, fragment_shader, nullptr);
        vkDestroyShaderModule(
            m_device, vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    VkVertexInputBindingDescription binding{};
    binding.binding = 0;
    binding.stride = kVertexStride;
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    VkVertexInputAttributeDescription position{};
    position.location = 0;
    position.binding = 0;
    position.format = VK_FORMAT_R32G32B32_SFLOAT;
    position.offset = 0;

    VkVertexInputAttributeDescription normal{};
    normal.location = 1;
    normal.binding = 0;
    normal.format = VK_FORMAT_R32G32B32_SFLOAT;
    normal.offset = 3 * sizeof(float);

    VkVertexInputAttributeDescription uv{};
    uv.location = 2;
    uv.binding = 0;
    uv.format = VK_FORMAT_R32G32_SFLOAT;
    uv.offset = 6 * sizeof(float);

    VulkanPipelineDesc desc{};
    desc.vert = vertex_shader;
    desc.frag = fragment_shader;
    desc.layout = m_pipeline_layout;
    desc.color_format = color_format;
    desc.depth_format = depth_format;
    desc.bindings = {binding};
    desc.attributes = {position, normal, uv};

    m_pipeline = create_graphics_pipeline(m_device, desc);

    vkDestroyShaderModule(
        m_device, fragment_shader, nullptr);
    vkDestroyShaderModule(
        m_device, vertex_shader, nullptr);

    if (m_pipeline == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    // Load the wireframe shader modules
    VkShaderModule wire_vertex_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/wire.vert.spv");

    if (wire_vertex_shader == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    VkShaderModule wire_fragment_shader = load_shader_module(
        m_device, GFX_SHADER_DIR "/wire.frag.spv");

    if (wire_fragment_shader == VK_NULL_HANDLE) {
        vkDestroyShaderModule(
            m_device, wire_vertex_shader, nullptr);
        destroy_pipeline();
        return false;
    }

    // Reuse the existing mesh vertex binding:
    // position + normal + uv, eight floats (32 bytes) per vertex
    VkVertexInputBindingDescription wire_binding{};
    wire_binding.binding = 0;
    wire_binding.stride = kVertexStride;
    wire_binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    // Wireframe reads only the position attribute
    VkVertexInputAttributeDescription wire_position{};
    wire_position.location = 0;
    wire_position.binding = 0;
    wire_position.format = VK_FORMAT_R32G32B32_SFLOAT;
    wire_position.offset = 0;

    // Create a second pipeline using the EXISTING pipeline layout
    VulkanPipelineDesc wire_desc{};
    wire_desc.vert = wire_vertex_shader;
    wire_desc.frag = wire_fragment_shader;
    wire_desc.layout = m_pipeline_layout;
    wire_desc.color_format = color_format;
    wire_desc.depth_format = depth_format;

    wire_desc.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    wire_desc.bindings = {wire_binding};
    wire_desc.attributes = {wire_position};

    wire_desc.depth_test = true;
    wire_desc.depth_write = true;
    wire_desc.blend = false;

    m_wire_pipeline = create_graphics_pipeline(
        m_device, wire_desc);

    // Shader modules can be destroyed after pipeline creation
    vkDestroyShaderModule(
        m_device, wire_fragment_shader, nullptr);
    vkDestroyShaderModule(
        m_device, wire_vertex_shader, nullptr);

    if (m_wire_pipeline == VK_NULL_HANDLE) {
        destroy_pipeline();
        return false;
    }

    LOG_INFO("Vulkan mesh pipeline created");
    return true;
}

MeshId VulkanMeshRenderer::create(const VulkanContext& ctx,
                                  VulkanUploader& uploader,
                                  const Mesh& mesh,
                                  const std::vector<TextureId>& textures,
                                  const VulkanTextureStore& store) {

    if (m_device == VK_NULL_HANDLE ||
        ctx.device() != m_device ||
        m_pipeline == VK_NULL_HANDLE) {
        LOG_ERROR("Vulkan mesh renderer is not initialized");
        return 0;
    }

    if (mesh.positions.empty() ||
        mesh.indices.empty() ||
        mesh.indices.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        LOG_ERROR("invalid mesh data or index count");
        return 0;
    }

    if (mesh.normals.size() != mesh.positions.size()) {
        LOG_ERROR(
            "mesh normals and positions have different sizes");
        return 0;
    }

    for (const std::uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size()) {
            LOG_ERROR("mesh contains an out-of-range index");
            return 0;
        }
    }

    // Edges are optional, but a present edge list must be valid
    if (mesh.edges.size() % 2 != 0 ||
        mesh.edges.size() >
            std::numeric_limits<std::uint32_t>::max()) {
        LOG_ERROR("invalid mesh edge data or edge index count");
        return 0;
    }

    for (const std::uint32_t index : mesh.edges) {
        if (index >= mesh.positions.size()) {
            LOG_ERROR("mesh contains an out-of-range edge index");
            return 0;
        }
    }

    // Texture coordinates are optional; without them every vertex gets
    // (0, 0) and the texture is just a single texel
    const bool has_uv = mesh.texcoords.size() == mesh.positions.size();

    if (!mesh.texcoords.empty() && !has_uv) {
        LOG_WARN("mesh texcoords and positions have different sizes; "
                 "ignoring the texcoords");
    }

    // Interleave each vertex as:
    // position.x, position.y, position.z,
    // normal.x, normal.y, normal.z,
    // u, v
    constexpr std::size_t kFloatsPerVertex = 8;

    if (mesh.positions.size() >
        std::numeric_limits<std::size_t>::max() /
            (kFloatsPerVertex * sizeof(float))) {
        LOG_ERROR("mesh vertex buffer size overflow");
        return 0;
    }

    // Triangles of the same material next to each other: one draw call
    // per material instead of switching the material per triangle
    std::vector<std::uint32_t> grouped_indices;
    const std::vector<MaterialRange> material_ranges =
        group_triangles_by_material(mesh, grouped_indices);

    if (grouped_indices.empty()) {
        LOG_ERROR("mesh has no complete triangle");
        return 0;
    }

    if (grouped_indices.size() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint32_t)) {
        LOG_ERROR("mesh index buffer size overflow");
        return 0;
    }

    std::vector<float> vertex_data;
    vertex_data.reserve(
        mesh.positions.size() * kFloatsPerVertex);

    for (std::size_t i = 0; i < mesh.positions.size(); ++i) {
        const Vec3& p = mesh.positions[i];
        const Vec3& n = mesh.normals[i];

        vertex_data.push_back(p.x);
        vertex_data.push_back(p.y);
        vertex_data.push_back(p.z);

        vertex_data.push_back(n.x);
        vertex_data.push_back(n.y);
        vertex_data.push_back(n.z);

        vertex_data.push_back(has_uv ? mesh.texcoords[i].x : 0.0f);
        vertex_data.push_back(has_uv ? mesh.texcoords[i].y : 0.0f);
    }

    const VkDeviceSize vertex_bytes =
        static_cast<VkDeviceSize>(
            vertex_data.size() * sizeof(float));

    const VkDeviceSize index_bytes =
        static_cast<VkDeviceSize>(
            grouped_indices.size() * sizeof(std::uint32_t));

    auto gpu_mesh = std::make_unique<GpuMesh>();

    // One draw call per material range, with the material's colour and
    // texture. An unknown material or texture falls back to the default
    // colour and the white texture (the store hands out the white
    // texture's set for TextureId 0)
    for (const MaterialRange& range : material_ranges) {
        const Material fallback;
        const Material& material =
            static_cast<std::size_t>(range.material) <
                    mesh.materials.size()
                ? mesh.materials[static_cast<std::size_t>(range.material)]
                : fallback;

        TextureId texture = 0;

        if (material.base_color_image >= 0 &&
            static_cast<std::size_t>(material.base_color_image) <
                textures.size()) {
            texture = textures[static_cast<std::size_t>(
                material.base_color_image)];
        }

        GpuRange gpu_range;
        gpu_range.first_index = range.first_index;
        gpu_range.index_count = range.index_count;
        gpu_range.base_color = material.base_color;
        gpu_range.texture_set = store.descriptor_set(texture);
        gpu_mesh->ranges.push_back(gpu_range);
    }

    gpu_mesh->vertex_buffer =
        std::make_unique<VulkanBuffer>();

    if (!uploader.upload_to_device_buffer(
            ctx,
            *gpu_mesh->vertex_buffer,
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_data.data(),
            vertex_bytes)) {
        LOG_ERROR("failed to upload mesh vertex buffer");
        return 0;
    }

    gpu_mesh->index_buffer =
        std::make_unique<VulkanBuffer>();

    if (!uploader.upload_to_device_buffer(
            ctx,
            *gpu_mesh->index_buffer,
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            grouped_indices.data(),
            index_bytes)) {
        LOG_ERROR("failed to upload mesh index buffer");
        return 0;
    }

    // Upload an edge buffer only when the mesh has edges
    if (!mesh.edges.empty()) {
        if (mesh.edges.size() >
            std::numeric_limits<std::size_t>::max() /
                sizeof(std::uint32_t)) {
            LOG_ERROR("mesh edge buffer size overflow");
            return 0;
        }

        const VkDeviceSize edge_bytes =
            static_cast<VkDeviceSize>(
                mesh.edges.size() * sizeof(std::uint32_t));

        gpu_mesh->edge_buffer =
            std::make_unique<VulkanBuffer>();

        if (!uploader.upload_to_device_buffer(
                ctx,
                *gpu_mesh->edge_buffer,
                VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                mesh.edges.data(),
                edge_bytes)) {
            LOG_ERROR("failed to upload mesh edge buffer");
            return 0;
        }

        gpu_mesh->edge_index_count =
            static_cast<std::uint32_t>(mesh.edges.size());
    }

    if (m_meshes.size() >=
        static_cast<std::size_t>(
            std::numeric_limits<MeshId>::max())) {
        LOG_ERROR("maximum Vulkan mesh count exceeded");
        return 0;
    }

    m_meshes.push_back(std::move(gpu_mesh));

    return static_cast<MeshId>(m_meshes.size());
}

void VulkanMeshRenderer::draw(VkCommandBuffer cmd, VkExtent2D extent,
        VkDescriptorSet descriptor_set, const std::vector<DrawItem>& items) {

    if (m_pipeline == VK_NULL_HANDLE ||
        m_pipeline_layout == VK_NULL_HANDLE ||
        cmd == VK_NULL_HANDLE ||
        descriptor_set == VK_NULL_HANDLE ||
        extent.width == 0 ||
        extent.height == 0) {
        return;
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    vkCmdBindPipeline(
        cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, m_pipeline);

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Set 0: the frame uniforms, the same for every draw call
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);

    // Set 1 (the texture) is bound per range, but only when it changes
    VkDescriptorSet bound_texture_set = VK_NULL_HANDLE;

    const VkDeviceSize offset = 0;

    for (const DrawItem& item : items) {
        if (item.mesh == 0 ||
            static_cast<std::size_t>(item.mesh) > m_meshes.size()) {
            continue;
        }

        const GpuMesh& mesh =
            *m_meshes[static_cast<std::size_t>(item.mesh) - 1];

        if (mesh.vertex_buffer == nullptr ||
            mesh.index_buffer == nullptr ||
            mesh.ranges.empty()) {
            continue;
        }

        const VkBuffer vertex_buffer =
            mesh.vertex_buffer->buffer();

        vkCmdBindVertexBuffers(
            cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdBindIndexBuffer(
            cmd,
            mesh.index_buffer->buffer(),
            0,
            VK_INDEX_TYPE_UINT32);

        // One draw call per material range
        for (const GpuRange& range : mesh.ranges) {
            if (range.texture_set == VK_NULL_HANDLE ||
                range.index_count == 0) {
                continue;
            }

            if (range.texture_set != bound_texture_set) {
                vkCmdBindDescriptorSets(
                    cmd,
                    VK_PIPELINE_BIND_POINT_GRAPHICS,
                    m_pipeline_layout,
                    1,
                    1,
                    &range.texture_set,
                    0,
                    nullptr);

                bound_texture_set = range.texture_set;
            }

            const MeshPush push = make_push(item, range.base_color);

            vkCmdPushConstants(
                cmd,
                m_pipeline_layout,
                kPushStages,
                0,
                sizeof(MeshPush),
                &push);

            vkCmdDrawIndexed(
                cmd, range.index_count, 1, range.first_index, 0, 0);
        }
    }
}

void VulkanMeshRenderer::draw_wireframe(VkCommandBuffer cmd,
                            VkExtent2D extent,
                            VkDescriptorSet descriptor_set,
                            const std::vector<DrawItem>& items) {

    if (m_wire_pipeline == VK_NULL_HANDLE ||
        m_pipeline_layout == VK_NULL_HANDLE ||
        cmd == VK_NULL_HANDLE ||
        descriptor_set == VK_NULL_HANDLE ||
        extent.width == 0 ||
        extent.height == 0) {
        return;
    }

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;

    vkCmdBindPipeline(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_wire_pipeline);

    vkCmdSetViewport(cmd, 0, 1, &viewport);
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    // Only set 0 is needed: the wire shaders use no texture
    vkCmdBindDescriptorSets(
        cmd,
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        m_pipeline_layout,
        0,
        1,
        &descriptor_set,
        0,
        nullptr);

    const VkDeviceSize offset = 0;

    for (const DrawItem& item : items) {
        if (item.mesh == 0 ||
            static_cast<std::size_t>(item.mesh) > m_meshes.size()) {
            continue;
        }

        const GpuMesh& mesh =
            *m_meshes[static_cast<std::size_t>(item.mesh) - 1];

        // A mesh without edges is valid; it contributes no lines
        if (mesh.vertex_buffer == nullptr ||
            mesh.edge_buffer == nullptr ||
            mesh.edge_index_count == 0) {
            continue;
        }

        // The push block is shared with the mesh pipeline, but the
        // wire shader only reads the model matrix. The stage flags must
        // be the same ones as in the layout's push constant range
        const MeshPush push = make_push(item, Vec3{0.0f, 0.0f, 0.0f});

        vkCmdPushConstants(
            cmd,
            m_pipeline_layout,
            kPushStages,
            0,
            sizeof(MeshPush),
            &push);

        const VkBuffer vertex_buffer =
            mesh.vertex_buffer->buffer();

        vkCmdBindVertexBuffers(
            cmd, 0, 1, &vertex_buffer, &offset);

        vkCmdBindIndexBuffer(
            cmd,
            mesh.edge_buffer->buffer(),
            0,
            VK_INDEX_TYPE_UINT32);

        vkCmdDrawIndexed(
            cmd,
            mesh.edge_index_count,
            1,
            0,
            0,
            0);
    }
}

void VulkanMeshRenderer::shutdown() {
    // The renderer must be idle before this is called
    m_meshes.clear();
    destroy_pipeline();
    m_device = VK_NULL_HANDLE;
}

} // namespace gfx
