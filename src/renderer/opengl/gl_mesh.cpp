#include "renderer/opengl/gl_mesh.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"
#include "core/log.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace gfx {

static_assert(sizeof(Vec3) == 12,
              "Vec3 must contain exactly three floats");

namespace {

constexpr const char* kMeshVertexShader = R"GLSL(
#version 460 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;

uniform mat4 u_model;
uniform mat4 u_normal_model;
uniform mat4 u_view;
uniform mat4 u_view_proj;

out vec3 v_view_pos;
out vec3 v_view_normal;
out vec2 v_uv;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);

    v_view_pos = (u_view * world).xyz;
    v_view_normal =
        mat3(u_view) * (mat3(u_normal_model) * a_normal);
    v_uv = a_uv;

    gl_Position = u_view_proj * world;
}
)GLSL";

constexpr const char* kMeshFragmentShader = R"GLSL(
#version 460 core

in vec3 v_view_pos;
in vec3 v_view_normal;
in vec2 v_uv;

uniform mat4 u_view;

// Material of the range being drawn. The texture is 1x1 white for
// surfaces without one, so no branch is needed for that case
uniform vec3 u_base_color;
uniform sampler2D u_base_texture;
uniform bool u_show_materials;

out vec4 frag_color;

void main() {
    vec3 n = normalize(v_view_normal);

    // Light both front-facing and back-facing surfaces.
    if (!gl_FrontFacing) {
        n = -n;
    }

    // Direction from the surface toward the camera.
    vec3 v = normalize(-v_view_pos);

    // Key light positioned up-left of the camera.
    vec3 key_dir = normalize(vec3(-0.5, 0.6, 0.7));
    float diff = max(dot(n, key_dir), 0.0);

    // World +Z expressed in view space.
    vec3 up_view = mat3(u_view)[2];
    float hemi = dot(n, up_view) * 0.5 + 0.5;

    vec3 ambient = mix(
        vec3(0.10, 0.10, 0.12),
        vec3(0.30, 0.32, 0.36),
        hemi
    );

    // Blinn-Phong specular highlight.
    vec3 h = normalize(key_dir + v);
    float spec = diff > 0.0
        ? pow(max(dot(n, h), 0.0), 32.0) * 0.25
        : 0.0;

    // Colours: this shading works directly in display values, with no
    // linear / sRGB conversion. Textures are uploaded as plain RGBA8 and
    // the base colour factor is used as given, which keeps the grey look
    // and the three backends consistent (the path tracer will do proper
    // linear light)
    //
    // Our UVs have (0, 0) at the bottom-left of the image and v pointing
    // up, but the image rows are stored top row first, so row 0 of the
    // texture is v = 1: sample at 1 - v
    vec3 base = u_show_materials
        ? u_base_color *
              texture(u_base_texture, vec2(v_uv.x, 1.0 - v_uv.y)).rgb
        : vec3(0.72, 0.74, 0.78);
    vec3 color = base * (ambient + 0.75 * diff) + vec3(spec);

    frag_color = vec4(min(color, vec3(1.0)), 1.0);
}
)GLSL";

constexpr const char* kWireVertexShader = R"GLSL(
#version 460 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_model;
uniform mat4 u_view_proj;

void main() {
    gl_Position =
        u_view_proj * u_model * vec4(a_position, 1.0);
}
)GLSL";

constexpr const char* kWireFragmentShader = R"GLSL(
#version 460 core

uniform vec3 u_color;

out vec4 frag_color;

void main() {
    frag_color = vec4(u_color, 1.0);
}

)GLSL";

} // namespace

GLMesh::~GLMesh() {
    shutdown();
}

bool GLMesh::upload(const Mesh& mesh,
                    const std::vector<TextureId>& textures,
                    const GLTextureStore& store) {
    shutdown();

    if (mesh.positions.empty() ||
        mesh.indices.empty() ||
        mesh.indices.size() >
            static_cast<std::size_t>(
                std::numeric_limits<GLsizei>::max())) {
        LOG_ERROR("invalid mesh data or index count");
        return false;
    }

    if (mesh.normals.size() != mesh.positions.size()) {
        LOG_ERROR("mesh normals and positions have different sizes");
        return false;
    }

    for (const std::uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size()) {
            LOG_ERROR("mesh contains an out-of-range index");
            return false;
        }
    }

    if (!mesh.edges.empty()) {
        if (mesh.edges.size() % 2 != 0 ||
            mesh.edges.size() >
                static_cast<std::size_t>(
                    std::numeric_limits<GLsizei>::max())) {
            LOG_ERROR("invalid mesh edge data or edge index count");
            return false;
        }

        for (const std::uint32_t index : mesh.edges) {
            if (index >= mesh.positions.size()) {
                LOG_ERROR("mesh contains an out-of-range edge index");
                return false;
            }
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
            kFloatsPerVertex / sizeof(float)) {
        LOG_ERROR("mesh vertex buffer size overflow");
        return false;
    }

    std::vector<float> vertex_data;
    vertex_data.reserve(mesh.positions.size() * kFloatsPerVertex);

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

    // Triangles of the same material next to each other: one draw call
    // per material instead of switching the material per triangle
    std::vector<std::uint32_t> grouped_indices;
    const std::vector<MaterialRange> material_ranges =
        group_triangles_by_material(mesh, grouped_indices);

    if (grouped_indices.empty()) {
        LOG_ERROR("mesh has no complete triangle");
        return false;
    }

    const std::size_t vertex_buffer_bytes =
        vertex_data.size() * sizeof(float);

    if (vertex_buffer_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<GLsizeiptr>::max())) {
        LOG_ERROR("mesh vertex buffer exceeds OpenGL size limit");
        return false;
    }

    if (grouped_indices.size() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint32_t)) {
        LOG_ERROR("mesh index buffer size overflow");
        return false;
    }

    const std::size_t index_buffer_bytes =
        grouped_indices.size() * sizeof(std::uint32_t);

    if (index_buffer_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<GLsizeiptr>::max())) {
        LOG_ERROR("mesh index buffer exceeds OpenGL size limit");
        return false;
    }

    if (mesh.edges.size() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint32_t)) {
        LOG_ERROR("mesh edge buffer size overflow");
        return false;
    }

    const std::size_t edge_buffer_bytes =
        mesh.edges.size() * sizeof(std::uint32_t);

    if (edge_buffer_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<GLsizeiptr>::max())) {
        LOG_ERROR("mesh edge buffer exceeds OpenGL size limit");
        return false;
    }

    // One draw call per material range, with the material's colour and
    // texture. An unknown material or texture falls back to the default
    // colour and the white texture
    m_ranges.clear();

    for (const MaterialRange& range : material_ranges) {
        const Material fallback;
        const Material& material =
            static_cast<std::size_t>(range.material) <
                    mesh.materials.size()
                ? mesh.materials[static_cast<std::size_t>(range.material)]
                : fallback;

        unsigned int texture = 0;

        if (material.base_color_image >= 0 &&
            static_cast<std::size_t>(material.base_color_image) <
                textures.size()) {
            texture = store.gl_name(
                textures[static_cast<std::size_t>(
                    material.base_color_image)]);
        }

        GLDrawRange draw_range;
        draw_range.first_index_byte =
            static_cast<std::size_t>(range.first_index) *
            sizeof(std::uint32_t);
        draw_range.index_count = static_cast<GLsizei>(range.index_count);
        draw_range.base_color = material.base_color;
        draw_range.texture = texture != 0 ? texture : store.white();
        m_ranges.push_back(draw_range);
    }

    m_edge_index_count = static_cast<GLsizei>(mesh.edges.size());

    // Discard stale OpenGL errors without an unbounded loop
    for (int i = 0; i < 16 && glGetError() != GL_NO_ERROR; ++i) {
    }

    glGenVertexArrays(1, &m_vao);
    if (m_vao == 0) {
        LOG_ERROR("failed to create mesh VAO");
        shutdown();
        return false;
    }

    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    if (m_vbo == 0) {
        LOG_ERROR("failed to create mesh VBO");
        glBindVertexArray(0);
        shutdown();
        return false;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(vertex_buffer_bytes),
        vertex_data.data(),
        GL_STATIC_DRAW
    );

    constexpr GLsizei kStride =
        static_cast<GLsizei>(kFloatsPerVertex * sizeof(float));

    // Attribute 0: position XYZ, stride 32 bytes, offset 0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        kStride,
        nullptr
    );

    // Attribute 1: normal XYZ, stride 32 bytes, offset 12 bytes
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        kStride,
        reinterpret_cast<const void*>(3 * sizeof(float))
    );

    // Attribute 2: texture coordinates UV, stride 32 bytes, offset 24 bytes
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2,
        2,
        GL_FLOAT,
        GL_FALSE,
        kStride,
        reinterpret_cast<const void*>(6 * sizeof(float))
    );

    glGenBuffers(1, &m_ebo);
    if (m_ebo == 0) {
        LOG_ERROR("failed to create mesh index EBO");
        glBindVertexArray(0);
        shutdown();
        return false;
    }

    // The EBO binding belongs to the currently bound VAO
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(index_buffer_bytes),
        grouped_indices.data(),
        GL_STATIC_DRAW
    );

    // Do not unbind GL_ELEMENT_ARRAY_BUFFER while this VAO is bound
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Create a separate VAO for wireframe drawing
    glGenVertexArrays(1, &m_edge_vao);
    if (m_edge_vao == 0) {
        LOG_ERROR("failed to create mesh edge VAO");
        shutdown();
        return false;
    }

    glBindVertexArray(m_edge_vao);

    // Reuse the same interleaved vertex buffer
    // Attribute 0 is position; normals are not needed for wireframe
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(kFloatsPerVertex * sizeof(float)),
        nullptr
    );

    // Create and upload the edge EBO while the edge VAO is bound
    glGenBuffers(1, &m_edge_ebo);
    if (m_edge_ebo == 0) {
        LOG_ERROR("failed to create mesh edge EBO");
        glBindVertexArray(0);
        shutdown();
        return false;
    }

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_edge_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(edge_buffer_bytes),
        mesh.edges.data(),
        GL_STATIC_DRAW
    );

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (glGetError() != GL_NO_ERROR) {
        LOG_ERROR("OpenGL error while uploading mesh");
        shutdown();
        return false;
    }

    return true;
}

void GLMesh::draw(int base_color_location) const {
    if (m_vao == 0 || m_ranges.empty()) {
        return;
    }

    glBindVertexArray(m_vao);
    glActiveTexture(GL_TEXTURE0);

    for (const GLDrawRange& range : m_ranges) {
        glUniform3f(
            base_color_location,
            range.base_color.x,
            range.base_color.y,
            range.base_color.z
        );

        glBindTexture(GL_TEXTURE_2D, range.texture);

        // With an element buffer bound, the last argument is a byte
        // offset into it, not a pointer
        glDrawElements(
            GL_TRIANGLES,
            range.index_count,
            GL_UNSIGNED_INT,
            reinterpret_cast<const void*>(range.first_index_byte)
        );
    }

    glBindTexture(GL_TEXTURE_2D, 0);
    glBindVertexArray(0);
}

void GLMesh::draw_edges() const {
    if (m_edge_vao == 0 || m_edge_index_count <= 0) {
        return;
    }

    glBindVertexArray(m_edge_vao);
    glDrawElements(
        GL_LINES,
        m_edge_index_count,
        GL_UNSIGNED_INT,
        nullptr
    );
    glBindVertexArray(0);
}

void GLMesh::shutdown() {
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    if (m_ebo != 0) {
        glDeleteBuffers(1, &m_ebo);
        m_ebo = 0;
    }

    if (m_edge_vao != 0) {
        glDeleteVertexArrays(1, &m_edge_vao);
        m_edge_vao = 0;
    }

    if (m_edge_ebo != 0) {
        glDeleteBuffers(1, &m_edge_ebo);
        m_edge_ebo = 0;
    }

    m_ranges.clear();
    m_edge_index_count = 0;
}

GLMeshRenderer::~GLMeshRenderer() {
    shutdown();
}

bool GLMeshRenderer::init() {
    shutdown();

    m_program = gl_create_program(
        kMeshVertexShader,
        kMeshFragmentShader
    );

    if (m_program == 0) {
        LOG_ERROR("failed to create mesh shader program");
        return false;
    }

    m_view_location =
        gl_uniform_location(m_program, "u_view");

    m_view_proj_location =
        gl_uniform_location(m_program, "u_view_proj");
    
    m_model_location =
        gl_uniform_location(m_program, "u_model");

    m_normal_model_location =
        gl_uniform_location(m_program, "u_normal_model");

    m_base_color_location =
        gl_uniform_location(m_program, "u_base_color");

    m_base_texture_location =
        gl_uniform_location(m_program, "u_base_texture");

    m_show_materials_location =
        gl_uniform_location(m_program, "u_show_materials");

    if (m_view_location < 0 ||
        m_view_proj_location < 0 ||
        m_model_location < 0 ||
        m_normal_model_location < 0 ||
        m_base_color_location < 0 ||
        m_base_texture_location < 0 ||
        m_show_materials_location < 0) {
        LOG_ERROR("mesh shader uniforms are missing");
        shutdown();
        return false;
    }

    m_wire_program = gl_create_program(
        kWireVertexShader,
        kWireFragmentShader
    );

    if (m_wire_program == 0) {
        LOG_ERROR("failed to create wireframe shader program");
        shutdown();
        return false;
    }

    m_wire_view_proj_location =
        gl_uniform_location(m_wire_program, "u_view_proj");

    m_wire_model_location =
        gl_uniform_location(m_wire_program, "u_model");

    m_wire_color_location =
        gl_uniform_location(m_wire_program, "u_color");

    if (m_wire_view_proj_location < 0 ||
        m_wire_model_location < 0 ||
        m_wire_color_location < 0) {
        LOG_ERROR("wireframe shader uniforms are missing");
        shutdown();
        return false;
    }

    return true;
}

MeshId GLMeshRenderer::add(const Mesh& mesh,
                           const std::vector<TextureId>& textures,
                           const GLTextureStore& store) {
    if (m_program == 0 ||
        m_meshes.size() >=
            static_cast<std::size_t>(
                std::numeric_limits<MeshId>::max())) {
        return 0;
    }

    std::unique_ptr<GLMesh> gpu_mesh =
        std::make_unique<GLMesh>();

    if (!gpu_mesh->upload(mesh, textures, store)) {
        return 0;
    }

    m_meshes.push_back(std::move(gpu_mesh));

    return static_cast<MeshId>(m_meshes.size());
}

void GLMeshRenderer::draw(const std::vector<DrawItem>& items,
                          const Mat4& view_proj, const Mat4& view,
                          bool show_materials) const {
    if (m_program == 0 || items.empty()) {
        return;
    }

    glUseProgram(m_program);

    // These matrices are shared by every object in this frame
    glUniformMatrix4fv(
        m_view_location,
        1,
        GL_FALSE,
        &view.col[0].x
    );

    glUniformMatrix4fv(
        m_view_proj_location,
        1,
        GL_FALSE,
        &view_proj.col[0].x
    );

    // Texture unit 0 holds the base-colour texture
    glUniform1i(m_base_texture_location, 0);
    glUniform1i(m_show_materials_location, show_materials ? 1 : 0);

    // Each object has its own model and normal matrices
    for (const DrawItem& item : items) {
        if (item.mesh == 0 ||
            static_cast<std::size_t>(item.mesh) > m_meshes.size()) {
            continue;
        }

        glUniformMatrix4fv(
            m_model_location,
            1,
            GL_FALSE,
            &item.model.col[0].x
        );

        glUniformMatrix4fv(
            m_normal_model_location,
            1,
            GL_FALSE,
            &item.normal_matrix.col[0].x
        );

        m_meshes[static_cast<std::size_t>(item.mesh) - 1]->draw(
            m_base_color_location);
    }
}

void GLMeshRenderer::draw_wireframe(const std::vector<DrawItem>& items,
                                    const Mat4& view_proj) const {
    if (m_wire_program == 0 || items.empty()) {
        return;
    }

    glUseProgram(m_wire_program);

    // Shared by all meshes in this frame
    glUniformMatrix4fv(
        m_wire_view_proj_location,
        1,
        GL_FALSE,
        &view_proj.col[0].x
    );

    // Uniform, unlit wire color
    glUniform3f(
        m_wire_color_location,
        0.85f,
        0.85f,
        0.85f
    );

    for (const DrawItem& item : items) {
        if (item.mesh == 0 ||
            static_cast<std::size_t>(item.mesh) > m_meshes.size()) {
            continue;
        }

        glUniformMatrix4fv(
            m_wire_model_location,
            1,
            GL_FALSE,
            &item.model.col[0].x
        );

        m_meshes[
            static_cast<std::size_t>(item.mesh) - 1
        ]->draw_edges();
    }
}

void GLMeshRenderer::shutdown() {
    // Release GPU meshes while the OpenGL context is alive
    m_meshes.clear();

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }

    if (m_wire_program != 0) {
        glDeleteProgram(m_wire_program);
        m_wire_program = 0;
    }

    m_view_location = -1;
    m_view_proj_location = -1;
    m_model_location = -1;
    m_normal_model_location = -1;
    m_base_color_location = -1;
    m_base_texture_location = -1;
    m_show_materials_location = -1;

    m_wire_view_proj_location = -1;
    m_wire_model_location = -1;
    m_wire_color_location = -1;
}

} // namespace gfx
