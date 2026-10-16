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

uniform mat4 u_model;
uniform mat4 u_normal_model;
uniform mat4 u_view;
uniform mat4 u_view_proj;

out vec3 v_view_pos;
out vec3 v_view_normal;

void main() {
    vec4 world = u_model * vec4(a_position, 1.0);

    v_view_pos = (u_view * world).xyz;
    v_view_normal =
        mat3(u_view) * (mat3(u_normal_model) * a_normal);

    gl_Position = u_view_proj * world;
}
)GLSL";

constexpr const char* kMeshFragmentShader = R"GLSL(
#version 460 core

in vec3 v_view_pos;
in vec3 v_view_normal;

uniform mat4 u_view;

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

    vec3 base = vec3(0.72, 0.74, 0.78);
    vec3 color = base * (ambient + 0.75 * diff) + vec3(spec);

    frag_color = vec4(min(color, vec3(1.0)), 1.0);
}
)GLSL";

} // namespace

GLMesh::~GLMesh() {
    shutdown();
}

bool GLMesh::upload(const Mesh& mesh) {
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

    // Interleave each vertex as:
    // position.x, position.y, position.z,
    // normal.x, normal.y, normal.z.
    constexpr std::size_t kFloatsPerVertex = 6;

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
    }

    const std::size_t vertex_buffer_bytes =
        vertex_data.size() * sizeof(float);

    if (vertex_buffer_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<GLsizeiptr>::max())) {
        LOG_ERROR("mesh vertex buffer exceeds OpenGL size limit");
        return false;
    }

    if (mesh.indices.size() >
        std::numeric_limits<std::size_t>::max() /
            sizeof(std::uint32_t)) {
        LOG_ERROR("mesh index buffer size overflow");
        return false;
    }

    const std::size_t index_buffer_bytes =
        mesh.indices.size() * sizeof(std::uint32_t);

    if (index_buffer_bytes >
        static_cast<std::size_t>(
            std::numeric_limits<GLsizeiptr>::max())) {
        LOG_ERROR("mesh index buffer exceeds OpenGL size limit");
        return false;
    }

    m_index_count = static_cast<GLsizei>(mesh.indices.size());

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

    // Attribute 0: position XYZ, stride 24 bytes, offset 0
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        kStride,
        nullptr
    );

    // Attribute 1: normal XYZ, stride 24 bytes, offset 12 bytes
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        kStride,
        reinterpret_cast<const void*>(3 * sizeof(float))
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
        mesh.indices.data(),
        GL_STATIC_DRAW
    );

    // Do not unbind GL_ELEMENT_ARRAY_BUFFER while this VAO is bound
    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    if (glGetError() != GL_NO_ERROR) {
        LOG_ERROR("OpenGL error while uploading mesh");
        shutdown();
        return false;
    }

    return true;
}

void GLMesh::draw() const {
    if (m_vao == 0 || m_index_count <= 0) {
        return;
    }

    glBindVertexArray(m_vao);
    glDrawElements(
        GL_TRIANGLES,
        m_index_count,
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

    m_index_count = 0;
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

    if (m_view_location < 0 ||
        m_view_proj_location < 0 ||
        m_model_location < 0 ||
        m_normal_model_location < 0) {
        LOG_ERROR("mesh shader uniforms are missing");
        shutdown();
        return false;
    }

    return true;
}

MeshId GLMeshRenderer::add(const Mesh& mesh) {
    if (m_program == 0 ||
        m_meshes.size() >=
            static_cast<std::size_t>(
                std::numeric_limits<MeshId>::max())) {
        return 0;
    }

    std::unique_ptr<GLMesh> gpu_mesh =
        std::make_unique<GLMesh>();

    if (!gpu_mesh->upload(mesh)) {
        return 0;
    }

    m_meshes.push_back(std::move(gpu_mesh));

    return static_cast<MeshId>(m_meshes.size());
}

void GLMeshRenderer::draw(const std::vector<DrawItem>& items,
                          const Mat4& view_proj,
                          const Mat4& view) const {
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

        m_meshes[static_cast<std::size_t>(item.mesh) - 1]->draw();
    }
}

void GLMeshRenderer::shutdown() {
    // Release GPU meshes while the OpenGL context is alive
    m_meshes.clear();

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }

    m_view_location = -1;
    m_view_proj_location = -1;
    m_model_location = -1;
    m_normal_model_location = -1;
}

} // namespace gfx
