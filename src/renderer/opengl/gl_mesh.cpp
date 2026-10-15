#include "renderer/opengl/gl_mesh.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"
#include "core/log.h"

#include <limits>
#include <memory>
#include <vector>
#include <cstddef>
#include <limits>

namespace gfx {

static_assert(sizeof(Vec3) == 12,
              "Vec3 must contain exactly three floats");

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

    for (const std::uint32_t index : mesh.indices) {
        if (index >= mesh.positions.size()) {
            LOG_ERROR("mesh contains an out-of-range index");
            return false;
        }
    }

    m_index_count = static_cast<GLsizei>(mesh.indices.size());

    // Discard any previous GL error before checking this upload
    while (glGetError() != GL_NO_ERROR) {
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
        LOG_ERROR("failed to create mesh position VBO");
        glBindVertexArray(0);
        shutdown();
        return false;
    }

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            mesh.positions.size() * sizeof(Vec3)),
        mesh.positions.data(),
        GL_STATIC_DRAW
    );

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vec3)),
        nullptr
    );

    glGenBuffers(1, &m_ebo);
    if (m_ebo == 0) {
        LOG_ERROR("failed to create mesh index EBO");
        glBindVertexArray(0);
        shutdown();
        return false;
    }

    // The VAO must be bound before attaching its element buffer
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_ebo);
    glBufferData(
        GL_ELEMENT_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(
            mesh.indices.size() * sizeof(std::uint32_t)),
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
        static_cast<GLsizei>(m_index_count),
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

namespace {

constexpr const char* kMeshVertexShader = R"GLSL(
#version 460 core

layout(location = 0) in vec3 a_position;

uniform mat4 u_view;
uniform mat4 u_view_proj;

out vec3 v_view_pos;

void main() {
    v_view_pos = (u_view * vec4(a_position, 1.0)).xyz;
    gl_Position = u_view_proj * vec4(a_position, 1.0);
}
)GLSL";

constexpr const char* kMeshFragmentShader = R"GLSL(
#version 460 core

in vec3 v_view_pos;
out vec4 frag_color;

void main() {
    vec3 n = normalize(cross(dFdx(v_view_pos), dFdy(v_view_pos)));
    vec3 l = normalize(-v_view_pos);

    float lit = abs(dot(n, l));

    frag_color = vec4(
        vec3(0.72, 0.74, 0.78) * (0.25 + 0.75 * lit),
        1.0
    );
}
)GLSL";

} // namespace

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

    if (m_view_location < 0 || m_view_proj_location < 0) {
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

void GLMeshRenderer::draw(MeshId id, const Mat4& view_proj, const Mat4& view) const {
    if (id == 0 ||
        id > m_meshes.size() ||
        m_program == 0) {
        return;
    }

    glUseProgram(m_program);

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

    m_meshes[id - 1]->draw();
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
}

} // namespace gfx
