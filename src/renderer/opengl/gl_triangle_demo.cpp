#include "renderer/opengl/gl_triangle_demo.h"
#include "renderer/opengl/gl_shader.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <cstddef>

namespace {

constexpr const char* kVertexSource = R"GLSL(#version 460 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_color;

uniform mat4 u_mvp;

out vec3 v_color;

void main() {
    v_color = a_color;
    gl_Position = u_mvp * vec4(a_position, 1.0);
}
)GLSL";

constexpr const char* kFragmentSource = R"GLSL(#version 460 core
in vec3 v_color;
out vec4 frag_color;

void main() {
    frag_color = vec4(v_color, 1.0);
}
)GLSL";

struct Vertex {
    float position[3];
    float color[3];
};

constexpr Vertex kVertices[3] = {
    {{-0.6f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}},
    {{ 0.6f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}},
    {{ 0.0f,  0.6f, 0.0f}, {0.0f, 0.0f, 1.0f}},
};

} // namespace

namespace gfx {

TriangleDemo::~TriangleDemo() {
    shutdown();
}

bool TriangleDemo::init() {
    if (m_program != 0 || m_vao != 0 || m_vbo != 0) {
        LOG_ERROR("TriangleDemo::init called more than once");
        return false;
    }

    m_program = gl_create_program(
        kVertexSource,
        kFragmentSource
    );

    if (m_program == 0) {
        return false;
    }

    m_mvp_location = glGetUniformLocation(m_program, "u_mvp");

    if (m_mvp_location == -1) {
        LOG_WARN("u_mvp uniform not found in triangle shader");
    }

    // The VAO must be bound before configuring vertex attributes
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glGenBuffers(1, &m_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

    glBufferData(
        GL_ARRAY_BUFFER,
        static_cast<GLsizeiptr>(sizeof(kVertices)),
        kVertices,
        GL_STATIC_DRAW
    );

    // Attribute 0: position
    glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vertex)),
        reinterpret_cast<const void*>(
            offsetof(Vertex, position))
    );

    glEnableVertexAttribArray(0);

    // Attribute 1: color
    glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        static_cast<GLsizei>(sizeof(Vertex)),
        reinterpret_cast<const void*>(
            offsetof(Vertex, color))
    );

    glEnableVertexAttribArray(1);

    glBindVertexArray(0);

    const GLenum error = glGetError();

    if (error != GL_NO_ERROR) {
        LOG_ERROR("TriangleDemo initialization failed: "
                  "OpenGL error 0x%04X",
                  static_cast<unsigned int>(error));

        shutdown();
        return false;
    }

    LOG_INFO("TriangleDemo initialized");
    return true;
}

void TriangleDemo::draw(const Mat4& view_proj, float spin_angle) const {
    if (m_program == 0 || m_vao == 0) {
        return;
    }

    glUseProgram(m_program);
    glBindVertexArray(m_vao);

    const auto draw_object = [this, &view_proj](
        const Mat4& model) {

        const Mat4 mvp = view_proj * model;

        glUniformMatrix4fv(
            m_mvp_location,
            1,
            GL_FALSE,
            &mvp.col[0].x
        );

        glDrawArrays(GL_TRIANGLES, 0, 3);
    };

    // Triangle A: rotate around the world Z axis.
    const Mat4 model_a = mat4_translate({0.0f, 0.0f, 0.5f})
                       * mat4_rotate({0.0f, 0.0f, 1.0f}, spin_angle);

    draw_object(model_a);

    // Triangle B: rotate 90 degrees around X.
    // It stands upright and intersects triangle A.
    const Mat4 model_b = mat4_translate({0.0f, 0.0f, 0.3f})
                       * mat4_rotate({1.0f, 0.0f, 0.0f}, kPi * 0.5f);

    draw_object(model_b);
}

void TriangleDemo::shutdown() {
    // Each resource is checked independently to support
    // cleanup after partially failed initialization
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_vbo != 0) {
        glDeleteBuffers(1, &m_vbo);
        m_vbo = 0;
    }

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}

} // namespace gfx
