#include "renderer/opengl/gl_presenter.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"
#include "core/log.h"

namespace gfx {

namespace {

constexpr const char* kPresenterVertexShader = R"GLSL(
#version 460 core

out vec2 uv;

void main() {
    vec2 ndc =
        vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;

    uv = ndc * 0.5 + 0.5;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

constexpr const char* kPresenterFragmentShader = R"GLSL(
#version 460 core

uniform sampler2D u_texture;

in vec2 uv;
out vec4 frag_color;

void main() {
    frag_color = texture(u_texture, vec2(uv.x, 1.0 - uv.y));
}
)GLSL";

} // namespace

GLPresenter::~GLPresenter() {
    shutdown();
}

bool GLPresenter::init() {
    shutdown();

    m_program = gl_create_program(
        kPresenterVertexShader,
        kPresenterFragmentShader
    );

    if (m_program == 0) {
        LOG_ERROR("Failed to create CUDA presenter shader program");
        return false;
    }

    m_sampler_location =
        gl_uniform_location(m_program, "u_texture");

    if (m_sampler_location < 0) {
        LOG_ERROR("CUDA presenter sampler uniform is missing");
        shutdown();
        return false;
    }

    glGenVertexArrays(1, &m_vao);

    if (m_vao == 0) {
        LOG_ERROR("Failed to create CUDA presenter VAO");
        shutdown();
        return false;
    }

    return true;
}

bool GLPresenter::resize(int width, int height) {
    if (width <= 0 || height <= 0) {
        return false;
    }

    if (m_texture != 0) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }

    glGenTextures(1, &m_texture);

    if (m_texture == 0) {
        LOG_ERROR("Failed to create CUDA presenter texture");
        return false;
    }

    glBindTexture(GL_TEXTURE_2D, m_texture);

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MIN_FILTER,
        static_cast<GLint>(GL_NEAREST)
    );

    glTexParameteri(
        GL_TEXTURE_2D,
        GL_TEXTURE_MAG_FILTER,
        static_cast<GLint>(GL_NEAREST)
    );

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        static_cast<GLint>(GL_RGBA8),
        width,
        height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        nullptr
    );

    glBindTexture(GL_TEXTURE_2D, 0);

    const GLenum error = glGetError();

    if (error != GL_NO_ERROR) {
        LOG_ERROR(
            "Failed to allocate CUDA presenter texture: GL error 0x%04X",
            error
        );

        glDeleteTextures(1, &m_texture);
        m_texture = 0;
        return false;
    }

    return true;
}

unsigned int GLPresenter::texture() const {
    return m_texture;
}

void GLPresenter::draw(int width, int height) {
    if (width <= 0 || height <= 0 ||
        m_texture == 0 || m_program == 0 || m_vao == 0) {
        return;
    }

    glViewport(0, 0, width, height);

    glUseProgram(m_program);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glUniform1i(m_sampler_location, 0);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glBindVertexArray(0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);
}

void GLPresenter::shutdown() {
    if (m_texture != 0) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }

    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }

    m_sampler_location = -1;
}

} // namespace gfx
