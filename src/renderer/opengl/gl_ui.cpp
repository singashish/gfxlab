#include "renderer/opengl/gl_ui.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"

#include "core/log.h"

#include <cstddef>

namespace gfx {

namespace {

constexpr const char* kUiVertexShader = R"GLSL(
#version 460 core

layout(location = 0) in vec2 a_position;  // window pixels, Y down
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;

uniform vec2 u_viewport;

out vec2 v_uv;
out vec4 v_color;

void main() {
    v_uv = a_uv;
    v_color = a_color;

    // Pixels to NDC. OpenGL's NDC has Y up, the UI has Y down
    vec2 ndc = vec2(a_position.x / u_viewport.x * 2.0 - 1.0,
                    1.0 - a_position.y / u_viewport.y * 2.0);

    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

constexpr const char* kUiFragmentShader = R"GLSL(
#version 460 core

uniform sampler2D u_atlas;

in vec2 v_uv;
in vec4 v_color;

out vec4 frag_color;

void main() {
    // The atlas stores coverage in its only (red) channel
    frag_color = vec4(v_color.rgb, v_color.a * texture(u_atlas, v_uv).r);
}
)GLSL";

} // namespace

GLUiRenderer::~GLUiRenderer() {
    shutdown();
}

bool GLUiRenderer::init() {
    shutdown();

    m_program = gl_create_program(kUiVertexShader, kUiFragmentShader);

    if (m_program == 0) {
        LOG_ERROR("Failed to create UI shader program");
        return false;
    }

    m_viewport_location = gl_uniform_location(m_program, "u_viewport");
    m_atlas_location = gl_uniform_location(m_program, "u_atlas");

    if (m_viewport_location < 0 || m_atlas_location < 0) {
        LOG_ERROR("One or more UI uniforms are missing");
        shutdown();
        return false;
    }

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vertex_buffer);
    glGenBuffers(1, &m_index_buffer);

    if (m_vao == 0 || m_vertex_buffer == 0 || m_index_buffer == 0) {
        LOG_ERROR("Failed to create UI buffers");
        shutdown();
        return false;
    }

    // The vertex layout and the index buffer binding are stored in the
    // VAO. The buffers' contents are replaced every frame
    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);

    constexpr GLsizei stride = static_cast<GLsizei>(sizeof(UiVertex));

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(
        0, 2, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void*>(offsetof(UiVertex, x)));

    glEnableVertexAttribArray(1);
    glVertexAttribPointer(
        1, 2, GL_FLOAT, GL_FALSE, stride,
        reinterpret_cast<const void*>(offsetof(UiVertex, u)));

    // Four bytes R, G, B, A, read as 0..1
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(
        2, 4, GL_UNSIGNED_BYTE, GL_TRUE, stride,
        reinterpret_cast<const void*>(offsetof(UiVertex, color)));

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_index_buffer);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return true;
}

bool GLUiRenderer::upload_font(const FontAtlas& font) {
    if (m_texture != 0) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }

    m_font = nullptr;

    glGenTextures(1, &m_texture);

    if (m_texture == 0) {
        LOG_ERROR("Failed to create the UI font texture");
        return false;
    }

    glBindTexture(GL_TEXTURE_2D, m_texture);

    // Glyphs are drawn at their exact size on whole pixels, so each
    // pixel samples one texel centre: nearest filtering is exact
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    static_cast<GLint>(GL_NEAREST));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    static_cast<GLint>(GL_NEAREST));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    static_cast<GLint>(GL_CLAMP_TO_EDGE));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    static_cast<GLint>(GL_CLAMP_TO_EDGE));

    // One byte per texel. kFontAtlasWidth is a multiple of 4, so the
    // default 4-byte row alignment matches the data
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_R8),
                 font.width, font.height, 0,
                 GL_RED, GL_UNSIGNED_BYTE, font.alpha.data());

    glBindTexture(GL_TEXTURE_2D, 0);

    const GLenum error = glGetError();

    if (error != GL_NO_ERROR) {
        LOG_ERROR("Failed to upload the UI font texture: GL error 0x%04X",
                  error);
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
        return false;
    }

    m_font = &font;
    return true;
}

void GLUiRenderer::draw(const UiDrawList& list, int width, int height) {
    const FontAtlas* font = list.font();

    if (list.empty() || font == nullptr || width <= 0 || height <= 0 ||
        m_program == 0) {
        return;
    }

    if (font != m_font && !upload_font(*font)) {
        return;
    }

    const std::vector<UiVertex>& vertices = list.vertices();
    const std::vector<std::uint32_t>& indices = list.indices();

    glViewport(0, 0, width, height);

    // The UI is flat and drawn last: no depth, blended over the scene
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(m_program);
    glUniform2f(m_viewport_location,
                static_cast<float>(width),
                static_cast<float>(height));

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_texture);
    glUniform1i(m_atlas_location, 0);

    glBindVertexArray(m_vao);

    // Replacing the whole buffer every frame lets the driver hand out
    // fresh memory instead of waiting for the previous frame to finish
    glBindBuffer(GL_ARRAY_BUFFER, m_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(vertices.size() * sizeof(UiVertex)),
                 vertices.data(), GL_STREAM_DRAW);

    glBufferData(GL_ELEMENT_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(indices.size() *
                                         sizeof(std::uint32_t)),
                 indices.data(), GL_STREAM_DRAW);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(indices.size()),
                   GL_UNSIGNED_INT, nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindTexture(GL_TEXTURE_2D, 0);
    glUseProgram(0);

    glDisable(GL_BLEND);
    glEnable(GL_DEPTH_TEST);
}

void GLUiRenderer::shutdown() {
    if (m_texture != 0) {
        glDeleteTextures(1, &m_texture);
        m_texture = 0;
    }

    if (m_index_buffer != 0) {
        glDeleteBuffers(1, &m_index_buffer);
        m_index_buffer = 0;
    }

    if (m_vertex_buffer != 0) {
        glDeleteBuffers(1, &m_vertex_buffer);
        m_vertex_buffer = 0;
    }

    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }

    m_viewport_location = -1;
    m_atlas_location = -1;
    m_font = nullptr;
}

} // namespace gfx
