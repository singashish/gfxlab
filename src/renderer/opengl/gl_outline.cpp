#include "renderer/opengl/gl_outline.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"
#include "renderer/renderer.h"

#include "core/log.h"

namespace gfx {

namespace {

constexpr const char* kOutlineVertexShader = R"GLSL(
#version 460 core

void main() {
    // Vertex IDs 0, 1, 2 generate (-1,-1), (3,-1), (-1,3)
    vec2 ndc =
        vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

// For every pixel that is not part of a marked object: look at the pixels
// within u_radius around it. If one belongs to a marked object (mask > 0)
// and this pixel is not clearly in front of that object, this pixel is
// part of the outline. "Clearly in front" compares distances from the
// camera, so the test means the same near and far from it
constexpr const char* kOutlineFragmentShader = R"GLSL(
#version 460 core

uniform sampler2D u_mask;
uniform sampler2D u_depth;

uniform float u_near;
uniform float u_far;
uniform float u_tolerance;
uniform int u_radius;
uniform vec3 u_selected_color;
uniform vec3 u_active_color;

out vec4 frag_color;

// Window depth (0..1) -> distance from the camera
float distance_of(float d) {
    return u_near * u_far / (u_far - d * (u_far - u_near));
}

void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_mask, 0);

    // Inside a marked object: no outline here
    if (texelFetch(u_mask, p, 0).r > 0.25) {
        discard;
    }

    float here = distance_of(texelFetch(u_depth, p, 0).r);
    float best = 0.0;

    for (int dy = -u_radius; dy <= u_radius; ++dy) {
        for (int dx = -u_radius; dx <= u_radius; ++dx) {
            // A rounded neighbourhood: the corners of the square are
            // too far away
            if (dx * dx + dy * dy > u_radius * u_radius + 1) {
                continue;
            }

            ivec2 q = p + ivec2(dx, dy);

            if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) {
                continue;
            }

            float m = texelFetch(u_mask, q, 0).r;

            if (m > 0.25) {
                float there = distance_of(texelFetch(u_depth, q, 0).r);

                if (here >= there * (1.0 - u_tolerance)) {
                    best = max(best, m);
                }
            }
        }
    }

    if (best <= 0.25) {
        discard;
    }

    frag_color = vec4(best > 0.75 ? u_active_color : u_selected_color, 1.0);
}
)GLSL";

} // namespace

GLOutline::~GLOutline() {
    shutdown();
}

bool GLOutline::init() {
    shutdown();

    m_program = gl_create_program(
        kOutlineVertexShader,
        kOutlineFragmentShader
    );

    if (m_program == 0) {
        LOG_ERROR("failed to create the outline shader program");
        return false;
    }

    m_mask_location = gl_uniform_location(m_program, "u_mask");
    m_depth_location = gl_uniform_location(m_program, "u_depth");
    m_near_location = gl_uniform_location(m_program, "u_near");
    m_far_location = gl_uniform_location(m_program, "u_far");
    m_tolerance_location = gl_uniform_location(m_program, "u_tolerance");
    m_radius_location = gl_uniform_location(m_program, "u_radius");
    m_selected_location =
        gl_uniform_location(m_program, "u_selected_color");
    m_active_location = gl_uniform_location(m_program, "u_active_color");

    if (m_mask_location < 0 ||
        m_depth_location < 0 ||
        m_near_location < 0 ||
        m_far_location < 0 ||
        m_tolerance_location < 0 ||
        m_radius_location < 0 ||
        m_selected_location < 0 ||
        m_active_location < 0) {
        LOG_ERROR("one or more outline uniforms are missing");
        shutdown();
        return false;
    }

    // Core profile needs a VAO for drawing, even without vertex buffers
    glGenVertexArrays(1, &m_vao);

    if (m_vao == 0) {
        LOG_ERROR("failed to create the outline VAO");
        shutdown();
        return false;
    }

    return true;
}

void GLOutline::destroy_targets() {
    if (m_fbo != 0) {
        glDeleteFramebuffers(1, &m_fbo);
        m_fbo = 0;
    }

    if (m_mask_texture != 0) {
        glDeleteTextures(1, &m_mask_texture);
        m_mask_texture = 0;
    }

    if (m_depth_texture != 0) {
        glDeleteTextures(1, &m_depth_texture);
        m_depth_texture = 0;
    }

    m_width = 0;
    m_height = 0;
}

void GLOutline::shutdown() {
    destroy_targets();

    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }
}

bool GLOutline::create_targets(int width, int height) {
    destroy_targets();

    // Texel fetches only, so no filtering; NEAREST also makes the
    // texture complete without mipmaps
    const auto setup = [](unsigned int texture) {
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    };

    glGenTextures(1, &m_mask_texture);
    setup(m_mask_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_R8),
                 width, height, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

    glGenTextures(1, &m_depth_texture);
    setup(m_depth_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_DEPTH_COMPONENT32F),
                 width, height, 0, GL_DEPTH_COMPONENT, GL_FLOAT, nullptr);

    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &m_fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, m_mask_texture, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT,
                           GL_TEXTURE_2D, m_depth_texture, 0);

    const GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);

    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        LOG_ERROR("outline framebuffer is incomplete (status 0x%x)",
                  static_cast<unsigned int>(status));
        destroy_targets();
        return false;
    }

    m_width = width;
    m_height = height;
    return true;
}

bool GLOutline::begin(int width, int height) {
    if (m_program == 0 || width <= 0 || height <= 0) {
        return false;
    }

    if (m_fbo == 0 || width != m_width || height != m_height) {
        if (!create_targets(width, height)) {
            return false;
        }
    }

    glBindFramebuffer(GL_FRAMEBUFFER, m_fbo);
    glViewport(0, 0, width, height);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    return true;
}

void GLOutline::finish(float near_plane, float far_plane,
                       int width, int height) {
    // Back to the window's framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glViewport(0, 0, width, height);

    glUseProgram(m_program);

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, m_mask_texture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, m_depth_texture);

    glUniform1i(m_mask_location, 0);
    glUniform1i(m_depth_location, 1);
    glUniform1f(m_near_location, near_plane);
    glUniform1f(m_far_location, far_plane);
    glUniform1f(m_tolerance_location, kOutlineDepthTolerance);
    glUniform1i(m_radius_location, kOutlineRadius);

    const Vec3 selected = edge_color(Highlight::HIGHLIGHT_SELECTED);
    const Vec3 active = edge_color(Highlight::HIGHLIGHT_ACTIVE);

    glUniform3f(m_selected_location, selected.x, selected.y, selected.z);
    glUniform3f(m_active_location, active.x, active.y, active.z);

    // The outline replaces what is under it: no depth test, no blending
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);

    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, 0);
}

} // namespace gfx
