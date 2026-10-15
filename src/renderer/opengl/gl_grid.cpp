#include "renderer/opengl/gl_grid.h"
#include "renderer/opengl/gl_api.h"
#include "renderer/opengl/gl_shader.h"

#include "core/log.h"

namespace gfx {

namespace {

constexpr const char* kGridVertexShader = R"GLSL(
#version 460 core

uniform vec3 u_forward;
uniform vec3 u_right;
uniform vec3 u_up;

out vec3 v_ray;

void main() {
    // Vertex IDs 0, 1, 2 generate (-1,-1), (3,-1), (-1,3).
    vec2 ndc =
        vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;

    v_ray = u_forward + ndc.x * u_right + ndc.y * u_up;
    gl_Position = vec4(ndc, 0.0, 1.0);
}
)GLSL";

constexpr const char* kGridFragmentShader = R"GLSL(
#version 460 core

uniform mat4 u_view_proj;
uniform vec3 u_eye;
uniform float u_fade_end;

in vec3 v_ray;
out vec4 frag_color;

// Coverage of a line of width_px, given the pixel distance to its centre.
float aa_line(float dist_px, float width_px) {
    return clamp(0.5 * width_px + 0.5 - dist_px, 0.0, 1.0);
}

// One grid level: a line every spacing world units.
// The per-family fade is only an anti-moire guard.
float grid(vec2 p, float spacing, float width_px) {
    vec2 c = p / spacing;
    vec2 w = max(fwidth(c), vec2(1e-6));

    vec2 dist_px = abs(fract(c - 0.5) - 0.5) / w;
    vec2 line = vec2(aa_line(dist_px.x, width_px),
                     aa_line(dist_px.y, width_px));

    vec2 lod = smoothstep(vec2(2.0), vec2(8.0), 1.0 / w);

    return max(line.x * lod.x, line.y * lod.y);
}

// Standard "over" compositing for non-premultiplied colour.
vec4 layer(vec4 dst, vec3 src_rgb, float src_a) {
    float a = src_a + dst.a * (1.0 - src_a);

    vec3 rgb =
        (src_rgb * src_a + dst.rgb * dst.a * (1.0 - src_a))
        / max(a, 1e-6);

    return vec4(rgb, a);
}

void main() {
    float t = -u_eye.z / v_ray.z;
    vec3 p = u_eye + t * v_ray;

    float minor = grid(p.xy, 1.0, 1.0);
    float major = grid(p.xy, 10.0, 1.0);

    vec2 fw = max(fwidth(p.xy), vec2(1e-6));
    float axis_x = aa_line(abs(p.y) / fw.y, 1.25);
    float axis_y = aa_line(abs(p.x) / fw.x, 1.25);

    vec4 c = vec4(0.0);
    c = layer(c, vec3(0.55), 0.10 * minor);
    c = layer(c, vec3(0.65), 0.24 * major);
    c = layer(c, vec3(0.85, 0.25, 0.30), 0.75 * axis_x);
    c = layer(c, vec3(0.45, 0.78, 0.25), 0.75 * axis_y);

    // Distance fade, relative to the zoom: gradual from 20% of u_fade_end,
    // gone at u_fade_end. This ends well before the horizon.
    float dist = t * length(v_ray);
    c.a *= 1.0 - smoothstep(0.2 * u_fade_end, u_fade_end, dist);

    bool valid =
        t > 0.0 &&
        abs(v_ray.z) > 1e-6;

    vec4 clip = u_view_proj * vec4(p, 1.0);

    valid = valid && abs(clip.w) > 1e-6;

    float ndc_z = clip.z / clip.w;

    if (!valid ||
        ndc_z < -1.0 ||
        ndc_z > 1.0 ||
        c.a < 0.004) {
        discard;
    }

    gl_FragDepth = ndc_z * 0.5 + 0.5;
    frag_color = c;
}
)GLSL";

} // namespace

GridRenderer::~GridRenderer() {
    shutdown();
}

bool GridRenderer::init() {
    shutdown();

    m_program = gl_create_program(
        kGridVertexShader,
        kGridFragmentShader
    );

    if (m_program == 0) {
        LOG_ERROR("Failed to create grid shader program");
        return false;
    }

    m_view_proj_location = gl_uniform_location(m_program, "u_view_proj");
    m_eye_location = gl_uniform_location(m_program, "u_eye");
    m_forward_location = gl_uniform_location(m_program, "u_forward");
    m_right_location = gl_uniform_location(m_program, "u_right");
    m_up_location = gl_uniform_location(m_program, "u_up");
    m_fade_end = gl_uniform_location(m_program, "u_fade_end");

    if (m_view_proj_location < 0 ||
        m_eye_location < 0 ||
        m_forward_location < 0 ||
        m_right_location < 0 ||
        m_up_location < 0 ||
        m_fade_end < 0) {
        LOG_ERROR("One or more grid uniforms are missing");
        shutdown();
        return false;
    }

    // Core profile requires a VAO for drawing, even without vertex buffers.
    glGenVertexArrays(1, &m_vao);

    if (m_vao == 0) {
        LOG_ERROR("Failed to create grid VAO");
        shutdown();
        return false;
    }

    return true;
}

void GridRenderer::draw(const GridParams& params) const {
    if (m_program == 0 || m_vao == 0) {
        return;
    }

    glUseProgram(m_program);

    glUniformMatrix4fv(
        m_view_proj_location,
        1,
        GL_FALSE,
        &params.view_proj.col[0].x
    );

    glUniform3f(m_eye_location, params.eye.x, params.eye.y, params.eye.z);
    glUniform3f(m_forward_location, params.forward.x, params.forward.y, params.forward.z);
    glUniform3f(m_right_location, params.right.x, params.right.y, params.right.z);
    glUniform3f(m_up_location, params.up.x, params.up.y, params.up.z);
    glUniform1f(m_fade_end, params.fade_end);

    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glDepthMask(GL_FALSE);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

void GridRenderer::shutdown() {
    if (m_vao != 0) {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_program != 0) {
        glDeleteProgram(m_program);
        m_program = 0;
    }

    m_view_proj_location = -1;
    m_eye_location = -1;
    m_forward_location = -1;
    m_right_location = -1;
    m_up_location = -1;
    m_fade_end = -1;
}

} // namespace gfx
