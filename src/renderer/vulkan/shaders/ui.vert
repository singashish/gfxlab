#version 450

// UI vertices are in window pixels with Y pointing down
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_uv;
layout(location = 2) in vec4 a_color;   // R8G8B8A8_UNORM -> 0..1

layout(push_constant) uniform Push {
    vec2 viewport;   // framebuffer size in pixels
} push;

layout(location = 0) out vec2 v_uv;
layout(location = 1) out vec4 v_color;

void main() {
    v_uv = a_uv;
    v_color = a_color;

    // Vulkan's NDC already has Y pointing down, like the UI, so unlike
    // the OpenGL version there is no flip
    vec2 ndc = a_position / push.viewport * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
