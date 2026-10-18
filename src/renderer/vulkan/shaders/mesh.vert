#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 view_proj;
} frame;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normal_model;
} pc;

layout(location = 0) out vec3 v_view_pos;
layout(location = 1) out vec3 v_view_normal;

void main() {
    vec4 world = pc.model * vec4(in_pos, 1.0);

    v_view_pos = (frame.view * world).xyz;
    v_view_normal =
        mat3(frame.view) * (mat3(pc.normal_model) * in_normal);

    gl_Position = frame.view_proj * world;
}
