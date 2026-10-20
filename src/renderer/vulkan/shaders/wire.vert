#version 450

layout(location = 0) in vec3 in_pos;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 view_proj;
    vec4 eye;
    vec4 forward;
    vec4 right;
    vec4 up;
    vec4 params;
} frame;

layout(push_constant) uniform Push {
    mat4 model;
    mat4 normal_model;
} pc;

void main() {
    gl_Position = frame.view_proj * pc.model * vec4(in_pos, 1.0);
}
