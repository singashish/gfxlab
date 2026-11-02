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

// Same declaration as in mesh.vert: the mask pipeline shares the mesh
// pipeline layout and its 128 byte push constant range
layout(push_constant) uniform Push {
    mat4 model;        // offset 0
    vec4 normal_col0;  // offset 64
    vec4 normal_col1;  // offset 80
    vec4 normal_col2;  // offset 96
    vec4 base_color;   // offset 112: x = mask value (mask.frag)
} pc;

void main() {
    gl_Position = frame.view_proj * pc.model * vec4(in_pos, 1.0);
}
