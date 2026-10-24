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

// Same declaration in mesh.vert, mesh.frag and wire.vert: they share one
// pipeline layout and one 128 byte push constant range. The normal matrix
// is sent as three vec4 columns (its upper 3x3, padded) so the material
// colour fits in the last 16 bytes
layout(push_constant) uniform Push {
    mat4 model;        // offset 0
    vec4 normal_col0;  // offset 64
    vec4 normal_col1;  // offset 80
    vec4 normal_col2;  // offset 96
    vec4 base_color;   // offset 112 (rgb used)
} pc;

void main() {
    gl_Position = frame.view_proj * pc.model * vec4(in_pos, 1.0);
}
