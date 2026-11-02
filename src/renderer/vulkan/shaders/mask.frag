#version 450

layout(location = 0) out vec4 out_color;

// Same declaration as in mesh.vert. Only the mask value is read
layout(push_constant) uniform Push {
    mat4 model;        // offset 0
    vec4 normal_col0;  // offset 64
    vec4 normal_col1;  // offset 80
    vec4 normal_col2;  // offset 96
    vec4 base_color;   // offset 112: x = 0 unmarked, 0.5 selected, 1 active
} pc;

void main() {
    out_color = vec4(pc.base_color.x, 0.0, 0.0, 1.0);
}
