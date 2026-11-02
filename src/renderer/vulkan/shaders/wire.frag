#version 450

layout(location = 0) out vec4 out_color;

// Same declaration as in mesh.vert, mesh.frag and wire.vert: one pipeline
// layout and one 128 byte push constant range. This shader only reads the
// line colour (base_color.rgb)
layout(push_constant) uniform Push {
    mat4 model;        // offset 0
    vec4 normal_col0;  // offset 64
    vec4 normal_col1;  // offset 80
    vec4 normal_col2;  // offset 96
    vec4 base_color;   // offset 112 (rgb used)
} pc;

void main() {
    out_color = vec4(pc.base_color.rgb, 1.0);
}
