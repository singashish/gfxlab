#version 450

layout(location = 0) in vec3 in_pos;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;

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

layout(location = 0) out vec3 v_view_pos;
layout(location = 1) out vec3 v_view_normal;
layout(location = 2) out vec2 v_uv;

void main() {
    vec4 world = pc.model * vec4(in_pos, 1.0);

    v_view_pos = (frame.view * world).xyz;

    // The upper 3x3 of the normal matrix, rebuilt from its three columns
    mat3 normal_matrix =
        mat3(pc.normal_col0.xyz, pc.normal_col1.xyz, pc.normal_col2.xyz);

    v_view_normal = mat3(frame.view) * (normal_matrix * in_normal);
    v_uv = in_uv;

    gl_Position = frame.view_proj * world;
}
