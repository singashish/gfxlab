#version 450

layout(location = 0) out vec3 out_color;

const vec2 positions[3] = vec2[](
    vec2( 0.0, -0.6), // Top: red
    vec2( 0.6,  0.6), // Bottom-right: green
    vec2(-0.6,  0.6)  // Bottom-left: blue
);

const vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    out_color = colors[gl_VertexIndex];
}
