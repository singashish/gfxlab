#version 450

layout(location = 0) out vec3 out_color;

const vec3 positions[6] = vec3[](
    // Near triangle: RGB
    vec3( 0.0, -0.6, 0.3),
    vec3( 0.6,  0.6, 0.3),
    vec3(-0.6,  0.6, 0.3),

    // Far triangle: larger, yellow
    vec3( 0.0, -0.9, 0.7),
    vec3( 0.9,  0.9, 0.7),
    vec3(-0.9,  0.9, 0.7)
);

const vec3 colors[6] = vec3[](
    // Near triangle
    vec3(1.0, 0.0, 0.0),
    vec3(0.0, 1.0, 0.0),
    vec3(0.0, 0.0, 1.0),

    // Far triangle
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0),
    vec3(1.0, 1.0, 0.0)
);

void main() {
    vec3 position = positions[gl_VertexIndex];

    gl_Position = vec4(position, 1.0);
    out_color = colors[gl_VertexIndex];
}
