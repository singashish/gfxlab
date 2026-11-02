#version 450

void main() {
    // Vertex IDs 0, 1, 2 generate (-1,-1), (3,-1), (-1,3): one triangle
    // that covers the screen
    vec2 ndc =
        vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2) * 2.0 - 1.0;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
