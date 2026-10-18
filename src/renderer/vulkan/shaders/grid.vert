#version 450

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 view_proj;
    vec4 eye;
    vec4 forward;
    vec4 right;
    vec4 up;
    vec4 params;
} frame;

layout(location = 0) out vec3 v_ray;

void main() {
    // Generate a fullscreen triangle from the vertex index.
    vec2 ndc =
        vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2)
        * 2.0 - 1.0;

    // Vulkan's NDC Y convention requires the sign flip here.
    // Keep the CPU-side up vector in the same convention as OpenGL.
    v_ray = frame.forward.xyz
          + ndc.x * frame.right.xyz
          - ndc.y * frame.up.xyz;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
