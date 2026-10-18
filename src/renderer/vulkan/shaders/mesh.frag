#version 450

layout(location = 0) in vec3 v_view_pos;
layout(location = 1) in vec3 v_view_normal;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 view_proj;
    vec4 eye;
    vec4 forward;
    vec4 right;
    vec4 up;
    vec4 params;
} frame;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = normalize(v_view_normal);

    // Light both front-facing and back-facing surfaces.
    if (!gl_FrontFacing) {
        n = -n;
    }

    // Direction from the surface toward the camera.
    vec3 v = normalize(-v_view_pos);

    // Key light positioned up-left of the camera.
    vec3 key_dir = normalize(vec3(-0.5, 0.6, 0.7));
    float diff = max(dot(n, key_dir), 0.0);

    // World +Z expressed in view space.
    vec3 up_view = mat3(frame.view)[2];
    float hemi = dot(n, up_view) * 0.5 + 0.5;
    vec3 ambient = mix(
        vec3(0.10, 0.10, 0.12),
        vec3(0.30, 0.32, 0.36),
        hemi
    );

    // Blinn-Phong specular highlight.
    vec3 h = normalize(key_dir + v);
    float spec = diff > 0.0
        ? pow(max(dot(n, h), 0.0), 32.0) * 0.25
        : 0.0;

    vec3 base = vec3(0.72, 0.74, 0.78);
    vec3 color = base * (ambient + 0.75 * diff) + vec3(spec);

    out_color = vec4(min(color, vec3(1.0)), 1.0);
}
