#version 450

layout(location = 0) in vec3 v_view_pos;
layout(location = 1) in vec3 v_view_normal;
layout(location = 2) in vec2 v_uv;

layout(set = 0, binding = 0) uniform Frame {
    mat4 view;
    mat4 view_proj;
    vec4 eye;
    vec4 forward;
    vec4 right;
    vec4 up;
    vec4 params;
} frame;

// Set 1: the base-colour texture of the draw call (a 1x1 white texture
// for surfaces without one, so no branch is needed for that case)
layout(set = 1, binding = 0) uniform sampler2D u_base_texture;

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

    // Colours: this shading works directly in display values, with no
    // linear / sRGB conversion. Textures are uploaded as plain RGBA8
    // (UNORM, not sRGB) and the base colour factor is used as given,
    // which keeps the grey look and the three backends consistent.
    //
    // Our UVs have (0, 0) at the bottom-left of the image and v pointing
    // up, but the image rows are stored top row first, so row 0 of the
    // texture is v = 1: sample at 1 - v.
    //
    // The texture is sampled unconditionally (not inside the ?: branch)
    // so the implicit derivatives used for mip selection never depend on
    // control flow; params.y is 1 when materials are shown, else 0
    vec3 texel = texture(u_base_texture, vec2(v_uv.x, 1.0 - v_uv.y)).rgb;
    vec3 base = frame.params.y > 0.5
        ? pc.base_color.rgb * texel
        : vec3(0.72, 0.74, 0.78);
    vec3 color = base * (ambient + 0.75 * diff) + vec3(spec);

    out_color = vec4(min(color, vec3(1.0)), 1.0);
}
