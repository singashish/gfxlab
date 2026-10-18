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

layout(location = 0) in vec3 v_ray;
layout(location = 0) out vec4 out_color;

// Coverage of a line of width_px, given pixel distance to its centre.
float aa_line(float dist_px, float width_px) {
    return clamp(0.5 * width_px + 0.5 - dist_px, 0.0, 1.0);
}

// One grid level: a line every spacing world units.
float grid(vec2 p, float spacing, float width_px) {
    vec2 c = p / spacing;
    vec2 w = max(fwidth(c), vec2(1e-6));

    vec2 dist_px = abs(fract(c - 0.5) - 0.5) / w;
    vec2 line = vec2(
        aa_line(dist_px.x, width_px),
        aa_line(dist_px.y, width_px));

    vec2 lod = smoothstep(vec2(2.0), vec2(8.0), 1.0 / w);

    return max(line.x * lod.x, line.y * lod.y);
}

// Standard "over" compositing for non-premultiplied colour.
vec4 layer(vec4 dst, vec3 src_rgb, float src_a) {
    float a = src_a + dst.a * (1.0 - src_a);

    vec3 rgb =
        (src_rgb * src_a + dst.rgb * dst.a * (1.0 - src_a))
        / max(a, 1e-6);

    return vec4(rgb, a);
}

void main() {
    float t = -frame.eye.z / v_ray.z;
    vec3 p = frame.eye.xyz + t * v_ray;

    float minor = grid(p.xy, 1.0, 1.0);
    float major = grid(p.xy, 10.0, 1.0);

    vec2 fw = max(fwidth(p.xy), vec2(1e-6));
    float axis_x = aa_line(abs(p.y) / fw.y, 1.25);
    float axis_y = aa_line(abs(p.x) / fw.x, 1.25);

    vec4 c = vec4(0.0);
    c = layer(c, vec3(0.55), 0.10 * minor);
    c = layer(c, vec3(0.65), 0.24 * major);
    c = layer(c, vec3(0.85, 0.25, 0.30), 0.75 * axis_x);
    c = layer(c, vec3(0.45, 0.78, 0.25), 0.75 * axis_y);

    // Fade gradually from 20% of fade_end to fade_end.
    float dist = t * length(v_ray);
    c.a *= 1.0 - smoothstep(
        0.2 * frame.params.x,
        frame.params.x,
        dist);

    bool valid = t > 0.0 && abs(v_ray.z) > 1e-6;

    vec4 clip = frame.view_proj * vec4(p, 1.0);
    valid = valid && abs(clip.w) > 1e-6;

    float ndc_z = clip.z / clip.w;

    if (!valid ||
        ndc_z < 0.0 ||
        ndc_z > 1.0 ||
        c.a < 0.004) {
        discard;
    }

    // Vulkan clip-space depth is already in [0, 1].
    gl_FragDepth = ndc_z;
    out_color = c;
}
