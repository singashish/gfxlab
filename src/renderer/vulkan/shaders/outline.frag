#version 450

// texelFetch / textureSize on a texture2D that has no sampler
#extension GL_EXT_samplerless_texture_functions : require

// The mask of the marked objects (0 unmarked, 0.5 selected, 1 active) and
// the depth buffer that was rendered with it (it holds the whole scene)
layout(set = 0, binding = 0) uniform texture2D u_mask;
layout(set = 0, binding = 1) uniform texture2D u_depth;

layout(push_constant) uniform Push {
    vec4 selected_color;  // rgb
    vec4 active_color;    // rgb
    vec4 params;          // x = near, y = far, z = depth tolerance,
                          // w = radius in pixels
} pc;

layout(location = 0) out vec4 out_color;

// Window depth (0..1) -> distance from the camera
float distance_of(float d) {
    float n = pc.params.x;
    float f = pc.params.y;
    return n * f / (f - d * (f - n));
}

// For every pixel that is not part of a marked object: look at the pixels
// within the radius around it. If one belongs to a marked object and this
// pixel is not clearly in front of that object, this pixel is part of the
// outline
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    ivec2 size = textureSize(u_mask, 0);
    int radius = int(pc.params.w);

    // Inside a marked object: no outline here
    if (texelFetch(u_mask, p, 0).r > 0.25) {
        discard;
    }

    float here = distance_of(texelFetch(u_depth, p, 0).r);
    float best = 0.0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            // A rounded neighbourhood: the corners of the square are
            // too far away
            if (dx * dx + dy * dy > radius * radius + 1) {
                continue;
            }

            ivec2 q = p + ivec2(dx, dy);

            if (q.x < 0 || q.y < 0 || q.x >= size.x || q.y >= size.y) {
                continue;
            }

            float m = texelFetch(u_mask, q, 0).r;

            if (m > 0.25) {
                float there = distance_of(texelFetch(u_depth, q, 0).r);

                if (here >= there * (1.0 - pc.params.z)) {
                    best = max(best, m);
                }
            }
        }
    }

    if (best <= 0.25) {
        discard;
    }

    out_color = vec4(best > 0.75 ? pc.active_color.rgb
                                 : pc.selected_color.rgb, 1.0);
}
