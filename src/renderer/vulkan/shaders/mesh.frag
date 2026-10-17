#version 450

layout(location = 0) in vec3 v_normal;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(normalize(v_normal) * 0.5 + 0.5, 1.0);
}
