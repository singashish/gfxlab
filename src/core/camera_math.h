#pragma once

#include "core/mat4.h"

#include <cmath>

namespace gfx {

enum class ClipSpace {
    CLIP_SPACE_OPENGL,
    CLIP_SPACE_VULKAN
};

// Build a world-to-camera view matrix
// The camera looks down local -Z, with +X right and +Y up
GFX_HD inline Mat4 look_at(Vec3 eye, Vec3 target, Vec3 up) {
    const Vec3 f = normalize(target - eye);
    const Vec3 s = normalize(cross(f, up));
    const Vec3 u = cross(s, f);

    return {{
        {s.x,  u.x, -f.x, 0.0f},
        {s.y,  u.y, -f.y, 0.0f},
        {s.z,  u.z, -f.z, 0.0f},
        {-dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f}
    }};
}

// Build a perspective projection matrix
// fov_y is in radians; near and far are positive distances
GFX_HD inline Mat4 perspective(float fov_y, float aspect, float near_plane,
                               float far_plane, ClipSpace clip_space) {
    const float f = 1.0f / std::tan(fov_y * 0.5f);

    if (clip_space == ClipSpace::CLIP_SPACE_OPENGL) {
        return {{
            {f / aspect, 0.0f, 0.0f,  0.0f},
            {0.0f,       f,    0.0f,  0.0f},
            {0.0f,       0.0f,
             (far_plane + near_plane) / (near_plane - far_plane), -1.0f},
            {0.0f,       0.0f,
             (2.0f * far_plane * near_plane) / (near_plane - far_plane), 0.0f}
        }};
    }

    // Vulkan convention: NDC depth [0, 1], Y flipped
    return {{
        {f / aspect, 0.0f, 0.0f, 0.0f},
        {0.0f,      -f,    0.0f, 0.0f},
        {0.0f,       0.0f, far_plane / (near_plane - far_plane), -1.0f},
        {0.0f,       0.0f, (far_plane * near_plane) / (near_plane - far_plane), 0.0f}
    }};
}

} // namespace gfx
