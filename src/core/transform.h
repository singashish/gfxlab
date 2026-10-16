#pragma once

#include "core/mat4.h"

#include <algorithm>
#include <cmath>

namespace gfx {

struct Transform {
    Vec3 position = {0.0f, 0.0f, 0.0f};
    Vec3 rotation = {0.0f, 0.0f, 0.0f}; // Euler angles in radians
    Vec3 scale = {1.0f, 1.0f, 1.0f};

    Mat4 rotation_matrix() const {
        return mat4_rotate({0.0f, 0.0f, 1.0f}, rotation.z)
             * mat4_rotate({0.0f, 1.0f, 0.0f}, rotation.y)
             * mat4_rotate({1.0f, 0.0f, 0.0f}, rotation.x);
    }

    Mat4 model_matrix() const {
        return mat4_translate(position)
             * rotation_matrix()
             * mat4_scale(scale);
    }

    Mat4 normal_matrix() const {
        constexpr float kMinScale = 1.0e-6f;

        const auto safe_scale = [](float s) {
            const float sign = s < 0.0f ? -1.0f : 1.0f;
            return sign * std::max(std::fabs(s), kMinScale);
        };

        const Vec3 inverse_scale = {
            1.0f / safe_scale(scale.x),
            1.0f / safe_scale(scale.y),
            1.0f / safe_scale(scale.z)
        };

        return rotation_matrix() * mat4_scale(inverse_scale);
    }
};

inline Vec3 euler_from_matrix(const Mat4& m) {
    // Mat4 stores columns. Access matrix entries as row, column
    const float r00 = m.col[0].x;
    const float r01 = m.col[1].x;
    const float r10 = m.col[0].y;
    const float r11 = m.col[1].y;
    const float r20 = m.col[0].z;
    const float r21 = m.col[1].z;
    const float r22 = m.col[2].z;

    const float sin_y = std::max(-1.0f, std::min(-r20, 1.0f));
    const float ry = std::asin(sin_y);

    float rx;
    float rz;

    if (std::fabs(r20) > 0.99999f) {
        // Gimbal lock: choose rx = 0 as a valid representative
        rx = 0.0f;
        rz = std::atan2(-r01, r11);
    } else {
        rx = std::atan2(r21, r22);
        rz = std::atan2(r10, r00);
    }

    return {rx, ry, rz};
}

} // namespace gfx
