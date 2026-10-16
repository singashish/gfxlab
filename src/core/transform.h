#pragma once

#include "core/mat4.h"

#include <algorithm>
#include <cmath>

namespace gfx {

struct Transform {
    Vec3 position = {0.0f, 0.0f, 0.0f};
    Vec3 rotation = {0.0f, 0.0f, 0.0f}; // Euler angles, radians
    Vec3 scale = {1.0f, 1.0f, 1.0f};

    Mat4 model_matrix() const {
        const Mat4 rx = mat4_rotate({1.0f, 0.0f, 0.0f}, rotation.x);
        const Mat4 ry = mat4_rotate({0.0f, 1.0f, 0.0f}, rotation.y);
        const Mat4 rz = mat4_rotate({0.0f, 0.0f, 1.0f}, rotation.z);

        // Blender-style XYZ Euler rotation:
        // X first, then Y, then Z for column vectors.
        const Mat4 r = rz * ry * rx;

        return mat4_translate(position) * r * mat4_scale(scale);
    }

    Mat4 normal_matrix() const {
        constexpr float kMinScale = 1.0e-6f;

        const Mat4 rx = mat4_rotate({1.0f, 0.0f, 0.0f}, rotation.x);
        const Mat4 ry = mat4_rotate({0.0f, 1.0f, 0.0f}, rotation.y);
        const Mat4 rz = mat4_rotate({0.0f, 0.0f, 1.0f}, rotation.z);

        const Mat4 r = rz * ry * rx;

        // Clamp the magnitude before inversion. Preserve the sign of
        // negative scale values, while treating zero as positive.
        const auto safe_scale = [](float s) {
            const float sign = s < 0.0f ? -1.0f : 1.0f;
            return sign * std::max(std::fabs(s), kMinScale);
        };

        const Vec3 inverse_scale = {
            1.0f / safe_scale(scale.x),
            1.0f / safe_scale(scale.y),
            1.0f / safe_scale(scale.z)
        };

        return r * mat4_scale(inverse_scale);
    }
};

} // namespace gfx
