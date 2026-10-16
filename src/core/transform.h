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

} // namespace gfx
