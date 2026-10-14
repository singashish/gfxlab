#pragma once

#include "core/vec3.h"
#include "core/camera_math.h"
#include "core/math_util.h"

#include <algorithm>
#include <cmath>

namespace gfx {

struct OrbitCamera {
    Vec3 target = {0.0f, 0.0f, 0.0f};

    float distance = 5.0f;
    float yaw = to_radians(-55.0f);
    float pitch = to_radians(25.0f);

    Vec3 eye() const {
        const float cos_pitch = std::cos(pitch);

        const Vec3 offset = {
            distance * cos_pitch * std::cos(yaw),
            distance * cos_pitch * std::sin(yaw),
            distance * std::sin(pitch)
        };

        return target + offset;
    }

    Mat4 view() const {
        return look_at(eye(), target, {0.0f, 0.0f, 1.0f});
    }

    void orbit(float dx_pixels, float dy_pixels) {
        constexpr float kOrbitSensitivity = 0.005f;
        constexpr float kPitchLimit =
            kPi * 0.5f - 0.0175f;

        yaw -= dx_pixels * kOrbitSensitivity;
        pitch += dy_pixels * kOrbitSensitivity;

        pitch = std::clamp(pitch, -kPitchLimit, kPitchLimit);
    }

    void pan(float dx_pixels, float dy_pixels, float world_per_pixel_value) {
        const CameraBasis b = basis();

        target = target
               - b.right * (dx_pixels * world_per_pixel_value)
               + b.up * (dy_pixels * world_per_pixel_value);
    }

    void zoom(float notches) {
        distance *= std::pow(0.9f, notches);
        distance = std::clamp(distance, 0.2f, 500.0f);
    }

    float world_per_pixel(float fov_y, float viewport_height) const {
        if (viewport_height <= 0.0f) {
            return 0.0f;
        }

        return 2.0f * distance * std::tan(fov_y * 0.5f) / viewport_height;
    }

    CameraBasis basis() const {
        const Vec3 forward = normalize(target - eye());
        const Vec3 world_up = {0.0f, 0.0f, 1.0f};

        const Vec3 right = normalize(cross(forward, world_up));
        const Vec3 up = cross(right, forward);

        return {right, up, forward};
    }
};

} // namespace gfx
