#pragma once

#include "core/common.h"
#include "core/vec3.h"
#include "core/vec4.h"

#include <cmath>

namespace gfx {

struct Mat4 {
    Vec4 col[4];
};

static_assert(sizeof(Mat4) == 64,
              "Mat4 must be tightly packed for GPU upload");

// Identity matrix
GFX_HD inline Mat4 mat4_identity() {
    return {{
        {1.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};
}

// Matrix multiplied by a column vector
GFX_HD inline Vec4 operator*(Mat4 m, Vec4 v) {
    return m.col[0] * v.x
         + m.col[1] * v.y
         + m.col[2] * v.z
         + m.col[3] * v.w;
}

// Matrix multiplied by a matrix
// Each result column is A multiplied by the corresponding column of B
GFX_HD inline Mat4 operator*(Mat4 a, Mat4 b) {
    return {{
        a * b.col[0],
        a * b.col[1],
        a * b.col[2],
        a * b.col[3]
    }};
}

// Translation matrix
GFX_HD inline Mat4 mat4_translate(Vec3 t) {
    Mat4 m = mat4_identity();

    m.col[3] = {t.x, t.y, t.z, 1.0f};

    return m;
}

// Scaling matrix
GFX_HD inline Mat4 mat4_scale(Vec3 s) {
    return {{
        {s.x,   0.0f, 0.0f, 0.0f},
        {0.0f, s.y,   0.0f, 0.0f},
        {0.0f, 0.0f, s.z,   0.0f},
        {0.0f, 0.0f, 0.0f, 1.0f}
    }};
}

// Rotation around a normalized axis, using radians
GFX_HD inline Mat4 mat4_rotate(Vec3 axis, float radians) {
    const float x = axis.x;
    const float y = axis.y;
    const float z = axis.z;

    const float c = std::cos(radians);
    const float s = std::sin(radians);
    const float t = 1.0f - c;

    return {{
        {
            t * x * x + c,
            t * x * y + s * z,
            t * x * z - s * y,
            0.0f
        },
        {
            t * x * y - s * z,
            t * y * y + c,
            t * y * z + s * x,
            0.0f
        },
        {
            t * x * z + s * y,
            t * y * z - s * x,
            t * z * z + c,
            0.0f
        },
        {
            0.0f, 0.0f, 0.0f, 1.0f
        }
    }};
}

// Transform a point: translation applies because w = 1
GFX_HD inline Vec3 transform_point(Mat4 m, Vec3 p) {
    const Vec4 result = m * make_vec4(p, 1.0f);

    return {result.x, result.y, result.z};
}

// Transform a direction: translation is ignored because w = 0
GFX_HD inline Vec3 transform_direction(Mat4 m, Vec3 d) {
    const Vec4 result = m * make_vec4(d, 0.0f);

    return {result.x, result.y, result.z};
}

} // namespace gfx
