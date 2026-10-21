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

// General 4x4 inverse (cofactor expansion). Returns false and leaves
// out unchanged when the matrix is singular (determinant ~0).
// Used to move rays into an object's own space in the ray tracer
GFX_HD inline bool mat4_inverse(const Mat4& m, Mat4& out) {
    // Column-major: a[c * 4 + r] is row r of column c
    const float a[16] = {
        m.col[0].x, m.col[0].y, m.col[0].z, m.col[0].w,
        m.col[1].x, m.col[1].y, m.col[1].z, m.col[1].w,
        m.col[2].x, m.col[2].y, m.col[2].z, m.col[2].w,
        m.col[3].x, m.col[3].y, m.col[3].z, m.col[3].w
    };

    float inv[16];

    inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] -
             a[9] * a[6] * a[15] + a[9] * a[7] * a[14] +
             a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
    inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] +
             a[8] * a[6] * a[15] - a[8] * a[7] * a[14] -
             a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
    inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] -
             a[8] * a[5] * a[15] + a[8] * a[7] * a[13] +
             a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
    inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] +
              a[8] * a[5] * a[14] - a[8] * a[6] * a[13] -
              a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
    inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] +
             a[9] * a[2] * a[15] - a[9] * a[3] * a[14] -
             a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
    inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] -
             a[8] * a[2] * a[15] + a[8] * a[3] * a[14] +
             a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
    inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] +
             a[8] * a[1] * a[15] - a[8] * a[3] * a[13] -
             a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
    inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] -
              a[8] * a[1] * a[14] + a[8] * a[2] * a[13] +
              a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
    inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] -
             a[5] * a[2] * a[15] + a[5] * a[3] * a[14] +
             a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
    inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] +
             a[4] * a[2] * a[15] - a[4] * a[3] * a[14] -
             a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
    inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] -
              a[4] * a[1] * a[15] + a[4] * a[3] * a[13] +
              a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
    inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] +
              a[4] * a[1] * a[14] - a[4] * a[2] * a[13] -
              a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
    inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] +
             a[5] * a[2] * a[11] - a[5] * a[3] * a[10] -
             a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
    inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] -
             a[4] * a[2] * a[11] + a[4] * a[3] * a[10] +
             a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
    inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] +
              a[4] * a[1] * a[11] - a[4] * a[3] * a[9] -
              a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
    inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] -
              a[4] * a[1] * a[10] + a[4] * a[2] * a[9] +
              a[8] * a[1] * a[6] - a[8] * a[2] * a[5];

    const float det =
        a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];

    if (!(fabsf(det) > 1.0e-30f)) {
        return false;
    }

    const float inv_det = 1.0f / det;

    for (int c = 0; c < 4; ++c) {
        out.col[c] = {inv[c * 4 + 0] * inv_det,
                      inv[c * 4 + 1] * inv_det,
                      inv[c * 4 + 2] * inv_det,
                      inv[c * 4 + 3] * inv_det};
    }

    return true;
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
