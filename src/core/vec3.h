#pragma once

/*
 * Math conventions:
 * - Right-handed coordinate system.
 * - Z-up: +Z is up, +X cross +Y = +Z.
 * - All vector components use float.
 * - Vec3 is a plain aggregate with public members and no constructors.
 * - Vec3 remains trivially copyable for GPU buffers and CUDA interoperability.
 * - Math functions are header-only and inline.
 * - normalize(v) returns the zero vector when v has zero length,
 *   avoiding division by zero and NaN results.
 */

#include "core/common.h"

#include <cmath>

namespace gfx {

struct Vec3 {
    float x, y, z;
};

// Component-wise addition
GFX_HD inline Vec3 operator+(Vec3 a, Vec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

// Component-wise subtraction
GFX_HD inline Vec3 operator-(Vec3 a, Vec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

// Scalar multiplication
GFX_HD inline Vec3 operator*(Vec3 v, float s) {
    return {v.x * s, v.y * s, v.z * s};
}

// Scalar multiplication with scalar on the left
GFX_HD inline Vec3 operator*(float s, Vec3 v) {
    return v * s;
}

// Unary negation
GFX_HD inline Vec3 operator-(Vec3 v) {
    return {-v.x, -v.y, -v.z};
}

// Dot product
GFX_HD inline float dot(Vec3 a, Vec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Right-handed cross product
GFX_HD inline Vec3 cross(Vec3 a, Vec3 b) {
    return {
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    };
}

// Euclidean length
GFX_HD inline float length(Vec3 v) {
    return std::sqrt(dot(v, v));
}

// Normalize to unit length, or return zero for a zero-length vector
GFX_HD inline Vec3 normalize(Vec3 v) {
    const float len = length(v);

    if (len == 0.0f) {
        return {0.0f, 0.0f, 0.0f};
    }

    return v * (1.0f / len);
}

} // namespace gfx
