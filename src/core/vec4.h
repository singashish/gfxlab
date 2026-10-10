#pragma once

#include "core/common.h"
#include "core/vec3.h"

namespace gfx {

struct Vec4 {
    float x, y, z, w;
};

GFX_HD inline Vec4 operator+(Vec4 a, Vec4 b) {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z,
        a.w + b.w
    };
}

GFX_HD inline Vec4 operator*(Vec4 v, float s) {
    return {
        v.x * s,
        v.y * s,
        v.z * s,
        v.w * s
    };
}

GFX_HD inline Vec4 make_vec4(Vec3 v, float w) {
    return {v.x, v.y, v.z, w};
}

} // namespace gfx
