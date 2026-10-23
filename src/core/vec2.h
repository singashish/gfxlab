#pragma once

#include "core/common.h"

namespace gfx {

// Two floats: texture coordinates (u, v) for now. A plain aggregate like
// Vec3, so it can be copied to the GPU as is
struct Vec2 {
    float x, y;
};

GFX_HD inline Vec2 operator+(Vec2 a, Vec2 b) {
    return {a.x + b.x, a.y + b.y};
}

GFX_HD inline Vec2 operator-(Vec2 a, Vec2 b) {
    return {a.x - b.x, a.y - b.y};
}

GFX_HD inline Vec2 operator*(Vec2 v, float s) {
    return {v.x * s, v.y * s};
}

} // namespace gfx
