#pragma once

#include "core/common.h"

namespace gfx {

constexpr float kPi = 3.14159265358979323846f;

GFX_HD constexpr float to_radians(float degrees) {
    return degrees * (kPi / 180.0f);
}

} // namespace gfx
