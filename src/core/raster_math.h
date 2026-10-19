#pragma once

#include "core/common.h"

#include <cmath>

namespace gfx {

// Convert screen coordinates to 1/16-pixel fixed-point coordinates
GFX_HD inline int snap_subpixel(float v) {
    return static_cast<int>(floorf(v * 16.0f + 0.5f));
}

// Signed edge function. Cast before multiplication to prevent
// intermediate overflow when coordinates are large
GFX_HD inline long long edge_function(int ax, int ay,
                            int bx, int by, int px, int py) {
    const long long dx =
        static_cast<long long>(bx) - ax;
    const long long dy =
        static_cast<long long>(by) - ay;
    const long long qx =
        static_cast<long long>(px) - ax;
    const long long qy =
        static_cast<long long>(py) - ay;

    return dx * qy - dy * qx;
}

// Screen coordinates have Y increasing downward
GFX_HD inline bool is_top_left(int ax, int ay, int bx, int by) {
    const int dx = bx - ax;
    const int dy = by - ay;

    return dy < 0 || (dy == 0 && dx > 0);
}

} // namespace gfx
