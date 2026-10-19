#pragma once

#include "core/common.h"

#include <cmath>
#include <cstdint>

namespace gfx {

// Coordinates are snapped to 1/16-pixel fixed-point precision
constexpr int kRasterSubpixelScale = 16;
constexpr int kRasterSubpixelHalf = 8;

GFX_HD inline int snap_subpixel(float v) {
    return static_cast<int>(
        floorf(v * static_cast<float>(kRasterSubpixelScale) + 0.5f));
}

// Signed edge function. Positive values indicate one side of the
// directed edge; zero means the point lies exactly on the edge
GFX_HD inline long long edge_function(
    int ax, int ay,
    int bx, int by,
    int px, int py) {

    const long long dx = static_cast<long long>(bx) - ax;
    const long long dy = static_cast<long long>(by) - ay;
    const long long qx = static_cast<long long>(px) - ax;
    const long long qy = static_cast<long long>(py) - ay;

    return dx * qy - dy * qx;
}

// Top-left convention for screen coordinates whose Y axis points down
GFX_HD inline bool is_top_left(
    int ax, int ay,
    int bx, int by) {

    const int dx = bx - ax;
    const int dy = by - ay;

    return dy < 0 || (dy == 0 && dx > 0);
}

// Shared result of triangle setup.
//
// sx/sy: snapped coordinates in positive-area order
// order: maps normalized vertices back to the original 0,1,2 order
// signed_area: original signed area, before winding normalization
// area: positive area after normalization.
struct RasterTriangleSetup {
    bool valid = false;
    int sx[3] = {};
    int sy[3] = {};
    int order[3] = {0, 1, 2};
    long long signed_area = 0;
    long long area = 0;
};

GFX_HD inline RasterTriangleSetup setup_raster_triangle(
    float x0, float y0,
    float x1, float y1,
    float x2, float y2) {

    RasterTriangleSetup result{};

    result.order[0] = 0;
    result.order[1] = 1;
    result.order[2] = 2;

    result.sx[0] = snap_subpixel(x0);
    result.sy[0] = snap_subpixel(y0);
    result.sx[1] = snap_subpixel(x1);
    result.sy[1] = snap_subpixel(y1);
    result.sx[2] = snap_subpixel(x2);
    result.sy[2] = snap_subpixel(y2);

    result.signed_area = edge_function(
        result.sx[0], result.sy[0],
        result.sx[1], result.sy[1],
        result.sx[2], result.sy[2]);

    if (result.signed_area == 0) {
        return result;
    }

    result.area = result.signed_area;

    if (result.signed_area < 0) {
        const int tmp_x = result.sx[1];
        result.sx[1] = result.sx[2];
        result.sx[2] = tmp_x;

        const int tmp_y = result.sy[1];
        result.sy[1] = result.sy[2];
        result.sy[2] = tmp_y;

        const int tmp_order = result.order[1];
        result.order[1] = result.order[2];
        result.order[2] = tmp_order;

        result.area = -result.signed_area;
    }

    result.valid = true;
    return result;
}

// Test the top-left fill convention at a fixed-point pixel centre
// px/py are already in 1/16-pixel coordinates
GFX_HD inline bool raster_triangle_covers_sample(
    const RasterTriangleSetup& triangle,
    int px,
    int py) {

    if (!triangle.valid) {
        return false;
    }

    const long long e0 = edge_function(
        triangle.sx[1], triangle.sy[1],
        triangle.sx[2], triangle.sy[2],
        px, py);

    const long long e1 = edge_function(
        triangle.sx[2], triangle.sy[2],
        triangle.sx[0], triangle.sy[0],
        px, py);

    const long long e2 = edge_function(
        triangle.sx[0], triangle.sy[0],
        triangle.sx[1], triangle.sy[1],
        px, py);

    const bool tl0 = is_top_left(
        triangle.sx[1], triangle.sy[1],
        triangle.sx[2], triangle.sy[2]);

    const bool tl1 = is_top_left(
        triangle.sx[2], triangle.sy[2],
        triangle.sx[0], triangle.sy[0]);

    const bool tl2 = is_top_left(
        triangle.sx[0], triangle.sy[0],
        triangle.sx[1], triangle.sy[1]);

    return
        e0 + (tl0 ? 0LL : -1LL) >= 0 &&
        e1 + (tl1 ? 0LL : -1LL) >= 0 &&
        e2 + (tl2 ? 0LL : -1LL) >= 0;
}

// Convenience helper: test a pixel using its centre
GFX_HD inline bool raster_triangle_covers_pixel(
    const RasterTriangleSetup& triangle,
    int pixel_x,
    int pixel_y) {

    return raster_triangle_covers_sample(
        triangle,
        pixel_x * kRasterSubpixelScale + kRasterSubpixelHalf,
        pixel_y * kRasterSubpixelScale + kRasterSubpixelHalf);
}

} // namespace gfx
