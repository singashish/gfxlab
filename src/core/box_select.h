#pragma once

/*
 * Box select: which objects can be seen through a rectangle of the screen?
 *
 * The rectangle is covered with a grid of pick rays (core/pick.h), one
 * ray every few pixels, and every object that is the nearest hit of at
 * least one ray is part of the result. Because every ray stops at its
 * nearest object, an object that is completely hidden behind another one
 * is not selected, exactly as it would not be in Blender's solid view.
 *
 * What this costs and what it misses:
 *   - The number of rays is capped (kBoxSelectMaxRays), so a box as big as
 *     the window uses a coarser grid than a small one.
 *   - An object smaller than the grid spacing, or one whose visible part
 *     is thinner than the spacing, can fall between the rays and is then
 *     not selected. With the default spacing of 4 pixels this only
 *     concerns objects of a few pixels.
 */

#include "core/pick.h"
#include "core/scene.h"
#include "core/transform_edit.h"

#include <cstddef>
#include <vector>

namespace gfx {

// Distance between two rays, in pixels, and the most rays one box select
// may shoot
constexpr float kBoxSelectSpacing = 4.0f;
constexpr std::size_t kBoxSelectMaxRays = 16384;

// The sample points of a box: nx * ny rays, the ray (i, j) goes through
// pixel (x0 + (i + 0.5) * step_x, y0 + (j + 0.5) * step_y). nx = ny = 0
// means that there is nothing to sample
struct BoxGrid {
    float x0 = 0.0f;
    float y0 = 0.0f;
    float step_x = 0.0f;
    float step_y = 0.0f;
    int nx = 0;
    int ny = 0;
};

// The grid for the rectangle between the two corners (in any order;
// pixels from the top-left corner of a view_width x view_height window).
// The rectangle is cut to the window first; one outside of it gives an
// empty grid. The spacing is at most `spacing` pixels, and grows when
// that would need more than max_rays rays. A rectangle with no width or
// no height (the mouse moved in a straight line) still gets one row or
// column of rays
BoxGrid make_box_grid(float x0, float y0, float x1, float y1,
                      float view_width, float view_height, float spacing,
                      std::size_t max_rays);

// The visible objects whose pixels lie inside the rectangle, as indices
// into Scene::objects in ascending order, each once. Hidden objects and
// objects with an empty mesh never appear (pick_object skips them)
std::vector<int> box_select_objects(const Scene& scene, const Model& model,
                                    PickIndex& index, const EditView& view,
                                    float x0, float y0, float x1, float y1,
                                    float spacing = kBoxSelectSpacing,
                                    std::size_t max_rays = kBoxSelectMaxRays);

} // namespace gfx
