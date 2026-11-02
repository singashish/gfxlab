#include "core/box_select.h"

#include <algorithm>
#include <cmath>

namespace gfx {

namespace {

// How many sample columns/rows cover `extent` pixels at `spacing`
int cells_for(float extent, float spacing) {
    const float cells = std::ceil(extent / spacing);

    return cells < 1.0f ? 1 : static_cast<int>(cells);
}

} // namespace

BoxGrid make_box_grid(float x0, float y0, float x1, float y1,
                      float view_width, float view_height, float spacing,
                      std::size_t max_rays) {
    BoxGrid grid;

    if (!(view_width > 0.0f) || !(view_height > 0.0f)) {
        return grid;
    }

    if (x1 < x0) {
        std::swap(x0, x1);
    }

    if (y1 < y0) {
        std::swap(y0, y1);
    }

    // Cut to the window. The comparisons are written so that a NaN
    // coordinate ends in the "nothing to sample" case
    x0 = std::max(x0, 0.0f);
    y0 = std::max(y0, 0.0f);
    x1 = std::min(x1, view_width);
    y1 = std::min(y1, view_height);

    if (!(x1 >= x0) || !(y1 >= y0)) {
        return grid;
    }

    if (!(spacing > 0.0f)) {
        spacing = kBoxSelectSpacing;
    }

    max_rays = std::max<std::size_t>(max_rays, 1);

    const float width = x1 - x0;
    const float height = y1 - y0;

    int nx = cells_for(width, spacing);
    int ny = cells_for(height, spacing);

    // Too many rays: make the grid coarser until it fits. Every step
    // grows the spacing by 10 %, and a spacing larger than the box
    // gives one ray, so this ends
    while (static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) >
           max_rays) {
        spacing *= 1.1f;
        nx = cells_for(width, spacing);
        ny = cells_for(height, spacing);
    }

    grid.x0 = x0;
    grid.y0 = y0;
    grid.nx = nx;
    grid.ny = ny;
    grid.step_x = width / static_cast<float>(nx);
    grid.step_y = height / static_cast<float>(ny);

    return grid;
}

std::vector<int> box_select_objects(const Scene& scene, const Model& model,
                                    PickIndex& index, const EditView& view,
                                    float x0, float y0, float x1, float y1,
                                    float spacing, std::size_t max_rays) {
    std::vector<int> result;

    const BoxGrid grid = make_box_grid(x0, y0, x1, y1, view.width,
                                       view.height, spacing, max_rays);

    if (grid.nx == 0 || grid.ny == 0 || scene.objects.empty()) {
        return result;
    }

    std::vector<bool> seen(scene.objects.size(), false);

    for (int j = 0; j < grid.ny; ++j) {
        const float py = grid.y0 + (static_cast<float>(j) + 0.5f) *
                                       grid.step_y;

        for (int i = 0; i < grid.nx; ++i) {
            const float px = grid.x0 + (static_cast<float>(i) + 0.5f) *
                                           grid.step_x;

            PickRay ray;
            PickHit hit;

            if (make_pick_ray(view, px, py, ray) &&
                pick_object(scene, model, index, ray, hit) &&
                hit.object >= 0 &&
                static_cast<std::size_t>(hit.object) < seen.size()) {
                seen[static_cast<std::size_t>(hit.object)] = true;
            }
        }
    }

    for (std::size_t k = 0; k < seen.size(); ++k) {
        if (seen[k]) {
            result.push_back(static_cast<int>(k));
        }
    }

    return result;
}

} // namespace gfx
