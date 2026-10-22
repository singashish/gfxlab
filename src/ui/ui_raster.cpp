#include "ui/ui_raster.h"

#include <algorithm>
#include <cmath>

namespace gfx {

UiRasterBounds build_ui_raster_tris(const UiDrawList& list,
                                    int width,
                                    int height,
                                    std::vector<UiRasterTri>& out) {
    out.clear();

    UiRasterBounds bounds;

    if (width <= 0 || height <= 0 || list.font() == nullptr) {
        return bounds;
    }

    const std::vector<UiVertex>& vertices = list.vertices();
    const std::vector<std::uint32_t>& indices = list.indices();

    bounds.x0 = width;
    bounds.y0 = height;
    bounds.x1 = -1;
    bounds.y1 = -1;

    for (std::size_t i = 0; i + 2 < indices.size(); i += 3) {
        if (indices[i] >= vertices.size() ||
            indices[i + 1] >= vertices.size() ||
            indices[i + 2] >= vertices.size()) {
            continue;
        }

        const UiVertex* v[3] = {
            &vertices[indices[i]],
            &vertices[indices[i + 1]],
            &vertices[indices[i + 2]]
        };

        const RasterTriangleSetup setup = setup_raster_triangle(
            v[0]->x, v[0]->y, v[1]->x, v[1]->y, v[2]->x, v[2]->y);

        if (!setup.valid) {
            continue;  // zero area
        }

        // Conservative pixel bounds: a pixel whose centre lies inside
        // the triangle is always within floor(min) .. floor(max)
        const float min_x = std::min({v[0]->x, v[1]->x, v[2]->x});
        const float max_x = std::max({v[0]->x, v[1]->x, v[2]->x});
        const float min_y = std::min({v[0]->y, v[1]->y, v[2]->y});
        const float max_y = std::max({v[0]->y, v[1]->y, v[2]->y});

        UiRasterTri t;
        t.min_x = std::max(0, static_cast<int>(std::floor(min_x)));
        t.min_y = std::max(0, static_cast<int>(std::floor(min_y)));
        t.max_x = std::min(width - 1, static_cast<int>(std::floor(max_x)));
        t.max_y = std::min(height - 1, static_cast<int>(std::floor(max_y)));

        if (t.min_x > t.max_x || t.min_y > t.max_y) {
            continue;  // entirely off screen
        }

        for (int k = 0; k < 3; ++k) {
            const UiVertex& src = *v[setup.order[k]];

            t.sx[k] = setup.sx[k];
            t.sy[k] = setup.sy[k];
            t.u[k] = src.u;
            t.v[k] = src.v;
            t.color[k] = src.color;
        }

        // Edge k runs between the two vertices other than k
        for (int k = 0; k < 3; ++k) {
            const int a = (k + 1) % 3;
            const int b = (k + 2) % 3;

            t.bias[k] = is_top_left(t.sx[a], t.sy[a], t.sx[b], t.sy[b])
                ? 0
                : -1;
        }

        t.inv_area = 1.0f / static_cast<float>(setup.area);

        bounds.x0 = std::min(bounds.x0, t.min_x);
        bounds.y0 = std::min(bounds.y0, t.min_y);
        bounds.x1 = std::max(bounds.x1, t.max_x);
        bounds.y1 = std::max(bounds.y1, t.max_y);

        out.push_back(t);
    }

    if (out.empty()) {
        bounds = UiRasterBounds{};
    }

    return bounds;
}

} // namespace gfx
