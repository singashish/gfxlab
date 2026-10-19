#include "core/raster_math.h"
#include "check.h"

#include <array>
#include <cstdio>

namespace {

struct Point {
    float x;
    float y;
};

using Triangle = std::array<Point, 3>;

gfx::RasterTriangleSetup setup(const Triangle& t) {
    return gfx::setup_raster_triangle(
        t[0].x, t[0].y,
        t[1].x, t[1].y,
        t[2].x, t[2].y);
}

bool covers_pixel(const Triangle& triangle, int x, int y) {
    return gfx::raster_triangle_covers_pixel(setup(triangle), x, y);
}

void test_snap_subpixel() {
    CHECK(gfx::snap_subpixel(1.0f) == 16);
    CHECK(gfx::snap_subpixel(1.5f) == 24);
    CHECK(gfx::snap_subpixel(1.0f / 16.0f) == 1);
}

void test_degenerate_triangle() {
    const Triangle triangle = {{
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {3.0f, 3.0f}
    }};

    const auto result = setup(triangle);

    CHECK(!result.valid);
    CHECK(result.area == 0);
    CHECK(!covers_pixel(triangle, 1, 1));
}

void check_quad_split(
    const Triangle& first,
    const Triangle& second) {

    // Every sample inside the quad must be covered exactly once
    for (int y = 2; y < 10; ++y) {
        for (int x = 2; x < 10; ++x) {
            const int count =
                static_cast<int>(covers_pixel(first, x, y)) +
                static_cast<int>(covers_pixel(second, x, y));

            CHECK(count == 1);
        }
    }
}

void test_quad_diagonals_and_windings() {
    const Point a{2.0f, 2.0f};
    const Point b{10.0f, 2.0f};
    const Point c{10.0f, 10.0f};
    const Point d{2.0f, 10.0f};

    // Diagonal A-C
    check_quad_split(
        Triangle{{a, b, c}},
        Triangle{{a, c, d}});

    // Same diagonal, reversed winding
    check_quad_split(
        Triangle{{c, b, a}},
        Triangle{{d, c, a}});

    // Diagonal B-D.
    check_quad_split(
        Triangle{{a, b, d}},
        Triangle{{b, c, d}});

    // Same diagonal, reversed winding
    check_quad_split(
        Triangle{{d, b, a}},
        Triangle{{d, c, b}});
}

void test_horizontal_shared_edge() {
    const Triangle upper = {{
        {2.0f, 6.5f},
        {10.0f, 6.5f},
        {6.0f, 2.5f}
    }};

    const Triangle lower = {{
        {10.0f, 6.5f},
        {2.0f, 6.5f},
        {6.0f, 10.5f}
    }};

    // Pixel centres on row 6 lie exactly on the shared edge
    for (int x = 2; x < 10; ++x) {
        const int count =
            static_cast<int>(covers_pixel(upper, x, 6)) +
            static_cast<int>(covers_pixel(lower, x, 6));

        CHECK(count == 1);
    }

    // Repeat with both triangles reversed
    const Triangle upper_reversed = {{
        upper[2], upper[1], upper[0]
    }};

    const Triangle lower_reversed = {{
        lower[2], lower[1], lower[0]
    }};

    for (int x = 2; x < 10; ++x) {
        const int count =
            static_cast<int>(
                covers_pixel(upper_reversed, x, 6)) +
            static_cast<int>(
                covers_pixel(lower_reversed, x, 6));

        CHECK(count == 1);
    }
}

void test_setup_preserves_original_winding() {
    const auto positive = gfx::setup_raster_triangle(
        2.0f, 2.0f,
        10.0f, 2.0f,
        10.0f, 10.0f);

    const auto negative = gfx::setup_raster_triangle(
        10.0f, 10.0f,
        10.0f, 2.0f,
        2.0f, 2.0f);

    CHECK(positive.valid);
    CHECK(negative.valid);
    CHECK(positive.signed_area > 0);
    CHECK(negative.signed_area < 0);
    CHECK(positive.area > 0);
    CHECK(negative.area > 0);
}

} // namespace


int main() {
    test_snap_subpixel();
    test_degenerate_triangle();
    test_quad_diagonals_and_windings();
    test_horizontal_shared_edge();
    test_setup_preserves_original_winding();

    if (g_failures == 0) {
        std::puts("All raster tests passed");
        return 0;
    }

    std::printf("Raster tests failed: %d failure(s)\n", g_failures);
    return 1;
}
