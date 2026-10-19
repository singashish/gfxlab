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

bool covers_pixel(Triangle triangle, int pixel_x, int pixel_y) {
    int x[3];
    int y[3];

    for (int i = 0; i < 3; ++i) {
        x[i] = gfx::snap_subpixel(triangle[i].x);
        y[i] = gfx::snap_subpixel(triangle[i].y);
    }

    long long area = gfx::edge_function(
        x[0], y[0], x[1], y[1], x[2], y[2]
    );

    if (area == 0) {
        return false;
    }

    // Normalize winding so the signed area is positive
    if (area < 0) {
        std::swap(x[1], x[2]);
        std::swap(y[1], y[2]);
        area = -area;
    }

    // Pixel center, in the same 1/16-pixel units
    const int px = pixel_x * 16 + 8;
    const int py = pixel_y * 16 + 8;

    const long long e0 = gfx::edge_function(
        x[1], y[1], x[2], y[2], px, py
    );
    const long long e1 = gfx::edge_function(
        x[2], y[2], x[0], y[0], px, py
    );
    const long long e2 = gfx::edge_function(
        x[0], y[0], x[1], y[1], px, py
    );

    const long long bias0 =
        gfx::is_top_left(x[1], y[1], x[2], y[2]) ? 0 : -1;
    const long long bias1 =
        gfx::is_top_left(x[2], y[2], x[0], y[0]) ? 0 : -1;
    const long long bias2 =
        gfx::is_top_left(x[0], y[0], x[1], y[1]) ? 0 : -1;

    return e0 + bias0 >= 0 &&
           e1 + bias1 >= 0 &&
           e2 + bias2 >= 0;
}

void test_snap_subpixel() {
    CHECK(gfx::snap_subpixel(1.0f) == 16);
    CHECK(gfx::snap_subpixel(1.5f) == 24);
    CHECK(gfx::snap_subpixel(1.0f / 16.0f) == 1);
}

void test_degenerate_triangle() {
    const Triangle degenerate{{
        {1.0f, 1.0f},
        {2.0f, 2.0f},
        {3.0f, 3.0f}
    }};

    const int ax = gfx::snap_subpixel(degenerate[0].x);
    const int ay = gfx::snap_subpixel(degenerate[0].y);
    const int bx = gfx::snap_subpixel(degenerate[1].x);
    const int by = gfx::snap_subpixel(degenerate[1].y);
    const int cx = gfx::snap_subpixel(degenerate[2].x);
    const int cy = gfx::snap_subpixel(degenerate[2].y);

    CHECK(gfx::edge_function(ax, ay, bx, by, cx, cy) == 0);
    CHECK(!covers_pixel(degenerate, 1, 1));
}

void check_quad_split(Triangle first, Triangle second) {
    // Pixel centers inside the square [2,10] x [2,10]
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
        Triangle{{a, c, d}}
    );

    // Reverse both windings
    check_quad_split(
        Triangle{{c, b, a}},
        Triangle{{d, c, a}}
    );

    // Diagonal B-D
    check_quad_split(
        Triangle{{a, b, d}},
        Triangle{{b, c, d}}
    );

    check_quad_split(
        Triangle{{d, b, a}},
        Triangle{{d, c, b}}
    );
}

void test_horizontal_shared_edge() {
    // The shared edge lies exactly through pixel centers at y = 6.5
    const Triangle upper{{
        {2.0f, 6.5f},
        {10.0f, 6.5f},
        {6.0f, 2.5f}
    }};

    const Triangle lower{{
        {10.0f, 6.5f},
        {2.0f, 6.5f},
        {6.0f, 10.5f}
    }};

    for (int x = 2; x < 10; ++x) {
        const int count =
            static_cast<int>(covers_pixel(upper, x, 6)) +
            static_cast<int>(covers_pixel(lower, x, 6));

        CHECK(count == 1);
    }

    // Reverse winding independently
    Triangle upper_reversed{{upper[2], upper[1], upper[0]}};
    Triangle lower_reversed{{lower[2], lower[1], lower[0]}};

    for (int x = 2; x < 10; ++x) {
        const int count =
            static_cast<int>(covers_pixel(upper_reversed, x, 6)) +
            static_cast<int>(covers_pixel(lower_reversed, x, 6));

        CHECK(count == 1);
    }
}

} // namespace


int main() {
    test_snap_subpixel();
    test_degenerate_triangle();
    test_quad_diagonals_and_windings();
    test_horizontal_shared_edge();

    if (g_failures == 0) {
        std::printf("All raster tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d raster test(s) failed\n", g_failures);
    return 1;
}
