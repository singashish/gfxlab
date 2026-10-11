#include "check.h"
#include "core/timer.h"

#include <cstdio>


int main() {
    constexpr double dt = 1.0 / 64.0;

    // Test 1: reset state
    {
        gfx::FrameTimer timer;
        timer.advance(1.0);
        timer.reset();

        CHECK(timer.delta_seconds() == 0.0);
        CHECK(timer.fps() == 0.0);
        CHECK(!timer.fps_updated());
    }

    // Test 2: one frame
    {
        gfx::FrameTimer timer;
        timer.advance(dt);

        CHECK(timer.delta_seconds() == dt);
        CHECK(!timer.fps_updated());
    }

    // Test 3: 64 frames produce exactly two updates
    {
        gfx::FrameTimer timer;
        int updates = 0;

        for (int frame = 1; frame <= 64; ++frame) {
            timer.advance(dt);

            if (timer.fps_updated()) {
                ++updates;

                CHECK(frame == 32 || frame == 64);
                CHECK(approx(timer.fps(), 64.0));
            }
        }

        CHECK(updates == 2);
    }

    // Test 4: fps_updated lasts only one advance
    {
        gfx::FrameTimer timer;

        for (int i = 0; i < 32; ++i) {
            timer.advance(dt);
        }

        CHECK(timer.fps_updated());

        timer.advance(dt);

        CHECK(!timer.fps_updated());
        CHECK(approx(timer.fps(), 64.0));
    }

    // Test 5: long stall clamps delta but not FPS time
    {
        gfx::FrameTimer timer;
        timer.advance(5.0);

        CHECK(timer.delta_seconds() == 0.25);
        CHECK(timer.fps_updated());
        CHECK(approx(timer.fps(), 0.2));
    }

    // Test 6: negative input
    {
        gfx::FrameTimer timer;
        timer.advance(-1.0);

        CHECK(timer.delta_seconds() == 0.0);
        CHECK(!timer.fps_updated());
    }

    // Test 7: real-clock smoke test
    {
        gfx::FrameTimer timer;
        timer.tick();
        timer.tick();

        CHECK(timer.delta_seconds() >= 0.0);
        CHECK(timer.delta_seconds() <= 0.25);
    }

    
    if (g_failures != 0) {
        std::printf("Timer tests failed: %d\n", g_failures);
        return 1;
    }

    std::printf("Timer tests passed\n");
    return 0;
}
