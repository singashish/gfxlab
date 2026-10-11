#include "platform/input.h"
#include "check.h"

#include <cstdio>


int main() {
    using gfx::InputState;
    using gfx::Key;
    using gfx::MouseButton;

    InputState input;

    const Key key_a = Key::GFX_KEY_A;
    const MouseButton middle = MouseButton::GFX_MOUSE_MIDDLE;

    // 1. Press and held state across frames
    input.begin_frame();
    input.set_key(key_a, true);

    CHECK(input.key_down(key_a));
    CHECK(input.key_pressed(key_a));
    CHECK(!input.key_released(key_a));

    input.end_frame();
    input.begin_frame();

    CHECK(input.key_down(key_a));
    CHECK(!input.key_pressed(key_a));
    CHECK(!input.key_released(key_a));

    // 2. Key auto-repeat must not generate another press
    input.set_key(key_a, true);

    CHECK(input.key_down(key_a));
    CHECK(!input.key_pressed(key_a));

    // 3. Release
    input.set_key(key_a, false);

    CHECK(!input.key_down(key_a));
    CHECK(!input.key_pressed(key_a));
    CHECK(input.key_released(key_a));

    input.end_frame();
    input.begin_frame();

    CHECK(!input.key_released(key_a));

    // 4. Press and release during the same frame
    input.set_key(key_a, true);
    input.set_key(key_a, false);

    CHECK(!input.key_down(key_a));
    CHECK(input.key_pressed(key_a));
    CHECK(input.key_released(key_a));

    // The following frame clears both edge flags
    input.end_frame();
    input.begin_frame();

    CHECK(!input.key_pressed(key_a));
    CHECK(!input.key_released(key_a));

    // 5. Mouse-button press and release
    input.set_mouse_button(middle, true);

    CHECK(input.mouse_down(middle));
    CHECK(input.mouse_pressed(middle));
    CHECK(!input.mouse_released(middle));

    input.set_mouse_button(middle, false);

    CHECK(!input.mouse_down(middle));
    CHECK(input.mouse_pressed(middle));
    CHECK(input.mouse_released(middle));

    input.end_frame();
    input.begin_frame();

    CHECK(!input.mouse_pressed(middle));
    CHECK(!input.mouse_released(middle));

    // 6. Mouse delta
    // Even if an earlier frame had no position event, the first position
    // received must not produce a delta from the origin
    input.end_frame();
    input.begin_frame();

    input.set_mouse_position(10.0f, 10.0f);
    input.end_frame();

    CHECK(approx(input.mouse_x(), 10.0f));
    CHECK(approx(input.mouse_y(), 10.0f));
    CHECK(approx(input.mouse_dx(), 0.0f));
    CHECK(approx(input.mouse_dy(), 0.0f));

    input.begin_frame();
    input.set_mouse_position(15.0f, 8.0f);
    input.end_frame();

    CHECK(approx(input.mouse_dx(), 5.0f));
    CHECK(approx(input.mouse_dy(), -2.0f));

    input.begin_frame();
    input.end_frame();

    CHECK(approx(input.mouse_dx(), 0.0f));
    CHECK(approx(input.mouse_dy(), 0.0f));

    // 7. Scroll accumulates within a frame and resets at begin_frame()
    input.begin_frame();
    input.add_scroll(1.0f);
    input.add_scroll(0.5f);

    CHECK(approx(input.scroll(), 1.5f));

    input.end_frame();
    input.begin_frame();

    CHECK(approx(input.scroll(), 0.0f));

    // 8. Losing focus releases every held input and resets mouse history
    const MouseButton left = MouseButton::GFX_MOUSE_LEFT;

    input.set_key(key_a, true);
    input.set_mouse_button(left, true);

    input.end_frame();
    input.begin_frame();

    CHECK(input.key_down(key_a));
    CHECK(input.mouse_down(left));

    input.release_all();

    CHECK(!input.key_down(key_a));
    CHECK(input.key_released(key_a));
    CHECK(!input.mouse_down(left));
    CHECK(input.mouse_released(left));

    input.set_mouse_position(100.0f, 200.0f);
    input.end_frame();

    CHECK(approx(input.mouse_dx(), 0.0f));
    CHECK(approx(input.mouse_dy(), 0.0f));

    // 9. Focus loss must not leave a stale baseline
    input.set_mouse_position(100.0f, 100.0f); 
    input.end_frame();

    input.set_mouse_position(110.0f, 100.0f); 
    input.end_frame();                                             // dx = 10
    
    input.release_all();
    input.end_frame();                                             // frame with no new position

    input.set_mouse_position(500.0f, 400.0f); input.end_frame();   // cursor comes back elsewhere
    CHECK(approx(input.mouse_dx(), 0.0f));
    CHECK(approx(input.mouse_dy(), 0.0f));


    if (g_failures == 0) {
        std::printf("All input tests passed\n");
        return 0;
    }

    std::fprintf(stderr, "%d input test(s) failed\n", g_failures);
    return 1;
}
