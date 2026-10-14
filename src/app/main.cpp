#include "core/log.h"
#include "core/timer.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "platform/window.h"
#include "renderer/renderer.h"

#include <algorithm>
#include <cstdio>
#include <memory>

int main() {
    // gfx::log_set_level(gfx::LogLevel::LOG_LEVEL_TRACE);

    LOG_INFO("gfxlab starting...");

#if defined(GFX_HAS_OPENGL)
    LOG_INFO("OpenGL backend compiled in");
#endif

#if defined(GFX_HAS_VULKAN)
    LOG_INFO("Vulkan backend compiled in");
#endif

#if defined(GFX_HAS_CUDA)
    LOG_INFO("CUDA backend compiled in");
#endif

    gfx::Window window;

    if (!window.create({"gfxlab", 1280, 720})) {
        LOG_ERROR("failed to create window");
        return 1;
    }

    // The renderer is destroyed before the window
    std::unique_ptr<gfx::IRenderer> renderer =
        gfx::create_renderer(
            gfx::RendererBackend::GFX_BACKEND_OPENGL
        );

    if (!renderer || !renderer->init(window.native_handle())) {
        LOG_ERROR("failed to initialize the renderer");
        return 1;
    }

    LOG_INFO("renderer: %s", renderer->name());
    renderer->set_vsync(true);

    gfx::FrameTimer timer;

    bool was_minimized = false;
    float spin_angle = 0.0f;

    constexpr float kNear = 0.1f;
    constexpr float kFar = 1000.0f;
    constexpr float kGridFadeFactor = 1.5f;

    gfx::OrbitCamera camera;
    constexpr float kFovY = gfx::to_radians(60.0f);

    // Shared by the normal main loop and the Windows modal loop
    auto draw_frame = [&]() {
        timer.tick();

        spin_angle +=
            static_cast<float>(timer.delta_seconds());

        gfx::FrameData frame;

        frame.width = window.width();
        frame.height = window.height();

        frame.view = camera.view();
        frame.eye = camera.eye();
        frame.basis = camera.basis();

        frame.fov_y = kFovY;
        frame.near_plane = kNear;
        frame.far_plane = kFar;

        frame.grid_fade_end = std::min(
            kGridFadeFactor *
                std::max(camera.distance, 10.0f),
            0.9f * kFar
        );

        frame.demo_spin_angle = spin_angle;

        renderer->render(frame);

        if (timer.fps_updated() && timer.fps() > 0.0) {
            char title[96];

            std::snprintf(
                title,
                sizeof(title),
                "gfxlab | %.1f FPS | %.4f ms",
                timer.fps(),
                1000.0 / timer.fps()
            );

            window.set_title(title);
        }
    };

    // Windows calls this while its own modal loop is active
    window.set_modal_frame_callback(draw_frame);

    while (!window.should_close()) {
        window.poll_events();

        if (window.is_minimized()) {
            if (!was_minimized) {
                LOG_INFO("window minimized");
                was_minimized = true;
            }

            window.wait_events();

            // Discard elapsed time while the window is minimized
            timer.reset();

            if (window.should_close()) {
                break;
            }

            if (window.is_minimized()) {
                continue;
            }
        }

        was_minimized = false;

        // Input and camera updates remain in the main loop
        const gfx::InputState& in = window.input();

        if (in.mouse_down(gfx::MouseButton::GFX_MOUSE_MIDDLE)) {
            if (in.key_down(gfx::Key::GFX_KEY_SHIFT)) {
                camera.pan(
                    in.mouse_dx(),
                    in.mouse_dy(),
                    camera.world_per_pixel(
                        kFovY,
                        static_cast<float>(window.height())
                    )
                );
            } else {
                camera.orbit(in.mouse_dx(), in.mouse_dy());
            }
        }

        camera.zoom(in.scroll());

        if (window.was_resized()) {
            const int width = window.width();
            const int height = window.height();

            if (width > 0 && height > 0) {
                const float aspect =
                    static_cast<float>(width) /
                    static_cast<float>(height);

                LOG_INFO(
                    "resized: %dx%d (aspect %.3f)",
                    width,
                    height,
                    aspect
                );
            }
        }

        if (in.key_pressed(gfx::Key::GFX_KEY_ESCAPE)) {
            window.request_close();
        }

        for (int k = 1;
             k < static_cast<int>(gfx::Key::GFX_KEY_COUNT);
             ++k) {
            const gfx::Key key = static_cast<gfx::Key>(k);

            if (in.key_pressed(key)) {
                LOG_INFO("key %d pressed", k);
            }

            if (in.key_released(key)) {
                LOG_INFO("key %d released", k);
            }
        }

        // Render the next frame
        // draw_frame() also advances the animation timer and updates FPS
        draw_frame();
    }

    // The callback is no longer needed. Clear it before shutdown
    window.set_modal_frame_callback({});

    LOG_INFO("shutting down");
    return 0;
}
