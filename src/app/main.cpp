#include "core/log.h"
#include "core/timer.h"
#include "core/camera_math.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "platform/window.h"

#if defined(GFX_HAS_OPENGL)
#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_triangle_demo.h"
#endif

#include <cstdio>

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

#if !defined(GFX_HAS_OPENGL) && !defined(GFX_HAS_VULKAN) && !defined(GFX_HAS_CUDA)
    LOG_WARN("No graphics backends were compiled in");
#endif

    gfx::Window window;
    if (!window.create({"gfxlab", 1280, 720})) {
        LOG_ERROR("failed to create window");
        return 1;
    }

#if defined(GFX_HAS_OPENGL)
    // Declare after window: GLContext must be destroyed first
    gfx::GLContext gl;

    if (!gl.create(window.native_handle())) {
        LOG_ERROR("failed to create the OpenGL context");
        return 1;
    }

    gl.set_vsync(true);

    // Testing demo
    gfx::TriangleDemo triangle;

    if (!triangle.init()) {
        LOG_ERROR("failed to create the triangle demo");
        return 1;
    }
#endif

    gfx::FrameTimer timer;
    bool was_minimized = false;
    float spin_angle = 0.0f;

    gfx::OrbitCamera camera;
    constexpr float kFovY = gfx::to_radians(60.0f);

    while (!window.should_close()) {
        timer.tick();
        window.poll_events();

        if (window.is_minimized()) {
            if (!was_minimized) {
                LOG_INFO("window minimized");
                was_minimized = true;
            }

            window.wait_events();

            // Discard time spent waiting while minimized
            timer.reset();

            // If still minimized, return to waiting. If restored, continue
            // processing this iteration so resize/input events aren't skipped
            if (window.should_close()) {
                break;
            }

            if (window.is_minimized()) {
                continue;
            }
        }

        was_minimized = false;

        // Orbit camera controls
        const gfx::InputState& in = window.input();

        if (in.mouse_down(gfx::MouseButton::GFX_MOUSE_MIDDLE)) {
            if (in.key_down(gfx::Key::GFX_KEY_SHIFT)) {
                camera.pan(in.mouse_dx(), in.mouse_dy(),
                    camera.world_per_pixel(kFovY,
                        static_cast<float>(window.height())));
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
                    static_cast<float>(width) / static_cast<float>(height);

                LOG_INFO("resized: %dx%d (aspect %.3f)",
                        width, height, aspect);
            }
        }

        if (in.key_pressed(gfx::Key::GFX_KEY_ESCAPE)) {
            window.request_close();
        }

        for (int k = 1; k < static_cast<int>(gfx::Key::GFX_KEY_COUNT); ++k) {
            const gfx::Key key = static_cast<gfx::Key>(k);
            if (in.key_pressed(key)) 
                LOG_INFO("key %d pressed", k);
            if (in.key_released(key))
                LOG_INFO("key %d released", k);
        }

        if (timer.fps_updated() && timer.fps() > 0.0) {
            char title[96];

            std::snprintf(title, sizeof(title),
                "gfxlab | %.1f FPS | %.4f ms",
                timer.fps(), 1000.0 / timer.fps());

            window.set_title(title);
        }

#if defined(GFX_HAS_OPENGL)
        spin_angle += static_cast<float>(timer.delta_seconds());

        const float aspect =
            static_cast<float>(window.width()) /
            static_cast<float>(window.height());

        const gfx::Mat4 view = camera.view();

        const gfx::Mat4 proj = gfx::perspective(
            kFovY,
            aspect,
            0.1f,
            100.0f,
            gfx::ClipSpace::CLIP_SPACE_OPENGL
        );

        const gfx::Mat4 view_proj = proj * view;

        gfx::gl_clear_frame(window.width(), window.height());
        triangle.draw(view_proj, spin_angle);
        gl.swap_buffers();
#endif
    }

    LOG_INFO("shutting down");
    return 0;
}
