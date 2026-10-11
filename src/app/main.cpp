#include "core/log.h"
#include "platform/window.h"

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

    LOG_WARN("Example warning, value = %d", 42);

    gfx::Window window;
    if (!window.create({"gfxlab", 1280, 720})) {
        LOG_ERROR("failed to create window");
        return 1;
    }

    while (!window.should_close()) {
        window.poll_events();
        // rendering will go here
        const gfx::InputState& in = window.input();

        if (in.key_pressed(gfx::Key::GFX_KEY_ESCAPE)) window.request_close();

        for (int k = 1; k < (int)gfx::Key::GFX_KEY_COUNT; ++k) {
            const gfx::Key key = (gfx::Key)k;
            if (in.key_pressed(key))  LOG_INFO("key %d pressed", k);
            if (in.key_released(key)) LOG_INFO("key %d released", k);
        }
    }

    LOG_INFO("shutting down");
    return 0;
}
