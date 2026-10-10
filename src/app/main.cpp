#include "core/log.h"

int main() {
    // gfx::log_set_level(gfx::LogLevel::LOG_LEVEL_TRACE);

    LOG_INFO("gfxlab starting");

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

    return 0;
}
