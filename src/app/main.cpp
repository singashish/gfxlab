#include "core/log.h"

int main() {
    // gfx::log_set_level(gfx::LogLevel::LOG_LEVEL_TRACE);

    LOG_DEBUG("this should be hidden at the default level");
    LOG_INFO("gfxlab starting");
    LOG_WARN("example warning, value = %d", 42);
    return 0;
}
