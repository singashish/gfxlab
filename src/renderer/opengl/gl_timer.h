#pragma once

#include "renderer/opengl/gl_api.h"

#include <array>
#include <cstddef>

namespace gfx {

class GLGpuTimer {
public:
    static constexpr std::size_t kSlots = 4;

    GLGpuTimer() = default;
    ~GLGpuTimer();

    GLGpuTimer(const GLGpuTimer&) = delete;
    GLGpuTimer& operator=(const GLGpuTimer&) = delete;

    // Must be called after the OpenGL context is current
    bool init();

    // Start timing this frame if a query slot is available
    void begin();

    // End the current query, if timing was started
    void end();

    // Consume the latest newly available measurement
    bool take_latest(double& ms);

private:
    struct Slot {
        GLuint query = 0;
        bool pending = false;
    };

    std::array<Slot, kSlots> m_slots{};
    std::size_t m_current = 0;

    bool m_initialized = false;
    bool m_active = false;

    bool m_has_latest = false;
    double m_latest_ms = 0.0;
};

} // namespace gfx
