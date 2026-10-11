#pragma once

#include <algorithm>
#include <chrono>

namespace gfx {

class FrameTimer {
public:
    FrameTimer() {
        reset();
    }

    void reset() {
        m_last = Clock::now();

        m_delta = 0.0;
        m_fps = 0.0;
        m_fps_updated = false;

        m_accum_time = 0.0;
        m_accum_frames = 0;
    }

    void tick() {
        const auto now = Clock::now();

        const double raw_seconds =
            std::chrono::duration<double>(now - m_last).count();

        m_last = now;
        advance(raw_seconds);
    }

    void advance(double raw_seconds) {
        raw_seconds = std::max(0.0, raw_seconds);

        m_delta = std::min(raw_seconds, 0.25);

        m_accum_time += raw_seconds;
        ++m_accum_frames;

        m_fps_updated = false;

        if (m_accum_time >= 0.5) {
            m_fps = static_cast<double>(m_accum_frames)
                    / m_accum_time;

            m_accum_time = 0.0;
            m_accum_frames = 0;
            m_fps_updated = true;
        }
    }

    double delta_seconds() const {
        return m_delta;
    }

    double fps() const {
        return m_fps;
    }

    bool fps_updated() const {
        return m_fps_updated;
    }

private:
    using Clock = std::chrono::steady_clock;

    Clock::time_point m_last{};

    double m_delta = 0.0;
    double m_fps = 0.0;
    double m_accum_time = 0.0;

    unsigned int m_accum_frames = 0;
    bool m_fps_updated = false;
};

} // namespace gfx
