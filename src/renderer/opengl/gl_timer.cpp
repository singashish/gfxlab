#include "renderer/opengl/gl_timer.h"

#include "core/log.h"

namespace gfx {

GLGpuTimer::~GLGpuTimer() {
    if (!m_initialized) {
        return;
    }

    GLuint queries[kSlots]{};

    for (std::size_t i = 0; i < kSlots; ++i) {
        queries[i] = m_slots[i].query;
    }

    glDeleteQueries(
        static_cast<GLsizei>(kSlots),
        queries);

    m_initialized = false;
}

bool GLGpuTimer::init() {
    if (m_initialized) {
        return true;
    }

    GLuint queries[kSlots]{};

    glGenQueries(
        static_cast<GLsizei>(kSlots),
        queries);

    for (std::size_t i = 0; i < kSlots; ++i) {
        if (queries[i] == 0) {
            glDeleteQueries(
                static_cast<GLsizei>(kSlots),
                queries);

            LOG_WARN("OpenGL GPU timer: failed to create queries");
            return false;
        }
    }

    for (std::size_t i = 0; i < kSlots; ++i) {
        m_slots[i].query = queries[i];
    }

    m_initialized = true;

    return true;
}

void GLGpuTimer::begin() {
    if (!m_initialized) {
        return;
    }

    m_active = false;

    Slot& slot = m_slots[m_current];

    if (slot.pending) {
        GLint available = 0;

        glGetQueryObjectiv(
            slot.query,
            GL_QUERY_RESULT_AVAILABLE,
            &available);

        if (available == 0) {
            // Keep checking the oldest pending slot next frame
            // Never wait for the GPU or skip ahead in the ring
            return;
        }

        GLuint64 nanoseconds = 0;

        glGetQueryObjectui64v(
            slot.query,
            GL_QUERY_RESULT,
            &nanoseconds);

        m_latest_ms =
            static_cast<double>(nanoseconds) * 1.0e-6;

        m_has_latest = true;
        slot.pending = false;
    }

    glBeginQuery(GL_TIME_ELAPSED, slot.query);
    m_active = true;
}

void GLGpuTimer::end() {
    if (!m_active) {
        return;
    }

    glEndQuery(GL_TIME_ELAPSED);

    m_slots[m_current].pending = true;
    m_active = false;

    m_current = (m_current + 1) % kSlots;
}

bool GLGpuTimer::take_latest(double& ms) {
    if (!m_has_latest) {
        return false;
    }

    ms = m_latest_ms;
    m_has_latest = false;

    return true;
}

} // namespace gfx
