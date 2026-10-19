#include "renderer/cuda/cuda_timer.h"

#include "core/log.h"
#include "renderer/cuda/cuda_api.h"

namespace gfx {

CudaStageTimer::~CudaStageTimer() {
    for (int i = 0; i < kEventCount; ++i) {
        if (m_event_created[i]) {
            (void)cuda_ok(
                cudaEventDestroy(m_events[i]),
                "cudaEventDestroy stage timer");
            m_event_created[i] = false;
        }
    }
}

bool CudaStageTimer::init() {
    if (m_initialized) {
        return true;
    }

    for (int i = 0; i < kEventCount; ++i) {
        if (!cuda_ok(
                cudaEventCreate(&m_events[i]),
                "cudaEventCreate stage timer")) {
            return false;
        }

        m_event_created[i] = true;
    }

    m_sums_ms.fill(0.0);
    m_frames = 0;
    m_report_start = std::chrono::steady_clock::now();
    m_initialized = true;

    return true;
}

void CudaStageTimer::mark(int index) {
    if (!m_initialized ||
        index < 0 || index >= kEventCount) {
        return;
    }

    (void)cuda_ok(
        cudaEventRecord(m_events[index], 0),
        "cudaEventRecord stage timer");
}

void CudaStageTimer::end_frame() {
    if (!m_initialized) {
        return;
    }

    // Wait for the final marked GPU stage to finish
    if (!cuda_ok(
            cudaEventSynchronize(m_events[kEventCount - 1]),
            "cudaEventSynchronize stage timer")) {
        return;
    }

    for (int i = 0; i < kStageCount; ++i) {
        float ms = 0.0f;

        if (!cuda_ok(
                cudaEventElapsedTime(
                    &ms, m_events[i], m_events[i + 1]),
                "cudaEventElapsedTime stage timer")) {
            return;
        }

        m_sums_ms[i] += ms;
    }

    ++m_frames;

    const auto now = std::chrono::steady_clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(
            now - m_report_start).count();

    if (elapsed_seconds < 1.0 || m_frames == 0) {
        return;
    }

    const double divisor = static_cast<double>(m_frames);

    const double clear = m_sums_ms[0] / divisor;
    const double geometry = m_sums_ms[1] / divisor;
    const double shade = m_sums_ms[2] / divisor;
    const double grid = m_sums_ms[3] / divisor;
    const double total = clear + geometry + shade + grid;

    LOG_INFO(
        "CUDA GPU ms: clear %.2f geometry+raster %.2f "
        "shade %.2f grid %.2f total %.2f",
        clear, geometry, shade, grid, total);

    m_sums_ms.fill(0.0);
    m_frames = 0;
    m_report_start = now;
}

} // namespace gfx
