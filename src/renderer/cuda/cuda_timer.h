#pragma once

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>

namespace gfx {

class CudaStageTimer {
public:
    CudaStageTimer() = default;
    ~CudaStageTimer();

    CudaStageTimer(const CudaStageTimer&) = delete;
    CudaStageTimer& operator=(const CudaStageTimer&) = delete;

    bool init();
    void mark(int index);
    void end_frame();

private:
    static constexpr int kEventCount = 5;
    static constexpr int kStageCount = kEventCount - 1;

    std::array<cudaEvent_t, kEventCount> m_events{};
    std::array<bool, kEventCount> m_event_created{};
    std::array<double, kStageCount> m_sums_ms{};

    std::uint64_t m_frames = 0;
    bool m_initialized = false;

    std::chrono::steady_clock::time_point m_report_start{};
};

} // namespace gfx
