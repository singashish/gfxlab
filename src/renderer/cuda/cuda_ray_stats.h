#pragma once

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cstdint>

namespace gfx {

// Counts the rays the ray tracer traces per frame (camera, shadow and
// ambient occlusion rays) and logs the averages once per second.
//
// The counts are produced on the GPU. Reading them back right away would
// make the CPU wait for the GPU, so this uses the same idea as the GPU
// timers: a ring of slots, each with its own device counters, a pinned
// host copy and an event. A slot is read only once its event says the
// copy has arrived
class CudaRayStats {
public:
    static constexpr int kSlots = 4;

    CudaRayStats() = default;
    ~CudaRayStats();

    CudaRayStats(const CudaRayStats&) = delete;
    CudaRayStats& operator=(const CudaRayStats&) = delete;

    bool init();

    // Collects a finished slot if there is one, then zeroes the next
    // slot's counters and returns them for the kernel. Returns null when
    // the slot is still busy (this frame is then not counted)
    unsigned long long* begin_frame();

    // Queues the copy of this frame's counters to the host
    void end_frame();

    // Logs the averages about once per second
    void report(unsigned int ao_samples, unsigned int ao_max_samples);

private:
    struct Slot {
        unsigned long long* device = nullptr;  // 3 counters on the GPU
        unsigned long long* host = nullptr;    // 3 counters, pinned
        cudaEvent_t copied = nullptr;
        bool pending = false;
    };

    std::array<Slot, kSlots> m_slots{};
    int m_current = 0;
    bool m_initialized = false;
    bool m_active = false;

    // Sums over the frames collected since the last report
    std::array<double, 3> m_sums{};
    std::uint64_t m_frames = 0;
    std::chrono::steady_clock::time_point m_report_start{};
};

} // namespace gfx
