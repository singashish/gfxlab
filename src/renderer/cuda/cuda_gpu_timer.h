#pragma once

#include <cuda_runtime.h>

#include <array>

namespace gfx {

class CudaGpuTimer {
public:
    static constexpr int kSlots = 4;

    CudaGpuTimer() = default;
    ~CudaGpuTimer();

    CudaGpuTimer(const CudaGpuTimer&) = delete;
    CudaGpuTimer& operator=(const CudaGpuTimer&) = delete;

    bool init();

    // Returns false when this frame cannot be timed without waiting
    bool begin();

    // Records the end after all timed CUDA work has been queued
    void end();

    bool take_latest(double& ms);

private:
    struct Slot {
        cudaEvent_t start = nullptr;
        cudaEvent_t end = nullptr;
        bool pending = false;
    };

    std::array<Slot, kSlots> m_slots{};
    int m_current = 0;

    bool m_initialized = false;
    bool m_active = false;

    bool m_has_latest = false;
    double m_latest_ms = 0.0;
};

} // namespace gfx
