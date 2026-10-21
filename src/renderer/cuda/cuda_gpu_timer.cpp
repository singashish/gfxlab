#include "renderer/cuda/cuda_gpu_timer.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

namespace gfx {

CudaGpuTimer::~CudaGpuTimer() {
    for (Slot& slot : m_slots) {
        if (slot.start != nullptr) {
            (void)cuda_ok(
                cudaEventDestroy(slot.start),
                "cudaEventDestroy GPU timer start");
            slot.start = nullptr;
        }

        if (slot.end != nullptr) {
            (void)cuda_ok(
                cudaEventDestroy(slot.end),
                "cudaEventDestroy GPU timer end");
            slot.end = nullptr;
        }
    }

    m_initialized = false;
}

bool CudaGpuTimer::init() {
    if (m_initialized) {
        return true;
    }

    // cudaEventCreate enables timing by default.
    // Do not use cudaEventDisableTiming.
    for (Slot& slot : m_slots) {
        cudaError_t result = cudaEventCreate(&slot.start);

        if (result != cudaSuccess) {
            LOG_WARN(
                "CUDA GPU timing unavailable: "
                "creating start event failed: %s",
                cudaGetErrorString(result));
            return false;
        }

        result = cudaEventCreate(&slot.end);

        if (result != cudaSuccess) {
            LOG_WARN(
                "CUDA GPU timing unavailable: "
                "creating end event failed: %s",
                cudaGetErrorString(result));
            return false;
        }
    }

    m_current = 0;
    m_active = false;
    m_has_latest = false;
    m_initialized = true;

    LOG_INFO("CUDA GPU frame timing enabled");
    return true;
}

bool CudaGpuTimer::begin() {
    if (!m_initialized) {
        return false;
    }

    m_active = false;

    Slot& slot = m_slots[m_current];

    if (slot.pending) {
        const cudaError_t query = cudaEventQuery(slot.end);

        if (query == cudaErrorNotReady) {
            // Expected when the GPU is behind. Do not log this as an error.
            // Stay on the oldest pending slot.
            return false;
        }

        if (query != cudaSuccess) {
            (void)cuda_ok(query, "cudaEventQuery GPU timer");
            return false;
        }

        float elapsed_ms = 0.0f;

        if (!cuda_ok(
                cudaEventElapsedTime(
                    &elapsed_ms, slot.start, slot.end),
                "cudaEventElapsedTime GPU frame")) {
            return false;
        }

        m_latest_ms = static_cast<double>(elapsed_ms);
        m_has_latest = true;
        slot.pending = false;
    }

    if (!cuda_ok(
            cudaEventRecord(slot.start, 0),
            "cudaEventRecord GPU frame start")) {
        return false;
    }

    m_active = true;
    return true;
}

void CudaGpuTimer::end() {
    if (!m_initialized || !m_active) {
        return;
    }

    Slot& slot = m_slots[m_current];

    if (!cuda_ok(
            cudaEventRecord(slot.end, 0),
            "cudaEventRecord GPU frame end")) {
        m_active = false;
        return;
    }

    slot.pending = true;
    m_active = false;

    m_current = (m_current + 1) % kSlots;
}

bool CudaGpuTimer::take_latest(double& ms) {
    if (!m_has_latest) {
        return false;
    }

    ms = m_latest_ms;
    m_has_latest = false;
    return true;
}

} // namespace gfx
