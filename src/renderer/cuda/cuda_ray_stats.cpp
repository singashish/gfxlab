#include "renderer/cuda/cuda_ray_stats.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

namespace gfx {

namespace {

constexpr std::size_t kCounterBytes = 3 * sizeof(unsigned long long);

} // namespace

CudaRayStats::~CudaRayStats() {
    for (Slot& slot : m_slots) {
        if (slot.device != nullptr) {
            (void)cuda_ok(cudaFree(slot.device),
                          "cudaFree ray counters");
        }

        if (slot.host != nullptr) {
            (void)cuda_ok(cudaFreeHost(slot.host),
                          "cudaFreeHost ray counters");
        }

        if (slot.copied != nullptr) {
            (void)cuda_ok(cudaEventDestroy(slot.copied),
                          "cudaEventDestroy ray counters");
        }

        slot = {};
    }

    m_initialized = false;
}

bool CudaRayStats::init() {
    if (m_initialized) {
        return true;
    }

    for (Slot& slot : m_slots) {
        // Pinned (page-locked) host memory: the GPU can copy into it
        // directly, so cudaMemcpyAsync really runs in the background
        if (cudaMalloc(reinterpret_cast<void**>(&slot.device),
                       kCounterBytes) != cudaSuccess ||
            cudaMallocHost(reinterpret_cast<void**>(&slot.host),
                           kCounterBytes) != cudaSuccess ||
            cudaEventCreateWithFlags(&slot.copied,
                                     cudaEventDisableTiming) !=
                cudaSuccess) {
            LOG_WARN("CUDA ray statistics unavailable: "
                     "allocation failed");
            return false;
        }
    }

    m_report_start = std::chrono::steady_clock::now();
    m_initialized = true;
    return true;
}

unsigned long long* CudaRayStats::begin_frame() {
    m_active = false;

    if (!m_initialized) {
        return nullptr;
    }

    Slot& slot = m_slots[m_current];

    if (slot.pending) {
        const cudaError_t query = cudaEventQuery(slot.copied);

        if (query == cudaErrorNotReady) {
            return nullptr;  // still in flight; try again next frame
        }

        if (!cuda_ok(query, "cudaEventQuery ray counters")) {
            return nullptr;
        }

        for (int i = 0; i < 3; ++i) {
            m_sums[i] += static_cast<double>(slot.host[i]);
        }

        ++m_frames;
        slot.pending = false;
    }

    if (!cuda_ok(cudaMemsetAsync(slot.device, 0, kCounterBytes, 0),
                 "cudaMemsetAsync ray counters")) {
        return nullptr;
    }

    m_active = true;
    return slot.device;
}

void CudaRayStats::end_frame() {
    if (!m_active) {
        return;
    }

    Slot& slot = m_slots[m_current];

    if (!cuda_ok(cudaMemcpyAsync(slot.host, slot.device, kCounterBytes,
                                 cudaMemcpyDeviceToHost, 0),
                 "cudaMemcpyAsync ray counters") ||
        !cuda_ok(cudaEventRecord(slot.copied, 0),
                 "cudaEventRecord ray counters")) {
        m_active = false;
        return;
    }

    slot.pending = true;
    m_active = false;
    m_current = (m_current + 1) % kSlots;
}

void CudaRayStats::report(unsigned int ao_samples,
                          unsigned int ao_max_samples) {
    const auto now = std::chrono::steady_clock::now();
    const double elapsed =
        std::chrono::duration<double>(now - m_report_start).count();

    if (elapsed < 1.0) {
        return;
    }

    if (m_frames > 0) {
        const double frames = static_cast<double>(m_frames);

        for (int i = 0; i < 3; ++i) {
            m_latest[i] = m_sums[i] / frames;
        }

        m_has_latest = true;

        LOG_INFO(
            "CUDA rays per frame: camera %.2f M, shadow %.2f M, "
            "AO %.2f M (AO samples %u / %u)",
            m_sums[0] / frames * 1.0e-6,
            m_sums[1] / frames * 1.0e-6,
            m_sums[2] / frames * 1.0e-6,
            ao_samples,
            ao_max_samples);
    }

    m_sums.fill(0.0);
    m_frames = 0;
    m_report_start = now;
}

bool CudaRayStats::latest(double& camera, double& shadow, double& ao) const {
    if (!m_has_latest) {
        return false;
    }

    camera = m_latest[0];
    shadow = m_latest[1];
    ao = m_latest[2];
    return true;
}

} // namespace gfx
