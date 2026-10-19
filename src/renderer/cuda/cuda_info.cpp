#include "renderer/cuda/cuda_info.h"
#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

#include <cstddef>

namespace gfx {
namespace {

bool log_version(const char* label, cudaError_t error, int version) {
    if (!cuda_ok(error, label)) {
        return false;
    }

    // CUDA versions are encoded as major * 1000 + minor * 10
    const int major = version / 1000;
    const int minor = (version % 1000) / 10;

    LOG_INFO("%s: %d.%d", label, major, minor);
    return true;
}

} // namespace

bool cuda_log_devices() {
    int runtime_version = 0;
    int driver_version = 0;
    bool success = true;

    const cudaError_t runtime_error =
        cudaRuntimeGetVersion(&runtime_version);

    if (!log_version(
            "CUDA runtime version",
            runtime_error,
            runtime_version)) {
        success = false;
    }

    const cudaError_t driver_error =
        cudaDriverGetVersion(&driver_version);

    if (!log_version(
            "CUDA driver version",
            driver_error,
            driver_version)) {
        success = false;
    }

    int device_count = 0;

    if (!cuda_ok(
            cudaGetDeviceCount(&device_count),
            "cudaGetDeviceCount")) {
        return false;
    }

    if (device_count == 0) {
        LOG_ERROR("CUDA: no NVIDIA CUDA devices found");
        return false;
    }

    LOG_INFO("CUDA devices found: %d", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp properties{};

        if (!cuda_ok(
                cudaGetDeviceProperties(&properties, i),
                "cudaGetDeviceProperties")) {
            success = false;
            continue;
        }

        const double vram_gib =
            static_cast<double>(properties.totalGlobalMem) /
            (1024.0 * 1024.0 * 1024.0);

        LOG_INFO("CUDA device %d: %s", i, properties.name);
        LOG_INFO(
            "  Compute capability: %d.%d",
            properties.major,
            properties.minor
        );
        LOG_INFO(
            "  SM count: %d",
            properties.multiProcessorCount
        );
        LOG_INFO(
            "  Total VRAM: %.2f GiB",
            vram_gib
        );
    }

    return success;
}

} // namespace gfx
