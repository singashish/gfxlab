#include "renderer/cuda/cuda_smoke.h"
#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

#include <cuda_runtime.h>
#include <array>
#include <cstddef>

namespace gfx {
namespace {

constexpr int kElementCount = 1024;
constexpr int kThreadsPerBlock = 256;

__global__ void smoke_kernel(int* data, int n) {
    const int i =
        static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

    if (i >= n) {
        return;
    }

    data[i] = i * 2;
}

} // namespace

bool cuda_run_smoke_test() {
    LOG_INFO("CUDA smoke test started");

    int* device_data = nullptr;

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&device_data),
                sizeof(int) * kElementCount),
            "cudaMalloc")) {
        return false;
    }

    bool success = true;

    const int blocks =
        (kElementCount + kThreadsPerBlock - 1) /
        kThreadsPerBlock;

    smoke_kernel<<<blocks, kThreadsPerBlock>>>(
        device_data, kElementCount);

    // Check 1: Did the kernel launch successfully?
    if (!cuda_ok(cudaGetLastError(), "smoke_kernel launch")) {
        success = false;
    }

    // Check 2: Did the kernel finish without a runtime error?
    if (!cuda_ok(cudaDeviceSynchronize(), "cudaDeviceSynchronize")) {
        success = false;
    }

    std::array<int, kElementCount> host_data{};

    if (success) {
        if (!cuda_ok(
                cudaMemcpy(
                    host_data.data(),
                    device_data,
                    sizeof(int) * kElementCount,
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy device to host")) {
            success = false;
        }
    }

    if (success) {
        for (int i = 0; i < kElementCount; ++i) {
            const int expected = i * 2;

            if (host_data[static_cast<std::size_t>(i)] != expected) {
                LOG_ERROR(
                    "CUDA smoke test mismatch at index %d: "
                    "expected %d, got %d",
                    i,
                    expected,
                    host_data[static_cast<std::size_t>(i)]
                );

                success = false;
                break;
            }
        }
    }

    // Release the allocation even when an earlier operation failed
    if (!cuda_ok(cudaFree(device_data), "cudaFree")) {
        success = false;
    }

    if (success) {
        LOG_INFO(
            "CUDA smoke test PASSED: verified %d elements",
            kElementCount
        );
    } else {
        LOG_ERROR("CUDA smoke test FAILED");
    }

    return success;
}

} // namespace gfx
