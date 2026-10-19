#include "core/log.h"
#include "renderer/cuda/cuda_info.h"
#include "renderer/cuda/cuda_smoke.h"


int main() {
    LOG_INFO("GFXLab CUDA diagnostic");

    const bool devices_ok = gfx::cuda_log_devices();
    const bool smoke_ok = gfx::cuda_run_smoke_test();

    return (devices_ok && smoke_ok) ? 0 : 1;
}
