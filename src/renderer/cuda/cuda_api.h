#pragma once

#include "core/log.h"

#include <cuda_runtime.h>

namespace gfx {

inline bool cuda_ok(cudaError_t error, const char* what) {
    if (error == cudaSuccess) {
        return true;
    }

    LOG_ERROR(
        "%s failed: %s (%s)",
        what,
        cudaGetErrorName(error),
        cudaGetErrorString(error)
    );

    return false;
}

} // namespace gfx
