#include "renderer/cuda/cuda_ui.h"

#include "renderer/cuda/cuda_api.h"
#include "core/log.h"

#include <algorithm>
#include <cstring>

namespace gfx {

namespace {

// Smallest triangle capacity; growing doubles it
constexpr std::size_t kMinTriangles = 1024;

std::size_t grown_capacity(std::size_t current, std::size_t needed) {
    return std::max({needed, current * 2, kMinTriangles});
}

} // namespace

CudaUiRenderer::~CudaUiRenderer() {
    shutdown();
}

void CudaUiRenderer::shutdown() {
    for (Slot& slot : m_slots) {
        if (slot.copied != nullptr) {
            // The pinned memory must not be freed while a copy reads it
            (void)cuda_ok(cudaEventSynchronize(slot.copied),
                          "cudaEventSynchronize UI copy");
            (void)cuda_ok(cudaEventDestroy(slot.copied),
                          "cudaEventDestroy UI copy");
        }

        if (slot.host != nullptr) {
            (void)cuda_ok(cudaFreeHost(slot.host),
                          "cudaFreeHost UI triangles");
        }

        slot = {};
    }

    if (m_device_tris != nullptr) {
        (void)cuda_ok(cudaFree(m_device_tris), "cudaFree UI triangles");
        m_device_tris = nullptr;
    }

    m_device_capacity = 0;

    if (m_atlas != nullptr) {
        (void)cuda_ok(cudaFree(m_atlas), "cudaFree UI font atlas");
        m_atlas = nullptr;
    }

    m_font = nullptr;
    m_atlas_width = 0;
    m_atlas_height = 0;
    m_current = 0;
}

bool CudaUiRenderer::upload_font(const FontAtlas& font) {
    if (m_atlas != nullptr) {
        // Earlier frames' UI kernels may still read the old atlas. This
        // happens at most a few times per run, so just wait
        (void)cuda_ok(cudaDeviceSynchronize(),
                      "cudaDeviceSynchronize before replacing the UI font");
        (void)cuda_ok(cudaFree(m_atlas), "cudaFree old UI font atlas");
        m_atlas = nullptr;
    }

    m_font = nullptr;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&m_atlas),
                            font.alpha.size()),
                 "cudaMalloc UI font atlas")) {
        m_atlas = nullptr;
        return false;
    }

    // A one-time synchronous copy
    if (!cuda_ok(cudaMemcpy(m_atlas, font.alpha.data(), font.alpha.size(),
                            cudaMemcpyHostToDevice),
                 "cudaMemcpy UI font atlas")) {
        return false;
    }

    m_font = &font;
    m_atlas_width = font.width;
    m_atlas_height = font.height;

    LOG_INFO("CUDA UI font uploaded: %dx%d", font.width, font.height);
    return true;
}

bool CudaUiRenderer::ensure_device_capacity(std::size_t count) {
    if (count <= m_device_capacity) {
        return true;
    }

    // Kernels queued earlier may still read the old buffer; cudaFree
    // waits for the device before releasing it
    if (m_device_tris != nullptr) {
        (void)cuda_ok(cudaFree(m_device_tris), "cudaFree UI triangles");
        m_device_tris = nullptr;
    }

    const std::size_t capacity = grown_capacity(m_device_capacity, count);
    m_device_capacity = 0;

    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&m_device_tris),
                            capacity * sizeof(UiRasterTri)),
                 "cudaMalloc UI triangles")) {
        m_device_tris = nullptr;
        return false;
    }

    m_device_capacity = capacity;
    return true;
}

bool CudaUiRenderer::prepare_slot(Slot& slot, std::size_t count) {
    if (slot.copied == nullptr &&
        !cuda_ok(cudaEventCreateWithFlags(&slot.copied,
                                          cudaEventDisableTiming),
                 "cudaEventCreate UI copy")) {
        slot.copied = nullptr;
        return false;
    }

    // The previous copy out of this slot must have run before the CPU
    // overwrites it. Two frames later it almost always has
    if (slot.pending) {
        if (!cuda_ok(cudaEventSynchronize(slot.copied),
                     "cudaEventSynchronize UI copy")) {
            return false;
        }

        slot.pending = false;
    }

    if (count <= slot.capacity) {
        return true;
    }

    if (slot.host != nullptr) {
        (void)cuda_ok(cudaFreeHost(slot.host), "cudaFreeHost UI triangles");
        slot.host = nullptr;
    }

    const std::size_t capacity = grown_capacity(slot.capacity, count);
    slot.capacity = 0;

    if (!cuda_ok(cudaMallocHost(reinterpret_cast<void**>(&slot.host),
                                capacity * sizeof(UiRasterTri)),
                 "cudaMallocHost UI triangles")) {
        slot.host = nullptr;
        return false;
    }

    slot.capacity = capacity;
    return true;
}

bool CudaUiRenderer::draw(uchar4* color,
                          int width,
                          int height,
                          const UiDrawList& list) {
    const FontAtlas* font = list.font();

    if (list.empty() || font == nullptr || color == nullptr) {
        return true;  // nothing to draw
    }

    if (font != m_font && !upload_font(*font)) {
        return false;
    }

    // 1. Triangle setup on the CPU
    const UiRasterBounds bounds =
        build_ui_raster_tris(list, width, height, m_tris);

    if (bounds.empty()) {
        return true;  // everything was off screen
    }

    const std::size_t count = m_tris.size();
    const std::size_t bytes = count * sizeof(UiRasterTri);

    // 2. Stage the triangles in pinned memory and queue the copy
    Slot& slot = m_slots[static_cast<std::size_t>(m_current)];
    m_current = (m_current + 1) % kSlots;

    if (!prepare_slot(slot, count) || !ensure_device_capacity(count)) {
        return false;
    }

    std::memcpy(slot.host, m_tris.data(), bytes);

    if (!cuda_ok(cudaMemcpyAsync(m_device_tris, slot.host, bytes,
                                 cudaMemcpyHostToDevice, 0),
                 "cudaMemcpyAsync UI triangles") ||
        !cuda_ok(cudaEventRecord(slot.copied, 0),
                 "cudaEventRecord UI copy")) {
        return false;
    }

    slot.pending = true;

    // 3. Composite. Same stream, so it runs after the copy and after
    // the scene kernels queued before it
    return cuda_draw_ui(color, width, height, m_device_tris,
                        static_cast<int>(count), m_atlas,
                        m_atlas_width, m_atlas_height, bounds);
}

} // namespace gfx
