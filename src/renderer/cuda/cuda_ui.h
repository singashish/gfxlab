#pragma once

#include <cuda_runtime.h>

#include "ui/ui_draw.h"
#include "ui/ui_raster.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

// Kernel launcher (cuda_ui.cu): blends tris[0 .. count) over the pixels
// inside bounds, in order. atlas is the font atlas in device memory
bool cuda_draw_ui(uchar4* color,
                  int width,
                  int height,
                  const UiRasterTri* tris,
                  int count,
                  const std::uint8_t* atlas,
                  int atlas_width,
                  int atlas_height,
                  UiRasterBounds bounds);

// Draws a UiDrawList into the CUDA colour buffer, the CUDA counterpart
// of the OpenGL and Vulkan UI passes. The CPU sets up the triangles
// (build_ui_raster_tris) and a kernel composites them, so the result
// does not depend on how the frame is shown afterwards
class CudaUiRenderer {
public:
    CudaUiRenderer() = default;
    ~CudaUiRenderer();

    CudaUiRenderer(const CudaUiRenderer&) = delete;
    CudaUiRenderer& operator=(const CudaUiRenderer&) = delete;

    // Queues the UI on CUDA's default stream, after the frame's other
    // kernels. Returns false only on a CUDA error
    bool draw(uchar4* color, int width, int height, const UiDrawList& list);

    void shutdown();

private:
    // The triangles travel through pinned host memory so the copy runs
    // in the background. Two slots: the CPU fills one while the copy
    // from the other may still be queued; an event says when a slot's
    // copy has been done
    static constexpr int kSlots = 2;

    struct Slot {
        UiRasterTri* host = nullptr;   // pinned
        std::size_t capacity = 0;      // in triangles
        cudaEvent_t copied = nullptr;
        bool pending = false;
    };

    bool upload_font(const FontAtlas& font);
    bool ensure_device_capacity(std::size_t count);
    bool prepare_slot(Slot& slot, std::size_t count);

    // Font atlas in device memory. The app keeps one atlas alive for
    // the whole run, so comparing the pointer is enough
    const FontAtlas* m_font = nullptr;
    std::uint8_t* m_atlas = nullptr;
    int m_atlas_width = 0;
    int m_atlas_height = 0;

    // Triangle setups: built on the CPU, copied to m_device_tris
    std::vector<UiRasterTri> m_tris;
    std::array<Slot, kSlots> m_slots{};
    int m_current = 0;

    UiRasterTri* m_device_tris = nullptr;
    std::size_t m_device_capacity = 0;
};

} // namespace gfx
