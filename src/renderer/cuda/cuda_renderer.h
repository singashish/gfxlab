#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <memory>
#include <vector>

#include "renderer/renderer.h"
#include "renderer/cuda/cuda_gpu_timer.h"
#include "renderer/cuda/cuda_mesh.h"
#include "renderer/cuda/cuda_presenter.h"
#include "renderer/cuda/cuda_raster.h"
#include "renderer/cuda/cuda_ray_stats.h"
#include "renderer/cuda/cuda_texture.h"
#include "renderer/cuda/cuda_timer.h"
#include "renderer/cuda/cuda_ui.h"

namespace gfx {

class CudaRenderer final : public IRenderer {
public:
    CudaRenderer() = default;
    ~CudaRenderer() override;

    CudaRenderer(const CudaRenderer&) = delete;
    CudaRenderer& operator=(const CudaRenderer&) = delete;

    const char* name() const override {
        return "CUDA software rasterizer";
    }

    bool init(void* native_window) override;
    void render(const FrameData& frame) override;
    void set_vsync(bool enabled) override;
    bool last_gpu_ms(double& ms) override;
    RendererStats stats() const override;
    TextureId create_texture(const Image& image) override;
    MeshId create_mesh(const Mesh& mesh,
                       const std::vector<TextureId>& textures) override;

private:
    bool resize_targets(int width, int height);
    bool ensure_scratch_capacity(std::uint32_t vertex_count);
    bool ensure_raster_capacity(std::uint32_t record_count);
    void free_raster_work();

    // Declaration order is intentional: the presenter owns the display
    // API's context, so it is destroyed last. The mesh and texture stores
    // free device memory, so they must go before it
    std::unique_ptr<ICudaPresenter> m_presenter;
    CudaMeshStore m_meshes;
    CudaTextureStore m_textures;
    CudaUiRenderer m_ui;

    CudaStageTimer m_stage_timer;
    CudaGpuTimer m_gpu_timer;
    CudaRayStats m_ray_stats;

    void* m_color = nullptr;
    void* m_visibility = nullptr;

    ProjectedVertex* m_scratch = nullptr;
    std::uint32_t m_scratch_capacity = 0;

    // Triangle worklist: records, small/large lists and GPU counters.
    // m_raster.capacity is the number of records the buffers can hold
    RasterWork m_raster{};

    // Number of SMs, used to size the grid-stride rasterizer kernels
    int m_sm_count = 0;

    // Debug builds report a worklist overflow only once
    bool m_overflow_reported = false;

    // The ray tracer handles up to kMaxRtInstances items; warn once
    bool m_rt_limit_reported = false;

    // Ambient occlusion accumulation: one float per pixel, the number of
    // AO samples per pixel already in it, and a snapshot of everything
    // that affects the picture so a change can restart the accumulation
    float* m_ao_accum = nullptr;
    unsigned int m_ao_samples = 0;
    std::vector<float> m_rt_signature;

    // Buffers for the wavefront ray tracer, one hit per pixel
    RtWavefront m_wave{};
    void free_wavefront();

    // The last frame was ray traced (for stats())
    bool m_last_frame_traced = false;

    // Path tracer accumulation: one float4 per pixel (the sum of the
    // linear radiance of all samples, and the closest camera-hit depth),
    // the samples per pixel already in it, and a snapshot of everything
    // that affects the picture so a change restarts the accumulation.
    // 16 bytes per pixel: 33 MB at 1080p, 133 MB at 4K
    float4* m_pt_accum = nullptr;
    unsigned int m_pt_samples = 0;
    unsigned int m_pt_max_samples = 0;  // the limit of the last frame
    std::vector<float> m_pt_signature;

    // The last frame was path traced (for stats())
    bool m_last_frame_path_traced = false;

    int m_width = 0;
    int m_height = 0;
};

} // namespace gfx
