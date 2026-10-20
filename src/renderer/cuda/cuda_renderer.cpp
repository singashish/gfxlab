#include "renderer/cuda/cuda_renderer.h"

#include <cuda_runtime.h>

#include "renderer/cuda/cuda_api.h"
#include "renderer/cuda/cuda_raster.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <cstdint>
#include <limits>
#include <vector>
#include <cmath>

namespace gfx {

namespace {

constexpr bool kDebugTriangleIds = false;
constexpr bool kProfileStages = false;

} // namespace

CudaRenderer::~CudaRenderer() {
    if (m_scratch != nullptr) {
        (void)cuda_ok(
            cudaFree(m_scratch),
            "cudaFree projected vertex scratch buffer");
        m_scratch = nullptr;
        m_scratch_capacity = 0;
    }

    if (m_visibility != nullptr) {
        (void)cuda_ok(
            cudaFree(m_visibility),
            "cudaFree visibility buffer");
        m_visibility = nullptr;
    }

    if (m_color != nullptr) {
        (void)cuda_ok(
            cudaFree(m_color),
            "cudaFree renderer colour buffer");
        m_color = nullptr;
    }

    m_width = 0;
    m_height = 0;
}

bool CudaRenderer::init(void* native_window) {
    if (!m_context.create(native_window)) {
        return false;
    }

    // The OpenGL context must be current before querying interop devices
    if (!cuda_select_gl_device()) {
        return false;
    }

    if constexpr (kProfileStages) {
        if (!m_stage_timer.init()) {
            return false;
        }
    }

    if (!m_presenter.init()) {
        return false;
    }

    LOG_INFO("CUDA renderer initialized");
    return true;
}

bool CudaRenderer::resize_targets(int width, int height) {
    m_width = 0;
    m_height = 0;

    if (width <= 0 || height <= 0) {
        return false;
    }

    // Release the old registration before changing the texture
    m_image.unregister();

    if (!m_presenter.resize(width, height)) {
        return false;
    }

    if (!m_image.register_texture(m_presenter.texture())) {
        return false;
    }

    if (m_color != nullptr) {
        if (!cuda_ok(cudaFree(m_color), "cudaFree old colour buffer")) {
            return false;
        }
        m_color = nullptr;
    }

    if (m_visibility != nullptr) {
        if (!cuda_ok(
                cudaFree(m_visibility),
                "cudaFree old visibility buffer")) {
            return false;
        }
        m_visibility = nullptr;
    }

    const std::size_t pixels =
        static_cast<std::size_t>(width) *
        static_cast<std::size_t>(height);

    if (!cuda_ok(
            cudaMalloc(&m_color, pixels * sizeof(uchar4)),
            "cudaMalloc colour buffer")) {
        m_color = nullptr;
        return false;
    }

    if (!cuda_ok(
            cudaMalloc(
                &m_visibility,
                pixels * sizeof(std::uint64_t)),
            "cudaMalloc visibility buffer")) {
        (void)cuda_ok(
            cudaFree(m_color),
            "cudaFree colour buffer after visibility allocation failure");
        m_color = nullptr;
        m_visibility = nullptr;
        return false;
    }

    m_width = width;
    m_height = height;

    LOG_INFO("CUDA render targets resized: %dx%d", width, height);
    return true;
}

bool CudaRenderer::ensure_scratch_capacity(
    std::uint32_t vertex_count) {

    if (vertex_count <= m_scratch_capacity) {
        return true;
    }

    if (m_scratch != nullptr) {
        if (!cuda_ok(
                cudaFree(m_scratch),
                "cudaFree old projected vertex scratch buffer")) {
            return false;
        }

        m_scratch = nullptr;
        m_scratch_capacity = 0;
    }

    if (vertex_count == 0) {
        return true;
    }

    const std::size_t bytes =
        static_cast<std::size_t>(vertex_count) *
        sizeof(ProjectedVertex);

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_scratch),
                bytes),
            "cudaMalloc projected vertex scratch buffer")) {
        m_scratch = nullptr;
        return false;
    }

    m_scratch_capacity = vertex_count;
    return true;
}

void CudaRenderer::render(const FrameData& frame) {
    if (frame.width <= 0 || frame.height <= 0) {
        return;
    }

    if (frame.width != m_width || frame.height != m_height) {
        if (!resize_targets(frame.width, frame.height)) {
            LOG_ERROR("Failed to resize CUDA render targets");
            return;
        }
    }

    auto* color = static_cast<uchar4*>(m_color);
    auto* visibility = static_cast<std::uint64_t*>(m_visibility);

    if (!color || !visibility) {
        LOG_ERROR("CUDA renderer: render targets are not allocated");
        return;
    }

    // Profile marker 0: start of measured GPU work
    if constexpr (kProfileStages) {
        m_stage_timer.mark(0);
    }

    // Pass 1: clear colour and visibility
    const uchar4 background = make_uchar4(26, 28, 33, 255);

    if (!cuda_clear(color, m_width, m_height, background) ||
        !cuda_clear_visibility(visibility, m_width, m_height)) {
        return;
    }

    // Profile marker 1: both clears have been submitted
    if constexpr (kProfileStages) {
        m_stage_timer.mark(1);
    }

    // Pass 2: build the same projection used by GLRenderer
    const float aspect =
        static_cast<float>(frame.width) /
        static_cast<float>(frame.height);

    const Mat4 projection = perspective(
        frame.fov_y,
        aspect,
        frame.near_plane,
        frame.far_plane,
        ClipSpace::CLIP_SPACE_OPENGL);

    const Mat4 view_proj = projection * frame.view;

    struct DrawRecord {
        const DeviceMesh* mesh;
        ProjectedVertex* vertices;
        std::uint32_t triangle_base;
    };

    // Calculate total vertex capacity and validate triangle IDs
    std::size_t total_vertices = 0;
    std::uint64_t total_triangles = 0;

    for (const DrawItem& item : frame.draw_list) {
        const DeviceMesh* mesh = m_meshes.find(item.mesh);

        if (mesh == nullptr) {
            continue;
        }

        if (mesh->index_count % 3 != 0) {
            LOG_ERROR("CUDA renderer: index count is not a triangle list");
            return;
        }

        if (mesh->vertex_count > 0 && !mesh->positions) {
            LOG_ERROR("CUDA renderer: mesh has no position buffer");
            return;
        }

        if (mesh->index_count > 0 && !mesh->indices) {
            LOG_ERROR("CUDA renderer: mesh has no index buffer");
            return;
        }

        if (mesh->vertex_count >
            std::numeric_limits<std::size_t>::max() - total_vertices) {
            LOG_ERROR("CUDA renderer: total vertex count overflow");
            return;
        }

        total_vertices += mesh->vertex_count;
        total_triangles += mesh->index_count / 3;

        if (total_vertices >
            std::numeric_limits<std::uint32_t>::max()) {
            LOG_ERROR("CUDA renderer: scratch vertex count overflow");
            return;
        }

        // Zero is reserved as the invalid visibility ID
        if (total_triangles >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) - 1ULL) {
            LOG_ERROR("CUDA renderer: triangle ID range overflow");
            return;
        }
    }

    if (total_vertices > 0 &&
        !ensure_scratch_capacity(
            static_cast<std::uint32_t>(total_vertices))) {
        return;
    }

    std::vector<DrawRecord> records;
    records.reserve(frame.draw_list.size());

    // Transform and rasterize every draw item
    std::size_t vertex_offset = 0;
    std::uint32_t triangle_base = 0;

    for (const DrawItem& item : frame.draw_list) {
        const DeviceMesh* mesh = m_meshes.find(item.mesh);

        if (mesh == nullptr) {
            continue;
        }

        ProjectedVertex* item_vertices =
            mesh->vertex_count > 0
                ? m_scratch + vertex_offset
                : nullptr;

        if (mesh->vertex_count > 0) {
            const Mat4 mvp = view_proj * item.model;
            const Mat4 model_view = frame.view * item.model;
            const Mat4 normal_to_view =
                frame.view * item.normal_matrix;

            if (!cuda_transform_vertices(
                    *mesh,
                    mvp,
                    model_view,
                    normal_to_view,
                    frame.width,
                    frame.height,
                    item_vertices)) {
                return;
            }
        }

        if (mesh->index_count > 0) {
            if (!item_vertices) {
                LOG_ERROR("CUDA renderer: indexed mesh has no vertices");
                return;
            }

            if (frame.wireframe) {
                // Wireframe uses the same transformed vertices,
                // but writes edge depths instead of triangle IDs
                if (!cuda_draw_edges(
                        visibility,
                        m_width,
                        m_height,
                        *mesh,
                        item_vertices)) {
                    return;
                }
            } else {
                // Solid mode retains the existing deferred shading path
                records.push_back({
                    mesh,
                    item_vertices,
                    triangle_base
                });

                if (!cuda_rasterize_triangles(
                        visibility,
                        m_width,
                        m_height,
                        *mesh,
                        item_vertices,
                        triangle_base)) {
                    return;
                }
            }
        }

        vertex_offset += mesh->vertex_count;
        triangle_base += mesh->index_count / 3;
    }

    // Profile marker 2: transform and rasterize have been submitted
    if constexpr (kProfileStages) {
        m_stage_timer.mark(2);
    }

    // Pass 3: resolve wire pixels or shade solid triangles
    if (frame.wireframe) {
        if (!cuda_resolve_wire(
                color,
                visibility,
                m_width,
                m_height)) {
            return;
        }
    } else if constexpr (kDebugTriangleIds) {
        if (!cuda_resolve_ids(
                color, visibility, m_width, m_height)) {
            return;
        }
    } else {
        const Vec3 up_view =
            transform_direction(
                frame.view, Vec3{0.0f, 0.0f, 1.0f});

        for (const DrawRecord& record : records) {
            if (!cuda_shade_item(
                    color,
                    visibility,
                    m_width,
                    m_height,
                    *record.mesh,
                    record.vertices,
                    record.triangle_base,
                    up_view)) {
                return;
            }
        }
    }

    // Profile marker 3: shading or wire resolution has been submitted
    if constexpr (kProfileStages) {
        m_stage_timer.mark(3);
    }

    // Pass 4: draw the floor grid over the shaded scene
    const float tan_half = std::tan(frame.fov_y * 0.5f);

    CudaGridParams grid_params;
    grid_params.view_proj = view_proj;
    grid_params.eye = frame.eye;
    grid_params.forward = frame.basis.forward;
    grid_params.right =
        frame.basis.right * (tan_half * aspect);
    grid_params.up = frame.basis.up * tan_half;
    grid_params.fade_end = frame.grid_fade_end;

    if (!cuda_draw_grid(
            color,
            visibility,
            m_width,
            m_height,
            grid_params)) {
        return;
    }

    // Profile marker 4: grid has been submitted
    // end_frame waits for this event and reports the stage timings
    if constexpr (kProfileStages) {
        m_stage_timer.mark(4);
        m_stage_timer.end_frame();
    }

#ifndef NDEBUG
    // Catch asynchronous kernel execution errors once per frame
    if (!cuda_ok(
            cudaDeviceSynchronize(),
            "CUDA frame execution")) {
        return;
    }
#endif

    // Present through the existing CUDA/OpenGL interop path
    if (!m_image.copy_from_device(
            color,
            m_width,
            m_height,
            static_cast<std::size_t>(m_width) * sizeof(uchar4))) {
        return;
    }

    m_presenter.draw(m_width, m_height);
    m_context.swap_buffers();
}

void CudaRenderer::set_vsync(bool enabled) {
    m_context.set_vsync(enabled);
}

MeshId CudaRenderer::create_mesh(const Mesh& mesh) {
    return m_meshes.add(mesh);
}

} // namespace gfx
