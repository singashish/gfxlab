#include "renderer/cuda/cuda_renderer.h"

#include <cuda_runtime.h>
#include "renderer/cuda/cuda_api.h"
#include "renderer/cuda/cuda_raster.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <limits>

namespace gfx {

CudaRenderer::~CudaRenderer() {
    if (m_scratch != nullptr) {
        (void)cuda_ok(
            cudaFree(m_scratch),
            "cudaFree vertex scratch buffer"
        );
        m_scratch = nullptr;
        m_scratch_capacity = 0;
    }

    if (m_visibility != nullptr) {
        (void)cuda_ok(
            cudaFree(m_visibility),
            "cudaFree visibility buffer"
        );
        m_visibility = nullptr;
    }

    if (m_color != nullptr) {
        (void)cuda_ok(
            cudaFree(m_color),
            "cudaFree renderer colour buffer"
        );
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

    // Release the old colour buffer
    if (m_color != nullptr) {
        if (!cuda_ok(cudaFree(m_color), "cudaFree old colour buffer")) {
            return false;
        }
        m_color = nullptr;
    }

    // Release the old visibility buffer
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
            cudaMalloc(&m_visibility, pixels * sizeof(std::uint64_t)),
            "cudaMalloc visibility buffer")) {
        (void)cuda_ok(
            cudaFree(m_color),
            "cudaFree colour buffer after visibility allocation failure"
        );
        m_color = nullptr;
        m_visibility = nullptr;
        return false;
    }

    m_width = width;
    m_height = height;

    LOG_INFO(
        "CUDA render targets resized: %dx%d",
        width, height
    );

    return true;
}

bool CudaRenderer::ensure_scratch_capacity(
                        std::uint32_t vertex_count) {
    if (vertex_count == 0) {
        return false;
    }

    if (m_scratch != nullptr &&
        m_scratch_capacity >= vertex_count) {
        return true;
    }

    if (m_scratch != nullptr) {
        if (!cuda_ok(
                cudaFree(m_scratch),
                "cudaFree old vertex scratch buffer")) {
            return false;
        }

        m_scratch = nullptr;
        m_scratch_capacity = 0;
    }

    const std::size_t bytes =
        static_cast<std::size_t>(vertex_count) * sizeof(float4);

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_scratch),
                bytes),
            "cudaMalloc vertex scratch buffer")) {
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

    // 1. Clear colour and visibility
    const uchar4 background = make_uchar4(26, 28, 33, 255);

    if (!cuda_clear(color, m_width, m_height, background)) {
        return;
    }

    if (!cuda_clear_visibility(visibility, m_width, m_height)) {
        return;
    }

    // 2. Build the same projection used by GLRenderer
    const float aspect =
        static_cast<float>(frame.width) /
        static_cast<float>(frame.height);

    const Mat4 projection = perspective(
        frame.fov_y,
        aspect,
        frame.near_plane,
        frame.far_plane,
        ClipSpace::CLIP_SPACE_OPENGL
    );

    const Mat4 view_proj = projection * frame.view;

    // 3. Transform vertices and rasterize each draw item
    std::uint32_t base = 0;

    for (const DrawItem& item : frame.draw_list) {
        const DeviceMesh* mesh = m_meshes.find(item.mesh);

        if (mesh == nullptr) {
            continue;
        }

        const std::uint32_t triangle_count = mesh->index_count / 3;

        if (triangle_count >
            std::numeric_limits<std::uint32_t>::max() - base) {
            LOG_ERROR("CUDA rasterizer: triangle ID range overflow");
            return;
        }

        if (!ensure_scratch_capacity(mesh->vertex_count)) {
            return;
        }

        const Mat4 mvp = view_proj * item.model;

        if (!cuda_transform_vertices(
                *mesh,
                mvp,
                m_width,
                m_height,
                m_scratch)) {
            return;
        }

        if (!cuda_rasterize_triangles(
                visibility,
                m_width,
                m_height,
                *mesh,
                m_scratch,
                base)) {
            return;
        }

        base += triangle_count;
    }

    // 4. Convert the winning triangle IDs into diagnostic colours
    if (!cuda_resolve_ids(
            color, visibility, m_width, m_height)) {
        return;
    }

#ifndef NDEBUG
    // Catch asynchronous kernel execution errors once per frame
    if (!cuda_ok(
            cudaDeviceSynchronize(),
            "CUDA frame execution")) {
        return;
    }
#endif

    // 5. Present through the existing CUDA/OpenGL interop path
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
