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

// Ambient occlusion: rays per pixel added each frame, and the total
// after which the result is considered converged and no more are traced
constexpr int kAoRaysPerFrame = 4;
constexpr unsigned int kAoMaxSamples = 1024;

// Measuring aid: when true, ambient occlusion restarts every frame, so
// every frame traces kAoRaysPerFrame AO rays per pixel and its cost can
// be timed. The picture stays noisy. Keep false for normal use
constexpr bool kBenchmarkAo = false;

// true: the ray tracer runs as three passes with one thread per shadow
// or AO ray (wavefront). false: one thread per pixel does all of its
// rays (megakernel). Same picture; switch to compare their GPU times
constexpr bool kRtWavefront = true;

// AO rays reach this fraction of the largest model's diagonal
constexpr float kAoRadiusFraction = 0.2f;

void append_matrix(std::vector<float>& out, const Mat4& m) {
    for (int c = 0; c < 4; ++c) {
        out.push_back(m.col[c].x);
        out.push_back(m.col[c].y);
        out.push_back(m.col[c].z);
        out.push_back(m.col[c].w);
    }
}

// Length of the world-space diagonal of an object-space box after the
// model transform (all eight corners, since rotation changes the extent)
float world_diagonal(const DeviceMesh& mesh, const Mat4& model) {
    Vec3 lo{1.0e30f, 1.0e30f, 1.0e30f};
    Vec3 hi{-1.0e30f, -1.0e30f, -1.0e30f};

    for (int i = 0; i < 8; ++i) {
        const Vec3 corner{
            (i & 1) ? mesh.bounds_max.x : mesh.bounds_min.x,
            (i & 2) ? mesh.bounds_max.y : mesh.bounds_min.y,
            (i & 4) ? mesh.bounds_max.z : mesh.bounds_min.z
        };

        const Vec3 p = transform_point(model, corner);

        lo = {std::fmin(lo.x, p.x), std::fmin(lo.y, p.y),
              std::fmin(lo.z, p.z)};
        hi = {std::fmax(hi.x, p.x), std::fmax(hi.y, p.y),
              std::fmax(hi.z, p.z)};
    }

    return length(hi - lo);
}

constexpr bool kDebugTriangleIds = false;
constexpr bool kProfileStages = false;

} // namespace

CudaRenderer::~CudaRenderer() {
    free_raster_work();
    free_wavefront();

    if (m_ao_accum != nullptr) {
        (void)cuda_ok(
            cudaFree(m_ao_accum),
            "cudaFree ambient occlusion buffer");
        m_ao_accum = nullptr;
    }

    if (m_raster.counts != nullptr) {
        (void)cuda_ok(
            cudaFree(m_raster.counts),
            "cudaFree raster counters");
        m_raster.counts = nullptr;
    }

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

    // The OpenGL context must be current before
    // querying interop devices
    if (!cuda_select_gl_device()) {
        return false;
    }

    // The worklist rasterizers launch a fixed grid sized by the SM count
    int device = 0;

    if (!cuda_ok(cudaGetDevice(&device), "cudaGetDevice") ||
        !cuda_ok(
            cudaDeviceGetAttribute(
                &m_sm_count,
                cudaDevAttrMultiProcessorCount,
                device),
            "cudaDeviceGetAttribute SM count")) {
        return false;
    }

    // The counters are tiny and live for the whole renderer
    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_raster.counts),
                sizeof(RasterCounts)),
            "cudaMalloc raster counters")) {
        m_raster.counts = nullptr;
        return false;
    }

    // Frame timing and ray statistics are optional;
    // keep rendering if event creation fails
    (void)m_gpu_timer.init();
    (void)m_ray_stats.init();

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

    // The AO accumulation is per pixel, so a new size starts it over
    if (m_ao_accum != nullptr) {
        if (!cuda_ok(
                cudaFree(m_ao_accum),
                "cudaFree old ambient occlusion buffer")) {
            return false;
        }
        m_ao_accum = nullptr;
    }

    m_ao_samples = 0;
    m_rt_signature.clear();

    free_wavefront();

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

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_ao_accum),
                pixels * sizeof(float)),
            "cudaMalloc ambient occlusion buffer")) {
        (void)cuda_ok(cudaFree(m_visibility),
                      "cudaFree visibility buffer after AO failure");
        (void)cuda_ok(cudaFree(m_color),
                      "cudaFree colour buffer after AO failure");
        m_color = nullptr;
        m_visibility = nullptr;
        m_ao_accum = nullptr;
        return false;
    }

    // Wavefront buffers: at most one camera-ray hit per pixel
    if (!cuda_ok(cudaMalloc(reinterpret_cast<void**>(&m_wave.hits),
                            pixels * sizeof(RtHit)),
                 "cudaMalloc ray tracer hits") ||
        !cuda_ok(cudaMalloc(
                     reinterpret_cast<void**>(&m_wave.shadow_blocked),
                     pixels * sizeof(unsigned int)),
                 "cudaMalloc ray tracer shadow results") ||
        !cuda_ok(cudaMalloc(
                     reinterpret_cast<void**>(&m_wave.ao_escaped),
                     pixels * sizeof(unsigned int)),
                 "cudaMalloc ray tracer AO results") ||
        !cuda_ok(cudaMalloc(reinterpret_cast<void**>(&m_wave.counts),
                            sizeof(RtWavefrontCounts)),
                 "cudaMalloc ray tracer hit count")) {
        free_wavefront();
        return false;
    }

    m_wave.capacity = static_cast<unsigned int>(pixels);

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

void CudaRenderer::free_wavefront() {
    if (m_wave.hits != nullptr) {
        (void)cuda_ok(cudaFree(m_wave.hits), "cudaFree ray tracer hits");
    }

    if (m_wave.shadow_blocked != nullptr) {
        (void)cuda_ok(cudaFree(m_wave.shadow_blocked),
                      "cudaFree ray tracer shadow results");
    }

    if (m_wave.ao_escaped != nullptr) {
        (void)cuda_ok(cudaFree(m_wave.ao_escaped),
                      "cudaFree ray tracer AO results");
    }

    if (m_wave.counts != nullptr) {
        (void)cuda_ok(cudaFree(m_wave.counts),
                      "cudaFree ray tracer hit count");
    }

    m_wave = {};
}

void CudaRenderer::free_raster_work() {
    if (m_raster.records != nullptr) {
        (void)cuda_ok(
            cudaFree(m_raster.records),
            "cudaFree raster records");
        m_raster.records = nullptr;
    }

    if (m_raster.small_list != nullptr) {
        (void)cuda_ok(
            cudaFree(m_raster.small_list),
            "cudaFree small triangle list");
        m_raster.small_list = nullptr;
    }

    if (m_raster.large_list != nullptr) {
        (void)cuda_ok(
            cudaFree(m_raster.large_list),
            "cudaFree large triangle list");
        m_raster.large_list = nullptr;
    }

    m_raster.capacity = 0;
}

bool CudaRenderer::ensure_raster_capacity(
    std::uint32_t record_count) {

    if (record_count <= m_raster.capacity) {
        return true;
    }

    // Grow-only, like the scratch buffer. The counters are kept
    free_raster_work();

    const std::size_t count = record_count;

    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_raster.records),
                count * sizeof(RasterTri)),
            "cudaMalloc raster records") ||
        !cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_raster.small_list),
                count * sizeof(unsigned int)),
            "cudaMalloc small triangle list") ||
        !cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_raster.large_list),
                count * sizeof(unsigned int)),
            "cudaMalloc large triangle list")) {
        free_raster_work();
        return false;
    }

    m_raster.capacity = record_count;
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

    // Start before clearing colour and visibility
    (void)m_gpu_timer.begin();

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

    // Solid mode: clipping can split one triangle into up to six.
    // Few triangles are clipped in practice, so reserve twice the
    // triangle count plus headroom; the overflow counter catches the
    // rare worst case
    // Ray tracing replaces rasterization in solid mode only;
    // wireframe always uses the line rasterizer
    const bool trace = frame.ray_trace && !frame.wireframe;

    const bool solid =
        !frame.wireframe && !trace && total_triangles > 0;

    if (solid) {
        const std::uint64_t record_capacity =
            total_triangles * 2ULL + 1024ULL;

        // Zero is reserved as the invalid visibility ID
        if (record_capacity >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) - 1ULL) {
            LOG_ERROR("CUDA renderer: worklist capacity overflow");
            return;
        }

        if (!ensure_raster_capacity(
                static_cast<std::uint32_t>(record_capacity)) ||
            !cuda_reset_raster_counts(m_raster.counts)) {
            return;
        }
    }

    // Ray tracing: collect the instances; no vertex work is needed.
    // The signature records everything that changes the traced picture,
    // so ambient occlusion can keep accumulating only while it is stable
    RtScene rt_scene{};
    std::vector<float> signature;
    float largest_diagonal = 0.0f;

    if (trace) {
        for (const DrawItem& item : frame.draw_list) {
            const DeviceMesh* mesh = m_meshes.find(item.mesh);

            if (mesh == nullptr || mesh->bvh_node_count == 0) {
                continue;
            }

            if (rt_scene.count >= kMaxRtInstances) {
                if (!m_rt_limit_reported) {
                    LOG_WARN(
                        "CUDA ray tracer: only the first %d draw items "
                        "are ray traced",
                        kMaxRtInstances);
                    m_rt_limit_reported = true;
                }
                break;
            }

            RtInstance& inst = rt_scene.items[rt_scene.count];

            // A model scaled to zero on an axis cannot be inverted;
            // it has no visible surface either, so skip it
            if (!mat4_inverse(item.model, inst.world_to_object)) {
                continue;
            }

            inst.mesh = *mesh;
            inst.normal_to_world = item.normal_matrix;
            ++rt_scene.count;

            append_matrix(signature, item.model);
            largest_diagonal = std::fmax(
                largest_diagonal, world_diagonal(*mesh, item.model));
        }
    }

    // Transform every draw item, then draw its edges (wireframe)
    // or append its triangles to the worklist (solid)
    std::size_t vertex_offset = 0;

    for (const DrawItem& item : frame.draw_list) {
        if (trace) {
            break;  // handled by the ray tracer below
        }

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
                // Setup only: rasterization runs once for all items
                if (!cuda_setup_triangles(
                        m_raster,
                        m_width,
                        m_height,
                        *mesh,
                        item_vertices)) {
                    return;
                }
            }
        }

        vertex_offset += mesh->vertex_count;
    }

    // Rasterize the whole worklist in one go
    if (solid &&
        !cuda_rasterize_worklist(
            visibility,
            m_width,
            m_raster,
            m_sm_count)) {
        return;
    }

    // Profile marker 2: transform and rasterize have been submitted
    if constexpr (kProfileStages) {
        m_stage_timer.mark(2);
    }

    // Pass 3: resolve wire pixels or shade solid triangles
    if (trace) {
        const GridRayBasis rays = make_grid_ray_basis(frame, aspect);

        rt_scene.view = frame.view;
        rt_scene.view_proj = view_proj;
        rt_scene.eye = rays.eye;
        rt_scene.forward = rays.forward;
        rt_scene.right = rays.right;
        rt_scene.up = rays.up;
        rt_scene.up_view = transform_direction(
            frame.view, Vec3{0.0f, 0.0f, 1.0f});

        // Rays start at the near plane and stop at the far plane,
        // exactly the depth range the rasterizer keeps
        rt_scene.t_min = frame.near_plane;
        rt_scene.t_max = frame.far_plane;

        if (frame.sun_enabled) {
            // The sun is fixed in the world: shadow rays use it as it
            // is, the lighting needs it in view space like the normals
            rt_scene.sun_world = normalize(frame.sun_dir);
            rt_scene.sun_view = normalize(
                transform_direction(frame.view, rt_scene.sun_world));
            rt_scene.shadows = true;
        } else {
            // Sun off: light like the rasterizer, without shadows
            rt_scene.sun_view = key_light_view_dir();
            rt_scene.sun_world = rt_scene.sun_view;
            rt_scene.shadows = false;
        }

        // Ambient occlusion: restart whenever anything that affects the
        // picture changed since the last frame, otherwise keep adding
        // rays until kAoMaxSamples, then reuse the converged result
        append_matrix(signature, frame.view);
        append_matrix(signature, view_proj);
        signature.push_back(rt_scene.sun_world.x);
        signature.push_back(rt_scene.sun_world.y);
        signature.push_back(rt_scene.sun_world.z);
        signature.push_back(frame.sun_enabled ? 1.0f : 0.0f);
        signature.push_back(frame.ambient_occlusion ? 1.0f : 0.0f);

        if (signature != m_rt_signature || kBenchmarkAo) {
            m_rt_signature = signature;
            m_ao_samples = 0;
        }

        rt_scene.ao = frame.ambient_occlusion;
        rt_scene.ao_accum = m_ao_accum;
        rt_scene.ao_samples_before = m_ao_samples;
        rt_scene.ao_rays =
            (frame.ambient_occlusion && m_ao_samples < kAoMaxSamples)
                ? kAoRaysPerFrame
                : 0;
        rt_scene.ao_radius = kAoRadiusFraction * largest_diagonal;

        if (rt_scene.count > 0) {
            rt_scene.ray_counts = m_ray_stats.begin_frame();

            if (!cuda_ray_trace(
                    color,
                    visibility,
                    m_width,
                    m_height,
                    rt_scene,
                    kRtWavefront ? &m_wave : nullptr,
                    m_sm_count)) {
                return;
            }

            m_ray_stats.end_frame();

            m_ao_samples +=
                static_cast<unsigned int>(rt_scene.ao_rays);
        }

        m_ray_stats.report(m_ao_samples, kAoMaxSamples);
    } else if (frame.wireframe) {
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
    } else if (solid) {
        const Vec3 up_view =
            transform_direction(
                frame.view, Vec3{0.0f, 0.0f, 1.0f});

        // One pass: every pixel shades the record that won it
        if (!cuda_shade(
                color,
                visibility,
                m_width,
                m_height,
                m_raster,
                up_view)) {
            return;
        }
    }

    // Profile marker 3: shading or wire resolution has been submitted
    if constexpr (kProfileStages) {
        m_stage_timer.mark(3);
    }

    // Pass 4: draw the floor grid over the shaded scene
    const GridRayBasis grid = make_grid_ray_basis(frame, aspect);

    CudaGridParams grid_params;
    grid_params.view_proj = view_proj;
    grid_params.eye = grid.eye;
    grid_params.forward = grid.forward;
    grid_params.right = grid.right;
    grid_params.up = grid.up;
    grid_params.fade_end = grid.fade_end;

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

    // The device is idle here, so reading the counters costs nothing extra
    if (solid && !m_overflow_reported) {
        RasterCounts counts{};

        if (cuda_ok(
                cudaMemcpy(
                    &counts,
                    m_raster.counts,
                    sizeof(counts),
                    cudaMemcpyDeviceToHost),
                "cudaMemcpy raster counters") &&
            counts.overflow_count > 0) {
            LOG_WARN(
                "CUDA worklist overflow: %u triangles dropped "
                "(capacity %u)",
                counts.overflow_count,
                m_raster.capacity);
            m_overflow_reported = true;
        }
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

    // Include CUDA/OpenGL interop copy cost in the frame measurement
    // The small OpenGL presentation blit is intentionally excluded
    m_gpu_timer.end();

    m_presenter.draw(m_width, m_height);
    m_context.swap_buffers();
}

void CudaRenderer::set_vsync(bool enabled) {
    m_context.set_vsync(enabled);
}

bool CudaRenderer::last_gpu_ms(double& ms) {
    return m_gpu_timer.take_latest(ms);
}

MeshId CudaRenderer::create_mesh(const Mesh& mesh) {
    return m_meshes.add(mesh);
}

} // namespace gfx
