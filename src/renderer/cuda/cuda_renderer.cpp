#include "renderer/cuda/cuda_renderer.h"

#include <cuda_runtime.h>

#include "renderer/cuda/cuda_api.h"
#include "renderer/cuda/cuda_raster.h"
#include "core/light_sampling.h"
#include "core/log.h"
#include "core/path_settings.h"

#include <algorithm>
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

// The ray tracer can run two ways (same picture):
//  - megakernel: one thread per pixel does all of its rays
//  - wavefront:  three passes, one thread per shadow or AO ray
// Measured on suzanne_s: the wavefront makes refining AO ~3x cheaper,
// but costs ~0.02 ms extra when there are no AO rays to spread out. So
// with kRtWavefront true it is used only while AO is refining, and
// the megakernel otherwise. false: always the megakernel
constexpr bool kRtWavefront = true;

// true: rays walk the 4-wide BVH; false: the binary one. Both give the
// same picture; switch to compare their GPU times
constexpr bool kUseBvh4 = true;

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

// Fills in what both ray-based modes share: the camera, the near and far
// plane, the BVH choice and the materials switch (the caller sets the
// material and texture tables). Rays start at the near plane and stop at
// the far plane, exactly the depth range the rasterizer keeps
void set_rt_camera(RtScene& scene,
                   const FrameData& frame,
                   const Mat4& view_proj,
                   float aspect) {
    const GridRayBasis rays = make_grid_ray_basis(frame, aspect);

    scene.view = frame.view;
    scene.view_proj = view_proj;
    scene.eye = rays.eye;
    scene.forward = rays.forward;
    scene.right = rays.right;
    scene.up = rays.up;
    scene.up_view = transform_direction(
        frame.view, Vec3{0.0f, 0.0f, 1.0f});

    scene.t_min = frame.near_plane;
    scene.t_max = frame.far_plane;

    scene.use_bvh4 = kUseBvh4;
    scene.surfaces.show_materials = frame.show_materials;
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

    if (m_pt_accum != nullptr) {
        (void)cuda_ok(
            cudaFree(m_pt_accum),
            "cudaFree path tracer accumulation buffer");
        m_pt_accum = nullptr;
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
    // The presenter goes first: it picks the CUDA device that can share
    // memory with the display API
    m_presenter = create_cuda_presenter();

    if (!m_presenter || !m_presenter->init(native_window)) {
        LOG_ERROR("CUDA renderer: no presenter for the window");
        return false;
    }

    LOG_INFO("CUDA presenter: %s", m_presenter->name());

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

    // Texture 0 is a 1x1 white one that materials without a texture
    // use, so it has to exist before any mesh is added
    if (!m_textures.init()) {
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

    LOG_INFO("CUDA renderer initialized");
    return true;
}

bool CudaRenderer::resize_targets(int width, int height) {
    m_width = 0;
    m_height = 0;

    if (width <= 0 || height <= 0) {
        return false;
    }

    if (!m_presenter || !m_presenter->resize(width, height)) {
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

    // Same for the path tracer's accumulation
    if (m_pt_accum != nullptr) {
        if (!cuda_ok(
                cudaFree(m_pt_accum),
                "cudaFree old path tracer accumulation buffer")) {
            return false;
        }
        m_pt_accum = nullptr;
    }

    m_ao_samples = 0;
    m_rt_signature.clear();
    m_pt_samples = 0;
    m_pt_signature.clear();

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

    // Path tracer accumulation: a float4 per pixel. The kernel ignores
    // the contents when it starts over, so it needs no clearing
    if (!cuda_ok(
            cudaMalloc(
                reinterpret_cast<void**>(&m_pt_accum),
                pixels * sizeof(float4)),
            "cudaMalloc path tracer accumulation buffer")) {
        (void)cuda_ok(cudaFree(m_ao_accum),
                      "cudaFree AO buffer after path tracer failure");
        (void)cuda_ok(cudaFree(m_visibility),
                      "cudaFree visibility buffer after path tracer "
                      "failure");
        (void)cuda_ok(cudaFree(m_color),
                      "cudaFree colour buffer after path tracer failure");
        m_color = nullptr;
        m_visibility = nullptr;
        m_ao_accum = nullptr;
        m_pt_accum = nullptr;
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
    // The path tracer ("Rendered" mode) takes priority over the ray
    // tracer; both gather the same instances
    const bool pathtrace = frame.path_trace && !frame.wireframe;
    const bool trace =
        frame.ray_trace && !frame.path_trace && !frame.wireframe;
    const bool ray_based = trace || pathtrace;

    const bool solid =
        !frame.wireframe && !ray_based && total_triangles > 0;

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

    if (ray_based) {
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
            inst.object_to_world = item.model;

            // Area lights exist only while materials are shown (emission
            // is off otherwise)
            inst.light_weight = frame.show_materials
                ? area_scale(item.model) * mesh->light_weight
                : 0.0f;
            rt_scene.light_total += inst.light_weight;
            ++rt_scene.count;

            append_matrix(signature, item.model);

            // Which meshes are drawn (the path tracer's restart check)
            if (pathtrace) {
                signature.push_back(static_cast<float>(item.mesh));
            }

            largest_diagonal = std::fmax(
                largest_diagonal, world_diagonal(*mesh, item.model));
        }
    }

    // Transform every draw item, then draw its edges (wireframe)
    // or append its triangles to the worklist (solid)
    std::size_t vertex_offset = 0;

    for (const DrawItem& item : frame.draw_list) {
        if (ray_based) {
            break;  // handled by the ray or path tracer below
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

    m_last_frame_traced = trace;
    m_last_frame_path_traced = pathtrace;

    // Leaving the path tracer: the next time starts a fresh accumulation
    if (!pathtrace) {
        m_pt_samples = 0;
        m_pt_signature.clear();
    }

    // Pass 3: resolve wire pixels or shade solid triangles
    if (pathtrace) {
        set_rt_camera(rt_scene, frame, view_proj, aspect);

        rt_scene.surfaces.materials = m_meshes.device_materials();
        rt_scene.surfaces.textures = m_textures.device_table();

        // The sun is fixed in the world. Its direction is all the path
        // tracer takes from the scene; sun_enabled goes in PtParams
        rt_scene.sun_world = normalize(frame.sun_dir);

        // Restart whenever anything that affects the picture changed
        // since the last frame: the camera, the models' transforms (in
        // the signature already), the sun, the materials switch, and the
        // set of meshes and textures (a mesh or texture uploaded after
        // the first frame changes the scene). A new window size already
        // restarted in resize_targets
        append_matrix(signature, frame.view);
        append_matrix(signature, view_proj);
        signature.push_back(rt_scene.sun_world.x);
        signature.push_back(rt_scene.sun_world.y);
        signature.push_back(rt_scene.sun_world.z);
        signature.push_back(frame.sun_enabled ? 1.0f : 0.0f);
        signature.push_back(frame.show_materials ? 1.0f : 0.0f);
        signature.push_back(static_cast<float>(m_meshes.count()));
        signature.push_back(static_cast<float>(m_textures.count()));

        // A material edit changes the colours and the lamps
        signature.push_back(static_cast<float>(m_meshes.revision()));

        // The settings that change the collected light restart too; not
        // the exposure (applied when the sums are resolved) and not the
        // sample counts (they only decide when to stop)
        const PathTraceSettings settings =
            sanitized(frame.path_settings);

        signature.push_back(static_cast<float>(settings.max_bounces));
        signature.push_back(settings.sun_strength);
        signature.push_back(settings.sky_strength);
        signature.push_back(settings.firefly_limit ? 1.0f : 0.0f);

        // Just switched on: the signature was cleared above
        if (signature != m_pt_signature) {
            m_pt_signature = signature;
            m_pt_samples = 0;
        }

        PtParams pt;
        pt.accum = m_pt_accum;
        pt.samples_before = m_pt_samples;

        // Samples are added until the limit; a limit below the count
        // adds nothing (the picture stays), a raised one continues
        const unsigned int sample_limit =
            static_cast<unsigned int>(settings.max_samples);
        const unsigned int per_frame =
            static_cast<unsigned int>(settings.samples_per_frame);

        m_pt_max_samples = sample_limit;
        pt.samples_to_add = m_pt_samples < sample_limit
            ? std::min(per_frame, sample_limit - m_pt_samples)
            : 0u;
        pt.max_bounces = settings.max_bounces;
        pt.exposure = kPtExposure * std::exp2(settings.exposure_ev);
        pt.sun_enabled = frame.sun_enabled;
        pt.sun_irradiance = settings.sun_strength;
        pt.sky_scale = settings.sky_strength;
        pt.sun_cos_half_angle = sun_cone_cos(kSunHalfAngle);
        pt.indirect_clamp =
            settings.firefly_limit ? kPtIndirectClamp : 1.0e30f;

        // Camera rays that miss see the same background colour the clear
        // pass painted (one variable, so they cannot drift apart), as the
        // linear radiance that the display pipeline turns back into it.
        // The sky and the sun only light the scene, they are not drawn.
        // It is made for the base exposure, not the exposure setting:
        // the setting is applied when the sums are resolved, so the
        // background (like everything else) brightens or darkens with it
        // and the accumulated sums stay valid when only it changes
        pt.background = background_radiance(
            Vec3{static_cast<float>(background.x) / 255.0f,
                 static_cast<float>(background.y) / 255.0f,
                 static_cast<float>(background.z) / 255.0f},
            kPtExposure);

        // Runs even with nothing to draw: the picture is then the plain
        // background
        if (!cuda_path_trace(
                color, visibility, m_width, m_height, rt_scene, pt)) {
            return;
        }

        m_pt_samples += pt.samples_to_add;
    } else if (trace) {
        set_rt_camera(rt_scene, frame, view_proj, aspect);

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

        rt_scene.surfaces.materials = m_meshes.device_materials();
        rt_scene.surfaces.textures = m_textures.device_table();

        // Wavefront only when it pays off: while AO rays are being added
        const bool use_wavefront =
            kRtWavefront && rt_scene.ao && rt_scene.ao_rays > 0;

        if (rt_scene.count > 0) {
            rt_scene.ray_counts = m_ray_stats.begin_frame();

            if (!cuda_ray_trace(
                    color,
                    visibility,
                    m_width,
                    m_height,
                    rt_scene,
                    use_wavefront ? &m_wave : nullptr,
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

        MaterialTables tables;
        tables.materials = m_meshes.device_materials();
        tables.textures = m_textures.device_table();
        tables.show_materials = frame.show_materials;

        // One pass: every pixel shades the record that won it
        if (!cuda_shade(
                color,
                visibility,
                m_width,
                m_height,
                m_raster,
                tables,
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

    // Pass 5: the UI on top of the finished picture. Drawn into the
    // colour buffer here, so it does not depend on the presenter
    if (frame.ui != nullptr &&
        !m_ui.draw(color, m_width, m_height, *frame.ui)) {
        return;
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

    // Hand the finished frame to the presenter
    if (!m_presenter->copy(color, m_width, m_height)) {
        return;
    }

    // Include the presenter's copy in the frame measurement.
    // The small presentation blit is intentionally excluded
    m_gpu_timer.end();

    m_presenter->present(m_width, m_height);
}

void CudaRenderer::set_vsync(bool enabled) {
    if (m_presenter) {
        m_presenter->set_vsync(enabled);
    }
}

bool CudaRenderer::last_gpu_ms(double& ms) {
    return m_gpu_timer.take_latest(ms);
}

RendererStats CudaRenderer::stats() const {
    RendererStats stats;
    stats.ray_traced = m_last_frame_traced;
    stats.path_traced = m_last_frame_path_traced;

    if (m_last_frame_path_traced) {
        stats.pt_samples = m_pt_samples;
        stats.pt_max_samples = m_pt_max_samples;
    }

    if (m_last_frame_traced) {
        stats.has_ray_counts = m_ray_stats.latest(
            stats.camera_rays, stats.shadow_rays, stats.ao_rays);
        stats.ao_samples = m_ao_samples;
        stats.ao_max_samples = kAoMaxSamples;
    }

    return stats;
}

TextureId CudaRenderer::create_texture(const Image& image) {
    return m_textures.create(image);
}

MeshId CudaRenderer::create_mesh(const Mesh& mesh,
                                 const std::vector<TextureId>& textures) {
    return m_meshes.add(mesh, textures, m_textures);
}

bool CudaRenderer::update_material(MeshId mesh, int material_index,
                                   const MaterialParams& params) {
    return m_meshes.update_material(mesh, material_index, params);
}

} // namespace gfx
