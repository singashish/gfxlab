#include "core/log.h"
#include "core/timer.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "core/mesh.h"
#include "core/model.h"
#include "core/transform.h"
#include "core/transform_edit.h"
#include "platform/ui_font.h"
#include "platform/window.h"
#include "renderer/renderer.h"
#include "ui/ui.h"
#include "ui/ui_draw.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>
#include <string>
#include <filesystem>

namespace {

// Finds a model file. Tries the name as given (relative to the current
// directory, or absolute), then assets/, then assets/local/.
// Every location tried is appended to 'tried'.
// Returns the first existing file, or an empty string if none exists.
std::string resolve_model_path(const std::string& name,
                               std::vector<std::string>& tried) {
    namespace fs = std::filesystem;

    const fs::path requested(name);

    std::vector<fs::path> candidates;
    candidates.push_back(requested);

    // An absolute path is only used as given
    if (!requested.is_absolute()) {
        candidates.push_back(fs::path(GFX_ASSETS_DIR) / requested);
        candidates.push_back(fs::path(GFX_ASSETS_DIR) / "local" / requested);
    }

    for (const fs::path& candidate : candidates) {
        tried.push_back(candidate.string());

        std::error_code ec;   // the error_code overload never throws

        if (fs::is_regular_file(candidate, ec)) {
            return candidate.string();
        }
    }

    return {};
}

} // namespace


int main(int argc, char** argv) {
    LOG_INFO("gfxlab starting...");

#if defined(GFX_HAS_OPENGL)
    LOG_INFO("OpenGL backend compiled in");
#endif

#if defined(GFX_HAS_VULKAN)
    LOG_INFO("Vulkan backend compiled in");
#endif

#if defined(GFX_HAS_CUDA)
    LOG_INFO("CUDA backend compiled in");
#endif

    gfx::RendererBackend backend =
        gfx::RendererBackend::GFX_BACKEND_OPENGL;

    const char* model_path = nullptr;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--backend=vulkan") == 0) {
            backend = gfx::RendererBackend::GFX_BACKEND_VULKAN;
        } else if (std::strcmp(argv[i], "--backend=opengl") == 0) {
            backend = gfx::RendererBackend::GFX_BACKEND_OPENGL;
        } else if (std::strcmp(argv[i], "--backend=cuda") == 0) {
            backend = gfx::RendererBackend::GFX_BACKEND_CUDA;
        } else if (std::strncmp(argv[i], "--", 2) == 0) {
            LOG_ERROR("unknown command-line option: %s", argv[i]);
            return 1;
        } else if (model_path == nullptr) {
            model_path = argv[i];
        } else {
            LOG_ERROR("unexpected extra argument: %s", argv[i]);
            return 1;
        }
    }

    // Load and convert the optional model before creating the window.
    // A Model is one or more meshes, the images their materials use, and
    // the instances placing the meshes (OBJ: one mesh, one instance;
    // glTF: one per node that has a mesh)
    gfx::Model model;
    const bool has_model = model_path != nullptr;
    gfx::Aabb bounds{};
    gfx::Transform model_transform;
    gfx::TransformEditor editor;

    // File name shown in the panel
    std::string model_name;

    if (has_model) {
        std::string error;
        std::vector<std::string> tried;
        const std::string resolved = resolve_model_path(model_path, tried);

        if (resolved.empty()) {
            LOG_ERROR("model not found: %s", model_path);
            for (const std::string& t : tried) {
                LOG_ERROR("  tried: %s", t.c_str());
            }
            return 1;
        }

        LOG_INFO("model: %s", resolved.c_str());

        model_name = std::filesystem::path(resolved).filename().string();

        // Loads, converts to Z-up, generates missing normals and edges,
        // decodes the textures and centres the model on the origin
        if (!gfx::load_model(resolved, model, error)) {
            LOG_ERROR("failed to load model: %s", error.c_str());
            return 1;
        }

        std::size_t vertex_count = 0;
        std::size_t texcoord_meshes = 0;

        for (const gfx::Mesh& mesh : model.meshes) {
            vertex_count += mesh.positions.size();

            if (mesh.has_texcoords()) {
                ++texcoord_meshes;
            }
        }

        LOG_INFO(
            "loaded model: %zu mesh%s, %zu instance%s, %zu vertices, "
            "%zu triangles",
            model.meshes.size(), model.meshes.size() == 1 ? "" : "es",
            model.instances.size(), model.instances.size() == 1 ? "" : "s",
            vertex_count, gfx::model_triangle_count(model)
        );

        if (model.meshes.size() == 1) {
            LOG_INFO("texture coordinates: %s",
                     texcoord_meshes == 1 ? "yes" : "no");
        } else {
            LOG_INFO("texture coordinates: %zu of %zu meshes",
                     texcoord_meshes, model.meshes.size());
        }

        // One line per distinct material name (glTF gives every mesh a
        // copy of the whole list)
        std::vector<std::string> logged_materials;

        for (const gfx::Mesh& mesh : model.meshes) {
            for (const gfx::Material& material : mesh.materials) {
                if (std::find(logged_materials.begin(),
                              logged_materials.end(),
                              material.name) != logged_materials.end()) {
                    continue;
                }

                logged_materials.push_back(material.name);

                LOG_INFO(
                    "material '%s': base colour %.2f %.2f %.2f, roughness "
                    "%.2f, metallic %.2f%s%s",
                    material.name.c_str(),
                    material.base_color.x,
                    material.base_color.y,
                    material.base_color.z,
                    material.roughness,
                    material.metallic,
                    material.base_color_texture.empty() ? "" : ", texture ",
                    material.base_color_texture.c_str());
            }
        }

        // The decoded images (the renderers will use them from Step 3).
        // The material that uses an image knows its name
        std::vector<bool> image_logged(model.images.size(), false);

        for (const gfx::Mesh& mesh : model.meshes) {
            for (const gfx::Material& material : mesh.materials) {
                const int image = material.base_color_image;

                if (image < 0 ||
                    static_cast<std::size_t>(image) >= model.images.size() ||
                    image_logged[static_cast<std::size_t>(image)]) {
                    continue;
                }

                image_logged[static_cast<std::size_t>(image)] = true;

                const gfx::Image& decoded =
                    model.images[static_cast<std::size_t>(image)];

                LOG_INFO("texture %s: %dx%d",
                         material.base_color_texture.c_str(),
                         decoded.width, decoded.height);
            }
        }

        bounds = gfx::model_bounds(model);

        // Stand on the floor: the model is centred on the origin, so
        // half its height lifts the lowest point to z = 0
        model_transform.position.z =
            (bounds.max.z - bounds.min.z) * 0.5f;
    }

    gfx::Window window;

    if (!window.create({"gfxlab", 1280, 720})) {
        LOG_ERROR("failed to create window");
        return 1;
    }

    // The renderer is destroyed before the window
    std::unique_ptr<gfx::IRenderer> renderer = gfx::create_renderer(backend);

    if (!renderer || !renderer->init(window.native_handle())) {
        LOG_ERROR("failed to initialize the renderer");
        return 1;
    }

    LOG_INFO("renderer: %s", renderer->name());

    // UI font and the per-frame UI draw list. Without a font the app
    // still runs, just without the overlay
    gfx::FontAtlas ui_font;

    if (!gfx::build_ui_font(window.native_handle(), "Segoe UI", 12.0f,
                            ui_font)) {
        LOG_WARN("UI font unavailable; the overlay is disabled");
    }

    gfx::UiDrawList ui_list;
    gfx::Ui ui;

    // H shows and hides the panel
    bool show_info_panel = true;

    // True while a middle-button drag that started in the 3D view
    // (not on the UI, and not during a G/R/S edit) orbits or pans
    // the camera
    bool camera_dragging = false;

    // Panel line with the timings, refreshed with the title
    char stats_text[96] = "-- FPS";

    // Ray tracing exists only in the CUDA backend
    const bool is_cuda = backend == gfx::RendererBackend::GFX_BACKEND_CUDA;

    // Which panel sections are unfolded
    bool view_open = true;
    bool model_open = true;
    bool path_tracing_open = is_cuda;
    bool ray_tracing_open = is_cuda;
    bool controls_open = false;

    bool vsync = true;
    bool wireframe = false;
    bool ray_trace = false;

    // The path tracer ("Rendered" mode, CUDA only); P toggles it. It and
    // the ray tracer exclude each other
    bool path_trace = false;

    // Material colours and textures; off = plain grey shading. M toggles it
    bool show_materials = true;

    // Sun for the ray and path tracers, as two angles in degrees: azimuth
    // around the Z axis (0 = +X, 90 = +Y) and elevation above the floor.
    // The default lights the front of the model from the default camera
    float sun_azimuth = 250.0f;
    float sun_elevation = 45.0f;

    // L turns the sun on and off. Off, the ray tracer uses the same
    // camera-attached light as the rasterizer, without shadows
    bool sun_enabled = true;

    // True while a right-button drag is moving the sun
    bool sun_dragging = false;

    // O turns the ray tracer's ambient occlusion on and off
    bool ambient_occlusion = true;

    renderer->set_vsync(vsync);

    gfx::FrameTimer timer;

    // GPU samples used for the title-bar average
    double gpu_sum = 0.0;
    unsigned int gpu_count = 0;

    // Independent accumulators for one-second logging
    double log_gpu_sum = 0.0;
    unsigned int log_gpu_count = 0;
    double log_elapsed_seconds = 0.0;

    bool was_minimized = false;
    float spin_angle = 0.0f;

    constexpr float kNear = 0.1f;
    constexpr float kFar = 1000.0f;
    constexpr float kGridFadeFactor = 2.5f;

    gfx::OrbitCamera camera;
    constexpr float kFovY = gfx::to_radians(60.0f);

    // One renderer mesh per Model mesh; the instances refer to them by
    // index
    std::vector<gfx::MeshId> mesh_ids;

    // Half the model's diagonal: frames the camera, and scales how far a
    // drag in the panel's Position fields moves the model
    const float model_radius =
        has_model ? 0.5f * gfx::length(bounds.max - bounds.min) : 1.0f;

    // Starting transform, for the panel's Reset transform button
    const gfx::Transform initial_transform = model_transform;

    // The start-up view: default angles, framed on the model
    const auto reset_camera = [&]() {
        camera = gfx::OrbitCamera{};

        if (has_model) {
            camera.frame(model_transform.position, model_radius, kFovY);
        }
    };

    if (has_model) {
        reset_camera();

        // Upload every image once; textures[i] belongs to model.images[i]
        // (0 = this backend has no textures, or the upload failed).
        // model.images stays in memory: the path tracer will need it
        std::vector<gfx::TextureId> textures;
        textures.reserve(model.images.size());

        std::size_t textures_uploaded = 0;

        for (const gfx::Image& image : model.images) {
            const gfx::TextureId texture = renderer->create_texture(image);

            if (texture != 0) {
                ++textures_uploaded;
            }

            textures.push_back(texture);
        }

        if (!model.images.empty()) {
            LOG_INFO("textures uploaded: %zu of %zu", textures_uploaded,
                     model.images.size());
        }

        for (const gfx::Mesh& mesh : model.meshes) {
            const gfx::MeshId id = renderer->create_mesh(mesh, textures);

            if (id == 0) {
                LOG_ERROR("failed to upload model to the renderer");
                return 1;
            }

            mesh_ids.push_back(id);
        }
    }

    // Convert the current camera and viewport into backend-independent
    // data for TransformEditor
    auto make_edit_view = [&]() -> gfx::EditView {
        return {
            camera.eye(),
            camera.basis(),
            kFovY,
            static_cast<float>(window.width()),
            static_cast<float>(window.height())
        };
    };

    auto mode_name = [](gfx::EditMode mode) -> const char* {
        switch (mode) {
        case gfx::EditMode::EDIT_GRAB:
            return "GRAB";
        case gfx::EditMode::EDIT_ROTATE:
            return "ROTATE";
        case gfx::EditMode::EDIT_SCALE:
            return "SCALE";
        default:
            return "";
        }
    };

    auto axis_name = [](gfx::EditAxis axis) -> const char* {
        switch (axis) {
        case gfx::EditAxis::EDIT_AXIS_X:
            return "X";
        case gfx::EditAxis::EDIT_AXIS_Y:
            return "Y";
        case gfx::EditAxis::EDIT_AXIS_Z:
            return "Z";
        default:
            return "";
        }
    };

    auto draw_frame = [&]() {
        timer.tick();

        spin_angle +=
            static_cast<float>(timer.delta_seconds());

        gfx::FrameData frame;

        frame.width = window.width();
        frame.height = window.height();
        frame.wireframe = wireframe;
        frame.show_materials = show_materials;
        frame.ray_trace = ray_trace;
        frame.path_trace = path_trace;
        frame.sun_enabled = sun_enabled;
        frame.ambient_occlusion = ambient_occlusion;

        {
            const float az = gfx::to_radians(sun_azimuth);
            const float el = gfx::to_radians(sun_elevation);

            frame.sun_dir = {
                std::cos(el) * std::cos(az),
                std::cos(el) * std::sin(az),
                std::sin(el)
            };
        }

        frame.view = camera.view();
        frame.eye = camera.eye();
        frame.basis = camera.basis();

        frame.fov_y = kFovY;
        frame.near_plane = kNear;
        frame.far_plane = kFar;

        frame.grid_fade_end = std::min(
            kGridFadeFactor * std::max(camera.distance, 10.0f),
            0.9f * kFar
        );

        frame.demo_spin_angle = spin_angle;

        // --------------------------------------------------------
        // UI overlay
        // --------------------------------------------------------

        ui_list.begin(&ui_font);

        // The UI sees the left mouse button, except during a G/R/S edit:
        // then every click belongs to the edit (confirm or cancel)
        gfx::UiInput ui_input;

        if (!editor.active()) {
            const gfx::InputState& in = window.input();

            ui_input.mouse_x = in.mouse_x();
            ui_input.mouse_y = in.mouse_y();
            ui_input.mouse_down =
                in.mouse_down(gfx::MouseButton::GFX_MOUSE_LEFT);
            ui_input.mouse_pressed =
                in.mouse_pressed(gfx::MouseButton::GFX_MOUSE_LEFT);
            ui_input.mouse_released =
                in.mouse_released(gfx::MouseButton::GFX_MOUSE_LEFT);
        }

        ui.begin_frame(ui_list, ui_input);

        if (show_info_panel) {
            ui.begin_panel("gfxlab", 10.0f, 10.0f, 280.0f);

            ui.text(renderer->name());
            ui.text(stats_text);

            // ---- View --------------------------------------------
            if (ui.header("View", &view_open)) {
                if (ui.checkbox("Wireframe", &wireframe)) {
                    LOG_INFO("wireframe: %s", wireframe ? "on" : "off");
                }

                if (ui.checkbox("Materials", &show_materials)) {
                    LOG_INFO("materials: %s",
                             show_materials ? "on" : "off");
                }

                if (ui.checkbox("Vsync", &vsync)) {
                    renderer->set_vsync(vsync);
                    LOG_INFO("vsync: %s", vsync ? "on" : "off");
                }

                if (ui.button("Reset camera")) {
                    reset_camera();
                    LOG_INFO("camera reset");
                }
            }

            // ---- Model -------------------------------------------
            if (has_model && ui.header("Model", &model_open)) {
                // Every glTF mesh holds the whole material list, so the
                // first mesh's count is the model's
                const std::size_t material_count =
                    model.meshes.empty() ? 0
                                         : model.meshes[0].materials.size();

                char info[160];
                int length = std::snprintf(
                    info, sizeof(info), "%s | %zu triangles | %zu material%s",
                    model_name.c_str(), gfx::model_triangle_count(model),
                    material_count, material_count == 1 ? "" : "s");

                if (model.meshes.size() > 1 && length > 0 &&
                    static_cast<std::size_t>(length) < sizeof(info)) {
                    std::snprintf(info + length, sizeof(info) -
                                  static_cast<std::size_t>(length),
                                  " | %zu meshes", model.meshes.size());
                }

                ui.text(info);

                // The fields show the live transform, so during a
                // G/R/S edit they follow it (the UI ignores the mouse
                // then). Rotation is stored in radians, shown in degrees
                float position[3] = {
                    model_transform.position.x,
                    model_transform.position.y,
                    model_transform.position.z
                };

                if (ui.drag_float3("Position", position,
                                   model_radius * 0.005f, "%.2f")) {
                    model_transform.position = {
                        position[0], position[1], position[2]};
                }

                float rotation[3] = {
                    gfx::to_degrees(model_transform.rotation.x),
                    gfx::to_degrees(model_transform.rotation.y),
                    gfx::to_degrees(model_transform.rotation.z)
                };

                if (ui.drag_float3("Rotation", rotation, 0.5f, "%.1f")) {
                    model_transform.rotation = {
                        gfx::to_radians(rotation[0]),
                        gfx::to_radians(rotation[1]),
                        gfx::to_radians(rotation[2])};
                }

                float scale[3] = {
                    model_transform.scale.x,
                    model_transform.scale.y,
                    model_transform.scale.z
                };

                if (ui.drag_float3("Scale", scale, 0.005f, "%.3f")) {
                    model_transform.scale = {scale[0], scale[1], scale[2]};
                }

                if (ui.button("Reset transform")) {
                    model_transform = initial_transform;
                    LOG_INFO("transform reset");
                }

                if (editor.active()) {
                    char edit_text[48];
                    std::snprintf(edit_text, sizeof(edit_text),
                                  "Editing: %s %s",
                                  mode_name(editor.mode()),
                                  axis_name(editor.axis()));
                    ui.text(edit_text);
                }
            }

            // ---- Path tracing (CUDA only) -------------------------
            if (ui.header("Path tracing", &path_tracing_open)) {
                if (!is_cuda) {
                    ui.set_enabled(false);
                    ui.text("Needs --backend=cuda");
                }

                if (ui.checkbox("Path tracing", &path_trace)) {
                    LOG_INFO("path tracing: %s",
                             path_trace ? "on" : "off");

                    // The two tracers exclude each other
                    if (path_trace && ray_trace) {
                        ray_trace = false;
                        LOG_INFO("ray tracing: off");
                    }
                }

                // How far the accumulation has come, from the renderer
                const gfx::RendererStats pt_stats = renderer->stats();

                if (is_cuda && path_trace && pt_stats.path_traced) {
                    char line[64];
                    std::snprintf(line, sizeof(line),
                                  "Samples: %u / %u",
                                  pt_stats.pt_samples,
                                  pt_stats.pt_max_samples);
                    ui.text(line);
                }

                ui.set_enabled(true);
            }

            // ---- Ray tracing (CUDA only) --------------------------
            if (ui.header("Ray tracing", &ray_tracing_open)) {
                if (!is_cuda) {
                    ui.set_enabled(false);
                    ui.text("Needs --backend=cuda");
                }

                if (ui.checkbox("Ray tracing", &ray_trace)) {
                    LOG_INFO("ray tracing: %s", ray_trace ? "on" : "off");

                    // The two tracers exclude each other
                    if (ray_trace && path_trace) {
                        path_trace = false;
                        LOG_INFO("path tracing: off");
                    }
                }

                // The sun controls drive both tracers' sun
                const bool sun_active =
                    is_cuda && (ray_trace || path_trace);

                ui.set_enabled(sun_active);

                if (ui.checkbox("Sun", &sun_enabled)) {
                    LOG_INFO("sun: %s", sun_enabled ? "on" : "off");
                }

                ui.set_enabled(sun_active && sun_enabled);

                ui.slider("Sun azimuth", &sun_azimuth, 0.0f, 360.0f,
                          "%.0f deg");
                ui.slider("Sun elevation", &sun_elevation, 5.0f, 90.0f,
                          "%.0f deg");

                if (ui.button("Reset sun")) {
                    sun_azimuth = 250.0f;
                    sun_elevation = 45.0f;
                }

                ui.set_enabled(is_cuda && ray_trace);

                if (ui.checkbox("Ambient occlusion", &ambient_occlusion)) {
                    LOG_INFO("ambient occlusion: %s",
                             ambient_occlusion ? "on" : "off");
                }

                // What the ray tracer did, from the renderer
                const gfx::RendererStats stats = renderer->stats();

                if (stats.ray_traced) {
                    char line[128];

                    if (stats.has_ray_counts) {
                        std::snprintf(
                            line, sizeof(line),
                            "Rays/frame: camera %.2f M, shadow %.2f M",
                            stats.camera_rays * 1.0e-6,
                            stats.shadow_rays * 1.0e-6);
                        ui.text(line);

                        std::snprintf(line, sizeof(line),
                                      "AO rays/frame: %.2f M",
                                      stats.ao_rays * 1.0e-6);
                        ui.text(line);
                    }

                    if (ambient_occlusion && stats.ao_max_samples > 0) {
                        if (stats.ao_samples >= stats.ao_max_samples) {
                            std::snprintf(line, sizeof(line),
                                          "AO converged (%u samples)",
                                          stats.ao_samples);
                        } else {
                            std::snprintf(line, sizeof(line),
                                          "AO samples: %u / %u",
                                          stats.ao_samples,
                                          stats.ao_max_samples);
                        }

                        ui.text(line);
                    }
                }

                ui.set_enabled(true);
            }

            // ---- Controls (keyboard and mouse help) ---------------
            if (ui.header("Controls", &controls_open)) {
                ui.text("Middle drag: orbit, Shift + middle: pan");
                ui.text("Scroll: zoom");
                ui.text("G / R / S: grab, rotate, scale");
                ui.text("  X / Y / Z: axis, left click: confirm");
                ui.text("  Right click / Esc: cancel");
                ui.text("W wireframe, M materials, V vsync");
                ui.text("H hide panel");
                ui.text("T ray tracing, L sun, O AO (CUDA)");
                ui.text("P path tracing (CUDA, excludes T)");
                ui.text("Right drag: move the sun (CUDA)");
                ui.text("Esc: quit");
            }

            ui.end_panel();
        }

        ui.end_frame();

        frame.ui = &ui_list;

        // The application owns the transform. The renderer receives
        // the resulting matrices and draws the meshes: each instance
        // is placed by the model transform (G/R/S edit the whole model)
        // times its own transform inside the model
        if (!mesh_ids.empty()) {
            const gfx::Mat4 model_matrix = model_transform.model_matrix();

            for (const gfx::ModelInstance& instance : model.instances) {
                if (instance.mesh >= mesh_ids.size()) {
                    continue;
                }

                const gfx::Mat4 world = model_matrix * instance.transform;

                frame.draw_list.push_back({
                    mesh_ids[instance.mesh],
                    world,
                    gfx::normal_matrix_of(world)
                });
            }
        }

        renderer->render(frame);

        double gpu_ms = 0.0;

        if (renderer->last_gpu_ms(gpu_ms)) {
            gpu_sum += gpu_ms;
            ++gpu_count;

            log_gpu_sum += gpu_ms;
            ++log_gpu_count;
        }

        // Update the title at the same cadence as the FPS display
        if (timer.fps_updated() && timer.fps() > 0.0) {
            const double frame_ms = 1000.0 / timer.fps();

            char gpu_text[48];

            if (gpu_count > 0) {
                const double average_gpu_ms =
                    gpu_sum / static_cast<double>(gpu_count);

                std::snprintf(
                    gpu_text,
                    sizeof(gpu_text),
                    "GPU %.3f ms",
                    average_gpu_ms);
            } else {
                std::snprintf(
                    gpu_text,
                    sizeof(gpu_text),
                    "GPU --");
            }

            // The title keeps the essentials; the panel has the rest
            char title[160];
            std::snprintf(title, sizeof(title), "gfxlab | %s | %.1f FPS",
                          renderer->name(), timer.fps());
            window.set_title(title);

            std::snprintf(stats_text, sizeof(stats_text),
                          "%.1f FPS | frame %.3f ms | %s",
                          timer.fps(), frame_ms, gpu_text);

            // Start a fresh averaging interval for the next title update
            gpu_sum = 0.0;
            gpu_count = 0;
        }

        // Print one timing summary approximately every second
        log_elapsed_seconds += timer.delta_seconds();

        if (log_elapsed_seconds >= 1.0) {
            const double frame_ms =
                timer.fps() > 0.0
                    ? 1000.0 / timer.fps()
                    : 0.0;

            if (log_gpu_count > 0) {
                const double average_gpu_ms =
                    log_gpu_sum /
                    static_cast<double>(log_gpu_count);

                LOG_INFO(
                    "%s | frame ms %.3f | GPU ms %.3f",
                    renderer->name(),
                    frame_ms,
                    average_gpu_ms);
            } else {
                LOG_INFO(
                    "%s | frame ms %.3f | GPU ms --",
                    renderer->name(),
                    frame_ms);
            }

            log_gpu_sum = 0.0;
            log_gpu_count = 0;

            // Retain the fractional remainder to reduce timing drift
            log_elapsed_seconds -= 1.0;
        }
    };

    window.set_modal_frame_callback(draw_frame);

    while (!window.should_close()) {
        window.poll_events();

        if (window.is_minimized()) {
            if (!was_minimized) {
                LOG_INFO("window minimized");
                was_minimized = true;
            }

            window.wait_events();
            timer.reset();

            if (window.should_close()) {
                break;
            }

            if (window.is_minimized()) {
                continue;
            }
        }

        was_minimized = false;

        const gfx::InputState& in = window.input();

        // Remember whether an edit was active before processing this
        // frame. Escape cancels an edit first; it must not also quit
        const bool was_editing_at_frame_start = editor.active();

        // ------------------------------------------------------------
        // Camera input
        // ------------------------------------------------------------

        // The mouse belongs to the UI when it was over a panel or holding
        // a widget last frame. During a G/R/S edit the UI ignores the
        // mouse, so the edit always gets it
        const bool ui_has_mouse = ui.wants_mouse() && !editor.active();

        // A middle-button drag orbits (or pans) only if it started in
        // the 3D view; once started it continues even over the UI.
        // During a G/R/S edit the camera is locked: the edit turns mouse
        // movement into a transform using the current view, so moving
        // the view would make the object jump. A drag in progress when
        // an edit starts stops, and does not resume when the edit ends;
        // press the middle button again
        if (in.mouse_pressed(gfx::MouseButton::GFX_MOUSE_MIDDLE)) {
            camera_dragging = !ui_has_mouse && !was_editing_at_frame_start;
        }

        if (!in.mouse_down(gfx::MouseButton::GFX_MOUSE_MIDDLE) ||
            was_editing_at_frame_start) {
            camera_dragging = false;
        }

        if (camera_dragging) {
            if (in.key_down(gfx::Key::GFX_KEY_SHIFT)) {
                camera.pan(
                    in.mouse_dx(),
                    in.mouse_dy(),
                    camera.world_per_pixel(
                        kFovY,
                        static_cast<float>(window.height())
                    )
                );
            } else {
                camera.orbit(in.mouse_dx(), in.mouse_dy());
            }
        }

        // Scrolling over the UI does not zoom
        if (!ui_has_mouse) {
            camera.zoom(in.scroll());
        }

        // Right-button drag moves the sun while ray or path tracing:
        // left/right turns it around the scene, up/down raises and
        // lowers it. Right click also cancels a G/R/S edit, so a drag
        // only starts when the button goes down with no edit active
        if (in.mouse_pressed(gfx::MouseButton::GFX_MOUSE_RIGHT)) {
            sun_dragging =
                (ray_trace || path_trace) && sun_enabled && !wireframe &&
                !was_editing_at_frame_start && !ui_has_mouse;
        }

        if (!in.mouse_down(gfx::MouseButton::GFX_MOUSE_RIGHT)) {
            if (sun_dragging) {
                LOG_INFO("sun: azimuth %.0f deg, elevation %.0f deg",
                         sun_azimuth, sun_elevation);
            }

            sun_dragging = false;
        }

        if (sun_dragging) {
            constexpr float kSunDegreesPerPixel = 0.25f;

            // Dragging right moves a sun on the camera's side of the
            // scene to the right; dragging up raises it
            sun_azimuth += in.mouse_dx() * kSunDegreesPerPixel;
            sun_elevation -= in.mouse_dy() * kSunDegreesPerPixel;

            // Keep the azimuth in [0, 360) and the sun above the floor
            sun_azimuth = std::fmod(sun_azimuth, 360.0f);

            if (sun_azimuth < 0.0f) {
                sun_azimuth += 360.0f;
            }

            sun_elevation =
                std::min(std::max(sun_elevation, 5.0f), 90.0f);
        }

        if (window.was_resized()) {
            const int width = window.width();
            const int height = window.height();

            if (width > 0 && height > 0) {
                const float aspect =
                    static_cast<float>(width) /
                    static_cast<float>(height);

                LOG_INFO(
                    "resized: %dx%d (aspect %.3f)",
                    width,
                    height,
                    aspect
                );
            }
        }

        // Use the camera state after processing camera input
        const gfx::EditView edit_view = make_edit_view();

        // ------------------------------------------------------------
        // Modal G / R / S editing
        // ------------------------------------------------------------

        if (has_model) {
            const auto begin_edit = [&](gfx::EditMode mode) {
                editor.begin(
                    mode,
                    model_transform,
                    in.mouse_x(),
                    in.mouse_y(),
                    edit_view
                );

                LOG_INFO("%s started", mode_name(mode));
            };

            gfx::EditMode requested_mode = gfx::EditMode::EDIT_NONE;

            if (in.key_pressed(gfx::Key::GFX_KEY_G)) {
                requested_mode = gfx::EditMode::EDIT_GRAB;
            } else if (in.key_pressed(gfx::Key::GFX_KEY_R)) {
                requested_mode = gfx::EditMode::EDIT_ROTATE;
            } else if (in.key_pressed(gfx::Key::GFX_KEY_S)) {
                requested_mode = gfx::EditMode::EDIT_SCALE;
            }

            if (!editor.active()) {
                if (requested_mode != gfx::EditMode::EDIT_NONE) {
                    begin_edit(requested_mode);
                }
            } else if (requested_mode != gfx::EditMode::EDIT_NONE) {
                // Switching G / R / S cancels the previous operation
                // and starts the next one from the restored transform
                model_transform = editor.cancel();
                LOG_INFO("cancelled; switching edit mode");

                begin_edit(requested_mode);
            } else {
                // Axis constraints are toggled by pressing the same
                // axis key twice
                if (in.key_pressed(gfx::Key::GFX_KEY_X)) {
                    editor.set_axis(gfx::EditAxis::EDIT_AXIS_X);
                    LOG_INFO("axis: %s", axis_name(editor.axis()));
                } else if (in.key_pressed(gfx::Key::GFX_KEY_Y)) {
                    editor.set_axis(gfx::EditAxis::EDIT_AXIS_Y);
                    LOG_INFO("axis: %s", axis_name(editor.axis()));
                } else if (in.key_pressed(gfx::Key::GFX_KEY_Z)) {
                    editor.set_axis(gfx::EditAxis::EDIT_AXIS_Z);
                    LOG_INFO("axis: %s", axis_name(editor.axis()));
                }
            }

            if (editor.active()) {
                // Calculate the result from the starting transform and
                // current mouse position
                editor.update(
                    in.mouse_x(),
                    in.mouse_y(),
                    edit_view,
                    model_transform
                );

                // Confirm takes precedence if both mouse buttons are
                // reported pressed during the same frame
                if (in.mouse_pressed(
                        gfx::MouseButton::GFX_MOUSE_LEFT)) {
                    editor.confirm();
                    LOG_INFO("confirmed");
                } else if (
                    in.mouse_pressed(
                        gfx::MouseButton::GFX_MOUSE_RIGHT) ||
                    in.key_pressed(gfx::Key::GFX_KEY_ESCAPE)
                ) {
                    model_transform = editor.cancel();
                    LOG_INFO("cancelled");
                }
            }
        }

        // Escape closes the application only if no edit was active
        // at the beginning of this frame. Thus, the first Escape
        // cancels an edit and the next Escape quits
        if (in.key_pressed(gfx::Key::GFX_KEY_ESCAPE) &&
            !was_editing_at_frame_start &&
            !editor.active()) {
            window.request_close();
        }

        // Toggle vsync with V
        if (in.key_pressed(gfx::Key::GFX_KEY_V)) {
            vsync = !vsync;
            renderer->set_vsync(vsync);
            LOG_INFO("vsync: %s", vsync ? "on" : "off");
        }

        // Toggle wireframe with W
        if (in.key_pressed(gfx::Key::GFX_KEY_W)) {
            wireframe = !wireframe;
            LOG_INFO("wireframe: %s", wireframe ? "on" : "off");
        }

        // Toggle material colours and textures with M
        if (in.key_pressed(gfx::Key::GFX_KEY_M)) {
            show_materials = !show_materials;
            LOG_INFO("materials: %s", show_materials ? "on" : "off");
        }

        // Toggle ray tracing with T (CUDA backend only). It turns the
        // path tracer off, and P turns this one off
        if (in.key_pressed(gfx::Key::GFX_KEY_T)) {
            ray_trace = !ray_trace;

            if (backend == gfx::RendererBackend::GFX_BACKEND_CUDA) {
                LOG_INFO("ray tracing: %s", ray_trace ? "on" : "off");

                if (ray_trace && path_trace) {
                    path_trace = false;
                    LOG_INFO("path tracing: off");
                }
            } else {
                LOG_INFO("ray tracing is only available with "
                         "--backend=cuda");
                ray_trace = false;
            }
        }

        // Toggle path tracing with P (CUDA backend only)
        if (in.key_pressed(gfx::Key::GFX_KEY_P)) {
            path_trace = !path_trace;

            if (backend == gfx::RendererBackend::GFX_BACKEND_CUDA) {
                LOG_INFO("path tracing: %s", path_trace ? "on" : "off");

                if (path_trace && ray_trace) {
                    ray_trace = false;
                    LOG_INFO("ray tracing: off");
                }
            } else {
                LOG_INFO("path tracing is only available with "
                         "--backend=cuda");
                path_trace = false;
            }
        }

        // Toggle the ray tracer's ambient occlusion with O
        if (in.key_pressed(gfx::Key::GFX_KEY_O)) {
            ambient_occlusion = !ambient_occlusion;
            LOG_INFO("ambient occlusion: %s",
                     ambient_occlusion ? "on" : "off");
        }

        // Show or hide the info panel with H
        if (in.key_pressed(gfx::Key::GFX_KEY_H)) {
            show_info_panel = !show_info_panel;
            LOG_INFO("panel: %s", show_info_panel ? "shown" : "hidden");
        }

        // Toggle the ray tracer's sun with L
        if (in.key_pressed(gfx::Key::GFX_KEY_L)) {
            sun_enabled = !sun_enabled;
            LOG_INFO("sun: %s", sun_enabled ? "on" : "off");
        }

        // ------------------------------------------------------------
        // Existing key diagnostics
        // ------------------------------------------------------------

        for (int k = 1;
             k < static_cast<int>(gfx::Key::GFX_KEY_COUNT);
             ++k) {
            const gfx::Key key = static_cast<gfx::Key>(k);

            if (in.key_pressed(key)) {
                LOG_INFO("key %d pressed", k);
            }

            if (in.key_released(key)) {
                LOG_INFO("key %d released", k);
            }
        }

        draw_frame();
    }

    window.set_modal_frame_callback({});

    LOG_INFO("shutting down");
    return 0;
}
