#include "core/log.h"
#include "core/timer.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "core/mesh.h"
#include "core/obj_loader.h"
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
    gfx::Mesh mesh;
    const bool has_model = model_path != nullptr;
    gfx::Aabb bounds{};
    gfx::Transform model_transform;
    gfx::TransformEditor editor;

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

        if (!gfx::load_obj(resolved, mesh, error)) {
            LOG_ERROR("failed to load model: %s", error.c_str());
            return 1;
        }

        gfx::mesh_y_up_to_z_up(mesh);

        const bool normals_generated = gfx::ensure_normals(mesh);

        gfx::ensure_edges(mesh);

        if (normals_generated) {
            LOG_INFO("normals generated (flat)");
        } else {
            LOG_INFO("normals in file: yes");
        }

        LOG_INFO(
            "loaded model: %zu vertices, %zu triangles",
            mesh.positions.size(),
            mesh.triangle_count()
        );

        gfx::mesh_center_at_origin(mesh);
        bounds = gfx::compute_bounds(mesh);

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

    bool vsync = true;
    bool wireframe = false;
    bool ray_trace = false;

    // Sun for the ray tracer, as two angles in degrees: azimuth around
    // the Z axis (0 = +X, 90 = +Y) and elevation above the floor.
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

    gfx::MeshId mesh_id = 0;

    if (has_model) {
        const float radius =
            0.5f * gfx::length(bounds.max - bounds.min);

        camera.frame(model_transform.position, radius, kFovY);

        mesh_id = renderer->create_mesh(mesh);

        if (mesh_id == 0) {
            LOG_ERROR("failed to upload model to the renderer");
            return 1;
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
        frame.ray_trace = ray_trace;
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
            char view_text[96];

            if (wireframe) {
                std::snprintf(view_text, sizeof(view_text),
                              "View: wireframe");
            } else if (ray_trace &&
                       backend == gfx::RendererBackend::GFX_BACKEND_CUDA) {
                std::snprintf(view_text, sizeof(view_text),
                              "View: ray traced");
            } else {
                std::snprintf(view_text, sizeof(view_text), "View: solid");
            }

            ui.begin_panel("gfxlab", 10.0f, 10.0f, 280.0f);

            ui.text(renderer->name());
            ui.text(stats_text);
            ui.text(view_text);

            if (editor.active()) {
                char edit_text[48];
                std::snprintf(edit_text, sizeof(edit_text), "Edit: %s %s",
                              mode_name(editor.mode()),
                              axis_name(editor.axis()));
                ui.text(edit_text);
            }

            ui.separator();

            if (ui.checkbox("Wireframe", &wireframe)) {
                LOG_INFO("wireframe: %s", wireframe ? "on" : "off");
            }

            if (ui.checkbox("Vsync", &vsync)) {
                renderer->set_vsync(vsync);
                LOG_INFO("vsync: %s", vsync ? "on" : "off");
            }

            // Ray tracing exists only in the CUDA backend; its options
            // only matter while it is on
            const bool is_cuda =
                backend == gfx::RendererBackend::GFX_BACKEND_CUDA;

            ui.separator();
            ui.set_enabled(is_cuda);

            if (ui.checkbox("Ray tracing", &ray_trace)) {
                LOG_INFO("ray tracing: %s", ray_trace ? "on" : "off");
            }

            ui.set_enabled(is_cuda && ray_trace);

            if (ui.checkbox("Sun", &sun_enabled)) {
                LOG_INFO("sun: %s", sun_enabled ? "on" : "off");
            }

            ui.set_enabled(is_cuda && ray_trace && sun_enabled);

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

            ui.end_panel();
        }

        ui.end_frame();

        frame.ui = &ui_list;

        // The application owns the transform. The renderer receives
        // the resulting matrices and draws the mesh
        if (mesh_id != 0) {
            frame.draw_list.push_back({
                mesh_id,
                model_transform.model_matrix(),
                model_transform.normal_matrix()
            });
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

            char title[192];

            // Only the CUDA backend ray traces; say so in the title,
            // together with the sun's angles
            char mode_suffix[64] = "";

            if (ray_trace && !wireframe &&
                backend == gfx::RendererBackend::GFX_BACKEND_CUDA) {
                const char* ao_text = ambient_occlusion ? ", AO" : "";

                if (sun_enabled) {
                    std::snprintf(
                        mode_suffix,
                        sizeof(mode_suffix),
                        " (ray traced, sun %.0f / %.0f deg%s)",
                        sun_azimuth,
                        sun_elevation,
                        ao_text);
                } else {
                    std::snprintf(
                        mode_suffix,
                        sizeof(mode_suffix),
                        " (ray traced, sun off%s)",
                        ao_text);
                }
            }

            if (editor.active()) {
                std::snprintf(
                    title,
                    sizeof(title),
                    "gfxlab | %s%s | %.1f FPS | frame %.3f ms | %s | %s%s%s",
                    renderer->name(),
                    mode_suffix,
                    timer.fps(),
                    frame_ms,
                    gpu_text,
                    mode_name(editor.mode()),
                    axis_name(editor.axis())[0] != '\0' ? " " : "",
                    axis_name(editor.axis()));
            } else {
                std::snprintf(
                    title,
                    sizeof(title),
                    "gfxlab | %s%s | %.1f FPS | frame %.3f ms | %s",
                    renderer->name(),
                    mode_suffix,
                    timer.fps(),
                    frame_ms,
                    gpu_text);
            }

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

        // Right-button drag moves the sun while ray tracing: left/right
        // turns it around the scene, up/down raises and lowers it.
        // Right click also cancels a G/R/S edit, so a drag only starts
        // when the button goes down with no edit active
        if (in.mouse_pressed(gfx::MouseButton::GFX_MOUSE_RIGHT)) {
            sun_dragging =
                ray_trace && sun_enabled && !wireframe &&
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

        // Toggle ray tracing with T (CUDA backend only)
        if (in.key_pressed(gfx::Key::GFX_KEY_T)) {
            ray_trace = !ray_trace;

            if (backend == gfx::RendererBackend::GFX_BACKEND_CUDA) {
                LOG_INFO("ray tracing: %s", ray_trace ? "on" : "off");
            } else {
                LOG_INFO("ray tracing is only available with "
                         "--backend=cuda");
                ray_trace = false;
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
