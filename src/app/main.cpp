#include "core/log.h"
#include "core/timer.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "core/mesh.h"
#include "core/obj_loader.h"
#include "core/transform.h"
#include "core/transform_edit.h"
#include "platform/window.h"
#include "renderer/renderer.h"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

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

    // Load and convert the optional model before creating the window
    gfx::Mesh mesh;
    const bool has_model = argc > 1;
    gfx::Aabb bounds{};
    gfx::Transform model_transform;
    gfx::TransformEditor editor;

    if (has_model) {
        std::string error;

        if (!gfx::load_obj(argv[1], mesh, error)) {
            LOG_ERROR("failed to load model: %s", error.c_str());
            return 1;
        }

        gfx::mesh_y_up_to_z_up(mesh);

        const bool normals_generated = gfx::ensure_normals(mesh);

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

        // Center the mesh around its local origin
        gfx::mesh_center_at_origin(mesh);
        bounds = gfx::compute_bounds(mesh);

        // Place the bottom of the model on the Z = 0 floor
        model_transform.position.z =
            (bounds.max.z - bounds.min.z) * 0.5f;
    }

    gfx::Window window;

    if (!window.create({"gfxlab", 1280, 720})) {
        LOG_ERROR("failed to create window");
        return 1;
    }

    // The renderer is destroyed before the window
    std::unique_ptr<gfx::IRenderer> renderer =
        gfx::create_renderer(
            gfx::RendererBackend::GFX_BACKEND_OPENGL
        );

    if (!renderer || !renderer->init(window.native_handle())) {
        LOG_ERROR("failed to initialize the renderer");
        return 1;
    }

    LOG_INFO("renderer: %s", renderer->name());
    renderer->set_vsync(true);

    gfx::FrameTimer timer;
    bool was_minimized = false;
    float spin_angle = 0.0f;

    constexpr float kNear = 0.1f;
    constexpr float kFar = 1000.0f;
    constexpr float kGridFadeFactor = 1.5f;

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

        if (timer.fps_updated() && timer.fps() > 0.0) {
            char title[128];

            if (editor.active()) {
                std::snprintf(
                    title,
                    sizeof(title),
                    "gfxlab | %.1f FPS | %.4f ms | %s%s%s",
                    timer.fps(),
                    1000.0 / timer.fps(),
                    mode_name(editor.mode()),
                    axis_name(editor.axis())[0] != '\0' ? " " : "",
                    axis_name(editor.axis())
                );
            } else {
                std::snprintf(
                    title,
                    sizeof(title),
                    "gfxlab | %.1f FPS | %.4f ms",
                    timer.fps(),
                    1000.0 / timer.fps()
                );
            }

            window.set_title(title);
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

        if (in.mouse_down(gfx::MouseButton::GFX_MOUSE_MIDDLE)) {
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

        camera.zoom(in.scroll());

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
