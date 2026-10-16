#include "core/log.h"
#include "core/timer.h"
#include "core/orbit_camera.h"
#include "core/math_util.h"
#include "core/mesh.h"
#include "core/obj_loader.h"
#include "platform/window.h"
#include "renderer/renderer.h"

#include <algorithm>
#include <cstdio>
#include <memory>

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

    if (has_model) {
        std::string error;

        if (!gfx::load_obj(argv[1], mesh, error)) {
            LOG_ERROR("failed to load model: %s", error.c_str());
            return 1;
        }

        gfx::mesh_y_up_to_z_up(mesh);

        const bool normals_generated = gfx::ensure_normals(mesh);

        // Log after ensure_normals(), because it may change the vertex count
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
        const gfx::Aabb bounds = gfx::compute_bounds(mesh);

        const gfx::Vec3 center =
            (bounds.min + bounds.max) * 0.5f;

        const float radius =
            0.5f * gfx::length(bounds.max - bounds.min);

        camera.frame(center, radius, kFovY);

        mesh_id = renderer->create_mesh(mesh);

        if (mesh_id == 0) {
            LOG_ERROR("failed to upload model to the renderer");
            return 1;
        }
    }

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
        frame.demo_mesh = mesh_id;

        renderer->render(frame);

        if (timer.fps_updated() && timer.fps() > 0.0) {
            char title[96];

            std::snprintf(
                title,
                sizeof(title),
                "gfxlab | %.1f FPS | %.4f ms",
                timer.fps(),
                1000.0 / timer.fps()
            );

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

        if (in.key_pressed(gfx::Key::GFX_KEY_ESCAPE)) {
            window.request_close();
        }

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
