#pragma once

#include "core/camera_math.h"
#include "core/mat4.h"
#include "core/vec3.h"

#include <memory>

namespace gfx {

enum class RendererBackend {
    GFX_BACKEND_OPENGL,
    GFX_BACKEND_VULKAN,
    GFX_BACKEND_CUDA
};

struct FrameData {
    // Framebuffer dimensions in pixels
    int width = 0;
    int height = 0;

    // Backend-neutral camera information
    Mat4 view = mat4_identity();
    Vec3 eye = {0.0f, 0.0f, 0.0f};
    CameraBasis basis = {};

    // Projection parameters. FOV is in radians
    float fov_y = 1.0f;
    float near_plane = 0.1f;
    float far_plane = 1000.0f;

    // World distance at which the grid completely fades
    float grid_fade_end = 100.0f;

    // Temporary demo animation state, replaced by a draw list in M3
    float demo_spin_angle = 0.0f;
};

class IRenderer {
public:
    virtual ~IRenderer() = default;

    virtual const char* name() const = 0;
    virtual bool init(void* native_window) = 0;
    virtual void render(const FrameData& frame) = 0;
    virtual void set_vsync(bool enabled) = 0;
};

// Returns nullptr and logs an error when the backend is unavailable
std::unique_ptr<IRenderer> create_renderer(RendererBackend backend);

} // namespace gfx
