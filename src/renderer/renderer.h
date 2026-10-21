#pragma once

#include "core/camera_math.h"
#include "core/mat4.h"
#include "core/vec3.h"
#include "core/mesh.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

using MeshId = std::uint32_t; // 0 means no mesh

struct DrawItem {
    MeshId mesh = 0;
    Mat4 model = mat4_identity();
    Mat4 normal_matrix = mat4_identity();
};

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

    // Render polygon edges instead of filled triangles
    bool wireframe = false;

    // Temporary demo animation state, replaced by a draw list in M3
    float demo_spin_angle = 0.0f;

    // Object instances to render
    std::vector<DrawItem> draw_list;
};

// Everything the ray-cast floor grid needs from the camera.
// The ray through a pixel is forward + ndc_x * right + ndc_y * up,
// so right and up are pre-scaled to the size of the view at
// distance 1. Shared by all three backends so the maths lives once
struct GridRayBasis {
    Vec3 eye;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float fade_end;
};

// aspect is passed in because a backend may render at a size that
// differs from frame.width / frame.height for a frame (the Vulkan
// swapchain extent, for example)
inline GridRayBasis make_grid_ray_basis(const FrameData& frame,
                                        float aspect) {
    const float tan_half = std::tan(frame.fov_y * 0.5f);

    GridRayBasis basis;
    basis.eye = frame.eye;
    basis.forward = frame.basis.forward;
    basis.right = frame.basis.right * (tan_half * aspect);
    basis.up = frame.basis.up * tan_half;
    basis.fade_end = frame.grid_fade_end;
    return basis;
}

class IRenderer {
public:
    virtual ~IRenderer() = default;

    virtual const char* name() const = 0;
    virtual bool init(void* native_window) = 0;
    virtual void render(const FrameData& frame) = 0;
    // GPU time of the most recent finished frame, in milliseconds
    // Returns false if no new measurement is available yet
    virtual bool last_gpu_ms(double& ms) {
        (void)ms;
        return false;
    }
    virtual void set_vsync(bool enabled) = 0;
    virtual MeshId create_mesh(const Mesh& mesh) = 0;
};

// Returns nullptr and logs an error when the backend is unavailable
std::unique_ptr<IRenderer> create_renderer(RendererBackend backend);

} // namespace gfx
