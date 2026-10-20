#pragma once

#include "core/camera_math.h"
#include "core/mat4.h"
#include "core/vec3.h"
#include "core/mesh.h"

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
