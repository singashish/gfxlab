#pragma once

#include "core/camera_math.h"
#include "core/image.h"
#include "core/mat4.h"
#include "core/vec3.h"
#include "core/mesh.h"
#include "ui/ui_draw.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

using MeshId = std::uint32_t; // 0 means no mesh
using TextureId = std::uint32_t; // 0 means no texture

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

    // Shade with the materials' colours and base-colour textures.
    // False is the plain grey viewport shading used before materials
    bool show_materials = true;

    // Ray trace the solid view instead of rasterizing it.
    // Only the CUDA backend implements it; the others ignore it.
    // Wireframe takes priority over ray tracing
    bool ray_trace = false;

    // Path trace the solid view ("Rendered" mode): a progressive Monte
    // Carlo renderer with the sun and a procedural sky as lights, refined
    // over frames while nothing changes. Only the CUDA backend implements
    // it; the others ignore it. It takes priority over ray_trace, and
    // wireframe takes priority over both. Uses sun_dir, sun_enabled and
    // show_materials
    bool path_trace = false;

    // Direction towards the sun in world space (unit length). Used only
    // by the ray tracer, which lights the scene with this sun instead of
    // the camera-attached key light, so shadows stay put while orbiting,
    // and by the path tracer. With sun_enabled off the ray tracer uses
    // the camera light, without shadows, and the path tracer only the sky
    Vec3 sun_dir = {0.0f, 0.0f, 1.0f};
    bool sun_enabled = true;

    // Ray tracer only: darken creases and contact areas with ambient
    // occlusion, refined over frames while nothing moves
    bool ambient_occlusion = true;

    // Temporary demo animation state, replaced by a draw list in M3
    float demo_spin_angle = 0.0f;

    // Object instances to render
    std::vector<DrawItem> draw_list;

    // 2D overlay drawn on top of the 3D view, in window pixels.
    // Owned by the app; nullptr or an empty list draws nothing.
    // Every backend draws it (CUDA in software, see ui/ui_raster.h)
    const UiDrawList* ui = nullptr;
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

// Numbers a backend can show in the UI beyond the frame timings.
// Fields a backend has nothing for keep their defaults
struct RendererStats {
    // The ray tracer drew the last frame (CUDA only)
    bool ray_traced = false;

    // Rays traced per frame, averaged over about a second
    bool has_ray_counts = false;
    double camera_rays = 0.0;
    double shadow_rays = 0.0;
    double ao_rays = 0.0;

    // Ambient occlusion samples per pixel so far, and where it stops
    unsigned int ao_samples = 0;
    unsigned int ao_max_samples = 0;

    // The path tracer drew the last frame (CUDA only), the samples per
    // pixel accumulated so far and where it stops
    bool path_traced = false;
    unsigned int pt_samples = 0;
    unsigned int pt_max_samples = 0;
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
    // Extra statistics for the UI. The default has none
    virtual RendererStats stats() const {
        return {};
    }
    virtual void set_vsync(bool enabled) = 0;

    // Uploads an RGBA8 image (rows top to bottom) as a texture with
    // mipmaps. Returns 0 if the backend does not support textures (yet) or
    // the upload fails
    virtual TextureId create_texture(const Image& image) {
        (void)image;
        return 0;
    }

    // textures[i] is the TextureId of Model::images[i] (0 = none); a
    // material's base_color_image indexes this list
    virtual MeshId create_mesh(const Mesh& mesh,
                               const std::vector<TextureId>& textures) = 0;
};

// Returns nullptr and logs an error when the backend is unavailable
std::unique_ptr<IRenderer> create_renderer(RendererBackend backend);

} // namespace gfx
