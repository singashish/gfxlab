#pragma once

#include "core/mat4.h"
#include "core/vec3.h"

namespace gfx {

struct GridParams {
    Mat4 view_proj;
    Vec3 eye;
    Vec3 forward;
    Vec3 right;
    Vec3 up;
    float fade_end;
};

class GridRenderer {
public:
    GridRenderer() = default;
    ~GridRenderer();

    GridRenderer(const GridRenderer&) = delete;
    GridRenderer& operator=(const GridRenderer&) = delete;

    bool init();
    void draw(const GridParams& params) const;
    void shutdown();

private:
    unsigned int m_program = 0;
    unsigned int m_vao = 0;

    int m_view_proj_location = -1;
    int m_eye_location = -1;
    int m_forward_location = -1;
    int m_right_location = -1;
    int m_up_location = -1;
    int m_fade_end = -1;
};

} // namespace gfx
