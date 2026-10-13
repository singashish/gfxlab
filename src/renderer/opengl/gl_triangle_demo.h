#pragma once

#include "core/mat4.h"
#include "core/math_util.h"

namespace gfx {

class TriangleDemo {
public:
    TriangleDemo() = default;
    ~TriangleDemo();

    TriangleDemo(const TriangleDemo&) = delete;
    TriangleDemo& operator=(const TriangleDemo&) = delete;

    bool init();
    void draw(const Mat4& view_proj, float spin_angle) const;
    void shutdown();

private:
    unsigned int m_program = 0;
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    int m_mvp_location = -1;
};

} // namespace gfx
