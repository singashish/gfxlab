#pragma once

#include "core/mesh.h"
#include "renderer/renderer.h"

#include <memory>
#include <vector>
#include <cstdint>

namespace gfx {

class GLMesh {
public:
    GLMesh() = default;
    ~GLMesh();

    GLMesh(const GLMesh&) = delete;
    GLMesh& operator=(const GLMesh&) = delete;

    bool upload(const Mesh& mesh);
    void draw() const;
    void shutdown();

private:
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    unsigned int m_ebo = 0;
    int m_index_count = 0;
};

class GLMeshRenderer {
public:
    GLMeshRenderer() = default;
    ~GLMeshRenderer();

    GLMeshRenderer(const GLMeshRenderer&) = delete;
    GLMeshRenderer& operator=(const GLMeshRenderer&) = delete;

    bool init();
    MeshId add(const Mesh& mesh);
    void draw(MeshId id, const Mat4& view_proj, const Mat4& view) const;
    void shutdown();

private:
    unsigned int m_program = 0;
    int m_view_location = -1;
    int m_view_proj_location = -1;

    std::vector<std::unique_ptr<GLMesh>> m_meshes;
};

} // namespace gfx
