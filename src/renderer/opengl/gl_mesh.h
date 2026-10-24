#pragma once

#include "core/mesh.h"
#include "renderer/renderer.h"
#include "renderer/opengl/gl_texture.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace gfx {

// One draw call of a mesh: a run of triangles with the same material
struct GLDrawRange {
    std::size_t first_index_byte = 0; // offset into the index buffer
    int index_count = 0;
    Vec3 base_color = {0.8f, 0.8f, 0.8f};
    unsigned int texture = 0;         // OpenGL texture name (never 0 once uploaded)
};

class GLMesh {
public:
    GLMesh() = default;
    ~GLMesh();

    GLMesh(const GLMesh&) = delete;
    GLMesh& operator=(const GLMesh&) = delete;

    // textures[i] belongs to Model::images[i], see IRenderer::create_mesh.
    // Materials without a usable texture get the store's white texture
    bool upload(const Mesh& mesh, const std::vector<TextureId>& textures,
                const GLTextureStore& store);

    // Draws one range at a time. base_color_location is the shader's
    // u_base_color uniform; the range's texture is bound to unit 0
    void draw(int base_color_location) const;
    void draw_edges() const;
    void shutdown();

private:
    unsigned int m_vao = 0;
    unsigned int m_vbo = 0;
    unsigned int m_ebo = 0;

    // Separate VAO and EBO for wireframe edges
    unsigned int m_edge_vao = 0;
    unsigned int m_edge_ebo = 0;

    int m_edge_index_count = 0;

    // One entry per material that has triangles
    std::vector<GLDrawRange> m_ranges;
};

class GLMeshRenderer {
public:
    GLMeshRenderer() = default;
    ~GLMeshRenderer();

    GLMeshRenderer(const GLMeshRenderer&) = delete;
    GLMeshRenderer& operator=(const GLMeshRenderer&) = delete;

    bool init();
    MeshId add(const Mesh& mesh, const std::vector<TextureId>& textures,
               const GLTextureStore& store);

    // show_materials false = plain grey shading
    void draw(const std::vector<DrawItem>& items,
              const Mat4& view_proj, const Mat4& view,
              bool show_materials) const;
    void draw_wireframe(const std::vector<DrawItem>& items,
                        const Mat4& view_proj) const;
    void shutdown();

private:
    unsigned int m_program = 0;
    int m_view_location = -1;
    int m_view_proj_location = -1;
    int m_model_location = -1;
    int m_normal_model_location = -1;
    int m_base_color_location = -1;
    int m_base_texture_location = -1;
    int m_show_materials_location = -1;

    unsigned int m_wire_program = 0;
    int m_wire_view_proj_location = -1;
    int m_wire_model_location = -1;
    int m_wire_color_location = -1;

    std::vector<std::unique_ptr<GLMesh>> m_meshes;
};

} // namespace gfx
