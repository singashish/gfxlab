#pragma once

#include "core/image.h"
#include "renderer/renderer.h"

#include <vector>

namespace gfx {

// The OpenGL texture objects of a model's images, plus a 1x1 white
// texture for surfaces without one. Owned by the OpenGL backend; the
// textures are freed in shutdown (call it while the context is alive)
class GLTextureStore {
public:
    GLTextureStore() = default;
    ~GLTextureStore();

    GLTextureStore(const GLTextureStore&) = delete;
    GLTextureStore& operator=(const GLTextureStore&) = delete;

    // Needs a current OpenGL context
    bool init();

    // Uploads an RGBA8 image (top row first) with mipmaps.
    // Returns 0 on failure; otherwise a 1-based TextureId
    TextureId create(const Image& image);

    // OpenGL texture name of an id from create, or 0 if the id is unknown
    unsigned int gl_name(TextureId id) const;

    // 1x1 white texture: base colour * white = base colour, so a surface
    // without a texture needs no special case in the shader
    unsigned int white() const {
        return m_white;
    }

    void shutdown();

private:
    unsigned int m_white = 0;
    float m_max_anisotropy = 1.0f;

    // m_textures[id - 1] is the OpenGL name of TextureId id
    std::vector<unsigned int> m_textures;
};

} // namespace gfx
