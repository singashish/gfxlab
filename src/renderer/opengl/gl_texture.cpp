#include "renderer/opengl/gl_texture.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace gfx {

namespace {

// Anisotropic filtering keeps textures sharp on surfaces seen at a grazing
// angle (a floor far away). 8x is plenty; the hardware limit may be lower
constexpr float kWantedAnisotropy = 8.0f;

// Discard stale OpenGL errors without an unbounded loop, so an error
// check after our own calls only sees our own errors
void clear_gl_errors() {
    for (int i = 0; i < 16 && glGetError() != GL_NO_ERROR; ++i) {
    }
}

// Creates a mipmapped RGBA8 texture from tightly packed pixels, top row
// first. Returns the OpenGL name, or 0 on failure.
//
// Colour note: the format is plain GL_RGBA8, not sRGB. The viewport
// shading works directly in display values (no linear / sRGB conversion),
// so the image bytes are used as they are. That keeps the grey look and
// the three backends consistent. The path tracer (Step 7/8) will do proper
// linear light with sRGB output
GLuint create_rgba8_texture(const std::uint8_t* pixels, int width,
                            int height, float max_anisotropy) {
    clear_gl_errors();

    GLuint texture = 0;
    glGenTextures(1, &texture);

    if (texture == 0) {
        return 0;
    }

    glBindTexture(GL_TEXTURE_2D, texture);

    // Rows go to the GPU as given (top row first); the mesh shader flips v
    // to match. RGBA rows are a multiple of 4 bytes, so the default
    // unpack alignment fits
    glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(GL_RGBA8),
                 width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels);

    // Smaller and smaller copies of the image: far-away surfaces sample
    // those instead of skipping over texels (which would shimmer)
    glGenerateMipmap(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    static_cast<GLint>(GL_LINEAR_MIPMAP_LINEAR));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    static_cast<GLint>(GL_LINEAR));

    // glTF's default: the image repeats outside 0..1
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    static_cast<GLint>(GL_REPEAT));
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    static_cast<GLint>(GL_REPEAT));

    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY,
                    max_anisotropy);

    glBindTexture(GL_TEXTURE_2D, 0);

    const GLenum error = glGetError();

    if (error != GL_NO_ERROR) {
        LOG_ERROR("OpenGL error 0x%04X while creating a %dx%d texture",
                  error, width, height);
        glDeleteTextures(1, &texture);
        return 0;
    }

    return texture;
}

} // namespace

GLTextureStore::~GLTextureStore() {
    shutdown();
}

bool GLTextureStore::init() {
    shutdown();

    // The hardware limit for anisotropic filtering, at most 8x
    clear_gl_errors();

    GLfloat hardware_max = 1.0f;
    glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY, &hardware_max);

    if (glGetError() != GL_NO_ERROR) {
        hardware_max = 1.0f;
    }

    m_max_anisotropy =
        std::max(1.0f, std::min(hardware_max, kWantedAnisotropy));

    const std::uint8_t white_pixel[4] = {255, 255, 255, 255};

    m_white = create_rgba8_texture(white_pixel, 1, 1, 1.0f);

    if (m_white == 0) {
        LOG_ERROR("failed to create the white fallback texture");
        return false;
    }

    return true;
}

TextureId GLTextureStore::create(const Image& image) {
    if (m_white == 0) {
        return 0;
    }

    if (!image.valid()) {
        LOG_ERROR("cannot upload an invalid image");
        return 0;
    }

    if (m_textures.size() >=
        static_cast<std::size_t>(std::numeric_limits<TextureId>::max())) {
        return 0;
    }

    const GLuint texture = create_rgba8_texture(
        image.pixels.data(), image.width, image.height, m_max_anisotropy);

    if (texture == 0) {
        return 0;
    }

    m_textures.push_back(texture);

    return static_cast<TextureId>(m_textures.size());
}

unsigned int GLTextureStore::gl_name(TextureId id) const {
    if (id == 0 || static_cast<std::size_t>(id) > m_textures.size()) {
        return 0;
    }

    return m_textures[static_cast<std::size_t>(id) - 1];
}

void GLTextureStore::shutdown() {
    for (const unsigned int texture : m_textures) {
        glDeleteTextures(1, &texture);
    }

    m_textures.clear();

    if (m_white != 0) {
        glDeleteTextures(1, &m_white);
        m_white = 0;
    }
}

} // namespace gfx
