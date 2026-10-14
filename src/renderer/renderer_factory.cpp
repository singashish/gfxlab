#include "renderer/renderer.h"
#include "core/log.h"

#if defined(GFX_HAS_OPENGL)
#include "renderer/opengl/gl_renderer.h"
#endif

namespace gfx {

std::unique_ptr<IRenderer> create_renderer(RendererBackend backend) {
    switch (backend) {
#if defined(GFX_HAS_OPENGL)
    case RendererBackend::GFX_BACKEND_OPENGL:
        return std::make_unique<GLRenderer>();
#endif

    default:
        break;
    }

    LOG_ERROR("requested renderer backend is not available in this build");
    return nullptr;
}

} // namespace gfx
