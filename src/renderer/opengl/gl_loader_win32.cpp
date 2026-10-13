#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <windows.h>
#include <cstdint>

#define GFX_GL_DEFINE(ret, name, args) \
    PFN_gl##name gl##name = nullptr;

GFX_GL_FUNCTIONS(GFX_GL_DEFINE)

#undef GFX_GL_DEFINE

namespace {

void* get_proc(const char* name) {
    PROC proc = wglGetProcAddress(name);

    // Windows drivers can return these sentinel values on failure
    const auto value = reinterpret_cast<std::intptr_t>(proc);

    if (proc == nullptr ||
        value == 1 ||
        value == 2 ||
        value == 3 ||
        value == -1) {

        static HMODULE opengl32 =
            GetModuleHandleW(L"opengl32.dll");

        if (!opengl32) {
            return nullptr;
        }

        proc = GetProcAddress(opengl32, name);
    }

    return reinterpret_cast<void*>(proc);
}

} // namespace

namespace gfx {

bool gl_load_functions() {
    int loaded = 0;
    int missing = 0;

#define GFX_GL_LOAD(ret, name, args) \
    do { \
        gl##name = reinterpret_cast<PFN_gl##name>( \
            get_proc("gl" #name)); \
        if (gl##name) { \
            ++loaded; \
        } else { \
            LOG_ERROR("missing OpenGL function: gl" #name); \
            ++missing; \
        } \
    } while (false);

    GFX_GL_FUNCTIONS(GFX_GL_LOAD)

#undef GFX_GL_LOAD

    if (missing != 0) {
        LOG_ERROR(
            "OpenGL loader failed: %d loaded, %d missing",
            loaded, missing);
        return false;
    }

    LOG_INFO("loaded %d OpenGL functions", loaded);
    return true;
}

} // namespace gfx
