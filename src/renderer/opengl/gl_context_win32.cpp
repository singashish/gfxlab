#include "renderer/opengl/gl_context.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <windows.h>
#include <cstdio>

namespace {

constexpr wchar_t DUMMY_CLASS_NAME[] = L"GfxLabGLDummy";

// WGL_ARB_pixel_format
constexpr int WGL_DRAW_TO_WINDOW_ARB    = 0x2001;
constexpr int WGL_ACCELERATION_ARB      = 0x2003;
constexpr int WGL_SUPPORT_OPENGL_ARB    = 0x2010;
constexpr int WGL_DOUBLE_BUFFER_ARB     = 0x2011;
constexpr int WGL_PIXEL_TYPE_ARB        = 0x2013;
constexpr int WGL_COLOR_BITS_ARB        = 0x2014;
constexpr int WGL_DEPTH_BITS_ARB        = 0x2022;
constexpr int WGL_STENCIL_BITS_ARB      = 0x2023;
constexpr int WGL_FULL_ACCELERATION_ARB = 0x2027;
constexpr int WGL_TYPE_RGBA_ARB         = 0x202B;

// WGL_ARB_create_context
constexpr int WGL_CONTEXT_MAJOR_VERSION_ARB = 0x2091;
constexpr int WGL_CONTEXT_MINOR_VERSION_ARB = 0x2092;
constexpr int WGL_CONTEXT_FLAGS_ARB         = 0x2094;
constexpr int WGL_CONTEXT_PROFILE_MASK_ARB  = 0x9126;
constexpr int WGL_CONTEXT_DEBUG_BIT_ARB     = 0x0001;
constexpr int WGL_CONTEXT_CORE_PROFILE_BIT_ARB = 0x0001;

using PFN_wglChoosePixelFormatARB =
    BOOL (WINAPI*)(HDC, const int*, const FLOAT*, UINT, int*, UINT*);

using PFN_wglCreateContextAttribsARB =
    HGLRC (WINAPI*)(HDC, HGLRC, const int*);

using PFN_wglSwapIntervalEXT =
    BOOL (WINAPI*)(int);

PFN_wglChoosePixelFormatARB wglChoosePixelFormatARB = nullptr;
PFN_wglCreateContextAttribsARB wglCreateContextAttribsARB = nullptr;
PFN_wglSwapIntervalEXT wglSwapIntervalEXT = nullptr;

bool register_dummy_class() {
    static bool registered = false;

    if (registered) {
        return true;
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = DefWindowProcW;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.lpszClassName = DUMMY_CLASS_NAME;

    if (!RegisterClassExW(&wc)) {
        const DWORD error = GetLastError();

        // Another call may already have registered this class
        if (error != ERROR_CLASS_ALREADY_EXISTS) {
            LOG_ERROR("RegisterClassExW for GL dummy failed: %lu",
                static_cast<unsigned long>(error));
            return false;
        }
    }

    registered = true;
    return true;
}

bool load_wgl_extensions() {
    if (!register_dummy_class()) {
        return false;
    }

    const HWND hwnd = CreateWindowExW(
        0,
        DUMMY_CLASS_NAME,
        L"",
        WS_POPUP,
        0, 0, 1, 1,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        nullptr
    );

    if (!hwnd) {
        LOG_ERROR("CreateWindowExW for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        return false;
    }

    HDC hdc = GetDC(hwnd);
    HGLRC hglrc = nullptr;
    bool current = false;

    if (!hdc) {
        LOG_ERROR("GetDC for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        DestroyWindow(hwnd);
        return false;
    }

    PIXELFORMATDESCRIPTOR pfd{};
    pfd.nSize = sizeof(pfd);
    pfd.nVersion = 1;
    pfd.dwFlags =
        PFD_DRAW_TO_WINDOW |
        PFD_SUPPORT_OPENGL |
        PFD_DOUBLEBUFFER;
    pfd.iPixelType = PFD_TYPE_RGBA;
    pfd.cColorBits = 32;
    pfd.cDepthBits = 24;
    pfd.cStencilBits = 8;
    pfd.iLayerType = PFD_MAIN_PLANE;

    const int format = ChoosePixelFormat(hdc, &pfd);

    if (format == 0) {
        LOG_ERROR("ChoosePixelFormat for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        goto cleanup;
    }

    if (!SetPixelFormat(hdc, format, &pfd)) {
        LOG_ERROR("SetPixelFormat for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        goto cleanup;
    }

    hglrc = wglCreateContext(hdc);
    if (!hglrc) {
        LOG_ERROR("wglCreateContext for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        goto cleanup;
    }

    if (!wglMakeCurrent(hdc, hglrc)) {
        LOG_ERROR("wglMakeCurrent for GL dummy failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        goto cleanup;
    }
    current = true;

    wglChoosePixelFormatARB =
        reinterpret_cast<PFN_wglChoosePixelFormatARB>(
            wglGetProcAddress("wglChoosePixelFormatARB"));

    wglCreateContextAttribsARB =
        reinterpret_cast<PFN_wglCreateContextAttribsARB>(
            wglGetProcAddress("wglCreateContextAttribsARB"));

    wglSwapIntervalEXT =
        reinterpret_cast<PFN_wglSwapIntervalEXT>(
            wglGetProcAddress("wglSwapIntervalEXT"));

cleanup:
    // Release the current context before deleting it
    if (current) {
        wglMakeCurrent(nullptr, nullptr);
    }

    if (hglrc) {
        wglDeleteContext(hglrc);
    }

    ReleaseDC(hwnd, hdc);
    DestroyWindow(hwnd);

    if (!wglChoosePixelFormatARB ||
        !wglCreateContextAttribsARB) {
        LOG_ERROR("Required WGL extensions are unavailable");
        return false;
    }

    if (!wglSwapIntervalEXT) {
        LOG_WARN("WGL_EXT_swap_control unavailable; vsync cannot be set");
    }

    return true;
}

void GFX_GL_CALL gl_debug_callback(GLenum, GLenum, GLuint, GLenum severity,
                            GLsizei, const GLchar* message, const void*) {
    switch (severity) {
    case GL_DEBUG_SEVERITY_HIGH:
        LOG_ERROR("OpenGL: %s", message);
        break;

    case GL_DEBUG_SEVERITY_MEDIUM:
        LOG_WARN("OpenGL: %s", message);
        break;

    case GL_DEBUG_SEVERITY_LOW:
        LOG_INFO("OpenGL: %s", message);
        break;

    case GL_DEBUG_SEVERITY_NOTIFICATION:
        LOG_TRACE("OpenGL: %s", message);
        break;

    default:
        LOG_INFO("OpenGL: %s", message);
        break;
    }
}

} // namespace

namespace gfx {

GLContext::~GLContext() {
    destroy();
}

bool GLContext::create(void* native_window) {
    if (!native_window) {
        LOG_ERROR("GLContext::create received a null window");
        return false;
    }

    if (m_hdc || m_hglrc) {
        LOG_ERROR("GLContext has already been created");
        return false;
    }

    if (!load_wgl_extensions()) {
        return false;
    }

    const HWND hwnd = static_cast<HWND>(native_window);
    HDC hdc = GetDC(hwnd);

    if (!hdc) {
        LOG_ERROR("GetDC for real window failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        return false;
    }

    const int pixel_attribs[] = {
        WGL_DRAW_TO_WINDOW_ARB, 1,
        WGL_SUPPORT_OPENGL_ARB, 1,
        WGL_DOUBLE_BUFFER_ARB, 1,
        WGL_ACCELERATION_ARB, WGL_FULL_ACCELERATION_ARB,
        WGL_PIXEL_TYPE_ARB, WGL_TYPE_RGBA_ARB,
        WGL_COLOR_BITS_ARB, 32,
        WGL_DEPTH_BITS_ARB, 24,
        WGL_STENCIL_BITS_ARB, 8,
        0
    };

    int format = 0;
    UINT format_count = 0;

    if (!wglChoosePixelFormatARB(
            hdc, pixel_attribs, nullptr,
            1, &format, &format_count) ||
        format_count < 1) {
        LOG_ERROR("wglChoosePixelFormatARB failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        ReleaseDC(hwnd, hdc);
        return false;
    }

    PIXELFORMATDESCRIPTOR pfd{};

    if (!DescribePixelFormat(
            hdc, format, sizeof(pfd), &pfd)) {
        LOG_ERROR("DescribePixelFormat failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        ReleaseDC(hwnd, hdc);
        return false;
    }

    if (!SetPixelFormat(hdc, format, &pfd)) {
        LOG_ERROR("SetPixelFormat for real window failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        ReleaseDC(hwnd, hdc);
        return false;
    }

    int context_attribs[] = {
        WGL_CONTEXT_MAJOR_VERSION_ARB, 4,
        WGL_CONTEXT_MINOR_VERSION_ARB, 6,
        WGL_CONTEXT_PROFILE_MASK_ARB,
            WGL_CONTEXT_CORE_PROFILE_BIT_ARB,
#ifndef NDEBUG
        WGL_CONTEXT_FLAGS_ARB, WGL_CONTEXT_DEBUG_BIT_ARB,
#endif
        0
    };

    HGLRC hglrc = wglCreateContextAttribsARB(
        hdc, nullptr, context_attribs);

    if (!hglrc) {
        LOG_ERROR("wglCreateContextAttribsARB failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        ReleaseDC(hwnd, hdc);
        return false;
    }

    if (!wglMakeCurrent(hdc, hglrc)) {
        LOG_ERROR("wglMakeCurrent for real window failed: %lu",
            static_cast<unsigned long>(GetLastError()));
        wglDeleteContext(hglrc);
        ReleaseDC(hwnd, hdc);
        return false;
    }

    m_hdc = hdc;
    m_hglrc = hglrc;

    if (!gl_load_functions()) {
        destroy();
        return false;
    }

    GLint major = 0;
    GLint minor = 0;

    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    LOG_INFO("OpenGL context %d.%d core", major, minor);

    const auto* vendor = glGetString(GL_VENDOR);
    const auto* renderer = glGetString(GL_RENDERER);
    const auto* version = glGetString(GL_VERSION);
    const auto* glsl = glGetString(GL_SHADING_LANGUAGE_VERSION);

    LOG_INFO("OpenGL vendor: %s",
        vendor ? reinterpret_cast<const char*>(vendor) : "unknown");

    LOG_INFO("OpenGL renderer: %s",
        renderer ? reinterpret_cast<const char*>(renderer) : "unknown");

    LOG_INFO("OpenGL version: %s",
        version ? reinterpret_cast<const char*>(version) : "unknown");

    LOG_INFO("GLSL version: %s",
        glsl ? reinterpret_cast<const char*>(glsl) : "unknown");

    GLint flags = 0;
    glGetIntegerv(GL_CONTEXT_FLAGS, &flags);

    if ((flags & GL_CONTEXT_FLAG_DEBUG_BIT) != 0) {
        glEnable(GL_DEBUG_OUTPUT);
        glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);

        glDebugMessageCallback(gl_debug_callback, nullptr);

        LOG_INFO("OpenGL debug output enabled");
    } else {
#ifndef NDEBUG
       LOG_WARN("OpenGL debug context unavailable");
#else
       LOG_INFO("OpenGL debug output disabled (release build)");
#endif
    }

    return true;
}

void GLContext::destroy() {
    const HGLRC hglrc = static_cast<HGLRC>(m_hglrc);
    const HDC hdc = static_cast<HDC>(m_hdc);

    if (hglrc && wglGetCurrentContext() == hglrc) {
        wglMakeCurrent(nullptr, nullptr);
    }

    if (hglrc) {
        wglDeleteContext(hglrc);
    }

    if (hdc) {
        ReleaseDC(WindowFromDC(hdc), hdc);
    }

    m_hglrc = nullptr;
    m_hdc = nullptr;
}

void GLContext::swap_buffers() {
    if (m_hdc) {
        SwapBuffers(static_cast<HDC>(m_hdc));
    }
}

void GLContext::set_vsync(bool enabled) {
    if (!m_hglrc || !wglSwapIntervalEXT) {
        return;
    }

    if (wglGetCurrentContext() != static_cast<HGLRC>(m_hglrc)) {
        LOG_WARN("Cannot set vsync: GL context is not current");
        return;
    }

    if (!wglSwapIntervalEXT(enabled ? 1 : 0)) {
        LOG_WARN("wglSwapIntervalEXT failed: %lu",
            static_cast<unsigned long>(GetLastError()));
    }
}

void gl_clear_frame(int width, int height) {
    if (width <= 0 || height <= 0) {
        return;
    }

    glViewport(0, 0, width, height);
    glClearColor(0.10f, 0.11f, 0.13f, 1.0f);
    
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

} // namespace gfx
