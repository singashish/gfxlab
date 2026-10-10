#pragma once

namespace gfx {

struct WindowDesc {
    const char* title;   // UTF-8
    int width;           // client area in pixels
    int height;
};

struct WindowImpl;       // defined in the platform .cpp

class Window {
public:
    Window();
    ~Window();
    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool create(const WindowDesc& desc);
    void poll_events();
    bool should_close() const;

    int width() const;
    int height() const;

    // HWND on Windows
    // needed later to create the GL context / Vulkan surface
    void* native_handle() const;

private:
    WindowImpl* m_impl;
};

} // namespace gfx
