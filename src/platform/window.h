#pragma once

#include "platform/input.h"

#include <functional>

namespace gfx {

struct WindowDesc {
    const char* title;   // UTF-8
    int width;           // client area in pixels
    int height;
};

struct WindowImpl;       // defined in the platform\win32\window_win32.cpp

class Window {
public:
    Window();
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool create(const WindowDesc& desc);
    void poll_events();
    bool should_close() const;
    void request_close();

    bool is_minimized() const;   // true while the window is minimized
    bool was_resized() const;    // true for one frame after the client size changed
    void wait_events();          // block until a message arrives, then poll_events()

    int width() const;
    int height() const;

    // HWND on Windows
    // needed later to create the GL context / Vulkan surface
    void* native_handle() const;

    void set_title(const char* utf8);

    const InputState& input() const;

    // Called while Windows runs its modal loop during dragging/resizing
    void set_modal_frame_callback(std::function<void()> callback);

private:
    WindowImpl* m_impl;
};

} // namespace gfx
