#include "platform/window.h"
#include "core/log.h"

// WIN32_LEAN_AND_MEAN, NOMINMAX and UNICODE are defined
// project-wide in the root CMakeLists.txt
#include <windows.h>
#include <windowsx.h>

#include <string>

namespace gfx {

struct WindowImpl {
    HWND hwnd = nullptr;
    bool close_requested = false;
    int width = 0;
    int height = 0;
    bool minimized = false;
    bool resized = false;
    InputState input;
};

} // namespace gfx

namespace {

constexpr wchar_t WINDOW_CLASS_NAME[] = L"GfxLabWindow";

// Forward declaration
static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);
static gfx::Key translate_vk(WPARAM vk);
static bool any_mouse_down(const gfx::InputState& input);

bool make_process_dpi_aware() {
    static bool initialized = false;

    if (initialized) {
        return true;
    }

    if (!SetProcessDpiAwarenessContext(
            DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)) {
        const DWORD error = GetLastError();

        // The process may already have its DPI awareness set by a manifest
        // or another component.
        if (error != ERROR_ACCESS_DENIED) {
            LOG_ERROR(
                "SetProcessDpiAwarenessContext failed: %lu",
                static_cast<unsigned long>(error)
            );
            return false;
        }
    }

    initialized = true;
    return true;
}

bool register_window_class() {
    static bool registered = false;

    if (registered) {
        return true;
    }

    WNDCLASSEXW wc{};
    wc.cbSize = sizeof(wc);
    wc.style = CS_OWNDC;
    wc.lpfnWndProc = wnd_proc;
    wc.hInstance = GetModuleHandleW(nullptr);
    wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
    wc.hbrBackground =
        reinterpret_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    wc.lpszClassName = WINDOW_CLASS_NAME;

    if (!RegisterClassExW(&wc)) {
        const DWORD error = GetLastError();

        LOG_ERROR(
            "RegisterClassExW failed: %lu",
            static_cast<unsigned long>(error)
        );
        return false;
    }

    registered = true;
    return true;
}

static std::wstring utf8_to_wide(const char* utf8) {
    if (utf8 == nullptr || utf8[0] == '\0') {
        return L"";
    }

    const int length = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        utf8,
        -1,
        nullptr,
        0
    );

    if (length == 0) {
        return L"";
    }

    std::wstring result(
        static_cast<std::size_t>(length),
        L'\0'
    );

    if (MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            utf8,
            -1,
            result.data(),
            length
        ) == 0) {
        return L"";
    }

    result.resize(static_cast<std::size_t>(length - 1));

    return result;
}

static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp) {
    gfx::WindowImpl* impl = nullptr;

    if (msg == WM_NCCREATE) {
        auto* cs = reinterpret_cast<CREATESTRUCTW*>(lp);
        impl = static_cast<gfx::WindowImpl*>(cs->lpCreateParams);

        SetWindowLongPtrW(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(impl));
    } else {
        impl = reinterpret_cast<gfx::WindowImpl*>(
            GetWindowLongPtrW(hwnd, GWLP_USERDATA)
        );
    }

    if (impl != nullptr) {
        switch (msg) {
        case WM_CLOSE:
            // Defer actual destruction until Window's destructor
            impl->close_requested = true;
            return 0;

        case WM_SIZE:
            if (wp == SIZE_MINIMIZED) {
                impl->minimized = true;
            } else {
                impl->minimized = false;

                const int new_width = static_cast<int>(LOWORD(lp));
                const int new_height = static_cast<int>(HIWORD(lp));

                if (new_width != impl->width || new_height != impl->height) {
                    impl->width = new_width;
                    impl->height = new_height;
                    impl->resized = true;
                }
            }
            return 0;

        case WM_NCDESTROY:
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            impl->hwnd = nullptr;
            break;

        case WM_KEYDOWN:
        case WM_KEYUP: {
            const gfx::Key key = translate_vk(wp);
            if (key != gfx::Key::GFX_KEY_UNKNOWN) {
                impl->input.set_key(key, msg == WM_KEYDOWN);
            }
            return 0;
        }

        case WM_SYSKEYDOWN:
        case WM_SYSKEYUP: {
            const gfx::Key key = translate_vk(wp);
            if (key != gfx::Key::GFX_KEY_UNKNOWN) {
                impl->input.set_key(key, msg == WM_SYSKEYDOWN);
            }
            // Preserve default system-key processing, including Alt+F4.
            break;
        }

        case WM_SYSCOMMAND:
            if ((wp & 0xFFF0) == SC_KEYMENU) {
                return 0;
            }
            break;

        case WM_MOUSEMOVE:
            impl->input.set_mouse_position(
                static_cast<float>(GET_X_LPARAM(lp)),
                static_cast<float>(GET_Y_LPARAM(lp))
            );
            return 0;

        case WM_LBUTTONDOWN:
        case WM_MBUTTONDOWN:
        case WM_RBUTTONDOWN: {
            const bool was_down = any_mouse_down(impl->input);

            gfx::MouseButton button = gfx::MouseButton::GFX_MOUSE_LEFT;
            if (msg == WM_MBUTTONDOWN) {
                button = gfx::MouseButton::GFX_MOUSE_MIDDLE;
            } else if (msg == WM_RBUTTONDOWN) {
                button = gfx::MouseButton::GFX_MOUSE_RIGHT;
            }

            impl->input.set_mouse_button(button, true);

            if (!was_down) {
                SetCapture(hwnd);
            }
            return 0;
        }

        case WM_LBUTTONUP:
        case WM_MBUTTONUP:
        case WM_RBUTTONUP: {
            gfx::MouseButton button = gfx::MouseButton::GFX_MOUSE_LEFT;
            if (msg == WM_MBUTTONUP) {
                button = gfx::MouseButton::GFX_MOUSE_MIDDLE;
            } else if (msg == WM_RBUTTONUP) {
                button = gfx::MouseButton::GFX_MOUSE_RIGHT;
            }

            impl->input.set_mouse_button(button, false);

            if (!any_mouse_down(impl->input) && GetCapture() == hwnd) {
                ReleaseCapture();
            }
            return 0;
        }

        case WM_MOUSEWHEEL:
            impl->input.add_scroll(
                static_cast<float>(GET_WHEEL_DELTA_WPARAM(wp))
                / static_cast<float>(WHEEL_DELTA)
            );
            return 0;

        case WM_KILLFOCUS:
            if (GetCapture() == hwnd) {
                ReleaseCapture();
            }
            impl->input.release_all();
            break;

        default:
            break;
        }
    }

    return DefWindowProcW(hwnd, msg, wp, lp);
}

static gfx::Key translate_vk(WPARAM vk) {
    using gfx::Key;

    if (vk >= 'A' && vk <= 'Z') {
        return static_cast<Key>(
            static_cast<unsigned char>(Key::GFX_KEY_A)
            + static_cast<unsigned char>(vk - 'A')
        );
    }

    if (vk >= '0' && vk <= '9') {
        return static_cast<Key>(
            static_cast<unsigned char>(Key::GFX_KEY_0)
            + static_cast<unsigned char>(vk - '0')
        );
    }

    if (vk >= VK_NUMPAD0 && vk <= VK_NUMPAD9) {
        return static_cast<Key>(
            static_cast<unsigned char>(Key::GFX_KEY_NUMPAD_0)
            + static_cast<unsigned char>(vk - VK_NUMPAD0)
        );
    }

    switch (vk) {
    case VK_ESCAPE:  return Key::GFX_KEY_ESCAPE;
    case VK_SPACE:   return Key::GFX_KEY_SPACE;
    case VK_TAB:     return Key::GFX_KEY_TAB;
    case VK_DELETE:  return Key::GFX_KEY_DELETE;
    case VK_LEFT:    return Key::GFX_KEY_LEFT;
    case VK_RIGHT:   return Key::GFX_KEY_RIGHT;
    case VK_UP:      return Key::GFX_KEY_UP;
    case VK_DOWN:    return Key::GFX_KEY_DOWN;
    case VK_SHIFT:   return Key::GFX_KEY_SHIFT;
    case VK_CONTROL: return Key::GFX_KEY_CTRL;
    case VK_MENU:    return Key::GFX_KEY_ALT;
    default:         return Key::GFX_KEY_UNKNOWN;
    }
}

static bool any_mouse_down(const gfx::InputState& input) {
    using gfx::MouseButton;

    return input.mouse_down(MouseButton::GFX_MOUSE_LEFT)
        || input.mouse_down(MouseButton::GFX_MOUSE_MIDDLE)
        || input.mouse_down(MouseButton::GFX_MOUSE_RIGHT);
}

} // namespace

namespace gfx {

Window::Window()
    : m_impl(new WindowImpl{}) {
}

Window::~Window() {
    if (m_impl != nullptr) {
        if (m_impl->hwnd != nullptr) {
            DestroyWindow(m_impl->hwnd);
        }

        delete m_impl;
        m_impl = nullptr;
    }
}

bool Window::create(const WindowDesc& desc) {
    if (m_impl == nullptr || m_impl->hwnd != nullptr) {
        LOG_ERROR("Window is invalid or has already been created");
        return false;
    }

    if (desc.title == nullptr || desc.width <= 0 || desc.height <= 0) {
        LOG_ERROR("Invalid window description");
        return false;
    }

    // 1. Set process DPI awareness before creating the window
    if (!make_process_dpi_aware()) {
        return false;
    }

    // 2. Convert the UTF-8 title into UTF-16
    const std::wstring wide_title = utf8_to_wide(desc.title);

    // 3. Register the window class
    if (!register_window_class()) {
        return false;
    }

    // 4. Convert the requested client size to an outer window size
    constexpr DWORD style = WS_OVERLAPPEDWINDOW;
    constexpr DWORD ex_style = 0;

    RECT rect{
        0,
        0,
        static_cast<LONG>(desc.width),
        static_cast<LONG>(desc.height)
    };

    const UINT dpi = GetDpiForSystem();

    if (!AdjustWindowRectExForDpi(&rect, style, FALSE, ex_style, dpi)) {
        LOG_ERROR("AdjustWindowRectExForDpi failed: %lu",
            static_cast<unsigned long>(GetLastError())
        );
        return false;
    }

    const int outer_width = rect.right - rect.left;
    const int outer_height = rect.bottom - rect.top;

    // Initialize the requested client dimensions. WM_SIZE will update them
    // with the actual client dimensions when Windows creates the window.
    m_impl->width = desc.width;
    m_impl->height = desc.height;

    // 5. Create the native window. m_impl arrives through WM_NCCREATE
    HWND hwnd = CreateWindowExW(
        ex_style,
        WINDOW_CLASS_NAME,
        wide_title.c_str(),
        style,
        CW_USEDEFAULT,
        CW_USEDEFAULT,
        outer_width,
        outer_height,
        nullptr,
        nullptr,
        GetModuleHandleW(nullptr),
        m_impl
    );

    if (hwnd == nullptr) {
        LOG_ERROR("CreateWindowExW failed: %lu",
            static_cast<unsigned long>(GetLastError())
        );
        return false;
    }

    m_impl->hwnd = hwnd;

    // 6. Display the window
    ShowWindow(hwnd, SW_SHOW);

    LOG_INFO(
        "Window created: %dx%d client area",
        m_impl->width,
        m_impl->height
    );

    return true;
}

void Window::poll_events() {
    m_impl->resized = false;
    m_impl->input.begin_frame();

    MSG msg{};

    while (PeekMessageW(&msg, nullptr, 0, 0, PM_REMOVE)) {
        if (msg.message == WM_QUIT) {
            m_impl->close_requested = true;
            continue;
        }

        TranslateMessage(&msg);
        DispatchMessageW(&msg);
    }

    m_impl->input.end_frame();
}

bool Window::should_close() const {
    return m_impl == nullptr || m_impl->close_requested;
}

void Window::request_close() {
    if (m_impl != nullptr) {
        m_impl->close_requested = true;
    }
}

bool Window::is_minimized() const {
    return m_impl == nullptr || m_impl->minimized;
}

bool Window::was_resized() const {
    return m_impl != nullptr && m_impl->resized;
}

void Window::wait_events() {
    if (m_impl == nullptr || m_impl->close_requested) {
        return;
    }

    WaitMessage();
    poll_events();
}

int Window::width() const {
    return m_impl != nullptr ? m_impl->width : 0;
}

int Window::height() const {
    return m_impl != nullptr ? m_impl->height : 0;
}

void* Window::native_handle() const {
    return m_impl != nullptr
        ? static_cast<void*>(m_impl->hwnd)
        : nullptr;
}

void Window::set_title(const char* utf8) {
    if (m_impl == nullptr || m_impl->hwnd == nullptr) {
        return;
    }

    const std::wstring wide_title = utf8_to_wide(utf8);

    SetWindowTextW(m_impl->hwnd, wide_title.c_str());
}

const InputState& Window::input() const {
    static const InputState empty{};
    return m_impl != nullptr ? m_impl->input : empty;
}

} // namespace gfx
