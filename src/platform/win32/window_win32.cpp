#include "platform/window.h"
#include "core/log.h"

// (set via CMake below: WIN32_LEAN_AND_MEAN, NOMINMAX, UNICODE)
#include <windows.h>
#include <windowsx.h>

#include <string>
#include <vector>

namespace gfx {

struct WindowImpl {
    HWND hwnd = nullptr;
    bool close_requested = false;
    int width = 0;
    int height = 0;
    InputState input;
};

} // namespace gfx

namespace {

constexpr wchar_t WINDOW_CLASS_NAME[] = L"GfxLabWindow";

// Forward declaration
static LRESULT CALLBACK wnd_proc(HWND hwnd, UINT msg, WPARAM wp, LPARAM lp);

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

bool utf8_to_utf16(const char* input, std::vector<wchar_t>& output) {
    if (input == nullptr) {
        return false;
    }

    const int length = MultiByteToWideChar(
        CP_UTF8,
        MB_ERR_INVALID_CHARS,
        input,
        -1,
        nullptr,
        0
    );

    if (length == 0) {
        LOG_ERROR(
            "MultiByteToWideChar length query failed: %lu",
            static_cast<unsigned long>(GetLastError())
        );
        return false;
    }

    output.resize(static_cast<std::size_t>(length));

    if (MultiByteToWideChar(
            CP_UTF8,
            MB_ERR_INVALID_CHARS,
            input,
            -1,
            output.data(),
            length) == 0) {
        LOG_ERROR(
            "MultiByteToWideChar conversion failed: %lu",
            static_cast<unsigned long>(GetLastError())
        );
        return false;
    }

    return true;
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
            impl->width = static_cast<int>(LOWORD(lp));
            impl->height = static_cast<int>(HIWORD(lp));
            return 0;

        case WM_NCDESTROY:
            SetWindowLongPtrW(hwnd, GWLP_USERDATA, 0);
            impl->hwnd = nullptr;
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
    std::vector<wchar_t> title;

    if (!utf8_to_utf16(desc.title, title)) {
        return false;
    }

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
        title.data(),
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

const InputState& Window::input() const {
    return m_impl->input;
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

void Window::request_close() {
    if (m_impl != nullptr) {
        m_impl->close_requested = true;
    }
}

} // namespace gfx
