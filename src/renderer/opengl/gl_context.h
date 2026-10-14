#pragma once

namespace gfx {

class GLContext {
public:
    GLContext() = default;
    ~GLContext();

    GLContext(const GLContext&) = delete;
    GLContext& operator=(const GLContext&) = delete;

    bool create(void* native_window);
    void destroy();

    void swap_buffers();
    void set_vsync(bool enabled);

private:
    void* m_hdc = nullptr;
    void* m_hglrc = nullptr;
};

} // namespace gfx
