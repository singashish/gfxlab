#pragma once

#include <memory>

namespace gfx {

// Shows the CUDA renderer's finished frame in the window.
//
// The CUDA renderer draws into an ordinary device buffer and knows
// nothing about the graphics API that puts it on screen; that is the
// presenter's job. Today the only presenter uses OpenGL interop
// (CudaGLPresenter). M10 adds a DX12 one behind the same interface
class ICudaPresenter {
public:
    virtual ~ICudaPresenter() = default;

    // Name for the log, e.g. "OpenGL interop"
    virtual const char* name() const = 0;

    // Sets up the display API for the window and makes the CUDA device
    // that can share memory with it the current CUDA device. Must run
    // before any other CUDA call of the renderer
    virtual bool init(void* native_window) = 0;

    // (Re)creates the display image for a new framebuffer size
    virtual bool resize(int width, int height) = 0;

    // Copies the frame into the display image. pixels is device memory:
    // width * height RGBA8 pixels, rows tightly packed, top row first.
    // The copy is queued on CUDA's default stream, after the kernels
    // that drew the frame
    virtual bool copy(const void* pixels, int width, int height) = 0;

    // Puts the display image on screen (and waits for vsync if on)
    virtual void present(int width, int height) = 0;

    virtual void set_vsync(bool enabled) = 0;
};

// The presenter for this build. nullptr if none is available
std::unique_ptr<ICudaPresenter> create_cuda_presenter();

} // namespace gfx
