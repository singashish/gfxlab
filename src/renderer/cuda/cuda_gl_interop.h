#pragma once

#include <cstddef>

struct cudaGraphicsResource;

namespace gfx {

bool cuda_select_gl_device();

class CudaGLImage {
public:
    CudaGLImage() = default;
    ~CudaGLImage();

    CudaGLImage(const CudaGLImage&) = delete;
    CudaGLImage& operator=(const CudaGLImage&) = delete;

    bool register_texture(unsigned int gl_texture);

    bool copy_from_device(const void* d_pixels, int width,
                          int height, std::size_t pitch_bytes);

    void unregister();

private:
    cudaGraphicsResource* m_resource = nullptr;
};

} // namespace gfx
