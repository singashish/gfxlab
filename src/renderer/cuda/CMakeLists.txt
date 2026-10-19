add_library(gfx_cuda STATIC
    cuda_info.cpp
    cuda_smoke.cu
)

target_include_directories(gfx_cuda PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gfx_cuda PUBLIC
    gfx_core
    CUDA::cudart_static
)

# The CUDA diagnostic executable remains available even when
# OpenGL is unavailable
if(TARGET gfx_gl)
    target_sources(gfx_cuda PRIVATE
        cuda_gl_image.cpp
        cuda_renderer.cpp
        cuda_mesh.cpp
        cuda_raster.cu
        cuda_timer.cpp
    )

    target_link_libraries(gfx_cuda PUBLIC
        gfx_gl
    )
endif()

add_executable(cuda_info
    cuda_info_main.cpp
)

target_link_libraries(cuda_info PRIVATE
    gfx_cuda
)
