add_library(gfx_cuda STATIC
    cuda_info.cpp
    cuda_smoke.cu
)

target_include_directories(gfx_cuda PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gfx_cuda PUBLIC
    gfx_core
    CUDA::cudart
)

add_executable(cuda_info
    cuda_info_main.cpp
)

target_link_libraries(cuda_info PRIVATE
    gfx_cuda
)
