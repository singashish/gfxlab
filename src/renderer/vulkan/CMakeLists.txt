gfx_compile_shaders(
    gfx_shaders
    "${CMAKE_BINARY_DIR}/shaders"
    shaders/triangle.vert
    shaders/triangle.frag
    shaders/mesh.vert
    shaders/mesh.frag
    shaders/grid.vert
    shaders/grid.frag
    shaders/wire.vert
    shaders/wire.frag
)

add_library(gfx_vk STATIC
    vk_buffer.cpp
    vk_context.cpp
    vk_depth.cpp
    vk_frames.cpp
    vk_grid.cpp
    vk_memory.cpp
    vk_mesh.cpp
    vk_pipeline.cpp
    vk_renderer.cpp
    vk_shader.cpp
    vk_swapchain.cpp
    vk_triangle_demo.cpp
    vk_uniforms.cpp
    vk_upload.cpp
)

add_dependencies(gfx_vk gfx_shaders)

target_include_directories(gfx_vk PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(gfx_vk PUBLIC
    VK_USE_PLATFORM_WIN32_KHR
)

target_compile_definitions(gfx_vk PRIVATE
    GFX_SHADER_DIR="${CMAKE_BINARY_DIR}/shaders"
)

target_link_libraries(gfx_vk PUBLIC
    gfx_core
    Vulkan::Vulkan
)
