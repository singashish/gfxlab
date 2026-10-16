add_library(gfx_vk STATIC
    vk_context.cpp
    vk_renderer.cpp
)

target_include_directories(gfx_vk PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_definitions(gfx_vk PUBLIC
    VK_USE_PLATFORM_WIN32_KHR
)

target_link_libraries(gfx_vk PUBLIC
    gfx_core
    Vulkan::Vulkan
)
