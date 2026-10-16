add_library(gfx_renderer STATIC
    renderer_factory.cpp
)

target_include_directories(gfx_renderer PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gfx_renderer PUBLIC
    gfx_core
    gfx_config
)

if(TARGET gfx_gl)
    target_link_libraries(gfx_renderer PUBLIC gfx_gl)
endif()

if(TARGET gfx_vk)
    target_link_libraries(gfx_renderer PUBLIC gfx_vk)
endif()
