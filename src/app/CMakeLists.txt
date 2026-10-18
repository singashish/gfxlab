add_executable(gfxlab main.cpp)

target_link_libraries(gfxlab PRIVATE
    gfx_core
    gfx_platform
    gfx_renderer
)

target_compile_definitions(gfxlab PRIVATE
    GFX_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
)
