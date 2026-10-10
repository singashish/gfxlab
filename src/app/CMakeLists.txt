add_executable(gfxlab main.cpp)

target_link_libraries(gfxlab PRIVATE
    gfx_core
    gfx_config
)
