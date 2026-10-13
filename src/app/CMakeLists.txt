add_executable(gfxlab main.cpp)

target_link_libraries(gfxlab PRIVATE
    gfx_core
    gfx_platform
    gfx_config
)

if(TARGET gfx_gl)
    target_link_libraries(gfxlab PRIVATE gfx_gl)
endif()
