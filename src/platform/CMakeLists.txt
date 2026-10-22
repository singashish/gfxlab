if(NOT WIN32)
    message(FATAL_ERROR "gfx_platform currently supports Windows only")
endif()

add_library(gfx_platform STATIC
    win32/ui_font_win32.cpp
    win32/window_win32.cpp
)

target_include_directories(gfx_platform PUBLIC ${PROJECT_SOURCE_DIR}/src)

# gdi32: the UI font is rasterized with GDI
target_link_libraries(gfx_platform
    PUBLIC gfx_core gfx_ui
    PRIVATE gdi32
)
