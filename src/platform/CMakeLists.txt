if(NOT WIN32)
    message(FATAL_ERROR "gfx_platform currently supports Windows only")
endif()

add_library(gfx_platform STATIC
    win32/window_win32.cpp
)

target_include_directories(gfx_platform PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(gfx_platform PUBLIC gfx_core)
target_compile_definitions(gfx_platform PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
    UNICODE
    _UNICODE
)
