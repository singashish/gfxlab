add_library(gfx_gl STATIC
    gl_context_win32.cpp
    gl_grid.cpp
    gl_loader_win32.cpp
    gl_mesh.cpp
    gl_presenter.cpp
    gl_renderer.cpp
    gl_timer.cpp
    gl_shader.cpp
    gl_triangle_demo.cpp
    gl_ui.cpp
)

target_include_directories(gfx_gl PUBLIC 
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gfx_gl 
    PUBLIC gfx_core gfx_ui
    PRIVATE OpenGL::GL
)
