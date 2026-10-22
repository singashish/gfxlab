# Immediate-mode UI: font atlas packing and the UI draw list.
# Portable (no platform or graphics API code), so the tests build
# everywhere. Glyph rasterization lives in the platform layer and the
# drawing in each renderer backend
add_library(gfx_ui STATIC
    font_atlas.cpp
    ui_draw.cpp
)

target_include_directories(gfx_ui PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)

target_link_libraries(gfx_ui PUBLIC
    gfx_core
)
