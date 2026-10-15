add_library(gfx_core STATIC
    log.cpp
    mesh.cpp
    obj_loader.cpp
)

target_include_directories(gfx_core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
