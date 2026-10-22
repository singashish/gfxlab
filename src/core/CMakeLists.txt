add_library(gfx_core STATIC
    bvh.cpp
    bvh4.cpp
    log.cpp
    mesh.cpp
    obj_loader.cpp
    transform_edit.cpp
)

target_include_directories(gfx_core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
