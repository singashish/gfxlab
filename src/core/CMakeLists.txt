add_library(gfx_core STATIC
    base64.cpp
    bvh.cpp
    bvh4.cpp
    gltf_loader.cpp
    image.cpp
    inflate.cpp
    json.cpp
    log.cpp
    mesh.cpp
    model.cpp
    obj_loader.cpp
    transform_edit.cpp
)

target_include_directories(gfx_core PUBLIC
    ${PROJECT_SOURCE_DIR}/src
)
