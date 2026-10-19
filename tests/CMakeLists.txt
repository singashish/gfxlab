add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE gfx_core)
add_test(NAME math COMMAND test_math)

add_executable(test_input test_input.cpp)
target_link_libraries(test_input PRIVATE gfx_core)
add_test(NAME input COMMAND test_input)

add_executable(test_timer test_timer.cpp)
target_link_libraries(test_timer PRIVATE gfx_core)
add_test(NAME timer COMMAND test_timer)

add_executable(test_camera test_camera.cpp)
target_link_libraries(test_camera PRIVATE gfx_core)
add_test(NAME camera COMMAND test_camera)

add_executable(test_obj test_obj.cpp)
target_link_libraries(test_obj PRIVATE gfx_core)
target_compile_definitions(test_obj PRIVATE
    GFX_ASSETS_DIR="${PROJECT_SOURCE_DIR}/assets"
)
add_test(NAME obj COMMAND test_obj)

add_executable(test_transform test_transform.cpp)
target_link_libraries(test_transform PRIVATE gfx_core)
add_test(NAME transform COMMAND test_transform)

add_executable(test_transform_edit test_transform_edit.cpp)
target_link_libraries(test_transform_edit PRIVATE gfx_core)
add_test(NAME transform_edit COMMAND test_transform_edit)

add_executable(test_raster test_raster.cpp)
target_link_libraries(test_raster PRIVATE gfx_core)
add_test(NAME raster COMMAND test_raster)
