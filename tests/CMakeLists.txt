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
add_test(NAME obj COMMAND test_obj)

add_executable(test_transform test_transform.cpp)
target_link_libraries(test_transform PRIVATE gfx_core)
add_test(NAME transform COMMAND test_transform)
