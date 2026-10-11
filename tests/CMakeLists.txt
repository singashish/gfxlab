add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE gfx_core)
add_test(NAME math COMMAND test_math)

add_executable(test_input test_input.cpp)
target_link_libraries(test_input PRIVATE gfx_core)
add_test(NAME input COMMAND test_input)
