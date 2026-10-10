add_executable(test_math test_math.cpp)
target_link_libraries(test_math PRIVATE gfx_core)
add_test(NAME math COMMAND test_math)
