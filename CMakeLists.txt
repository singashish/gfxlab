cmake_minimum_required(VERSION 3.20)
project(gfxlab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options(/W4 /permissive- /utf-8)
endif()

# Optional backend detection
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(GfxBackends)

add_subdirectory(src/core)
add_subdirectory(src/platform)
add_subdirectory(src/app)

enable_testing()
add_subdirectory(tests)
