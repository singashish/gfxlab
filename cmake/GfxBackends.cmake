# Optional graphics backend detection for GFXLab.
# ON means "enable if the required development tools are available."

option(GFX_ENABLE_OPENGL "Build the OpenGL backend" ON)
option(GFX_ENABLE_VULKAN "Build the Vulkan backend" ON)
option(GFX_ENABLE_CUDA   "Build the CUDA backend"   ON)

# Detection results
set(GFX_HAVE_OPENGL OFF)
set(GFX_HAVE_VULKAN OFF)
set(GFX_HAVE_CUDA   OFF)

# --------------------------------------------------
# OpenGL
# --------------------------------------------------
if(GFX_ENABLE_OPENGL)
    find_package(OpenGL QUIET)

    if(OpenGL_FOUND)
        set(GFX_HAVE_OPENGL ON)
        message(STATUS "GFXLab: OpenGL found")
    else()
        message(STATUS "GFXLab: OpenGL not found, backend disabled")
    endif()
else()
    message(STATUS "GFXLab: OpenGL disabled by option")
endif()

# --------------------------------------------------
# Vulkan + glslc
# --------------------------------------------------
if(GFX_ENABLE_VULKAN)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        # CMake 3.24+ supports Vulkan components
        find_package(Vulkan QUIET COMPONENTS glslc)

        if(Vulkan_FOUND)
            set(GFX_HAVE_VULKAN ON)
            message(STATUS "GFXLab: Vulkan and glslc found")
        else()
            message(STATUS
                "GFXLab: Vulkan SDK or glslc not found, backend disabled")
        endif()
    else()
        # Backward-compatible path for CMake 3.20–3.23
        find_package(Vulkan QUIET)

        if(Vulkan_FOUND AND TARGET Vulkan::glslc)
            set(GFX_HAVE_VULKAN ON)
            message(STATUS "GFXLab: Vulkan and glslc found")
        else()
            message(STATUS
                "GFXLab: Vulkan SDK or glslc not found, backend disabled")
        endif()
    endif()
else()
    message(STATUS "GFXLab: Vulkan disabled by option")
endif()

# --------------------------------------------------
# CUDA
# --------------------------------------------------
# Optional list of CUDA architectures, provided by the user
set(GFX_CUDA_ARCHITECTURES "" CACHE STRING
    "CUDA architectures to compile for, e.g. 75;86;89;120"
)

if(GFX_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)
        # Finding nvcc alone is insufficient. We also need the
        # CUDA toolkit, including the CUDA runtime library
        find_package(CUDAToolkit QUIET)

        if(CUDAToolkit_FOUND)
            if(GFX_CUDA_ARCHITECTURES)
                set(CMAKE_CUDA_ARCHITECTURES
                    "${GFX_CUDA_ARCHITECTURES}"
                )
            endif()

            set(CMAKE_CUDA_STANDARD 17)
            set(CMAKE_CUDA_STANDARD_REQUIRED ON)

            enable_language(CUDA)

            set(GFX_HAVE_CUDA ON)

            message(STATUS
                "GFXLab: CUDA compiler found; architectures="
                "${CMAKE_CUDA_ARCHITECTURES}"
            )
        else()
            message(STATUS
                "GFXLab: CUDA toolkit not found, backend disabled"
            )
        endif()
    else()
        message(STATUS
            "GFXLab: CUDA compiler not found, backend disabled"
        )
    endif()
else()
    message(STATUS "GFXLab: CUDA disabled by option")
endif()

# --------------------------------------------------
# Shared build configuration
# --------------------------------------------------
add_library(gfx_config INTERFACE)

if(GFX_HAVE_OPENGL)
    target_compile_definitions(gfx_config INTERFACE
        GFX_HAS_OPENGL=1
    )
endif()

if(GFX_HAVE_VULKAN)
    target_compile_definitions(gfx_config INTERFACE
        GFX_HAS_VULKAN=1
    )
endif()

if(GFX_HAVE_CUDA)
    target_compile_definitions(gfx_config INTERFACE
        GFX_HAS_CUDA=1
    )
endif()

# --------------------------------------------------
# Summary
# --------------------------------------------------
message(STATUS
    "Backends: OpenGL=${GFX_HAVE_OPENGL} "
    "Vulkan=${GFX_HAVE_VULKAN} "
    "CUDA=${GFX_HAVE_CUDA}"
)
