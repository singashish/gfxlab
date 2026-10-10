# gfxlab

A small 3D model viewer, written from scratch to learn graphics programming.
It is a deliberately minimal take on a Blender-style viewport: an infinite
grid floor, an orbit camera, and the ability to load a model and view,
move, rotate and scale it (no mesh editing).

The same scene is rendered by three independent backends, so each API can be
compared side by side:

- **OpenGL 4.6**
- **Vulkan 1.3+**
- **CUDA**, a software rasterizer (CUDA ray tracing may follow later)

The only dependencies are the graphics APIs themselves. Windowing, input,
math, logging, the OpenGL function loader and the model parser are all
written in this repository.

## Status

Early development. See the milestones below.

| Milestone | Goal | State |
|-----------|------|-------|
| M0 | Build system, core library (logging, math), tests | done |
| M1 | Win32 window and input | |
| M2 | OpenGL backend: grid and orbit camera | |
| M3 | OBJ loading, lit mesh, object transforms | |
| M4 | Vulkan backend | |
| M5 | CUDA rasterizer | |
| M6 | Polish and backend comparison | |

## Requirements

- Windows 10/11, MSVC (Visual Studio 2022 Build Tools), CMake 3.21+, Ninja
- Optional: [Vulkan SDK](https://vulkan.lunarg.com/) for the Vulkan backend
- Optional: CUDA Toolkit 12.8+ for the CUDA backend (NVIDIA GPU)

Backends whose dependencies are not found are skipped automatically.

## Building

From VS Code with the CMake Tools extension, pick the `Debug` or `Release`
configure preset, then build. From the command line, use a Visual Studio
Native Tools prompt:

```
cmake --preset windows-debug
cmake --build --preset debug
ctest --preset debug
```

## Project layout

```
cmake/         build helper modules
src/core/      math, logging, common macros
src/app/       application entry point
tests/         unit tests
```

## Conventions

Right-handed, Z-up world, column-major matrices, `float` throughout.
