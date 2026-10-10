# gfxlab

A small 3D model viewer, written from scratch to learn graphics programming.
The goal is a deliberately minimal take on a Blender-style viewport: an
infinite grid floor, an orbit camera, and the ability to load a model and
view, move, rotate and scale it (no mesh editing).

The plan is to render the same scene with three independent backends, so each
API can be compared side by side:

- **OpenGL 4.6**
- **Vulkan 1.3+**
- **CUDA**, as a software rasterizer (CUDA ray tracing may follow later)

The only external dependencies are the graphics APIs themselves. Windowing,
input, math, logging, the OpenGL function loader and the model parser are all
written in this repository, as they are added.

The full design and milestone plan is in [`docs/plan.txt`](docs/plan.txt).

## Status

Early development. **M0 (project foundation) is complete.** No window or
renderer exists yet; the next milestone starts the Win32 window layer.

| Milestone | Goal | State |
|-----------|------|-------|
| M0 | Build system, core library (logging, math), tests | DONE |
| M1 | Win32 window and input | |
| M2 | OpenGL backend: grid and orbit camera | |
| M3 | OBJ loading, lit mesh, object transforms | |
| M4 | Vulkan backend | |
| M5 | CUDA rasterizer | |
| M6 | Polish and backend comparison | |
| M7 | CUDA ray tracer | |

### What M0 contains

- **CMake build** with presets (Debug and Release), `/W4` warnings on MSVC,
  and automatic detection of the optional backends
- **`gfx_core` static library**: a printf-style logger with levels and
  file:line output
- **Header-only math**: `Vec3`, `Vec4`, `Mat4` (identity, multiply, translate,
  scale, rotate, point and direction transforms), and camera matrices
  (`look_at`, and `perspective` for both OpenGL and Vulkan clip space)
- **Unit tests** using a tiny assert harness registered with CTest
- **`gfxlab` executable**: currently only logs which backends were detected
  at configure time

## Requirements

- Windows 10/11
- MSVC (Visual Studio 2022 Build Tools, no IDE needed), CMake 3.21+, Ninja
- Optional: [Vulkan SDK](https://vulkan.lunarg.com/) for the Vulkan backend
- Optional: CUDA Toolkit 12.8+ for the CUDA backend (NVIDIA GPU; the build is
  set up for compute capability 12.0, an RTX 50 series card)

Backends whose dependencies are not found are skipped automatically.

## Building

All commands need the MSVC environment. In VS Code, the CMake Tools extension
sets it up when you select a configure preset. Otherwise use an
"x64 Native Tools Command Prompt for VS 2022".

### With presets

```
cmake --preset windows-debug
cmake --build --preset debug
ctest --preset debug
```

Use `windows-release` / `release` for an optimized build. Each preset builds
into its own directory (`build/windows-debug`, `build/windows-release`), and
the executable is at `build/windows-debug/src/app/gfxlab.exe`.

### With `build.bat`

`build.bat` is a convenience script for quick iteration. It deletes `build/`,
configures with Ninja (without a preset, so no explicit build type), builds,
and then shows a menu to run `gfxlab.exe`, run `test_math.exe`, or run all
tests through CTest. Its output goes to `build/src/app/`.

### Backend options

Each backend is controlled by a CMake option, all `ON` by default, where `ON`
means "build it if its tools are found":

| Option | Requires |
|--------|----------|
| `GFX_ENABLE_OPENGL` | OpenGL (ships with the Windows SDK) |
| `GFX_ENABLE_VULKAN` | Vulkan SDK, including `glslc` |
| `GFX_ENABLE_CUDA` | CUDA Toolkit (found as a CMake language) |

Configuration prints a summary line such as
`Backends: OpenGL=ON Vulkan=OFF CUDA=OFF`. Each detected backend defines
`GFX_HAS_OPENGL`, `GFX_HAS_VULKAN` or `GFX_HAS_CUDA` for the code that links
`gfx_config`. To turn one off explicitly:

```
cmake --preset windows-debug -DGFX_ENABLE_CUDA=OFF
```

Detection results are cached. After installing the Vulkan SDK or the CUDA
Toolkit, delete the cache (or the build directory) and configure again.

## Project layout

```
cmake/          build helper modules (backend detection)
docs/           design plan
src/core/       logging, math, common macros      -> gfx_core library
src/app/        application entry point            -> gfxlab executable
tests/          unit tests                         -> test_math executable
CMakePresets.json
build.bat
```

Planned, not yet present: `src/platform/win32/` (window and input),
`src/scene/` (camera, mesh, OBJ loader), `src/renderer/{opengl,vulkan,cuda}/`,
`shaders/` and `assets/`.

## Conventions

- Right-handed world, **Z up**, column-major matrices, `float` throughout
- `Mat4` is four `Vec4` columns (64 bytes, no padding), so it can be uploaded
  to the GPU without a transpose
- The camera looks down -Z in view space, with +X right and +Y up
- `perspective()` takes a `ClipSpace` argument: OpenGL uses NDC depth in
  [-1, 1] with Y up, Vulkan uses [0, 1] with Y down
- Math functions are marked `GFX_HD` so the same code can later be compiled
  by `nvcc` for the CUDA backend
- Names avoid collisions with `<windows.h>` macros, for example
  `LOG_LEVEL_ERROR` instead of `ERROR`, and `near_plane` instead of `near`
