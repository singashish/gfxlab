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

## Status

Early development. **M0 (foundation) and M1 (window and input) are complete.**
The app opens a window and reports keyboard, mouse, resize and frame-timing
events, but nothing is rendered yet. The next milestone is the OpenGL backend.

| Milestone | Goal | State |
|-----------|------|-------|
| M0 | Build system, core library (logging, math), tests | DONE |
| M1 | Win32 window, input, frame timer | DONE |
| M2 | OpenGL backend: grid and orbit camera | |
| M3 | OBJ loading, lit mesh, object transforms | |
| M4 | Vulkan backend | |
| M5 | CUDA rasterizer | |
| M6 | Polish and backend comparison | |
| M7 | CUDA ray tracer | |

### What is in the repository so far

- **Build system**: CMake with Debug and Release presets, `/W4` warnings on
  MSVC, and automatic detection of the optional backends
- **`gfx_core`**: a printf-style logger, header-only math (`Vec3`, `Vec4`,
  `Mat4`) and camera matrices (`look_at`, and `perspective` for both OpenGL and
  Vulkan clip space), plus a frame timer
- **`gfx_platform`**: a hand-written Win32 window (DPI-aware, resize and
  minimize handling), and a keyboard and mouse state tracker with mouse capture
  and focus-loss handling
- **Tests**: three unit-test programs (math, input, timer) registered with CTest
- **`gfxlab` app**: opens a 1280x720 window and logs input and resize events to
  the console. The title bar shows the FPS and frame time. Press **Esc** to quit

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/plan.txt`](docs/plan.txt) | Overall goal, key decisions, layout and milestones |
| [`docs/m0-design.md`](docs/m0-design.md) | Build system, backend detection, logging, math and camera conventions |
| [`docs/m1-design.md`](docs/m1-design.md) | Window, input model, frame timer, known limitations, manual test checklist |

## Requirements

- Windows 10 (version 1703 or later) or Windows 11
- MSVC (Visual Studio 2022 Build Tools, no IDE needed), CMake 3.21+, Ninja
- Optional: [Vulkan SDK](https://vulkan.lunarg.com/) for the Vulkan backend
- Optional: CUDA Toolkit for the CUDA backend (12.8 or newer for RTX 50 series)

Backends whose dependencies are not found are skipped automatically. The
project currently builds on Windows only.

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
configures with Ninja (without a preset, so no explicit build type, and with
`-DGFX_CUDA_ARCHITECTURES=120`), builds, and then shows a menu to run
`gfxlab.exe`, run `test_math.exe`, or run all tests through CTest. Its output
goes to `build/src/app/` and `build/tests/`. Edit the architecture line in the
script if your GPU is not an RTX 50 series card.

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

### CUDA architecture

The CUDA architecture is chosen with `GFX_CUDA_ARCHITECTURES`. If it is empty,
CMake and `nvcc` use their own default, which is unlikely to match a current
GPU, so set it when CUDA is enabled:

| GPU | Value |
|-----|-------|
| RTX 20 series | `75` |
| RTX 30 series | `86` |
| RTX 40 series | `89` |
| RTX 50 series | `120` |

```
cmake --preset windows-debug -DGFX_CUDA_ARCHITECTURES=120
cmake --preset windows-debug "-DGFX_CUDA_ARCHITECTURES=75;86;89;120"   # several GPUs
```

Detection results are cached. After installing the Vulkan SDK or the CUDA
Toolkit, delete the cache (or the build directory) and configure again.

## Running

```
build\windows-debug\src\app\gfxlab.exe
```

The app is a console program, so the log output is visible in the terminal. It
logs which backends were compiled in, then opens the window and reports:

- window resizes (with the aspect ratio) and minimize
- mouse button presses and releases, scroll steps, and cursor movement while
  the middle button is held
- key presses and releases (as numeric ids matching the `Key` enum order)

**Esc** quits.

## Project layout

```
cmake/            build helper modules (backend detection)
docs/             design documents
src/core/         logging, math, camera math, frame timer   -> gfx_core
src/platform/     window and input                          -> gfx_platform
  win32/          Win32 implementation of the window
src/app/          application entry point                   -> gfxlab
tests/            unit tests                                -> test_math, test_input, test_timer
CMakePresets.json
build.bat
```

Planned, not yet present: `src/scene/` (camera, mesh, OBJ loader),
`src/renderer/{opengl,vulkan,cuda}/`, `shaders/` and `assets/`.

## Conventions

- Right-handed world, **Z up**, column-major matrices, `float` throughout
- `Mat4` is four `Vec4` columns (64 bytes, no padding), so it can be uploaded
  to the GPU without a transpose
- The camera looks down -Z in view space, with +X right and +Y up
- `perspective()` takes a `ClipSpace` argument: OpenGL uses NDC depth in
  [-1, 1] with Y up, Vulkan uses [0, 1] with Y down
- Mouse coordinates are client pixels with the origin at the top left and Y
  pointing down
- Math functions are marked `GFX_HD` so the same code can later be compiled
  by `nvcc` for the CUDA backend
- Names avoid collisions with `<windows.h>` macros, for example
  `LOG_LEVEL_ERROR` instead of `ERROR`, and `near_plane` instead of `near`
