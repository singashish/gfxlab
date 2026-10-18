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

Early development. **M0 (foundation), M1 (window and input), M2 (OpenGL
renderer and orbit camera), M3 (models, lighting and object transforms) and M4
(the Vulkan backend) are complete.** The app opens a window with a
Blender-style viewport: an infinite, anti-aliased floor grid with colored X and
Y axes that fades with distance, and a mouse-controlled orbit camera. Give it
an OBJ file on the command line and the model is loaded, lit, placed on the
floor, and can be moved, rotated and scaled with Blender's G, R and S keys.
Without a file it shows two demo triangles. The scene keeps rendering while the
window is dragged or resized. It renders with OpenGL by default, or with Vulkan
when started with `--backend=vulkan`, and both backends produce the same
picture. The next milestone is M5: a CUDA software rasterizer.

| Milestone | Goal | State |
|-----------|------|-------|
| M0 | Build system, core library (logging, math), tests | DONE |
| M1 | Win32 window, input, frame timer | DONE |
| M2 | OpenGL backend: grid and orbit camera | DONE |
| M3 | OBJ loading, lit mesh, object transforms | DONE |
| M4 | Vulkan backend | DONE |
| M5 | CUDA rasterizer | |
| M6 | Polish and backend comparison | |
| M7 | CUDA ray tracer | |

### What is in the repository so far

- **Build system**: CMake with Debug and Release presets, `/W4` warnings on
  MSVC, and automatic detection of the optional backends
- **`gfx_core`**: a printf-style logger, header-only math (`Vec3`, `Vec4`,
  `Mat4`), camera matrices (`look_at`, and `perspective` for both OpenGL and
  Vulkan clip space), the orbit camera, a frame timer, the `Mesh` type with an
  own OBJ parser and normal generation, `Transform`, and the `TransformEditor`
  behind G/R/S
- **`gfx_platform`**: a hand-written Win32 window (DPI-aware, resize and
  minimize handling, rendering during the drag/resize loop), and a keyboard and
  mouse state tracker with mouse capture and focus-loss handling
- **`gfx_gl`**: an OpenGL 4.6 core context created through WGL, an own function
  loader, shader helpers, the ray-cast grid, lit mesh drawing (`GLMesh`), and
  `GLRenderer`
- **`gfx_vk`**: the Vulkan 1.3 backend: instance, validation layer and device
  selection (a discrete GPU is preferred), swapchain with live recreation,
  frames in flight, dynamic rendering, a depth buffer, staging uploads, per-frame
  uniform buffers and push constants, a shared pipeline helper, shaders compiled
  to SPIR-V at build time, and `VulkanRenderer`
- **`gfx_renderer`**: the backend-neutral `IRenderer` interface, `FrameData`
  with its draw list, and the `create_renderer()` factory
- **Tests**: seven unit-test programs (math, input, timer, camera, obj,
  transform, transform_edit) registered with CTest
- **`gfxlab` app**: opens a 1280x720 window and renders the viewport, loading
  the OBJ file given on the command line. The title bar shows the FPS, the frame
  time and the active edit mode. **Esc** cancels an edit, or quits when none is
  active. `--backend=opengl|vulkan` selects the renderer
- **`assets/`**: test models (a cube with and without normals, Suzanne flat and
  smooth)

### Controls

| Input | Action |
|-------|--------|
| Middle mouse drag | Orbit around the target |
| Shift + middle mouse drag | Pan |
| Scroll wheel | Zoom |
| **G** / **R** / **S** | Grab (move), rotate or scale the model (only when a model is loaded) |
| **X** / **Y** / **Z** during an edit | Constrain to that axis; press again to go back to free |
| Left click during an edit | Confirm |
| Right click or Esc during an edit | Cancel, restoring the model exactly |
| **V** | Toggle vsync |
| Esc (no edit active) | Quit |

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/plan.txt`](docs/plan.txt) | Overall goal, key decisions, layout and milestones |
| [`docs/m0-design.md`](docs/m0-design.md) | Build system, backend detection, logging, math and camera conventions |
| [`docs/m1-design.md`](docs/m1-design.md) | Window, input model, frame timer, known limitations, manual test checklist |
| [`docs/m2-design.md`](docs/m2-design.md) | OpenGL context and loader, shaders, orbit camera, ray-cast grid, renderer interface, rendering during window drag, tests, manual checklist |
| [`docs/m3-design.md`](docs/m3-design.md) | OBJ parser, mesh upload and lighting, normals, transforms and the normal matrix, interactive G/R/S editing, tests, manual checklist |
| [`docs/m4-design.md`](docs/m4-design.md) | Vulkan backend: instance and device, swapchain, frames in flight and synchronization, memory and uploads, descriptors and push constants, pipelines, the grid, live resizing, differences from OpenGL, manual checklist |

## Requirements

- Windows 10 (version 1703 or later) or Windows 11
- A GPU and driver that support OpenGL 4.6 (and Vulkan 1.3 for the Vulkan
  backend)
- MSVC (Visual Studio 2022 Build Tools, no IDE needed), CMake 3.21+, Ninja
- Optional: [Vulkan SDK](https://vulkan.lunarg.com/) for the Vulkan backend
- Optional: CUDA Toolkit for the CUDA backend (12.8 or newer for RTX 50 series)

Backends whose dependencies are not found are skipped automatically. The app
and the window layer build on Windows only. `gfx_core` and the unit tests are
portable, and also configure and build on other platforms.

### Installing the toolchain

In PowerShell, run the commands below. The `winget` package IDs can change over
time; if one is not found, run `winget search <name>` to get the current ID.

```powershell
winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
winget install Kitware.CMake
winget install Ninja-build.Ninja
```

The Build Tools install the MSVC compiler and the Windows SDK (the C++ workload).
Open a new terminal afterwards so the updated `PATH` is loaded, then check:

```powershell
cmake --version
ninja --version
```

`cl` is only available inside the MSVC environment (see [Building](#building)).
For VS Code, also install the C/C++ (`ms-vscode.cpptools`) and CMake Tools
(`ms-vscode.cmake-tools`) extensions.

The Vulkan SDK and the CUDA Toolkit are optional and only needed for their
backends. Install them from [LunarG](https://vulkan.lunarg.com/) (the SDK
includes `glslc` and the validation layers) and from
[NVIDIA](https://developer.nvidia.com/cuda-downloads) (12.8 or newer for
RTX 50 series).

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

### With the build scripts

`build.bat` builds both configurations in one run and then prints ready-to-copy
commands to run the tests and the app with OpenGL and Vulkan. It configures
with Ninja (without a preset, and with `-DGFX_CUDA_ARCHITECTURES=120`) and builds
incrementally; `build.bat clean` deletes both build folders first. It must be
run from a terminal that has the MSVC environment.

`build-debug.bat` and `build-release.bat` build one configuration each. Each
deletes its build directory, configures and builds, and then shows a menu to run
`gfxlab.exe`, run `test_math.exe`, or run all tests through CTest.

| Script | Build type | Output directory |
|--------|------------|------------------|
| `build.bat` | Debug and Release | `builds\build-debug\` and `builds\build-release\` |
| `build-debug.bat` | Debug | `builds\build-debug\` |
| `build-release.bat` | Release | `builds\build-release\` |

The executable is in `src\app\` and the tests are in `tests\` under the output
directory. Edit the architecture line in the scripts (and in
`CMakePresets.json`) if your GPU is not an RTX 50 series card.

### Core and tests only (any platform)

On a non-Windows host the app is skipped, and the portable core and tests build
with plain CMake:

```
cmake -S . -B build
cmake --build build
ctest --test-dir build
```

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

The CUDA architecture is chosen with `GFX_CUDA_ARCHITECTURES`. The presets and
the build scripts set it to `120` (RTX 50 series). If it is empty, CMake and `nvcc`
use their own default, which is unlikely to match a current GPU, so change it
to match yours when CUDA is enabled:

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
build\windows-debug\src\app\gfxlab.exe assets\suzanne_smooth.obj
build\windows-debug\src\app\gfxlab.exe --backend=vulkan assets\suzanne_smooth.obj
```

The second form loads a model. The name is looked up as given, then in `assets\`,
then in `assets\local\`, so a bare name such as `suzanne_smooth.obj` works from
any directory. The OBJ loader reads positions, normals and
faces, and assumes the file is Y-up (the default of most exporters). The third
form renders it with Vulkan instead of OpenGL (`--backend=opengl` is the
default); it needs the Vulkan SDK at build time and a Vulkan 1.3 GPU at run
time. In a Debug build it also enables the Khronos validation layer and prints
`Vulkan validation layer enabled`; any `[Vulkan validation]` line in the console
is a bug to investigate.

The app is a console program, so the log output is visible in the terminal. It
logs which backends were compiled in, creates the renderer (and prints the GPU
and API version), then opens the viewport. The console also reports:

- the loaded model (vertex and triangle counts, and whether normals were
  generated)
- window resizes (with the aspect ratio) and minimize
- edit events (`GRAB started`, `axis: X`, `confirmed`, `cancelled`)
- key presses and releases (as numeric ids matching the `Key` enum order)

See [Controls](#controls) for the mouse and keyboard actions.

## Project layout

```
cmake/               build helper modules (backend detection, shader compilation)
docs/                design documents
src/core/            logging, math, cameras, timer, mesh, OBJ parser,
                     transform and its editor                          -> gfx_core
src/platform/        window and input                                  -> gfx_platform
  win32/             Win32 implementation of the window
src/renderer/        IRenderer, FrameData, backend factory             -> gfx_renderer
  opengl/            WGL context, loader, shaders, grid, GLRenderer    -> gfx_gl
  vulkan/            instance/device, swapchain, frames, memory,
                     pipelines, mesh, grid, VulkanRenderer             -> gfx_vk
    shaders/         GLSL sources, compiled to SPIR-V by glslc at build time
src/app/             application entry point                           -> gfxlab
assets/              test models (.obj)
tests/               unit tests                                        -> test_math, test_input, test_timer,
                                                                          test_camera, test_obj, test_transform,
                                                                          test_transform_edit
CMakePresets.json
build.bat, build-debug.bat, build-release.bat
```

Planned, not yet present: `src/renderer/cuda/`. The OpenGL shaders are embedded
in the C++ sources, and the Vulkan shaders are separate files that `glslc`
compiles into `<build>/shaders`.

## Conventions

- Right-handed world, **Z up**, column-major matrices, `float` throughout
- `Mat4` is four `Vec4` columns (64 bytes, no padding), so it can be uploaded
  to the GPU without a transpose
- The camera looks down -Z in view space, with +X right and +Y up
- `perspective()` takes a `ClipSpace` argument: OpenGL uses NDC depth in
  [-1, 1] with Y up, Vulkan uses [0, 1] with Y down. Both backends show the
  same picture, with counter-clockwise triangles front-facing
- Mouse coordinates are client pixels with the origin at the top left and Y
  pointing down
- Math functions are marked `GFX_HD` so the same code can later be compiled
  by `nvcc` for the CUDA backend
- Names avoid collisions with `<windows.h>` macros, for example
  `LOG_LEVEL_ERROR` instead of `ERROR`, and `near_plane` instead of `near`
- Model files are assumed to be Y-up and are rotated into the Z-up world on
  load. Triangles are counter-clockwise when seen from outside
- Object rotation uses Blender's XYZ Euler order (`R = Rz * Ry * Rx`), and the
  model matrix is `T * R * S`
