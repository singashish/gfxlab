# gfxlab

A small 3D model viewer, written from scratch to learn graphics programming.
The goal is a deliberately minimal take on a Blender-style viewport: an
infinite grid floor, an orbit camera, and the ability to load a model and
view, move, rotate and scale it (no mesh editing).

The plan is to render the same scene with three independent backends, so each
API can be compared side by side:

- **OpenGL 4.6**
- **Vulkan 1.3+**
- **CUDA**, as a software rasterizer, a ray tracer and a path tracer written
  from scratch as compute kernels

The only external dependencies are the graphics APIs themselves. Windowing,
input, math, logging, the OpenGL function loader, the model, image and glTF
parsers and the user interface are all written in this repository, as they are
added.

## Status

Early development. **M0 (foundation), M1 (window and input), M2 (OpenGL
renderer and orbit camera), M3 (models, lighting and object transforms), M4
(the Vulkan backend), M5 (the CUDA software rasterizer), M6 (wireframe, GPU
timing and CUDA clipping), M7 (the CUDA ray tracer), M8 (a basic user
interface) and M9 (materials, textures and a CUDA path tracer) are complete.**
The app opens a window with a
Blender-style viewport: an infinite, anti-aliased floor grid with colored X and
Y axes that fades with distance, and a mouse-controlled orbit camera. Give it
an OBJ file (with its MTL materials and textures) or a glTF/GLB file on the
command line and the model is loaded, shaded with its materials and textures,
placed on the floor, and can be moved, rotated and scaled with Blender's G, R
and S keys.
Without a file it shows two demo triangles. The scene keeps rendering while the
window is dragged or resized. It renders with OpenGL by default, with Vulkan
when started with `--backend=vulkan`, or with a software rasterizer written in
CUDA when started with `--backend=cuda`, and all three backends produce the
same picture. **W** switches every backend to a Blender-style wireframe that
shows the model's original polygon edges, and each backend's measured GPU
time is shown next to the frame time. With `--backend=cuda`,
**T** switches to a ray tracer that adds shadows from a sun you can drag around
the model and ambient occlusion that refines while the view is still, and
**P** to a progressive path tracer ("Rendered" mode) with a sun, a sky, glossy
and metallic materials and emissive surfaces that light their surroundings. A
panel in the top-left corner, drawn by every backend, shows the timings and
lets the mouse change the view, the model's position, rotation and scale, the
ray tracer's and path tracer's settings and, with CUDA, the model's materials
live. The panel can be hidden to the left edge, and scrolls when the window is
shorter than it.

| Milestone | Goal | State |
|-----------|------|-------|
| M0 | Build system, core library (logging, math), tests | DONE |
| M1 | Win32 window, input, frame timer | DONE |
| M2 | OpenGL backend: grid and orbit camera | DONE |
| M3 | OBJ loading, lit mesh, object transforms | DONE |
| M4 | Vulkan backend | DONE |
| M5 | CUDA rasterizer | DONE |
| M6 | Polish and backend comparison | DONE |
| M7 | CUDA ray tracer | DONE |
| M8 | Basic user interface | DONE |
| M9 | Materials, textures and a CUDA path tracer | DONE |

### What is in the repository so far

- **Build system**: CMake with Debug and Release presets, `/W4` warnings on
  MSVC, and automatic detection of the optional backends
- **`gfx_core`**: a printf-style logger, header-only math (`Vec3`, `Vec4`,
  `Mat4`), camera matrices (`look_at`, and `perspective` for both OpenGL and
  Vulkan clip space), the orbit camera, a frame timer, the `Mesh` type with an
  own OBJ parser and normal generation, `Transform`, the `TransformEditor`
  behind G/R/S, ray/box and ray/triangle tests, and a BVH (binary, built with
  the surface area heuristic, and a 4-wide version collapsed from it) whose
  traversal code runs on both the CPU and the GPU. Since M9 it also holds the
  materials (`Material`, the MTL parser), the image decoders (Deflate, PNG and
  TGA), a JSON parser and the glTF/GLB loader behind a common `Model` type,
  software texture sampling with mip maps, the path tracer's maths (sRGB and
  tone mapping, the sky and sun, the GGX material model in `brdf.h`, sampling
  of emissive triangles), the path tracer's settings and the material editor's
  helpers, all header-only and shared with the GPU code where it matters
- **`gfx_ui`**: an immediate-mode UI: a font atlas packer, the `UiDrawList`
  of UI triangles every backend draws, a software rasterizer for that list
  (used by the CUDA backend, and by the tests), and widgets (panels, text,
  buttons, checkboxes, sliders, drag fields, foldable sections) with mouse
  capture and, since M9, panels that scroll when taller than the window
- **`gfx_platform`**: a hand-written Win32 window (DPI-aware, resize and
  minimize handling, rendering during the drag/resize loop), a keyboard and
  mouse state tracker with mouse capture and focus-loss handling, and the UI
  font, rasterized with GDI
- **`gfx_gl`**: an OpenGL 4.6 core context created through WGL, an own function
  loader, shader helpers, the ray-cast grid, lit mesh drawing (`GLMesh`), an
  edge-list wireframe, a non-blocking GPU timer (`GL_TIME_ELAPSED` queries), the
  UI pass, per-material drawing with textures and mip maps
  (`GLTextureStore`), and `GLRenderer`
- **`gfx_vk`**: the Vulkan 1.3 backend: instance, validation layer and device
  selection (a discrete GPU is preferred), swapchain with live recreation,
  frames in flight, dynamic rendering, a depth buffer, staging uploads, per-frame
  uniform buffers and push constants, a shared pipeline helper, shaders compiled
  to SPIR-V at build time, a line-list wireframe pipeline, timestamp-query GPU
  timing, the UI pass (with image uploads and per-frame UI buffers),
  per-material drawing with textures and mip maps (`VulkanTextureStore`, one
  descriptor set per texture), and `VulkanRenderer`
- **`gfx_cuda`**: the CUDA backend: a software rasterizer made of kernels
  (vertex transform; a triangle setup pass that clips against the near plane
  and a guard band and fills a worklist; small and large triangle rasterizers
  with fixed-point edge functions and the top-left rule; a 64-bit `atomicMin`
  visibility buffer; deferred shading that matches the OpenGL lighting; a
  clipped line rasterizer for wireframe; and the grid as a per-pixel ray
  kernel), a ray tracer (camera rays through a per-mesh BVH, sun shadows,
  progressive ambient occlusion, run as one kernel or as wavefront passes),
  a progressive path tracer (sun and sky light, a glossy GGX layer, emission,
  sampled emissive triangles, a firefly limit; its code is in
  `cuda_pathtrace.cuh` and `cuda_rt_device.cuh`), per-material drawing with
  software texture sampling (`CudaTextureStore`, mip chains, LOD from the
  triangle), a device material table with live edits (`CudaMeshStore`),
  the UI composited by a kernel, a presenter interface (`ICudaPresenter`) with
  the CUDA/OpenGL interop presenter that shows the result, per-frame and
  per-stage GPU timing, ray statistics, and `CudaRenderer`. The `cuda_info` tool reports the CUDA
  device
- **`gfx_renderer`**: the backend-neutral `IRenderer` interface (including
  `last_gpu_ms` for GPU timing and `stats` for the UI), `FrameData` with its
  draw list, flags and UI draw list, the shared grid ray basis, and the
  `create_renderer()` factory
- **Tests**: eighteen unit-test programs (math, input, timer, camera, obj,
  transform, transform_edit, raster, bvh, ui, image, texture, json, gltf, path,
  brdf, light, material_edit) registered with CTest
- **`gfxlab` app**: opens a 1280x720 window and renders the viewport, loading
  the OBJ or glTF/GLB file given on the command line, with the settings panel
  on top. The
  title bar shows the backend and the FPS, the panel the frame and GPU times,
  and the console logs the frame and GPU times once per second. **Esc** cancels
  an edit, or quits when none is active. `--backend=opengl|vulkan|cuda` selects
  the renderer
- **`assets/`**: test models (a textured cube, `textured_cube.obj` with its MTL
  file and `textures/uv_grid.png` and `label.tga`, glTF test files in
  `assets/gltf/` (`cubes.gltf` and `cubes.glb`), a cube with and without normals, a
  high-resolution Suzanne with flat and smooth shading, and `suzanne_fl.obj`,
  Blender's standard Suzanne exported with its quads, for comparing the
  wireframe with Blender). `assets/gltf/materials.glb` (made by
  `tests/make_material_test_model.py`) is the material test bench: ten
  spheres (dielectric above, metal below, roughness 0.05 to 1 from left to
  right; in the path tracer the gold metals go from mirror-like to rough and
  the blue dielectrics from sharp sun glints to a soft sheen), a floor and an
  emissive panel

### Controls

| Input | Action |
|-------|--------|
| Middle mouse drag | Orbit around the target (not during a G/R/S edit) |
| Shift + middle mouse drag | Pan |
| Scroll wheel | Zoom (over the 3D view); over the panel it scrolls the panel when the panel is taller than the window |
| **G** / **R** / **S** | Grab (move), rotate or scale the model (only when a model is loaded) |
| **X** / **Y** / **Z** during an edit | Constrain to that axis; press again to go back to free |
| Left click during an edit | Confirm |
| Right click or Esc during an edit | Cancel, restoring the model exactly |
| **V** | Toggle vsync |
| **W** | Toggle wireframe (the model's polygon edges, unlit, in every backend) |
| **M** | Toggle materials: the models' colours and textures, or plain grey shading (in every backend; in the path tracer also gloss, metal and emission) |
| **T** | Toggle ray tracing (`--backend=cuda` only; wireframe takes priority) |
| **P** | Toggle the path tracer, the progressive "Rendered" mode, lit by a sun and a sky that are not drawn themselves (`--backend=cuda` only; excludes **T**; wireframe takes priority). Emissive materials glow and light their surroundings, and the roughness and metallic values of the materials show as gloss and metal (GGX highlights and reflections; try `gltf\materials.glb`); **M** turns the materials, and so the emission and the gloss, off (plain matte grey). The panel's Path tracing > Settings group (folded by default) sets the sample limit, samples per frame, bounces, exposure, sun and sky strength and the firefly limit |
| Panel > Materials | The collapsible Materials section (folded by default; `--backend=cuda` only) picks one of the loaded model's materials and edits its base colour, roughness, metallic and emission colour and strength; the rasterizer, **T** and **P** show it live (**P** restarts its accumulation). Textures are not editable (the colour still multiplies them) and the edits are in memory only, never saved to the file |
| Right mouse drag (ray or path tracing) | Move the sun: left/right around the model, up/down higher or lower |
| **L** (ray or path tracing) | Sun on/off (ray tracing: the viewport's camera light, no shadows; path tracing: sky light only) |
| **O** (ray tracing) | Ambient occlusion on/off |
| **H** | Hide or show the panel (the panel's **<< Hide** button hides it, the **>>** tab at the left edge shows it again) |
| Left click / drag on the panel | Use its checkboxes, buttons, sliders and fields; click a section header to fold it |
| Esc (no edit active) | Quit |

The 3D view ignores the mouse while it is over the panel (no zoom, orbit or
sun drag starts there); a middle drag started in the view continues over it.
During a G/R/S edit the panel ignores the mouse, so clicks always confirm or
cancel the edit, and the camera stays locked except for zoom.

## Documentation

| Document | Contents |
|----------|----------|
| [`docs/plan.txt`](docs/plan.txt) | Overall goal, key decisions, layout and milestones |
| [`docs/m0-design.md`](docs/m0-design.md) | Build system, backend detection, logging, math and camera conventions |
| [`docs/m1-design.md`](docs/m1-design.md) | Window, input model, frame timer, known limitations, manual test checklist |
| [`docs/m2-design.md`](docs/m2-design.md) | OpenGL context and loader, shaders, orbit camera, ray-cast grid, renderer interface, rendering during window drag, tests, manual checklist |
| [`docs/m3-design.md`](docs/m3-design.md) | OBJ parser, mesh upload and lighting, normals, transforms and the normal matrix, interactive G/R/S editing, tests, manual checklist |
| [`docs/m4-design.md`](docs/m4-design.md) | Vulkan backend: instance and device, swapchain, frames in flight and synchronization, memory and uploads, descriptors and push constants, pipelines, the grid, live resizing, differences from OpenGL, manual checklist |
| [`docs/m5-design.md`](docs/m5-design.md) | CUDA backend: toolchain, CUDA/OpenGL interop, device meshes, the rasterizer (fixed point, edge functions, top-left rule, atomic visibility buffer), deferred shading, the grid kernel, the large-triangle performance fix, known limitations, manual checklist |
| [`docs/m6-design.md`](docs/m6-design.md) | Polygon-edge wireframe in all three backends, non-blocking GPU timing (GL queries, Vulkan timestamps, CUDA events), the backend comparison and how to read it, the CUDA triangle worklist, near-plane and guard-band clipping, the shared grid ray basis, manual checklist |
| [`docs/m7-design.md`](docs/m7-design.md) | CUDA ray tracer: ray tests, the SAH BVH and its 4-wide version, instancing with the inverse model matrix, sun shadows, ambient occlusion with progressive refinement, ray statistics, megakernel versus wavefront, measurements and what they taught, manual checklist |
| [`docs/m8-design.md`](docs/m8-design.md) | User interface: the GDI font atlas, the UI draw list, the UI pass in OpenGL and Vulkan, the CUDA presenter interface and the software UI rasterizer, immediate-mode widgets and mouse ownership, the settings panel, the cost of the UI, manual checklist |
| [`docs/m9-design.md`](docs/m9-design.md) | Materials and textures: the MTL parser, Deflate, PNG and TGA decoders, the JSON parser and glTF/GLB loader, textured drawing in all three backends, the CUDA path tracer (linear light, sun and sky, the GGX material model, emissive triangles and MIS, the firefly limit), the settings panel, hiding and scrolling, the material editor, manual checklist |

## Requirements

- Windows 10 (version 1703 or later) or Windows 11
- A GPU and driver that support OpenGL 4.6 (and Vulkan 1.3 for the Vulkan
  backend)
- MSVC (Visual Studio 2022 Build Tools, no IDE needed), CMake 3.21+, Ninja
- Optional: [Vulkan SDK](https://vulkan.lunarg.com/) for the Vulkan backend
- Optional: CUDA Toolkit for the CUDA backend (12.8 or newer for RTX 50 series;
  developed with 13.4). The CUDA backend also needs OpenGL, which it uses to
  show the finished picture

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
into its own directory (`builds/build-debug`, `builds/build-release`, the same
folders `build.bat` uses), and the executable is at
`builds/build-debug/src/app/gfxlab.exe`.

### With build.bat

`build.bat` builds both configurations in one run and then prints ready-to-copy
commands to run the tests and the app with OpenGL and Vulkan. It configures
with Ninja (without a preset, and with `-DGFX_CUDA_ARCHITECTURES=120`) and builds
incrementally; `build.bat clean` deletes both build folders first. It must be
run from a terminal that has the MSVC environment.

| Script | Build type | Output directory |
|--------|------------|------------------|
| `build.bat` | Debug and Release | `builds\build-debug\` and `builds\build-release\` |

The executable is in `src\app\` and the tests are in `tests\` under the output
directory. Edit the architecture line in `build.bat` (and in
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

The CUDA architecture is chosen with `GFX_CUDA_ARCHITECTURES` (`CUDA_ARCH` in
`build.bat`). The presets and the build script set it to `120` (RTX 50 series).
`nvcc` builds machine code for each listed generation plus portable PTX that
the driver can compile for newer GPUs, never for older ones, so a build for
`120` does not run on an RTX 30 or 40 card. Set the value for your GPU, or list
several to build one executable for many cards:

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

If the value is empty, CMake uses `nvcc`'s default, which for CUDA 13 is the
oldest supported generation (`75`, RTX 20 series). That runs on every RTX card
through the driver's compiler, with a slower first start and code not tuned
for newer chips. CUDA 13 does not support GPUs older than the RTX 20 series.
Run `build.bat clean` after changing the value.

On a laptop with an integrated GPU as well as an NVIDIA one, Windows may run
OpenGL on the integrated GPU. The CUDA backend then fails to start with "no
CUDA device is associated with the current OpenGL context", because it shows
its picture through OpenGL. Set `gfxlab.exe` to "High performance" in Windows
graphics settings or in the NVIDIA Control Panel.

Detection results are cached. After installing the Vulkan SDK or the CUDA
Toolkit, delete the cache (or the build directory) and configure again.

## Running

```
builds\build-debug\src\app\gfxlab.exe
builds\build-debug\src\app\gfxlab.exe assets\suzanne_s.obj
builds\build-debug\src\app\gfxlab.exe --backend=vulkan assets\suzanne_s.obj
builds\build-debug\src\app\gfxlab.exe --backend=cuda assets\suzanne_s.obj
```

The second form loads a model. The name is looked up as given, then in `assets\`,
then in `assets\local\`, so a bare name such as `suzanne_s.obj` works from
any directory. The loader chooses the format by extension: `.obj` (positions,
normals, texture coordinates and faces, with the materials of the `.mtl` file
and its `map_Kd` texture) or `.gltf` / `.glb` (see `assets\gltf\`); PNG and
TGA textures are supported. Both formats are assumed to be Y-up, as most
exporters write them, and are converted to the viewer's Z-up. The third
form renders it with Vulkan instead of OpenGL (`--backend=opengl` is the
default); it needs the Vulkan SDK at build time and a Vulkan 1.3 GPU at run
time. In a Debug build it also enables the Khronos validation layer and prints
`Vulkan validation layer enabled`; any `[Vulkan validation]` line in the console
is a bug to investigate. The fourth form renders with the CUDA software
rasterizer; it needs the CUDA Toolkit at build time and an NVIDIA GPU that
OpenGL also runs on. It logs the CUDA device it uses (`CUDA/OpenGL interop
device: ...`) and the presenter (`CUDA presenter: OpenGL interop`). With no model it shows only the grid. For a per-stage breakdown
of the CUDA frame, set `kProfileStages` to `true` in `cuda_renderer.cpp`: a
`CUDA GPU ms: clear ... geometry+raster ... shade ... grid ...` line then
appears once per second. It waits for the GPU every frame, so leave it `false`
when measuring frame rates.

The panel and the once-per-second log line (`<backend> | frame ms ... |
GPU ms ...`) show two different numbers. The frame time is the wall-clock time
from one frame to the next, including CPU work, vsync and presentation. The GPU
time is how long the GPU spent executing the frame's commands. To compare
backends, use a Release build with vsync off (**V**), the same window size and
the startup camera.

With `--backend=cuda` and a model loaded, **T** (or the panel's Ray tracing
checkbox) switches to the ray tracer. The panel's Ray tracing section shows the
rays traced per frame and the AO samples, and the console adds a
once-per-second line with the same numbers:
`CUDA rays per frame: camera ... M, shadow ... M, AO ... M (AO samples n / 1024)`.
Ambient occlusion keeps adding rays while nothing moves and stops at 1024
samples per pixel, after which it costs nothing. Measuring aids in
`cuda_renderer.cpp`: `kBenchmarkAo` (restart AO every frame so its cost can be
timed), `kRtWavefront` and `kUseBvh4` (compare the two ways of running the ray
tracer and the two BVH layouts). Keep the defaults for normal use, and keep the
window focused while measuring: unfocused it measures a few percent faster.

`builds\build-debug\src\renderer\cuda\cuda_info.exe` prints the CUDA runtime,
driver and device properties and runs a small kernel. It is the quickest way to
check the CUDA installation.

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
src/core/            logging, math, cameras, timer, mesh and materials,
                     OBJ/MTL parser, Deflate/PNG/TGA, JSON, glTF loader,
                     texture sampling, path tracer maths (sRGB, BRDF,
                     light sampling, settings), material editor helpers,
                     transform and its editor, ray tests, BVH          -> gfx_core
src/ui/              font atlas, UI draw list, software UI rasterizer,
                     immediate-mode widgets (scrolling panels)         -> gfx_ui
src/platform/        window, input and the UI font                     -> gfx_platform
  win32/             Win32 window and GDI font rasterization
src/renderer/        IRenderer, FrameData, grid ray basis, factory     -> gfx_renderer
  opengl/            WGL context, loader, shaders, grid, wireframe,
                     GPU timer, UI pass, GLRenderer                    -> gfx_gl
  vulkan/            instance/device, swapchain, frames, memory,
                     pipelines, mesh, wireframe, grid, GPU timer,
                     UI pass, VulkanRenderer                           -> gfx_vk
    shaders/         GLSL sources, compiled to SPIR-V by glslc at build time
  cuda/              presenter interface and OpenGL interop presenter,
                     device meshes and materials, textures, clipping,
                     rasterizer, ray tracer, path tracer and UI kernels,
                     GPU and stage timers, ray statistics, CudaRenderer,
                     cuda_info                                         -> gfx_cuda
src/app/             application entry point                           -> gfxlab
assets/              test models (.obj, .gltf, .glb), textures
tests/               unit tests                                        -> test_math, test_input, test_timer,
                                                                          test_camera, test_obj, test_transform,
                                                                          test_transform_edit, test_raster, test_bvh,
                                                                          test_ui, test_image, test_texture, test_json,
                                                                          test_gltf, test_path, test_brdf, test_light,
                                                                          test_material_edit
CMakePresets.json
build.bat
```

The OpenGL shaders are embedded in the C++ sources, and the Vulkan shaders are
separate files that `glslc` compiles into `<build>/shaders`. The CUDA backend
has no shader files: its kernels are in `cuda_raster.cu` and `cuda_ui.cu`,
compiled by `nvcc`.

## Conventions

- Right-handed world, **Z up**, column-major matrices, `float` throughout
- `Mat4` is four `Vec4` columns (64 bytes, no padding), so it can be uploaded
  to the GPU without a transpose
- The camera looks down -Z in view space, with +X right and +Y up
- `perspective()` takes a `ClipSpace` argument: OpenGL uses NDC depth in
  [-1, 1] with Y up, Vulkan uses [0, 1] with Y down. The CUDA backend uses the
  OpenGL clip space and a viewport with row 0 at the top. All backends show the
  same picture, with counter-clockwise triangles front-facing
- Mouse coordinates are client pixels with the origin at the top left and Y
  pointing down
- Math functions are marked `GFX_HD` so the same code compiles for both the
  host and the GPU: the CUDA kernels use the same `Vec3`, `Mat4` and
  `raster_math.h` code as the CPU and the unit tests
- Names avoid collisions with `<windows.h>` macros, for example
  `LOG_LEVEL_ERROR` instead of `ERROR`, and `near_plane` instead of `near`
- Model files are assumed to be Y-up and are rotated into the Z-up world on
  load. Triangles are counter-clockwise when seen from outside
- Object rotation uses Blender's XYZ Euler order (`R = Rz * Ry * Rx`), and the
  model matrix is `T * R * S`
