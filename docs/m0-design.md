# M0 — Foundation: Build System, Core Library and Tests

> **Status:** Complete; implementation documented through Step 7  
> **Scope:** Set up the Windows toolchain, create the CMake project and its presets, build a small core library (logging and math, including camera matrices), add optional backend detection, and cover the math with unit tests. Nothing in M0 opens a window or draws anything.

## 1. Purpose

M0 builds everything the rest of the project stands on. Later milestones add a window, a renderer and a camera, and each of them depends on decisions that are cheap to make once and expensive to change later:

- how the project is built, and how that build is reproduced on another machine;
- which coordinate system, matrix layout and clip-space convention every piece of math uses;
- how the code reports what it is doing;
- how correctness is checked without staring at a rendered image.

The overall plan is in [`plan.txt`](plan.txt). The window, input and timer work that follows is described in [`m1-design.md`](m1-design.md), and the first renderer in [`m2-design.md`](m2-design.md).

At the end of Step 7, the project configures and builds from a clean checkout with a preset, `ctest` passes, and the app starts and logs which graphics backends were detected.

## 2. Architecture and dependency boundaries

M0 creates the first layers of the dependency stack. Layers depend downward only.

```
gfxlab        EXECUTABLE  src/app        entry point
gfx_core      STATIC      src/core       logging (log.cpp); math headers
gfx_config    INTERFACE   (no sources)   compile definitions for detected backends
test_*        EXECUTABLE  tests/         one program per area, registered with CTest
```

(`gfx_platform` is added in M1, and `gfx_gl` and `gfx_renderer` in M2.)

- `gfx_core` has no dependency on Windows, OpenGL or any other library. It exposes `src/` as a `PUBLIC` include directory, so anything linking it writes `#include "core/log.h"` with `src/` as the include root.
- `gfx_config` is an `INTERFACE` library. It has no code and only carries compile definitions to whatever links it.
- Each module has its own `CMakeLists.txt`. The root file holds only project-wide settings and the module list.
- On Windows, the root file defines `WIN32_LEAN_AND_MEAN`, `NOMINMAX`, `UNICODE` and `_UNICODE` for every target. The window layer and the app are added only when `WIN32` is set, while `gfx_core` and the tests are added everywhere, so the portable part configures, builds and passes `ctest` on any platform.

## 3. Toolchain setup (before Step 1)

The setup is deliberately small, because each backend's dependency is needed only once its milestone is reached.

| When | Install |
|---|---|
| Now (M0 to M3, OpenGL) | VS Build Tools 2022, CMake, Ninja, VS Code extensions, Git |
| Before M4 (Vulkan) | Vulkan SDK |
| Before M5 (CUDA) | CUDA Toolkit 12.8 or newer (required for RTX 50 series) |

### Decisions

| Area | Decision | Reason |
|---|---|---|
| Language | C++17 with `CMAKE_CXX_EXTENSIONS OFF` | Modern enough, portable, supported by `nvcc` |
| Compiler | MSVC from the Build Tools (no Visual Studio IDE) | `nvcc` on Windows needs `cl.exe` as its host compiler, so Clang or MinGW would block the CUDA backend |
| Generator | Ninja | Fast parallel builds. It works with `nvcc` without Visual Studio's CUDA integration, which can fail to install when only Build Tools are present |
| Editor | VS Code with C/C++ and CMake Tools | Preferred workflow |
| Dependencies | Only OpenGL, Vulkan SDK and CUDA Toolkit | Learning goal: write everything else in this repository |
| Floating point | `float` everywhere | Matches GPU data and shader code |

### Setup steps

1. **Install the tools.** Build Tools 2022 with the C++ workload (it brings the MSVC compiler and the Windows SDK, which the Win32 headers need), CMake and Ninja, via `winget`. Reopen the terminal so the new `PATH` entries load.
2. **Install the VS Code extensions:** C/C++ (`ms-vscode.cpptools`) for IntelliSense and debugging, and CMake Tools (`ms-vscode.cmake-tools`) for configure, build and debug.
3. **Make the MSVC environment available.** MSVC needs environment variables that a normal terminal does not have. The recommended route is a CMake Tools kit (**CMake: Select a Kit**, then *Visual Studio Build Tools 2022 Release - amd64*). The fallback is to start VS Code from the "x64 Native Tools Command Prompt for VS 2022".
4. **Verify.** `cl`, `cmake --version` and `ninja --version` all run in that environment.

## 4. Step 1 — Create the project and prove the toolchain

The first project is a one-file program that does nothing except confirm that MSVC, CMake and Ninja work together:

```cpp
#include <cstdio>

int main() {
    std::puts("gfxlab: toolchain OK");
    return 0;
}
```

The root `CMakeLists.txt` at this point:

- pins the CMake feature level with `cmake_minimum_required`, so older versions fail with a clear message instead of misbehaving;
- calls `project(gfxlab LANGUAGES CXX)`, which enables only the C++ compiler. This is deliberate: CUDA is enabled conditionally later, because enabling it unconditionally would break the build on any machine without the toolkit;
- requests C++17 and turns off compiler extensions to keep the code portable;
- sets `CMAKE_EXPORT_COMPILE_COMMANDS`, which writes `compile_commands.json` for VS Code's IntelliSense;
- declares one executable target with `add_executable`.

### How the flow works

*Configure* reads `CMakeLists.txt` and generates Ninja's build files. *Build* calls Ninja, which invokes `cl.exe` to compile and link. CMake itself never compiles anything. Everything in CMake is about targets, and the later steps add more of them.

## 5. Step 2 — Core library and logging

### Why logging comes first

Every later module (platform, renderer, app) needs to report what it is doing, so logging sits at the bottom of the layer stack. This step also establishes the CMake pattern repeated for every module: a static library with its own `CMakeLists.txt`, a `PUBLIC` include directory, and an executable that links it.

### The logging API

```cpp
enum class LogLevel { LOG_LEVEL_TRACE, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO,
                      LOG_LEVEL_WARN,  LOG_LEVEL_ERROR };
void log_set_level(LogLevel min_level);
void log_write(LogLevel level, const char* file, int line, const char* fmt, ...);
```

Application code uses the macros `LOG_TRACE`, `LOG_DEBUG`, `LOG_INFO`, `LOG_WARN` and `LOG_ERROR`, which call `log_write` with `__FILE__` and `__LINE__`.

- **Macros, not functions**, so the call site's file and line are captured. A plain function would always report its own location.
- **Variadic `__VA_ARGS__` forwarding** keeps calls with no format arguments legal. A `(fmt, ...)` macro breaks on the trailing comma.
- **printf-style** formatting needs no allocation or streams, works from C-style code, and stays readable when logging numbers.
- Output goes to `stderr` as `[LEVEL] file.cpp:line  message`. The directory is stripped from the path (both `/` and `\` are handled), and level names are padded to five characters (`[INFO ]`, `[WARN ]`) so the columns line up. The default minimum level is `INFO`.
- Logging is not thread-safe. That is acceptable while everything runs on one thread, and needs revisiting if a second thread ever logs.

### CMake visibility

`target_include_directories(gfx_core PUBLIC ...)` applies to `gfx_core` itself and to everything that links it. `PRIVATE` would apply only to the target, and `INTERFACE` only to its consumers. The same three keywords decide how every later target is wired.

## 6. Step 3 — Math conventions, `Vec3` and the test harness

### Conventions fixed once

These apply to the whole project:

- **Right-handed**, **Z up** (like Blender): `X cross Y = Z`.
- **Column vectors**, so a transform reads `M * v` and chained transforms read right to left. In `T * R * S`, scale applies first.
- **Column-major storage**, so matrices upload to the GPU without a transpose (see Step 4).
- **`w` distinguishes points from directions**: `w = 1` is a point (translation applies) and `w = 0` is a direction (translation is ignored).
- **Plain structs** with public members and no constructors, so they stay trivially copyable and can be copied into GPU buffers.
- **`GFX_HD`** (defined in `common.h`) expands to `__host__ __device__` under `nvcc` and to nothing otherwise. Every math function is declared `GFX_HD inline`, so the same code can later run inside CUDA kernels without being rewritten.

### `Vec3`

`vec3.h` provides `Vec3`; `+`, `-`, scalar `*` (both operand orders) and unary `-`; and `dot`, `cross`, `length` and `normalize`. `normalize` returns the zero vector for zero-length input instead of producing NaNs. This is a deliberate, documented choice.

### The test harness

There is no third-party test framework. `tests/check.h` provides:

- a `CHECK(cond)` macro that prints the file, line and expression on failure and counts failures;
- an `approx(a, b, eps)` comparison, computed in `double` so that float arguments do not trigger narrowing warnings.

A test program exits 0 only if no check failed. That is all CTest needs, because it treats a zero exit code as a pass.

## 7. Step 4 — `Vec4` and `Mat4`

`vec4.h` provides `Vec4`, `+`, scalar `*` and `make_vec4(Vec3, w)`.

`mat4.h` provides `Mat4`, stored as **four `Vec4` columns, 64 bytes, no padding**, enforced by a `static_assert`. The API:

| Function | Purpose |
|---|---|
| `mat4_identity()` | Identity matrix |
| `Mat4 * Vec4` | Written as `col0*x + col1*y + col2*z + col3*w` |
| `Mat4 * Mat4` | Applies the matrix-vector product to each column of the right operand |
| `mat4_translate`, `mat4_scale` | Translation and non-uniform scale |
| `mat4_rotate(axis, radians)` | Axis-angle (Rodrigues) rotation. It **expects a unit axis** and does not normalize |
| `transform_point`, `transform_direction` | Convenience wrappers that set `w` to 1 or 0 |

Because the layout is column-major and tightly packed, a `Mat4` can be passed to `glUniformMatrix4fv(..., GL_FALSE, ...)` directly. A test copies a translation matrix into a `float[16]` and checks that elements 12, 13 and 14 hold the translation, which is exactly what OpenGL reads. That guards the "upload without a transpose" assumption.

## 8. Step 5 — Camera matrices

### View matrix: `look_at(eye, target, up)`

Builds the world-to-camera transform. The camera looks down **-Z** in view space, with **+X right** and **+Y up**. With `f = normalize(target - eye)`, `s = normalize(cross(f, up))` and `u = cross(s, f)`, the columns are:

```
col0 = (  s.x,  u.x, -f.x, 0 )
col1 = (  s.y,  u.y, -f.y, 0 )
col2 = (  s.z,  u.z, -f.z, 0 )
col3 = ( -dot(s,eye), -dot(u,eye), dot(f,eye), 1 )
```

The rotation part is the transpose of the camera's basis, so it maps world axes into camera axes, and the last column moves the eye to the origin. If the camera looks exactly along `up`, `cross(f, up)` is zero and the matrix collapses. The orbit camera in M2 prevents this by clamping pitch just short of straight up or down.

### Projection: `perspective(fov_y, aspect, near_plane, far_plane, clip_space)`

`fov_y` is the vertical field of view in radians, and `f = 1 / tan(fov_y / 2)`. OpenGL and Vulkan disagree about clip space, so there is one function with a `ClipSpace` parameter instead of two functions:

| | OpenGL (`CLIP_SPACE_OPENGL`) | Vulkan (`CLIP_SPACE_VULKAN`) |
|---|---|---|
| NDC depth | -1 to 1 | 0 to 1 |
| NDC Y | up | down |

OpenGL:

```
col0 = ( f/aspect, 0,  0,                        0 )
col1 = ( 0,        f,  0,                        0 )
col2 = ( 0,        0,  (far+near)/(near-far),   -1 )
col3 = ( 0,        0,  2*far*near/(near-far),    0 )
```

Vulkan:

```
col0 = ( f/aspect, 0,  0,                    0 )
col1 = ( 0,       -f,  0,                    0 )
col2 = ( 0,        0,  far/(near-far),      -1 )
col3 = ( 0,        0,  far*near/(near-far),  0 )
```

The `-1` in `col2.w` copies `-z` into `w`, which makes the GPU's later divide by `w` the perspective divide. Points behind the camera end up with negative `w`.

### Parameter naming

The parameters are named `near_plane` and `far_plane`, not `near` and `far`, because `<windows.h>` defines `near` and `far` as empty macros (a leftover from 16-bit Windows) that would corrupt the function signature. See section 12.

## 9. Step 6 — Version control, presets and warnings

### `.gitignore` and the first commit

Build output never goes into Git, so the ignore file excludes `build/` and similar directories. The first commit captures the toolchain, core library, camera math and tests, and every step after it ends with a commit.

### `CMakePresets.json`

Presets pin the build configuration in the repository instead of in VS Code's state.

| Preset | Build directory | Purpose |
|---|---|---|
| `windows-debug` | `build/windows-debug` | Stepping through code |
| `windows-release` | `build/windows-release` | Honest performance numbers |

- **Two configure presets, not one.** Ninja is a single-configuration generator, so the build type is fixed at configure time and each type needs its own build directory.
- Both inherit from a hidden `windows-base` preset that sets the generator, `cl.exe`, the `compile_commands.json` export, the CUDA architecture (`GFX_CUDA_ARCHITECTURES=120`, added in Step 7) and `"strategy": "external"` for the x64 toolset. The external strategy makes CMake Tools set up the MSVC environment itself.
- A condition limits the presets to Windows hosts.
- Matching build and test presets (`debug`, `release`) bind to the configure presets, and test output is shown on failure.
- Version 3 presets need CMake 3.21. The root `cmake_minimum_required` is 3.20, so a manual configure without presets still works on 3.20.

### Warnings

On MSVC the root file adds `/W4 /permissive- /utf-8`: a high warning level to catch non-portable code early, standards-conforming behavior, and consistent source encoding. Ordering matters. `add_compile_options` and `add_compile_definitions` must appear **before** the subdirectories they should affect, and libraries must be added before the targets that use them.

## 10. Step 7 — Optional backends (closes M0)

### Detection (`cmake/GfxBackends.cmake`)

Each backend has an option, all `ON` by default, where `ON` means "build it if its tools can be found". A missing dependency is a status message, never an error, so the project configures on a machine with none of them.

| Option | Detection | Result variable |
|---|---|---|
| `GFX_ENABLE_OPENGL` | `find_package(OpenGL)` | `GFX_HAVE_OPENGL` |
| `GFX_ENABLE_VULKAN` | `find_package(Vulkan COMPONENTS glslc)` on CMake 3.24+; on older CMake, `Vulkan_FOUND` plus the `Vulkan::glslc` target | `GFX_HAVE_VULKAN` |
| `GFX_ENABLE_CUDA` | `check_language(CUDA)`, then `enable_language(CUDA)` | `GFX_HAVE_CUDA` |

`glslc` is part of what "Vulkan found" means, because the Vulkan backend needs it to compile shaders.

### CUDA specifics

CUDA is a language in CMake rather than a package, which has two consequences:

- `enable_language(CUDA)` has to run at top-level scope, so the file is pulled in with `include()`, not a function or subdirectory.
- The target architectures must be set before it runs. They are controlled by the optional cache variable `GFX_CUDA_ARCHITECTURES`.

If that variable is empty, nothing is set and CMake and `nvcc` pick their own default, which is unlikely to match a current GPU.

| GPU | Value |
|---|---|
| RTX 20 series | `75` |
| RTX 30 series | `86` |
| RTX 40 series | `89` |
| RTX 50 series (Blackwell) | `120` |

A list such as `"75;86;89;120"` builds for several. The presets, `build.bat` and the later `build-debug.bat` and `build-release.bat` all set `120`, so change it everywhere if the GPU is different.

### Shared configuration

Detected backends define `GFX_HAS_OPENGL`, `GFX_HAS_VULKAN` and `GFX_HAS_CUDA` on the `gfx_config` interface target. Configuration ends with a summary such as `Backends: OpenGL=ON Vulkan=OFF CUDA=OFF`. Detection results are cached, so after installing the Vulkan SDK or the CUDA Toolkit, delete the cache or the build directory and configure again.

### Build scripts

`build.bat` is a convenience script for quick iteration. It deletes `build/`, configures with Ninja (without a preset, so no explicit build type) and the CUDA architecture flag, builds, and offers a menu to run `gfxlab.exe`, run `test_math.exe`, or run all tests through CTest. Its output goes to `build/src/app/` and `build/tests/`, unlike the preset builds. `build-debug.bat` and `build-release.bat` were added during M2 as the same idea with an explicit build type, writing to `builds\build-debug\` and `builds\build-release\`.

## 11. Tests and correctness checks

`tests/test_math.cpp` covers:

1. `Vec3` arithmetic, dot, cross handedness, length, and `normalize` including the zero vector.
2. `Mat4`: identity, translate, scale, the right-hand rule for rotation, composition order, memory layout, and trivial copyability.
3. Camera math: the eye maps to the origin, axes map as expected, the near and far planes land on the right NDC depth for both clip-space conventions, the Vulkan Y flip, the field-of-view edge, and negative `w` behind the camera.

Tests are registered with `add_test`, one executable per area, and run with `ctest --preset debug`. Math errors are painful to debug visually, which is why the math was tested before any renderer existed.

## 12. Naming rules

`<windows.h>` defines macros with common names. The project avoids the collisions up front:

- `LOG_LEVEL_ERROR`, not `ERROR` (`wingdi.h` defines `ERROR` as `0`).
- `near_plane` and `far_plane`, not `near` and `far`.
- Enum constants are prefixed (`GFX_KEY_*`, `GFX_MOUSE_*`, `CLIP_SPACE_*`), because Windows also defines names such as `KEY_READ`.
- `NOMINMAX` is defined project-wide, so `min` and `max` are never macros.

## 13. Build and configure sequence

From a clean checkout:

1. `cmake --preset windows-debug` selects Ninja and `cl.exe`, applies `/W4 /permissive- /utf-8` and the Windows definitions, runs backend detection and prints the backend summary.
2. `cmake --build --preset debug` builds `gfx_core`, the tests and the app.
3. `ctest --preset debug` runs the registered tests.
4. `build\windows-debug\src\app\gfxlab.exe` starts the app, which logs which backends were compiled in.

## 14. Known scope and non-goals

M0 is a foundation, not a product. It does not include:

- running the app on any platform other than Windows (only `gfx_core` and the tests are portable);
- a window, input handling or a frame timer (M1);
- any rendering (M2 onward);
- a third-party test framework, logging library or math library.

Known gaps in the code:

- `perspective` divides by `aspect` and by `near - far` without checking. A zero `aspect` or `near == far` gives infinities or NaNs. M1 keeps zero sizes away from the renderer in the platform layer, and an assert macro will be added when the renderer needs it.
- `mat4_rotate` does not normalize its axis.
- The logger is not thread-safe.

## 15. Completion checkpoint

The expected state at the end of M0:

- the project configures and builds from a clean checkout using a preset, with `/W4` enabled;
- `ctest` passes;
- the app starts and logs which backends were detected;
- the configure output reports Vulkan and CUDA honestly when they are absent;
- `gfx_core` and the tests also configure, build and pass on a non-Windows host.