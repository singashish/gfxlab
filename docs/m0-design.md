# M0 design: project foundation

M0 builds everything the rest of the project stands on: the build system, a
small core library (logging and math), the optional-backend detection, and a
unit-test setup. Nothing in M0 opens a window or draws anything.

This document records what was built and why. The overall plan is in
[`plan.txt`](plan.txt), and the window, input and timer work is described in
[`m1-design.md`](m1-design.md).

## 1. Goals and non-goals

**Goals**

- A reproducible build on Windows with MSVC, CMake and Ninja, usable from VS Code
- Dependencies limited to the graphics APIs (OpenGL, Vulkan, CUDA), each optional
- Math and camera conventions fixed once, in one place, before any renderer
  depends on them
- Tests for the math, because math errors are painful to debug visually

**Non-goals**

- Any platform other than Windows (the portable parts stay portable, but only
  Windows is built and tested)
- A third-party test framework, logging library or math library

## 2. Decisions

| Area | Decision | Reason |
|------|----------|--------|
| Language | C++17 (`CMAKE_CXX_EXTENSIONS OFF`) | Modern enough, portable, supported by `nvcc` |
| Compiler | MSVC (Visual Studio 2022 Build Tools, no IDE) | `nvcc` on Windows needs `cl.exe` as its host compiler |
| Generator | Ninja | Fast parallel builds; works with `nvcc` without Visual Studio's CUDA integration |
| Editor | VS Code with CMake Tools | Preferred workflow |
| Dependencies | Only OpenGL, Vulkan SDK, CUDA Toolkit | Learning goal: write the rest ourselves |
| Floating point | `float` everywhere | Matches GPU data and shader code |
| Warnings | `/W4 /permissive- /utf-8` on MSVC | Catch non-portable code early; consistent source encoding |

## 3. Build system

### 3.1 Targets

```
gfx_core      STATIC     src/core        logging (log.cpp); math headers
gfx_platform  STATIC     src/platform    added in M1
gfx_config    INTERFACE  (no sources)    compile definitions for detected backends
gfxlab        EXECUTABLE src/app
test_*        EXECUTABLE tests/          one per test file, registered with CTest
```

Each module has its own `CMakeLists.txt`, and the root file only holds
project-wide settings and the module list. Order in the root file matters
twice: `add_compile_options` must come before the subdirectories it should
affect, and libraries are added before the targets that use them.

**Visibility.** `gfx_core` exposes `src/` as a `PUBLIC` include directory, so
anything linking it can write `#include "core/log.h"` or
`#include "platform/input.h"` with `src/` as the include root. The
`gfx_config` target is an `INTERFACE` library: it has no code and only carries
compile definitions to whatever links it.

### 3.2 Presets

`CMakePresets.json` pins the build configuration in the repository:

| Preset | Build dir | Purpose |
|--------|-----------|---------|
| `windows-debug` | `build/windows-debug` | Stepping through code |
| `windows-release` | `build/windows-release` | Honest performance numbers |

There are two configure presets, not one, because Ninja is a single-config
generator: the build type is fixed when you configure, so each type needs its
own build directory. Both inherit from a hidden `windows-base` preset that sets
the generator, `cl.exe`, `compile_commands.json` export, and
`"strategy": "external"` for the architecture, which makes CMake Tools set up
the MSVC environment automatically. The presets are guarded by a condition so
they only appear on Windows. Matching build and test presets (`debug`,
`release`) bind to them.

### 3.3 Optional backends (`cmake/GfxBackends.cmake`)

Each backend has an option, all `ON` by default, where `ON` means "build it if
its tools can be found". A missing dependency is a status message, never an
error, so the project configures on a machine with none of them.

| Option | Detection | Result variable |
|--------|-----------|-----------------|
| `GFX_ENABLE_OPENGL` | `find_package(OpenGL)` | `GFX_HAVE_OPENGL` |
| `GFX_ENABLE_VULKAN` | `find_package(Vulkan COMPONENTS glslc)` on CMake 3.24+; on older CMake, `Vulkan_FOUND` and the `Vulkan::glslc` target | `GFX_HAVE_VULKAN` |
| `GFX_ENABLE_CUDA` | `check_language(CUDA)`, then `enable_language(CUDA)` | `GFX_HAVE_CUDA` |

`glslc` is part of what "Vulkan found" means, because the Vulkan backend needs
it to compile shaders.

CUDA is a language in CMake rather than a package, which has two consequences.
`enable_language(CUDA)` has to run at the top-level scope (so the file is
pulled in with `include()`, not a function or subdirectory), and the target
architectures must be set before it runs. They are controlled by an optional
cache variable:

```
cmake --preset windows-debug -DGFX_CUDA_ARCHITECTURES=120
```

If `GFX_CUDA_ARCHITECTURES` is empty, nothing is set and CMake and `nvcc` pick
their own default, which is unlikely to match the architecture of a current
GPU. Use
`120` for RTX 50 series (Blackwell), `89` for RTX 40, `86` for RTX 30, `75`
for RTX 20, or a list such as `"75;86;89;120"` to build for several.
`build.bat` currently passes `120` explicitly. The presets do not set it.

Detected backends define `GFX_HAS_OPENGL`, `GFX_HAS_VULKAN` and `GFX_HAS_CUDA`
on `gfx_config`, and configuration ends with a summary such as
`Backends: OpenGL=ON Vulkan=OFF CUDA=OFF`. Detection results are cached, so
after installing the Vulkan SDK or CUDA Toolkit, delete the cache or the build
directory and configure again.

### 3.4 `build.bat`

A convenience script for quick iteration. It deletes `build/`, configures with
Ninja (no preset, so no explicit build type) and the CUDA architecture flag,
builds, and then offers a menu to run `gfxlab.exe`, run `test_math.exe`, or run
all tests through CTest. Because it does not use presets, its output goes to
`build/src/app/` and `build/tests/`, unlike the preset builds.

## 4. Core library (`src/core`)

### 4.1 Logging (`log.h`, `log.cpp`)

```cpp
enum class LogLevel { LOG_LEVEL_TRACE, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO,
                      LOG_LEVEL_WARN,  LOG_LEVEL_ERROR };
void log_set_level(LogLevel min_level);
void log_write(LogLevel level, const char* file, int line, const char* fmt, ...);
```

Application code uses the macros `LOG_TRACE`, `LOG_DEBUG`, `LOG_INFO`,
`LOG_WARN` and `LOG_ERROR`, which call `log_write` with `__FILE__` and
`__LINE__`.

- **Macros, not functions**, so the call site's file and line are captured. A
  plain function would always report its own location.
- **Variadic `__VA_ARGS__` forwarding** keeps calls with no format arguments
  legal. A `(fmt, ...)` macro breaks on the trailing comma.
- **printf-style** formatting: no allocation, no streams, and easy to read when
  logging numbers.
- Output goes to `stderr` as `[LEVEL] file.cpp:line  message`, with the
  directory stripped from the path (both `/` and `\` are handled). The default
  minimum level is `INFO`.
- Not thread-safe. This is acceptable while everything runs on one thread, and
  needs revisiting if a second thread ever logs.

### 4.2 `common.h`

Defines `GFX_HD`, which expands to `__host__ __device__` under `nvcc`
(`__CUDACC__`) and to nothing otherwise. Every math function is declared
`GFX_HD inline`, so the same code can later run inside CUDA kernels without
being rewritten.

### 4.3 Math conventions

These are fixed for the whole project:

- **Right-handed**, **Z up** (like Blender): `X cross Y = Z`
- **Column vectors**, so a transform reads `M * v`, and chained transforms read
  right to left (in `T * R * S`, scale applies first)
- **Column-major storage**: `Mat4` is four `Vec4` columns, 64 bytes, no padding
  (enforced by a `static_assert`), so it uploads to the GPU without a transpose
- **`w` distinguishes points from directions**: `w = 1` is a point (translation
  applies), `w = 0` is a direction (translation is ignored)
- Plain structs with public members and no constructors, so they stay
  trivially copyable and can be copied into GPU buffers

### 4.4 Types and functions

| Header | Contents |
|--------|----------|
| `vec3.h` | `Vec3`; `+`, `-`, scalar `*` (both orders), unary `-`, `dot`, `cross`, `length`, `normalize` |
| `vec4.h` | `Vec4`; `+`, scalar `*`, `make_vec4(Vec3, w)` |
| `mat4.h` | `Mat4`; `mat4_identity`, `Mat4 * Vec4`, `Mat4 * Mat4`, `mat4_translate`, `mat4_scale`, `mat4_rotate(axis, radians)`, `transform_point`, `transform_direction` |
| `camera_math.h` | `ClipSpace`, `look_at`, `perspective` |

Design notes:

- `normalize` returns the zero vector for zero-length input rather than
  producing NaNs. This is a deliberate, documented choice.
- Matrix times vector is written as a combination of columns,
  `col0*x + col1*y + col2*z + col3*w`, and matrix times matrix applies that to
  each column of the right operand.
- `mat4_rotate` uses the axis-angle (Rodrigues) form and **expects a unit
  axis**. It does not normalize.

### 4.5 Camera math

**View matrix, `look_at(eye, target, up)`.** Builds the world-to-camera
transform. The camera looks down **-Z** in view space, with **+X right** and
**+Y up**. With `f = normalize(target - eye)`, `s = normalize(cross(f, up))`
and `u = cross(s, f)`, the columns are:

```
col0 = (  s.x,  u.x, -f.x, 0 )
col1 = (  s.y,  u.y, -f.y, 0 )
col2 = (  s.z,  u.z, -f.z, 0 )
col3 = ( -dot(s,eye), -dot(u,eye), dot(f,eye), 1 )
```

The rotation part is the transpose of the camera's basis (it maps world axes
into camera axes), and the last column moves the eye to the origin. If the
camera looks exactly along `up`, `cross(f, up)` is zero and the matrix
collapses. The orbit camera will prevent this by clamping pitch just short of
straight up or down.

**Projection, `perspective(fov_y, aspect, near_plane, far_plane, clip_space)`.**
`fov_y` is the vertical field of view in radians, and `f = 1 / tan(fov_y / 2)`.
The two APIs disagree about clip space, so there is one function with a
parameter, not two:

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

The `-1` in `col2.w` copies `-z` into `w`, which makes the GPU's later divide
by `w` the perspective divide. Points behind the camera end up with negative
`w`.

**Known gaps.** `perspective` divides by `aspect` and by `near - far` without
checking. A zero `aspect` (a minimized window reports 0x0) or `near == far`
gives infinities or NaNs. M1 keeps zero sizes away from the renderer in the
platform layer, and an assert macro will be added when the renderer needs it.

## 5. Testing

- **`tests/check.h`** provides a `CHECK(cond)` macro that prints the file, line
  and expression on failure and counts failures, and an `approx(a, b, eps)`
  comparison. A test program exits 0 only if no check failed. This is enough
  for CTest, which treats a zero exit code as a pass.
- **`tests/test_math.cpp`** covers `Vec3` (arithmetic, dot, cross handedness,
  length, normalize including the zero vector), `Mat4` (identity, translate,
  scale, the right-hand rule, composition order, memory layout, trivial
  copyability) and camera math (the eye maps to the origin, axes map as
  expected, the near and far planes land on the right NDC depth for both
  conventions, the Y flip, the field-of-view edge, and negative `w` behind the
  camera).
- Tests are registered with `add_test`, one executable per area, and run with
  `ctest --preset debug`.

The memory-layout test copies a translation matrix into a `float[16]` and
checks that elements 12, 13 and 14 hold the translation. That is exactly what
OpenGL reads, so it guards the "upload without a transpose" assumption.

## 6. Naming rules

`<windows.h>` defines macros with common names. The project avoids the
collisions up front:

- `LOG_LEVEL_ERROR`, not `ERROR` (`wingdi.h` defines `ERROR` as `0`)
- `near_plane` and `far_plane`, not `near` and `far` (empty macros left over
  from 16-bit Windows)
- Enum constants are prefixed (`GFX_KEY_*`, `GFX_MOUSE_*`, `CLIP_SPACE_*`)
  because Windows also defines names such as `KEY_READ`

## 7. Definition of done for M0

- Configures and builds from a clean checkout using a preset, with `/W4` on
- `ctest` passes
- The app starts and logs which backends were detected
- The configure output reports the backends honestly when Vulkan or CUDA are
  absent
