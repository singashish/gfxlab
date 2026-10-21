# M5 — The CUDA Software Rasterizer

> **Status:** Complete (Steps 1–7)  
> **Scope:** Render the same scene as the OpenGL and Vulkan backends through the same `IRenderer` interface, but without the graphics pipeline: a rasterizer written as CUDA kernels draws into a buffer on the GPU, and the result is shown on screen through CUDA/OpenGL interop.

## 1. Purpose

M2 to M4 used the GPU's fixed-function hardware: vertex fetch, triangle setup, rasterization, depth testing and blending are all done by the driver and the silicon. M5 replaces all of it with code written in this repository, running as ordinary compute kernels. The point is to learn what the hardware does, and why it is built the way it is:

- how a triangle is turned into pixels (edge functions, the top-left fill rule, fixed-point snapping);
- how a depth test works when thousands of threads write the same pixel (atomics);
- how to separate "which triangle is visible" from "what colour is it" (a visibility buffer and deferred shading);
- why a naive "one thread per triangle" mapping is fast for small triangles and terrible for big ones, and how to fix it;
- how a CUDA program hands a finished image to OpenGL without copying it through the CPU.

Backend selection is again a command-line choice:

```
gfxlab.exe --backend=cuda                 CUDA software rasterizer, no model (demo triangles are not drawn; see Section 17)
gfxlab.exe --backend=cuda suzanne_s.obj   CUDA with a model
```

The CUDA backend needs both the CUDA Toolkit and OpenGL at build time: OpenGL is used only to own the window's surface and to present the finished picture. All drawing is done by CUDA.

## 2. Architecture and what M5 adds

```
gfxlab        main.cpp: --backend=opengl|vulkan|cuda
gfx_renderer  IRenderer, FrameData, create_renderer() (one new factory case)
gfx_cuda      CudaRenderer and the kernels below
gfx_gl        GLContext, plus the new GLPresenter (texture + fullscreen blit)
gfx_core      raster_math.h (shared by the kernels and the tests)
```

New files:

| Area | Files |
|---|---|
| Common helpers | `src/renderer/cuda/cuda_api.h` (`cuda_ok`) |
| Device diagnostics | `cuda_info.*`, `cuda_info_main.cpp`, `cuda_smoke.*` (the `cuda_info` executable) |
| CUDA/OpenGL interop | `cuda_gl_image.h`, `cuda_gl_image.cpp` |
| Presenting | `src/renderer/opengl/gl_presenter.h`, `gl_presenter.cpp` |
| Device meshes | `cuda_mesh.h`, `cuda_mesh.cpp` |
| Kernels | `cuda_raster.h`, `cuda_raster.cu` |
| Profiling | `cuda_timer.h`, `cuda_timer.cpp` |
| The renderer | `cuda_renderer.h`, `cuda_renderer.cpp` |
| Shared maths | `src/core/raster_math.h`, `tests/test_raster.cpp` |

`CudaRenderer` owns its parts in dependency order, so C++ destroys them in reverse:

```cpp
GLContext      m_context;     // the window's OpenGL context (destroyed last)
GLPresenter    m_presenter;   // texture + shader that shows the picture
CudaGLImage    m_image;       // the texture registered with CUDA
CudaMeshStore  m_meshes;      // meshes copied to GPU memory
CudaStageTimer m_stage_timer; // per-stage GPU timings (optional)

void*          m_color;       // width * height * uchar4
void*          m_visibility;  // width * height * uint64
ProjectedVertex* m_scratch;   // projected vertices of all draw items
```

The frame data flows like this:

```
FrameData + draw list
      |
      v
 clear colour, clear visibility          (CUDA)
      |
      v
 per draw item: transform vertices       (CUDA, one thread per vertex)
                rasterize triangles      (CUDA, atomicMin into the visibility buffer)
      |
      v
 per draw item: shade visible pixels     (CUDA, one thread per pixel)
      |
      v
 grid                                    (CUDA, one thread per pixel, alpha blended)
      |
      v
 colour buffer --cudaMemcpy2DToArray--> OpenGL texture --fullscreen triangle--> window
```

Only the last arrow involves OpenGL. Nothing is copied through the CPU.

## 3. Step 1 — Toolchain: the CUDA Toolkit and a smoke test

The CUDA backend needs `nvcc` and the CUDA Toolkit. For an RTX 50 series card the toolkit must be 12.8 or newer; this project was built with 13.4 and `GFX_CUDA_ARCHITECTURES=120`.

`cmake/GfxBackends.cmake` enables the backend in three steps: `check_language(CUDA)` (is `nvcc` there?), then `find_package(CUDAToolkit)` (are the libraries there?), then `enable_language(CUDA)`. Only if both succeed does it set `GFX_HAVE_CUDA` and define `GFX_HAS_CUDA`. The language is enabled after the checks so that a machine without CUDA still configures normally.

Three build-system details mattered:

- **Compile options are per language.** `/W4` is an MSVC flag that `nvcc` rejects, so the root `CMakeLists.txt` uses generator expressions: `$<$<COMPILE_LANGUAGE:CXX>:/W4>` for C++ and `-Xcompiler=/W4,/utf-8` for the host part of CUDA files.
- **`add_subdirectory` order matters.** `src/renderer/cuda` is added before `src/renderer`, and the link to `gfx_cuda` lives in `src/renderer/CMakeLists.txt` (see Section 14).
- **The static runtime.** `gfx_cuda` links `CUDA::cudart_static`, so the executable has no dependency on `cudart64_13.dll`.

`cuda_ok(cudaError_t, what)` is the common error check: it logs the error name and string and returns `false`. Kernel launches have two error channels, so `check_launch()` calls `cudaGetLastError()` after each launch (bad configuration), while asynchronous failures during execution are caught by `cudaDeviceSynchronize()`. The synchronize is done **once per frame, in Debug builds only** (`#ifndef NDEBUG`), instead of after every kernel.

The `cuda_info` executable prints the runtime and driver versions and the device properties (RTX 5080, compute capability 12.0, 84 SMs, 15.89 GiB), then runs a trivial kernel. If `cuda_info` passes, the toolchain works.

## 4. Step 2 — Showing a CUDA image through OpenGL

CUDA has no window. To get its output on screen, `CudaRenderer` creates a normal OpenGL context (`GLContext`, unchanged from M2), and uses OpenGL only as a display path.

### Choosing the device

This machine has two GPUs (an RTX 5080 and an Intel iGPU). `cudaGLGetDevices` returns the CUDA devices that the *current* OpenGL context runs on, so it is called after the context exists and is current, and `cudaSetDevice` selects the first one. If OpenGL ran on the Intel GPU there would be no CUDA device and `init()` fails with a clear message.

### The presenter

`GLPresenter` (in `gfx_gl`) owns an `RGBA8` texture with `NEAREST` filtering, a shader program, and an empty VAO. It draws one fullscreen triangle generated from `gl_VertexID`. The fragment shader flips `v`: the CUDA buffer stores row 0 at the top of the image, while row 0 of an OpenGL texture is at `v = 0`, the bottom.

### The interop path

`CudaGLImage` wraps the interop calls:

| Call | When | Purpose |
|---|---|---|
| `cudaGraphicsGLRegisterImage` (write-discard) | when the texture is created or resized | tells CUDA about the GL texture |
| `cudaGraphicsMapResources` | every frame | gets the texture for CUDA's use |
| `cudaGraphicsSubResourceGetMappedArray` | every frame | the `cudaArray` behind the texture |
| `cudaMemcpy2DToArray` (device to device) | every frame | copies the colour buffer into the texture |
| `cudaGraphicsUnmapResources` | every frame | hands the texture back to OpenGL |
| `cudaGraphicsUnregisterResource` | before the texture is replaced, and at shutdown | |

The colour buffer is a plain linear `uchar4` array in CUDA memory (`m_color`). Kernels write to linear memory, which is simple and fast, and a single copy puts the result into the texture. The copy is GPU to GPU, a few megabytes per frame.

One practical problem: `cuda_gl_interop.h` includes `<GL/gl.h>`, but this project's own `gl_api.h` deliberately refuses to be included next to it (`#error`), because it declares the same functions through its own loader. So all interop code lives in `cuda_gl_image.cpp`, which includes `<windows.h>` and then `<cuda_gl_interop.h>` and never `gl_api.h`. Only plain `unsigned int` texture ids cross that boundary.

### Resizing

`render()` compares the frame size with the buffers' size. On a change, `resize_targets()` unregisters the texture, resizes it, registers it again, and reallocates the colour and visibility buffers. `m_width` and `m_height` are zeroed at the top, so a failed resize can never leave stale sizes that look valid. Zero-size frames (a minimized window) return immediately.

Vsync (`V`) is the context's swap interval, so it works exactly like OpenGL.

## 5. Step 3 — Device meshes and the vertex stage

`create_mesh()` returns an id for a mesh already copied to GPU memory. `CudaMeshStore` keeps a `DeviceMesh`:

```cpp
struct DeviceMesh {
    Vec3* positions;          // device pointers
    Vec3* normals;
    std::uint32_t* indices;
    std::uint32_t vertex_count, index_count;
};
```

Ids are the slot index plus one, so 0 stays "no mesh". The CPU validates the mesh before copying (index range, triangle-list size) so the kernels do not need to defend against bad indices.

`GFX_HD` made this step easy: the `Vec3`, `Vec4`, `Mat4` types and their operators from `gfx_core` expand to `__host__ __device__` under `nvcc`, so the kernels use the same maths code as the CPU, with no copy.

### The vertex kernel

`transform_vertices_kernel` runs one thread per vertex and writes a `ProjectedVertex`:

```cpp
struct ProjectedVertex {
    float4 screen;       // x, y in pixels, depth in [0,1], clip-space w
    float4 view_pos;     // view-space position, for lighting
    float4 view_normal;  // view-space normal, for lighting
};
```

For each vertex:

1. `clip = MVP * position`.
2. If `clip.w <= 0` (the vertex is behind the camera) or not finite, mark it invalid.
3. Divide by `w`: `ndc = clip.xyz / clip.w`.
4. Viewport transform, with row 0 at the top: `x = (ndc.x*0.5 + 0.5) * width`, `y = (1 - (ndc.y*0.5 + 0.5)) * height`, `depth01 = ndc.z*0.5 + 0.5`.

The projection is built with OpenGL clip space, the same as `GLRenderer`, so depth is already in the range that OpenGL's `glDepthRange(0, 1)` would give.

The view-space position and normal are computed here once per vertex, and the shading stage interpolates them. The normal uses `frame.view * item.normal_matrix`, which is the normal matrix from M3.

All items of a frame share one scratch buffer (`m_scratch`, grown when needed and never shrunk). Each item gets its own offset, so the shading stage can still find an item's vertices after the next item has been rasterized.

## 6. Step 4 — The rasterizer

### Coordinates and fixed point

Screen positions are snapped to **1/16 of a pixel** (`snap_subpixel` in `core/raster_math.h`: `floor(v*16 + 0.5)`), and the edge tests are done with integers. Floating-point edge tests can give a different answer for the same point depending on which triangle asks, which opens cracks between neighbouring triangles. Integers give identical answers for a shared edge from both sides.

Limits: a vertex with `|x|` or `|y|` greater than 16384 pixels is rejected, so 16384 × 16 fits comfortably in an `int`, and the edge function products need 64-bit (`long long`).

### Edge functions

For an edge from `a` to `b` and a point `p`:

```
E(a, b, p) = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x)
```

`E` is positive on one side of the edge, negative on the other and zero on it. A point is inside a triangle if the three edge functions agree in sign. The triangle's own signed area is `E(v0, v1, v2)`.

`setup_raster_triangle` normalizes the winding: if the signed area is negative it swaps two vertices, so the area is positive and "inside" always means "all three edge values are non-negative". It also keeps the **original** signed area, because the shading stage needs it to decide whether the triangle faces the camera, and the vertex `order` so barycentrics can be mapped back.

Pixel samples are taken at pixel centres: `(x + 0.5, y + 0.5)`, which is `x*16 + 8` in 1/16 units.

### The top-left rule

If a sample lies *exactly* on an edge shared by two triangles, both triangles would draw it (double blending, a bright seam) or neither would (a hole). The standard solution is the top-left rule: a sample on an edge belongs to the triangle only if the edge is a *top* or *left* edge. With a y-down screen and positive-area winding, that is:

```cpp
is_top_left = (dy < 0) || (dy == 0 && dx > 0);     // for the edge a -> b
inside      = E + (is_top_left ? 0 : -1) >= 0;
```

The `-1` turns "E ≥ 0" into "E > 0" for edges that are not top-left, using integers.

This logic is **shared**: `raster_math.h` is plain C++ (`GFX_HD`), compiled by both `nvcc` and the host compiler. `tests/test_raster.cpp` runs the same functions on the CPU and checks that a quad split into two triangles is covered **exactly once** per sample (both diagonals, both windings), that a horizontal shared edge has no double or missing coverage, and that a degenerate triangle covers nothing.

### Depth and the visibility buffer

The rasterizer does not compute colour. It only decides which triangle is in front at each pixel, and stores the answer in a **visibility buffer** with one 64-bit value per pixel:

```
bits 63..32   depth, as the bit pattern of a float in [0, 1]
bits 31..0    triangle id + 1   (0 is never a valid id)
empty pixel   0xFFFFFFFFFFFFFFFF
```

Two facts make this work. First, for non-negative IEEE floats, the integer order of the bit patterns equals the numeric order, so comparing the packed 64-bit integers compares depths. Second, `atomicMin` on a 64-bit integer is a single atomic operation, so the depth test and the write happen together with no race, even when thousands of threads hit the same pixel. When two depths are exactly equal, the lower triangle id wins, which is deterministic.

The visibility buffer is cleared to all ones (`clear_visibility_kernel`), so any real value is smaller.

Depth is interpolated linearly in screen space (`depth01` is already linear in screen space after the perspective divide), using the barycentric weights `b_i = E_i / area`. A depth outside `[0, 1]` means the sample is beyond the near or far plane, and the sample is **discarded** (not clamped), like OpenGL's clipping.

### One thread per triangle (Step 4), and why it was not enough

The first version launched one thread per triangle. Each thread computes the triangle's bounding box and loops over every pixel in it. This is excellent for small triangles, but the frame takes as long as its **slowest thread**. A close-up of a cube has 12 triangles and each can cover most of the screen: one thread walked hundreds of thousands of pixels alone, and the GPU's thousands of other lanes sat idle. That is Section 9.

## 7. Step 5 — Deferred shading

After all items are rasterized, the visibility buffer holds, for every pixel, the nearest triangle among **all** items. Shading is a separate pass: `shade_item_kernel` runs one thread per pixel, once per draw item.

For each pixel it:

1. reads the packed value; if empty, or if the id does not belong to this item (`triangle_base` and the item's triangle count tell), it returns;
2. rebuilds the triangle's setup from the scratch vertices (the same `setup_triangle` the rasterizer used, so both agree);
3. recomputes the three edge values at the pixel centre and the barycentrics `b_i`;
4. makes them **perspective-correct**: `λ_i = (b_i / w_i) / Σ(b_j / w_j)`. Screen-space barycentrics are linear in screen space, but view-space attributes are not; dividing by each vertex's clip-space `w` corrects for it;
5. interpolates the view-space position and normal with `λ`;
6. decides facing and lights the pixel.

Shading only the visible pixel is the benefit of a visibility buffer: each pixel is lit once, however many triangles overlapped it. The cost is that the shading pass must re-derive the barycentrics, which the rasterizer threw away.

### Facing

The screen is y-down, which mirrors the picture compared with OpenGL's y-up window space. A triangle that is counter-clockwise (front-facing) in OpenGL therefore has a **negative original signed area** here:

```cpp
front_facing = (s.signed_area < 0);
if (!front_facing) n = -n;       // two-sided lighting, like the GL shader
```

### The lighting

Ported line by line from `gl_mesh.cpp`, so the picture matches OpenGL and Vulkan:

- key direction `normalize(-0.5, 0.6, 0.7)` in view space (a headlight-style key light that follows the camera);
- hemisphere ambient: `mix(low, high, dot(n, up_view)*0.5 + 0.5)` with `low = (0.10, 0.10, 0.12)`, `high = (0.30, 0.32, 0.36)`, and `up_view` the world +Z axis rotated into view space;
- diffuse `max(dot(n, key), 0)` scaled by 0.75;
- Blinn–Phong specular, exponent 32, scaled by 0.25, only when the surface faces the light;
- base colour `(0.72, 0.74, 0.78)`; the result is clamped to 1 and converted to bytes.

### Debug view

`kDebugTriangleIds` (a `constexpr bool` in `cuda_renderer.cpp`, `false`) replaces shading with `resolve_ids_kernel`, which colours every pixel by a hash of its triangle id. It makes the rasterizer's output visible without any lighting, which is how the top-left rule and the winding were checked on screen.

## 8. Step 6 — The grid as a compute kernel

The grid is the same infinite, ray-cast floor as in OpenGL and Vulkan, but here it is a kernel with one thread per pixel (`draw_grid_kernel`), run after shading. Per pixel:

1. Build the ray through the pixel centre from the camera basis (`forward`, and `right`/`up` already scaled by `tan(fov/2)` and the aspect ratio, as `GLRenderer` does) and intersect it with the plane `z = 0`: `t = -eye.z / ray.z`. No hit, no work.
2. Estimate how big one pixel is on the floor. OpenGL has `fwidth()` for this; a kernel has no neighbouring-pixel derivatives, so the same ray is cast through the pixel to the right `(x+1.5, y+0.5)` and the pixel below `(x+0.5, y+1.5)`, and the world-space differences are used. If a neighbour's ray misses the floor (near the horizon) the footprint falls back to a huge value, so the lines fade out instead of aliasing.
3. Evaluate the minor lines (spacing 1), major lines (spacing 10) and the two axes, with analytic anti-aliasing from the footprint, and combine the layers: grey, lighter grey, red X axis, green Y axis, with the same alphas as the GL shader.
4. Fade with distance from the camera: `1 - smoothstep(0.4 * fade_end, fade_end, distance)`.
5. Depth test against the mesh. The visibility buffer already holds the nearest depth, so the grid compares its own depth with it using `GL_LESS` semantics: the grid is drawn only where it is nearer. Writing the grid's depth is not needed because nothing is drawn after it.
6. Blend over the colour buffer with `SRC_ALPHA, ONE_MINUS_SRC_ALPHA`.

### Depth bias for coplanar geometry

A model's base face lies in the same plane as the grid, so their depths differ only by rounding. The grid's depth is computed from the point `eye + (t * 1.001) * ray`, 0.1% farther along the ray, while its colour and fade still use the true hit point. This is the same fix as in the OpenGL and Vulkan grids.

### Why the grid is not a rasterized quad

It could have been two huge triangles through the rasterizer. A ray cast per pixel has no clipping problem (the plane is infinite and crosses the camera), costs a fixed amount per pixel, and matches the other two backends exactly.

## 9. Step 7 — Measuring, then fixing the large-triangle problem

After Step 6 (the grid) the cube ran at around **80 fps**, while Suzanne, with 15,744 triangles, ran at about 480 fps. That is backwards: the cube is nearly free for every other renderer. Zooming in made the cube *slower*, which pointed at the problem: work per frame depended on triangle **size**, not triangle count.

### Measure first

`CudaStageTimer` puts `cudaEvent_t` markers between the stages and logs the average GPU time of each once per second:

```
CUDA GPU ms: clear 0.03 geometry+raster ... shade ... grid ... total ...
```

Events are recorded in the same stream as the kernels, so they measure GPU time, not CPU launch time. The timer is controlled by `kProfileStages` in `cuda_renderer.cpp`. It waits on the last event every frame, which forces a CPU/GPU sync, so it should be `false` for normal running and fps measurement.

### The cause

With one thread per triangle, the cost of the rasterization kernel is the pixel count of the **largest** triangle, because every other thread finishes early and waits. A cube close-up has a few huge triangles.

### The fix: two tiers

Triangles are classified by the area of their pixel bounding box:

| Tier | Bounding box | Kernel | Mapping |
|---|---|---|---|
| Small | ≤ 1024 pixels | `rasterize_triangles_kernel` | one thread per triangle |
| Large | > 1024 pixels | `rasterize_large_triangles_kernel` | one block row per slice: `gridDim = (triangle_count, 32)`, 128 threads per block |

In the large kernel, block `(t, s)` handles triangle `t` and rows `s, s+32, s+64, …` of its bounding box, with its 128 threads striding across the columns of each row. A full-screen triangle is therefore shared by 32 × 128 = 4096 threads instead of one.

Both kernels use the same helper functions (`setup_triangle`, `triangle_pixel_bounds`, `rasterize_pixel`) and the same area calculation, so each triangle is handled by exactly one of them. Blocks of the other tier return at once.

### Results

Measured with `kProfileStages = true`, so these include the per-frame sync:

| Scene | Before | After |
|---|---|---|
| Cube, close-up | about 80 fps | about 2800 to 3000 fps |
| Cube, zoomed out | not recorded | about 2200 to 2500 fps |
| `suzanne_s.obj` | about 480 fps | about 900 fps |

Suzanne changes by only 20 to 30 fps between near and far, which is expected: her triangles are small, so the cost depends mostly on the triangle count. The cube is slower when zoomed out because the fixed per-frame costs (grid, copy, present) become the bigger part of a very short frame.

### What is still wasteful

The large kernel is launched with one block per triangle per slice **whether or not** the triangle is large. For Suzanne that is about 500,000 blocks (15,744 × 32), nearly all of which return immediately. This is cheap per block but not free, and is one reason Suzanne is slower than the cube. The fix is *compaction*: the small kernel appends the ids of large triangles to a list with an atomic counter, and the large kernel is launched only for that list. It is planned for M6.

## 10. Per-frame sequence (CUDA)

1. Input and editing in `main.cpp`, as in M3. `V` toggles vsync.
2. `draw_frame()` fills `FrameData` and calls `render()`.
3. `render()` returns if the frame has no area; if the size changed, `resize_targets()` runs.
4. Clear the colour buffer to the background `(26, 28, 33)` and the visibility buffer to all ones.
5. Build the projection (OpenGL clip space), validate the draw list (index counts, id range), and make sure the scratch buffer is large enough for all items.
6. For each draw item: transform its vertices into its slice of the scratch buffer, then rasterize its triangles (small kernel, then large kernel).
7. For each draw item: shade the pixels whose visible triangle belongs to it.
8. Draw the grid.
9. Debug builds: `cudaDeviceSynchronize()` and check for kernel errors.
10. Copy the colour buffer into the registered texture (map, `cudaMemcpy2DToArray`, unmap).
11. Draw the fullscreen triangle and swap buffers.

All CUDA work is issued on the default stream, so the kernels run in order and the copy sees their results without extra synchronization.

## 11. Memory

| Buffer | Size | Lifetime |
|---|---|---|
| Colour | `width * height * 4` bytes | reallocated on resize |
| Visibility | `width * height * 8` bytes | reallocated on resize |
| Scratch (projected vertices) | 48 bytes × the total vertex count of the frame | grows when needed, never shrinks |
| Meshes | positions, normals, indices | until the renderer is destroyed |
| GL texture | `width * height * 4` bytes | reallocated on resize |

At 1280×720 the colour buffer is 3.5 MB and the visibility buffer 7 MB. The scratch buffer for `suzanne_s.obj` (8,159 vertices) is about 400 KB.

## 12. Differences between the three backends

| Concern | OpenGL / Vulkan | CUDA |
|---|---|---|
| Rasterization | Fixed-function hardware | `rasterize_*` kernels, edge functions in integers |
| Depth test | Depth buffer and ROP hardware | 64-bit `atomicMin` on a packed depth + id value |
| Shading | Fragment shader, run for every fragment that passes the depth test (or early-Z) | Deferred: one thread per pixel, only for the winning triangle |
| Interpolation | Done by the hardware, perspective-correct | `shade_item_kernel` computes barycentrics itself |
| Blending | ROP hardware | Done by hand in the grid kernel |
| Clipping | Near and far planes clip triangles | No near clipping: triangles with any vertex behind the camera are dropped (Section 17) |
| Anti-aliasing | Optional multisampling | None; only the grid is analytically anti-aliased |
| Presenting | `SwapBuffers` / swapchain | Copy into a GL texture, then `SwapBuffers` |
| Synchronization | Driver / explicit | One stream; `cudaDeviceSynchronize` once per frame in Debug only |

## 13. Verification

- `tests/test_raster.cpp` (the eighth CTest program) tests the shared rasterization maths on the CPU. It does not need a GPU.
- `cuda_info` checks the toolchain and the device.
- The side-by-side comparison with `--backend=opengl` is the correctness test for the kernels: same model, same camera, same picture.
- `kDebugTriangleIds` shows the rasterizer's own output.
- Debug builds synchronize and check for errors once per frame, so an out-of-bounds access or a bad launch shows up as a logged `cuda_ok` failure.
- The manual checklist in Section 15.

`ctest` shows 8 passing tests.

## 14. Problems found during M5

| Problem | Cause and fix |
|---|---|
| `gfxlab.exe` failed to link with unresolved `CudaRenderer` symbols | The `if(TARGET gfx_cuda) target_link_libraries(gfx_renderer ...)` block was in the root `CMakeLists.txt` before `add_subdirectory(src/renderer/cuda)`, so the condition was false and nothing warned. The block moved into `src/renderer/CMakeLists.txt`, after the subdirectory exists |
| `/W4` rejected by `nvcc` | `add_compile_options` applies to every language. Replaced by per-language generator expressions |
| `gl_api.h` and `<cuda_gl_interop.h>` clash | Both declare OpenGL; `gl_api.h` `#error`s on `<GL/gl.h>`. Interop code lives in one file that never includes `gl_api.h` |
| Link to `gfx_gl` removed by accident | `gfx_cuda` linked `gfx_gl` only through `gfx_renderer`, which worked by luck. Restored `target_link_libraries(gfx_cuda PUBLIC gfx_gl)` |
| Stale size after a failed resize | `m_width` and `m_height` kept their old values while the buffers were gone. They are zeroed first in `resize_targets()` |
| Depth clamped instead of discarded | A sample beyond the near or far plane was clamped into `[0, 1]` and drawn. Now `continue` |
| `cudaDeviceProp::clockRate` missing | Removed in CUDA 13; the diagnostic no longer prints it |
| Cube at 80 fps | Work per frame depended on the size of the largest triangle. Section 9 |
| `kProfileStages = true` committed | The timer syncs every frame. Documented, and should be `false` when measuring fps |

## 15. Manual checklist

Run all of these with `--backend=cuda`, in Debug and then Release.

1. The log shows `CUDA/OpenGL interop device: NVIDIA GeForce RTX 5080` (not the Intel GPU) and `CUDA renderer initialized`.
2. `gfxlab.exe --backend=cuda suzanne_s.obj`: Suzanne looks the same as with `--backend=opengl` from the same camera: key light up-left, bluish ambient from above, a small highlight.
3. `suzanne_f.obj` stays faceted, `suzanne_s.obj` smooth. `cube.obj`: one face brighter than the others.
4. The grid matches OpenGL: red X axis, green Y axis, minor and major lines, fade with distance, visible from below the floor, hidden behind the model, and no dotted pattern where the model's base meets the floor.
5. Orbit, pan and zoom are smooth. Zooming a cube until its faces fill the screen keeps the frame rate high (no drop to double digits).
6. **G**, **R**, **S** with X/Y/Z constraints, confirm and cancel work, with correct lighting under non-uniform scale and no flicker or missing triangles while editing.
7. Drag every edge and corner of the window: the picture re-renders live, never stretched or black. Maximize and restore.
8. Minimize and restore: no errors, and the first frame is correct.
9. Press **V** repeatedly, also just before and after a resize.
10. With `kProfileStages = true`, a `CUDA GPU ms:` line appears once per second.
11. No `CUDA error` line appears at any point (Debug build).
12. `ctest` passes 8 tests.

## 16. Controls and flags

| Item | Where | Default |
|---|---|---|
| `--backend=cuda` | command line | OpenGL is the default backend |
| `kProfileStages` | `cuda_renderer.cpp` | `true` while developing; set `false` to measure fps |
| `kDebugTriangleIds` | `cuda_renderer.cpp` | `false` |
| `kLargeTriangleArea` (1024), `kLargeSlices` (32), `kLargeThreads` (128) | `cuda_raster.cu` | tuning constants for the two-tier rasterizer |
| `GFX_CUDA_ARCHITECTURES` | CMake cache | `120` in the presets and `build.bat` |

## 17. Known scope and limitations

| Item | Notes |
|---|---|
| **No near-plane clipping** | A triangle with any vertex behind the camera (`w <= 0`) or with a screen coordinate beyond ±16384 pixels is dropped entirely, instead of being clipped. Walking the camera into a large model makes triangles vanish earlier than in OpenGL. Fixed in M6: triangles and edges are now clipped against the near plane and a guard band (see `m6-design.md`) |
| **No anti-aliasing** | One sample per pixel at the pixel centre; edges are aliased. Only the grid is anti-aliased, analytically |
| **Large-triangle launch overhead** | The large kernel launches a block for every triangle and slice, about 500,000 mostly empty blocks for Suzanne. Fixed in M6 by a triangle setup pass that fills a worklist (see `m6-design.md`) |
| **Bounding-box rasterization** | Every thread tests every pixel of the bounding box. Thin diagonal triangles waste work; tiled or hierarchical rasterizers avoid this |
| **Atomic contention** | Many triangles covering the same pixel serialize on one `atomicMin`. Fine for this scene |
| **Depth precision** | Depth is a 32-bit float in `[0, 1]`, the same precision as the other backends, but ties are resolved by triangle id, not by draw order |
| **Facing rule** | Front-facing means negative original signed area because the screen is y-down. Any change to the viewport transform must keep this in step |
| **Grid depth bias** | The 0.1% bias is a fixed fraction of the ray length, like in the other backends |
| **Two GPUs** | The CUDA device is the one OpenGL runs on. If OpenGL picks the Intel GPU, the CUDA backend refuses to start |
| **Demo triangles** | With no model, the CUDA backend draws only the grid; the OpenGL and Vulkan demo triangles are not ported |
| **Single stream** | Everything runs on the default stream; there is no overlap of the copy with the next frame |
| **Profiling cost** | `kProfileStages` forces a sync per frame |
| **Platform** | Windows only (WGL interop) |

## 18. Completion checkpoint

The expected state at the end of M5:

- `gfxlab.exe --backend=cuda model.obj` opens the same viewport as the other two backends, with the orbit camera, the infinite grid and a lit model that can be grabbed, rotated and scaled;
- the picture matches the OpenGL backend for the same camera and model;
- resizing, minimizing, maximizing and vsync changes work;
- a close-up of a cube runs in the thousands of frames per second, and Suzanne at about 900;
- the Debug build is silent in the console;
- `ctest` shows 8 passing tests.

Next is M6: polish and a backend comparison (wireframe, frame times of the three backends side by side, near-plane clipping and triangle compaction for the CUDA backend). M7 follows with a CUDA ray tracer using a CPU-built BVH.
