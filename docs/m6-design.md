# M6 — Polish and Backend Comparison

> **Status:** Complete (Steps 1–8)  
> **Scope:** A Blender-style wireframe in all three backends, GPU timing that never stalls the CPU, a measured comparison of OpenGL, Vulkan and CUDA, and two fixes to the CUDA rasterizer found by that comparison: a triangle worklist and near-plane clipping.

## 1. Purpose

M5 finished the third backend. M6 makes the three comparable and closes the gaps the comparison exposes:

- **Wireframe (W)** that looks like Blender's: the model's original polygon edges in one flat colour, not the triangles the GPU actually draws.
- **GPU timing** in every backend, shown next to the frame time, so the backends can be compared fairly and each number can be read for what it is.
- **A triangle worklist** for the CUDA rasterizer. The measurements showed the M5 large-triangle kernel spending most of its time on empty blocks.
- **Clipping** for the CUDA rasterizer, so triangles that cross the camera plane are cut instead of disappearing, as they are in OpenGL and Vulkan.

## 2. What M6 adds

| Area | Files |
|---|---|
| Polygon edges | `core/mesh.{h,cpp}` (`Mesh::edges`, `ensure_edges`), `core/obj_loader.cpp`, `tests/test_obj.cpp` |
| Wireframe toggle | `renderer/renderer.h` (`FrameData::wireframe`), `app/main.cpp` (W key) |
| OpenGL wireframe | `opengl/gl_mesh.{h,cpp}` (edge VAO/EBO, wire program) |
| Vulkan wireframe | `vulkan/vk_pipeline.h` (`topology`), `vulkan/vk_mesh.{h,cpp}`, `vulkan/shaders/wire.{vert,frag}` |
| CUDA wireframe | `cuda/cuda_mesh.{h,cpp}` (device edges), `cuda/cuda_raster.cu` (`draw_edges_kernel`, `resolve_wire_kernel`) |
| GPU timing | `renderer/renderer.h` (`IRenderer::last_gpu_ms`), `opengl/gl_timer.{h,cpp}`, `vulkan/vk_timer.{h,cpp}`, `cuda/cuda_gpu_timer.{h,cpp}`, `app/main.cpp` (title and log) |
| CUDA worklist | `cuda/cuda_raster.{h,cu}` (`RasterTri`, `RasterCounts`, `RasterWork`, setup and grid-stride kernels), `cuda/cuda_renderer.{h,cpp}` |
| CUDA clipping | `cuda/cuda_raster.{h,cu}` (`ProjectedVertex::clip`, `clip_to_screen`, `plane_distance`, Sutherland–Hodgman in the setup kernel, segment clipping in the edge kernel) |
| Shared grid maths | `renderer/renderer.h` (`GridRayBasis`, `make_grid_ray_basis`) |

`IRenderer` gained one method, with a default so a backend can opt out:

```cpp
virtual bool last_gpu_ms(double& ms) { return false; }
```

## 3. Steps 1–3 — Wireframe

### Why `glPolygonMode` was not enough

The first attempt drew the triangles with `glPolygonMode(GL_LINE)`. It looked wrong next to Blender for two reasons:

1. **It showed triangles, not polygons.** The OBJ loader splits every quad and n-gon into triangles, and the GPU only ever sees triangles, so every diagonal looked like an edge. Blender draws the edges of its original faces.
2. **It was lit.** The lines went through the lit mesh shader, so lines facing away from the light faded into the background and looked broken.

### Keeping the polygon outlines (`Mesh::edges`)

`Mesh` has a second index list, `edges`, with two vertex indices per edge. The OBJ loader fills it from each face's outline **before** fan triangulation: `(v[i], v[(i + 1) % n])`. The fan's inner diagonals are never added.

- Edges are deduplicated by the pair of **OBJ position indices**, `(min, max)` packed into a 64-bit key. Vertex indices would not work: in a flat-shaded file the same corner has a different vertex on each face (different normals), so a shared edge would be stored twice.
- A face that repeats a position produces no zero-length edge.
- `ensure_normals` unwelds meshes without normals into one vertex per triangle corner. It records the first copy of every old vertex and remaps the edge list through it.
- `ensure_edges` builds deduplicated triangle edges for meshes that have none (meshes built in code). `main.cpp` calls it after `ensure_normals`.
- Tests: a quad has 4 edges, two quads sharing an edge have 7, the cube has 12 (with and without normals), a triangle has 3, and the fallback builds 5 edges for a two-triangle quad and does not duplicate them when called again.

An OBJ that is already triangulated has only triangles, so its wireframe shows triangles. That is correct for the file. `assets/suzanne_fl.obj` is Blender's standard Suzanne exported without triangulation (468 quads and 32 triangles) and is the model to compare against Blender.

### OpenGL

`GLMesh` uploads the edges to a second element buffer and owns a **second VAO** that reuses the same vertex buffer with only attribute 0 (position). The element-buffer binding is part of a VAO's state, which is why a second VAO is the clean way to switch index lists. A tiny program draws `GL_LINES` with `u_view_proj * u_model * pos` and a uniform colour of 0.85 grey. A mesh without edges uploads normally and draws nothing in wireframe.

### Vulkan

A Vulkan pipeline bakes its primitive topology in when it is created, so wireframe is a second pipeline:

- `VulkanPipelineDesc` gained `topology` (default triangle list).
- The wire pipeline uses `VK_PRIMITIVE_TOPOLOGY_LINE_LIST`, one 24-byte binding with only the position attribute, and the mesh pipeline's layout (same descriptor set, same 128-byte push constants).
- Each mesh has an optional edge index buffer, drawn with `vkCmdDrawIndexed`.
- This needs no device feature. `VK_POLYGON_MODE_LINE` would need `fillModeNonSolid`, and would bring back the triangle diagonals.
- `wire.vert` declares the full `Frame` block, the M4 convention for every shader using the descriptor set.

### CUDA

There is no line hardware, so the CUDA backend has a small line rasterizer that reuses the visibility buffer:

- `draw_edges_kernel`, one thread per edge:
  - clips the segment in clip space (Section 8);
  - converts both ends to screen space;
  - clips them to the viewport with **Liang–Barsky**, so long off-screen lines cost nothing;
  - steps along the line with a DDA, `ceil(max(|dx|, |dy|)) + 1` samples;
  - depth is interpolated linearly, which is correct because depth is linear in screen space after the divide;
  - each sample does an `atomicMin` of `(depth bits << 32) | 1` into the visibility buffer.
- `resolve_wire_kernel` paints every covered pixel 0.85 grey (217).
- The grid kernel is unchanged: it already depth-tests against the visibility buffer, so wires hide the grid behind them.

## 4. Steps 4–5 — GPU timing

### Frame time and GPU time

- **Frame time** is wall-clock time from one frame to the next: CPU work, driver work, waiting for vsync, presenting.
- **GPU time** is how long the GPU spent executing one frame's commands.

GPU time close to the frame time means the GPU is the bottleneck. GPU time much smaller means the frame is limited by the CPU, the driver or presentation. Turning vsync on is the quickest demonstration: the frame time locks to the refresh interval and the GPU time does not change.

### The rule: never wait for the GPU

A timer result is ready only after the GPU has finished that frame, one to three frames after it was submitted. Asking for it immediately would make the CPU wait for the GPU and destroy the overlap being measured. All three timers therefore keep a small ring of timers, read a result only when it is known to be ready, and otherwise try again next frame. Results arrive late, which does not matter for averages.

| Backend | Mechanism | How "ready" is known |
|---|---|---|
| OpenGL (`GLGpuTimer`) | 4 `GL_TIME_ELAPSED` query objects around clear … grid | `GL_QUERY_RESULT_AVAILABLE`; if the oldest query is not ready, the frame is not timed and the same slot is checked next frame |
| Vulkan (`VulkanGpuTimer`) | A timestamp query pool with 2 queries per frame in flight: reset and `TOP_OF_PIPE` before rendering, `BOTTOM_OF_PIPE` after `vkCmdEndRendering` | The frame slot's fence: once it has been waited on, that slot's timestamps are complete, so `vkGetQueryPoolResults` is called without the wait flag |
| CUDA (`CudaGpuTimer`) | 4 pairs of events; start before the clears, end after the copy into the GL texture | `cudaEventQuery`: `cudaErrorNotReady` is the normal "not yet" answer and is not logged |

Vulkan details: `timestampPeriod` converts ticks to nanoseconds, and the raw values are masked to the queue family's `timestampValidBits` before subtracting. If the queue has no valid bits, timing is disabled with a warning. Every timer's `init` may fail without stopping the renderer.

`main.cpp` averages the samples over each FPS interval:

```
gfxlab | OpenGL 4.6 | 2850.0 FPS | frame 0.351 ms | GPU 0.112 ms
```

and logs one line per second: `<backend> | frame ms ... | GPU ms ...`. With no samples it shows `GPU --`.

`kProfileStages` in `cuda_renderer.cpp` (M5) is separate and still useful for a per-stage breakdown, but it waits for the GPU every frame.

## 5. The comparison

Release build, vsync off, 1280×720 window, startup camera. Frame / GPU time in milliseconds, measured on the RTX 5080 after Step 5 (before the worklist):

| Scene | OpenGL | Vulkan | CUDA |
|---|---|---|---|
| No model (grid only) | 0.060 / 0.006 | 0.175 / 0.010 | 0.263 / 0.257 |
| `cube.obj` solid | 0.068 / 0.007 | 0.206 / 0.012 | 0.319 / 0.296 |
| `suzanne_s.obj` solid | 0.065 / 0.009 | 0.198 / 0.016 | 1.131 / 1.081 |
| `suzanne_s.obj` wireframe | 0.068 / 0.010 | 0.213 / 0.016 | 0.299 / 0.280 |
| `suzanne_fl.obj` solid | 0.078 / 0.007 | 0.195 / 0.012 | 0.530 / 0.505 |

How to read it:

1. **OpenGL and Vulkan: the GPU is almost idle.** 6–16 µs is close to what the timers can resolve, and the two backends do not bracket exactly the same commands. These numbers do not show that one API is faster than the other.
2. **OpenGL and Vulkan: the frame is CPU-bound.** About 90% of OpenGL's 0.06 ms frame is CPU, driver and presentation. Vulkan's extra ~0.13 ms is also on the CPU, in this project's own per-frame code: the fence wait, acquire, command recording, mapping and unmapping the uniform buffer, submit and present. The OpenGL driver hides and optimizes the equivalent work. Vulkan gives control, not automatic speed: when the GPU work is tiny, per-frame fixed costs decide the result, and in Vulkan those costs belong to the application.
3. **CUDA is GPU-bound**, the only backend that keeps the GPU busy, with a fixed cost of about 0.25 ms per frame: clearing 11 MB of buffers, the grid kernel (three rays per pixel), and mapping, copying into and unmapping the GL texture.
4. **CUDA's solid cost grew with the triangle count, not the pixel count.** Above the grid-only baseline: cube +0.04 ms, `suzanne_fl` +0.25 ms, `suzanne_s` +0.82 ms, while Suzanne covers about the same pixels in both versions. The M5 large-triangle kernel launched `triangles × 32` blocks of 128 threads, about 504,000 blocks for `suzanne_s`, and every thread repeated the full triangle setup before exiting. Step 6 removes this.

After Step 6, `suzanne_s.obj` solid: OpenGL 0.056 / 0.009, Vulkan 0.180 / 0.014, **CUDA 0.367 / 0.366**, down from 1.081 ms. The triangle work above the fixed baseline went from about 0.82 ms to about 0.11 ms.

## 6. Step 6 — The triangle worklist

Before, both rasterizer kernels read the mesh's index list and each re-derived every triangle. Now a **setup pass** looks at each triangle once and writes compact work for the rasterizers, the way a hardware pipeline's setup stage feeds its rasterizer.

```cpp
struct RasterTri {
    RasterTriangleSetup setup;       // snapped coordinates, original signed area
    int min_x, max_x, min_y, max_y;  // pixel bounds, clamped to the screen
    float4 screen[3];                // x, y, depth01, w      (setup order)
    float4 view_pos[3];              //                       (setup order)
    float4 view_normal[3];           //                       (setup order)
};

struct RasterCounts {
    unsigned int record_count, small_count, large_count, overflow_count;
};
```

The fields are called `small_count` and so on because `small` is a macro in `<windows.h>`.

The frame, in solid mode:

1. `cuda_reset_raster_counts`: `cudaMemsetAsync` on stream 0, ordered with the kernels.
2. Per draw item: transform the vertices, then `setup_triangles_kernel` (one thread per triangle). It validates the triangle, computes its bounds, **drops it if it is entirely off-screen** (free culling), takes a record slot with `atomicAdd`, writes the record and appends its index to the small or the large list (bounding box above or below 1024 pixels).
3. `rasterize_small_kernel` and `rasterize_large_kernel`, launched once for all items.
4. `shade_kernel`: one pass over all pixels. Visibility id − 1 is the record index; the record holds everything shading needs, so the per-item shading loop and `triangle_base` are gone.

**The CPU never reads the counts.** Copying a counter back each frame would make the CPU wait for the GPU. Instead the rasterizers are launched with a **fixed** grid sized from the SM count (`cudaDevAttrMultiProcessorCount`) and loop over work whose size only the GPU knows:

- small: SMs × 4 blocks of 256 threads; each thread grid-strides over `small_list[0 .. small_count)`;
- large: SMs × 8 blocks of 128 threads; blocks stride over the `(triangle, slice)` pairs, `large_count × 32` of them, and threads share the columns of each row as before.

The buffers grow and never shrink. Records are appended in whatever order the threads run, so two triangles at exactly the same depth could swap winners between frames; that only happens with duplicated coplanar faces. Debug builds read the counters after the existing per-frame sync and warn once if the worklist overflowed.

## 7. Step 7 — Clipping

### The problem

The vertex stage divides by `w`, the distance in front of the camera. For a vertex behind the camera `w ≤ 0`, and the divide flips the point to the wrong side of the screen or sends it to infinity. M5 therefore dropped any triangle with such a vertex, and any triangle beyond ±16,384 pixels. Faces vanished when the camera came close to a model.

### Clip before the divide

Clipping happens in **clip space**, the 4D position after the MVP multiply and before the divide by `w`. A straight edge is still straight there, and every attribute (the view-space position and normal included) is a linear function of the same original position, so the point a fraction `t` along an edge has all its attributes exactly `t` of the way between the ends. `ProjectedVertex` therefore stores `clip`, and `clip_to_screen()` is the one place where clip space becomes screen space.

### The planes

`plane_distance(c, plane, g) ≥ 0` means kept:

| Plane | Distance |
|---|---|
| Near | `z + w` (OpenGL clip space; it also guarantees `w > 0`) |
| Guard band left / right | `g·w + x`, `g·w − x` |
| Guard band bottom / top | `g·w + y`, `g·w − y` |

The guard band is far outside the screen: `g = 0.5 · (2 · 16384 / max(width, height) − 1)`, about 12 screen widths at 1280×720. It does not trim anything visible; it keeps clipped vertices well inside the ±16,384-pixel range of the fixed-point rasterizer. There is no far plane: the per-sample depth test already drops samples beyond it.

### Triangles

In `setup_triangles_kernel`:

1. Classify the three corners against the five planes.
2. If every corner is outside the same plane, drop the triangle.
3. **Fast path**: if every corner is inside every plane, which is almost every triangle, use the stored screen positions as before.
4. Otherwise **Sutherland–Hodgman**: clip the polygon against one plane at a time, alternating between two local arrays of up to 8 corners (3 plus at most one per plane). For each edge, keep the start corner if it is inside, and add the crossing point at `t = d_a / (d_a − d_b)` if the edge crosses the plane, interpolating clip position, view position and normal with the same `t`.
5. Fan-triangulate the result, `(0, i, i + 1)`, into at most 6 triangles, convert each to screen space and write it with `emit_triangle`. Clipping keeps the winding, so each piece faces the same way as the original.

The worklist now holds `2 × triangles + 1024` records; the overflow counter covers the rare worst case.

### Edges

`draw_edges_kernel` clips the segment against the same planes in clip space: for each plane it computes the two distances, rejects the edge if both are negative, and otherwise narrows `[t0, t1]` to the part that is inside. The two clipped ends then go through `clip_to_screen`, the viewport clip and the DDA, unchanged.

### Checking it without the GPU

The kernels are plain C++ apart from the launch syntax, so they were also compiled for the CPU with stand-in CUDA declarations and compared against a ray-cast reference. A floor quad passing under the camera, which M5 dropped entirely, matched the reference pixel for pixel, and an edge crossing the camera plane left the screen at the predicted row.

## 8. Step 8 — One place for the grid's camera rays

All three backends built the grid's ray basis the same way: `tan_half = tan(fov_y / 2)`, `right × tan_half × aspect`, `up × tan_half`. It now lives in `renderer.h`:

```cpp
struct GridRayBasis { Vec3 eye, forward, right, up; float fade_end; };

GridRayBasis make_grid_ray_basis(const FrameData& frame, float aspect);
```

The aspect ratio is a parameter because Vulkan uses its swapchain extent. OpenGL fills `GridParams`, Vulkan the `FrameUniforms` vectors and CUDA `CudaGridParams` from it.

## 9. Problems found during M6

| Problem | Cause and fix |
|---|---|
| Wireframe did not look like Blender | `glPolygonMode` draws triangles, including diagonals, through the lit shader. Replaced by the polygon edge list and an unlit wire shader |
| Triangulated models show triangles in wireframe | The file has no quads to show. Added `suzanne_fl.obj`, exported from Blender with quads |
| `GLMesh::upload` failed for meshes without edges | It required an edge list. Edges are now optional in all three backends |
| Zero-length edges | A face repeating a position produced one. The loader skips them, like `ensure_edges` |
| `wire.vert` declared only part of the `Frame` block | Valid in Vulkan (a prefix of a block), so validation was silent, but against the M4 rule. It now declares the full block |
| A CUDA profile marker recorded twice | Left over when the wireframe branch was added. Removed |
| OpenGL timer skipped ahead when a result was not ready | Moving to the next slot broke the ring order and starved measurements when the GPU was behind. It now waits on the oldest slot |
| `small` is a Windows macro | `<windows.h>` defines `small`, so the counters are named `small_count` and so on |
| `r` declared twice in the shade kernel | The record was named `r`, and the lighting code already uses `r` for red. Renamed to `rec`; caught by a compile check before the first build |
| Vulkan timer slot count duplicated | Tied to the frames-in-flight count instead of a separate constant |
| CUDA solid cost scaled with triangle count | The large kernel's empty blocks. Fixed by the worklist (Step 6) |
| Faces vanished near the camera in CUDA | No clipping. Fixed in Step 7 |

## 10. Verification

- `ctest` runs 8 test programs; `test_obj` covers the edge lists.
- Each backend's wireframe is compared with the others and with Blender (`suzanne_fl.obj`).
- The GPU times are checked by turning vsync on and off, and by confirming that timing does not change the frame rate.
- The CUDA worklist and clipping are compared with OpenGL at the same camera, in solid, wireframe and debug-id modes, and the clipping maths was run on the CPU against a reference (Section 7).
- The Vulkan validation layer is silent in Debug; the CUDA backend logs no errors and no worklist overflow.

## 11. Manual checklist

1. With each backend, **W** toggles wireframe; `cube.obj` shows 12 edges and no diagonals.
2. `suzanne_fl.obj` in wireframe matches Blender's wireframe and looks the same in all three backends (lines may differ by a pixel between APIs).
3. Wire colour is even everywhere, with no shading and no gaps; the grid hides behind the wires.
4. The title shows `GPU … ms` for every backend; with vsync on the frame time locks to the refresh rate and the GPU time does not.
5. FPS with timing on matches FPS without it.
6. `--backend=cuda suzanne_s.obj` solid runs at about 0.37 ms GPU time (RTX 5080, 1280×720).
7. Zoom through Suzanne with `--backend=cuda` and compare with `--backend=opengl`: faces crossing the camera are cut along a straight line instead of disappearing, in solid and wireframe.
8. Put the camera inside the cube: the inside faces are visible.
9. An extreme close-up of a cube face shows no missing faces and no flicker.
10. G/R/S, orbit, resize, minimize and V work in solid and wireframe with every backend.
11. Debug: no `[Vulkan validation]` messages, no `CUDA error` lines, no worklist overflow warning.
12. `ctest` passes 8 tests.

## 12. Known scope and limitations

| Item | Notes |
|---|---|
| **Wireframe style** | 1-pixel aliased lines in one colour. Blender draws anti-aliased lines; that needs a quad-based line shader |
| **Wireframe depth** | All edges are drawn, including hidden ones (like Blender's wireframe with X-ray). There is no hidden-line mode |
| **Triangulated files** | A file without polygons shows triangles; quads cannot be recovered from triangles |
| **Timer resolution** | OpenGL and Vulkan GPU times of a few microseconds are near the resolution of the timers and are not a reliable way to rank the two APIs for this scene |
| **Timing brackets** | Each backend times slightly different work: OpenGL clear to grid, Vulkan the rendering plus its first barrier, CUDA everything up to the texture copy. None includes presentation |
| **CUDA fixed cost** | About 0.25 ms per frame of clears, grid and interop copy, now the largest part of the CUDA frame |
| **Worklist order** | Record ids vary between frames; triangles at exactly equal depth may swap winners |
| **Worklist capacity** | `2 × triangles + 1024`; a scene where many triangles each clip into several pieces could overflow it (warned in Debug) |
| **Near-plane depth** | Samples exactly on the near plane can be lost to rounding at depth 0, as with any near clip |
| **Vulkan CPU overhead** | Per-frame uniform map/unmap and the other CPU work in `render()` dominate Vulkan's frame time in this small scene |

## 13. Completion checkpoint

The expected state at the end of M6:

- **W** shows a Blender-style polygon wireframe in OpenGL, Vulkan and CUDA;
- every backend reports its GPU time without slowing the frame, and the title shows it next to the frame time;
- the CUDA backend handles the camera close to and inside models like the other two, and `suzanne_s.obj` costs about a third of its M5 GPU time;
- the grid's ray basis is computed in one place;
- `ctest` shows 8 passing tests.

Next is M7: a CUDA ray tracer using a BVH built on the CPU.
