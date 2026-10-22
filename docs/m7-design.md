# M7 — The CUDA Ray Tracer

> **Status:** Complete (Steps 1–5c)  
> **Scope:** A ray tracer inside the CUDA backend, switched on with **T**: camera rays through a BVH built on the CPU, shadows from a sun that can be dragged around the model, ambient occlusion that refines while the view is still, ray statistics, and two rounds of measured optimization.

## 1. Purpose

The rasterizer of M5 and M6 starts from triangles: for each triangle, which pixels does it cover? A ray tracer starts from pixels: for each pixel, which surface does a ray from the camera hit first? For the plain viewport both give the same picture. The difference is what else a ray tracer can ask. From the point it hit, it can fire more rays: towards a light (is it in shadow?) or into the surroundings (how enclosed is it?). A rasterizer has no direct way to ask either question.

M7 builds that ray tracer in CUDA, from the ray/triangle test upwards, and measures what each effect costs.

## 2. What M7 adds

| Area | Files |
|---|---|
| Ray tests | `src/core/ray.h` (slab test, Möller–Trumbore) |
| Binary BVH | `src/core/bvh.h`, `bvh.cpp` (SAH builder, `bvh_trace`, `bvh_occluded`) |
| 4-wide BVH | `src/core/bvh4.h`, `bvh4.cpp` (collapse from the binary tree, `bvh4_trace`, `bvh4_occluded`) |
| Matrix inverse | `src/core/mat4.h` (`mat4_inverse`) |
| Tests | `tests/test_bvh.cpp` (new, the 9th CTest program), `tests/test_math.cpp` |
| Frame flags | `renderer.h`: `ray_trace`, `sun_dir`, `sun_enabled`, `ambient_occlusion` |
| Controls | `app/main.cpp`: T, right-button drag, L, O; the title shows the mode |
| Device data | `cuda/cuda_mesh.*`: both BVHs and the bounding box per mesh |
| Ray tracer | `cuda/cuda_raster.h`, `cuda_raster.cu`: `RtScene`, building blocks, megakernel and wavefront kernels |
| Ray statistics | `cuda/cuda_ray_stats.h`, `cuda_ray_stats.cpp` |
| Renderer | `cuda/cuda_renderer.*`: scene setup, AO accumulation, path and tree switches |

All ray maths in `gfx_core` is `GFX_HD`, so the CPU tests and the GPU kernels run the same functions.

## 3. Step 1 — Ray tests and a BVH built on the CPU

### The two tests

- **`ray_hits_box`** is the slab test: the ray's entry and exit distances for the three pairs of parallel planes of an axis-aligned box; it hits if the latest entry is before the earliest exit. A zero direction component gives an infinite reciprocal, which the comparison handles.
- **`ray_hits_triangle`** is Möller–Trumbore. It returns the distance `t` and the barycentric weights `u, v` of the second and third corners, and is two-sided like the viewport lighting.

### Why a BVH

Testing every triangle is about 15,000 tests per ray for `suzanne_s.obj`, times about a million pixels. A **bounding volume hierarchy** puts the model in a box, splits the triangles into two groups with their own boxes, and repeats until each box holds a few triangles. A ray that misses a box skips everything inside it, so a typical ray tests a few dozen boxes and a handful of triangles.

### The builder (`build_bvh`)

- Each triangle's box and centre are computed once.
- For a node, the centres are sorted into 16 buckets along each axis, and every plane between buckets is scored with the **surface area heuristic**: `left count × left area + right count × right area`. The chance that a random ray hits a box is roughly proportional to its surface area, so this favours small boxes around many triangles.
- Nodes with more than 4 triangles are always split (when a split exists); smaller ones only when the split is cheaper than a leaf. Depth is limited to 48 (`kBvhMaxDepth`).
- Nodes are 32 bytes (`BvhNode`: box, count, first child or first triangle). The two children of a node are stored next to each other, so one index finds both.
- `suzanne_s.obj`: 17,097 nodes, 8,549 leaves, depth 18, at most 4 triangles per leaf, built in about 13 ms.

### Traversal

`bvh_trace` finds the closest hit with a small fixed-size stack, visiting the nearer child first and skipping boxes farther away than the best hit so far. `bvh_occluded` (Step 3) answers "is anything in the way?" and stops at the first triangle it finds. On the CPU a ray through the BVH cost about 0.2 µs, against about 130 µs testing every triangle.

### Tests

`test_bvh` checks the ray tests on known answers, an empty mesh and a single triangle, the tree's structure (every triangle in exactly one leaf, every box containing its children) on 2,000 random triangles and on `suzanne_s.obj`, and that thousands of random rays give exactly the same answers as testing every triangle, for both the closest hit and the any-hit query. The tests found a real builder bug before the first delivery: combining a box with an empty box leaked infinities into the result, which produced leaves of 140 triangles.

## 4. Step 2 — Camera rays in the CUDA backend

**T** switches the CUDA backend between the rasterizer and the ray tracer (wireframe takes priority). The other backends ignore it.

- **The camera ray** for a pixel is `forward + ndc_x × right + ndc_y × up`, the grid's ray basis from M6. It is not normalized: its view-space depth component is exactly 1, so the distance `t` along it equals the depth in front of the camera, and rays simply run from `t = near` to `t = far`, the same range the rasterizer keeps.
- **Instances.** Each mesh's BVH is built once, in the mesh's own space, when the mesh is added. Instead of rebuilding it when the model moves, the ray is moved into the model's space with the **inverse model matrix** (`mat4_inverse`). The direction is not renormalized, so `t` is the same in both spaces and hits from several models compare directly. Up to 8 instances are passed to the kernel by value in `RtScene`.
- **Shading.** The hit's barycentrics blend the three vertex normals. Which side was hit is decided by the sign of `dot(triangle normal, ray direction)`, which is the same in object and world space. The lighting is `lit_color()`, the same function the rasterizer's shading pass uses.
- **The grid.** The ray tracer writes the same depth into the visibility buffer that the rasterizer would, so the grid pass composites unchanged.

A CPU run of both paths covered the same 1,916 pixels of a small Suzanne image except 2, with 7 pixels visibly different, all on silhouette edges.

## 5. Step 3 — Shadows and the sun

### Shadow rays

From a hit, a second ray goes towards the light. If anything blocks it, the light's diffuse and highlight are removed; the ambient fill stays, so shadows are dark but not black. Shadow rays only need to know whether *anything* is in the way, so they use `bvh_occluded`. Surfaces facing away from the light skip the ray.

**Shadow acne.** A shadow ray starts on a surface, and rounding can put its start just below it, so the surface shadows itself in speckles. The start point is lifted along the triangle's normal by `1e-4 × max(1, largest coordinate)`; float precision is relative, so the lift scales with the coordinates.

### The sun (Step 3b)

The first version used the viewport's light, which is attached to the camera; shadows then slid around while orbiting. In ray-traced mode the light is now a **sun** fixed in the world:

- Two angles: azimuth around the vertical axis and elevation above the floor, default 250° / 45°, which lights the front of the model from the startup camera. They become a direction `(cos el · cos az, cos el · sin az, sin el)`.
- **Right mouse button drag** moves it (0.25° per pixel; sideways turns it, up and down raises and lowers it, between 5° and 90°). A drag only starts when no G/R/S edit is active, because the right button also cancels edits.
- **L** turns the sun off: the ray tracer then uses the camera light without shadows, which matches the rasterized picture.
- The sun is passed in world space for shadow rays and in view space for the lighting.

An early version also darkened the floor under the model. It was removed: the grid is a viewport aid, not a floor surface.

## 6. Step 4 — Ambient occlusion

Ambient light stands in for light arriving from everywhere around. A point in a crease sees less of its surroundings and should get less of it. **Ambient occlusion** measures that: fire short rays in random directions above the point, and the fraction that escape is how open it is. The result scales only the ambient term.

- **Directions** are **cosine-weighted**: a random point on a disc lifted onto the hemisphere, so directions near the normal are chosen more often, matching how much each direction contributes. Tangents come from the branchless construction of Duff et al. (2017).
- **Random numbers** come from a hash (PCG) of the pixel and sample numbers, so every thread has its own repeatable sequence without a shared generator.
- **Reach**: 20% of the largest model's diagonal (from the BVH root box), so only the neighbourhood counts.
- **Progressive refinement.** 4 rays per pixel are noisy, so each frame adds 4 more to a per-pixel running total (`m_ao_accum`). A **signature** of everything that affects the picture (camera, model matrices, sun, toggles) restarts the total when anything changes. After 1,024 samples per pixel the result is kept and no more AO rays are traced.
- **O** turns AO on and off.

The step also removed five build warnings: a shadowed test variable, hidden padding in `RasterTri` (now explicit, with `static_assert`s), an unused function, a warning in `nvcc`'s generated stub files (`/wd4211`), and a false "divide by zero" in the BVH test's deliberate edge cases.

## 7. Step 5 — Measuring

### Ray statistics

Each block of the ray-tracing kernel adds its threads' ray counts in shared memory, then one thread adds the block's totals to three global counters (camera, shadow, AO). `CudaRayStats` reads them back the same way the GPU timers do: a ring of slots, each with device counters, a pinned host copy and an event, read only when the event says the copy has arrived. Once per second it logs `CUDA rays per frame: camera ... shadow ... AO ... (AO samples n / 1024)`.

`kBenchmarkAo` restarts AO every frame, so its cost can be timed.

### Measurement conditions

All numbers: `suzanne_s.obj`, startup camera, 1280×720, Release, vsync off, RTX 5080, GPU time in ms. The window must stay focused: an unfocused window measured 3–5% faster. Other GPU programs should be closed.

At the startup camera Suzanne covers about 6% of the screen: every pixel fires a camera ray (0.92 M), but only about 55,000 hit her and fire shadow (0.05 M) and AO (0.22 M) rays.

### Results with the binary BVH

| Row | Megakernel | Wavefront |
|---|---|---|
| 1. Rasterizer | 0.35 | |
| 2. Camera rays only | 0.37 | |
| 3. + sun shadows | 0.74 | 0.74 |
| 4. + AO refining | 1.44 | **1.00** |
| 5. AO converged | about row 3 | |

- **Camera rays cost almost nothing here**: most rays miss the model's root box with one test.
- **Shadow rays were expensive**: about 0.37 ms for 50,000 rays, about 7–8 ns each. They start inside the model's boxes and, when not blocked, must pass every box along the way.
- **AO was the most expensive** in total, at four rays per hit pixel.
- **Convergence works**: once AO reaches 1,024 samples, the frame costs the same as without AO.

## 8. Step 5b — Wavefront: one thread per ray

With one thread per pixel (the **megakernel**), the 55,000 hit pixels each traced a shadow ray and then 4 AO rays one after another. That is too little independent work to keep an RTX 5080 busy: BVH traversal is mostly waiting for memory, and the GPU hides waiting by switching between many warps.

The **wavefront** version splits the work into passes connected through GPU memory:

1. `rt_camera_kernel`, one thread per pixel: trace the camera ray; append each hit (`RtHit`: position, pixel, normals) to a list.
2. `rt_secondary_kernel`, one thread per **ray**: work item `i` belongs to hit `i / 5`; slot 0 is the shadow ray, slots 1–4 the AO rays. A fixed grid loops over the list, whose length only the GPU knows. AO results are added per hit with `atomicAdd`.
3. `rt_finish_kernel`, one thread per hit: combine the results into colour and depth.

Both versions are built from the same device functions (`closest_surface`, `trace_shadow`, `trace_ao_ray`, `finish_pixel`) and produce byte-identical images.

Result: AO went from 0.70 ms to 0.24 ms (about 1 ns per AO ray instead of 3). With AO off there is nothing to spread out, and the extra passes cost about 0.02 ms.

## 9. Step 5c — Automatic choice and the 4-wide BVH

- **Automatic choice.** The wavefront runs only while AO is refining; otherwise the megakernel runs. `kRtWavefront = false` forces the megakernel.
- **The 4-wide BVH** (`bvh4.h`). `Bvh4Node` (128 bytes) holds up to four child boxes side by side. `build_bvh4` makes it from the binary tree by repeatedly opening the largest inner child until a node has four children, keeping the triangle order (Suzanne: 4,186 nodes instead of 17,097). The idea is fewer, larger memory reads per ray. `kUseBvh4` selects the tree; both give identical images.

### What the measurements showed

| Build | Row 3: sun, AO off | Row 4: AO refining |
|---|---|---|
| Without the 4-wide code (binary tree) | 0.74 | 1.00 |
| With the 4-wide code, `kUseBvh4 = true` | **0.39** | about 0.63 |
| With the 4-wide code, `kUseBvh4 = false` | 0.38 | about 0.65 |

The build with the 4-wide code is about **twice as fast for shadows**, but inside that build the two trees perform the same. So the gain does not come from the 4-wide layout itself. Code that never runs can still change how the compiler builds a kernel: how many registers each thread uses (and so how many warps fit on an SM to hide memory waits), and whether the traversal is inlined. Comparing `nvcc -Xptxas=-v` register and spill counts of the two builds would confirm which; that is left open. The 4-wide build is kept because it is the fastest measured.

**Lessons**

- Measure every optimization against the version before, in one sitting, under the same conditions.
- Wavefront helps when it creates parallelism (several AO rays per pixel), not otherwise.
- On a GPU, how the compiler allocates registers can matter as much as the algorithm.

## 10. Problems found during M7

| Problem | Cause and fix |
|---|---|
| Leaves of 140 triangles | Combining a box with an empty box leaked infinities. Fixed (min of mins, max of maxes); caught by the tests |
| Shadows slid while orbiting | The light was attached to the camera. A world-fixed sun in ray-traced mode |
| Sun on arrow keys | Replaced by right-button drag, which only starts when no edit is active |
| Floor shadow on the grid | Removed; the grid is not a surface |
| Five build warnings | Fixed in Step 4 (see Section 6) |
| Wrong first explanation of the slowness | Half-empty warps were suspected; the real issue was too few parallel rays. The wavefront measurement showed it |
| An unexpected 2× speedup | Came from the 4-wide build, not the tree layout (Section 9) |

## 11. Verification

- `ctest`: 9 programs; `test_bvh` checks both trees against brute force and `test_math` checks `mat4_inverse`.
- The CUDA kernels were also compiled as plain C++ and run on the CPU: rasterizer versus ray tracer, megakernel versus wavefront, binary versus 4-wide BVH, Step 4 versus Step 5 images (identical where expected).
- On the RTX 5080: the checkpoints of each step and the measurements of Sections 7–9.

## 12. Manual checklist

1. `--backend=cuda suzanne_s.obj`, **T**: same shape and grid as the rasterizer, lit by the sun; the title shows `(ray traced, sun 250 / 45 deg, AO)`.
2. Shadows under the brows and around the ears; no speckled acne on `cube.obj`.
3. Right-drag moves the sun; the log prints its angles on release; orbiting leaves the shadows in place.
4. **L**: camera light, no shadows (like the rasterizer). **O**: AO off and on.
5. Holding still, AO goes from grainy to smooth; moving restarts it; the GPU time drops once it reaches 1024 / 1024.
6. G/R/S and zooming through the model work in ray-traced mode; **W** still shows the wireframe.
7. A G edit cancelled with a right click does not move the sun.
8. No `CUDA error` lines in Debug; `ctest` passes 9 tests.

## 13. Known scope and limitations

| Item | Notes |
|---|---|
| **One bounce** | Camera, shadow and AO rays only; no reflections or indirect light (path tracing is planned for M9) |
| **Materials** | One grey material, as in the viewport |
| **Instances** | Up to 8 draw items are ray traced; more are skipped with a warning |
| **Hard shadows** | The sun is a direction, so shadow edges are sharp |
| **AO noise** | Visible while moving; it refines only when the view is still |
| **Memory** | Wavefront buffers are sized to one hit per pixel (about 44 MB at 1280×720) |
| **Software traversal** | BVH traversal runs on the regular CUDA cores, not on the RT cores (the ray tracing hardware in every RTX GPU, reachable only through DXR, Vulkan ray tracing or OptiX). Measured here: about 0.1 to 1 billion rays per second, against several billion or more with hardware ray tracing |
| **Open question** | The exact cause of the 4-wide build's speedup (Section 9) |
| **Wireframe** | Always rasterized, also with T on |

## 14. Completion checkpoint

- **T** switches the CUDA backend to a ray tracer that matches the rasterized picture, with sun shadows and ambient occlusion on top;
- the sun can be dragged and turned off; AO refines and then stops costing time;
- ray counts and GPU times are logged, and the measurements are recorded above;
- `ctest` shows 9 passing tests.

Next is M8: a small user interface to view and change these settings without keyboard shortcuts.
