# M4 — The Vulkan Backend

> **Status:** Complete (Steps 1–10)  
> **Scope:** Render the same scene as the OpenGL backend through the same `IRenderer` interface using Vulkan 1.3: an instance and device, a swapchain, frames in flight, a depth buffer, uploaded meshes, per-frame uniforms, lighting that matches OpenGL, the ray-cast infinite grid, and live resizing.

## 1. Purpose

M2 and M3 produced a complete viewer on OpenGL. M4 draws the same thing with Vulkan, so that the two APIs can be compared side by side. Both backends receive the same `FrameData` and the same draw list, and they should produce the same picture.

OpenGL hides a lot of work inside the driver: the context, synchronization, memory, state tracking, the swapchain, and the layout of images. Vulkan makes the application do all of it explicitly. Most of M4 is therefore about learning what the driver used to do:

- creating and choosing a device;
- owning the swapchain and recreating it when the window changes;
- synchronizing the CPU with the GPU, and the GPU with the presentation engine;
- allocating memory and moving data into GPU-only memory;
- describing every piece of pipeline state up front;
- passing data to shaders through descriptors and push constants.

Backend selection is a command-line choice:

```
gfxlab.exe                                  OpenGL (the default)
gfxlab.exe --backend=vulkan                 Vulkan
gfxlab.exe --backend=vulkan model.obj       Vulkan with a model
```

The first argument that does not start with `--` is the model path. It is looked up as given, then in `assets\`, then in `assets\local\` (see `m3-design.md`), so a bare name such as `cube.obj` is enough.

## 2. Architecture and what M4 adds

```
gfxlab        main.cpp: --backend=opengl|vulkan, V toggles vsync
gfx_renderer  IRenderer, FrameData, create_renderer() (unchanged apart from one factory case)
gfx_vk        VulkanRenderer and the classes below
gfx_gl        (unchanged)
gfx_core      (unchanged)
```

New files, all in `src/renderer/vulkan/` unless noted:

| Area | Files |
|---|---|
| Common helpers | `vk_api.h` (`vk_ok`, readable `VkResult` names) |
| Instance, device, surface | `vk_context.h`, `vk_context.cpp` |
| Swapchain | `vk_swapchain.h`, `vk_swapchain.cpp` |
| Frames in flight and synchronization | `vk_frames.h`, `vk_frames.cpp` |
| Memory and depth | `vk_memory.h`, `vk_memory.cpp`, `vk_depth.h`, `vk_depth.cpp` |
| Buffers and uploads | `vk_buffer.h`, `vk_buffer.cpp`, `vk_upload.h`, `vk_upload.cpp` |
| Shaders | `vk_shader.h`, `vk_shader.cpp`, `shaders/*.vert`, `shaders/*.frag`, `cmake/GfxShaders.cmake` |
| Pipelines | `vk_pipeline.h`, `vk_pipeline.cpp` |
| Per-frame uniforms | `vk_uniforms.h`, `vk_uniforms.cpp` |
| Drawing | `vk_triangle_demo.*`, `vk_mesh.*`, `vk_grid.*` |
| The renderer | `vk_renderer.h`, `vk_renderer.cpp` |

`gfx_vk` links the Vulkan loader (`Vulkan::Vulkan`) and `gfx_core`. The factory in `renderer_factory.cpp` remains the only file that uses `#if GFX_HAS_*`.

The classes follow one rule: **each class owns one kind of Vulkan object, and destroys it in its destructor.** `VulkanRenderer` declares its members in dependency order, so that C++ destroys them in the reverse of creation:

```cpp
VulkanContext       m_context;    // instance, device, surface, queue (destroyed last)
VulkanSwapchain     m_swapchain;
VulkanDepthBuffer   m_depth;
VulkanFrames        m_frames;
VulkanUploader      m_uploader;
VulkanFrameUniforms m_uniforms;   // before the pipelines that use its layout
VulkanTriangleDemo  m_triangle;
VulkanMeshRenderer  m_meshes;
VulkanGridRenderer  m_grid;       // destroyed first
```

`~VulkanRenderer` calls `vkDeviceWaitIdle` first, so nothing is destroyed while the GPU is still using it.

## 3. Step 1 — Toolchain: the Vulkan SDK and `glslc`

The Vulkan SDK provides the headers, the loader, the validation layer and `glslc`, the compiler from GLSL to SPIR-V. `cmake/GfxBackends.cmake` runs `find_package(Vulkan COMPONENTS glslc)` (CMake 3.24 or newer) and the backend is only built when both are found. The configure output shows `GFXLab: Vulkan and glslc found`.

Machines often have more than one GPU. The development machine has an RTX 5080 and the processor's integrated Intel UHD 770. Selecting the right one is part of Step 2.

## 4. Step 2 — Instance, validation, surface and device

### The instance

`VulkanContext` creates a Vulkan **1.3** instance after checking that the loader itself reports 1.3 or newer. The instance enables `VK_KHR_surface` and `VK_KHR_win32_surface` (the way to present to a Win32 window).

### The validation layer and debug messenger

In a Debug build, `VK_LAYER_KHRONOS_validation` is enabled when it is installed, together with `VK_EXT_debug_utils`. The layer checks every API call against the specification and reports misuse. Without it, a Vulkan mistake is undefined behaviour that may simply work on one GPU. A debug messenger routes the layer's messages to the project logger: errors as `LOG_ERROR`, everything else as warnings.

- The messenger's create-info is also **chained into the instance's `pNext`**. This way, problems found while creating and destroying the instance itself are reported too.
- Whether validation is requested is `constexpr bool kRequestValidation`, rather than an `#ifndef`, so that the code in both configurations is always compiled and warning-free in a Release build.
- The log line **`Vulkan validation layer enabled`** should appear in every Debug run. If it does not, the layer is not installed, and silence from it means nothing.

### The surface

`vkCreateWin32SurfaceKHR` turns the `HWND` into a `VkSurfaceKHR`. The renderer receives the window handle through the same `init(void* hwnd)` call as OpenGL.

### Choosing the physical device

Every physical device is examined and rejected, with the reason logged, unless it:

- reports Vulkan 1.3 or newer;
- supports `VK_KHR_swapchain`;
- supports the features the renderer uses: **dynamic rendering**, **synchronization2** and **shaderDemoteToHelperInvocation** (see Section 14);
- has a queue family that supports both graphics and presentation to this surface.

Suitable devices are scored: **a discrete GPU +1000, an integrated GPU +100**, and the best wins. On the development machine the RTX 5080 is chosen over the Intel iGPU.

### The logical device

One queue is created from the chosen family, since graphics and present share it. The device enables the swapchain extension and a `VkPhysicalDeviceVulkan13Features` structure with the three features above.

### Why Vulkan 1.3 core

Two features that used to require extensions are core in 1.3 and make the code much shorter:

- **Dynamic rendering** replaces render passes and framebuffers. `vkCmdBeginRendering` takes the image views directly, and pipelines declare the attachment formats they expect (`VkPipelineRenderingCreateInfo`).
- **Synchronization2** has clearer barriers (`vkCmdPipelineBarrier2`) and submission (`vkQueueSubmit2`) with the stage and access flags in one place.

Core functions are exported by the loader. Only extension functions would need `vkGetInstanceProcAddr`.

## 5. Step 3 — The swapchain

A swapchain is a queue of images owned by the presentation engine. The application acquires one, draws into it, and hands it back. `VulkanSwapchain` chooses:

| Choice | Value | Reason |
|---|---|---|
| Format | `B8G8R8A8_UNORM` with `SRGB_NONLINEAR` (then `R8G8B8A8_UNORM`, then the first available, with a warning) | The OpenGL backend does no sRGB conversion, so a UNORM target reproduces the same colours |
| Present mode | vsync on: `FIFO`; vsync off: `IMMEDIATE`, else `MAILBOX`, else `FIFO` | FIFO is the only mode the spec guarantees, and it waits for the display's refresh |
| Extent | the surface's `currentExtent` | On Windows it equals the client area |
| Image count | `minImageCount + 1`, capped at `maxImageCount` | One spare image so the application is not waiting for the driver |
| Usage | `COLOR_ATTACHMENT` | The scene is rendered straight into the images |
| Composite alpha | opaque | The window is not transparent |
| Pre-transform | the surface's `currentTransform` | |

One image view is created per image, and the class owns and destroys them.

### Recreation

A swapchain becomes invalid when the window size changes. Recreation is `vkDeviceWaitIdle`, then destroying everything that depends on the old swapchain, then creating the new one. Step 10 refines how this is triggered and done. `set_vsync` also goes through the same path, because the present mode is fixed at creation.

## 6. Step 4 — Frames in flight, dynamic rendering and synchronization

### The problem

The CPU should record frame N+1 while the GPU is still drawing frame N. But anything the CPU touches for frame N+1 must not still be in use by frame N. Vulkan gives three primitives:

| Primitive | Synchronizes | Used for |
|---|---|---|
| **Fence** | GPU → CPU | "this frame slot's earlier work has finished" |
| **Semaphore** | GPU → GPU (queue operations) | "the image is acquired", "rendering has finished" |
| **Barrier** | inside a command buffer | ordering of work, memory visibility, image layouts |

### `VulkanFrames`

Two **frame slots** (`kFramesInFlight = 2`) each own a command buffer, a fence that starts *signaled* (so the first wait returns at once), and an `image_available` semaphore. A command pool serves both.

The `render_finished` semaphores are **per swapchain image, not per frame slot**. The presentation engine may still hold a semaphore long after a frame slot has been reused, and reusing it too soon is a validation error. Because their count depends on the image count, they are recreated with the swapchain.

### The frame, step by step

1. Wait for the slot's fence. The slot's command buffer and uniform buffer are now safe to reuse.
2. Acquire the next swapchain image, signalling `image_available`.
   `OUT_OF_DATE` → recreate the swapchain and skip the frame.
3. Validate everything needed (image, view, semaphore) **before** resetting the fence.
4. Reset the fence, reset and begin the command buffer.
5. Barrier: colour image `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL`.
6. Barrier: depth image `UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL`.
7. `vkCmdBeginRendering` with a colour attachment (cleared to 0.10, 0.11, 0.13) and a depth attachment (cleared to 1.0, contents not stored).
8. Draw: the mesh or the demo triangles, then the grid.
9. `vkCmdEndRendering`.
10. Barrier: colour image `COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR`.
11. End the command buffer. `vkQueueSubmit2`, waiting for `image_available` at the colour-attachment-output stage, signalling `render_finished` and the slot's fence.
12. `vkQueuePresentKHR`, waiting for `render_finished`. Then advance to the next slot.
    `OUT_OF_DATE` or `SUBOPTIMAL` → recreate.

### Rules that matter here

- **Reset the fence only when a submit is certain.** An early return after `vkResetFences` leaves the fence unsignaled forever, and the next `vkWaitForFences` blocks for ever. Everything that can fail is checked before the reset.
- **The first barrier's stage must match the semaphore wait stage.** The acquire semaphore is waited on at `COLOR_ATTACHMENT_OUTPUT`, and the layout transition uses the same stage, so the image is transitioned only after it has been acquired.
- **The depth image is shared by both frames in flight.** Its barrier therefore uses the early *and* late fragment-test stages on both sides, so that frame N+1's depth work is ordered after frame N's.
- A barrier helper, `transition_image(cmd, image, aspect, old, new, src_stage, src_access, dst_stage, dst_access)`, builds the `VkImageMemoryBarrier2` for all three transitions.

## 7. Step 5 — Shaders and the first triangle

### Building shaders

Vulkan consumes SPIR-V, not GLSL. `cmake/GfxShaders.cmake` defines `gfx_compile_shaders(target output_dir sources...)`, which adds one custom command per shader that runs:

```
glslc --target-env=vulkan1.3 -o <build>/shaders/<name>.spv <source>
```

and a target, `gfx_shaders`, that `gfx_vk` depends on. A change to a shader source therefore rebuilds only that shader. The output directory reaches the code as the `GFX_SHADER_DIR` compile definition, and `load_shader_module` reads the `.spv` file at start-up and creates a `VkShaderModule`.

`GFX_SHADER_DIR` is an absolute path into the build directory, so the executable can be run from anywhere, but it cannot yet be moved to another machine without the build tree.

### Shader conventions that differ from OpenGL

- `#version 450`, with explicit `layout(location = N)` on every interface variable.
- `gl_VertexIndex` replaces `gl_VertexID`.
- Uniforms live in blocks with `set` and `binding` numbers, or in `push_constant` blocks. There are no loose `uniform mat4 u_x;` declarations.

### The first triangle

`VulkanTriangleDemo` draws six hard-coded vertices generated from `gl_VertexIndex`, with no vertex buffer. The pipeline gets its formats from `VkPipelineRenderingCreateInfo`, uses an empty pipeline layout, and leaves viewport and scissor as **dynamic state** so a resize does not rebuild the pipeline. Clip-space Y points **down** in Vulkan, which matters for the grid in Step 9.

## 8. Step 6 — Memory and the depth buffer

### Memory

A `VkImage` or `VkBuffer` is only a handle; the memory behind it is a separate allocation. `find_memory_type(physical_device, type_bits, required_flags)` picks a memory type that is both allowed for the resource and has the properties the code needs. The project makes **one allocation per resource**. That is simple and fine for a handful of objects, but a real renderer would use a sub-allocator, because drivers limit the number of allocations (see Section 18).

### `VulkanDepthBuffer`

A `D32_SFLOAT` image in device-local memory, created at the swapchain's extent with a depth-stencil attachment usage and an image view. The format is checked against the device's optimal-tiling features, with a clear error if unsupported. It is recreated with the swapchain. Its barrier is described in Section 6.

The demo from Step 5 became two overlapping triangles (a near RGB one at depth 0.3 and a larger far yellow one at 0.7) to prove the depth test: the far triangle must be hidden where the near one overlaps, whatever the draw order.

### Format-aware pipelines

A pipeline is compatible only with the attachment formats it was created for. The renderer caches the colour and depth formats and **rebuilds the pipelines only when a recreated swapchain changes them**. A resize alone never rebuilds a pipeline.

## 9. Step 7 — Buffers, staging uploads and the first mesh

### `VulkanBuffer`

Wraps `VkBuffer` + `VkDeviceMemory` with the size and memory flags, with `create`, `destroy`, and `write` (map, copy, unmap). `write` requires host-visible **and host-coherent** memory, so no explicit flush is needed. It refuses anything else.

### Device-local memory and staging

GPU-only memory is the fastest for the GPU, but the CPU cannot map it. Static data such as meshes goes there through a **staging buffer**:

1. Create a host-visible, coherent staging buffer, and copy the data into it.
2. Create the destination buffer in `DEVICE_LOCAL` memory, with the final usage flags plus `TRANSFER_DST`.
3. Record `vkCmdCopyBuffer` in a one-time command buffer from a transient pool.
4. Submit with `vkQueueSubmit2` and a fence, and wait for it.
5. Reset the fence for the next upload and destroy the staging buffer.

`VulkanUploader::upload_to_device_buffer` does all of that. The wait makes uploads **synchronous**: simple, and acceptable at load time, but it would stall a renderer that streams data.

The fence is created *unsignaled*, waited on after the submit, and reset after the wait succeeds. A failed submit therefore leaves it unsignaled and consistent.

### `VulkanMeshRenderer`

`create(ctx, uploader, mesh)` validates the mesh (matching position and normal counts, indices in range, sizes that fit), **interleaves** it as `position.xyz normal.xyz` (24 bytes per vertex, the same as the OpenGL backend), uploads one vertex and one index buffer, and returns a `MeshId` (the vector index plus one, 0 meaning none).

The pipeline declares:

- one vertex binding with stride 24, attribute 0 (position) at offset 0 and attribute 1 (normal) at offset 12;
- `uint32` indices (`vkCmdBindIndexBuffer` with `VK_INDEX_TYPE_UINT32`);
- depth test on, depth write on, `LESS`;
- no culling, front face counter-clockwise (see Section 10).

The first shader pair coloured each pixel by its normal (`normal * 0.5 + 0.5`), which is an easy way to check the geometry and the vertex layout before any lighting exists.

## 10. Step 8 — Descriptors, per-frame uniforms and matching the lighting

OpenGL uploads each uniform by name with `glUniform*`. Vulkan needs two mechanisms, chosen by how often the data changes:

| OpenGL uniform | Changes | Vulkan |
|---|---|---|
| `u_view`, `u_view_proj` | once per frame | **uniform buffer**, bound through a descriptor set |
| `u_model`, `u_normal_model` | once per draw | **push constants**, 128 bytes |

### `VulkanFrameUniforms`

- One host-visible, coherent **uniform buffer per frame in flight**, because the CPU writes the next frame's data while the GPU may still be reading the previous one. This is the same reason there are two command buffers and two fences.
- One **descriptor set layout** (binding 0, uniform buffer, vertex and fragment stages), one descriptor pool, and one descriptor set per frame in flight.
- Each set is written **once** at start-up to point at its buffer. Only the buffer contents change afterwards.
- `update(frame_index, uniforms)` writes the current frame's buffer. It is safe because the frame slot's fence was waited on first. No barrier is needed for the host write, because coherent memory plus a queue submit makes it visible to the device.

### Push constants

`struct MeshPush { Mat4 model; Mat4 normal_matrix; }` is exactly 128 bytes, the minimum every device must support. It is pushed for each draw item with `vkCmdPushConstants`, vertex stage. `Mat4` is already four column-major `Vec4`s, so it matches GLSL's `mat4` layout without conversion.

### The lighting

The Vulkan shaders are the OpenGL lighting ported line by line (Section 5 of `m3-design.md`): view-space lighting, hemisphere ambient from `mat3(view)[2]`, Lambert diffuse, a small Blinn-Phong highlight and two-sided shading with `gl_FrontFacing`. Only the declarations change:

```glsl
layout(set = 0, binding = 0) uniform Frame { ... } frame;
layout(push_constant) uniform Push { mat4 model; mat4 normal_model; } pc;
```

### Why the Vulkan picture is not mirrored

`perspective(..., ClipSpace::CLIP_SPACE_VULKAN)` produces depth in [0, 1] and flips Y, so the projection already accounts for Vulkan's downward clip-space Y. The image looks the same as in OpenGL. Vulkan defines winding in framebuffer coordinates with that Y direction in mind, so a counter-clockwise OBJ mesh is still front-facing with `frontFace = COUNTER_CLOCKWISE`. Setting it the other way round makes every face look inside out, and the lighting lights the wrong side.

### The `std140` layout rule

`FrameUniforms` is the C++ mirror of a GLSL block. A `mat4` is 64 bytes, and in `std140` a `vec3` is aligned to 16 bytes, so a `float` that follows a `vec3` is packed into its padding, which is an easy way to read a wrong value with no error. This project avoids the problem by using only `mat4` and `vec4` members, and a `static_assert` on the size.

## 11. Step 9 — A shared pipeline helper and the infinite grid

### `create_graphics_pipeline`

Three pipelines (triangle, mesh, grid) share nearly all of their state. `VulkanPipelineDesc` carries only what differs:

```cpp
struct VulkanPipelineDesc {
    VkShaderModule vert, frag;  VkPipelineLayout layout;
    VkFormat color_format, depth_format;
    std::vector<VkVertexInputBindingDescription>   bindings;
    std::vector<VkVertexInputAttributeDescription> attributes;
    bool depth_test = true, depth_write = true, blend = false;
};
```

The function bakes in what never varies: triangle lists, no culling, counter-clockwise front face, one sample, dynamic viewport and scissor, depth compare `LESS`, and the rendering-info formats. With `blend` set, colour and alpha both use `SRC_ALPHA / ONE_MINUS_SRC_ALPHA`, which is what `glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)` does. Shader modules, pipeline layouts and destruction stay in each renderer class.

### The grid

The grid is the OpenGL algorithm (Sections 8 and 9 of `m2-design.md`): one fullscreen triangle without a vertex buffer, a ray per pixel intersected with the plane z = 0 in the fragment shader, anti-aliased lines, coloured axes, a distance fade, and `gl_FragDepth` written so models hide the grid correctly. Four things change for Vulkan:

1. **NDC Y is down.** The ray is built as `forward + ndc.x * right - ndc.y * up`. The CPU-side `up` keeps the same meaning as in OpenGL, and the flip lives in the shader.
2. **Depth is [0, 1].** `gl_FragDepth = clip.z / clip.w`, discarding outside [0, 1].
3. **`gl_VertexIndex`** replaces `gl_VertexID`.
4. **The camera data comes from the shared uniform buffer.** `FrameUniforms` was extended to 208 bytes:

```cpp
struct FrameUniforms {
    Mat4 view;  Mat4 view_proj;
    Vec4 eye;       // xyz
    Vec4 forward;   // xyz
    Vec4 right;     // xyz, scaled by tan(fov/2) * aspect
    Vec4 up;        // xyz, scaled by tan(fov/2)
    Vec4 params;    // x = grid fade distance
};
```

The grid pipeline has no vertex input, no push constants, depth test on, **depth write off** and blending on. It is drawn **after** the meshes so the depth test hides it behind models. Writing `gl_FragDepth` disables early depth testing for this shader, which is fine for one fullscreen triangle.

The set stays bound from the mesh draw, but the grid binds it again explicitly. Relying on descriptor-set compatibility rules between pipeline layouts is not worth the confusion for a very cheap call.

### Depth bias for geometry on the floor

A model's base face lies exactly in the grid's plane. Their depths come from different computations (interpolation versus the per-pixel ray), so with a `LESS` test they flip pixel by pixel and the axes appear as dots on the base when seen from below. As in the OpenGL backend, the shader computes the **depth** from a point 0.1% farther along the ray, `frame.eye.xyz + (t * 1.001) * v_ray`, so geometry on the floor always wins. Colour, line coverage and the fade still use the true hit point.

### The mesh and grid shaders must declare the same block

All four shaders (`mesh.vert`, `mesh.frag`, `grid.vert`, `grid.frag`) declare the full `Frame` block, even where a stage uses only part of it. The block must match the C++ structure and the descriptor set layout.

## 12. Step 10 — Resize, minimize and the modal loop

The main loop is not running while the user drags a window edge: Windows is in its own modal loop, and `Window::set_modal_frame_callback` calls the same `draw_frame` from inside it (the mechanism from M2). The Vulkan backend must keep up with the changing size.

### Detect the change instead of waiting for an error

`vkAcquireNextImageKHR` and present return `OUT_OF_DATE` or `SUBOPTIMAL` after a resize, but how soon a driver reports it varies, and until then frames can be shown stretched. `render()` therefore compares `FrameData` with the swapchain at the start of every frame:

1. A zero-sized frame (minimized) returns immediately and silently.
2. If there is no swapchain, or the frame's width or height differs from the swapchain's extent, the swapchain is recreated first. If that fails, the frame is skipped and the next frame retries.

The `OUT_OF_DATE` and `SUBOPTIMAL` handling after acquire and present stays as a safety net.

### Cheaper and quieter recreation

- `VulkanSwapchain::recreate` passes the old handle as `oldSwapchain`. The driver may reuse resources, and presentation of images already in flight can continue. The old image views are destroyed first, and the old swapchain is retired **after the attempt whether or not it succeeded**. After any failure the class holds no swapchain, so the retry condition above works.
- Recreation is `vkDeviceWaitIdle` + recreate the swapchain + depth buffer + `VulkanFrames`. The pipelines are rebuilt only if a format changed.
- The messages for each recreation are `LOG_DEBUG`, because a drag recreates many times per second.
- Pipeline creation for all three renderers is one function, `init_pipelines(color_format, depth_format)`, called from `init()` and from the recreate path.

### Vsync

`V` toggles vsync from `main.cpp` (a `vsync` flag, `renderer->set_vsync(vsync)` and a log line). The key works with both backends. In Vulkan it recreates the swapchain with `FIFO` or the non-vsync present mode.

## 13. Differences between the OpenGL and Vulkan backends

| Concern | OpenGL | Vulkan |
|---|---|---|
| Context / device | WGL context, one call | Instance, validation, surface, physical-device scoring, logical device |
| Presenting | `SwapBuffers`, vsync with an extension | Swapchain, acquire/present, present mode, recreation |
| Synchronization | Implicit in the driver | Fences, semaphores and barriers written by hand |
| Image layouts | Hidden | Explicit transitions every frame |
| Shaders | GLSL compiled at run time from strings | GLSL compiled to SPIR-V at build time with `glslc` |
| Uniforms | `glUniform*` by name | Descriptor set + uniform buffer, push constants |
| Pipeline state | Global state, changed at will | Baked into a pipeline object per combination |
| Memory | Hidden | Chosen memory types, allocations, staging copies |
| Clip space | Depth [-1, 1], Y up | Depth [0, 1], Y down |
| Errors | `glGetError` / debug callback | Return codes and the validation layer |
| Resize | The framebuffer follows the window | The swapchain and its dependents are recreated |

## 14. Problems found during M4

| Problem | Cause and fix |
|---|---|
| Unused function warning in a Release build (C4505) | `has_instance_layer` was only used when validation code was compiled in. Replaced the `#ifndef` with `constexpr kRequestValidation` so both paths always compile |
| Backend name said "Vulkan 1.4" | The instance requests 1.3. `name()` now returns "Vulkan 1.3" |
| Possible deadlock after an error | Some early returns came after `vkResetFences`, leaving the fence unsignaled for ever. All checks now happen before the reset |
| Pipelines rebuilt on every resize | The triangle pipeline was re-created in every swapchain recreation. They are now rebuilt only if the colour or depth format changes |
| A missing barrier that validation did not report | While refactoring, the colour `UNDEFINED → COLOR_ATTACHMENT_OPTIMAL` barrier was removed, then the depth barrier. The scene still rendered, and validation stayed silent. The check was to confirm that `Vulkan validation layer enabled` is logged and to remove a barrier on purpose to see the layer complain |
| `OpDemoteToHelperInvocation` rejected by validation | `glslc --target-env=vulkan1.3` compiles GLSL `discard` into that SPIR-V instruction, which needs the `shaderDemoteToHelperInvocation` feature. Vulkan 1.3 requires it, so device selection now requires it and the device enables it. Semantics are unchanged for the grid, since the `discard` is at the end of the shader |
| Stretched image during a drag | Relying on `OUT_OF_DATE` alone. `render()` now compares the frame size with the swapchain extent every frame |
| Dotted axes on a model's base seen from below | The grid and the base face are coplanar and their depths differ by rounding (z-fighting). The grid's depth is now biased 0.1% away from the camera, in both backends |
| Grid faded out too close to the model | The fade ended at 1.5 times the camera distance and started at 20% of that. Now 2.5 times, starting at 40% |

## 15. Verification

There are no new unit tests. The Vulkan code cannot run without a GPU and a window, so the correctness checks for M4 are:

- the **validation layer**, which is expected to be silent in a Debug build through start-up, rendering, resizing, vsync changes and shutdown;
- a **side-by-side comparison** with `--backend=opengl`, for the same model and camera;
- the manual checklist in Section 17.

The seven existing CTest programs are unchanged and still pass.

## 16. Frame-loop sequence (Vulkan)

1. Input and editing in `main.cpp`, as in M3. `V` toggles vsync.
2. `draw_frame()` fills `FrameData` and calls `render()`.
3. `render()`: return if the frame has no area; recreate the swapchain if its size differs from the frame.
4. Wait for the frame slot's fence.
5. Compute `projection` for Vulkan clip space and fill `FrameUniforms` (matrices, eye, camera basis, grid fade). Write the slot's uniform buffer.
6. Acquire a swapchain image.
7. Reset the fence, record: transition colour and depth, begin rendering, draw the mesh (or demo triangles), draw the grid, end rendering, transition for presentation.
8. Submit, present, advance the frame slot.
9. Recreate the swapchain if present reported it out of date.

## 17. Manual checklist

Run all of these with `--backend=vulkan` in a **Debug build** and read the console.

1. The log shows `Vulkan validation layer enabled`, the discrete GPU's name as `selected Vulkan GPU`, and the Intel GPU is not chosen.
2. With no model: two overlapping triangles in front of the grid (they are fixed to the screen), the larger one hiding behind the smaller one.
3. `gfxlab.exe --backend=vulkan assets\suzanne_smooth.obj`: Suzanne looks the same as in `--backend=opengl` from the same camera: key light up-left, bluish ambient from above, a small highlight.
4. `cube.obj`: lit edges, one face brighter than the others. `suzanne_flat.obj` stays faceted.
5. The grid matches OpenGL: red X axis, green Y axis, minor and major lines, fade with distance, visible from below the floor, hidden behind the model. Seen from below, the axes disappear cleanly under the model's base, with no dotted pattern.
6. **G**, **R**, **S** with X/Y/Z constraints, confirm and cancel work, with correct lighting under non-uniform scale.
7. Orbit, pan and zoom work. Zoom in and out does not make the grid shimmer.
8. Drag every edge and corner of the window: the picture re-renders live, never stretched or black.
9. Maximize, restore, and snap left and right.
10. Minimize: no log output and no CPU use. Restore: the first frame is correct.
11. Press **V** repeatedly, also just before and after a resize. The frame rate changes and the picture stays correct.
12. Close the window during a drag: it exits without a hang.
13. No `[Vulkan validation]` message appears at any point in steps 1 to 12.
14. A Release build runs the same scenes, without the validation layer.

## 18. Known scope and limitations

| Item | Notes |
|---|---|
| **Platform** | Win32 surface only |
| **Vulkan version** | Requires a 1.3 loader and device, with dynamic rendering, synchronization2 and shader demote. Older devices are rejected |
| **Single queue** | Graphics and present share one queue; there is no async transfer or compute |
| **Memory** | One allocation per resource; no sub-allocator. Drivers limit the number of allocations (the guaranteed minimum is 4096), which is plenty for this scene but would not scale to many small meshes |
| **Uploads** | Synchronous, with a fence wait per buffer. A mesh is two uploads |
| **Uniform writes** | `VulkanBuffer::write` maps and unmaps on each call instead of keeping a persistent mapping |
| **Swapchain format** | UNORM, matching OpenGL's lack of sRGB conversion. Colours are not gamma-correct in either backend |
| **Depth format** | Only `D32_SFLOAT` is requested; there is no fallback format |
| **Recreation** | Waits for the whole device to go idle and recreates the frame objects, which is simple but not the cheapest possible. The pipelines are kept unless a format changes |
| **Error recovery** | Out-of-date surfaces are handled. Device loss is not |
| **Persistent failure** | If recreation fails on every frame, it is retried on every frame and each failure is logged |
| **Shader location** | Shaders are loaded from the build directory (`GFX_SHADER_DIR`), an absolute path |
| **Pipeline cache** | None; pipelines are built at start-up and rebuilt on format changes |
| **Triangle demo** | Fixed to the screen in clip space, unlike the OpenGL demo, which is in the world and spins. It is the "no model loaded" fallback and a depth-test check |
| **Grid depth** | Writing `gl_FragDepth` disables early depth testing for the grid pass |
| **Anti-aliasing** | No multisampling; the grid's anti-aliasing is analytic |
| **Validation** | Only in Debug builds, and only if the SDK's layer is installed |

## 19. Completion checkpoint

The expected state at the end of M4:

- `gfxlab.exe --backend=vulkan` opens the same viewport as OpenGL, with an orbit camera, the infinite grid, and a lit model that can be grabbed, rotated and scaled;
- the picture matches the OpenGL backend for the same camera and model;
- the discrete GPU is chosen on a machine with several GPUs;
- resizing, minimizing, maximizing and vsync changes work without stretching, hangs or validation messages;
- the validation layer is silent in a Debug build, and the Release build runs;
- `ctest` still shows 7 passing tests.

Next is M5: a CUDA software rasterizer. It will draw the same scene into a buffer on the GPU and display the result through OpenGL interop, using the same `IRenderer` interface.
