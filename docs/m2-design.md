# M2 — OpenGL Renderer and Orbit Camera

> **Status:** Complete (Steps 1–9)  
> **Scope:** Create a native OpenGL 4.6 core context, load modern OpenGL entry points, render shader-driven geometry, add view/projection transforms and depth testing, add Blender-style orbit-camera controls, draw an infinite ray-cast floor grid, put the backend behind a renderer interface, and keep rendering while Windows runs its drag/resize loop.

## 1. Purpose

M2 establishes the first real GPU-rendering path in gfxlab. Instead of drawing through a high-level framework, the project creates its own WGL context, loads the OpenGL functions it needs, manages a small set of GPU objects, and draws geometry through shaders.

The milestone also connects the previously separate pieces:

- platform window and input from M1;
- vector, matrix, and camera math from `gfx_core`;
- OpenGL resources and drawing from `gfx_gl`;
- a backend-neutral renderer interface from `gfx_renderer`;
- application behavior and the frame loop in `main.cpp`.

At the end of M2 the app shows a Blender-style viewport: a dark background, an infinite grid floor with red X and green Y axes that fades into the distance, and two intersecting vertex-colored demo triangles. The camera orbits, pans, and zooms with the mouse, and the scene keeps animating while the window is being dragged or resized.

## 2. Architecture and dependency boundaries

```
gfxlab        EXECUTABLE  src/app              input handling, camera, frame loop
gfx_renderer  STATIC      src/renderer         IRenderer, FrameData, create_renderer()
gfx_gl        STATIC      src/renderer/opengl  WGL context, loader, shaders, grid, GLRenderer
gfx_platform  STATIC      src/platform         window and input
gfx_core      STATIC      src/core             math, orbit camera, timer, log
```

- `gfx_core` supplies math and reusable logic, including `OrbitCamera`.
- `gfx_platform` creates the window and supplies its native handle and input state.
- `gfx_gl` creates the OpenGL context, loads entry points, owns OpenGL resources, and issues draw calls. Everything OpenGL-specific, including the clip-space convention, stays inside it.
- `gfx_renderer` declares the interface that the app talks to and the factory that creates a backend. The factory is the only place that checks `GFX_HAS_OPENGL`, `GFX_HAS_VULKAN` and `GFX_HAS_CUDA`.
- `main.cpp` handles user input, updates the camera and animation, fills a `FrameData`, and calls the renderer. After Step 8 it contains no OpenGL.

`gfx_core` must not depend on `InputState`, Win32, or OpenGL. This keeps the orbit camera testable without a window or GPU context.

## 3. Step 1 — Create the OpenGL context

### Why WGL is needed

On Windows, OpenGL rendering requires a rendering context associated with a window's device context. The `GLContext` class owns the WGL context and exposes operations such as creation, destruction, buffer swapping, and vertical-sync configuration.

Modern context creation requires WGL extension functions. Windows' legacy OpenGL exports do not directly expose every modern entry point, so the implementation first creates a temporary dummy window and legacy context. It uses that context to retrieve the WGL extension functions (`wglChoosePixelFormatARB`, `wglCreateContextAttribsARB`), requests a real OpenGL 4.6 core context for the application's window, and then destroys the temporary bootstrap resources.

The bootstrap is a setup technique; the dummy window is not the application window and is not used for rendering.

### Context lifetime

The context depends on the window's native handle, so it must be destroyed first. In Step 1 this was done with declaration order in `main`; since Step 8 the context is the first member of `GLRenderer`, and the renderer is declared after the window:

```cpp
gfx::Window window;
std::unique_ptr<gfx::IRenderer> renderer = gfx::create_renderer(...);
```

C++ destroys locals and members in reverse declaration order, so the renderer (and with it every GL object and the context) is destroyed before the window.

Context setup also configures the swap interval when the relevant extension is available and logs the driver's vendor, renderer, OpenGL version, and GLSL version. A debug context is requested in debug builds (`!NDEBUG`). If the driver cannot provide one, a warning is logged in debug builds only.

## 4. Step 2 — Load OpenGL entry points explicitly

The project uses its own small OpenGL loader rather than an external loader library.

The function list is centralized in `gl_functions.h` using an X-macro list. The same list drives function-pointer declarations and loading, so a function cannot be declared in one place and forgotten in another.

`gl_api.h` provides the OpenGL types, constants, and function-pointer declarations used by the renderer. The implementation avoids including the legacy `<GL/gl.h>` after defining its own API declarations, preventing conflicting declarations.

### Loading sequence

1. Create the real OpenGL context.
2. Make that context current on the calling thread.
3. Load the required OpenGL functions.
4. Fail context creation if a required entry point cannot be loaded.
5. Query driver information and configure debug output.

The loader first tries `wglGetProcAddress`. Some functions are instead exported directly by `opengl32.dll`, so it falls back to `GetProcAddress` for those. `wglGetProcAddress` can return the sentinel values 0, 1, 2, 3 or -1 on failure, which must be treated as failures rather than valid function pointers.

In debug builds the debug-message callback prints driver diagnostics, which finds invalid API usage without every error having to be discovered visually.

## 5. Step 3 — Draw a shader-driven triangle

The first draw demo is intentionally small. `TriangleDemo` owns:

- a shader program;
- a vertex array object (VAO);
- a vertex buffer object (VBO).

The vertex data stores each vertex's position and color. The vertex shader multiplies the position by the MVP matrix (added in Step 4) and forwards the color; the fragment shader writes the interpolated color to the output pixel.

### What the GPU objects do

- **VBO:** stores the vertex data in GPU-accessible memory.
- **VAO:** records how vertex attributes are interpreted and where their data comes from.
- **Shader program:** tells the GPU how to process vertices and produce fragment colors.

The demo draws three vertices as `GL_TRIANGLES`. The vertex colors blend across the triangle because the rasterizer interpolates the varying color values across its surface.

Shader compilation and program linking are checked, and the info log is printed when a shader or program fails.

## 6. Step 4 — Model, view, projection, and depth

Step 4 moves the triangle from a flat demonstration into a 3D camera pipeline. The combined transform is:

```
MVP = Projection * View * Model
```

Each matrix has one job:

- **Model:** places or rotates one object in world space.
- **View:** expresses the world from the camera's point of view.
- **Projection:** maps the camera view into clip space using a perspective field of view and aspect ratio.

The vertex shader multiplies each input position by the MVP matrix. `Mat4` is column-major, so it is uploaded with `glUniformMatrix4fv(..., GL_FALSE, ...)` and no transpose.

### Two intersecting triangles

The demo draws two instances of the same triangle geometry with different model transforms: triangle A spins around world Z at a height of 0.5, and triangle B stands upright (rotated 90° around X, centred at height 0.3), so it cuts through A and also through the floor plane. The animated angle comes from elapsed frame time, so the rotation speed does not depend on the frame rate.

### Depth testing

The renderer enables depth testing (`GL_LESS`) and clears both the color and depth buffers every frame. The depth buffer stores how far each fragment is from the camera, so overlapping surfaces are resolved per pixel rather than by draw order. This is what lets the two triangles visibly cut through each other.

### Resize-safe perspective

The aspect ratio is recalculated from the current width and height every frame. `GLRenderer::render()` returns immediately when either dimension is not positive, which avoids a divide by zero, and the main loop does not render while the window is minimized.

## 7. Step 5 — Orbit camera

The camera is a pure-math type in `gfx_core` (`orbit_camera.h`), not part of the OpenGL renderer or the Win32 platform layer. It stores four values:

| State | Meaning |
|---|---|
| `target` | World-space point the camera looks at and orbits around |
| `distance` | Distance from the eye to the target |
| `yaw` | Horizontal angle around world Z |
| `pitch` | Elevation angle above or below the XY plane |

The eye position is derived from spherical coordinates and added to `target`:

```
eye = target + distance * ( cos(pitch)*cos(yaw),
                            cos(pitch)*sin(yaw),
                            sin(pitch) )
```

Because the eye is recomputed from the four values each time, the camera's distance stays exact while orbiting. Rotating an eye vector incrementally would accumulate numerical drift.

The view matrix is `look_at(eye(), target, up)` with world up set to +Z.

### Orbit

Middle-mouse dragging changes yaw and pitch from mouse pixel deltas. Pitch is clamped to ±(π/2 − 0.0175) radians. At the exact poles the forward direction would be parallel to the up vector, and the cross product used to build the view basis would be zero.

Mouse deltas are displacements, so they are not multiplied by frame delta time. Doing so would make camera movement depend on the frame rate.

### Pan

Shift + middle-drag moves the target along the camera's right and up directions, computed from the current orientation, so panning follows the screen even after the camera has orbited. The scale in world units per pixel is:

```
s = 2 * distance * tan(fov_y / 2) / viewport_height
```

At the target's distance, the view frustum is `2 * distance * tan(fov_y / 2)` world units tall and spans `viewport_height` pixels, so this scale makes the point under the cursor follow the cursor. A non-positive viewport height returns zero. Horizontal motion moves the target opposite to the camera's right vector, which gives the "grab the scene" feel.

### Zoom

Scroll-wheel input changes the distance multiplicatively:

```cpp
distance *= std::pow(0.9f, notches);
```

Each positive notch reduces the distance by 10%. The distance is clamped to [0.2, 500] so the camera cannot reach the target or move arbitrarily far away.

### Input mapping

The app translates platform input into camera calls: middle drag → `orbit`, Shift + middle drag → `pan`, scroll → `zoom`. The camera class does not include or depend on `InputState`.

## 8. Step 6 — The floor, ray-cast

### Why not draw a big quad

A floor mesh has an edge. A quad large enough to look infinite breaks down at the far plane and loses precision at grazing angles. Instead the grid is computed per pixel: for every pixel, the shader casts a ray from the eye and intersects it with the plane `z = 0`. Every pixel that hits the plane gets a color computed from where it hit. There are no floor vertices and no vertex buffer.

### Fullscreen triangle

`GridRenderer` draws one triangle with `glDrawArrays(GL_TRIANGLES, 0, 3)` and no buffers. The vertex shader derives the corner from `gl_VertexID`:

```glsl
vec2 ndc = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2) * 2.0 - 1.0;
```

IDs 0, 1, 2 give (-1,-1), (3,-1), (-1,3), a triangle that covers the whole screen with the excess clipped. A core-profile context still requires a VAO to be bound for any draw, so the renderer creates an empty one.

### The ray for each pixel

For a point `ndc = (x, y)` on the screen, the view-space direction is `(x * tan_half * aspect, y * tan_half, -1)`, where `tan_half = tan(fov_y / 2)`. In world space that becomes:

```
ray = forward + x * (right * tan_half * aspect) + y * (up * tan_half)
```

`GLRenderer` builds the scaled `right` and `up` vectors on the CPU from `FrameData::basis` and passes them as uniforms. The vertex shader evaluates the formula at the three corners, and the rasterizer interpolates `v_ray` across the screen. The triangle's `w` is 1 at every vertex, so interpolation is exactly linear in screen space and no perspective correction is involved. `v_ray` is not normalized, which is why the distance later uses `length(v_ray)`.

`test_ray_reconstruction` in `test_camera` checks this formula: it projects a known world point to NDC, rebuilds the ray from the camera basis, and verifies the ray points at the original point.

### Intersecting the plane

The ray is `p = eye + t * ray`. Setting `p.z = 0`:

```
t = -eye.z / ray.z
```

The hit is valid only if `t > 0` (the plane is in front of the camera, not behind it) and `|ray.z|` is not near zero (the ray is not parallel to the floor, as at the horizon). Pixels with an invalid hit are discarded.

### Writing correct depth

The grid has no vertices, so the hardware has no depth for it. The fragment shader projects the hit point with `u_view_proj`, divides by `w`, and writes `gl_FragDepth = ndc_z * 0.5 + 0.5` (the default depth range is [0, 1]). This makes the grid interleave correctly with the demo triangles: where triangle B passes through the floor, grid lines are hidden by the part of the triangle in front of the floor and drawn over the part behind it. A hit outside NDC depth [-1, 1] is discarded, which applies the near and far planes manually because the fullscreen triangle itself is never clipped.

Writing `gl_FragDepth` disables the early depth test for the grid's draw. This is acceptable for one fullscreen pass, and is a known cost to keep in mind.

## 9. Step 7 — Grid lines, axes, and anti-aliasing

### Lines from a distance field

The fragment shader works in floor coordinates `p.xy`. For a grid with spacing `s`:

```glsl
vec2 c = p / s;                      // position in grid cells
vec2 w = max(fwidth(c), 1e-6);       // cells per pixel on screen
vec2 dist_px = abs(fract(c - 0.5) - 0.5) / w;   // pixels to the nearest line
```

`fract(c - 0.5) - 0.5` is the signed distance to the nearest integer, so lines sit at whole numbers of cells. `fwidth` is how much `c` changes between neighbouring pixels, which converts that distance into pixels. Line width is therefore constant on screen at any zoom level. Coverage is a simple one-pixel ramp:

```glsl
float aa_line(float dist_px, float width_px) {
    return clamp(0.5 * width_px + 0.5 - dist_px, 0.0, 1.0);
}
```

### Two levels and the axes

| Layer | Spacing | Width | Color | Alpha |
|---|---|---|---|---|
| Minor | 1 | 1 px | grey 0.55 | 0.10 |
| Major | 10 | 1 px | grey 0.65 | 0.24 |
| X axis (the line `y = 0`) | — | 1.25 px | red (0.85, 0.25, 0.30) | 0.75 |
| Y axis (the line `x = 0`) | — | 1.25 px | green (0.45, 0.78, 0.25) | 0.75 |

The values were tuned against Blender's viewport. Thin, dim lines look right; thicker or brighter ones were visibly heavier than Blender's. The layers are composited in the shader with the standard "over" operator, so an axis lying on a grid line blends correctly. The result is a single RGBA color per pixel.

### Anti-moiré

When a cell is only a few pixels wide, line coverage aliases into moiré patterns. Each grid level therefore fades out by its own on-screen density:

```glsl
vec2 lod = smoothstep(vec2(2.0), vec2(8.0), 1.0 / w);
```

`1.0 / w` is pixels per cell. Below about 2 pixels the level is invisible, and above 8 it is fully visible. This is only an anti-aliasing guard. How the grid fades into the distance is handled separately.

### Distance fade

Without a fade, the grid shimmers into a solid band near the horizon. The shader fades by actual distance from the eye:

```glsl
float dist = t * length(v_ray);
c.a *= 1.0 - smoothstep(0.2 * u_fade_end, u_fade_end, dist);
```

The fade starts gradually at 20% of `u_fade_end` and ends at `u_fade_end`. The app derives it from the zoom so the visible patch of grid scales with the camera:

```cpp
grid_fade_end = min(1.5 * max(camera.distance, 10.0), 0.9 * far_plane);
```

The `0.9 * far_plane` cap guarantees the grid is gone before the far plane could cut it off with a visible edge. Early versions used a fade based only on line density, which left a visible horizon line when zoomed out. The distance fade, together with thinner and dimmer lines, is what made the result match Blender.

### Blending and depth mask

The grid is drawn after the opaque demo geometry, with:

```cpp
glEnable(GL_BLEND);
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
glDepthMask(GL_FALSE);
// draw
glDepthMask(GL_TRUE);
glDisable(GL_BLEND);
```

- Depth **testing** stays on, so geometry in front of the floor hides it.
- Depth **writing** is off, so the translucent floor never hides anything drawn after it.
- The depth mask must be restored afterwards. If it were left disabled, `glClear(GL_DEPTH_BUFFER_BIT)` on the next frame would not clear the depth buffer, and the scene would break.

### Shader rules that matter

- `fwidth` needs derivatives from neighbouring pixels, so every derivative is computed **before** any `discard`. Computing it after a divergent `discard` would give undefined results.
- A `discard` for low alpha (`c.a < 0.004`) skips work for pixels with nothing to show.

## 10. Step 8 — The renderer interface

### Why an interface now

Vulkan (M4) and the CUDA rasterizer (M5) must draw the same scene. If `main.cpp` called OpenGL directly, each new backend would mean rewriting the app. Step 8 puts one narrow seam in between.

```cpp
class IRenderer {
public:
    virtual ~IRenderer() = default;
    virtual const char* name() const = 0;
    virtual bool init(void* native_window) = 0;
    virtual void render(const FrameData& frame) = 0;
    virtual void set_vsync(bool enabled) = 0;
};

std::unique_ptr<IRenderer> create_renderer(RendererBackend backend);
```

`create_renderer` returns `nullptr` and logs an error when the requested backend was not compiled in. It is the only function that contains `#if defined(GFX_HAS_*)`.

### FrameData

`FrameData` is the backend-neutral description of one frame:

| Field | Purpose |
|---|---|
| `width`, `height` | framebuffer size in pixels |
| `view`, `eye`, `basis` | camera: world-to-view matrix, eye position, right/up/forward vectors |
| `fov_y`, `near_plane`, `far_plane` | projection parameters (radians for the FOV) |
| `grid_fade_end` | world distance at which the grid is fully faded |
| `demo_spin_angle` | temporary, drives the demo triangles; replaced by a draw list in M3 |

`FrameData` deliberately carries **no projection matrix**. OpenGL and Vulkan differ in NDC depth range and Y direction, so each backend builds its own with `perspective(..., ClipSpace)`. `GLRenderer` uses `CLIP_SPACE_OPENGL`, and the Vulkan backend will use `CLIP_SPACE_VULKAN`.

### GLRenderer

`GLRenderer` is a `final` class implementing `IRenderer`. It owns, in this order:

```cpp
GLContext m_context;
TriangleDemo m_triangle;
GridRenderer m_grid;
```

Members are destroyed in reverse order, so the grid and triangle delete their GL objects while the context still exists, and the context goes last. The order is intentional and commented in the header.

`render()` does, in order: return early if the size is not positive; build the projection and `view_proj`; set the viewport, clear color and depth state; clear; draw the triangles; build `GridParams` (scaled `right` and `up` as described in Step 6) and draw the grid; swap buffers.

### Build wiring

`gfx_renderer` is a static library containing the factory. It links `gfx_core` and `gfx_config`, plus `gfx_gl` when that target exists. The app links `gfx_renderer` only, and so never includes an OpenGL header.

## 11. Step 9 — Rendering during drag and resize

### The problem

When the user presses on the title bar or a window edge, Windows' default handling enters a modal loop inside `DispatchMessage`. The call does not return until the mouse is released, so `poll_events()` stalls, nothing is drawn, and the window freezes visually while it is being dragged or resized. This was listed as a known limitation of M1.

### The fix

Windows still delivers some messages to the window procedure from inside that loop. The window layer uses them to run a render callback:

| Message | Action |
|---|---|
| `WM_ENTERSIZEMOVE` | set `in_modal = true`; start a 1 ms timer (`SetTimer`) |
| `WM_TIMER` | if the timer id matches and `in_modal` is set, call the callback |
| `WM_SIZE` | after updating the stored size (not minimized), call the callback if `in_modal`, so a resize redraws immediately at the new size |
| `WM_EXITSIZEMOVE` | `KillTimer`; `in_modal = false`; `input.release_all()` |
| `WM_ERASEBKGND` | return 1, so Windows does not paint the background over the GL surface |

The callback is registered with `Window::set_modal_frame_callback(std::function<void()>)`. `main.cpp` registers the same `draw_frame` lambda that the normal loop calls, so both paths render identically: tick the timer, advance the animation, fill `FrameData`, call `render`, update the title.

### Details

- The 1 ms request is a ceiling, not a rate. Windows' timer is coarse (on the order of 10–16 ms), so the scene redraws at roughly that cadence during a drag. `WM_SIZE` triggers an extra redraw per resize step.
- `release_all()` runs on exit because the modal loop consumes the mouse messages. A button that was down when the drag began would otherwise stay "down" in `InputState`.
- Input and camera updates remain in the main loop. During a drag the scene keeps animating but the camera does not move, which is correct because the mouse is dragging the window.
- Because `draw_frame` also ticks the frame timer, there is no multi-second gap after a drag, so the FPS readout no longer shows a single low value afterwards.
- The lambda captures locals by reference. `main` clears the callback (`set_modal_frame_callback({})`) before shutdown so the window can never call into destroyed objects.
- `WM_ERASEBKGND` returning 1 removes the flicker caused by the system clearing the client area on every resize step.

## 12. Tests and correctness checks

Math errors are cheaper to find with a test than by looking at the screen. `test_camera` covers the orbit camera and the ray setup that Steps 5 and 6 depend on:

| # | Case |
|---|---|
| 1 | The distance between `eye()` and `target` equals `distance` |
| 2 | Orbiting changes the angles but preserves the distance |
| 3 | Pitch stays strictly inside the poles, and the view matrix has no NaN values |
| 4 | The target maps to approximately `(0, 0, -distance)` in view space |
| 5 | Zoom scales the distance and respects its bounds |
| 6 | Panning moves the target perpendicular to the view direction and keeps the distance |
| 7 | Horizontal panning moves the target opposite to the camera's right vector |
| 8 | `basis()` returns unit-length, mutually orthogonal vectors, `right` is horizontal, `forward` points at the target, and `right × up = -forward` (right-handed) |
| 9 | Reconstructing a pixel's ray from the basis and FOV points at the original world point |

`test_math` additionally covers `to_radians`. All tests use the `CHECK` macro from `tests/check.h`, so they also run in Release builds, and each program exits with a non-zero code on failure so CTest sees it.

The renderer, the shaders and the Win32 modal loop are not unit-tested because they need a window and a GPU. They are verified with the manual checklist below.

## 13. Frame-loop sequence

Each iteration of the main loop:

1. `window.poll_events()`.
2. If minimized: log once, `wait_events()`, `timer.reset()`, and skip rendering until restored.
3. Read the input and update the orbit camera (orbit, pan, zoom).
4. If the window was resized, log the new size and aspect ratio.
5. Handle Escape and log key edges.
6. `draw_frame()`:
   - `timer.tick()` and advance the demo angle by the delta time;
   - fill `FrameData` from the camera and the constants (`kFovY` = 60°, near 0.1, far 1000, grid fade);
   - `renderer->render(frame)`, which builds the projection, clears, draws the triangles and the grid, and swaps;
   - update the window title when the FPS value has just refreshed.

The same `draw_frame` is called by the window's modal-loop callback (Step 9). The field of view is one constant shared by the projection and the pan scale, so camera movement always matches what is visible.

## 14. Known scope and non-goals

M2 is a rendering and camera foundation, not a model viewer yet. It does not include:

- loading model files, a scene graph, materials, lighting or textures (M3);
- object move/rotate/scale (M3);
- a Vulkan renderer (M4) or a CUDA software rasterizer (M5);
- a UI or any text rendering.

Known limitations and temporary pieces:

| Item | Notes |
|---|---|
| **Demo triangles** | `TriangleDemo` and `FrameData::demo_spin_angle` are scaffolding; M3 replaces them with a draw list of meshes |
| **Fixed grid spacing** | Two levels (1 and 10 units); there is no adaptive subdivision when zooming in very far |
| **Grid cost** | Writing `gl_FragDepth` disables early-Z for the grid pass, which is one fullscreen draw |
| **Modal loop cadence** | Roughly the Windows timer resolution, not vsync; the camera does not respond to input during a drag |
| **DPI changes** | `WM_DPICHANGED` is still not handled (see the M1 limitations) |
| **Single backend** | Only OpenGL is implemented behind `IRenderer`; `create_renderer` returns `nullptr` for the others |

## 15. Manual checklist

1. The window opens with a dark background, a grid, a red X axis and a green Y axis. The two colored triangles spin and intersect each other.
2. The log shows `renderer: OpenGL 4.6` and the GPU details. A debug build prints no OpenGL debug-callback errors.
3. Middle-drag orbits, Shift + middle-drag pans, and scrolling zooms. The floor never flips at the poles.
4. Zoom far out: the grid fades smoothly and the horizon is not a solid band or a hard edge. Zoom in close: the lines stay one pixel wide.
5. Move the camera below the floor (orbit to a negative pitch): the grid is still visible from underneath.
6. The upright triangle cuts through the floor: the grid lines are hidden where the triangle is in front of the floor, and drawn over it where the triangle is behind.
7. Drag the window by its title bar: the scene keeps animating and does not freeze.
8. Resize by dragging an edge: the image re-renders at the new size, with no stretching or flicker.
9. Minimize and restore: CPU use drops while minimized, and nothing breaks afterwards.
10. The title shows FPS and frame time, and the FPS settles at the monitor's refresh rate (vsync).
11. Esc exits cleanly.

## 16. Completion checkpoint

The expected state at the end of M2:

- the application creates an OpenGL 4.6 core context and loads its functions through its own loader;
- shader-driven geometry renders with perspective projection and depth testing;
- middle-drag orbits, Shift + middle-drag pans, scrolling zooms, and the camera tests pass;
- an infinite floor grid with colored axes is drawn per pixel, anti-aliased, depth-correct, and fades with distance in the way Blender's viewport does;
- `main.cpp` talks only to `IRenderer` and `FrameData`, with the OpenGL backend selected through `create_renderer`;
- the scene keeps rendering while the window is dragged or resized;
- `ctest` passes in Debug and Release.

Next is M3: the OBJ loader, a lit mesh, and object move/rotate/scale, which replaces the demo triangles.