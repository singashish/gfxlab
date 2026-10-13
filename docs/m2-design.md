# M2 — OpenGL Renderer and Orbit Camera

> **Status:** In progress; implementation documented through Step 5  
> **Scope:** Create a native OpenGL 4.6 core context, load modern OpenGL entry points, render a shader-driven triangle, introduce view/projection transforms and depth testing, and add Blender-style orbit-camera controls.

## 1. Purpose

M2 establishes the first real GPU-rendering path in gfxlab. Instead of drawing through a high-level framework, the project creates its own WGL context, loads the OpenGL functions it needs, manages a small set of GPU objects, and draws geometry through shaders.

The milestone also connects the previously separate pieces:

- platform window and input from M1;
- vector, matrix, and camera math from `gfx_core`;
- OpenGL resources and drawing from `gfx_gl`;
- application behavior and frame loop in `main.cpp`.

At the end of Step 5, the demo should show two intersecting, vertex-colored triangles that can be orbited, panned, and zoomed with the mouse.

## 2. Architecture and dependency boundaries

The OpenGL implementation lives in `src/renderer/opengl` and is built as `gfx_gl`.

- `gfx_core` supplies math and reusable logic.
- The platform layer creates the window and supplies its native handle and input state.
- `gfx_gl` creates the OpenGL context, loads entry points, owns OpenGL resources, and issues draw calls.
- `main.cpp` handles user input, updates the camera and animation, prepares view/projection matrices, and calls the renderer.

`gfx_core` must not depend on `InputState`, Win32, or OpenGL. This keeps the orbit camera testable without a window or GPU context.

## 3. Step 1 — Create the OpenGL context

### Why WGL is needed

On Windows, OpenGL rendering requires a rendering context associated with a window's device context. The `GLContext` class owns the WGL context and exposes operations such as creation, destruction, buffer swapping, and vertical-sync configuration.

Modern context creation requires WGL extension functions. Windows' legacy OpenGL exports do not directly expose every modern entry point, so the implementation first creates a temporary dummy window and legacy context. It uses that context to retrieve WGL extension functions, requests a real OpenGL 4.6 core context for the application's window, and then releases the temporary bootstrap resources.

The bootstrap is a setup technique; the dummy window is not the application window and is not used for rendering the demo.

### Context lifetime

The application declares the window before the GL context:

```cpp
gfx::Window window;
gfx::GLContext gl;
```

C++ destroys local objects in reverse declaration order, so the GL context is destroyed before the window. This is important because the context depends on the window's native resources.

The context setup also configures swap interval when the relevant extension is available and logs useful driver information such as vendor, renderer, OpenGL version, and GLSL version. A debug context is requested in debug builds when supported by the setup.

## 4. Step 2 — Load OpenGL entry points explicitly

The project uses its own small OpenGL loader rather than an external loader library.

The function list is centralized in `gl_functions.h` using an X-macro list. The same list drives function-pointer declarations and loading, reducing the chance that a declaration is added in one place but forgotten in another.

`gl_api.h` provides the OpenGL types, constants, and function-pointer declarations used by the renderer. The implementation avoids including the legacy `<GL/gl.h>` after defining its own API declarations, preventing conflicting declarations.

### Loading sequence

1. Create the real OpenGL context.
2. Make that context current on the calling thread.
3. Load required OpenGL functions.
4. Fail context initialization if required entry points cannot be loaded.
5. Query driver information and configure debug output.

The loader first tries `wglGetProcAddress`. Some functions may instead be exported directly by `opengl32.dll`, so it falls back to `GetProcAddress` for those cases. WGL can return special invalid sentinel values, which must be treated as failures rather than valid function pointers.

The debug-message callback provides driver diagnostics in debug builds. It helps identify invalid API usage without requiring every error to be discovered visually.

## 5. Step 3 — Draw a shader-driven triangle

The first draw demo is intentionally small. It owns:

- a shader program;
- a vertex array object (VAO);
- a vertex buffer object (VBO).

The vertex data stores each vertex's position and color. The vertex shader passes position through the MVP transform (added in Step 4) and forwards color; the fragment shader writes the interpolated color to the output pixel.

### What the GPU objects do

- **VBO:** stores the vertex data in GPU-accessible memory.
- **VAO:** records how vertex attributes are interpreted and where their data comes from.
- **Shader program:** tells the GPU how to process vertices and produce fragment colors.

The demo uses vertex attributes for position and color, then draws three vertices as `GL_TRIANGLES`. The different vertex colors blend across the triangle because the rasterizer interpolates the varying color values across its surface.

Shader compilation and program linking are checked, with diagnostic logs retrieved when a shader or program fails.

## 6. Step 4 — Model, view, projection, and depth

Step 4 moves the triangle from a simple screen-space demonstration into a 3D camera pipeline.

The combined transform is:

\[
\text{MVP} = \text{Projection} \times \text{View} \times \text{Model}
\]

Each matrix has one job:

- **Model:** places or rotates one object in world space.
- **View:** expresses the world from the camera's point of view.
- **Projection:** maps the camera view into clip space using a perspective field of view and aspect ratio.

The vertex shader multiplies each input position by the MVP matrix. The project uses column-major `Mat4` storage and uploads it with `glUniformMatrix4fv(..., GL_FALSE, ...)`.

### Two intersecting triangles

The demo draws two instances of the same triangle geometry with different model transforms:

- one rotates around world Z;
- the other is rotated around X to occupy a different plane.

The first triangle's angle is updated using elapsed frame time. This is frame-rate-independent animation: the rotation advances by time elapsed, not by a fixed amount per rendered frame.

### Depth testing

The renderer enables depth testing and clears both the color buffer and depth buffer for each frame. The depth buffer stores how far a fragment is from the camera. When surfaces overlap on screen, depth testing helps determine which fragment is visible rather than simply letting the last submitted triangle cover the earlier one.

### Resize-safe perspective

The aspect ratio is recalculated from the current window dimensions and passed into the perspective matrix. Rendering is skipped while minimized, and dimensions should be checked before division to avoid a zero-height viewport.

## 7. Step 5 — Orbit camera

The camera is a pure-math type in `gfx_core`, not part of the OpenGL renderer or Win32 platform layer.

It stores four values:

| State | Meaning |
|---|---|
| `target` | World-space point the camera looks at and orbits around |
| `distance` | Distance from the eye to the target |
| `yaw` | Horizontal angle around world Z |
| `pitch` | Elevation angle above or below the XY plane |

The eye position is derived from the spherical coordinates:

\[
\begin{aligned}
x &= d\cos(p)\cos(y)\\
y &= d\cos(p)\sin(y)\\
z &= d\sin(p)
\end{aligned}
\]

The resulting offset is added to `target`. This representation keeps the camera's distance stable while orbiting and avoids repeatedly rotating an eye vector, which can accumulate numerical drift.

The view matrix is produced by looking from `eye()` toward `target` with world up set to positive Z.

### Orbit

Middle-mouse dragging changes yaw and pitch based on mouse pixel deltas. Pitch is clamped just inside ±90 degrees. At the exact poles, the camera's forward direction would become parallel to the up vector, making the cross product used to construct the view basis degenerate.

Mouse deltas are displacements, so they are not multiplied by frame delta time. Doing so would make camera movement depend on frame rate.

### Pan

Shift + middle-drag moves the target along the camera's right and up directions. Those directions are computed from the current view, so panning follows the screen even after the camera has orbited.

The movement scale is measured in world units per pixel:

\[
s = \frac{2d\tan(f/2)}{H}
\]

where \(d\) is camera distance, \(f\) is vertical field of view, and \(H\) is viewport height in pixels. This scale makes panning respond naturally to both zoom level and window size. A non-positive viewport height returns zero to avoid invalid calculations.

### Zoom

Scroll-wheel input changes distance multiplicatively:

```cpp
distance *= std::pow(0.9f, notches);
```

Each positive notch reduces the distance by 10%. The distance is clamped to a minimum and maximum so the camera cannot pass through the target or move arbitrarily far away.

### Input mapping

The app translates platform input into camera operations:

- middle mouse drag → orbit;
- Shift + middle mouse drag → pan;
- scroll wheel → zoom.

The camera class does not include or depend on `InputState`. This is intentional: the same camera mathematics can be tested independently and reused by another renderer or platform.

## 8. Tests and correctness checks

The camera's math should be tested before relying on visual inspection. Important invariants include:

1. The distance between `eye()` and `target` equals `distance`.
2. Orbiting changes the angles but preserves distance.
3. Pitch remains strictly inside the poles, and the view matrix contains no NaN values.
4. Transforming the target by the view matrix places it at approximately `(0, 0, -distance)`.
5. Zoom scales distance and respects its bounds.
6. Panning moves the target perpendicular to the view direction without changing camera distance.
7. Horizontal panning moves the target opposite the camera's right direction, producing the grab-the-scene feel.

The math test suite should also cover the degree-to-radian helper. These checks catch coordinate and sign errors more cheaply than debugging them by trial and error in the running application.

## 9. Frame-loop sequence

For each visible frame, the application should:

1. Tick the frame timer and poll platform events.
2. Handle minimized state and skip rendering if still minimized.
3. Read current input and update the orbit camera.
4. Update the animated triangle angle using delta time.
5. Calculate the current aspect ratio.
6. Build the view matrix from the orbit camera.
7. Build the perspective projection using the same field of view used by pan scaling.
8. Clear color and depth buffers.
9. Draw the demo and swap buffers.

Keeping the field of view shared between projection and pan scaling prevents the camera's movement scale from disagreeing with the visible perspective.

## 10. Known scope and non-goals

This milestone is a rendering and camera foundation. It should not be described as a complete model viewer yet. In particular, this design document does not claim that the following are complete:

- loading arbitrary 3D model files;
- a scene graph, materials, lighting, or texture system;
- a Vulkan renderer;
- a CUDA software rasterizer;
- a complete editor-style UI.

Those can be layered on after the basic rendering and navigation path is stable.

## 11. Completion checkpoint through Step 5

The expected checkpoint is:

- the application creates an OpenGL 4.6 core context;
- required modern OpenGL functions are loaded after the context is current;
- the shader-driven demo draws two intersecting colored triangles;
- depth testing and perspective projection are active;
- the view responds correctly to window resizing;
- middle-drag orbits, Shift + middle-drag pans, and scrolling zooms;
- camera math tests pass, including pitch limits, target view-space position, pan direction, and zoom limits.

If one of these behaviors is not yet verified in the current build, treat it as a checkpoint to validate rather than as a guaranteed result.
