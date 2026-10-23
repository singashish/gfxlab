# M8 — A Basic User Interface

> **Status:** Complete (Steps 1–5, plus 4b)  
> **Scope:** A small immediate-mode UI written from scratch: a font rasterized by Windows' GDI into an atlas, a backend-neutral list of UI triangles, a UI pass in each of the three backends (CUDA draws it in software), widgets, mouse capture, and a settings panel for the features of M3–M7. The CUDA backend's display code moved behind a presenter interface on the way.

## 1. Purpose

Up to M7 every setting was a keyboard shortcut and every number was in the title bar or the console. That worked, but the list of shortcuts kept growing, and values like the sun's angles or the model's transform could only be changed indirectly.

M8 adds a panel in the window that shows the state and lets the mouse change it. It is built the way the rest of the project is: no UI library. The interesting parts are not the widgets themselves but how text gets onto the GPU, how one UI is drawn identically by OpenGL, Vulkan and a CUDA kernel, and how the UI and the 3D view agree on who owns the mouse.

## 2. What M8 adds

| Area | Files |
|---|---|
| UI library (portable) | `src/ui/` → `gfx_ui`: `font_atlas.*` (atlas packing), `ui_draw.*` (draw list), `ui_raster.*` (software rasterization of the draw list), `ui.*` (immediate-mode widgets) |
| Font rasterization | `src/platform/ui_font.h`, `win32/ui_font_win32.cpp` (GDI, linked with `gdi32`) |
| Frame data | `renderer.h`: `FrameData::ui`, `RendererStats` and `IRenderer::stats()` |
| OpenGL UI pass | `opengl/gl_ui.*` |
| Vulkan UI pass | `vulkan/vk_ui.*`, `shaders/ui.vert`, `ui.frag`; `vk_upload.*` gained image uploads |
| CUDA presenter | `cuda/cuda_presenter.h` (interface), `cuda_gl_presenter.*` (OpenGL interop) |
| CUDA UI pass | `cuda/cuda_ui.h`, `cuda_ui.cpp`, `cuda_ui.cu` |
| Ray statistics for the UI | `cuda/cuda_ray_stats.*` (`latest`), `cuda_renderer.*` (`stats`) |
| App | `app/main.cpp`: the settings panel, mouse ownership, short title |
| Tests | `tests/test_ui.cpp` (new, the 10th CTest program), `tests/test_math.cpp` (`to_degrees`) |

## 3. Step 1 — Draw list, font atlas and the OpenGL UI pass

### The draw list

The UI's output is a `UiDrawList`: vertices and 32-bit indices forming triangles, in window pixels with the origin at the top left and Y down (the same space as the mouse). A vertex is 20 bytes: position, atlas coordinates and a colour packed as R, G, B, A bytes (`ui_rgba`). Every backend draws a list the same way:

```
colour = vertex.rgb,   alpha = vertex.a × atlas(uv)
```

blended "source over" on top of the finished 3D frame. The renderer only sees triangles; it knows nothing about panels or text. The app fills one list per frame and passes it in `FrameData::ui`.

### The font atlas

A GPU draws text as textured rectangles, so every character needs a small image. At start-up `build_ui_font()` asks GDI for each printable ASCII character (32–126) of Segoe UI with `GetGlyphOutlineW(GGO_GRAY8_BITMAP)`, which returns a grey coverage bitmap (65 levels, scaled to 0–255) and the glyph's metrics. Grey coverage is used rather than ClearType, whose coloured sub-pixels would not survive blending over a 3D picture.

`pack_font_atlas()` (portable, tested) places the 95 bitmaps in rows across a 256-pixel-wide single-channel image with a one-pixel gap, rounds the height up to a power of two, and records for each glyph its rectangle in the atlas, where to place it relative to the pen, and how far to advance the pen.

The atlas also holds a small fully covered 3×3 block. A solid rectangle is a quad whose texture coordinates all point at the middle of that block, so rectangles and text use the same shader, one texture and one draw call.

### Text layout and sharpness

`add_text` rounds the start position to whole pixels and emits one quad per visible glyph at its exact pixel size. Every screen pixel then lands on exactly one atlas texel, so nearest filtering reproduces GDI's output without blur. Space advances without a quad, `\n` starts a new line, and characters outside the range draw as `?`.

### DPI

The window is per-monitor DPI aware, so at 150% scaling everything has 1.5× the pixels. The font is created at its size × the monitor's DPI scale, and `FontAtlas::scale` records the factor so the layout scales its paddings the same way. The app uses **12 px at 96 DPI**.

### The OpenGL pass

After the mesh and the grid: depth test off, blending on, the vertices and indices uploaded with `glBufferData(..., GL_STREAM_DRAW)` (replacing the whole buffer lets the driver hand out fresh memory instead of waiting for the previous frame), one `glDrawElements`. The atlas is a `GL_R8` texture uploaded once. The UI is inside the GPU timer, because it is real work in the frame.

## 4. Step 2 — The Vulkan UI pass

The data is the same; what OpenGL's driver did implicitly is now explicit.

- **Atlas image.** An `R8_UNORM` image in device-local memory, filled through a staging buffer. `VulkanUploader::upload_to_image()` records the layout transitions `UNDEFINED → TRANSFER_DST → SHADER_READ_ONLY` around the copy. Its submit-and-wait part became a shared helper, `submit_and_wait()`, also used by buffer uploads.
- **Descriptor set.** One combined image sampler (nearest, clamp-to-edge) that the fragment shader reads as set 0, binding 0.
- **Vertex buffers per frame in flight.** The CPU records frame N+1 while the GPU may still read frame N's vertices, so each frame slot has its own host-visible vertex and index buffers (64 KB, doubled when needed). A slot is rewritten only after the renderer has waited for its fence.
- **Pipeline.** `UiVertex` layout (two `R32G32_SFLOAT`, one `R8G8B8A8_UNORM`), blending on, no depth test or write. It still names the depth format, because it draws inside the scene's dynamic rendering pass, which has a depth attachment. The framebuffer size is an 8-byte push constant.
- **No Y flip.** Vulkan's NDC has Y down like the UI, so `ui.vert` maps pixels to NDC directly; the OpenGL shader flips Y.

## 5. Step 3 — The CUDA presenter interface and the CUDA UI pass

### The presenter

`CudaRenderer` used to own an OpenGL context, a texture and the CUDA/OpenGL interop object. It now owns an `ICudaPresenter` with five operations: set up the display for the window (which also selects the CUDA device that can share memory with it), resize, copy the finished frame, present, and set vsync. `CudaGLPresenter` holds the previous interop code unchanged. `create_cuda_presenter()` chooses the presenter, so M10 can add a DX12 presenter without touching the renderer.

### Why the CUDA UI is drawn in CUDA

Drawing the UI with OpenGL after the interop copy would tie the UI to the OpenGL presenter. Instead the UI is composited into the CUDA colour buffer before the presenter sees the frame, so any presenter shows a finished picture. It also keeps the CUDA backend doing all of its own drawing.

### The software UI rasterizer (`ui/ui_raster.h`)

- **CPU setup** (`build_ui_raster_tris`): each triangle is snapped to 1/16 pixel and put in positive-area order with `setup_raster_triangle` from M5. The setup also records which edges are top or left edges and the triangle's pixel bounds clamped to the screen. It returns the rectangle containing the whole UI.
- **Per pixel** (`ui_blend_triangle`, `GFX_HD`): evaluate the three edge functions at the pixel centre; if inside, turn them into barycentric weights, interpolate the atlas coordinates and the colour, read the nearest texel, and blend.
- **Matching the hardware backends.** A quad's two triangles share a diagonal. Without the top-left rule a pixel centre exactly on it would be blended twice, which shows as a darker line on a half-transparent panel. The blend is also rounded to 8 bits after every triangle, as an RGBA8 framebuffer does.

### The kernel and the upload

One thread per pixel of the UI rectangle. Every pixel walks all triangles **in order**, because the panel background must be blended before its text. One thread per triangle would let two triangles blend into a pixel in random order. Each 16×16 block loads 256 triangles at a time into shared memory, so a block fetches each triangle once rather than once per thread.

The triangles travel through pinned host memory with `cudaMemcpyAsync`, on the same stream as the scene kernels, so the copy and the kernel run after the scene without the CPU waiting. Two pinned slots, each with an event, make sure the CPU never overwrites data a queued copy has not read yet.

The CUDA frame is now: clear → raster or ray trace → grid → **UI** → presenter copy (inside the GPU timer) → present.

## 6. Step 4 — Widgets and mouse ownership

### Immediate mode

There are no widget objects. Every frame the app describes the whole UI with function calls, and each call draws its widget and returns what the user did:

```cpp
ui.begin_panel("gfxlab", 10.0f, 10.0f, 280.0f);
if (ui.checkbox("Wireframe", &wireframe)) { /* it just changed */ }
ui.slider("Sun azimuth", &sun_azimuth, 0.0f, 360.0f, "%.0f deg");
ui.end_panel();
```

The app's variables are the state, so the UI cannot disagree with the keyboard shortcuts: press **W** and the checkbox shows it on the next frame because it reads `wireframe`.

### What `gfx::Ui` remembers

- **The held widget.** A widget's ID is a hash of its panel's title and its label (labels must be unique within a panel). A left press over a widget makes it **active**. While active it keeps receiving the mouse even outside its rectangle, which lets a slider follow a drag past its end, and other widgets ignore the mouse. Release clears it, and so does a frame in which the active widget is not drawn.
- **`wants_mouse()`.** True when the mouse is over a panel or a widget is held. A press on a panel's empty background also holds the mouse, so a drag that starts on the panel never becomes a camera move.

### Widgets

| Widget | Behaviour |
|---|---|
| `text` | One line |
| `separator` | Thin line |
| `button` | Clicked when the press and the release are both over it |
| `checkbox` | A button that flips a `bool` |
| `slider` | While held, the value follows the mouse's position along the track, between `min` and `max` |
| `set_enabled(false)` | The following widgets are drawn at about 40% opacity and ignore the mouse |

### Panels

The background must be drawn under the widgets, so it must come first in the list, but its height is known only after them. `begin_panel` adds **placeholder rectangles** (background and four border edges) with zero area, which no backend draws; `end_panel` sets their corners. A panel grows to fit wider text and remembers its final width, and the next frame lays the widgets out with that width.

### Who gets the mouse

The UI is built at the end of the frame, so the 3D-view code uses the previous frame's `wants_mouse()`. One frame of delay cannot be noticed.

- Middle-button orbit and pan run only while `camera_dragging` is set, which a middle **press outside the UI** sets. A drag that started in the 3D view continues over the panel.
- Scrolling zooms only when the mouse is not over the UI.
- The right-button sun drag starts only outside the UI.
- During a G/R/S edit the UI receives no mouse input at all, so clicks always confirm or cancel the edit.

### Step 4b — The camera is locked during G/R/S

G/R/S turns mouse movement into a transform using the current view. If the camera moved during the edit, the same mouse position would suddenly mean a different transform and the object would jump. A middle press during an edit no longer starts a drag, a drag in progress stops when an edit starts, and when the edit ends the camera waits for a new middle press. Scroll zoom stays available.

## 7. Step 5 — The settings panel

The panel has four foldable sections. `Ui::header` draws a full-width row with an arrow (the new `UiDrawList::add_triangle`) and flips a `bool` the app owns. A folded section's widgets are simply not called that frame. Headers stay clickable after `set_enabled(false)`.

| Section | Contents |
|---|---|
| (top) | Backend name; FPS, frame ms and GPU ms, refreshed with the title |
| View | Wireframe, Vsync, Reset camera |
| Model | File name and triangle count; Position, Rotation (degrees) and Scale as drag fields; Reset transform; the active edit (`Editing: GRAB X`) |
| Ray tracing | Ray tracing, Sun, Sun azimuth and elevation sliders, Reset sun, Ambient occlusion, rays per frame, AO samples or "AO converged". Dimmed with "Needs --backend=cuda" in OpenGL and Vulkan |
| Controls | The keyboard and mouse shortcuts (folded at start) |

- **Drag fields** (`Ui::drag_float3`): three values side by side with red, green and blue strips for X, Y and Z. While held, every frame adds `(mouse x now − mouse x last frame) × speed`, so the value changes with movement, not position, and has no limits. Speeds: Position 0.5% of the model's half-diagonal per pixel, Rotation 0.5° per pixel, Scale 0.005 per pixel.
- **One owner for the transform.** Each frame the panel copies `model_transform` into the fields and writes it back only when a field reports a change. G/R/S, the fields and Reset therefore never disagree, and the fields follow a G/R/S edit live.
- **Statistics from the renderer.** `IRenderer::stats()` returns a `RendererStats`; the default has nothing. The CUDA renderer fills in whether the last frame was ray traced, the rays per frame from the last once-per-second report of `CudaRayStats`, and the AO sample count. No extra GPU work or waiting is involved.
- **Resets.** Reset camera rebuilds the default orbit camera and frames the model, using the same code as start-up. Reset transform restores the transform taken after loading.
- **Title.** Now only `gfxlab | <backend> | <FPS> FPS`; the details are in the panel. The once-per-second log lines are unchanged.

## 8. Measurements

Release, vsync off, `suzanne_s.obj`, startup camera, measured after Step 3 with the information panel shown (**H** on) and hidden:

| Backend | Frame ms (off → on) | GPU ms (off → on) | Extra per frame |
|---|---|---|---|
| OpenGL | 0.073 → 0.082 | 0.010 → 0.011 | about 9 µs, almost all CPU and driver work |
| Vulkan | 0.183 → 0.184 | 0.013 → 0.014 | about 1 µs |
| CUDA | 0.353 → 0.373 | 0.352 → 0.372 | about 20 µs of GPU time |

The OpenGL change looks like over a thousand FPS (about 13,700 → 12,200), but it is 9 microseconds. When a frame takes a fraction of a millisecond, compare milliseconds, not FPS. CUDA's UI is real GPU work: a copy and a launch with a fixed cost each, plus every UI pixel testing every UI triangle.

## 9. Problems found during M8

| Problem | Cause and fix |
|---|---|
| UI font size | Changed from 15 px to 12 px at 96 DPI after Step 1 |
| Separators and sliders narrower than the panel | Seen in the Step 4 preview: a panel widened by long text kept its widgets at the requested width. Panels now lay out with last frame's final width |
| Panel width would follow the FPS digits | The minimum width (280 px) is wider than the stats line, so the panel keeps its size |
| Camera moved during G/R/S | Middle-button orbit and pan are locked during edits (Step 4b) |
| "UI on/off costs 200–300 FPS" | A reading of FPS at very high frame rates; the cost is microseconds (Section 8) |

## 10. Verification

- `ctest`: 10 programs. `test_ui` covers atlas packing and its error cases, colour packing, rectangles, text layout and fallbacks, the software UI rasterizer (single blending on a quad's diagonal, order, clipping, glyph texels landing 1:1 on pixels), placeholders, triangles, and the widgets: clicks, a click within one frame, releases away from the widget, slider dragging inside and outside the panel, drag fields, headers, disabled widgets, panel capture, a held widget disappearing, and panel sizing. `test_math` checks `to_degrees`.
- Before Steps 4 and 5 were delivered, previews of the panel were rendered on the CPU with `ui_blend_triangle` and a stand-in font.
- On the RTX 5080: the checkpoints of each step and the measurements of Section 8, with no Vulkan validation messages.

## 11. Manual checklist

1. Each backend shows the same panel; text is sharp, with no darker diagonal across the panel in CUDA.
2. **H** hides and shows the panel; the title shows only the backend and the FPS.
3. Sections fold and unfold; Ray tracing is dimmed outside `--backend=cuda` and open by default inside it.
4. Wireframe and Vsync follow both the checkboxes and **W** / **V**; Reset camera returns to the start-up view.
5. Dragging Position, Rotation and Scale moves the model live, also when the mouse leaves the panel; Reset transform restores it; during G/R/S the fields follow the edit and `Editing: ...` appears.
6. In CUDA with **T**: the sun sliders move the shadows, Reset sun returns to 250 / 45, the rays per frame update about once a second, and AO counts to 1024 and then shows "AO converged".
7. Scroll, middle drag and right drag over the panel do not move the camera or the sun; a middle drag started in the view continues over the panel.
8. During G/R/S, middle drag does nothing, clicks over the panel confirm or cancel the edit, and orbiting needs a new middle press after the edit.
9. Resizing and changing vsync keep the panel in place, with no validation errors in Debug Vulkan.

## 12. Known scope and limitations

| Item | Notes |
|---|---|
| **ASCII only** | The atlas holds characters 32–126 |
| **No keyboard input** | Values are changed with the mouse; there are no text fields |
| **Font size** | Made once at start-up for the monitor the window is on; moving to a monitor with a different DPI does not rebuild it |
| **No scrolling** | A panel taller than the window runs past its bottom; fold a section |
| **Fixed placement** | The panel cannot be moved or resized |
| **Left button only** | The UI uses the left button; wheel and other buttons over the panel are ignored |
| **Unique labels** | Widget IDs come from labels, so two widgets with the same label in one panel would share one |
| **CUDA UI cost** | Grows with the UI's area × its triangle count. If it passes about 0.05 ms, sorting triangles into 16×16 tiles first (binning) would cut it several times |
| **One frame of delay** | Mouse ownership uses the previous frame's UI |

## 13. Completion checkpoint

- all three backends draw the same UI from one `UiDrawList`, CUDA in software into its own colour buffer;
- the CUDA backend shows its frames through an `ICudaPresenter`, ready for a DX12 presenter;
- the settings panel shows and changes the view, the model's transform and the ray tracer, with the ray statistics;
- the UI and the 3D view agree on who owns the mouse, and the camera is locked during G/R/S;
- `ctest` shows 10 passing tests.

Next is M9: materials, textures and a CUDA path tracer.
