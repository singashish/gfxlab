# M1 — Window, Input and Frame Timing

> **Status:** Complete; implementation documented through Step 5  
> **Scope:** Create a Win32 window with no third-party library, translate keyboard and mouse messages into a polled input state, handle resize and minimize safely, and add a frame timer with an FPS readout. Nothing is drawn yet.

## 1. Purpose

M1 adds the first thing you can see and touch: a window that reports its size, a keyboard and mouse state tracker, and a frame timer. The window is the surface that the OpenGL and Vulkan backends render into in M2 and M4.

The goals:

- a window built directly on Win32, with no GLFW or SDL;
- an input model that fits a viewport, where the questions are "is the middle button held, and how far did the mouse move this frame?";
- correct behavior for the awkward cases: resize, minimize, Alt keys, losing focus mid-drag, and the cursor leaving the window during a drag;
- as much logic as possible written so it can be unit-tested without Windows.

For the build system, math and logging that M1 builds on, see [`m0-design.md`](m0-design.md). The renderer that uses the window is described in [`m2-design.md`](m2-design.md).

At the end of Step 5, the app opens a 1280x720 window and logs input and resize events, and the title bar shows the FPS and frame time.

## 2. Architecture and dependency boundaries

```
                 app (src/app/main.cpp)
                 loop: tick, poll_events, query input, update title
                    |                 |
          gfx::Window            gfx::FrameTimer
        (src/platform)            (src/core/timer.h)
                 |
        gfx::WindowImpl  ---- owns ---->  gfx::InputState
     (win32/window_win32.cpp)            (src/platform/input.h)
```

Dependencies point downward only: `app` uses `platform` and `core`, and `gfx_platform` links `gfx_core`. The public headers contain **no Windows types**:

- `window.h` hides everything behind a pointer to an implementation struct (the pimpl idiom), so `<windows.h>` and its macros (`min`, `max`, `ERROR`, `near`, `far`) cannot leak into the rest of the code. Only `window_win32.cpp` includes it.
- `WindowImpl` is declared at namespace scope rather than nested inside `Window`, because the free function `wnd_proc` has to name it, and a free function cannot name a private nested type.
- `input.h` and `timer.h` are header-only and platform-independent. That is why `test_input` and `test_timer` link only `gfx_core`, run without a window, and build on any platform.

`gfx_platform` is a static library that exposes `src/` as a public include path and links `gfx_core` publicly. The root `CMakeLists.txt` adds `src/platform` and `src/app` only when `WIN32` is set, and the platform `CMakeLists.txt` still stops configuration with an error if it is added on another host. The macros `WIN32_LEAN_AND_MEAN`, `NOMINMAX`, `UNICODE` and `_UNICODE` are defined for every target on Windows, not just `gfx_platform`, so any future file that includes `windows.h` is safe from its `min` and `max` macros.

## 3. Step 1 — A Win32 window

### Interface

| Member | Meaning |
|---|---|
| `create(WindowDesc)` | Create and show the window. `title` is UTF-8, and `width` and `height` are the **client area** in pixels. Returns false and logs on failure |
| `poll_events()` | Handle all pending messages without blocking, and advance the input frame |
| `wait_events()` | Block until a message arrives, then `poll_events()`. Used while minimized |
| `should_close()` / `request_close()` | Has closing been requested, and request it from code |
| `is_minimized()` | True while the window is minimized |
| `was_resized()` | True for one frame after the client size changed |
| `width()` / `height()` | Current client size in pixels (never 0 while minimized, see Step 4) |
| `native_handle()` | The `HWND` as `void*`, used by the GL context now and the Vulkan surface later |
| `set_title(utf8)` | Change the window title |
| `input()` | Read-only access to the `InputState` (a static empty state if the implementation is missing) |

`Window` is not copyable. Its destructor destroys the native window.

### Why polling, not blocking

`PeekMessage` returns immediately, while `GetMessage` blocks until a message arrives. A renderer must draw every iteration, so the main loop is "handle pending messages, render, repeat", and `poll_events()` uses `PeekMessageW`. Blocking is used only on purpose, in `wait_events()`, when there is nothing to draw.

### Creation sequence

1. **Validate** the description (non-null title, positive size).
2. **Make the process DPI-aware** (once) with `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`. Without it, Windows at 125% or 150% scaling reports virtualized pixel sizes and stretches the output, which is blurry for a renderer. "Access denied" is tolerated, because awareness may already be set by a manifest. This API needs Windows 10 version 1703 or later.
3. **Convert the title** from UTF-8 to UTF-16 with `MultiByteToWideChar`. The code uses the wide (`W`) Win32 functions throughout. Invalid UTF-8 yields an empty title.
4. **Register the window class** once, with the `CS_OWNDC` style (the window keeps one device context for life, which OpenGL context creation relies on), an arrow cursor and a black background brush, so the window has a defined look before a renderer takes over.
5. **Compute the outer size from the client size** with `AdjustWindowRectExForDpi` and the system DPI. The requested 1280x720 is the *renderable* area, and the title bar and borders are added on top. The plain `AdjustWindowRectEx` assumes 96 DPI and would be wrong under scaling.
6. **Create the window** with `CreateWindowExW`, passing the `WindowImpl` pointer as the creation parameter.
7. **Show it** and log the client size.

### How `wnd_proc` finds its object

The window procedure is a plain C callback with no `this`. The pointer to the `WindowImpl` is passed through `CreateWindowExW`, picked up in `WM_NCCREATE` (the first message sent) and stored in the window's `GWLP_USERDATA` slot. Every later call reads it back from there. `WM_NCDESTROY` clears the slot and the stored handle.

`WM_CLOSE` sets a flag and returns. The window is **not** destroyed there, because the app decides when to quit (and could later ask "save changes?"). The destructor does the destruction. `poll_events()` also treats `WM_QUIT` as a close request.

## 4. Step 2 — `InputState`

### Model

Input is **polled state**, not an event queue. The platform layer translates Win32 messages into calls on `InputState`, and the application reads it once per frame. That fits a viewport, where the questions are state questions.

The frame contract is owned by `Window::poll_events()`, so the app cannot get it wrong:

```
begin_frame()  ->  platform feeds events  ->  end_frame()  ->  app queries
```

There are three kinds of data:

| Kind | Examples | Lifetime |
|---|---|---|
| Held state | `key_down`, `mouse_down` | Until released |
| Edge state | `key_pressed`, `key_released`, `mouse_pressed`, `mouse_released` | One frame; cleared by `begin_frame()` |
| Per-frame motion | `mouse_dx`, `mouse_dy`, `scroll` | One frame; the delta is computed in `end_frame()`, and scroll accumulates during the frame and is cleared by `begin_frame()` |

Mouse coordinates are client pixels with the origin at the **top left** and Y increasing **downward** (the Windows convention). OpenGL window coordinates start at the bottom left, so convert at the point of use. Mouse deltas are in pixels per frame, not per second, so they must not be multiplied by delta time.

### Rules, each pinned by a test

| Situation | Behavior |
|---|---|
| Key goes down while up | `down` and `pressed` become true |
| Key goes down while **already down** | Ignored. This is OS **auto-repeat**, and it must not count as a new press |
| Key goes up while down | `down` false, `released` true |
| Press and release in one frame | Both `pressed` and `released` are true, `down` is false |
| First mouse position ever seen | Produces a delta of 0, not a jump from (0, 0) |
| A frame ends before any position has arrived (at startup, or after focus loss) | Delta is 0 and **no baseline is recorded**, so the default (0, 0) is never mistaken for a real position |
| Scroll events in one frame | Add up |
| Focus lost (`release_all`) | Every held key and button is released through the normal path, so each reports `released`; the mouse baseline is discarded |
| Out-of-range key or button index | Ignored on write, false on read |

The mouse-baseline rule has a subtlety. Clearing only the "has previous position" flag is not enough: the same frame's `end_frame()` would immediately store the stale cursor position as a new baseline, and the first move after focus returns would produce a large spurious delta. So `release_all()` also clears the "has a position" flag, and `end_frame()` waits for a **fresh** position before recording a baseline. Test case 9 in `test_input.cpp` covers exactly this sequence.

### Key mapping

`Key` and `MouseButton` are `enum class` types with a `GFX_` prefix. The keys are A to Z, 0 to 9, numpad 0 to 9, Escape, Space, Tab, Delete, the four arrows, Shift, Ctrl and Alt, plus `GFX_KEY_UNKNOWN` and `GFX_KEY_COUNT`.

Letters, digits and numpad digits are kept **contiguous and in order** in the enum. Win32 virtual-key codes for those ranges are also sequential, so translating them is arithmetic (`GFX_KEY_A + (vk - 'A')`), and a `switch` is needed only for the remaining keys.

## 5. Step 3 — Wire Win32 messages into `InputState`

### Message handling

| Message | Effect |
|---|---|
| `WM_KEYDOWN`, `WM_KEYUP` | Translate the virtual key and update `InputState`; handled |
| `WM_SYSKEYDOWN`, `WM_SYSKEYUP` | Same, then fall through to `DefWindowProc` so Alt+F4 still closes the window |
| `WM_SYSCOMMAND` with `SC_KEYMENU` | Swallowed (see below) |
| `WM_MOUSEMOVE` | Store the cursor position |
| `WM_LBUTTONDOWN`, `WM_MBUTTONDOWN`, `WM_RBUTTONDOWN` | Press the button, and start mouse capture if no button was already down |
| `WM_LBUTTONUP`, `WM_MBUTTONUP`, `WM_RBUTTONUP` | Release the button, and end capture when none are left down |
| `WM_MOUSEWHEEL` | Add `delta / WHEEL_DELTA` to the scroll |
| `WM_KILLFOCUS` | Release capture and call `release_all()`; then fall through |

### Mouse capture

The orbit camera drags with the middle button held, and the cursor will often leave the window mid-drag. Normally Windows then stops sending mouse messages, including the button-up, which would leave the state stuck at "down".

- When a button goes down and none was already down, the window calls `SetCapture`.
- When a button goes up and none are left down (and the capture is still ours), it calls `ReleaseCapture`.

With capture, the window keeps receiving `WM_MOUSEMOVE` and the final button-up wherever the cursor is. Mouse coordinates are read with `GET_X_LPARAM` and `GET_Y_LPARAM`, which are **signed**. `LOWORD` would turn a negative coordinate (outside the window, during capture) into a large positive one.

### Mouse wheel

The wheel delta is a multiple of `WHEEL_DELTA` (120). High-resolution mice and touchpads send fractions, which is why scroll is a `float`. Positive means away from the user. `WM_MOUSEWHEEL` carries *screen* coordinates, unlike the other mouse messages, so its position is deliberately unused.

### Alt and system keys

Windows sends `WM_SYSKEYDOWN` and `WM_SYSKEYUP` instead of the plain messages whenever Alt is involved. They are mapped into `InputState` like the plain ones, so Alt (and Alt-modified keys, which a Blender-style viewport uses) work. They then fall through to `DefWindowProc`, which keeps Alt+F4 working.

The default handler would also turn a tap of Alt, F10 or Alt+Space into menu mode, which freezes the render loop until the user clicks. The window therefore swallows `WM_SYSCOMMAND` with `SC_KEYMENU`. The cost is that Alt+Space no longer opens the system menu.

### Focus loss

`WM_KILLFOCUS` releases capture and calls `release_all()`. Without it, pressing Alt+Tab while holding a key or button would never deliver the release, and the orbit camera would stay "dragging" forever.

## 6. Step 4 — Resize and minimize handling

A minimized window reports a client size of **0x0**. If that were stored, it would flow into everything downstream: `perspective()` computes `aspect = width / height`, so a zero height gives infinity or NaN and the whole projection matrix is destroyed. An OpenGL viewport of 0x0 draws nothing, and Vulkan refuses to create a swapchain with a zero extent.

### `WM_SIZE` rules

- If the window is minimized, set the minimized flag and leave the stored size untouched.
- Otherwise, clear the flag, and if the new client size **differs** from the stored one, store it and set the resized flag.
- A zero size never overwrites the last valid size, and "minimized" is exposed as its own fact.
- `was_resized()` fires only when the size actually changes. Restoring from a minimize reports the same size as before and does not count, because recreating a swapchain or framebuffer for no reason would be wasted work.
- `resized` is cleared at the start of `poll_events()`, so it is a one-frame signal like the input edges.

### Idling while minimized

While minimized, the app calls `wait_events()`, which uses `WaitMessage()` to sleep the thread until a message arrives. This keeps CPU use near zero instead of spinning a core on a window that draws nothing. A `WM_CLOSE` is a message, so closing a minimized window still wakes the loop.

After waking, a restored window **continues the same iteration** instead of jumping back to the top, so the resize and input events that woke it are not skipped.

## 7. Step 5 — Frame timer and FPS in the title (closes M1)

`gfx::FrameTimer` (`src/core/timer.h`) provides a safe delta time for camera smoothing and animation, and a readable FPS figure.

```cpp
FrameTimer();                         // calls reset()
void   reset();                       // start timing from now; clears everything
void   tick();                        // measure real elapsed time, then advance()
void   advance(double raw_seconds);   // all of the logic
double delta_seconds() const;         // clamped to [0, 0.25]
double fps() const;                   // last computed average (0 until the first update)
bool   fps_updated() const;           // true only on the frame fps() was recomputed
```

### Design decisions

- **Clock:** `std::chrono::steady_clock`. It is monotonic, so it never jumps when the system clock is adjusted, and it is backed by `QueryPerformanceCounter` on MSVC. It also keeps `core` free of Windows headers.
- **Clamped delta.** After a window drag, a breakpoint or a minimize, the next frame's raw delta can be several seconds, and feeding that into smoothing or animation makes it overshoot wildly. `delta_seconds()` is therefore clamped to 0.25 s, and negative input counts as 0.
- **FPS is averaged**, not `1 / dt`, which flickers too fast to read. Frames and time accumulate, and about every 0.5 s the average is recomputed. The accumulation uses the **raw** (unclamped) time, so a stall shows up honestly as a dip. The division only happens inside the "at least 0.5 s" branch, so a zero denominator is impossible.
- **`advance()` and `tick()` are separate** so the logic can be unit-tested with fake time. Real time cannot be tested reliably, so the tests call `advance()` with made-up steps.

### Title readout

The app shows `gfxlab | <fps> FPS | <ms> ms` in the title, updated only when `fps_updated()` is true (about twice a second), because `SetWindowTextW` sends a synchronous message to the window. Frame time in milliseconds is the number to trust when comparing backends later, because FPS is nonlinear: going from 1000 to 2000 FPS saves 0.5 ms, while going from 30 to 60 FPS saves 16.7 ms.

In M1 the loop does no work, so FPS reads in the hundreds of thousands or more. Once M2 renders with vsync, it settles at the monitor's refresh rate.

### `set_title`

`set_title` reuses the UTF-8 to UTF-16 helper from `create()` and calls `SetWindowTextW`.

## 8. Tests and correctness checks

| Suite | Cases |
|---|---|
| `test_input` | 1 press and held state across frames; 2 auto-repeat; 3 release; 4 same-frame press and release; 5 mouse buttons; 6 mouse delta, including the first position and no-movement frames; 7 scroll accumulation and reset; 8 `release_all`; 9 no stale baseline after focus loss |
| `test_timer` | 1 `reset`; 2 one frame; 3 64 frames give exactly two FPS updates; 4 `fps_updated` lasts one advance; 5 a long stall is clamped for delta but reported in FPS; 6 negative input; 7 a real-clock smoke test |

The timer tests use a frame time of 1/64 s because it is exactly representable in binary, so 32 steps sum to exactly 0.5 with no rounding. With 1/60 the sum could land one rounding error short and make the test flaky.

The Win32 layer itself is **not** unit-tested, because it needs a real window. It is checked by hand with the checklist in section 11.

## 9. Frame-loop sequence

```
loop until the window asks to close:
    timer.tick()
    window.poll_events()

    if minimized:
        log once; wait_events(); timer.reset()
        if closing: break
        if still minimized: continue        # else carry on with this iteration

    if resized and size is positive: log the new size and aspect ratio

    read input: log button edges, scroll, middle-button drag deltas,
                key edges; Escape requests close

    if the FPS value just updated: set the window title
```

`timer.reset()` after waiting keeps the idle time out of the FPS average. In M2 this loop gains rendering and the orbit camera, and the drag-delta logging is replaced by camera control.

## 10. Known scope and non-goals

M1 does not include:

- rendering, an OpenGL context or a Vulkan surface (M2 and M4);
- text input, raw mouse input, gamepads, multiple windows or fullscreen.

### Known limitations

| Limitation | Notes |
|---|---|
| **Modal drag and resize loop** | **Resolved in M2 (Step 9).** While the user drags the title bar or an edge, Windows runs its own loop inside `DispatchMessage`, so `poll_events()` does not return. In M1 nothing rendered during the drag, and the stall showed up as one low FPS reading afterward. M2 renders from inside that loop with a timer and a render callback (see `m2-design.md`, Step 9). `was_resized()` still fires only once, after the drag ends |
| **Shared modifier keys** | Left and right Shift, Ctrl and Alt share one `Key`. Holding both and releasing one reads as released |
| **Capture lost without focus loss** | If another window takes mouse capture (for example, a system dialog) without `WM_KILLFOCUS`, the button-up may never arrive. `WM_CAPTURECHANGED` and `WM_CANCELMODE` are not handled yet |
| **Alt+Space** | The system menu is disabled, as described in Step 3 |
| **DPI changes** | `WM_DPICHANGED` (dragging the window to a monitor with different scaling) is not handled |
| **Text input** | `WM_CHAR` is not handled, and there is no raw mouse input or gamepad support |
| **Platform** | Windows only, and Windows 10 version 1703 or later for the DPI API |
| **Public size** | `width()` and `height()` are client pixels, which equal framebuffer pixels because the process is per-monitor DPI-aware |

## 11. Manual checklist for the Win32 layer

1. Open the window, then close it with the X button. It exits cleanly, with code 0.
2. Press and release A, a digit and an arrow. Each logs exactly one press and one release, even if held (no repeat spam).
3. Tap Alt, then Alt+Space. The window does not freeze. Alt+F4 still closes it.
4. Hold the middle button and drag the cursor **outside** the window. Deltas keep logging, and the release is logged when you let go.
5. Scroll the wheel. Steps of +1.0 and -1.0 are logged.
6. Hold a key, then Alt+Tab away. A `released` line appears for it.
7. Escape exits cleanly.
8. Drag-resize an edge: one `resized` line appears after release (from M2 onward the scene also keeps rendering during the drag). Maximize and restore: one line each.
9. Minimize: one `minimized` line, and CPU use drops to about 0%. Restore: no `resized` line, because the size is unchanged. Closing while minimized still exits.
10. The title updates about twice a second without flicker.

## 12. Completion checkpoint

The expected state at the end of M1:

- a resizable window opens at the requested client size and closes cleanly;
- keyboard, mouse buttons, cursor position and deltas, and the wheel are reported correctly, including during a drag outside the window and after losing focus;
- resize and minimize are reported without ever exposing a 0x0 size, and a minimized window uses no CPU;
- a tested frame timer reports delta time and FPS, shown in the title;
- `ctest` passes the `math`, `input` and `timer` suites.