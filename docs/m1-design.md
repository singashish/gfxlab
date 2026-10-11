# M1 design: window, input and frame timing

M1 adds the first thing you can see and touch: a Win32 window that reports its
size, a keyboard and mouse state tracker, and a frame timer with an FPS
readout. Nothing is drawn yet. The window is the surface that the OpenGL and
Vulkan backends will render into in M2 and M4.

For the build system, math and logging that this builds on, see
[`m0-design.md`](m0-design.md).

## 1. Goals and non-goals

**Goals**

- A window with no third-party library (no GLFW or SDL), built directly on Win32
- An input model that fits a viewport: "is the middle button held, and how far
  did the mouse move this frame?"
- Correct behaviour for the awkward cases: resize, minimize, Alt keys, losing
  focus mid-drag, and the cursor leaving the window during a drag
- As much logic as possible written so it can be unit-tested without Windows

**Non-goals**

- Rendering, an OpenGL context or a Vulkan surface (M2 and M4)
- Text input, raw mouse input, gamepads, multiple windows, fullscreen

## 2. Architecture

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

Dependencies point downward only: `app` uses `platform` and `core`, and
`gfx_platform` links `gfx_core`. The public headers contain **no Windows
types**:

- `window.h` hides everything behind a pointer to an implementation struct
  (the pimpl idiom), so `<windows.h>` and its macros (`min`, `max`, `ERROR`,
  `near`, `far`) cannot leak into the rest of the code. Only
  `window_win32.cpp` includes it.
- `WindowImpl` is declared at namespace scope rather than nested inside
  `Window`, because the free function `wnd_proc` has to name it, and a free
  function cannot name a private nested type.
- `input.h` and `timer.h` are header-only and platform-independent. That is
  why `test_input` and `test_timer` link only `gfx_core` and run without a
  window.

`gfx_platform` is a static library that exposes `src/` as a public include
path and links `gfx_core` publicly. It defines `WIN32_LEAN_AND_MEAN`,
`NOMINMAX`, `UNICODE` and `_UNICODE` privately, because only its own sources
include `windows.h`. Its `CMakeLists.txt` stops configuration with an error on
non-Windows hosts.

## 3. The window (`window.h`, `win32/window_win32.cpp`)

### 3.1 Interface

| Member | Meaning |
|--------|---------|
| `create(WindowDesc)` | Create and show the window. `title` is UTF-8, and `width` and `height` are the **client area** in pixels. Returns false and logs on failure |
| `poll_events()` | Handle all pending messages without blocking, and advance the input frame |
| `wait_events()` | Block until a message arrives, then `poll_events()`. Used while minimized |
| `should_close()` / `request_close()` | Has closing been requested, and request it from code |
| `is_minimized()` | True while the window is minimized |
| `was_resized()` | True for one frame after the client size changed |
| `width()` / `height()` | Current client size in pixels (never 0 while minimized, see below) |
| `native_handle()` | The `HWND` as `void*`, needed later for the GL context and the Vulkan surface |
| `set_title(utf8)` | Change the window title |
| `input()` | Read-only access to the `InputState` |

`Window` is not copyable. Its destructor destroys the native window.

### 3.2 Why polling, not blocking

`PeekMessage` returns immediately, `GetMessage` blocks until a message
arrives. A renderer must draw every iteration, so the main loop is "handle
pending messages, render, repeat", and `poll_events()` uses `PeekMessageW`.
Blocking is used only on purpose, in `wait_events()`, when there is nothing to
draw.

### 3.3 Creation sequence

1. **Validate** the description (non-null title, positive size).
2. **Make the process DPI-aware** (once), with
   `SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2)`.
   Without it, Windows at 125% or 150% scaling reports virtualized pixel sizes
   and stretches the output, which is blurry for a renderer. "Access denied"
   is tolerated, because awareness may already be set by a manifest. This API
   needs Windows 10 version 1703 or later.
3. **Convert the title** from UTF-8 to UTF-16 with `MultiByteToWideChar`. The
   code uses the wide (`W`) Win32 functions throughout. Invalid UTF-8 yields an
   empty title.
4. **Register the window class** once, with the `CS_OWNDC` style (the window
   keeps one device context for life, which OpenGL context creation relies on),
   an arrow cursor and a black background brush so the window has a defined
   look before a renderer takes over.
5. **Compute the outer size from the client size** with
   `AdjustWindowRectExForDpi` and the system DPI. The requested 1280x720 is the
   *renderable* area, and the title bar and borders are added on top. The plain
   `AdjustWindowRectEx` assumes 96 DPI and would be wrong under scaling.
6. **Create the window** with `CreateWindowExW`, passing the `WindowImpl`
   pointer as the creation parameter.
7. **Show it** and log the client size.

### 3.4 How `wnd_proc` finds its object

The window procedure is a plain C callback with no `this`. The pointer to the
`WindowImpl` is passed through `CreateWindowExW`, picked up in `WM_NCCREATE`
(the first message sent) and stored in the window's `GWLP_USERDATA` slot. Every
later call reads it back from there. `WM_NCDESTROY` clears the slot and the
stored handle.

### 3.5 Message handling

| Message | Effect |
|---------|--------|
| `WM_CLOSE` | Set a flag and return. The window is **not** destroyed here: the app decides when to quit (and could later ask "save changes?"), and the destructor does the destruction |
| `WM_SIZE` | If minimized: set the minimized flag and leave the stored size untouched. Otherwise clear the flag, and if the new client size differs from the stored one, store it and set the resized flag |
| `WM_KEYDOWN`, `WM_KEYUP` | Translate the virtual key and update `InputState`; handled |
| `WM_SYSKEYDOWN`, `WM_SYSKEYUP` | Same, but then fall through to `DefWindowProc` so Alt+F4 still closes the window |
| `WM_SYSCOMMAND` with `SC_KEYMENU` | Swallowed (see 4.5) |
| `WM_MOUSEMOVE` | Store the cursor position |
| `WM_LBUTTONDOWN`, `WM_MBUTTONDOWN`, `WM_RBUTTONDOWN` | Press the button, and start mouse capture if no button was already down |
| `WM_LBUTTONUP`, `WM_MBUTTONUP`, `WM_RBUTTONUP` | Release the button, and end capture when none are left down |
| `WM_MOUSEWHEEL` | Add `delta / WHEEL_DELTA` to the scroll |
| `WM_KILLFOCUS` | Release capture and call `release_all()`; then fall through |
| `WM_NCDESTROY` | Clear the stored pointer and handle |

`poll_events()` also treats `WM_QUIT` as a close request.

### 3.6 Resize and minimize

A minimized window reports a client size of **0x0**. If that were stored, it
would flow into everything downstream: `perspective()` computes
`aspect = width / height`, so a zero height gives infinity or NaN and the whole
projection matrix is destroyed. An OpenGL viewport of 0x0 draws nothing, and
Vulkan refuses to create a swapchain with a zero extent. So the rule is:

- A zero size never overwrites the last valid size, and "minimized" is exposed
  as its own fact.
- `was_resized()` fires only when the size actually **differs**. Restoring from
  a minimize reports the same size as before and does not count, because
  recreating a swapchain or framebuffer for no reason would be wasted work.
- `resized` is cleared at the start of `poll_events()`, so it is a one-frame
  signal like the input edges.

While minimized, the app calls `wait_events()`, which uses `WaitMessage()` to
sleep the thread until a message arrives. This keeps CPU use near zero instead
of spinning a core on a window that draws nothing. A `WM_CLOSE` is a message,
so closing a minimized window still wakes the loop.

### 3.7 `set_title`

Reuses the same UTF-8 to UTF-16 helper as `create()` and calls
`SetWindowTextW`. That call sends a synchronous message to the window, which is
why the app updates the title only when the FPS value changes (twice a second)
and not every frame.

## 4. Input (`input.h`)

### 4.1 Model

Input is **polled state**, not an event queue. The platform layer translates
Win32 messages into calls on `InputState`, and the application reads it once
per frame. That fits a viewport, where the questions are state questions.

The frame contract, owned by `Window::poll_events()` so the app cannot get it
wrong:

```
begin_frame()  ->  platform feeds events  ->  end_frame()  ->  app queries
```

Three kinds of data:

| Kind | Examples | Lifetime |
|------|----------|----------|
| Held state | `key_down`, `mouse_down` | Until released |
| Edge state | `key_pressed`, `key_released`, `mouse_pressed`, `mouse_released` | One frame; cleared by `begin_frame()` |
| Per-frame motion | `mouse_dx`, `mouse_dy`, `scroll` | One frame; the delta is computed in `end_frame()` and scroll accumulates during the frame and is cleared by `begin_frame()` |

Mouse coordinates are client pixels with the origin at the **top left** and Y
increasing **downward** (the Windows convention). OpenGL window coordinates
start at the bottom left, so convert at the point of use. Mouse deltas are in
pixels per frame, not per second, so they must not be multiplied by delta time.

### 4.2 Rules, each pinned by a test

| Situation | Behaviour |
|-----------|-----------|
| Key goes down while up | `down` and `pressed` become true |
| Key goes down while **already down** | Ignored. This is OS **auto-repeat**, and it must not count as a new press |
| Key goes up while down | `down` false, `released` true |
| Press and release in one frame | Both `pressed` and `released` are true, `down` is false |
| First mouse position ever seen | Produces a delta of 0, not a jump from (0, 0) |
| A frame ends before any position has arrived (at startup, or after focus loss) | Delta is 0 and **no baseline is recorded**, so the default (0, 0) is never mistaken for a real position |
| Scroll events in one frame | Add up |
| Focus lost (`release_all`) | Every held key and button is released through the normal path, so each reports `released`; the mouse baseline is discarded |
| Out-of-range key or button index | Ignored on write, false on read |

The last "mouse baseline" row has a subtlety worth recording. Clearing only the
"has previous position" flag is not enough: the same frame's `end_frame()`
would immediately store the stale cursor position as a new baseline, and the
first move after focus returns would produce a large spurious delta. So
`release_all()` also clears the "has a position" flag, and `end_frame()` waits
for a **fresh** position before recording a baseline. Test case 9 in
`test_input.cpp` covers exactly this sequence.

### 4.3 Key mapping

`Key` and `MouseButton` are `enum class` types with a `GFX_` prefix. The keys
are A to Z, 0 to 9, numpad 0 to 9, Escape, Space, Tab, Delete, the four arrows,
Shift, Ctrl and Alt, plus `GFX_KEY_UNKNOWN` and `GFX_KEY_COUNT`.

Letters, digits and numpad digits are kept **contiguous and in order** in the
enum. Win32 virtual-key codes for those ranges are also sequential, so
translating them is arithmetic (`GFX_KEY_A + (vk - 'A')`), and a `switch` is
needed only for the remaining keys.

### 4.4 Mouse capture

The orbit camera drags with the middle button held, and the cursor will often
leave the window mid-drag. Normally Windows then stops sending mouse messages,
including the button-up, which would leave the state stuck at "down".

- When a button goes down and none was already down, the window calls
  `SetCapture`.
- When a button goes up and none are left down (and the capture is still ours),
  it calls `ReleaseCapture`.

With capture, the window keeps receiving `WM_MOUSEMOVE` and the final button-up
wherever the cursor is. Mouse coordinates are read with `GET_X_LPARAM` and
`GET_Y_LPARAM`, which are **signed**. `LOWORD` would turn a negative coordinate
(outside the window, during capture) into a large positive one.

The wheel delta is a multiple of `WHEEL_DELTA` (120). High-resolution mice and
touchpads send fractions, which is why scroll is a `float`. Positive means away
from the user. `WM_MOUSEWHEEL` carries *screen* coordinates, unlike the other
mouse messages, so its position is deliberately unused.

### 4.5 Alt and system keys

Windows sends `WM_SYSKEYDOWN` and `WM_SYSKEYUP` instead of the plain messages
whenever Alt is involved. They are mapped into `InputState` like the plain
ones, so Alt (and Alt-modified keys, which a Blender-style viewport uses) work.
They then fall through to `DefWindowProc`, which keeps Alt+F4 working.

The default handler would also turn a tap of Alt, F10 or Alt+Space into
menu mode, which freezes the render loop until the user clicks. The window
therefore swallows `WM_SYSCOMMAND` with `SC_KEYMENU`. The cost is that
Alt+Space no longer opens the system menu.

### 4.6 Focus loss

`WM_KILLFOCUS` releases capture and calls `release_all()`. Without it, pressing
Alt+Tab while holding a key or button would never deliver the release, and the
orbit camera would stay "dragging" forever.

## 5. Frame timer (`src/core/timer.h`)

`gfx::FrameTimer` provides a safe delta time for camera smoothing and
animation, and a readable FPS figure.

```cpp
FrameTimer();                 // calls reset()
void   reset();               // start timing from now; clears everything
void   tick();                // measure real elapsed time, then advance()
void   advance(double raw_seconds);   // all of the logic
double delta_seconds() const; // clamped to [0, 0.25]
double fps() const;           // last computed average (0 until the first update)
bool   fps_updated() const;   // true only on the frame fps() was recomputed
```

Design decisions:

- **Clock:** `std::chrono::steady_clock`. It is monotonic, so it never jumps
  when the system clock is adjusted, and it is backed by
  `QueryPerformanceCounter` on MSVC. It also keeps `core` free of Windows
  headers.
- **Clamped delta.** After a window drag, a breakpoint or a minimize, the next
  frame's raw delta can be several seconds, and feeding that into camera
  smoothing makes it overshoot wildly. `delta_seconds()` is therefore clamped
  to 0.25 s, and negative input counts as 0.
- **FPS is averaged**, not `1 / dt`, which flickers too fast to read. Frames
  and time accumulate, and about every 0.5 s the average is recomputed. The
  accumulation uses the **raw** (unclamped) time, so a stall shows up honestly
  as a dip. The division only happens inside the "at least 0.5 s" branch, so a
  zero denominator is impossible.
- **`advance()` and `tick()` are separate** so the logic can be unit-tested
  with fake time. Real time cannot be tested reliably, so the tests call
  `advance()` with made-up steps.

The app shows `gfxlab | <fps> FPS | <ms> ms` in the title. Frame time in
milliseconds is the number to trust when comparing backends later, because FPS
is nonlinear: going from 1000 to 2000 FPS saves 0.5 ms, while going from 30 to
60 FPS saves 16.7 ms. Right now the loop does no work, so FPS reads in the
hundreds of thousands or more. Once M2 renders with vsync, it will settle at
the monitor's refresh rate.

## 6. The main loop

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

After waking from a minimize, a restored window **continues the same
iteration** instead of jumping back to the top, so the resize and input events
that woke it are not skipped. `timer.reset()` after waiting keeps the idle time
out of the FPS average.

## 7. Testing

| Suite | Cases |
|-------|-------|
| `test_input` | 1 press and held state across frames; 2 auto-repeat; 3 release; 4 same-frame press and release; 5 mouse buttons; 6 mouse delta, including the first position and no-movement frames; 7 scroll accumulation and reset; 8 `release_all`; 9 no stale baseline after focus loss |
| `test_timer` | 1 `reset`; 2 one frame; 3 64 frames give exactly two FPS updates; 4 `fps_updated` lasts one advance; 5 a long stall is clamped for delta but reported in FPS; 6 negative input; 7 a real-clock smoke test |

The timer tests use a frame time of 1/64 s because it is exactly representable
in binary, so 32 steps sum to exactly 0.5 with no rounding. With 1/60 the sum
could land one rounding error short and make the test flaky.

The Win32 layer itself is **not** unit-tested, because it needs a real window.
It is checked by hand with the list below.

### Manual checklist for the Win32 layer

1. Open the window, then close it with the X button. It exits cleanly, with code 0.
2. Press and release A, a digit and an arrow. Each logs exactly one press and
   one release, even if held (no repeat spam).
3. Tap Alt, then Alt+Space. The window does not freeze. Alt+F4 still closes it.
4. Hold the middle button and drag the cursor **outside** the window. Deltas
   keep logging, and the release is logged when you let go.
5. Scroll the wheel. Steps of +1.0 and -1.0 are logged.
6. Hold a key, then Alt+Tab away. A `released` line appears for it.
7. Escape exits cleanly.
8. Drag-resize an edge: one `resized` line appears after release. Maximize and
   restore: one line each.
9. Minimize: one `minimized` line, and CPU use drops to about 0%. Restore: no
   `resized` line, because the size is unchanged. Closing while minimized
   still exits.
10. The title updates about twice a second without flicker.

## 8. Known limitations

| Limitation | Notes |
|------------|-------|
| **Modal drag and resize loop** | While the user drags the title bar or an edge, Windows runs its own loop inside `DispatchMessage`, so `poll_events()` does not return. Nothing renders during the drag, and `was_resized()` fires once after it ends. The stall is also counted in one FPS window, so the title shows a single low reading afterward. The fix is to render from inside that loop (via `WM_ENTERSIZEMOVE` and `WM_EXITSIZEMOVE` and a timer, or a render callback), planned for M2 |
| **Shared modifier keys** | Left and right Shift, Ctrl and Alt share one `Key`. Holding both and releasing one reads as released |
| **Capture lost without focus loss** | If another window takes mouse capture (for example, a system dialog) without `WM_KILLFOCUS`, the button-up may never arrive. `WM_CAPTURECHANGED` and `WM_CANCELMODE` are not handled yet |
| **Alt+Space** | The system menu is disabled, as described in 4.5 |
| **DPI changes** | `WM_DPICHANGED` (dragging the window to a monitor with different scaling) is not handled |
| **Text input** | `WM_CHAR` is not handled, and there is no raw mouse input or gamepad support |
| **Platform** | Windows only, and Windows 10 version 1703 or later for the DPI API |
| **Public size** | `width()` and `height()` are client pixels, which equal framebuffer pixels because the process is per-monitor DPI-aware |

## 9. Definition of done for M1

- A resizable window opens at the requested client size and closes cleanly
- Keyboard, mouse buttons, cursor position and deltas, and wheel are reported
  correctly, including during a drag outside the window and after losing focus
- Resize and minimize are reported without ever exposing a 0x0 size, and a
  minimized window uses no CPU
- A tested frame timer reports delta time and FPS, shown in the title
- `ctest` passes all three suites (`math`, `input`, `timer`)
