# M3 — Models, Lighting, and Object Transforms

> **Status:** Complete (Steps 1–5)  
> **Scope:** Parse Wavefront OBJ files with an own parser, draw the mesh with proper lighting, represent it with a transform, and move, rotate and scale it interactively with Blender-style G, R and S keys.

## 1. Purpose

M2 produced an empty viewport: a grid, an orbit camera and two demo triangles. M3 puts a real model in it and makes it behave like an object in Blender:

- a model file is loaded from the command line (`gfxlab.exe model.obj`);
- it is converted into the project's coordinate system, centered, and placed on the floor;
- it is lit by view-space lights with smooth or flat shading, depending on the file;
- pressing **G**, **R** or **S** grabs, rotates or scales it, optionally constrained to the X, Y or Z axis, and a click confirms or Esc cancels.

The work is split so that the parts which can be tested without a GPU are tested without one. The parser, the mesh helpers, the transform and the editing math live in `gfx_core`, and the OpenGL code only uploads and draws.

## 2. Architecture and what M3 adds

```
gfxlab        main.cpp: load model, own the Transform and TransformEditor, build the draw list
gfx_renderer  DrawItem, MeshId, IRenderer::create_mesh, FrameData::draw_list
gfx_gl        GLMesh (VAO/VBO/EBO), GLMeshRenderer (shader + meshes)
gfx_core      Mesh, Aabb, OBJ parser, normals, Transform, TransformEditor
```

New files:

| Area | Files |
|---|---|
| Mesh data and helpers | `core/mesh.h`, `core/mesh.cpp` |
| OBJ parser | `core/obj_loader.h`, `core/obj_loader.cpp` |
| Transform | `core/transform.h` |
| Interactive editing | `core/transform_edit.h`, `core/transform_edit.cpp` |
| OpenGL meshes | `renderer/opengl/gl_mesh.h`, `gl_mesh.cpp` |
| Test models | `assets/cube.obj`, `cube_no_normals.obj`, `suzanne_f.obj`, `suzanne_s.obj` |
| Tests | `tests/test_obj.cpp`, `test_transform.cpp`, `test_transform_edit.cpp` |

The dependency rule from M2 still holds: `gfx_core` knows nothing about OpenGL, Win32 or `InputState`. The editor receives a camera description and mouse positions as plain numbers, in the same way the orbit camera receives pixel deltas.

## 3. Step 1 — Mesh and OBJ parser

### The OBJ format

OBJ is a line-based text format. The parser understands these lines and ignores everything else (`o`, `g`, `s`, `usemtl`, `mtllib`, comments):

```
v  x y z          a position (anything after z is ignored: a w component or vertex colours)
vn x y z          a normal
vt u v            a texture coordinate (accepted in faces, otherwise ignored)
f  a b c ...      a face with three or more corners
```

A face corner has four spellings: `v`, `v/vt`, `v//vn` and `v/vt/vn`. Indices are **1-based**, and **negative indices count backwards** from the most recent entry, so `-1` is the last `v` defined so far. A negative index must be resolved when its `f` line is read, using the counts at that moment.

### From the file's layout to the GPU's layout

A GPU draws from one index per vertex, and a vertex is a complete bundle of position and normal. OBJ allows position 7 to appear with normal 3 in one face and normal 4 in another. The loader therefore converts the file into the GPU layout:

```cpp
struct Mesh {
    std::vector<Vec3> positions;
    std::vector<Vec3> normals;           // empty if the file had no (complete) normals
    std::vector<std::uint32_t> indices;  // 3 per triangle
};
```

- Each unique `(position index, normal index)` pair becomes **one output vertex**. A `std::map` from the pair to the output index does the deduplication. A cube with face normals comes out as 24 vertices (8 corners × 3 normals), and without normals as 8.
- Polygons are **fan-triangulated**: a face with corners `0..n-1` becomes `(0,1,2), (0,2,3), …, (0,n-2,n-1)`. This is correct for convex polygons.
- If **any** corner of any face lacks a normal, the whole `normals` array is cleared at the end. A half-filled array would be worse than none, because the arrays must stay parallel. Step 3 generates normals in that case.

### Robust parsing

- Numbers are read with `std::from_chars` rather than `atof` or `strtof`. It does not depend on the locale (a system that writes `1,5` still reads `1.5`), and it reports failure instead of silently returning 0. A bad number is an error with its line number.
- Windows `\r\n` line endings, tabs, blank lines and trailing `#` comments are accepted.
- Every failure returns `false`, clears the output and sets an error message that begins with the line number, for example `line 7: face index 9 out of range`.
- A file without faces is an error. A face with fewer than three corners, index 0 and out-of-range indices are errors.
- Anything after the third number of a `v` line is ignored. Many exporters (scanners, MeshLab, Blender with vertex colours) append `r g b` there.

### API

```cpp
bool parse_obj(std::string_view text, Mesh& out, std::string& error);
bool load_obj(const std::string& path, Mesh& out, std::string& error);
Aabb compute_bounds(const Mesh& mesh);
```

`parse_obj` is separate from `load_obj` so the parser can be tested with strings in memory.

## 4. Step 2 — Getting the model on screen

### Axis conversion

Almost every OBJ exporter writes **Y-up**, but gfxlab is **Z-up**. Rotating by +90° about X converts one into the other:

```
(x, y, z) -> (x, -z, y)
```

`mesh_y_up_to_z_up` applies it to positions and normals. A rotation keeps the triangle winding, so nothing turns inside out. A model exported with Z up will therefore lie on its back, and the exporter's up-axis option should be changed.

### Framing the camera

`OrbitCamera::frame(center, radius, fov_y)` sets `target = center` and places the camera so the bounding sphere fits the vertical field of view. A sphere of radius `r` at distance `d` fills half the field of view when `sin(fov_y / 2) = r / d`, so:

```
distance = r / sin(fov_y / 2) * 1.2        (the 1.2 is breathing room)
```

The distance is clamped to the same range as zooming (0.2 to 500). Yaw and pitch are unchanged.

### Meshes behind the renderer interface

Backends must not leak into the app, so meshes are referred to by handle:

```cpp
using MeshId = std::uint32_t;   // 0 means "no mesh"
virtual MeshId create_mesh(const Mesh& mesh) = 0;   // in IRenderer; returns 0 on failure
```

`GLMesh` owns a VAO, a VBO and an EBO. `GLMeshRenderer` owns the shader program and a `std::vector<std::unique_ptr<GLMesh>>`. The vector holds pointers because `GLMesh` owns GL objects and must not be copied, and a mesh id is the vector index plus one.

Two OpenGL rules that this code depends on:

- The **element-buffer binding belongs to the VAO**. The EBO must be bound while the VAO is bound, and must not be unbound before the VAO is, or the VAO ends up with no indices and nothing draws.
- Every failure path deletes whichever GL objects were already created.

### Command line

`gfxlab.exe path\to\model.obj` loads the model. The name is looked up as given (relative to the current directory, or absolute), then in the repository's `assets\` folder, then in `assets\local\`, so `gfxlab.exe cube.obj` works from any directory. The `assets` folder reaches the app as a `GFX_ASSETS_DIR` compile definition, an absolute path into the source tree. The log prints the resolved path (`model: ...`). If no candidate exists, the error lists every location tried and the app exits with code 1. A model that exists but fails to load also logs the error and exits with code 1. With no argument the app behaves as in M2 and draws the demo triangles.

## 5. Step 3 — Normals and lighting

### Guaranteeing normals

`ensure_normals(Mesh&)` makes sure the mesh has exactly one unit-length normal per vertex and returns whether it had to generate them:

- **Normals present:** each one is normalized. A zero or non-finite vector becomes `(0, 0, 1)`.
- **Normals missing (or the wrong count):** the mesh is **unwelded**. A flat normal belongs to a triangle, not a vertex, and the welded cube has only eight shared vertices that cannot hold three different face directions. So every triangle gets its own three vertices, and all three carry the triangle's normal:

```
n = normalize(cross(pb - pa, pc - pa))      // outward for counter-clockwise winding
```

The indices become `0, 1, 2, 3, …`. Flat normals were chosen over smooth ones for files without normals, because smoothing a cube without a crease angle turns it into a blob.

A degenerate triangle has a zero cross product and would normalize to NaN, so a tiny-length test falls back to `(0, 0, 1)`. The test compares the **squared** length against `1e-30`. An earlier threshold of `1e-12` was too large: a cross product's length shrinks with the square of the edge length, so triangles with edges below about 0.5 mm in a model measured in metres were misreported as degenerate. A regression test covers this.

### Interleaved vertex buffer

`GLMesh::upload` builds one buffer with each vertex stored as `position.xyz normal.xyz` (24 bytes):

| Attribute | Components | Stride | Offset |
|---|---|---|---|
| 0, position | 3 floats | 24 | 0 |
| 1, normal | 3 floats | 24 | 12 |

It refuses a mesh whose `normals` and `positions` differ in size. `static_assert(sizeof(Vec3) == 12)` guards the layout assumption.

### The lighting model

Lighting is computed in **view space**, so the light follows the camera, like Blender's solid mode:

```glsl
vec3 n = normalize(v_view_normal);            // interpolation shortens normals
if (!gl_FrontFacing) n = -n;                  // light the back of a face too
vec3 v = normalize(-v_view_pos);              // toward the camera

vec3 key_dir = normalize(vec3(-0.5, 0.6, 0.7));   // a light up-left of the camera
float diff = max(dot(n, key_dir), 0.0);           // Lambert

vec3 up_view = mat3(u_view)[2];                   // world +Z in view space
float hemi = dot(n, up_view) * 0.5 + 0.5;
vec3 ambient = mix(vec3(0.10, 0.10, 0.12), vec3(0.30, 0.32, 0.36), hemi);

vec3 h = normalize(key_dir + v);                  // Blinn-Phong half vector
float spec = diff > 0.0 ? pow(max(dot(n, h), 0.0), 32.0) * 0.25 : 0.0;

color = base * (ambient + 0.75 * diff) + vec3(spec);
```

- **Lambert diffuse** is the cosine of the angle between the surface and the light.
- **Hemisphere ambient** approximates light bouncing from sky and ground. It uses the world up direction, so upward-facing surfaces stay lighter however the camera orbits.
- **Specular** is a small Blinn-Phong highlight, computed only for surfaces facing the light.
- `gl_FrontFacing` assumes counter-clockwise winding, which OBJ uses and the axis conversion preserves. It is also the reason a transform must never have a negative scale (see Step 5).

## 6. Step 4 — Transform and draw list

### The transform

```cpp
struct Transform {
    Vec3 position = {0, 0, 0};
    Vec3 rotation = {0, 0, 0};   // Euler angles in radians
    Vec3 scale    = {1, 1, 1};

    Mat4 rotation_matrix() const;   // Rz * Ry * Rx
    Mat4 model_matrix() const;      // T * R * S
    Mat4 normal_matrix() const;     // R * S^-1
};
```

- The Euler order is Blender's default XYZ: X is applied first, then Y, then Z. Written for column vectors that is `R = Rz * Ry * Rx`.
- `model_matrix = translate(position) * R * scale(scale)`. Reading right to left: scale first, then rotate, then translate.
- Euler angles suffer from gimbal lock at `ry = ±90°`. That is acceptable for a viewer. Quaternions would be the upgrade if it ever matters.

### Why normals need their own matrix

A model matrix with non-uniform scale tilts normals away from their surface. Take a surface with tangent `t` and normal `n`, where `dot(t, n) = 0`. After transforming by `M`, the tangent becomes `M t`. The normal must be transformed by the inverse transpose of the upper 3×3 of `M`, written `N`, to stay perpendicular. For `M = R * S` this simplifies:

```
N = (R S)^-T = R^-T S^-T = R S^-1       (R is orthogonal, S is diagonal)
```

so `normal_matrix()` is the rotation times the inverse scale. Each scale component is clamped to at least `1e-6` in magnitude before inverting, preserving its sign. `test_transform` checks both sides of this: the corrected normal stays perpendicular to the transformed tangent, and the plain model matrix does not.

### Placing the model

After loading, `main.cpp` runs:

1. `mesh_y_up_to_z_up`, then `ensure_normals`.
2. `mesh_center_at_origin`, so the bounding-box centre is the local origin. This makes the **pivot** the middle of the model, so rotations and scales happen about it, like Blender's "origin to geometry".
3. `compute_bounds`, then `transform.position.z = height / 2`, so the bottom of the model rests on the floor.

The camera is then framed on the transform's position.

### The draw list

```cpp
struct DrawItem {
    MeshId mesh;
    Mat4 model;
    Mat4 normal_matrix;
};
// FrameData::draw_list is a std::vector<DrawItem>
```

The app computes both matrices, as it does for the camera, and the backend only draws them. For each item `GLMeshRenderer` uploads `u_model` and `u_normal_model`, then draws the mesh. `u_view` and `u_view_proj` are uploaded once per frame. The vertex shader becomes:

```glsl
vec4 world = u_model * vec4(a_position, 1.0);
v_view_pos    = (u_view * world).xyz;
v_view_normal = mat3(u_view) * (mat3(u_normal_model) * a_normal);
gl_Position   = u_view_proj * world;
```

When the draw list is empty, `GLRenderer` still draws the demo triangles, which is what runs without a command-line argument.

## 7. Step 5 — Interactive move, rotate and scale

### Interaction, as in Blender

| Input | Effect |
|---|---|
| **G**, **R**, **S** | Start grabbing, rotating or scaling the model |
| **X**, **Y**, **Z** | Constrain to that axis; pressing the same key again goes back to free |
| **Left click** | Confirm |
| **Right click** or **Esc** | Cancel; the model returns exactly to where it started |
| **G** / **R** / **S** during an edit | Cancel the current edit and start the new one |

The active mode and axis appear in the window title.

### `TransformEditor`

All the maths lives in `TransformEditor` in `gfx_core`. It takes a camera description and mouse positions as numbers and never touches input or the window:

```cpp
struct EditView { Vec3 eye; CameraBasis basis; float fov_y, width, height; };

void begin(EditMode mode, const Transform& current, float mouse_x, float mouse_y, const EditView& view);
void set_axis(EditAxis axis);
void update(float mouse_x, float mouse_y, const EditView& view, Transform& out);
void confirm();
Transform cancel();
```

The key design rule: **every update is computed from the transform at `begin` plus the total mouse movement since then**, and never by adding up per-frame changes. Cancelling is therefore exact, and switching the axis mid-edit re-applies the same mouse movement with the new constraint, as in Blender.

### Projecting the pivot to the screen

Rotating and scaling are driven by the mouse's position around the object's pivot on screen. `project_to_screen` computes that without a projection matrix, from the same camera basis used for the grid's rays. With `p = world - eye`, `depth = dot(p, forward)` and `tan_half = tan(fov_y / 2)`:

```
ndc_x = dot(p, right) / (depth * tan_half * aspect)
ndc_y = dot(p, up)    / (depth * tan_half)
px = (ndc_x * 0.5 + 0.5) * width
py = (1 - (ndc_y * 0.5 + 0.5)) * height       (pixel y grows downward)
```

It fails when the point is behind the camera, and the editors then fall back to the viewport centre. Because it depends only on the basis and the field of view, it works for any backend.

### Grab

At `begin`, the editor stores the world distance covered by one pixel at the object's depth, `s = 2 * depth * tan_half / height`. The free movement is:

```
delta = right * (dx * s) - up * (dy * s)        (dy grows downward, so the model follows the cursor)
```

With a constraint, the movement is projected onto the world axis `a`: `a * dot(delta, a)`. When the axis is almost parallel to the view direction the projection is nearly zero and the model barely moves. That is inherent to this approach.

### Rotate

- The angle of the mouse around the projected pivot is `atan2(-(my - cy), mx - cx)`, which is counter-clockwise-positive on screen.
- Each update adds the **wrapped** difference from the previous angle, wrapped to (−π, π], to a running total. This lets the model turn through more than half a revolution.
- Free rotation turns about the view axis, `-forward`. Constrained rotation turns about the world axis, with its sign flipped when the axis points away from the camera. As a result the model always turns in the direction the mouse circles.
- The new orientation is `mat4_rotate(axis, total) * R_start`, converted back to Euler angles with `euler_from_matrix`:

```
ry = asin(clamp(-R(2,0), -1, 1))
rx = atan2(R(2,1), R(2,2))
rz = atan2(R(1,0), R(0,0))
gimbal lock (|R(2,0)| > 0.99999): rx = 0, rz = atan2(-R(0,1), R(1,1))
```

### Scale

`factor = distance(mouse, pivot) / max(distance(start_mouse, pivot), 8 px)`. The 8-pixel floor prevents a division by nearly zero when an edit starts with the cursor on the pivot. The factor is clamped to at least `0.01`, so the scale never reaches zero or goes negative. A negative scale would mirror the mesh and reverse its winding, which would invert the lighting because the shader relies on `gl_FrontFacing`. A constrained scale multiplies only that component. For scale the axes are the **object's** axes, because scale is stored in object space.

### Wiring in `main.cpp`

Each frame, after the camera input, the app builds an `EditView`, reads the G/R/S/X/Y/Z keys and the mouse buttons, and calls `begin`, `set_axis`, `update`, `confirm` or `cancel`. Pressing Esc closes the app only if no edit was active at the start of that frame (and none is active now), so the first Esc cancels an edit and a second one quits. The camera mapping is fixed at `begin`; orbiting the camera during an edit is allowed but the grab keeps its starting mapping.

## 8. Tests and correctness checks

| Suite | What it covers |
|---|---|
| `test_obj` | triangles, quads and fan triangulation; negative indices; all four corner spellings; the 24-vertex cube with normals; mixed normals; ignored statements, comments, blank lines and CRLF; scientific notation and vertex colours; all error cases with line numbers; a missing file; bounds; the axis conversion; flat-normal generation (single triangle, degenerate, small triangle, cube without normals, outward orientation); normalization of existing normals; loading the `assets/` cubes |
| `test_transform` | identity, translation, scale, rotation, the order of scale/rotate/translate, the XYZ Euler order, the normal-matrix property (and that the model matrix fails it), safe zero scale, centring a mesh |
| `test_transform_edit` | projection to the screen; free and constrained grab; cancel restoring the start; axis toggle; free and constrained scale, clamping and a start on the pivot; Z rotation with the camera above and below the floor; free rotation; rotation past 180° in several steps; the Euler round trip including gimbal lock |

The tests that load files get the `assets/` folder from a `GFX_ASSETS_DIR` compile definition set in CMake, so they pass from any build directory.

The core, including all of these tests, also builds and passes with GCC on Linux. Running it there found two problems that Windows hid (see the next section).

## 9. Problems found during M3

| Problem | Cause and fix |
|---|---|
| Degenerate-normal threshold too large | Compared a squared length against `1e-12`; use `1e-30` and `!(len2 >= eps)` so NaN is also caught |
| `*.obj` in `.gitignore` hid `assets/*.obj` | The rule was meant for compiler output. A later `!assets/**/*.obj` re-includes models. A fresh clone failed the OBJ tests until the assets were committed |
| Tests loaded assets relative to the working directory | Replaced with an absolute path from `GFX_ASSETS_DIR` |
| `kMinScale` used inside a lambda without capture | MSVC accepts it, GCC and Clang do not, because `std::max` takes a reference. Fixed with `static constexpr` |
| A test expected a Z rotation without selecting the axis | The test forgot `set_axis(Z)`. Free mode rotates about the view axis |

## 10. Frame-loop sequence

1. `window.poll_events()`, minimized handling as in M2.
2. Camera input: middle-drag orbit, Shift + middle-drag pan, scroll zoom.
3. Build the `EditView` from the camera and the window size.
4. Editing: start, switch, constrain, update, confirm or cancel the `TransformEditor`, writing into the model's `Transform`.
5. Esc: quit only if no edit was active.
6. `draw_frame()`: fill `FrameData` (camera, projection parameters, grid fade, the draw list with the model and normal matrices), render, and update the title with FPS and the edit mode.

## 11. Known scope and limitations

| Item | Notes |
|---|---|
| **Single object** | One model, always selected. There is no scene graph, picking or multiple objects |
| **OBJ subset** | Positions, normals and faces only. No texture coordinates, materials (`.mtl`), groups or smoothing groups |
| **Triangulation** | Fan triangulation is correct for convex polygons only |
| **Generated normals** | Flat only. Smooth normals with a crease angle are not generated |
| **Up axis** | Assumes Y-up input. There is no option for Z-up files |
| **Rotation** | Euler angles, with gimbal lock at `ry = ±90°`. Rotations about a world axis are converted back to Euler angles after each update |
| **Scale** | Always positive, with a minimum factor of 0.01 |
| **Constrained grab** | Barely moves when the axis points almost along the view direction |
| **Editing while orbiting** | A grab keeps the camera mapping from when it started |
| **Lighting** | One key light that follows the camera, a hemisphere ambient and a specular highlight. No shadows, materials or textures |
| **Large models** | Vertex data is unweld-expanded 3× when normals are missing. Dedup uses `std::map`, which is slow for models with hundreds of thousands of faces. The camera clamps at distance 500 and the far plane is 1000 |
| **Command line** | `argv` is narrow-character, so paths with non-ASCII characters may fail to open |
| **Demo triangles** | Remain only as the no-argument fallback |
| **Backend** | OpenGL only. `DrawItem` and `MeshId` are backend-neutral so the Vulkan and CUDA backends can reuse them |

## 12. Manual checklist

1. `gfxlab.exe` with no argument shows the spinning demo triangles, and G/R/S do nothing.
2. `gfxlab.exe assets\cube.obj`: a cube resting on the grid, centred over the origin, with crisp lit edges and one face brighter than the others.
3. `assets\cube_no_normals.obj` looks identical, and the log says the normals were generated.
4. `assets\suzanne_s.obj` shades smoothly and `suzanne_f.obj` stays faceted, both standing upright.
5. A path that does not exist logs an error and exits with code 1.
6. **G**: the model follows the cursor. **G** then **Z**: only up and down. **G** then **X** twice: free again.
7. **R**: the model turns the way the mouse circles, including through more than one revolution. **R** then **Z**: it spins about the vertical axis, also when the camera is below the floor.
8. **S**: moving away enlarges, moving onto the model shrinks without flipping inside out. **S** then **Z** stretches vertically, and the lighting stays correct on the stretched model.
9. Left click keeps the result. Right click or Esc restores the exact previous state. A second Esc quits.
10. **G** after **R** cancels the rotation and starts the grab.
11. No OpenGL debug-callback errors in a Debug build.

## 13. Completion checkpoint

The expected state at the end of M3:

- the OBJ parser handles the supported subset robustly and reports errors with line numbers;
- a model can be loaded from the command line, centred, placed on the floor and framed by the camera;
- it is lit with smooth or flat shading, with normals generated when the file has none;
- the object has a transform, drawn through a backend-neutral draw list, and lit correctly under non-uniform scale;
- G, R and S with axis constraints and confirm/cancel work as described;
- `ctest` shows 7 passing tests on Windows and on a non-Windows host.

Next is M4: the Vulkan backend, drawing the same scene through the same `IRenderer` interface.
