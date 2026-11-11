# M10 — Object Selection, an Outliner and Several Models

> **Status:** Complete
> **Scope:** Models become scenes of objects. Every instance of every loaded file is an object with its own name and transform; the mouse picks objects with a ray through the cursor; a click, Shift+click and a dragged box select them the way Blender does; the selection is drawn as an orange silhouette outline in all three backends (also in the ray traced and path traced views); G/R/S work on the whole selection; an Outliner panel lists the scene, hides objects and loads more model files while the program runs.

## 1. Purpose

Up to M9 a loaded model was one thing. A single transform (`model_transform` in `main.cpp`) moved the whole file, so the three spheres of `materials.glb`, or two files given on the command line, could not be moved separately, and there was no way to say "this object".

M10 adds that. The data model changes first (objects and collections), then the machinery to point at an object (picking), then the user-facing parts: selecting, seeing what is selected, moving a selection, and managing the scene from a list. As elsewhere in this project nothing is a library: the pick ray, the selection rules, the outline in each graphics API, the box select and the UI widgets are in this repository. Ashish chose to do this before the DX12 backend (2026-10-25), so DX12 will get the highlight from the start.

## 2. What M10 adds

| Area | Files |
|---|---|
| Objects and collections | `core/scene.{h,cpp}`: `Scene`, `SceneObject`, `SceneCollection`, `scene_add_model`, `scene_bounds`, `decompose_trs`; `append_model` in `core/model.{h,cpp}`; `ModelInstance::name` |
| Picking | `core/pick.{h,cpp}`: `make_pick_ray`, `PickIndex` (one BVH per mesh, built on first use), `pick_object` |
| Selection rules | `core/scene.{h,cpp}`: `scene_click_select`, `scene_box_select`, `scene_select_all`, `scene_deselect_all`, `scene_selected_count`, `scene_set_hidden`, `scene_set_collection_hidden`, `scene_collection_counts` |
| Box select | `core/box_select.{h,cpp}`: `make_box_grid`, `box_select_objects` |
| G/R/S on a selection | `core/group_edit.{h,cpp}`; `TransformEditor` itself is unchanged |
| Highlight | `renderer.h`: `Highlight`, `DrawItem::highlight`, `edge_color`, `kOutlineRadius`, `kOutlineDepthTolerance`, `any_highlight`; OpenGL `gl_outline.*`; Vulkan `vk_outline.*` with the shaders `mask.*` and `outline.*`; CUDA `outline_kernel` and the mask buffers in `cuda_raster.cu` / `cuda_renderer.cpp` |
| UI | `ui/ui.{h,cpp}`: `Ui::tree_row` (the "selectable" row: folder or item, mark colours, eye icon), `UiRowMark`, `UiRowEvent`, `Ui::right_aligned_x` |
| Loading at run time | `platform/file_dialog.h` and `platform/win32/file_dialog_win32.cpp` (the Open dialog, `comdlg32`); `core/dialog_result.{h,cpp}` (the dialog's multi-select text); in `main.cpp` `add_model_file`, `refresh_scene_info`, `upload_new_parts`, `refresh_material_entries`, `load_models_now` |
| Renderer interface | `FrameData::show_demo`; the draw list skips hidden objects and meshes that failed to upload |
| App | `app/main.cpp`: selection by click and box, group G/R/S, the Outliner panel, Load models |
| Tests | 5 new programs, so 23 in all: `test_scene` (new in M10, grew with each step), `test_pick`, `test_box_select`, `test_group_edit`, `test_dialog_result`; `test_ui` grew (tree rows, right alignment) |

## 3. Step 1a — Objects, collections and several files

A `Model` (M9) is geometry: meshes, images and *instances* ("draw mesh M with this matrix"). It does not know which instances belong to which file or how to move one of them.

The `Scene` adds that. A **collection** is one loaded file (Blender's word). An **object** is one instance, with a name, the index of its mesh, the collection it is in, an editable `Transform` (position, Euler rotation, scale), the `initial` transform (for "Reset transform"), and the flags `selected` and `hidden`. `Scene::active` is the object the panel shows.

- **One Model for all files.** `append_model` appends a file's meshes and images to the one `Model` the renderers see, shifting the new instances' mesh indices and the materials' image indices. The renderers still get one list of meshes and one list of textures.
- **World matrix = `transform.model_matrix() * local`.** An instance's matrix is split into position, rotation and scale by `decompose_trs`, so G/R/S turn and scale an object around its own origin. A matrix that cannot be written that way (shear, which glTF allows through non-uniformly scaled parent nodes) is kept whole in `local`; such an object turns around its collection's origin instead of its own. A mirroring matrix is accepted: the sign goes into `scale.x`.
- **Layout.** Each file is centred as in M9 and then stands on the floor (its lowest point at z = 0). The first file is centred on x = 0, y = 0; each later one is placed to the right (+x) of everything already in the scene, with a gap of 0.25 times the longest side of the larger of the two neighbours.
- **Names.** An object is named after its glTF node, or after the file; equal names inside one collection get `.001`, `.002`, like Blender.
- **The command line** takes any number of files.

The panel's Position, Rotation and Scale fields and the G/R/S keys edit the active object from this step on.

## 4. Step 1b — Picking

`make_pick_ray(view, px, py, ray)` turns a pixel into a ray from the camera. It is the exact inverse of `project_to_screen`, the function G/R/S already use, so a point drawn at pixel (x, y) lies on the ray made for (x, y). A test projects points and checks that the ray goes through them, for several window sizes and fields of view.

`pick_object(scene, model, index, ray, hit)` finds the nearest object under the ray. The triangles stay where they are in mesh space; instead of moving all triangles into the world, the ray is moved into each object's space by the inverse of its world matrix and tested against that mesh's BVH (the M7 `core/bvh`). One BVH per mesh serves any number of objects, and moving, rotating or scaling an object never needs a rebuild. `PickIndex` builds a mesh's BVH the first time it is hit, so a model nobody clicks costs nothing, and meshes appended later (Load models) are handled by the same call.

The ray's direction is **not** normalized when it is moved into object space. Then the distance `t` is the same number in object and world space (the inverse is an affine map), so the hits of differently scaled objects compare directly. Triangles are two-sided. Hidden objects, objects with an empty or missing mesh and objects whose matrix cannot be inverted (a zero scale) are skipped.

`test_pick` compares the BVH result against a brute-force test of every triangle in world space for 400 random rays against real meshes (Suzanne and `materials.glb`) with rotated, stretched and mirrored objects. Two deliberately injected bugs were caught.

## 5. Step 2 — Selecting and showing the selection

### The rules (Blender's)

| Input | Result |
|---|---|
| Click on an object | Only that object is selected, and it becomes active (also when it was one of several selected) |
| Click on empty space | Everything is deselected; `active` is kept |
| Shift+click on an unselected object | Added to the selection, becomes active |
| Shift+click on a selected object that is not active | Stays selected, becomes active |
| Shift+click on the active selected object | Deselected (it stays the active object) |
| Shift+click on empty space | Nothing |
| **A** / **Alt+A** | Select all visible / deselect all (`active` is kept) |

Nothing is selected after loading. The rules live in `scene_click_select` and friends, in portable code with tests, so the 3D view, the box select and the Outliner share them.

### The outline

A selected object gets an orange outline around its silhouette, the active one a lighter orange, like Blender. Every backend does the same two steps:

1. **Mask pass.** Every object is rendered flat into an offscreen mask with its own depth buffer. An unmarked object writes 0, a selected one 0.5, the active one 1. Because the depth test runs against the whole scene, an object in front hides the marked object behind it.
2. **Outline pass.** A fullscreen pass colours a pixel that is not part of a marked object but has a marked pixel within a rounded neighbourhood of `kOutlineRadius` (2) pixels. It skips the pixel if it belongs to something clearly nearer than that marked pixel (more than `kOutlineDepthTolerance` = 2 % of the camera distance), so nothing is drawn on an occluder; the occluder's own border is drawn instead. Active wins over selected. The distance is recovered from the window depth `d` with `z = n*f / (f - d*(f-n))`, which is the same for OpenGL's and Vulkan's depth range.

Only the solid views get the outline. In the wireframe view (**W**) the selected objects' edges are coloured instead (`edge_color`: grey, orange, lighter orange).

| Backend | How |
|---|---|
| OpenGL | `GLOutline`: a framebuffer with an R8 mask texture and a D32F depth texture (resized with the window), `GLMeshRenderer::draw_mask`, and a fullscreen-triangle shader that reads both with `texelFetch`. Drawn after the grid, before the UI |
| Vulkan | `VulkanOutline`: mask and depth images shared by the frames in flight; the mask pass is its own `vkCmdBeginRendering` before the main one (barriers: undefined → attachment → shader read), the outline is drawn inside the main rendering after the grid. `outline.frag` needs `GL_EXT_samplerless_texture_functions` to use `texelFetch` on a sampled image. The 128-byte push block is shared by the mesh, wire and mask pipelines (`base_color.x` carries the mask value) |
| CUDA | The selected and the active objects are rasterized *alone* (a worklist round of their own) into two extra visibility buffers; `outline_kernel` compares them with the main visibility buffer. A marked object counts as visible where its distance is at most the main distance times (1 + tolerance). Works in solid, **T** and **P** modes, because the tracers write the same visibility buffer |

## 6. Step 3 — Box select

A left press in the 3D view only records where it was pressed. If the mouse then moves more than 4 pixels it becomes a box: a translucent white rectangle (fill and 1-pixel outline, added to the UI draw list *before* the panels so panels draw over it) follows the mouse. The selection happens when the button is **released**; if the mouse stayed within 4 pixels it is the ordinary click at the press position. This is why selection moved from press to release in this step.

`box_select_objects` covers the rectangle with a grid of pick rays, one per cell of about 4 pixels, capped at 16384 rays (the cells grow when the box is large), and returns every object that is the nearest hit of at least one ray. An object completely behind another is therefore not selected, as in Blender's solid view. The grid is cut to the window; a box with no width or height still gets one row or column; a box outside the window or with a NaN coordinate gives nothing. Plain box: the selection becomes exactly the found objects (an empty box deselects everything); Shift: they are added. A box select never changes `active`, like Blender. Esc or a right click while dragging cancels the box (and that Esc does not also quit); starting G/R/S during a drag drops it.

The price of using pick rays instead of an ID buffer: an object smaller than the grid spacing, or with a visible sliver thinner than it, can fall between the rays and be missed. An ID buffer in every backend would be exact, but it was out of scope.

## 7. Step 4 — Groups, hiding, the Outliner and Load models

### G/R/S on the selection

`TransformEditor` turns the mouse into a change of *one* transform. For a group it is given a made-up transform at the **median** of the selected objects' positions (Blender's default pivot), with no rotation and scale 1. Whatever the editor does to that transform is applied to every selected object by `group_edit_apply`:

- **Grab:** all objects move by the same distance.
- **Rotate:** each object's offset from the pivot is turned, and its own orientation turns by the same rotation.
- **Scale:** the offset from the pivot is scaled along the world axes (only the constrained axis after **X**/**Y**/**Z**), and the object's own scale is multiplied by the same factors, as the editor does for one object.

With one selected object the pivot is its own position, and `test_group_edit` proves that the result equals what the editor gives for that object alone, for every mode and axis. Every update starts from the transforms saved at the start (like the editor, so nothing accumulates), and cancel restores them exactly. G/R/S need at least one selected object (no longer the active one); the panel's fields still edit the active object only.

### Hiding

`scene_set_hidden` hides or shows an object; hiding also deselects it, so G/R/S can never move something you cannot see (it stays the active object if it was). A hidden object is not drawn, not pickable and not box selectable. `scene_set_collection_hidden` does it for a whole file. OpenGL and Vulkan draw their demo triangles when the draw list is empty, which hiding everything would trigger; the app now sets `FrameData::show_demo = !has_model`.

### The Outliner

A second panel at the right edge (it shows and hides with the left one: **H**, **<< Hide**, the **>>** tab). From top to bottom: the **Load models...** button, a status line, the "Scene Collection" root, one folder row per collection (fold arrow, an eye that hides all its objects) and one row per object. A click on an object row uses `scene_click_select` (Shift extends); the selected rows are blue and the active one brighter, in step with the viewport. The panel scrolls when it is taller than the window.

The rows come from the new widget `Ui::tree_row(key, label, indent, mark, open, visible)`. A row is a folder when `open` is given (any click flips it) and an item otherwise (a click is reported); `visible` adds the eye icon, which flips the flag and is reported separately from the label click. The `key` identifies the row (like a label does for the other widgets), so two objects with the same name in different files do not mix up. `Ui::right_aligned_x` puts a panel flush to the window's right edge using the width it had last frame, so a panel that grows stays inside the window.

### Load models

The button only sets a flag. The main loop opens the dialog at the top of the next frame, because the dialog blocks (it is modal) and must never run inside the render lambda (which also runs while the window is dragged).

- `open_model_files_dialog` calls `GetOpenFileNameW` with `OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_NOCHANGEDIR` and returns UTF-8 paths. With several files the dialog returns the folder and then the names, separated by zero characters; the portable `parse_multi_select_buffer` turns that into full paths and is tested on any system.
- The per-file code that used to be in the start-up loop is now `add_model_file`. It either adds a complete collection or changes nothing and returns a reason. Missing files are looked up like command-line arguments.
- `upload_new_parts` creates only the textures and meshes the renderer does not have yet (`texture_ids` and `mesh_ids` stay parallel to `model.images` and `model.meshes`). A mesh that fails to upload keeps id 0 and is skipped when drawing; the status line says so. The three renderers needed **no change**: `create_texture` and `create_mesh` already worked after start-up (the Vulkan descriptor pools grow, the CUDA and OpenGL stores append). The CUDA path tracer's change signature includes every draw item, so new objects restart its accumulation.
- `refresh_material_entries` (CUDA): the material editor's list is rebuilt from all meshes, but an entry that existed before (same index and name) keeps its edited values, the selected entry stays selected, and the edited values are sent to the new meshes that share the material.
- The new collections stand to the right of the old ones, the camera frames the whole scene (the first load also resets the view angles) and the selection is left alone. Starting without a file and loading one with the button works: the demo triangles go away.

## 8. Design decisions and why

| Decision | Why |
|---|---|
| Objects and collections in a separate `Scene`, over one `Model` | The renderers keep one list of meshes and textures; many files and many objects cost no new renderer concept |
| Pick ray moved into object space, direction not normalized | One BVH per mesh for any number of objects, no rebuild when an object moves, and distances comparable across scaled objects |
| Selection rules in portable code (`scene_click_select`, `scene_box_select`) | The 3D view, the box and the Outliner share them, and they are unit tested |
| Silhouette outline (mask + dilate) instead of an edge overlay | The first version, an edge overlay with a depth bias, flickered at grazing angles and outlined every edge of dense meshes. The mask version looks like Blender's and needs one extra pass only while something is selected |
| Only three mask levels (none, selected, active), not an ID per object | Cheap in all three backends; the price is that two overlapping selected objects get no border between them |
| Box select by pick rays instead of an ID buffer | Reuses `pick_object`, respects occlusion, and needs no change in the renderers; the price is a 4 px ray spacing |
| Selection on release, not press | A click and a box drag start the same way; only the release tells them apart |
| Median pivot, applied through the existing `TransformEditor` | Blender's default; the editor and its tests stay untouched, and one object behaves exactly as before |
| Hiding deselects | Group G/R/S then never touches an invisible object |
| Dialog opened by the main loop, not by the UI code | The dialog is modal and blocks; it must not run inside the render lambda |
| Upload only what is new | `mesh_ids` and `texture_ids` stay aligned with the model, and a later load never re-uploads old meshes |
| The Outliner shows and hides with the left panel | One key (**H**) for all panels; no second visibility state |
| Measurements dropped | Ashish's decision (2026-11-11): selection and the list cost nothing worth a table; the only number kept is the BVH build time from Step 1b (Section 10) |

## 9. Problems found during M10

| Problem | Cause and fix |
|---|---|
| Flickering selection edges | The first highlight drew the mesh edges over the solid view with a depth bias; at grazing angles it z-fought. Replaced by the mask and outline passes |
| Vulkan shader build error in `outline.frag` | `texelFetch` and `textureSize` on a sampled image need `GL_EXT_samplerless_texture_functions`; the `#extension` line was added |
| Orange dots inside objects in the path tracer (**P**) | The path tracer writes the nearest depth of its jittered samples, a little nearer than the raster depth, so the 0.1 % slack of the CUDA visibility test was too small. It now uses the same 2 % tolerance as the outline |
| Hiding every object showed the demo triangles (found while designing the eye icon) | OpenGL and Vulkan draw them for an empty draw list. `FrameData::show_demo` |

## 10. Measurements

Dropped by decision. The only reference number is from Step 1b on Linux (Release): the first pick on `suzanne_s` builds its BVH (15,744 triangles) in about 20 ms, and later picks take well under a microsecond each. The outline passes, the box select (at most 16384 rays) and the Outliner were not timed.

## 11. Verification

- `ctest`: 23 programs, all passing in Debug and Release on Linux with `-Wall -Wextra -Wconversion -Wshadow`. New in M10: `test_scene` (decomposition and its round trip, mirrors and rejects, appending models, the layout of one and of two files, names, shear, the selection rules, hiding, the real asset files), `test_pick` (ray corners and the inverse of the projection, hit and miss, two sides, object transforms, the nearest object, hidden and empty objects, the BVH cache, 400 random rays against brute force), `test_box_select` (the grid, its cap, clipping, degenerate boxes, occlusion, hidden objects, replace and extend), `test_group_edit` (the median pivot, grab, rotate, scale, repeatability and restore, one object against the editor, cancel through the editor) and `test_dialog_result`. `test_ui` gained tree rows, a scrolling tree and right alignment.
- Files that need the GPU or Windows could not be built in the sandbox (no MSVC, nvcc, glslc or GPU). They were type-checked against stubs and the real Vulkan headers, the CUDA outline and wireframe kernels were run on the CPU against known answers, and the shaders were reviewed by eye. Ashish built every step on Windows and checked the checkpoints before pushing; that is the real test of the GPU parts.

## 12. Manual checklist

1. `gltf\materials.glb` in each backend: nothing is selected at start. A click on a sphere gives it an orange outline (lighter for the active one), a click on empty space clears the selection, **Shift**+click adds and removes, **A** selects all, **Alt+A** none.
2. The outline stays on the silhouette and does not flicker while orbiting; it is hidden where another object is in front; with **T** and with **P** (CUDA) there are no dots inside objects; **W** colours the selected objects' edges orange.
3. Drag a rectangle over empty space: a translucent rectangle follows the mouse; on release the objects seen through it are selected. A small move (under 4 pixels) is a click. **Shift**+drag adds. An object hidden behind another is not selected. Esc or a right click during the drag cancels it without quitting.
4. Two or three objects selected: **G**, **R** and **S** move, turn and scale them together around their middle; **X**/**Y**/**Z** constrain the axis; left click confirms; right click or Esc restores everything. With one object selected the behaviour is the same as before M10; with none selected the keys do nothing.
5. `gltf\materials.glb gltf\cubes.glb`: the Outliner lists "Scene Collection", two collections and their objects. Clicking rows selects like clicks in the viewport; selection in the viewport lights the rows. The folder arrow folds a collection; the eye hides an object or a whole file; hidden objects cannot be clicked or boxed and are deselected; hiding everything shows an empty scene, not the demo triangles.
6. **Load models...** with several files: they appear to the right of the existing ones, the camera frames everything, the status line says what was loaded; a bad file gives a "Failed" line and changes nothing. Try it in OpenGL, Vulkan and CUDA, and starting without any file.
7. CUDA: edit a material, load another file, and the edit is kept; moving or hiding objects restarts the path tracer.
8. **H** and **<< Hide** hide both panels, the **>>** tab brings them back; a long file name widens the Outliner to the left, not out of the window; a short window makes both panels scroll.

## 13. Known scope and limitations

| Item | Notes |
|---|---|
| **Outline** | A hard 2 px line without anti-aliasing; two overlapping selected objects get no border between them (three mask levels, not one id per object); a surface more than 2 % nearer than the marked object it touches gets no outline (for example a very steep grazing floor) |
| **Box select** | Rays every 4 px, at most 16384: objects smaller than that can be missed |
| **Instance limit** | The CUDA ray and path tracers handle `kMaxRtInstances` = 8 draw items; objects beyond that are left out (a warning is logged once), so a big scene loaded later is only partly traced |
| **Outliner** | No collection selection, rename, delete, drag and drop or parenting; two files with the same name show the same folder label; with a narrow window the two panels can overlap |
| **Load models** | The Windows dialog is modal: the window does not repaint while it is open. Loading several large files blocks the window until they are all uploaded |
| **Pivot** | Always the median of the selected objects' origins; there is no choice of pivot, no local or view orientation and no snapping |
| **Panel fields** | Position, Rotation and Scale edit the active object only, not the selection |
| **Transforms are not saved** | Edits live in memory, as the material edits do |
| **Sheared instances** | An object whose matrix has shear turns around its collection's origin, not its own |
| **Not selectable** | There is no way to select by name or by material, or to select a whole collection |

## 14. Completion checkpoint

- every instance of every loaded file is an object with its own name and transform, and a file is a collection; several files can be given on the command line or loaded while the program runs;
- a ray through the cursor finds the nearest object using one BVH per mesh;
- click, Shift+click, a dragged box, **A** and **Alt+A** select the way Blender does, and the selection is shown as an outline in OpenGL, Vulkan and CUDA (also in **T** and **P**);
- G/R/S act on the whole selection around its median;
- an Outliner panel lists the scene, selects in step with the viewport, hides objects and collections, and loads more files;
- `ctest` shows 23 passing tests.

Next is M11, the DX12 backend.
