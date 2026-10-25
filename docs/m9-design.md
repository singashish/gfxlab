# M9 — Materials, Textures and a CUDA Path Tracer

> **Status:** Complete (Steps 1–8b)  
> **Scope:** Materials and textures end to end: texture coordinates and a MTL parser, PNG and TGA decoders written from scratch (including Deflate), a JSON parser and a glTF/GLB loader, textured and per-material drawing in all three rasterizers, a progressive CUDA path tracer ("Rendered" mode, **P**) with a glossy GGX material model, emissive surfaces and sampled lights, a panel for the path tracer's settings, and a live material editor. The panel also learned to hide and to scroll.

## 1. Purpose

Up to M8 a model was a grey, lit shape: the only thing a triangle knew about its look was the shading of the viewport's key light. Real models come with materials (colour, roughness, metal, glow) and textures, in formats that need real parsers.

M9 adds all of that. A model can now come from an OBJ with an MTL file or from a glTF/GLB file; each triangle has a material; the three rasterizers draw the base colour and its texture; and the CUDA backend gets a second, physically based way to look at the same scene: a path tracer that follows light bouncing around, so emission lights its surroundings, metals reflect, and rough surfaces blur the reflections. A material editor in the panel changes the material values while the picture converges. As elsewhere in this project, nothing is a library: image decoding, JSON, glTF, the GPU material model and the sampling maths are all in this repository.

## 2. What M9 adds

| Area | Files |
|---|---|
| Model and materials | `core/material.h` (`Material`), `core/mesh.*` (`texcoords`, `materials`, `triangle_materials`, `group_triangles_by_material`), `core/obj_loader.*` (texture coordinates, `usemtl`/`mtllib`, `parse_mtl`), `core/model.*` (`Model`, `load_model`, `model_bounds`, `center_model_at_origin`) |
| Images and containers | `core/inflate.*` (Deflate and zlib), `core/image.*` (PNG and TGA decoders, `load_image`), `core/base64.*` |
| JSON and glTF | `core/json.*` (parser), `core/gltf_loader.*` (`.gltf` and `.glb`) |
| Sampling maths (portable, also compiled for the GPU) | `core/texture_sampling.h` (mip chains, bilinear, trilinear, LOD), `core/path_math.h` (sRGB, tone mapping, sky, sun, cosine sampling, Russian roulette, firefly clamp), `core/brdf.h` (GGX, Smith, Fresnel, VNDF sampling, MIS helpers), `core/light_sampling.h` and `core/light_list.h` (emissive triangles as lights), `core/path_settings.h` (the panel's settings), `core/material_edit.h` (editable values, clamping rules, the editor's material list) |
| Renderer interface | `renderer.h`: `Image`/`TextureId`, `IRenderer::create_texture`, `create_mesh(mesh, textures)`, `update_material`, `FrameData::show_materials`, the sun and `path_settings` fields |
| Textures per backend | `opengl/gl_texture.*` (`GLTextureStore`), `vulkan/vk_texture.*` (`VulkanTextureStore`), `cuda/cuda_texture.*` (`CudaTextureStore`); material-aware drawing in `gl_mesh.cpp`, `vk_mesh.*` with `mesh.vert`/`mesh.frag`, and `cuda_raster.cu` |
| CUDA material table and lights | `cuda/cuda_mesh.*`: `CudaMaterial` table, per-mesh light lists, `update_material`, `revision()` |
| CUDA path tracer | `cuda/cuda_pathtrace.cuh` (materials, light sampling, `trace_path`, kernels) and `cuda/cuda_rt_device.cuh` (rays shared with the ray tracer); both are included into `cuda_raster.cu`. The host side is in `cuda_renderer.cpp` |
| UI | `ui/ui.*`: wheel and viewport height in `UiInput`, scrolling panels |
| App | `app/main.cpp`: Path tracing section with Settings group, panel hide/show, Materials editor, **P** and **M** |
| Tests | 8 new programs, so 18 in all: `test_image`, `test_texture`, `test_json`, `test_gltf`, `test_path`, `test_brdf`, `test_light`, `test_material_edit`; `test_obj` and `test_ui` grew. Generators: `tests/make_image_test_data.py` (into `image_test_data.h`), `make_gltf_test_assets.py`, `make_material_test_model.py` |
| Assets | `textured_cube.obj` + `.mtl`, `textures/uv_grid.png`, `textures/label.tga`, `gltf/cubes.gltf` (+ `.bin`) and `cubes.glb`, `gltf/materials.glb` |

## 3. Step 1 and 2 — Materials, OBJ/MTL and image loading

### Material and texture coordinates (Step 1)

`Material` is a small version of Blender's Principled BSDF: `base_color`, `roughness`, `metallic`, `emission`, and four texture slots (`base_color_image`, `metallic_roughness_image`, `normal_image`, `emissive_image`) that are indices into `Model::images` (-1 = none). The images live in the model, not in the materials, so materials stay cheap to copy and many can share one image.

A `Mesh` gained `texcoords` (one per vertex, v up as in OBJ), `materials` and `triangle_materials` (one material index per triangle). `group_triangles_by_material` reorders the triangles so each material's are contiguous, which is what a rasterizer needs to draw one range per material. A mesh without the list (or with a list of the wrong length) counts as all material 0.

The OBJ parser reads `vt` and the `v/vt/vn` face forms, splits vertices at texture seams, drops partial texture coordinates, and handles `usemtl`/`mtllib`. `parse_mtl` reads `newmtl`, `Kd`, `Ke`, `Ns`, `Pr`, `Pm` and `map_Kd` (its options such as `-s` are skipped); roughness comes from `Pr` when given and otherwise from the Phong exponent with the mapping Blender's importer uses (`Ns` 1000 gives 0, `Ns` 0 gives 1).

### Images (Step 2)

PNG needs Deflate, so `core/inflate.*` implements RFC 1951 (stored, fixed and dynamic Huffman blocks, LZ77 copies) and its zlib wrapper (RFC 1950, with the Adler-32 check). `core/image.*` decodes PNG with CRC-32 checks, all colour types and bit depths (1 to 16 bits), the five row filters, Adam7 interlacing and `tRNS` transparency, and TGA (uncompressed and RLE, 24/32-bit true colour and 8-bit grey). Every decoder returns the same thing: 8-bit RGBA, top row first (16-bit channels keep their high byte). `load_image` picks the decoder by extension (`kMaxImageSize` is 16384 pixels per side).

## 4. Step 3 and 4 — JSON and the glTF/GLB loader

`core/json.*` is an RFC 8259 parser. A `JsonValue` tree is read-only and forgiving to query: a missing key, a bad index or a value of another type gives a "null" value, never an exception, so `doc["nodes"][2]["mesh"].as_int(-1)` needs no checks at each step. Errors carry line and column; nesting depth and number ranges are limited.

`core/gltf_loader.*` turns a `.gltf` (external or base64 data) or a `.glb` (binary container) into a `Model`:

- **Axes and texture coordinates.** glTF is Y-up and puts (0, 0) at the top-left of an image. The loader rotates vertices, normals and instance transforms into gfxlab's Z-up at the end, and stores `v` as `1 - v` to follow the OBJ convention of `Mesh::texcoords`.
- **Scene graph.** The node tree is flattened to `ModelInstance`s (a mesh index plus a transform), so a mesh used by several nodes is stored once. Cycles and excessive depth are rejected.
- **Accessors.** Offsets, strides, 8/16/32-bit indices and triangle strips and fans (converted to triangle lists) are handled, and accessors shared between primitives are read from the same buffer.
- **Materials.** Base colour, metallic and roughness factors, emissive factor, and the four texture slots. `KHR_materials_emissive_strength` multiplies the emissive factor. Every `Mesh` gets a copy of the file's whole material list (plus a "default" material when a primitive names none), so `triangle_materials` indexes into any mesh's list directly.
- **Out of scope.** JPEG textures (a warning; the material keeps its factors), texture samplers and `KHR_texture_transform` (warnings), sparse accessors (the primitive is skipped), cameras, lights, skins, animations and morph targets (ignored). Files that require an extension the loader cannot honour, such as Draco or meshopt compression, are refused with an error.

`load_model` chooses OBJ or glTF by extension (`.obj`, `.gltf`, `.glb`), makes sure every mesh has normals and edges, and centres the model by moving the instances, never the vertices (instances share vertices).

## 5. Step 5 — Materials in the three rasterizers

All three backends draw the same thing: one draw per material range of each mesh, with the material's base colour multiplied by its base-colour texture. The shading stays in display values, as before: textures are uploaded as plain RGBA8 and no sRGB conversion happens in the viewport, so the key-light shading of M3 is unchanged. **M** switches back to the plain grey shading.

- **Texture ids.** `IRenderer::create_texture(image)` returns a `TextureId` (0 means "not supported or failed"); `create_mesh(mesh, textures)` receives `textures[i]` for `Model::images[i]`. A surface with no texture uses a 1x1 white texture owned by the store, so the shaders never branch on "has a texture".
- **OpenGL (5a).** `GLTextureStore` uploads with `glGenerateMipmap` and trilinear filtering; the base colour is a per-draw uniform.
- **Vulkan (5b).** `VulkanTextureStore` creates one descriptor set per texture (set 1, one shared sampler and layout, a full mip chain per image); the mesh renderer binds the set for each draw. The base colour travels in the push constants (`vec4 base_color` at offset 112, inside the shared 128-byte range declared identically in `mesh.vert`, `mesh.frag` and `wire.vert`).
- **CUDA (5c).** The CUDA backend does not use CUDA texture objects. `CudaTextureStore` keeps each texture as a device buffer holding its whole mip chain (RGBA8) with a `TextureView` in a small device table; entry 0 is the white texture, and a `TextureId` is its own table index. The kernels sample with `core/texture_sampling.h`, the same code the tests run on the CPU. The deferred shading kernel needs the mip level, and a thread shading one pixel has no 2x2 quad to take derivatives from, so it evaluates the triangle's UV interpolation one pixel to the right and one pixel down itself and takes the differences (`texture_lod`, OpenGL's formula: the longer side of the footprint, no anisotropy). A neighbour past the surface's horizon gets the blurriest level. Texture coordinates travel through near-plane and guard-band clipping in the free `w` components of the clip vertices, so they interpolate correctly across clipped polygons.

## 6. Step 6 and 6b — The path tracer

**P** (CUDA backend, a model loaded) switches to "Rendered" mode, a progressive path tracer that excludes the ray tracer (**T**). Each frame adds `samples_per_frame` samples per pixel to a float4 accumulation buffer (the sum of the linear radiance, and the closest camera-hit depth); the picture shown is the sum divided by the sample count, multiplied by the exposure, tone mapped and encoded. It stops adding at `max_samples` (default 1024), after which a frame costs almost nothing.

- **Linear light.** All arithmetic is in linear light: stored colours are sRGB-decoded (`decode_srgb`), and only the final picture goes through the ACES fit by Narkowicz (`tonemap_aces`) and `encode_srgb`.
- **Bounces.** Diffuse surfaces are sampled with a cosine-weighted hemisphere, which cancels the cosine and the pi, so one bounce multiplies the throughput by the albedo. A path shades at most `max_bounces` surfaces (default 4); from the third bounce on Russian roulette ends paths with a probability from the throughput, dividing the survivors so the average is unchanged.
- **Sun.** A cone of 1 degree half angle (`kSunHalfAngle`; the real sun is smaller, a bigger disc gives softer shadows) with an irradiance of `kSunIrradiance` (5.0). At every hit one shadow ray (next-event estimation) goes to a random point of the disc; a bounce ray that happens to hit the sun adds nothing, so the sun is never counted twice.
- **Sky.** A procedural sky (`sky_radiance`, scaled by `kSkyScale`) lights surfaces for bounce rays that miss.
- **Sun and sky are lights, not scenery (6b).** Camera rays that miss everything see the viewport background colour, not the sky or the sun. `background_radiance` runs the display pipeline backwards (sRGB decode, inverse tone map, divide by the exposure), so the background stays exactly the viewport colour and edge pixels blend with it correctly.
- **Restarting.** The accumulation restarts when anything that changes the picture changes: the camera, the models' transforms, the sun direction and on/off, the materials switch, the number of meshes and textures, the material edit counter (`CudaMeshStore::revision()`), and the settings that change the collected light (bounces, sun strength, sky strength, firefly limit). The exposure and the two sample counts do not restart it.

## 7. Step 7a–7f — Materials in the path tracer

### Material data on the GPU (7a)

`CudaMeshStore` appends every mesh's materials to one renderer-wide table in device memory. A `CudaMaterial` is 48 bytes (three 16-byte groups): `base_color` and `texture`; `emission` and `emissive_texture`; `roughness`, `metallic` and `metallic_roughness_texture`. A triangle's entry is `material_base + triangle_materials[t]`, and each mesh's slice ends with a fallback entry (the default material) for out-of-range numbers. Texture slots always hold a valid table index (0 = white). The rasterizer reads only the first 16 bytes. Values are cleaned on the CPU (roughness and metallic to 0..1, base colour to 0..1, emission finite and not negative, but unlimited above), by the same `sanitized()` the editor uses.

The path tracer multiplies the base colour by its texture (sRGB) and decodes it to linear for the albedo, multiplies roughness and metallic by the G and B channels of the metallic-roughness texture (which is data, used as stored), and adds emission times the emissive texture. A ray that reaches an emissive surface by chance picks up its radiance.

### The glossy layer (7b)

A surface is a Lambert diffuse layer plus a GGX specular lobe, the metallic-roughness model of glTF (`core/brdf.h`): `D` (GGX normal distribution) times `Vis` (height-correlated Smith shadowing and masking, divided by the BRDF's denominator) times `F` (Schlick's Fresnel). The specular `F0` blends from 0.04 (`kDielectricF0`) for dielectrics to the base colour for metals; the diffuse colour is zero for metals and, for dielectrics, the albedo reduced by what the glossy layer already reflects. The roughness slider maps to `alpha = roughness^2` with a floor (`kGgxMinAlpha`) so a perfect mirror stays a very sharp but hittable lobe. For each bounce the lobe is chosen with a probability from how much each lobe reflects (limited to 0.1 to 0.9, and exactly 0 or 1 when one lobe reflects nothing), and the glossy direction is sampled from the visible normals (VNDF), which wastes no samples on microfacets facing away. `M` off removes the glossy layer and emission and gives the plain matte grey of the viewport.

### Combined sun sampling (7c)

For a glossy surface the sun can be found by the shadow ray (good for rough surfaces) or by the bounce ray (good for smooth ones, where the lobe is narrow and the sun small). Multiple importance sampling (MIS) with the power heuristic combines both, weighting each sample by how likely the other strategy would have been to produce the same direction.

### Emissive surfaces as lights (7d)

A small lamp found only by chance makes a speckled picture. Every mesh now has a light list (`core/light_list.h`): its emissive triangles in ascending order and the cumulative sum of their weights (object-space area times the luminance of the emission). At each hit the path tracer picks an instance by its scaled total weight, a triangle by binary search on the cumulative sum, a point on the triangle with barycentric sampling, and traces a shadow ray; the density is converted from area to solid angle (`pdf_solid_angle = pdf_area * dist^2 / cos_light`) and combined with the bounce rays by MIS, so aimed rays win for small lamps and bounce rays for large ones close by.

### The firefly limit (7e, 7f)

A "firefly" is a pixel whose average jumps because one sample carried an enormous value. Here the usual cause is a caustic path such as camera, floor, shiny sphere, mirror bounce, sun: the sun is about 5000 times brighter than the sky, no shadow ray can be aimed at it through a mirror, and bounce rays find it only rarely. `clamp_contribution` limits the value one indirect sample may add to a pixel to `kPtIndirectClamp` (10, in linear light, per channel). 7e clamped paths after a diffuse bounce; 7f widened it to every path from the third ray or vertex on (`bounce >= 2`), because the sun seen through two glossy reflections is just as rare and bright. Paths with no diffuse bounce and at most one bounce (a lamp seen directly, a sun glint, a mirror reflection of the sun) are never clamped.

The clamp is biased on purpose, so real caustics, such as the bright spot the sun makes under a shiny sphere, come out dimmer than in a fully converged unclamped render. It can be switched off in the panel (Section 8).

### The file split (7f)

The path tracer grew large, so it moved out of `cuda_raster.cu` into `cuda_pathtrace.cuh` and `cuda_rt_device.cuh`. They are headers included into `cuda_raster.cu` at the place where their text used to be, in the same translation unit and the same anonymous namespace, not separate `.cu` files: a second `.cu` would need relocatable device code and device linking, which this project does not use.

## 8. Step 8 — The panel

### Path tracing settings (8a)

The panel's Path tracing section has a folded **Settings** group (`core/path_settings.h`, `PathTraceSettings`): Max samples (16 to 4096), Samples / frame (1 to 16), Bounces (1 to 12), Exposure (-4 to +4 EV), Sun strength (0 to 20), Sky strength (0 to 4), a Firefly limit checkbox and Reset settings. The defaults are the constants of `path_math.h`, so a fresh `PathTraceSettings` reproduces the picture from before the settings existed. The sliders edit floats, so the three whole-number settings keep a float copy owned by the slider and the setting is the rounded copy each frame; the copy is never rewritten from the rounded value, so a drag stays smooth.

### Hiding the panel (8a)

The panel's **<< Hide** button (or **H**) collapses it; a small **>>** tab at the left edge brings it back. The mouse ownership rules of M8 apply to the tab as well.

### Scrolling (8a-2)

When the window is shorter than the panel, the panel used to run past the bottom and its lower widgets were unreachable. `UiInput` now carries the viewport height and the wheel. A panel taller than the window (minus a 10 px margin) is clamped to that height and goes into scroll mode: its title and separator stay, the body moves by a per-panel offset, a thin scrollbar sits inside the right edge, and the wheel (3 text lines per notch) or a drag on the thumb scrolls it. The draw list has no clipping, so rows not fully inside the visible body are simply not emitted and ignore the mouse (`header()` still returns its `*open` value, other widgets return false), and nothing is ever drawn outside the panel, in any of the three backends. The scrollbar's width is added to the panel width in scroll mode, while the remembered "wanted" width never includes it, so the width cannot grow from frame to frame. Zoom is skipped while the mouse is over the panel. A panel that fits keeps the old code path and an identical draw list. Whether to scroll is decided from the previous frame's content height, so after a resize or an unfold the layout needs one frame to catch up.

### The material editor (8b)

The **Materials** section (folded by default; `--backend=cuda` only, the other backends show "Needs --backend=cuda" and the section is disabled) edits the loaded model's materials live.

- **The list.** `build_material_entries` lists the materials used by at least one triangle of some mesh. This matters because every glTF mesh carries a copy of the whole file's material list. An entry is identified by (index k, name) and remembers its targets: every (mesh, k) with the same index and name, so one edit updates every mesh's copy.
- **The widgets.** A Material slider (not shown when there is only one entry) with a line "k: name"; Base R/G/B, Roughness, Metallic, Emission R/G/B (0 to 1) and Emission strength (0 to 50); Reset material and Reset all materials. Emission is kept as a colour and a strength (`split_emission` / `combine_emission`: the strength is the largest component), so a strength of 0 does not lose the colour. The base-colour texture is not editable: the colour factor still multiplies it.
- **Applying an edit.** Only on frames where something changed, the app calls `IRenderer::update_material(mesh, index, params)` for each target. `CudaMeshStore::update_material` validates the id and index, sanitizes the values, uploads the one 48-byte entry into the existing device table (the pointer stays valid, texture slots are untouched), and increments `revision()`, which restarts the path tracer's accumulation.
- **Lights.** If the luminance of the emission changed, the mesh's light list is rebuilt from the stored object-space triangle areas and per-triangle material numbers (8 bytes per triangle kept on the host), the new arrays are uploaded, and the old ones are freed only after the entry is in, so a failed upload leaves everything unchanged. Colour, roughness and metallic edits cost one small copy. The ray tracer (**T**) needs no restart: its accumulation holds only how many AO rays escaped, and colours are applied each frame.
- **Not saved.** The editor state is the truth for the session; the `Material` copies in the `Model` are not updated and nothing is written back to the file.

## 9. Design decisions and why

| Decision | Why |
|---|---|
| Software texture sampling in CUDA instead of texture objects | The same header-only code runs on the GPU and in the CPU tests, so the maths is tested; the LOD formula and the wrap mode are under our control. Costs some speed against the hardware units |
| The path tracer's maths in portable headers (`path_math.h`, `brdf.h`, `light_sampling.h`) | The tests run the very same functions the kernels use, so normalisation, sampling densities and MIS weights are checked on the CPU |
| Firefly clamp instead of unbiased handling | A caustic path through a mirror cannot be sampled by a shadow ray; a clamp removes the fireflies cheaply. The price is dimmer caustic light, and the setting can be turned off |
| `materials.glb` has ONE mesh with ten primitives | The ray and path tracers handle at most `kMaxRtInstances` (8) draw items, and each node is one item; ten spheres as ten nodes would have left items out |
| Single-scattering GGX | Simple and standard; the price is that rough metals lose energy (Section 14) |
| glTF: a copy of the material list in every mesh | Keeps `triangle_materials` a plain index into the mesh's own list, as in the OBJ loader. The consequences are a duplicated material table and light list per mesh, and the editor's (index, name) targets |
| Display-space shading in the rasterizers, linear light in the path tracer | The rasterized viewport keeps the look of M3 to M8 and stays identical in three backends; the path tracer needs linear light for correct energy |
| Sun and sky are only lights | A path-traced picture should not show a sun disc or a sky dome behind the model; the viewport background stays |
| Path tracer in headers included into `cuda_raster.cu` | Avoids device linking and relocatable device code |
| Scrolling without clip rectangles | The three UI backends consume one draw list with no clipping; not emitting hidden rows needed no backend change |

## 10. Problems found during M9

| Problem | Cause and fix |
|---|---|
| The sky and sun were visible in the path-traced picture | Camera rays that missed everything returned the sky. Step 6b: they return the viewport background (`background_radiance`), and sky and sun are only lights |
| Fireflies under glossy spheres | Caustic paths ending in the sun (7e). First limited after a diffuse bounce; the sun through two glossy reflections slipped past, so 7f clamps from the third ray on |
| The panel ran past the bottom of a small window | No scrolling (8a-2): scroll mode, pinned title, scrollbar |
| The Materials header and the View section's "Materials" checkbox share a label | Widget ids come from labels (M8). The section's header label ends in a space |
| Ten spheres did not fit as ten nodes | `kMaxRtInstances` is 8: one mesh with ten primitives |
| MSVC pitfalls in the code and tests | Raw string literals and `\u` escapes in test sources, `CHECK` on constants and lambdas inside `CHECK`, `near`/`far`/`min`/`max`/`small` as identifiers (`windows.h` macros), implicit narrowing and shadowing under `/W4 /permissive-`. They were avoided by habit from Step 2 on |

## 11. Measurements

Release, vsync off, 1280×720, `gltf\materials.glb` at the startup camera, RTX 5080, window focused. Frame ms / GPU ms from the once-per-second log line.

### Textures and materials in the rasterizers (M off → on)

| Backend | Frame ms | GPU ms |
|---|---|---|
| OpenGL | 0.081 → 0.082 | 0.011 → 0.010 |
| Vulkan | 0.183 → 0.184 | 0.014 → 0.014 |
| CUDA | 0.481 → 0.480 | 0.480 → 0.480 |

Textures and per-material colours cost nothing that can be measured in any backend. In CUDA the software trilinear sampling is hidden inside the shading kernel, whose time is dominated by other work at this resolution.

### CUDA path tracer (P), one sample per frame

| Case | GPU ms per sample | Time to 1024 samples |
|---|---|---|
| 1 bounce, sun on | 0.640 | 0.66 s |
| 4 bounces (default), sun on | 0.817 | 0.84 s |
| 8 bounces, sun on | 0.850 | 0.87 s |
| 4 bounces, sun off | 0.756 | 0.77 s |
| `suzanne_s.obj`, 4 bounces, sun on | 0.824 | 0.84 s |

- **Bounces.** Going from 1 to 4 bounces costs 28 % more, and from 4 to 8 only 4 % more. Most paths end early: they leave the scene (the model sits on a floor with open air around it) or are cut by Russian roulette from bounce 3, so the later bounces have few paths left to trace.
- **Sun.** Next-event estimation for the sun adds about 8 % (one extra shadow ray per hit).
- **Scene size.** The 15,744-triangle Suzanne costs the same as the small test room: at 1280×720 with one sample per pixel the time is dominated by ray count, not by the BVH depth.
- **Converged frames are cheap.** After the last sample (for example 1024/1024) the path tracer stops tracing and only runs the resolve and tone-mapping kernel, about 0.42 ms. Measuring in that state gives the same number for every setting. To time the tracing itself, raise Max samples to 4096 and read the log while the count is still rising.

## 12. Verification

- `ctest`: 18 programs. The new ones: `test_image` (checksums, zlib cases and errors, PNG and TGA cases and errors, `load_image`), `test_texture` (mip chain sizes and values including odd sizes, packing, bilinear, repeat, levels and trilinear, LOD), `test_json` (types, fallbacks, numbers and their range, string escapes, whitespace and BOM, errors with positions, depth, copy and move, a glTF-shaped document, a large document), `test_gltf` (base64, a basic model, Y-up to Z-up, accessors with offsets, strides and sparse storage, index types, primitive modes, node transforms, hierarchy and cycles, instancing, materials and their assignment, embedded and unusable textures, extensions, GLB, external files, the committed `cubes` and `materials` assets, damaged files), `test_path` (sRGB, tone mapping, the sky, basis and cosine sampling, the sun, the background, Russian roulette, the firefly clamp, settings), `test_brdf` (GGX normalisation, Smith, Fresnel, VNDF validity and pdf, albedo consistency, lobe probability, mixture pdf, MIS for the sun), `test_light` (area scale, triangle sampling, the cumulative sum, area to solid angle, the light list, MIS for area lights) and `test_material_edit` (clamping rules, split and combine of emission, the editor's list and targets, the light list from stored areas). `test_obj` gained texture coordinates, the MTL parser and material grouping; `test_ui` gained the scrolling panel (fit, wheel, hidden rows, thumb drag).
- The GPU code is also run on the CPU in tests where practical: `texture_sampling.h`, `path_math.h`, `brdf.h` and `light_sampling.h` are header-only code shared with the kernels.

## 13. Manual checklist

1. `gltf\cubes.glb` and `textured_cube.obj` (with `assets\textures\uv_grid.png`) in each backend: the same colours and textures, with **M** switching to plain grey shading and back; `gltf\materials.glb` in each backend shows the spheres, the floor and the panel in their base colours.
2. With `--backend=cuda`, **P** on `gltf\materials.glb`: the picture starts noisy and converges progressively up to 1024 samples per pixel, then stops; moving the camera or the sun restarts it. **P** again returns to the rasterized view.
3. In the path tracer the gold metals go from mirror-like to rough from left to right, the blue dielectrics from sharp sun glints to a soft sheen, and the emissive panel glows and lights the spheres and the floor. **M** removes gloss and emission.
4. Path tracing > Settings: Max samples, Samples / frame and Bounces change how it converges and how deep light travels; Exposure brightens or darkens the picture without restarting; Sun strength 0 leaves only the sky; Sky strength changes the ambient light; Reset settings restores the defaults.
5. The Firefly limit checkbox: off, bright isolated dots may appear in the sun's reflections and caustics; on, they are gone and the caustics are slightly dimmer.
6. Panel > Materials: pick the gold sphere's material and drag Roughness: the reflections blur live and the noise restarts. Set the panel's Emission strength to 0: the room darkens. Raise the floor's emission: the floor glows and lights the spheres. In the CUDA rasterized view (not **P**) colour edits show immediately. Reset material and Reset all materials restore the loaded values.
7. Click **<< Hide**: the panel collapses and a **>>** tab appears at the left edge; the tab or **H** brings the panel back. Clicks on the tab do not move the camera.
8. Make the window shorter than the panel: the panel fits the window height with a scrollbar on its right; the wheel and dragging the thumb scroll the body; nothing is drawn below the panel; the camera does not zoom while the wheel is over the panel; making the window taller removes the scrollbar.
9. With `--backend=opengl` and `--backend=vulkan` the Path tracing and Materials sections show "Needs --backend=cuda" and are disabled.
10. A glTF file with a JPEG texture loads with a warning and keeps its colours; a file requiring Draco is refused with an error message.

## 14. Known scope and limitations

| Item | Notes |
|---|---|
| **Normal maps** | Loaded and stored in the material, but not used by any backend yet; planned for all backends |
| **No mip level for rays** | The ray and path tracers sample textures at level 0 (bilinear) because camera rays carry no pixel footprint; distant textures can alias |
| **Single-scattering GGX** | Rough metals come out too dark (energy lost to light that bounces between microfacets); a multiple-scattering compensation is not implemented |
| **Instance limit** | The ray and path tracers handle `kMaxRtInstances` = 8 draw items; further ones are left out (a warning is logged once) |
| **Material table and light list per mesh** | Every mesh uploads its own copy of its materials and light list, which is wasteful when a glTF file repeats the same list in each mesh |
| **Editor** | Edits are in memory only and are not saved to the file; textures cannot be edited; the first target's values seed an entry, so an edit overwrites the other meshes' copies of the same (index, name) material even if they differed |
| **Exposure and the background** | In **P** the exposure setting also brightens or darkens the viewport background colour, like everything else |
| **Firefly limit** | Biased: caustic light is dimmer than a converged unclamped render |
| **Image and glTF coverage** | No JPEG; glTF samplers, `KHR_texture_transform`, sparse accessors, alpha modes, animation, skins and cameras are not supported; Draco and similar compression is refused |
| **Panel layout** | Scroll mode is decided from the previous frame, so a resize or an unfold shows one frame at the old size; a row partly outside the visible body is hidden whole |
| **ASCII only** | Unchanged from M8 |

## 15. Completion checkpoint

- models load from OBJ with MTL and from glTF/GLB, with textures from PNG and TGA decoded by our own Deflate, PNG and TGA code;
- all three rasterizers draw per-material colours and base-colour textures, with mip maps, and `M` switches to plain shading;
- the CUDA backend has a progressive path tracer (**P**) with sun and sky lighting, a GGX glossy layer, emission, sampled emissive triangles with MIS, and a firefly limit;
- the panel sets the path tracer's settings, hides to the left edge, scrolls when the window is short, and edits the model's materials live;
- the path tracer's maths is in portable headers tested on the CPU;
- `ctest` shows 18 passing tests.

Next is M10.
