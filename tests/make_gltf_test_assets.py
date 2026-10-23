"""Generates the glTF test models in assets/gltf/ (committed).

Only needed to change the models. Python standard library only.

  cubes.glb              binary glTF, everything in one file
  cubes.gltf + cubes.bin text glTF with an external buffer; the texture is
                         referenced as ../textures/uv_grid.png

Both describe the same scene, and tests/test_gltf.cpp checks that they
load to identical models:

  root  (translation, rotation of 30 degrees about Y)
   |- cube_right  mesh "cube", translation + scale 0.5
   |- cube_left   mesh "cube" (the SAME mesh), a 4x4 matrix transform
   `- pyramid     mesh "pyramid": plain TRIANGLES, no indices, no normals

The cube has 24 vertices (4 per face) in one interleaved buffer view with
a byteStride (position, normal, uv), u16 indices and two primitives:
material "grid" (the uv_grid.png texture) on five faces and material
"gold" (a plain colour) on the -X face. The vertex data is read from
assets/textured_cube.obj so the faces have the same orientation as in
that file; glTF puts (0, 0) of a texture at its top-left corner and OBJ
at the bottom-left, hence v_gltf = 1 - v_obj.

Run:  python tests/make_gltf_test_assets.py
"""

import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "..", "assets")
OUT = os.path.join(ASSETS, "gltf")

# glTF constants
FLOAT = 5126
USHORT = 5123
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def read_obj_cube():
    """Returns [(material name, [4 corners as (pos, uv, normal)])] per face."""
    positions, uvs, normals, faces = [], [], [], []
    material = None

    with open(os.path.join(ASSETS, "textured_cube.obj")) as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0].startswith("#"):
                continue
            if parts[0] == "v":
                positions.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "vt":
                uvs.append(tuple(float(x) for x in parts[1:3]))
            elif parts[0] == "vn":
                normals.append(tuple(float(x) for x in parts[1:4]))
            elif parts[0] == "usemtl":
                material = parts[1]
            elif parts[0] == "f":
                corners = []
                for corner in parts[1:]:
                    v, vt, vn = (int(x) - 1 for x in corner.split("/"))
                    u, tv = uvs[vt]
                    corners.append((positions[v], (u, 1.0 - tv), normals[vn]))
                faces.append((material, corners))

    return faces


def pad4(data, fill=b"\0"):
    return data + fill * (-len(data) % 4)


def pyramid_positions():
    b0 = (-0.5, 0.0, -0.5)
    b1 = (0.5, 0.0, -0.5)
    b2 = (0.5, 0.0, 0.5)
    b3 = (-0.5, 0.0, 0.5)
    apex = (0.0, 1.0, 0.0)

    # Counter-clockwise seen from outside: four sides and a two-triangle base
    triangles = [
        (b3, b2, apex),   # +Z side
        (b2, b1, apex),   # +X side
        (b1, b0, apex),   # -Z side
        (b0, b3, apex),   # -X side
        (b0, b1, b2),     # base, facing down
        (b0, b2, b3),
    ]

    # Sanity check: every normal points away from the pyramid's centre
    centre = (0.0, 0.3, 0.0)
    for a, b, c in triangles:
        u = [b[i] - a[i] for i in range(3)]
        v = [c[i] - a[i] for i in range(3)]
        n = (u[1] * v[2] - u[2] * v[1],
             u[2] * v[0] - u[0] * v[2],
             u[0] * v[1] - u[1] * v[0])
        away = sum(n[i] * (a[i] - centre[i]) for i in range(3))
        assert away > 0, "triangle faces inwards"

    return [p for tri in triangles for p in tri]


def quaternion_about_y(degrees):
    half = math.radians(degrees) / 2.0
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def left_cube_matrix():
    """Column-major 4x4: translate(-1.5, 0.25, 0) * rotateZ(45) * scale(0.75)."""
    c = math.cos(math.radians(45.0)) * 0.75
    s = math.sin(math.radians(45.0)) * 0.75
    return [c, s, 0.0, 0.0,
            -s, c, 0.0, 0.0,
            0.0, 0.0, 0.75, 0.0,
            -1.5, 0.25, 0.0, 1.0]


def min_max(points):
    return ([min(p[i] for p in points) for i in range(3)],
            [max(p[i] for p in points) for i in range(3)])


def build(embed_image):
    faces = read_obj_cube()
    assert len(faces) == 6

    # ---- Cube: 24 interleaved vertices (position, normal, uv) ------
    vertices = bytearray()
    cube_positions = []
    for _, corners in faces:
        for pos, uv, normal in corners:
            cube_positions.append(pos)
            vertices += struct.pack("<3f3f2f", *pos, *normal, *uv)
    assert len(vertices) == 24 * 32

    # Two primitives: the "grid" faces, and the "gold" one (obj material "label")
    grid_indices, gold_indices = [], []
    for face, (material, _) in enumerate(faces):
        base = face * 4
        quad = [base, base + 1, base + 2, base, base + 2, base + 3]
        (gold_indices if material == "label" else grid_indices).extend(quad)

    indices = struct.pack("<%dH" % len(grid_indices + gold_indices),
                          *(grid_indices + gold_indices))

    # ---- Pyramid: 18 positions, no indices, no normals ---------------
    pyramid = pyramid_positions()
    pyramid_bytes = b"".join(struct.pack("<3f", *p) for p in pyramid)

    # ---- Binary buffer: views laid out one after the other -----------
    blob = bytearray()
    views = []

    def add_view(data, **extra):
        offset = len(blob)
        blob.extend(pad4(bytes(data)))
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        view.update(extra)
        views.append(view)
        return len(views) - 1

    cube_view = add_view(vertices, byteStride=32, target=ARRAY_BUFFER)
    index_view = add_view(indices, target=ELEMENT_ARRAY_BUFFER)
    pyramid_view = add_view(pyramid_bytes, target=ARRAY_BUFFER)

    image = {"uri": "../textures/uv_grid.png"}
    if embed_image:
        with open(os.path.join(ASSETS, "textures", "uv_grid.png"), "rb") as f:
            png = f.read()
        image_view = add_view(png)
        image = {"bufferView": image_view, "mimeType": "image/png"}

    lo, hi = min_max(cube_positions)
    plo, phi = min_max(pyramid)
    n_grid = len(grid_indices)
    n_gold = len(gold_indices)

    accessors = [
        {"bufferView": cube_view, "byteOffset": 0, "componentType": FLOAT,
         "count": 24, "type": "VEC3", "min": lo, "max": hi},
        {"bufferView": cube_view, "byteOffset": 12, "componentType": FLOAT,
         "count": 24, "type": "VEC3"},
        {"bufferView": cube_view, "byteOffset": 24, "componentType": FLOAT,
         "count": 24, "type": "VEC2"},
        {"bufferView": index_view, "byteOffset": 0, "componentType": USHORT,
         "count": n_grid, "type": "SCALAR"},
        {"bufferView": index_view, "byteOffset": n_grid * 2,
         "componentType": USHORT, "count": n_gold, "type": "SCALAR"},
        {"bufferView": pyramid_view, "componentType": FLOAT,
         "count": len(pyramid), "type": "VEC3", "min": plo, "max": phi},
    ]

    attributes = {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}

    doc = {
        "asset": {"version": "2.0", "generator": "make_gltf_test_assets.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"name": "cubes", "nodes": [0]}],
        "nodes": [
            {"name": "root", "translation": [0.0, 0.5, 0.0],
             "rotation": quaternion_about_y(30.0), "children": [1, 2, 3]},
            {"name": "cube_right", "mesh": 0,
             "translation": [1.5, 0.0, 0.0], "scale": [0.5, 0.5, 0.5]},
            {"name": "cube_left", "mesh": 0, "matrix": left_cube_matrix()},
            {"name": "pyramid", "mesh": 1, "translation": [0.0, -0.5, 1.5]},
        ],
        "meshes": [
            {"name": "cube", "primitives": [
                {"attributes": attributes, "indices": 3, "material": 0},
                {"attributes": attributes, "indices": 4, "material": 1},
            ]},
            {"name": "pyramid", "primitives": [
                {"attributes": {"POSITION": 5}, "material": 2},
            ]},
        ],
        "materials": [
            {"name": "grid", "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0, "roughnessFactor": 0.5}},
            {"name": "gold", "pbrMetallicRoughness": {
                "baseColorFactor": [1.0, 0.766, 0.336, 1.0],
                "metallicFactor": 1.0, "roughnessFactor": 0.3}},
            {"name": "glow",
             "pbrMetallicRoughness": {
                 "baseColorFactor": [0.2, 0.2, 0.2, 1.0],
                 "metallicFactor": 0.0, "roughnessFactor": 0.8},
             "emissiveFactor": [1.0, 0.4, 0.1],
             "extensions": {"KHR_materials_emissive_strength":
                            {"emissiveStrength": 4.0}}},
        ],
        "textures": [{"sampler": 0, "source": 0}],
        "samplers": [{"magFilter": 9729, "minFilter": 9987,
                      "wrapS": 10497, "wrapT": 10497}],
        "images": [image],
        "accessors": accessors,
        "bufferViews": views,
    }

    return doc, bytes(blob)


def main():
    os.makedirs(OUT, exist_ok=True)

    # ---- cubes.glb --------------------------------------------------
    doc, blob = build(embed_image=True)
    doc["buffers"] = [{"byteLength": len(blob)}]
    json_bytes = pad4(json.dumps(doc, separators=(",", ":")).encode("utf-8"),
                      b" ")
    bin_bytes = pad4(blob)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(os.path.join(OUT, "cubes.glb"), "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))
        f.write(bin_bytes)

    # ---- cubes.gltf + cubes.bin ------------------------------------
    doc, blob = build(embed_image=False)
    doc["buffers"] = [{"uri": "cubes.bin", "byteLength": len(blob)}]

    with open(os.path.join(OUT, "cubes.bin"), "wb") as f:
        f.write(blob)

    with open(os.path.join(OUT, "cubes.gltf"), "w", newline="\n") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    print("wrote", os.path.normpath(OUT))


if __name__ == "__main__":
    main()
