"""Generates assets/gltf/materials.glb, the material test scene (committed).

Only needed to change the model. Python standard library only; the output
is deterministic (running it twice gives the same bytes).

The scene is a small "material test bench" for the CUDA path tracer:

  * Ten UV spheres (radius 0.45, 24 x 16 segments) in a 5 x 2 grid with
    1.1 spacing. The top row is dielectric (metallic 0, blue-ish base
    colour), the bottom row is metal (metallic 1, warm gold base colour).
    The columns, left to right, have roughness 0.05, 0.25, 0.5, 0.75, 1.0.
    Every sphere has its own material.
  * An emissive panel, 2 x 0.3 m, hovering above and behind the grid and
    tilted to face down and forward. Its base colour is almost black, so
    it is mostly a light source: emissiveFactor (1.0, 0.85, 0.6) with
    KHR_materials_emissive_strength 8.0, i.e. an emission of
    (8.0, 6.8, 4.8).
  * A large light-grey floor (roughness 1, metallic 0) under the spheres.

The file has three nodes: "spheres" (ONE mesh with ten primitives, one
per material, the positions already moved to the sphere's place in the
grid), "panel" and "floor". Why not ten nodes with ten meshes? The CUDA
ray and path tracers handle at most kMaxRtInstances (8) draw items, a
node being one item; ten spheres, a panel and a floor as separate nodes
would leave the last ones out of the picture.

glTF is Y-up, so the scene is written Y-up and the loader turns it into
gfxlab's Z-up world. The spheres stand in the XY plane, the floor is at
y = 0, and the camera of a glTF viewer looks along -Z. Positions, normals
and texture coordinates are plain floats; indices are 16-bit. Everything
is in one embedded buffer (no external files). Texture coordinates follow
glTF: (0, 0) is the top-left corner of an image, so v = 0 is the north
pole of a sphere.

Run:  python tests/make_material_test_model.py
"""

import json
import math
import os
import struct

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "assets", "gltf", "materials.glb")

# glTF constants
FLOAT = 5126
USHORT = 5123
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

# The scene, in metres
RADIUS = 0.45
SPACING = 1.1
SEGMENTS = 24            # around the Y axis
RINGS = 16               # from the north pole to the south pole
ROUGHNESS = [0.05, 0.25, 0.5, 0.75, 1.0]
BOTTOM_ROW_Y = 0.5       # sphere centres; the floor is at y = 0
TOP_ROW_Y = BOTTOM_ROW_Y + SPACING

GOLD = [1.0, 0.77, 0.34]
BLUE = [0.2, 0.4, 0.9]
FLOOR_GREY = [0.65, 0.65, 0.65]
PANEL_BASE = [0.04, 0.04, 0.04]
PANEL_EMISSIVE = [1.0, 0.85, 0.6]
PANEL_STRENGTH = 8.0
PANEL_CENTRE = (0.0, 3.2, -1.0)
PANEL_NORMAL = (0.0, -0.8, 0.6)    # unit length: faces down and forward
PANEL_SIZE = (2.0, 0.3)
FLOOR_HALF = (4.0, 3.0)            # half sizes in x and z


# ---------------------------------------------------------------------
# Small vector helpers
# ---------------------------------------------------------------------

def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def normalize(a):
    n = math.sqrt(dot(a, a))
    return (a[0] / n, a[1] / n, a[2] / n)


# ---------------------------------------------------------------------
# Geometry: each function returns (positions, normals, uvs, indices)
# ---------------------------------------------------------------------

def make_sphere():
    """A UV sphere centred at the origin, poles on the Y axis."""
    positions, normals, uvs, indices = [], [], [], []

    for ring in range(RINGS + 1):
        theta = math.pi * ring / RINGS          # 0 at +Y, pi at -Y
        for seg in range(SEGMENTS + 1):         # the seam vertex is doubled
            phi = 2.0 * math.pi * seg / SEGMENTS
            n = (math.sin(theta) * math.cos(phi),
                 math.cos(theta),
                 math.sin(theta) * math.sin(phi))
            positions.append(scale(n, RADIUS))
            normals.append(n)
            uvs.append((seg / SEGMENTS, ring / RINGS))

    row = SEGMENTS + 1
    for ring in range(RINGS):
        for seg in range(SEGMENTS):
            a = ring * row + seg            # upper left
            b = a + 1                       # upper right
            c = a + row                     # lower left
            d = c + 1                       # lower right
            # The first and last ring touch a pole, where one of the two
            # triangles of a quad would have no area: leave it out
            if ring != 0:
                indices += [a, b, c]
            if ring != RINGS - 1:
                indices += [b, d, c]

    check_outward(positions, indices, (0.0, 0.0, 0.0))
    return positions, normals, uvs, indices


def check_outward(positions, indices, centre):
    """Every triangle must be counter-clockwise seen from outside."""
    for i in range(0, len(indices), 3):
        p0, p1, p2 = (positions[indices[i + k]] for k in range(3))
        n = cross(sub(p1, p0), sub(p2, p0))
        c = scale(add(add(p0, p1), p2), 1.0 / 3.0)
        assert dot(n, sub(c, centre)) > 0.0, "triangle %d faces inward" % (i // 3)


def make_quad(corners, normal):
    """Two triangles; the corners are counter-clockwise seen from `normal`."""
    p0, p1, p2 = corners[0], corners[1], corners[2]
    assert dot(cross(sub(p1, p0), sub(p2, p0)), normal) > 0.0
    uvs = [(0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (0.0, 0.0)]
    return list(corners), [normal] * 4, uvs, [0, 1, 2, 0, 2, 3]


def make_panel():
    n = normalize(PANEL_NORMAL)
    x = (1.0, 0.0, 0.0)
    e = cross(n, x)                 # the panel's short side; x cross e = n
    hx = PANEL_SIZE[0] / 2.0
    he = PANEL_SIZE[1] / 2.0
    c = PANEL_CENTRE
    corners = [
        add(add(c, scale(x, -hx)), scale(e, -he)),
        add(add(c, scale(x, hx)), scale(e, -he)),
        add(add(c, scale(x, hx)), scale(e, he)),
        add(add(c, scale(x, -hx)), scale(e, he)),
    ]
    return make_quad(corners, n)


def make_floor():
    hx, hz = FLOOR_HALF
    # Seen from above (+Y) counter-clockwise: -x,+z  ->  +x,+z  ->  +x,-z ...
    corners = [(-hx, 0.0, hz), (hx, 0.0, hz), (hx, 0.0, -hz), (-hx, 0.0, -hz)]
    return make_quad(corners, (0.0, 1.0, 0.0))


# ---------------------------------------------------------------------
# Binary buffer and glTF document
# ---------------------------------------------------------------------

class Buffer:
    """Collects buffer views and accessors for one embedded buffer."""

    def __init__(self):
        self.data = bytearray()
        self.views = []
        self.accessors = []

    def add_view(self, blob, target):
        while len(self.data) % 4:       # every view starts 4-byte aligned
            self.data.append(0)
        self.views.append({"buffer": 0, "byteOffset": len(self.data),
                           "byteLength": len(blob), "target": target})
        self.data += blob
        return len(self.views) - 1

    def add_accessor(self, view, component, count, kind, bounds=None):
        accessor = {"bufferView": view, "componentType": component,
                    "count": count, "type": kind}
        if bounds is not None:
            accessor["min"], accessor["max"] = bounds
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def add_positions(self, positions):
        """A POSITION accessor; needs min and max, as single-precision values."""
        view = self.add_view(
            b"".join(struct.pack("<3f", *p) for p in positions), ARRAY_BUFFER)
        stored = [struct.unpack("<3f", struct.pack("<3f", *p))
                  for p in positions]
        lows = [min(p[k] for p in stored) for k in range(3)]
        highs = [max(p[k] for p in stored) for k in range(3)]
        return self.add_accessor(view, FLOAT, len(positions), "VEC3",
                                 (lows, highs))

    def add_normals(self, normals):
        view = self.add_view(
            b"".join(struct.pack("<3f", *n) for n in normals), ARRAY_BUFFER)
        return self.add_accessor(view, FLOAT, len(normals), "VEC3")

    def add_uvs(self, uvs):
        view = self.add_view(
            b"".join(struct.pack("<2f", *t) for t in uvs), ARRAY_BUFFER)
        return self.add_accessor(view, FLOAT, len(uvs), "VEC2")

    def add_indices(self, indices):
        view = self.add_view(
            b"".join(struct.pack("<H", i) for i in indices),
            ELEMENT_ARRAY_BUFFER)
        return self.add_accessor(view, USHORT, len(indices), "SCALAR")

    def add_geometry(self, positions, normals, uvs, indices):
        """Returns the accessor numbers (POSITION, NORMAL, TEXCOORD_0, indices)."""
        return (self.add_positions(positions), self.add_normals(normals),
                self.add_uvs(uvs), self.add_indices(indices))


def pbr_material(name, color, metallic, roughness):
    return {"name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": color + [1.0],
                "metallicFactor": metallic,
                "roughnessFactor": roughness}}


def primitive(position, normal, uv, index, material):
    return {"attributes": {"POSITION": position, "NORMAL": normal,
                           "TEXCOORD_0": uv},
            "indices": index, "material": material, "mode": 4}


def build():
    buf = Buffer()
    materials, meshes, nodes = [], [], []

    # The ten spheres: one primitive per material. The normals, the
    # texture coordinates and the indices are the same for all of them and
    # stored once; only the positions differ (the sphere moved to its place)
    positions, normals, uvs, indices = make_sphere()
    normal_accessor = buf.add_normals(normals)
    uv_accessor = buf.add_uvs(uvs)
    index_accessor = buf.add_indices(indices)
    sphere_primitives = []

    for kind, color, metallic, y in [("dielectric", BLUE, 0.0, TOP_ROW_Y),
                                     ("metal", GOLD, 1.0, BOTTOM_ROW_Y)]:
        for col, roughness in enumerate(ROUGHNESS):
            name = "%s_r%03d" % (kind, round(roughness * 100))
            materials.append(pbr_material(name, color, metallic, roughness))
            centre = ((col - 2) * SPACING, y, 0.0)
            moved = [add(p, centre) for p in positions]
            sphere_primitives.append(primitive(
                buf.add_positions(moved), normal_accessor, uv_accessor,
                index_accessor, len(materials) - 1))

    meshes.append({"name": "spheres", "primitives": sphere_primitives})
    nodes.append({"name": "spheres", "mesh": len(meshes) - 1})

    # The panel: an emitter over a nearly black base colour
    materials.append({
        "name": "panel",
        "pbrMetallicRoughness": {"baseColorFactor": PANEL_BASE + [1.0],
                                 "metallicFactor": 0.0,
                                 "roughnessFactor": 1.0},
        "emissiveFactor": PANEL_EMISSIVE,
        "extensions": {"KHR_materials_emissive_strength":
                       {"emissiveStrength": PANEL_STRENGTH}},
        "doubleSided": True})
    meshes.append({"name": "panel", "primitives": [primitive(
        *buf.add_geometry(*make_panel()), len(materials) - 1)]})
    nodes.append({"name": "panel", "mesh": len(meshes) - 1})

    materials.append(pbr_material("floor", FLOOR_GREY, 0.0, 1.0))
    meshes.append({"name": "floor", "primitives": [primitive(
        *buf.add_geometry(*make_floor()), len(materials) - 1)]})
    nodes.append({"name": "floor", "mesh": len(meshes) - 1})

    document = {
        "asset": {"version": "2.0",
                  "generator": "gfxlab tests/make_material_test_model.py"},
        "extensionsUsed": ["KHR_materials_emissive_strength"],
        "scene": 0,
        "scenes": [{"name": "materials", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": buf.accessors,
        "bufferViews": buf.views,
        "buffers": [{"byteLength": 0}],
    }
    return document, buf


def pad(data, filler):
    return data + filler * (-len(data) % 4)


def write_glb(document, buf):
    binary = pad(bytes(buf.data), b"\0")
    document["buffers"][0]["byteLength"] = len(binary)

    # sort_keys + fixed separators: the same bytes on every run
    text = json.dumps(document, sort_keys=True, separators=(",", ":"))
    js = pad(text.encode("ascii"), b" ")

    total = 12 + 8 + len(js) + 8 + len(binary)
    out = struct.pack("<4sII", b"glTF", 2, total)
    out += struct.pack("<I4s", len(js), b"JSON") + js
    out += struct.pack("<I4s", len(binary), b"BIN\0") + binary

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "wb") as f:
        f.write(out)
    print("wrote %s (%d bytes, %d meshes, %d materials)"
          % (os.path.normpath(OUT), len(out), len(document["meshes"]),
             len(document["materials"])))


if __name__ == "__main__":
    write_glb(*build())
