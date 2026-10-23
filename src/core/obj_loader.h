#pragma once

#include "core/material.h"
#include "core/mesh.h"

#include <string>
#include <string_view>
#include <vector>

namespace gfx {

// Parse OBJ text. On failure, returns false, sets an error message
// beginning with the line number, and leaves out empty.
//
// Faces may use v, v/vt, v//vn and v/vt/vn corners. "usemtl" selects the
// material of the faces that follow: out.materials gets one entry per
// material name in order of first use (with default values; the .mtl
// file fills them in), and out.triangle_materials one index per
// triangle. Faces before any usemtl use a material named "default".
// The file names after "mtllib" are appended to material_libraries
// when it is given
bool parse_obj(std::string_view text, Mesh& out, std::string& error,
               std::vector<std::string>* material_libraries = nullptr);

// Parse MTL text into materials (appended to out). Reads newmtl, Kd,
// Ke, Ns, Pr, Pm and map_Kd; everything else is ignored. Texture names
// are resolved relative to base_dir. Roughness comes from Pr when it is
// given, otherwise from Ns. On failure returns false with an error
// beginning with the line number
bool parse_mtl(std::string_view text, const std::string& base_dir,
               std::vector<Material>& out, std::string& error);

// Read an OBJ file and parse it, then read its material libraries
// (relative to the OBJ file) and fill in the materials it uses. Missing
// or broken material libraries and unknown material names are logged
// as warnings and keep the default values; they do not fail the load
bool load_obj(const std::string& path, Mesh& out, std::string& error);

} // namespace gfx
