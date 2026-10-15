#pragma once

#include "core/mesh.h"

#include <string>
#include <string_view>

namespace gfx {

// Parse OBJ text. On failure, returns false, sets an error message
// beginning with the line number, and leaves out empty
bool parse_obj(std::string_view text, Mesh& out, std::string& error);

// Read a file and parse its contents
bool load_obj(const std::string& path, Mesh& out, std::string& error);

} // namespace gfx
