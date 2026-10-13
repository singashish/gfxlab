#pragma once

namespace gfx {

// Compiles and links a vertex + fragment shader pair
// Returns the program ID or 0 on failure
unsigned int gl_create_program(const char* vertex_source, const char* fragment_source);

} // namespace gfx
