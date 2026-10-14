#pragma once

namespace gfx {

// Compiles and links a vertex + fragment shader pair
// Returns the program ID or 0 on failure
unsigned int gl_create_program(const char* vertex_source, const char* fragment_source);

// calls `glGetUniformLocation` and logs a warning naming
// the uniform when the result is -1
int gl_uniform_location(unsigned int program, const char* name);

} // namespace gfx
