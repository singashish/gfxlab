#pragma once

#if defined(__gl_h_) || defined(__GL_H__)
#error "Do not include <GL/gl.h>. Use renderer/opengl/gl_api.h instead."
#endif

#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
    #define GFX_GL_CALL __stdcall
#else
    #define GFX_GL_CALL
#endif

// OpenGL types
using GLenum = unsigned int;
using GLbitfield = unsigned int;
using GLuint = unsigned int;
using GLuint64 = std::uint64_t;
using GLint = int;
using GLsizei = int;
using GLboolean = unsigned char;
using GLubyte = unsigned char;
using GLfloat = float;
using GLchar = char;
using GLsizeiptr = std::ptrdiff_t;

using GLDEBUGPROC = void (GFX_GL_CALL*)(
    GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam
);

// Buffer bits
constexpr GLenum GL_COLOR_BUFFER_BIT   = 0x00004000;
constexpr GLenum GL_DEPTH_BUFFER_BIT   = 0x00000100;
constexpr GLenum GL_STENCIL_BUFFER_BIT  = 0x00000400;

// Primitive types
constexpr GLenum GL_LINES      = 0x0001;
constexpr GLenum GL_TRIANGLES  = 0x0004;

// Data types
constexpr GLenum GL_UNSIGNED_SHORT = 0x1403;
constexpr GLenum GL_UNSIGNED_INT   = 0x1405;
constexpr GLenum GL_FLOAT          = 0x1406;

// Render state
constexpr GLenum GL_CULL_FACE = 0x0B44;
constexpr GLenum GL_DEPTH_TEST = 0x0B71;
constexpr GLenum GL_BLEND = 0x0BE2;

// GPU timer queries
constexpr GLenum GL_TIME_ELAPSED = 0x88BF;
constexpr GLenum GL_QUERY_RESULT = 0x8866;
constexpr GLenum GL_QUERY_RESULT_AVAILABLE = 0x8867;

constexpr GLenum GL_LESS = 0x0201;
constexpr GLenum GL_LEQUAL = 0x0203;
constexpr GLenum GL_SRC_ALPHA = 0x0302;
constexpr GLenum GL_ONE_MINUS_SRC_ALPHA = 0x0303;

constexpr GLenum GL_FRONT_AND_BACK = 0x0408;
constexpr GLenum GL_LINE = 0x1B01;
constexpr GLenum GL_FILL = 0x1B02;

// Version and information strings
constexpr GLenum GL_VENDOR = 0x1F00;
constexpr GLenum GL_RENDERER = 0x1F01;
constexpr GLenum GL_VERSION = 0x1F02;
constexpr GLenum GL_SHADING_LANGUAGE_VERSION = 0x8B8C;
constexpr GLenum GL_MAJOR_VERSION = 0x821B;
constexpr GLenum GL_MINOR_VERSION = 0x821C;

// Shader constants
constexpr GLenum GL_FRAGMENT_SHADER = 0x8B30;
constexpr GLenum GL_VERTEX_SHADER = 0x8B31;
constexpr GLenum GL_COMPILE_STATUS = 0x8B81;
constexpr GLenum GL_LINK_STATUS = 0x8B82;
constexpr GLenum GL_INFO_LOG_LENGTH = 0x8B84;

// Textures
constexpr GLenum GL_TEXTURE_2D = 0x0DE1;
constexpr GLenum GL_TEXTURE0 = 0x84C0;
constexpr GLenum GL_RGBA = 0x1908;
constexpr GLenum GL_RGBA8 = 0x8058;
constexpr GLenum GL_UNSIGNED_BYTE = 0x1401;
constexpr GLenum GL_TEXTURE_MIN_FILTER = 0x2801;
constexpr GLenum GL_TEXTURE_MAG_FILTER = 0x2800;
constexpr GLenum GL_NEAREST = 0x2600;

// Buffer constants
constexpr GLenum GL_ARRAY_BUFFER = 0x8892;
constexpr GLenum GL_ELEMENT_ARRAY_BUFFER = 0x8893;
constexpr GLenum GL_STATIC_DRAW = 0x88E4;
constexpr GLenum GL_DYNAMIC_DRAW = 0x88E8;

// Error constants
constexpr GLenum GL_NO_ERROR = 0;
constexpr GLenum GL_INVALID_ENUM = 0x0500;
constexpr GLenum GL_INVALID_VALUE = 0x0501;
constexpr GLenum GL_INVALID_OPERATION = 0x0502;
constexpr GLenum GL_OUT_OF_MEMORY = 0x0505;
constexpr GLenum GL_INVALID_FRAMEBUFFER_OPERATION = 0x0506;

// Debug constants
constexpr GLenum GL_CONTEXT_FLAGS = 0x821E;
constexpr GLenum GL_CONTEXT_FLAG_DEBUG_BIT = 0x00000002;
constexpr GLenum GL_DEBUG_OUTPUT = 0x92E0;
constexpr GLenum GL_DEBUG_OUTPUT_SYNCHRONOUS = 0x8242;

constexpr GLenum GL_DEBUG_SEVERITY_NOTIFICATION = 0x826B;
constexpr GLenum GL_DEBUG_SEVERITY_HIGH = 0x9146;
constexpr GLenum GL_DEBUG_SEVERITY_MEDIUM = 0x9147;
constexpr GLenum GL_DEBUG_SEVERITY_LOW = 0x9148;

constexpr GLboolean GL_FALSE = 0;
constexpr GLboolean GL_TRUE = 1;

#include "renderer/opengl/gl_functions.h"

// Generate function-pointer types and declarations
#define GFX_GL_DECLARE(ret, name, args)                 \
    using PFN_gl##name = ret (GFX_GL_CALL*) args;       \
    extern PFN_gl##name gl##name;

GFX_GL_FUNCTIONS(GFX_GL_DECLARE)

#undef GFX_GL_DECLARE

namespace gfx {
    bool gl_load_functions();
}
