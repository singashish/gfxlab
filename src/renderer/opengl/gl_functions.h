#pragma once

// X(return type, name without "gl", (parameter list))
#define GFX_GL_FUNCTIONS(X) \
    /* Basic state and drawing */ \
    X(void, Clear, (GLbitfield mask)) \
    X(void, ClearColor, (GLfloat r, GLfloat g, GLfloat b, GLfloat a)) \
    X(void, Viewport, (GLint x, GLint y, GLsizei width, GLsizei height)) \
    X(const GLubyte*, GetString, (GLenum name)) \
    X(GLenum, GetError, (void)) \
    X(void, GetIntegerv, (GLenum pname, GLint* data)) \
    X(void, Enable, (GLenum cap)) \
    X(void, Disable, (GLenum cap)) \
    X(void, DepthFunc, (GLenum func)) \
    X(void, BlendFunc, (GLenum sfactor, GLenum dfactor)) \
    X(void, PolygonMode, (GLenum face, GLenum mode)) \
    X(void, DrawArrays, (GLenum mode, GLint first, GLsizei count)) \
    X(void, DrawElements, (GLenum mode, GLsizei count, GLenum type, const void* indices)) \
    /* GPU timer queries */ \
    X(void, GenQueries, (GLsizei n, GLuint* ids)) \
    X(void, DeleteQueries, (GLsizei n, const GLuint* ids)) \
    X(void, BeginQuery, (GLenum target, GLuint id)) \
    X(void, EndQuery, (GLenum target)) \
    X(void, GetQueryObjectiv, (GLuint id, GLenum pname, GLint* params)) \
    X(void, GetQueryObjectui64v, (GLuint id, GLenum pname, GLuint64* params)) \
    /* Shaders and programs */ \
    X(GLuint, CreateShader, (GLenum type)) \
    X(void, ShaderSource, (GLuint shader, GLsizei count, const GLchar* const* string, const GLint* length)) \
    X(void, CompileShader, (GLuint shader)) \
    X(void, GetShaderiv, (GLuint shader, GLenum pname, GLint* params)) \
    X(void, GetShaderInfoLog, (GLuint shader, GLsizei bufSize, GLsizei* length, GLchar* infoLog)) \
    X(void, DeleteShader, (GLuint shader)) \
    X(GLuint, CreateProgram, (void)) \
    X(void, AttachShader, (GLuint program, GLuint shader)) \
    X(void, LinkProgram, (GLuint program)) \
    X(void, GetProgramiv, (GLuint program, GLenum pname, GLint* params)) \
    X(void, GetProgramInfoLog, (GLuint program, GLsizei bufSize, GLsizei* length, GLchar* infoLog)) \
    X(void, UseProgram, (GLuint program)) \
    X(void, DeleteProgram, (GLuint program)) \
    X(GLint, GetUniformLocation, (GLuint program, const GLchar* name)) \
    X(void, UniformMatrix4fv, (GLint location, GLsizei count, GLboolean transpose, const GLfloat* value)) \
    /* Buffers and vertex arrays */ \
    X(void, GenBuffers, (GLsizei n, GLuint* buffers)) \
    X(void, BindBuffer, (GLenum target, GLuint buffer)) \
    X(void, BufferData, (GLenum target, GLsizeiptr size, const void* data, GLenum usage)) \
    X(void, DeleteBuffers, (GLsizei n, const GLuint* buffers)) \
    X(void, GenVertexArrays, (GLsizei n, GLuint* arrays)) \
    X(void, BindVertexArray, (GLuint array)) \
    X(void, DeleteVertexArrays, (GLsizei n, const GLuint* arrays)) \
    X(void, VertexAttribPointer, (GLuint index, GLint size, GLenum type, GLboolean normalized, GLsizei stride, const void* pointer)) \
    X(void, EnableVertexAttribArray, (GLuint index)) \
    /* Debug output */ \
    X(void, DebugMessageCallback, (GLDEBUGPROC callback, const void* userParam)) \
    X(void, Uniform3f, (GLint location, GLfloat v0, GLfloat v1, GLfloat v2)) \
    X(void, DepthMask, (GLboolean flag)) \
    X(void, Uniform1f, (GLint location, GLfloat v0)) \
    X(void, Uniform2f, (GLint location, GLfloat v0, GLfloat v1)) \
        /* Texture presentation */ \
    X(void, GenTextures, (GLsizei n, GLuint* textures)) \
    X(void, DeleteTextures, (GLsizei n, const GLuint* textures)) \
    X(void, BindTexture, (GLenum target, GLuint texture)) \
    X(void, TexImage2D, (GLenum target, GLint level, GLint internalformat, \
        GLsizei width, GLsizei height, GLint border, GLenum format, \
        GLenum type, const void* pixels)) \
    X(void, TexParameteri, (GLenum target, GLenum pname, GLint param)) \
    X(void, ActiveTexture, (GLenum texture)) \
    X(void, Uniform1i, (GLint location, GLint v0))
