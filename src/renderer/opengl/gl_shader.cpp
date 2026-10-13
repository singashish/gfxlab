#include "renderer/opengl/gl_shader.h"
#include "renderer/opengl/gl_api.h"
#include "core/log.h"

#include <string>

namespace {

GLuint compile_stage(GLenum type, const char* source, const char* label) {
    const GLuint shader = glCreateShader(type);

    if (shader == 0) {
        LOG_ERROR("failed to create %s shader", label);
        return 0;
    }

    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);

    if (!ok) {
        GLint length = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);

        std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');

        GLsizei written = 0;
        glGetShaderInfoLog(
            shader,
            static_cast<GLsizei>(log.size()),
            &written,
            log.data()
        );

        LOG_ERROR("%s shader failed to compile:\n%s", label, log.c_str());

        glDeleteShader(shader);
        return 0;
    }

    return shader;
}

} // namespace

namespace gfx {

unsigned int gl_create_program(const char* vertex_source, const char* fragment_source) {
    const GLuint vertex_shader = compile_stage(GL_VERTEX_SHADER,
                                               vertex_source, "vertex");

    const GLuint fragment_shader = compile_stage(
        GL_FRAGMENT_SHADER, fragment_source, "fragment");

    if (vertex_shader == 0 || fragment_shader == 0) {
        if (vertex_shader != 0) {
            glDeleteShader(vertex_shader);
        }

        if (fragment_shader != 0) {
            glDeleteShader(fragment_shader);
        }

        return 0;
    }

    const GLuint program = glCreateProgram();

    if (program == 0) {
        LOG_ERROR("failed to create OpenGL program");
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);

    if (!linked) {
        GLint length = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

        std::string log(static_cast<std::size_t>(length > 0 ? length : 1), '\0');

        GLsizei written = 0;
        glGetProgramInfoLog(
            program,
            static_cast<GLsizei>(log.size()),
            &written,
            log.data()
        );

        LOG_ERROR("shader program failed to link:\n%s",
            log.c_str());

        glDeleteProgram(program);
        glDeleteShader(vertex_shader);
        glDeleteShader(fragment_shader);
        return 0;
    }

    // The linked program no longer needs the shader objects
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    return program;
}

} // namespace gfx
