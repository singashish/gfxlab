#include "core/log.h"

#include <cstdio>
#include <cstdarg>
#include <cstring>

namespace gfx {

static LogLevel min_level = LogLevel::LOG_LEVEL_INFO;

static const char* log_level_name(LogLevel level) {
    switch (level) {
        case LogLevel::LOG_LEVEL_TRACE: return "TRACE";
        case LogLevel::LOG_LEVEL_DEBUG: return "DEBUG";
        case LogLevel::LOG_LEVEL_INFO:  return "INFO";
        case LogLevel::LOG_LEVEL_WARN:  return "WARN";
        case LogLevel::LOG_LEVEL_ERROR: return "ERROR";
        default:              return "?????";
    }
}

void log_set_level(LogLevel level) {
    min_level = level;
}

void log_write(LogLevel level, const char* file, int line, const char* fmt, ...) {
    if (level < min_level) {
        return;
    }

    // Extract the filename from the full path
    const char* filename = file;
    const char* slash = std::strrchr(file, '/');
    const char* backslash = std::strrchr(file, '\\');

    if (slash && (!backslash || slash > backslash)) {
        filename = slash + 1;
    } else if (backslash) {
        filename = backslash + 1;
    }

    // Print the log prefix
    std::fprintf(stderr, "[%s] %s:%d  ",
                 log_level_name(level), filename, line);

    // Print the formatted message
    va_list args;
    va_start(args, fmt);
    std::vfprintf(stderr, fmt, args);
    va_end(args);

    // Finish the log entry
    std::fprintf(stderr, "\n");
}

} // namespace gfx
