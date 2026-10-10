#pragma once

namespace gfx {

enum class LogLevel {
    LOG_LEVEL_TRACE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARN,
    LOG_LEVEL_ERROR
};

void log_set_level(LogLevel min_level);
void log_write(LogLevel level, const char* file, int line, const char* fmt, ...);

} // namespace gfx

#define LOG_TRACE(...) ::gfx::log_write(::gfx::LogLevel::LOG_LEVEL_TRACE, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_DEBUG(...) ::gfx::log_write(::gfx::LogLevel::LOG_LEVEL_DEBUG, __FILE__, __LINE__, __VA_ARGS__)
#define LOG_INFO(...)  ::gfx::log_write(::gfx::LogLevel::LOG_LEVEL_INFO,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_WARN(...)  ::gfx::log_write(::gfx::LogLevel::LOG_LEVEL_WARN,  __FILE__, __LINE__, __VA_ARGS__)
#define LOG_ERROR(...) ::gfx::log_write(::gfx::LogLevel::LOG_LEVEL_ERROR, __FILE__, __LINE__, __VA_ARGS__)
