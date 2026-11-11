#include "core/dialog_result.h"

namespace gfx {

namespace {

// The string that starts at `start` and ends at the first zero (or at the
// end of the buffer); `next` is where the following string starts
std::wstring take_string(const wchar_t* buffer, std::size_t capacity,
                         std::size_t start, std::size_t& next) {
    std::size_t end = start;

    while (end < capacity && buffer[end] != L'\0') {
        ++end;
    }

    next = end + 1;
    return std::wstring(buffer + start, end - start);
}

} // namespace

std::vector<std::wstring> parse_multi_select_buffer(const wchar_t* buffer,
                                                    std::size_t capacity) {
    std::vector<std::wstring> paths;

    if (buffer == nullptr || capacity == 0 || buffer[0] == L'\0') {
        return paths;
    }

    std::size_t next = 0;
    const std::wstring first = take_string(buffer, capacity, 0, next);

    // Only one string: the full path of the single file
    if (next >= capacity || buffer[next] == L'\0') {
        paths.push_back(first);
        return paths;
    }

    // Folder first, then the file names
    std::wstring folder = first;

    if (!folder.empty() && folder.back() != L'\\' && folder.back() != L'/') {
        folder += L'\\';
    }

    while (next < capacity && buffer[next] != L'\0') {
        const std::size_t start = next;
        const std::wstring name = take_string(buffer, capacity, start, next);

        paths.push_back(folder + name);
    }

    return paths;
}

} // namespace gfx
