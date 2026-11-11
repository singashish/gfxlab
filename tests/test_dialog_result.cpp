// The text a multi-select "Open" dialog returns (core/dialog_result.h)

#include "core/dialog_result.h"
#include "check.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

// Builds a buffer of `capacity` characters from strings that are each
// followed by a zero, and the extra zero at the end
std::vector<wchar_t> make_buffer(const std::vector<std::wstring>& parts,
                                 std::size_t capacity) {
    std::vector<wchar_t> buffer(capacity, L'\0');
    std::size_t at = 0;

    for (const std::wstring& part : parts) {
        for (const wchar_t c : part) {
            buffer[at++] = c;
        }

        ++at; // the zero
    }

    return buffer;
}

void test_one_file() {
    const std::vector<wchar_t> buffer =
        make_buffer({L"C:\\models\\a.obj"}, 64);

    const auto paths = gfx::parse_multi_select_buffer(buffer.data(), buffer.size());

    CHECK(paths.size() == 1);
    CHECK(paths[0] == L"C:\\models\\a.obj");
}

void test_several_files() {
    const std::vector<wchar_t> buffer =
        make_buffer({L"C:\\models", L"a.obj", L"b.glb", L"c d.gltf"}, 128);

    const auto paths = gfx::parse_multi_select_buffer(buffer.data(), buffer.size());

    CHECK(paths.size() == 3);
    CHECK(paths[0] == L"C:\\models\\a.obj");
    CHECK(paths[1] == L"C:\\models\\b.glb");
    CHECK(paths[2] == L"C:\\models\\c d.gltf");
}

// A folder that already ends in a separator (a drive root) gets no second
void test_folder_with_separator() {
    const std::vector<wchar_t> buffer =
        make_buffer({L"C:\\", L"a.obj", L"b.obj"}, 64);

    const auto paths = gfx::parse_multi_select_buffer(buffer.data(), buffer.size());

    CHECK(paths.size() == 2);
    CHECK(paths[0] == L"C:\\a.obj");
    CHECK(paths[1] == L"C:\\b.obj");
}

void test_empty_and_bad_input() {
    const std::vector<wchar_t> empty(16, L'\0');

    CHECK(gfx::parse_multi_select_buffer(empty.data(), empty.size()).empty());
    CHECK(gfx::parse_multi_select_buffer(nullptr, 10).empty());
    CHECK(gfx::parse_multi_select_buffer(empty.data(), 0).empty());
}

// A buffer without its closing zeros is read only up to its capacity
void test_no_terminator() {
    const wchar_t text[] = {L'C', L':', L'\\', L'a', L'.', L'o', L'b', L'j'};

    const auto one = gfx::parse_multi_select_buffer(text, 8);

    CHECK(one.size() == 1);
    CHECK(one[0] == L"C:\\a.obj");

    // Folder, zero, name, and then the buffer simply ends
    const wchar_t two[] = {L'D', L'\0', L'x', L'.', L'o', L'b', L'j'};

    const auto paths = gfx::parse_multi_select_buffer(two, 7);

    CHECK(paths.size() == 1);
    CHECK(paths[0] == L"D\\x.obj");
}

} // namespace

int main() {
    test_one_file();
    test_several_files();
    test_folder_with_separator();
    test_empty_and_bad_input();
    test_no_terminator();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d check(s) failed\n", g_failures);
        return 1;
    }

    std::printf("dialog result tests passed\n");
    return 0;
}
