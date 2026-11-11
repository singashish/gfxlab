#include "platform/file_dialog.h"

#include "core/dialog_result.h"
#include "core/log.h"

// WIN32_LEAN_AND_MEAN, NOMINMAX and UNICODE are defined
// project-wide in the root CMakeLists.txt
#include <windows.h>
#include <commdlg.h>

#include <string>
#include <vector>

namespace gfx {

namespace {

std::string wide_to_utf8(const std::wstring& wide) {
    if (wide.empty()) {
        return {};
    }

    const int length = WideCharToMultiByte(
        CP_UTF8, 0, wide.c_str(), static_cast<int>(wide.size()), nullptr, 0,
        nullptr, nullptr);

    if (length <= 0) {
        return {};
    }

    std::string utf8(static_cast<std::size_t>(length), '\0');

    if (WideCharToMultiByte(CP_UTF8, 0, wide.c_str(),
                            static_cast<int>(wide.size()), utf8.data(),
                            length, nullptr, nullptr) != length) {
        return {};
    }

    return utf8;
}

} // namespace

std::vector<std::string> open_model_files_dialog(void* owner) {
    std::vector<std::string> result;

    // Room for the names of very many files; the dialog fails with
    // FNERR_BUFFERTOOSMALL if even this is not enough
    std::vector<wchar_t> buffer(65536, L'\0');

    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = static_cast<HWND>(owner);

    // Pairs of "description", "patterns", ended by an extra zero (the
    // literal's own terminating zero is the extra one)
    dialog.lpstrFilter =
        L"3D models (*.obj, *.gltf, *.glb)\0*.obj;*.gltf;*.glb\0"
        L"All files (*.*)\0*.*\0";
    dialog.nFilterIndex = 1;
    dialog.lpstrFile = buffer.data();
    dialog.nMaxFile = static_cast<DWORD>(buffer.size());
    dialog.lpstrTitle = L"Load models";

    // NOCHANGEDIR: the dialog must not move the program's current
    // directory (model paths on the command line are relative to it)
    dialog.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST |
                   OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR | OFN_HIDEREADONLY;

    if (!GetOpenFileNameW(&dialog)) {
        const DWORD error = CommDlgExtendedError();

        // 0 means the user cancelled
        if (error != 0) {
            LOG_ERROR("open-file dialog failed: error 0x%lx",
                      static_cast<unsigned long>(error));
        }

        return result;
    }

    for (const std::wstring& path :
         parse_multi_select_buffer(buffer.data(), buffer.size())) {
        const std::string utf8 = wide_to_utf8(path);

        if (!utf8.empty()) {
            result.push_back(utf8);
        }
    }

    return result;
}

} // namespace gfx
