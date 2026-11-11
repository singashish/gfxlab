#pragma once

/*
 * The text a Windows "Open" dialog with multiple selection hands back.
 *
 * With OFN_EXPLORER | OFN_ALLOWMULTISELECT the dialog fills one buffer
 * with strings that end in a zero character, and an extra zero at the end:
 *
 *   one file:    "C:\models\a.obj" 0 0
 *   several:     "C:\models" 0 "a.obj" 0 "b.glb" 0 0
 *
 * (the first string is the folder, the others are file names in it). This
 * turns either form into full paths. It is plain C++ so it can be tested
 * on any system; only the dialog itself is Windows code
 * (platform/file_dialog.h).
 */

#include <cstddef>
#include <string>
#include <vector>

namespace gfx {

// `buffer` has room for `capacity` characters. Reads never go past it, so
// a buffer that lost its closing zeros still gives something sensible.
// Returns the full paths, in order, or nothing for an empty buffer
std::vector<std::wstring> parse_multi_select_buffer(const wchar_t* buffer,
                                                    std::size_t capacity);

} // namespace gfx
