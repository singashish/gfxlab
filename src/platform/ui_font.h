#pragma once

#include "ui/font_atlas.h"

namespace gfx {

// Rasterizes characters 32..126 of an installed font (face, ASCII name
// such as "Segoe UI") with the operating system's font rasterizer and
// packs them into out.
//
// size is the font's pixel height at 96 DPI. It is multiplied by the DPI
// scale of the monitor the window is on (1.5 at 144 DPI, for example),
// and out.scale records that factor for the UI layout.
// Returns false (and logs why) if the font cannot be created
bool build_ui_font(void* native_window,
                   const char* face,
                   float size,
                   FontAtlas& out);

} // namespace gfx
