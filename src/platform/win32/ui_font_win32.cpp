#include "platform/ui_font.h"
#include "core/log.h"

// WIN32_LEAN_AND_MEAN, NOMINMAX and UNICODE are defined
// project-wide in the root CMakeLists.txt
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace gfx {

namespace {

// GGO_GRAY8_BITMAP gives 65 coverage levels, 0..64
constexpr unsigned int kGdiGrayLevels = 64;

// Releases the GDI objects on every return path
struct GdiFont {
    HDC dc = nullptr;
    HFONT font = nullptr;
    HGDIOBJ previous = nullptr;

    ~GdiFont() {
        if (dc != nullptr && previous != nullptr) {
            SelectObject(dc, previous);
        }

        if (font != nullptr) {
            DeleteObject(font);
        }

        if (dc != nullptr) {
            DeleteDC(dc);
        }
    }
};

// One glyph from the selected font, or a blank glyph with the right
// advance when the font has no outline for it (space, for example)
GlyphBitmap rasterize_glyph(HDC dc, wchar_t ch, int ascent) {
    GlyphBitmap g;

    // Identity transform. FIXED is {fraction, integer part}
    MAT2 identity = {};
    identity.eM11.value = 1;
    identity.eM22.value = 1;

    GLYPHMETRICS gm = {};

    const DWORD size = GetGlyphOutlineW(
        dc, ch, GGO_GRAY8_BITMAP, &gm, 0, nullptr, &identity);

    if (size == GDI_ERROR) {
        INT width = 0;

        if (GetCharWidth32W(dc, ch, ch, &width)) {
            g.advance = width;
        }

        return g;
    }

    g.advance = gm.gmCellIncX;

    if (size == 0) {
        return g;  // blank glyph
    }

    // Each source row is padded to a multiple of 4 bytes
    const int width = static_cast<int>(gm.gmBlackBoxX);
    const int height = static_cast<int>(gm.gmBlackBoxY);
    const int pitch = (width + 3) & ~3;

    if (width <= 0 || height <= 0 ||
        size < static_cast<DWORD>(pitch * height)) {
        return g;
    }

    std::vector<BYTE> buffer(size);

    if (GetGlyphOutlineW(dc, ch, GGO_GRAY8_BITMAP, &gm, size,
                         buffer.data(), &identity) == GDI_ERROR) {
        return g;
    }

    g.width = width;
    g.height = height;

    // gmptGlyphOrigin is the black box's top-left corner relative to the
    // pen on the baseline, with Y pointing up. Our offsets are relative
    // to the top of the line, with Y pointing down
    g.offset_x = static_cast<int>(gm.gmptGlyphOrigin.x);
    g.offset_y = ascent - static_cast<int>(gm.gmptGlyphOrigin.y);

    g.alpha.resize(static_cast<std::size_t>(width) *
                   static_cast<std::size_t>(height));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const unsigned int level = std::min<unsigned int>(
                buffer[static_cast<std::size_t>(y * pitch + x)],
                kGdiGrayLevels);

            g.alpha[static_cast<std::size_t>(y * width + x)] =
                static_cast<std::uint8_t>(
                    (level * 255u + kGdiGrayLevels / 2) / kGdiGrayLevels);
        }
    }

    return g;
}

} // namespace

bool build_ui_font(void* native_window,
                   const char* face,
                   float size,
                   FontAtlas& out) {
    out = FontAtlas{};

    if (face == nullptr || size <= 0.0f) {
        return false;
    }

    // DPI of the monitor the window is on (96 = 100% scaling)
    UINT dpi = 0;

    if (native_window != nullptr) {
        dpi = GetDpiForWindow(static_cast<HWND>(native_window));
    }

    if (dpi == 0) {
        dpi = GetDpiForSystem();
    }

    const float scale = static_cast<float>(dpi) / 96.0f;
    const int pixel_height =
        std::max(6, static_cast<int>(std::lround(size * scale)));

    // The face name is ASCII, so widening each byte is enough
    const std::wstring wide_face(face, face + std::strlen(face));

    GdiFont gdi;
    gdi.dc = CreateCompatibleDC(nullptr);

    if (gdi.dc == nullptr) {
        LOG_ERROR("UI font: CreateCompatibleDC failed: %lu",
                  static_cast<unsigned long>(GetLastError()));
        return false;
    }

    // A negative height asks for that character height (without the
    // internal leading). ANTIALIASED_QUALITY gives grayscale coverage;
    // ClearType's coloured subpixels would not survive blending over
    // an arbitrary 3D view
    gdi.font = CreateFontW(
        -pixel_height, 0, 0, 0,
        FW_NORMAL, FALSE, FALSE, FALSE,
        DEFAULT_CHARSET,
        OUT_TT_ONLY_PRECIS,
        CLIP_DEFAULT_PRECIS,
        ANTIALIASED_QUALITY,
        DEFAULT_PITCH | FF_DONTCARE,
        wide_face.c_str());

    if (gdi.font == nullptr) {
        LOG_ERROR("UI font: CreateFontW(\"%s\") failed", face);
        return false;
    }

    gdi.previous = SelectObject(gdi.dc, gdi.font);

    TEXTMETRICW tm = {};

    if (!GetTextMetricsW(gdi.dc, &tm)) {
        LOG_ERROR("UI font: GetTextMetricsW failed");
        return false;
    }

    const int line_height = static_cast<int>(tm.tmHeight);
    const int ascent = static_cast<int>(tm.tmAscent);

    std::vector<GlyphBitmap> glyphs;
    glyphs.reserve(kFontCharCount);

    for (int c = kFontFirstChar; c <= kFontLastChar; ++c) {
        glyphs.push_back(rasterize_glyph(
            gdi.dc, static_cast<wchar_t>(c), ascent));
    }

    if (!pack_font_atlas(glyphs, line_height, ascent, scale, out)) {
        LOG_ERROR("UI font: packing the atlas failed");
        return false;
    }

    LOG_INFO("UI font: %s, %d px (DPI %u), atlas %dx%d",
             face, pixel_height, dpi, out.width, out.height);
    return true;
}

} // namespace gfx
