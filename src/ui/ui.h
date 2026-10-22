#pragma once

/*
 * A small immediate-mode UI.
 *
 * "Immediate mode" means there are no widget objects. Every frame the
 * app describes the whole UI again with plain function calls, and each
 * call draws the widget and reports what the user did with it:
 *
 *     ui.begin_frame(draw_list, input);
 *     ui.begin_panel("Settings", 10, 10, 260);
 *     ui.text("Hello");
 *     if (ui.checkbox("Wireframe", &wireframe)) { ...toggled... }
 *     ui.slider("Sun azimuth", &azimuth, 0, 360, "%.0f deg");
 *     ui.end_panel();
 *     ui.end_frame();
 *
 * The app's variables are the only state. The Ui object itself only
 * remembers which widget the mouse is holding (so a slider keeps
 * following the mouse while dragged) and whether the mouse belongs to
 * the UI (wants_mouse), so the 3D view can ignore it.
 *
 * Widgets are identified by a hash of their panel title and label, so
 * labels must be unique within a panel.
 *
 * Sizes passed in are in 96-DPI pixels and are multiplied by the font's
 * DPI scale. Everything here is portable and tested (test_ui).
 */

#include "ui/ui_draw.h"

#include <cstdint>
#include <utility>
#include <vector>

namespace gfx {

// The mouse as the UI sees it: window pixels, left button only
struct UiInput {
    float mouse_x = -1.0e9f;
    float mouse_y = -1.0e9f;
    bool mouse_down = false;      // left button held
    bool mouse_pressed = false;   // went down this frame
    bool mouse_released = false;  // went up this frame
};

class Ui {
public:
    // Starts a frame. list must already have been begun with the font;
    // widgets are appended to it
    void begin_frame(UiDrawList& list, const UiInput& input);
    void end_frame();

    // True when the mouse belongs to the UI: it is over a panel, or a
    // widget is being dragged. Computed at end_frame, so during a frame
    // it describes the previous one. The app checks it before using the
    // mouse for the 3D view
    bool wants_mouse() const { return m_wants_mouse; }

    // A panel: a titled box whose widgets are stacked top to bottom.
    // (x, y) is its top-left corner and width its minimum width. It
    // grows to fit wider text: widgets are laid out with the width the
    // panel ended up with last frame, so after one frame everything
    // lines up again. Its height fits its contents
    void begin_panel(const char* title, float x, float y, float width);
    void end_panel();

    // Widgets after set_enabled(false) are drawn dimmed and ignore the
    // mouse. Every panel starts enabled
    void set_enabled(bool enabled) { m_enabled = enabled; }

    // One line of text
    void text(const char* text);

    // A section header that folds the widgets under it. Click to flip
    // *open. Returns *open, so the section's widgets go in an if:
    //     if (ui.header("Model", &model_open)) { ...widgets... }
    bool header(const char* label, bool* open);

    // A thin horizontal line between groups of widgets
    void separator();

    // Returns true on the frame the button is clicked (pressed and
    // released over it)
    bool button(const char* label);

    // Flips *value when clicked. Returns true on the frame it changed
    bool checkbox(const char* label, bool* value);

    // Drag (or click) along the track to set *value in [min, max].
    // format is a printf format for the value shown on the track, such
    // as "%.0f deg". Returns true on frames where *value changed
    bool slider(const char* label, float* value, float min, float max,
                const char* format);

    // Three values side by side (x, y, z), each changed by dragging
    // left/right on it: every pixel of mouse movement adds speed. No
    // limits. Returns true on frames where a value changed
    bool drag_float3(const char* label, float values[3], float speed,
                     const char* format);

private:
    struct Rect {
        float x0;
        float y0;
        float x1;
        float y1;

        bool contains(float x, float y) const {
            return x >= x0 && x < x1 && y >= y0 && y < y1;
        }
    };

    std::uint32_t widget_id(const char* label) const;

    // Hover / press / release bookkeeping shared by all widgets.
    // Returns true while the mouse is over rect (and free or held by id)
    bool interact(std::uint32_t id, const Rect& rect, bool& pressed,
                  bool& released);

    // Next row of the given height inside the current panel
    Rect next_row(float height);

    float px(float v) const;  // 96-DPI pixels to real pixels

    std::uint32_t text_color() const;
    std::uint32_t widget_color(bool hovered, bool held) const;
    void note_width(float right);

    UiDrawList* m_list = nullptr;
    UiInput m_input{};
    float m_scale = 1.0f;

    // Widget the mouse is holding (pressed on and not yet released).
    // 0 = none. A panel id here means a press on a panel's background
    std::uint32_t m_active = 0;
    bool m_active_seen = false;

    // Mouse x when a drag field last applied movement
    float m_drag_x = 0.0f;

    bool m_mouse_over_panel = false;
    bool m_wants_mouse = false;

    // The panel being built
    bool m_in_panel = false;
    std::uint32_t m_panel_id = 0;
    float m_panel_x = 0.0f;
    float m_panel_y = 0.0f;
    float m_panel_width = 0.0f;     // layout width this frame
    float m_min_width = 0.0f;       // the width asked for
    float m_content_right = 0.0f;   // widest content so far
    float m_cursor_y = 0.0f;
    std::size_t m_background = 0;
    std::size_t m_border[4] = {};
    bool m_enabled = true;

    // Final width of each panel last frame, by panel id
    std::vector<std::pair<std::uint32_t, float>> m_panel_widths;
};

} // namespace gfx
