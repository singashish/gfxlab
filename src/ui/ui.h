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
 * A panel taller than the window (input.viewport_height) shrinks to fit
 * and scrolls: its title stays, the body moves with the mouse wheel or
 * a scrollbar on the right. Rows that are not fully inside the visible
 * body are not drawn and ignore the mouse; widgets still return what
 * they would (header returns *open, the others false). The draw list
 * has no clipping, so nothing is ever emitted outside the panel.
 *
 * Sizes passed in are in 96-DPI pixels and are multiplied by the font's
 * DPI scale. Everything here is portable and tested (test_ui).
 */

#include "ui/ui_draw.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gfx {

// The mouse as the UI sees it: window pixels, left button only, plus the
// wheel and the window height (for panels taller than the window)
struct UiInput {
    float mouse_x = -1.0e9f;
    float mouse_y = -1.0e9f;
    bool mouse_down = false;      // left button held
    bool mouse_pressed = false;   // went down this frame
    bool mouse_released = false;  // went up this frame
    float scroll = 0.0f;          // wheel notches this frame, up = positive
    float viewport_height = 0.0f; // window height in pixels; 0 = unknown
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

    // Scroll state of a panel as of its last end_panel (for tests and
    // the app): how far the body is scrolled and the largest offset it
    // can have, in pixels. max is 0 when the panel fits (no scrollbar)
    float panel_scroll(const char* title) const;
    float panel_max_scroll(const char* title) const;

    // A panel: a titled box whose widgets are stacked top to bottom.
    // (x, y) is its top-left corner and width its minimum width. It
    // grows to fit wider text: widgets are laid out with the width the
    // panel ended up with last frame, so after one frame everything
    // lines up again. Its height fits its contents, up to the window
    // bottom: then it scrolls (the layout needs one frame to notice)
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

    // Next row of the given height inside the current panel. Sets
    // m_row_visible: false when scrolling puts the row (even partly)
    // outside the visible body. Hidden rows must not draw or interact
    Rect next_row(float height);

    // Scroll mode, once the panel's title and separator are in: applies
    // the wheel and the scrollbar thumb to the panel's offset
    struct PanelState;
    void update_scroll(PanelState& state);
    std::uint32_t thumb_id() const;

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

    // Scroll mode (the panel is taller than the window)
    bool m_scrolling = false;      // this frame's body scrolls
    bool m_scroll_rows = false;    // title is in: rows are offset
    bool m_row_visible = true;     // the row next_row just made
    bool m_bar_hot = false;        // mouse over the scrollbar track
    float m_max_height = 0.0f;     // tallest the panel may be; 0 = no limit
    float m_bar_inset = 0.0f;      // right edge kept free for the bar
    float m_scroll = 0.0f;         // body offset in pixels
    float m_body_top = 0.0f;       // visible body band
    float m_body_bottom = 0.0f;
    float m_thumb_grab = 0.0f;     // mouse y inside the thumb when grabbed

    // What each panel looked like last frame, by panel id
    struct PanelState {
        std::uint32_t id = 0;
        float width = 0.0f;       // width its content needs (no scrollbar)
        float height = 0.0f;      // height its content needs
        float right = 0.0f;       // right edge as drawn
        float scroll = 0.0f;      // body offset
        float max_scroll = 0.0f;  // 0 when it fits
    };

    PanelState* find_panel(std::uint32_t id);
    const PanelState* find_panel(std::uint32_t id) const;

    std::vector<PanelState> m_panels;
};

} // namespace gfx
