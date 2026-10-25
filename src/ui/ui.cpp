#include "ui/ui.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace gfx {

namespace {

// Layout, in 96-DPI pixels
constexpr float kPadding = 8.0f;      // inside the panel border
constexpr float kSpacing = 4.0f;      // between rows
constexpr float kRowExtra = 6.0f;     // widget rows: line height + this
constexpr float kLabelFraction = 0.42f;  // slider: share of the label

// Scrolling panels, in 96-DPI pixels
constexpr float kBottomMargin = 10.0f;   // kept free below a tall panel
constexpr float kMinPanelHeight = 80.0f; // never squeezed below this
constexpr float kBarWidth = 5.0f;        // scrollbar track and thumb
constexpr float kBarMargin = 3.0f;       // scrollbar to the panel edge
constexpr float kMinThumb = 16.0f;
constexpr float kWheelLines = 3.0f;      // text lines per wheel notch

// Colours
constexpr std::uint32_t kPanelColor = ui_rgba(22, 24, 28, 225);
constexpr std::uint32_t kBorderColor = ui_rgba(255, 255, 255, 40);
constexpr std::uint32_t kTitleColor = ui_rgba(255, 255, 255, 255);
constexpr std::uint32_t kTextColor = ui_rgba(196, 202, 210, 255);
constexpr std::uint32_t kSeparatorColor = ui_rgba(255, 255, 255, 30);
constexpr std::uint32_t kWidgetColor = ui_rgba(52, 56, 64, 255);
constexpr std::uint32_t kHoverColor = ui_rgba(66, 71, 81, 255);
constexpr std::uint32_t kHeldColor = ui_rgba(84, 90, 102, 255);
constexpr std::uint32_t kAccentColor = ui_rgba(74, 138, 226, 255);
constexpr std::uint32_t kValueColor = ui_rgba(240, 242, 245, 255);
constexpr std::uint32_t kHeaderColor = ui_rgba(38, 42, 49, 255);
constexpr std::uint32_t kTrackColor = ui_rgba(255, 255, 255, 16);
constexpr std::uint32_t kThumbColor = ui_rgba(92, 98, 110, 255);
constexpr std::uint32_t kThumbHotColor = ui_rgba(120, 127, 141, 255);

// Axis colours for x, y, z fields, as on the floor grid
constexpr std::uint32_t kAxisColors[3] = {
    ui_rgba(217, 64, 77, 255),
    ui_rgba(115, 199, 64, 255),
    ui_rgba(74, 128, 226, 255)
};

// Disabled widgets: same colour, about 40% opacity
std::uint32_t dimmed(std::uint32_t color) {
    const std::uint32_t alpha = (color >> 24) * 2 / 5;
    return (color & 0x00FFFFFFu) | (alpha << 24);
}

// FNV-1a, continued from seed
std::uint32_t hash_string(const char* s, std::uint32_t seed) {
    std::uint32_t h = seed;

    for (; s != nullptr && *s != '\0'; ++s) {
        h ^= static_cast<unsigned char>(*s);
        h *= 16777619u;
    }

    return h;
}

constexpr std::uint32_t kFnvBasis = 2166136261u;

// The scrollbar thumb's top and bottom inside a track that starts at
// track_y0 and is track_h tall. The thumb is as long as the visible part
// is of the whole body, and rides the track as scroll goes to max_scroll
void thumb_span(float track_y0, float track_h, float visible, float content,
                float scroll, float max_scroll, float min_length,
                float& y0, float& y1) {
    const float length = std::clamp(
        content > 0.0f ? track_h * visible / content : track_h,
        std::min(min_length, track_h), track_h);
    const float t = max_scroll > 0.0f ? scroll / max_scroll : 0.0f;

    y0 = track_y0 + t * (track_h - length);
    y1 = y0 + length;
}

} // namespace

float Ui::px(float v) const {
    return std::round(v * m_scale);
}

std::uint32_t Ui::text_color() const {
    return m_enabled ? kTextColor : dimmed(kTextColor);
}

std::uint32_t Ui::widget_color(bool hovered, bool held) const {
    std::uint32_t color = kWidgetColor;

    if (held) {
        color = kHeldColor;
    } else if (hovered) {
        color = kHoverColor;
    }

    return m_enabled ? color : dimmed(color);
}

void Ui::note_width(float right) {
    m_content_right = std::max(m_content_right, right);
}

Ui::PanelState* Ui::find_panel(std::uint32_t id) {
    for (PanelState& panel : m_panels) {
        if (panel.id == id) {
            return &panel;
        }
    }

    return nullptr;
}

const Ui::PanelState* Ui::find_panel(std::uint32_t id) const {
    for (const PanelState& panel : m_panels) {
        if (panel.id == id) {
            return &panel;
        }
    }

    return nullptr;
}

float Ui::panel_scroll(const char* title) const {
    const PanelState* panel = find_panel(hash_string(title, kFnvBasis));
    return panel != nullptr ? panel->scroll : 0.0f;
}

float Ui::panel_max_scroll(const char* title) const {
    const PanelState* panel = find_panel(hash_string(title, kFnvBasis));
    return panel != nullptr ? panel->max_scroll : 0.0f;
}

std::uint32_t Ui::thumb_id() const {
    const std::uint32_t id = m_panel_id ^ 0xB5297A4Du;
    return id != 0 ? id : 1;
}

std::uint32_t Ui::widget_id(const char* label) const {
    const std::uint32_t id = hash_string(label, m_panel_id ^ 0x9E3779B9u);
    return id != 0 ? id : 1;  // 0 means "no widget"
}

void Ui::begin_frame(UiDrawList& list, const UiInput& input) {
    m_list = &list;
    m_input = input;
    m_scale = list.font() != nullptr ? list.font()->scale : 1.0f;
    m_active_seen = false;
    m_mouse_over_panel = false;
    m_in_panel = false;
}

void Ui::end_frame() {
    // The held widget was not drawn this frame (its panel was hidden,
    // for example): let go of it
    if (m_active != 0 && !m_active_seen) {
        m_active = 0;
    }

    // Widgets saw this frame's release; now the mouse is free again
    if (m_active != 0 && !m_input.mouse_down) {
        m_active = 0;
    }

    m_wants_mouse = m_mouse_over_panel || m_active != 0;
    m_list = nullptr;
}

void Ui::begin_panel(const char* title, float x, float y, float width) {
    if (m_list == nullptr) {
        return;
    }

    m_in_panel = true;
    m_enabled = true;
    m_panel_id = hash_string(title, kFnvBasis);
    m_panel_x = px(x);
    m_panel_y = px(y);
    m_min_width = px(width);
    m_panel_width = m_min_width;

    // Lay out with last frame's width if the panel had grown
    PanelState* state = find_panel(m_panel_id);

    if (state != nullptr) {
        m_panel_width = std::max(m_panel_width, state->width);
    }

    // Taller than the window last frame: scroll mode. The scrollbar gets
    // its own strip on the right, added to the panel's width. The width
    // remembered for next frame never includes it, so it cannot make the
    // width grow from frame to frame
    m_scrolling = false;
    m_scroll_rows = false;
    m_row_visible = true;
    m_bar_hot = false;
    m_bar_inset = 0.0f;
    m_scroll = 0.0f;
    m_max_height = 0.0f;

    if (m_input.viewport_height > 0.0f) {
        m_max_height = std::max(
            m_input.viewport_height - m_panel_y - px(kBottomMargin),
            px(kMinPanelHeight));
        m_scrolling = state != nullptr && state->height > m_max_height + 0.5f;
    }

    if (m_scrolling) {
        m_panel_width += px(kBarWidth);
    }

    // Content that fits the minimum width needs no growth
    m_content_right = m_panel_x + m_min_width - px(kPadding);

    // Background and border first, so they are drawn under the widgets.
    // Their size is filled in by end_panel
    m_background = m_list->add_rect_placeholder(kPanelColor);

    for (std::size_t& edge : m_border) {
        edge = m_list->add_rect_placeholder(kBorderColor);
    }

    m_cursor_y = m_panel_y + px(kPadding);

    // Title row, then a line under it
    const float title_x = m_panel_x + px(kPadding);
    const float title_width =
        m_list->add_text(title_x, m_cursor_y, title, kTitleColor);

    note_width(title_x + title_width);
    m_cursor_y += m_list->line_height() + px(kSpacing);

    separator();

    // From here on the rows are the body: they scroll, the title stays
    m_body_top = m_cursor_y;

    if (m_scrolling) {
        update_scroll(*state);
    } else if (state != nullptr) {
        state->scroll = 0.0f;
    }
}

void Ui::update_scroll(PanelState& state) {
    const float pad = px(kPadding);
    const float bar_width = px(kBarWidth);

    m_body_bottom = m_panel_y + m_max_height - pad;

    const float visible = m_body_bottom - m_body_top;
    const float content = state.height - (m_body_top - m_panel_y) - pad;
    const float max_scroll = std::max(0.0f, content - visible);
    float offset = state.scroll;

    // The track, where the bar was drawn last frame
    const float bar_right = state.right - px(kBarMargin);
    const Rect track{bar_right - bar_width, m_body_top, bar_right,
                     m_body_bottom};
    const std::uint32_t id = thumb_id();

    bool pressed = false;
    bool released = false;
    m_bar_hot = interact(id, track, pressed, released);

    float thumb_y0 = 0.0f;
    float thumb_y1 = 0.0f;
    thumb_span(track.y0, visible, visible, content, offset, max_scroll,
               px(kMinThumb), thumb_y0, thumb_y1);

    // Grab the thumb where it is, or jump it to the mouse when the press
    // is on the track beside it; then it follows the mouse, also outside
    // the track
    if (pressed) {
        const bool on_thumb =
            m_input.mouse_y >= thumb_y0 && m_input.mouse_y < thumb_y1;

        m_thumb_grab = on_thumb ? m_input.mouse_y - thumb_y0
                                : (thumb_y1 - thumb_y0) * 0.5f;
    }

    if (m_active == id) {
        const float travel = visible - (thumb_y1 - thumb_y0);

        if (travel > 0.0f) {
            const float t = std::clamp(
                (m_input.mouse_y - m_thumb_grab - track.y0) / travel,
                0.0f, 1.0f);

            offset = t * max_scroll;
        }
    } else if (m_input.scroll != 0.0f &&
               (m_active == 0 || m_active == m_panel_id)) {
        // The wheel scrolls while the mouse is over the panel
        const Rect panel{m_panel_x, m_panel_y, state.right,
                         m_panel_y + m_max_height};

        if (panel.contains(m_input.mouse_x, m_input.mouse_y)) {
            offset -= m_input.scroll * kWheelLines * m_list->line_height();
        }
    }

    // Whole pixels keep text sharp
    offset = std::clamp(std::round(offset), 0.0f, max_scroll);

    state.scroll = offset;
    m_scroll = offset;
    m_scroll_rows = true;
    m_bar_inset = bar_width;
}

void Ui::end_panel() {
    if (m_list == nullptr || !m_in_panel) {
        return;
    }

    const float pad = px(kPadding);
    const float x0 = m_panel_x;
    const float y0 = m_panel_y;
    // Wide enough for the minimum width, the widest content, and the
    // widgets already laid out this frame
    const float bar_width = m_scrolling ? px(kBarWidth) : 0.0f;
    const float x1 = std::max(m_panel_x + m_panel_width,
                              m_content_right + pad + bar_width);
    const float full_height = m_cursor_y - px(kSpacing) + pad - y0;

    // Too tall for the window: stop at the bottom margin, the body
    // scrolls. The bar is drawn only if the content really overflows
    // (it may have shrunk since last frame)
    const bool overflow = m_scrolling && full_height > m_max_height + 0.5f;
    const float y1 = y0 + (overflow ? m_max_height : full_height);
    const float border = std::max(1.0f, px(1.0f));

    m_list->set_rect(m_background, x0, y0, x1, y1);
    m_list->set_rect(m_border[0], x0, y0, x1, y0 + border);           // top
    m_list->set_rect(m_border[1], x0, y1 - border, x1, y1);           // bottom
    m_list->set_rect(m_border[2], x0, y0 + border, x0 + border,
                     y1 - border);                                    // left
    m_list->set_rect(m_border[3], x1 - border, y0 + border, x1,
                     y1 - border);                                    // right

    // The scrollbar: a faint track with the thumb on it, inside the right
    // edge, as long as the body is
    if (overflow) {
        const float visible = m_body_bottom - m_body_top;
        const float content = full_height - (m_body_top - y0) - pad;
        const float max_scroll = std::max(0.0f, content - visible);
        const float bar_right = x1 - px(kBarMargin);
        const float bar_left = bar_right - bar_width;
        float thumb_y0 = 0.0f;
        float thumb_y1 = 0.0f;

        thumb_span(m_body_top, visible, visible, content, m_scroll,
                   max_scroll, px(kMinThumb), thumb_y0, thumb_y1);

        const bool hot = m_bar_hot || m_active == thumb_id();

        m_list->add_rect(bar_left, m_body_top, bar_right, m_body_bottom,
                         kTrackColor);
        m_list->add_rect(bar_left, thumb_y0, bar_right, thumb_y1,
                         hot ? kThumbHotColor : kThumbColor);
    }

    // Remember the width the content needs (not the layout width, so a
    // panel shrinks again once its wide text is gone) and the height, so
    // next frame knows whether to scroll
    const float wanted = std::max(m_min_width, m_content_right + pad - x0);
    PanelState* state = find_panel(m_panel_id);

    if (state == nullptr) {
        m_panels.emplace_back();
        state = &m_panels.back();
        state->id = m_panel_id;
    }

    state->width = wanted;
    state->height = full_height;
    state->right = x1;
    state->max_scroll =
        overflow ? std::max(0.0f, full_height - m_max_height) : 0.0f;
    state->scroll =
        overflow ? std::min(m_scroll, state->max_scroll) : 0.0f;

    const Rect panel{x0, y0, x1, y1};

    if (panel.contains(m_input.mouse_x, m_input.mouse_y)) {
        m_mouse_over_panel = true;

        // A press on the panel's background holds the mouse too, so a
        // drag that starts on the panel never turns into a camera move
        if (m_input.mouse_pressed && m_active == 0) {
            m_active = m_panel_id;
        }
    }

    if (m_active == m_panel_id) {
        m_active_seen = true;
    }

    m_in_panel = false;
    m_scrolling = false;
    m_scroll_rows = false;
}

Ui::Rect Ui::next_row(float height) {
    const float pad = px(kPadding);

    // The cursor is the row's place without scrolling; scrolling moves
    // the body up. A row not entirely inside the body band is hidden
    const float y0 = m_scroll_rows ? m_cursor_y - m_scroll : m_cursor_y;

    Rect row{m_panel_x + pad, y0,
             m_panel_x + m_panel_width - pad - m_bar_inset, y0 + height};

    m_row_visible = !m_scroll_rows ||
        (row.y0 >= m_body_top - 0.5f && row.y1 <= m_body_bottom + 0.5f);

    m_cursor_y += height + px(kSpacing);
    return row;
}

bool Ui::interact(std::uint32_t id, const Rect& rect, bool& pressed,
                  bool& released) {
    pressed = false;
    released = false;

    if (!m_enabled) {
        return false;
    }

    if (m_active == id) {
        m_active_seen = true;
    }

    // While another widget holds the mouse, this one does not react
    const bool hovered =
        rect.contains(m_input.mouse_x, m_input.mouse_y) &&
        (m_active == 0 || m_active == id);

    if (hovered && m_input.mouse_pressed && m_active == 0) {
        m_active = id;
        m_active_seen = true;
        pressed = true;
    }

    if (m_active == id && m_input.mouse_released) {
        released = true;
    }

    return hovered;
}

void Ui::text(const char* text) {
    if (m_list == nullptr || !m_in_panel) {
        return;
    }

    const Rect row = next_row(m_list->line_height());

    // Hidden text still counts for the panel's width: scrolling must not
    // change it
    const float width = m_row_visible
        ? m_list->add_text(row.x0, row.y0, text, text_color())
        : m_list->text_width(text);

    note_width(row.x0 + width);
}

void Ui::separator() {
    if (m_list == nullptr || !m_in_panel) {
        return;
    }

    const Rect row = next_row(px(5.0f));

    if (!m_row_visible) {
        return;
    }

    const float thickness = std::max(1.0f, px(1.0f));
    const float y = std::round((row.y0 + row.y1 - thickness) * 0.5f);

    m_list->add_rect(row.x0, y, row.x1, y + thickness, kSeparatorColor);
}

bool Ui::button(const char* label) {
    if (m_list == nullptr || !m_in_panel) {
        return false;
    }

    const float line = m_list->line_height();
    const Rect row = next_row(line + px(kRowExtra));
    const std::uint32_t id = widget_id(label);
    const float width = m_list->text_width(label);

    note_width(row.x0 + width + 2.0f * px(kPadding));

    if (!m_row_visible) {
        return false;
    }

    bool pressed = false;
    bool released = false;
    const bool hovered = interact(id, row, pressed, released);
    const bool held = m_enabled && m_active == id && m_input.mouse_down;

    std::uint32_t background = kWidgetColor;

    if (held) {
        background = kHeldColor;
    } else if (hovered) {
        background = kHoverColor;
    }

    if (!m_enabled) {
        background = dimmed(background);
    }

    m_list->add_rect(row.x0, row.y0, row.x1, row.y1, background);

    // Label centred in the button
    m_list->add_text((row.x0 + row.x1 - width) * 0.5f,
                     row.y0 + (row.y1 - row.y0 - line) * 0.5f,
                     label, text_color());

    // A click is a press and a release both over the button
    return released && hovered;
}

bool Ui::checkbox(const char* label, bool* value) {
    if (m_list == nullptr || !m_in_panel || value == nullptr) {
        return false;
    }

    const float line = m_list->line_height();
    const Rect row = next_row(line + px(kRowExtra));
    const std::uint32_t id = widget_id(label);

    // The box: a square as tall as the text, centred in the row
    const float box = std::round(line * 0.9f);
    const float bx0 = row.x0;

    if (!m_row_visible) {
        note_width(bx0 + box + px(6.0f) + m_list->text_width(label));
        return false;
    }

    bool pressed = false;
    bool released = false;
    const bool hovered = interact(id, row, pressed, released);
    const float by0 = std::round(row.y0 + (row.y1 - row.y0 - box) * 0.5f);

    std::uint32_t background = hovered ? kHoverColor : kWidgetColor;

    if (!m_enabled) {
        background = dimmed(background);
    }

    m_list->add_rect(bx0, by0, bx0 + box, by0 + box, background);

    // Ticked: a filled square inside the box
    if (*value) {
        const float inset = std::max(2.0f, std::round(box * 0.22f));
        m_list->add_rect(bx0 + inset, by0 + inset,
                         bx0 + box - inset, by0 + box - inset,
                         m_enabled ? kAccentColor : dimmed(kAccentColor));
    }

    const float text_x = bx0 + box + px(6.0f);
    const float width = m_list->add_text(
        text_x, row.y0 + (row.y1 - row.y0 - line) * 0.5f, label,
        text_color());

    note_width(text_x + width);

    if (released && hovered) {
        *value = !*value;
        return true;
    }

    return false;
}

bool Ui::slider(const char* label, float* value, float min, float max,
                const char* format) {
    if (m_list == nullptr || !m_in_panel || value == nullptr ||
        !(max > min)) {
        return false;
    }

    const float line = m_list->line_height();
    const Rect row = next_row(line + px(kRowExtra));
    const std::uint32_t id = widget_id(label);

    if (!m_row_visible) {
        return false;
    }

    // Label on the left, track on the right
    const float label_width =
        std::round((row.x1 - row.x0) * kLabelFraction);
    const Rect track{row.x0 + label_width, row.y0, row.x1, row.y1};

    const float text_y = row.y0 + (row.y1 - row.y0 - line) * 0.5f;
    m_list->add_text(row.x0, text_y, label, text_color());

    bool pressed = false;
    bool released = false;
    const bool hovered = interact(id, track, pressed, released);

    // While held, the value follows the mouse, even outside the track
    bool changed = false;

    if (m_enabled && m_active == id) {
        const float t = std::clamp(
            (m_input.mouse_x - track.x0) / (track.x1 - track.x0), 0.0f, 1.0f);
        const float new_value = min + t * (max - min);

        if (new_value != *value) {
            *value = new_value;
            changed = true;
        }
    }

    const bool held = m_enabled && m_active == id;
    std::uint32_t background = kWidgetColor;

    if (held) {
        background = kHeldColor;
    } else if (hovered) {
        background = kHoverColor;
    }

    if (!m_enabled) {
        background = dimmed(background);
    }

    m_list->add_rect(track.x0, track.y0, track.x1, track.y1, background);

    // Filled part: how far the value is between min and max
    const float fraction =
        std::clamp((*value - min) / (max - min), 0.0f, 1.0f);
    const float fill_x = std::round(track.x0 + fraction * (track.x1 - track.x0));
    const std::uint32_t fill = ui_rgba(74, 138, 226, 150);

    m_list->add_rect(track.x0, track.y0, fill_x, track.y1,
                     m_enabled ? fill : dimmed(fill));

    // Value, centred on the track
    char text[64];
    std::snprintf(text, sizeof(text), format != nullptr ? format : "%.2f",
                  static_cast<double>(*value));

    const float value_width = m_list->text_width(text);
    m_list->add_text((track.x0 + track.x1 - value_width) * 0.5f, text_y,
                     text, m_enabled ? kValueColor : dimmed(kValueColor));

    return changed;
}

bool Ui::header(const char* label, bool* open) {
    if (m_list == nullptr || !m_in_panel || open == nullptr) {
        return false;
    }

    const float line = m_list->line_height();
    const Rect row = next_row(line + px(kRowExtra));
    const std::uint32_t id = widget_id(label);

    if (!m_row_visible) {
        // Scrolled out: it keeps its state and the section under it
        // still follows *open
        const float hidden_size = std::round(line * 0.4f);

        note_width(row.x0 + px(4.0f) + hidden_size + px(6.0f) +
                   m_list->text_width(label));
        return *open;
    }

    // Headers stay usable even after set_enabled(false)
    const bool enabled = m_enabled;
    m_enabled = true;

    bool pressed = false;
    bool released = false;
    const bool hovered = interact(id, row, pressed, released);

    m_list->add_rect(row.x0, row.y0, row.x1, row.y1,
                     hovered ? kHoverColor : kHeaderColor);

    // Arrow: pointing down when open, right when folded
    const float size = std::round(line * 0.4f);
    const float cx = row.x0 + px(4.0f) + size * 0.5f;
    const float cy = (row.y0 + row.y1) * 0.5f;

    if (*open) {
        m_list->add_triangle(cx - size * 0.5f, cy - size * 0.3f,
                             cx + size * 0.5f, cy - size * 0.3f,
                             cx, cy + size * 0.4f, kTitleColor);
    } else {
        m_list->add_triangle(cx - size * 0.3f, cy - size * 0.5f,
                             cx + size * 0.4f, cy,
                             cx - size * 0.3f, cy + size * 0.5f, kTitleColor);
    }

    const float text_x = row.x0 + px(4.0f) + size + px(6.0f);
    const float width = m_list->add_text(
        text_x, row.y0 + (row.y1 - row.y0 - line) * 0.5f, label, kTitleColor);

    note_width(text_x + width);

    if (released && hovered) {
        *open = !*open;
    }

    m_enabled = enabled;
    return *open;
}

bool Ui::drag_float3(const char* label, float values[3], float speed,
                     const char* format) {
    if (m_list == nullptr || !m_in_panel || values == nullptr) {
        return false;
    }

    const float line = m_list->line_height();
    const Rect row = next_row(line + px(kRowExtra));

    if (!m_row_visible) {
        return false;
    }

    const float text_y = row.y0 + (row.y1 - row.y0 - line) * 0.5f;

    m_list->add_text(row.x0, text_y, label, text_color());

    // Three equal fields to the right of the label
    const float label_width =
        std::round((row.x1 - row.x0) * kLabelFraction);
    const float gap = px(4.0f);
    const float field_width =
        (row.x1 - row.x0 - label_width - 2.0f * gap) / 3.0f;
    const float strip = std::max(2.0f, px(2.0f));

    const std::uint32_t base_id = widget_id(label);
    bool changed = false;

    for (int i = 0; i < 3; ++i) {
        const float x0 = std::round(row.x0 + label_width +
                                    static_cast<float>(i) * (field_width + gap));
        const float x1 = std::round(x0 + field_width);
        const Rect field{x0, row.y0, x1, row.y1};
        const std::uint32_t id = base_id + static_cast<std::uint32_t>(i) + 1u;

        bool pressed = false;
        bool released = false;
        const bool hovered = interact(id, field, pressed, released);
        const bool held = m_enabled && m_active == id;

        // While held, horizontal mouse movement changes the value
        if (pressed) {
            m_drag_x = m_input.mouse_x;
        } else if (held) {
            const float dx = m_input.mouse_x - m_drag_x;
            m_drag_x = m_input.mouse_x;

            if (dx != 0.0f) {
                values[i] += dx * speed;
                changed = true;
            }
        }

        m_list->add_rect(x0, row.y0, x1, row.y1, widget_color(hovered, held));

        // Thin axis-coloured strip on the left edge
        m_list->add_rect(x0, row.y0, x0 + strip, row.y1,
                         m_enabled ? kAxisColors[i] : dimmed(kAxisColors[i]));

        char text[32];
        std::snprintf(text, sizeof(text), format != nullptr ? format : "%.2f",
                      static_cast<double>(values[i]));

        const float width = m_list->text_width(text);
        m_list->add_text((x0 + x1 - width) * 0.5f, text_y, text,
                         m_enabled ? kValueColor : dimmed(kValueColor));
    }

    return changed;
}

} // namespace gfx
