#pragma once

#include <cstddef>

namespace gfx {

// Frame contract:
// begin_frame() -> platform feeds events -> end_frame() -> app queries.
//
// Mouse coordinates are client pixels, origin at top-left, with Y increasing
// downward. Pressed/released flags and scroll are cleared by begin_frame().
// Mouse deltas are calculated by end_frame().

enum class Key : unsigned char {
    GFX_KEY_UNKNOWN = 0,

    // Keep letters contiguous for Win32 virtual-key mapping
    GFX_KEY_A,
    GFX_KEY_B,
    GFX_KEY_C,
    GFX_KEY_D,
    GFX_KEY_E,
    GFX_KEY_F,
    GFX_KEY_G,
    GFX_KEY_H,
    GFX_KEY_I,
    GFX_KEY_J,
    GFX_KEY_K,
    GFX_KEY_L,
    GFX_KEY_M,
    GFX_KEY_N,
    GFX_KEY_O,
    GFX_KEY_P,
    GFX_KEY_Q,
    GFX_KEY_R,
    GFX_KEY_S,
    GFX_KEY_T,
    GFX_KEY_U,
    GFX_KEY_V,
    GFX_KEY_W,
    GFX_KEY_X,
    GFX_KEY_Y,
    GFX_KEY_Z,

    // Keep digits contiguous
    GFX_KEY_0,
    GFX_KEY_1,
    GFX_KEY_2,
    GFX_KEY_3,
    GFX_KEY_4,
    GFX_KEY_5,
    GFX_KEY_6,
    GFX_KEY_7,
    GFX_KEY_8,
    GFX_KEY_9,

    // Keep numpad digits contiguous
    GFX_KEY_NUMPAD_0,
    GFX_KEY_NUMPAD_1,
    GFX_KEY_NUMPAD_2,
    GFX_KEY_NUMPAD_3,
    GFX_KEY_NUMPAD_4,
    GFX_KEY_NUMPAD_5,
    GFX_KEY_NUMPAD_6,
    GFX_KEY_NUMPAD_7,
    GFX_KEY_NUMPAD_8,
    GFX_KEY_NUMPAD_9,

    GFX_KEY_ESCAPE,
    GFX_KEY_SPACE,
    GFX_KEY_TAB,
    GFX_KEY_DELETE,
    GFX_KEY_LEFT,
    GFX_KEY_RIGHT,
    GFX_KEY_UP,
    GFX_KEY_DOWN,
    GFX_KEY_SHIFT,
    GFX_KEY_CTRL,
    GFX_KEY_ALT,

    GFX_KEY_COUNT
};

enum class MouseButton : unsigned char {
    GFX_MOUSE_LEFT = 0,
    GFX_MOUSE_MIDDLE,
    GFX_MOUSE_RIGHT,
    GFX_MOUSE_COUNT
};

struct InputState {
    // Called by the platform layer
    inline void begin_frame() {
        for (std::size_t i = 0; i < key_count; ++i) {
            m_key_pressed[i] = false;
            m_key_released[i] = false;
        }

        for (std::size_t i = 0; i < mouse_button_count; ++i) {
            m_mouse_pressed[i] = false;
            m_mouse_released[i] = false;
        }

        m_scroll = 0.0f;
    }

    inline void end_frame() {
        if (!m_has_mouse_pos) {
            m_mouse_dx = 0.0f;
            m_mouse_dy = 0.0f;
            return;
        }

        if (m_has_prev_pos) {
            m_mouse_dx = m_mouse_x - m_prev_mouse_x;
            m_mouse_dy = m_mouse_y - m_prev_mouse_y;
        } else {
            m_mouse_dx = 0.0f;
            m_mouse_dy = 0.0f;
        }

        m_prev_mouse_x = m_mouse_x;
        m_prev_mouse_y = m_mouse_y;
        m_has_prev_pos = true;
    }

    inline void set_key(Key k, bool down) {
        const std::size_t i = static_cast<std::size_t>(k);

        if (i >= key_count || m_key_down[i] == down) {
            return;
        }

        m_key_down[i] = down;

        if (down) {
            m_key_pressed[i] = true;
        } else {
            m_key_released[i] = true;
        }
    }

    inline void set_mouse_button(MouseButton b, bool down) {
        const std::size_t i = static_cast<std::size_t>(b);

        if (i >= mouse_button_count || m_mouse_down[i] == down) {
            return;
        }

        m_mouse_down[i] = down;

        if (down) {
            m_mouse_pressed[i] = true;
        } else {
            m_mouse_released[i] = true;
        }
    }

    inline void set_mouse_position(float x, float y) {
        m_mouse_x = x;
        m_mouse_y = y;
        m_has_mouse_pos = true;
    }

    inline void add_scroll(float notches) {
        m_scroll += notches;
    }

    inline void release_all() {
        for (std::size_t i = 0; i < key_count; ++i) {
            set_key(static_cast<Key>(i), false);
        }

        for (std::size_t i = 0; i < mouse_button_count; ++i) {
            set_mouse_button(static_cast<MouseButton>(i), false);
        }

        // The next end_frame() must not calculate a delta from stale
        // coordinates after focus has been lost
        m_has_prev_pos = false;
        m_has_mouse_pos = false;   // wait for a fresh position after focus loss
    }

    // Queried by the application.
    inline bool key_down(Key k) const {
        const std::size_t i = static_cast<std::size_t>(k);
        return i < key_count && m_key_down[i];
    }

    inline bool key_pressed(Key k) const {
        const std::size_t i = static_cast<std::size_t>(k);
        return i < key_count && m_key_pressed[i];
    }

    inline bool key_released(Key k) const {
        const std::size_t i = static_cast<std::size_t>(k);
        return i < key_count && m_key_released[i];
    }

    inline bool mouse_down(MouseButton b) const {
        const std::size_t i = static_cast<std::size_t>(b);
        return i < mouse_button_count && m_mouse_down[i];
    }

    inline bool mouse_pressed(MouseButton b) const {
        const std::size_t i = static_cast<std::size_t>(b);
        return i < mouse_button_count && m_mouse_pressed[i];
    }

    inline bool mouse_released(MouseButton b) const {
        const std::size_t i = static_cast<std::size_t>(b);
        return i < mouse_button_count && m_mouse_released[i];
    }

    inline float mouse_x() const { return m_mouse_x; }
    inline float mouse_y() const { return m_mouse_y; }
    inline float mouse_dx() const { return m_mouse_dx; }
    inline float mouse_dy() const { return m_mouse_dy; }
    inline float scroll() const { return m_scroll; }

private:
    static constexpr std::size_t key_count =
        static_cast<std::size_t>(Key::GFX_KEY_COUNT);

    static constexpr std::size_t mouse_button_count =
        static_cast<std::size_t>(MouseButton::GFX_MOUSE_COUNT);

    bool m_key_down[key_count] = {};
    bool m_key_pressed[key_count] = {};
    bool m_key_released[key_count] = {};

    bool m_mouse_down[mouse_button_count] = {};
    bool m_mouse_pressed[mouse_button_count] = {};
    bool m_mouse_released[mouse_button_count] = {};

    float m_mouse_x = 0.0f;
    float m_mouse_y = 0.0f;
    float m_prev_mouse_x = 0.0f;
    float m_prev_mouse_y = 0.0f;
    float m_mouse_dx = 0.0f;
    float m_mouse_dy = 0.0f;
    float m_scroll = 0.0f;

    bool m_has_prev_pos = false;
    bool m_has_mouse_pos = false;
};

} // namespace gfx
