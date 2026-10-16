#include "core/transform_edit.h"

#include <algorithm>
#include <cmath>

namespace gfx {
namespace {

constexpr float kMinDepth = 0.01f;
constexpr float kGrabMinDepth = 0.1f;
constexpr float kMinScaleFactor = 0.01f;
constexpr float kMinPivotDistance = 8.0f;

float clamp_value(float value, float lo, float hi) {
    return std::max(lo, std::min(value, hi));
}

// Return an angle in (-pi, pi]
float wrap_angle(float angle) {
    constexpr float two_pi = 2.0f * kPi;

    angle = std::fmod(angle, two_pi);

    if (angle <= -kPi) {
        angle += two_pi;
    } else if (angle > kPi) {
        angle -= two_pi;
    }

    return angle;
}

float distance_2d(float x0, float y0, float x1, float y1) {
    const float dx = x0 - x1;
    const float dy = y0 - y1;
    return std::sqrt(dx * dx + dy * dy);
}

Vec3 axis_vector(EditAxis axis) {
    switch (axis) {
    case EditAxis::EDIT_AXIS_X:
        return {1.0f, 0.0f, 0.0f};

    case EditAxis::EDIT_AXIS_Y:
        return {0.0f, 1.0f, 0.0f};

    case EditAxis::EDIT_AXIS_Z:
        return {0.0f, 0.0f, 1.0f};

    case EditAxis::EDIT_AXIS_FREE:
    default:
        return {0.0f, 0.0f, 0.0f};
    }
}

Vec3 constrain_to_axis(Vec3 delta, EditAxis axis) {
    if (axis == EditAxis::EDIT_AXIS_FREE) {
        return delta;
    }

    const Vec3 a = axis_vector(axis);
    return a * dot(delta, a);
}

} // namespace

bool project_to_screen(const EditView& view,
                       const Vec3& world,
                       float& px, float& py) {
    if (view.width <= 0.0f ||
        view.height <= 0.0f ||
        view.fov_y <= 0.0f ||
        view.fov_y >= kPi) {
        return false;
    }

    const Vec3 p = world - view.eye;
    const float depth = dot(p, view.basis.forward);

    if (depth < kMinDepth) {
        return false;
    }

    const float tan_half = std::tan(view.fov_y * 0.5f);
    const float aspect = view.width / view.height;

    if (tan_half <= 0.0f || aspect <= 0.0f) {
        return false;
    }

    const float ndc_x =
        dot(p, view.basis.right) / (depth * tan_half * aspect);

    const float ndc_y =
        dot(p, view.basis.up) / (depth * tan_half);

    px = (ndc_x * 0.5f + 0.5f) * view.width;
    py = (1.0f - (ndc_y * 0.5f + 0.5f)) * view.height;

    return true;
}

bool TransformEditor::active() const {
    return m_active;
}

EditMode TransformEditor::mode() const {
    return m_mode;
}

EditAxis TransformEditor::axis() const {
    return m_axis;
}

void TransformEditor::begin(EditMode mode, const Transform& current,
                            float mouse_x, float mouse_y,
                            const EditView& view) {
    if (mode == EditMode::EDIT_NONE) {
        return;
    }

    m_active = true;
    m_mode = mode;
    m_axis = EditAxis::EDIT_AXIS_FREE;

    m_start = current;
    m_start_mouse_x = mouse_x;
    m_start_mouse_y = mouse_y;
    m_start_view = view;

    m_world_per_pixel = 0.0f;
    m_total_angle = 0.0f;
    m_previous_angle = 0.0f;
    m_start_pivot_distance = 0.0f;
    m_rotation_sign = 1.0f;

    const Vec3 to_pivot = current.position - view.eye;
    const float depth = std::max(
        dot(to_pivot, view.basis.forward),
        kGrabMinDepth
    );

    if (view.height > 0.0f) {
        m_world_per_pixel =
            2.0f * depth * std::tan(view.fov_y * 0.5f) / view.height;
    }

    float pivot_x = view.width * 0.5f;
    float pivot_y = view.height * 0.5f;

    project_to_screen(
        view,
        current.position,
        pivot_x,
        pivot_y
    );

    m_pivot_x = pivot_x;
    m_pivot_y = pivot_y;

    if (mode == EditMode::EDIT_ROTATE) {
        m_rotation_axis = -view.basis.forward;

        m_previous_angle = std::atan2(
            -(mouse_y - m_pivot_y),
            mouse_x - m_pivot_x
        );
    }

    if (mode == EditMode::EDIT_SCALE) {
        m_start_pivot_distance = distance_2d(
            mouse_x, mouse_y, m_pivot_x, m_pivot_y
        );
    }
}

void TransformEditor::set_axis(EditAxis axis) {
    if (!m_active) {
        return;
    }

    // Pressing the same constrained axis again returns to free mode
    if (axis != EditAxis::EDIT_AXIS_FREE && axis == m_axis) {
        m_axis = EditAxis::EDIT_AXIS_FREE;
    } else {
        m_axis = axis;
    }

    // Rotation's axis is selected in world space
    if (m_mode == EditMode::EDIT_ROTATE) {
        m_rotation_sign = 1.0f;

        if (m_axis == EditAxis::EDIT_AXIS_FREE) {
            m_rotation_axis = -m_start_view.basis.forward;
        } else {
            m_rotation_axis = axis_vector(m_axis);

            if (dot(m_rotation_axis, -m_start_view.basis.forward) < 0.0f) {
                m_rotation_sign = -1.0f;
            }
        }
    }
}

void TransformEditor::update(float mouse_x, float mouse_y,
                             const EditView& view, Transform& out) {
    if (!m_active) {
        return;
    }

    // Every update starts from the original transform, never the
    // result of the previous frame
    out = m_start;

    const float dx = mouse_x - m_start_mouse_x;
    const float dy = mouse_y - m_start_mouse_y;

    if (m_mode == EditMode::EDIT_GRAB) {
        const Vec3 free_delta =
            m_start_view.basis.right * (dx * m_world_per_pixel)
            - m_start_view.basis.up * (dy * m_world_per_pixel);

        out.position =
            m_start.position + constrain_to_axis(free_delta, m_axis);

        return;
    }

    if (m_mode == EditMode::EDIT_ROTATE) {
        // Re-project using the current view. The rotation axis and
        // starting orientation remain anchored to begin()
        float cx = view.width * 0.5f;
        float cy = view.height * 0.5f;

        project_to_screen(view, m_start.position, cx, cy);

        const float current_angle = std::atan2(
            -(mouse_y - cy),
            mouse_x - cx
        );

        const float angle_delta = wrap_angle(
            current_angle - m_previous_angle
        );

        m_total_angle += angle_delta;
        m_previous_angle = current_angle;

        const Mat4 delta_rotation =
            mat4_rotate(m_rotation_axis, m_total_angle * m_rotation_sign);

        const Mat4 new_rotation =
            delta_rotation * m_start.rotation_matrix();

        out.rotation = euler_from_matrix(new_rotation);
        return;
    }

    if (m_mode == EditMode::EDIT_SCALE) {
        float cx = view.width * 0.5f;
        float cy = view.height * 0.5f;

        project_to_screen(view, m_start.position, cx, cy);

        const float current_distance = distance_2d(
            mouse_x, mouse_y, cx, cy
        );

        const float denominator = std::max(
            m_start_pivot_distance,
            kMinPivotDistance
        );

        const float factor = std::max(
            current_distance / denominator,
            kMinScaleFactor
        );

        switch (m_axis) {
        case EditAxis::EDIT_AXIS_X:
            out.scale.x = m_start.scale.x * factor;
            break;

        case EditAxis::EDIT_AXIS_Y:
            out.scale.y = m_start.scale.y * factor;
            break;

        case EditAxis::EDIT_AXIS_Z:
            out.scale.z = m_start.scale.z * factor;
            break;

        case EditAxis::EDIT_AXIS_FREE:
            out.scale = {
                m_start.scale.x * factor,
                m_start.scale.y * factor,
                m_start.scale.z * factor
            };
            break;
        }
    }
}

void TransformEditor::confirm() {
    m_active = false;
    m_mode = EditMode::EDIT_NONE;
    m_axis = EditAxis::EDIT_AXIS_FREE;
}

Transform TransformEditor::cancel() {
    const Transform result = m_start;

    m_active = false;
    m_mode = EditMode::EDIT_NONE;
    m_axis = EditAxis::EDIT_AXIS_FREE;

    return result;
}

} // namespace gfx
