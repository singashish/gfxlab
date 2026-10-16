#pragma once

#include "core/orbit_camera.h"
#include "core/transform.h"

namespace gfx {

enum class EditMode {
    EDIT_NONE,
    EDIT_GRAB,
    EDIT_ROTATE,
    EDIT_SCALE
};

enum class EditAxis {
    EDIT_AXIS_FREE,
    EDIT_AXIS_X,
    EDIT_AXIS_Y,
    EDIT_AXIS_Z
};

struct EditView {
    Vec3 eye;
    CameraBasis basis;
    float fov_y = 0.0f;
    float width = 0.0f;
    float height = 0.0f;
};

bool project_to_screen(const EditView& view, const Vec3& world,
                       float& px, float& py);

class TransformEditor {
public:
    bool active() const;
    EditMode mode() const;
    EditAxis axis() const;

    void begin(EditMode mode, const Transform& current,
               float mouse_x, float mouse_y, const EditView& view);

    void set_axis(EditAxis axis);

    void update(float mouse_x, float mouse_y,
                const EditView& view, Transform& out);

    void confirm();
    Transform cancel();

private:
    bool m_active = false;
    EditMode m_mode = EditMode::EDIT_NONE;
    EditAxis m_axis = EditAxis::EDIT_AXIS_FREE;

    Transform m_start{};

    float m_start_mouse_x = 0.0f;
    float m_start_mouse_y = 0.0f;

    // Snapshot of the camera at the start of the operation
    EditView m_start_view{};

    // Grab sensitivity at the starting object's depth
    float m_world_per_pixel = 0.0f;

    // Rotation state
    float m_pivot_x = 0.0f;
    float m_pivot_y = 0.0f;
    float m_previous_angle = 0.0f;
    float m_total_angle = 0.0f;
    float m_rotation_sign = 1.0f;
    Vec3 m_rotation_axis{0.0f, 0.0f, 1.0f};

    // Scale state
    float m_start_pivot_distance = 0.0f;
};

} // namespace gfx
