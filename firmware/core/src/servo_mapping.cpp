#include "cansat/servo_mapping.hpp"

#include <algorithm>
#include <cmath>

namespace cansat {

namespace {

inline double clamp_brake(double val) noexcept {
    if (!std::isfinite(val) || val < 0.0) return 0.0;
    if (val > 1.0) return 1.0;
    return val;
}

} // namespace

ServoCommand brake_to_servo(
    const BrakeCommand& brake,
    const ServoConfig& left_config,
    const ServoConfig& right_config
) noexcept {
    const double b_left = clamp_brake(brake.left);
    const double b_right = clamp_brake(brake.right);

    const double left_angle = left_config.neutral_angle
        + b_left * (left_config.max_brake_angle - left_config.neutral_angle);

    const double right_angle = right_config.neutral_angle
        + b_right * (right_config.max_brake_angle - right_config.neutral_angle);

    return ServoCommand{left_angle, right_angle};
}

} // namespace cansat
