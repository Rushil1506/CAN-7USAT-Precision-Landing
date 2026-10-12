#ifndef CANSAT_SERVO_MAPPING_HPP_
#define CANSAT_SERVO_MAPPING_HPP_

#include "cansat/steering.hpp"

namespace cansat {

/**
 * @brief Servo linkage geometric calibration configuration.
 *
 * NOTE: Values such as 90°, 140°, 40° are synthetic test fixtures.
 * Actual hardware requires physical calibration of neutral angle,
 * maximum brake pull travel, and horn direction.
 */
struct ServoConfig {
    double neutral_angle{90.0};
    double max_brake_angle{140.0};
};

/**
 * @brief Commanded physical servo output angles.
 */
struct ServoCommand {
    double left_angle{90.0};
    double right_angle{90.0};
};

/**
 * @brief Convert normalized differential brake commands to physical servo angles.
 *
 * Linearly interpolates between neutral and maximum-brake angle limits:
 *   angle = neutral_angle + brake * (max_brake_angle - neutral_angle)
 *
 * Inputs are clamped to [0.0, 1.0] for physical actuator safety.
 */
ServoCommand brake_to_servo(
    const BrakeCommand& brake,
    const ServoConfig& left_config,
    const ServoConfig& right_config
) noexcept;

} // namespace cansat

#endif // CANSAT_SERVO_MAPPING_HPP_
