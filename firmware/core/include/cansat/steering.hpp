#ifndef CANSAT_STEERING_HPP_
#define CANSAT_STEERING_HPP_

namespace cansat {

/**
 * @brief Normalized differential brake commands for steerable parafoil.
 *
 * Range:
 *   0.0 = no brake (neutral)
 *   1.0 = maximum brake
 */
struct BrakeCommand {
    double left{0.0};
    double right{0.0};
};

/**
 * @brief Convert normalized steering command into differential brake commands.
 *
 * Convention:
 *   steering = -1.0 -> maximum LEFT brake (left = 1.0, right = 0.0)
 *   steering =  0.0 -> both brakes neutral (left = 0.0, right = 0.0)
 *   steering = +1.0 -> maximum RIGHT brake (left = 0.0, right = 1.0)
 *
 * @param steering Normalized steering command, clamped to [-1.0, +1.0].
 * @return BrakeCommand with values strictly in [0.0, 1.0].
 */
BrakeCommand steering_to_brakes(double steering) noexcept;

} // namespace cansat

#endif // CANSAT_STEERING_HPP_
