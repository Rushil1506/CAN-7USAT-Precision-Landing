#ifndef CANSAT_PID_HPP_
#define CANSAT_PID_HPP_

#include <cmath>

namespace cansat {

/**
 * @brief Status codes for PID update operations without exceptions.
 */
enum class PidStatus {
    OK = 0,
    ERROR_INVALID_DT = 1,
    ERROR_INVALID_ERROR = 2
};

/**
 * @brief Configuration parameters for the PID controller.
 *
 * Defaults match frozen guidance baseline:
 *   Kp = 10.0, Ki = 0.1, Kd = 0.5 (tuned for radian input)
 *   Output limits = [-1.0, +1.0] (normalized steering command)
 *   Integral anti-windup clamping = [-10.0, +10.0]
 */
struct PidConfig {
    double kp{10.0};
    double ki{0.1};
    double kd{0.5};

    double output_min{-1.0};
    double output_max{1.0};

    double integral_min{-10.0};
    double integral_max{10.0};
};

/**
 * @brief Discrete-time PID controller with output saturation, anti-windup clamping,
 *        derivative kick suppression, and exception-free error handling.
 *
 * Implements 1:1 mathematical parity with Python `control/pid.py`.
 */
class PidController {
public:
    explicit PidController(const PidConfig& config = PidConfig()) noexcept;

    /**
     * @brief Reset internal controller state (integral, previous error, initialization flag).
     */
    void reset() noexcept;

    /**
     * @brief Update PID controller state and compute saturated output command.
     *
     * @param error Current error input (typically heading error in radians).
     * @param dt Time delta since last update in seconds. Must be > 0.0 and finite.
     * @param out_steering Output reference receiving the saturated steering command [-1, +1].
     * @return PidStatus::OK if successful, or error code on invalid input.
     *         On invalid input, internal state is preserved without corruption.
     */
    PidStatus update(double error, double dt, double& out_steering) noexcept;

    /**
     * @brief Convenient single-value update method.
     *
     * If dt is non-positive or non-finite, returns last valid output without updating state.
     */
    double update(double error, double dt) noexcept;

    // Const inspection accessors
    const PidConfig& config() const noexcept { return config_; }
    double integral() const noexcept { return integral_; }
    double previous_error() const noexcept { return previous_error_; }
    double last_output() const noexcept { return last_output_; }
    bool is_initialized() const noexcept { return initialized_; }
    PidStatus last_status() const noexcept { return last_status_; }

    // Configuration setter
    void set_config(const PidConfig& config) noexcept;

private:
    PidConfig config_;
    double integral_{0.0};
    double previous_error_{0.0};
    double last_output_{0.0};
    bool initialized_{false};
    PidStatus last_status_{PidStatus::OK};
};

} // namespace cansat

#endif // CANSAT_PID_HPP_
