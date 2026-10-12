#include "cansat/pid.hpp"

#include <algorithm>
#include <cmath>

namespace cansat {

namespace {

inline double clamp_val(double val, double min_val, double max_val) noexcept {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

} // namespace

PidController::PidController(const PidConfig& config) noexcept
    : config_(config),
      integral_(0.0),
      previous_error_(0.0),
      last_output_(0.0),
      initialized_(false),
      last_status_(PidStatus::OK) {}

void PidController::reset() noexcept {
    integral_ = 0.0;
    previous_error_ = 0.0;
    last_output_ = 0.0;
    initialized_ = false;
    last_status_ = PidStatus::OK;
}

void PidController::set_config(const PidConfig& config) noexcept {
    config_ = config;
    // Re-clamp integral accumulator to new configuration bounds
    integral_ = clamp_val(integral_, config_.integral_min, config_.integral_max);
}

PidStatus PidController::update(double error, double dt, double& out_steering) noexcept {
    // Defensive input validation without exceptions
    if (!std::isfinite(dt) || dt <= 0.0) {
        last_status_ = PidStatus::ERROR_INVALID_DT;
        out_steering = last_output_;
        return last_status_;
    }

    if (!std::isfinite(error)) {
        last_status_ = PidStatus::ERROR_INVALID_ERROR;
        out_steering = last_output_;
        return last_status_;
    }

    last_status_ = PidStatus::OK;

    // 1. Proportional term
    const double proportional = config_.kp * error;

    // 2. Integral term with anti-windup clamping
    integral_ += error * dt;
    integral_ = clamp_val(integral_, config_.integral_min, config_.integral_max);
    const double integral_term = config_.ki * integral_;

    // 3. Derivative term with first-step derivative kick suppression
    double derivative = 0.0;
    if (!initialized_) {
        derivative = 0.0;
        initialized_ = true;
    } else {
        derivative = (error - previous_error_) / dt;
    }
    const double derivative_term = config_.kd * derivative;

    // 4. Raw PID output summation
    const double raw_output = proportional + integral_term + derivative_term;

    // 5. Output command saturation
    const double saturated_output = clamp_val(raw_output, config_.output_min, config_.output_max);

    // 6. Update internal history state
    previous_error_ = error;
    last_output_ = saturated_output;
    out_steering = saturated_output;

    return PidStatus::OK;
}

double PidController::update(double error, double dt) noexcept {
    double steering = 0.0;
    update(error, dt, steering);
    return steering;
}

} // namespace cansat
