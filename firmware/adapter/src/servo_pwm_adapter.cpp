#include "cansat/servo_pwm_adapter.hpp"

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

bool validate_channel_config(const ChannelPwmConfig& cfg) noexcept {
    // 1. All values must be finite
    if (!std::isfinite(cfg.angle_min_deg) || !std::isfinite(cfg.angle_max_deg) ||
        !std::isfinite(cfg.pulse_at_min_us) || !std::isfinite(cfg.pulse_at_max_us) ||
        !std::isfinite(cfg.clamp_min_us) || !std::isfinite(cfg.clamp_max_us)) {
        return false;
    }

    // 2. Ordered angle bounds with non-zero span
    if (cfg.angle_min_deg >= cfg.angle_max_deg) {
        return false;
    }

    // 3. Positive pulse-width values
    if (cfg.pulse_at_min_us <= 0.0 || cfg.pulse_at_max_us <= 0.0 ||
        cfg.clamp_min_us <= 0.0 || cfg.clamp_max_us <= 0.0) {
        return false;
    }

    // 4. Ordered clamp limits
    if (cfg.clamp_min_us >= cfg.clamp_max_us) {
        return false;
    }

    // 5. Calibrated pulse endpoints must be within absolute clamp bounds
    const double min_pulse = std::min(cfg.pulse_at_min_us, cfg.pulse_at_max_us);
    const double max_pulse = std::max(cfg.pulse_at_min_us, cfg.pulse_at_max_us);
    if (min_pulse < cfg.clamp_min_us || max_pulse > cfg.clamp_max_us) {
        return false;
    }

    return true;
}

bool validate_pwm_config(const ServoPwmConfig& cfg) noexcept {
    return validate_channel_config(cfg.left) && validate_channel_config(cfg.right);
}

AdapterStatus convert_channel_angle_to_pwm(
    double angle_deg,
    const ChannelPwmConfig& cfg,
    double& out_pulse_us
) noexcept {
    out_pulse_us = 0.0;

    // Validate configuration
    if (!validate_channel_config(cfg)) {
        return AdapterStatus::ERROR_INVALID_CONFIG;
    }

    // Validate input finiteness
    if (!std::isfinite(angle_deg)) {
        return AdapterStatus::ERROR_NON_FINITE_INPUT;
    }

    // Validate angle within allowable bounds
    if (angle_deg < cfg.angle_min_deg || angle_deg > cfg.angle_max_deg) {
        return AdapterStatus::ERROR_ANGLE_OUT_OF_BOUNDS;
    }

    // Linear interpolation from angle domain to pulse width domain
    const double span_angle = cfg.angle_max_deg - cfg.angle_min_deg;
    const double t = (angle_deg - cfg.angle_min_deg) / span_angle;
    const double raw_pulse = cfg.pulse_at_min_us + t * (cfg.pulse_at_max_us - cfg.pulse_at_min_us);

    // Apply absolute physical hardware safety clamp
    const double clamped_pulse = clamp_val(raw_pulse, cfg.clamp_min_us, cfg.clamp_max_us);

    out_pulse_us = clamped_pulse;
    return AdapterStatus::OK;
}

ServoPwmAdapter::ServoPwmAdapter(const ServoPwmConfig& config) noexcept
    : config_(config),
      is_valid_(validate_pwm_config(config)) {}

void ServoPwmAdapter::set_config(const ServoPwmConfig& config) noexcept {
    config_ = config;
    is_valid_ = validate_pwm_config(config);
}

AdapterStatus ServoPwmAdapter::convert(const ServoCommand& cmd, PwmPulseOutput& out) const noexcept {
    out.left_pulse_us = 0.0;
    out.right_pulse_us = 0.0;
    out.valid = false;

    if (!is_valid_) {
        out.status = AdapterStatus::ERROR_INVALID_CONFIG;
        return out.status;
    }

    // Convert left channel
    double left_us = 0.0;
    const AdapterStatus left_stat = convert_channel_angle_to_pwm(cmd.left_angle, config_.left, left_us);
    if (left_stat != AdapterStatus::OK) {
        out.status = left_stat;
        return left_stat;
    }

    // Convert right channel
    double right_us = 0.0;
    const AdapterStatus right_stat = convert_channel_angle_to_pwm(cmd.right_angle, config_.right, right_us);
    if (right_stat != AdapterStatus::OK) {
        out.status = right_stat;
        return right_stat;
    }

    out.left_pulse_us = left_us;
    out.right_pulse_us = right_us;
    out.valid = true;
    out.status = AdapterStatus::OK;

    return AdapterStatus::OK;
}

PwmPulseOutput ServoPwmAdapter::convert(const ServoCommand& cmd) const noexcept {
    PwmPulseOutput out{};
    convert(cmd, out);
    return out;
}

} // namespace cansat
