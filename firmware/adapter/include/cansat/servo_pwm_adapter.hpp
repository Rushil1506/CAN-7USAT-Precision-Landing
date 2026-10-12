#ifndef CANSAT_SERVO_PWM_ADAPTER_HPP_
#define CANSAT_SERVO_PWM_ADAPTER_HPP_

#include <cmath>
#include <cstdint>
#include "cansat/servo_mapping.hpp"

namespace cansat {

/**
 * @brief Status codes for servo angle to PWM conversion.
 *
 * Implements strict, exception-free validation.
 */
enum class AdapterStatus {
    OK = 0,
    ERROR_INVALID_CONFIG = 1,
    ERROR_NON_FINITE_INPUT = 2,
    ERROR_ANGLE_OUT_OF_BOUNDS = 3
};

/**
 * @brief Per-channel calibration configuration.
 *
 * All values must be explicitly configured. By default, bounds are zero,
 * rendering the unconfigured struct explicitly invalid. No unverified physical
 * positions (such as 1500 us or 90 deg) are assumed or defaulted.
 */
struct ChannelPwmConfig {
    double angle_min_deg{0.0};       // Minimum allowable input angle
    double angle_max_deg{0.0};       // Maximum allowable input angle (must be > angle_min_deg)

    double pulse_at_min_us{0.0};     // Calibrated pulse width at angle_min_deg
    double pulse_at_max_us{0.0};     // Calibrated pulse width at angle_max_deg

    double clamp_min_us{0.0};        // Absolute physical hardware clamp lower bound
    double clamp_max_us{0.0};        // Absolute physical hardware clamp upper bound
};

/**
 * @brief Dual-channel configuration for left and right parafoil steering servos.
 */
struct ServoPwmConfig {
    ChannelPwmConfig left;
    ChannelPwmConfig right;
};

/**
 * @brief Output container holding calculated pulse widths.
 *
 * If status != AdapterStatus::OK, valid is false and pulse widths are 0.0.
 * Under no circumstances is an undocumented neutral pulse substituted on error.
 */
struct PwmPulseOutput {
    double left_pulse_us{0.0};
    double right_pulse_us{0.0};
    bool valid{false};
    AdapterStatus status{AdapterStatus::ERROR_INVALID_CONFIG};

    uint16_t left_us_rounded() const noexcept {
        return static_cast<uint16_t>(std::round(left_pulse_us));
    }
    uint16_t right_us_rounded() const noexcept {
        return static_cast<uint16_t>(std::round(right_pulse_us));
    }
};

/**
 * @brief Validate an individual channel configuration.
 *
 * Checks:
 *   1. All values are finite.
 *   2. angle_min_deg < angle_max_deg.
 *   3. pulse_at_min_us > 0.0 and pulse_at_max_us > 0.0.
 *   4. clamp_min_us > 0.0 and clamp_max_us > clamp_min_us.
 *   5. Calibrated pulses are within [clamp_min_us, clamp_max_us].
 */
bool validate_channel_config(const ChannelPwmConfig& cfg) noexcept;

/**
 * @brief Validate dual-channel configuration.
 */
bool validate_pwm_config(const ServoPwmConfig& cfg) noexcept;

/**
 * @brief Convert a single channel angle into microsecond pulse width.
 *
 * @param angle_deg Input angle in degrees.
 * @param cfg Channel calibration configuration.
 * @param out_pulse_us Reference receiving calculated pulse width in microseconds.
 * @return AdapterStatus::OK on success, or error code on invalid input or config.
 */
AdapterStatus convert_channel_angle_to_pwm(
    double angle_deg,
    const ChannelPwmConfig& cfg,
    double& out_pulse_us
) noexcept;

/**
 * @brief Hardware-independent adapter converting ServoCommand angles into PWM pulse widths.
 */
class ServoPwmAdapter {
public:
    /**
     * @brief Construct with unconfigured/invalid default configuration.
     */
    ServoPwmAdapter() noexcept = default;

    /**
     * @brief Construct with specified calibration configuration.
     */
    explicit ServoPwmAdapter(const ServoPwmConfig& config) noexcept;

    /**
     * @brief Check whether currently loaded configuration is valid.
     */
    bool is_valid() const noexcept { return is_valid_; }

    /**
     * @brief Update configuration. Revalidates configuration parameters.
     */
    void set_config(const ServoPwmConfig& config) noexcept;

    /**
     * @brief Get active configuration.
     */
    const ServoPwmConfig& config() const noexcept { return config_; }

    /**
     * @brief Convert ServoCommand angles to microsecond PWM pulse widths.
     *
     * @param cmd Input commanded angles for left and right channels.
     * @param out Output receiving pulse widths, validity flag, and status code.
     * @return AdapterStatus::OK if both channels convert successfully, or error code.
     */
    AdapterStatus convert(const ServoCommand& cmd, PwmPulseOutput& out) const noexcept;

    /**
     * @brief Convenience conversion returning output struct directly.
     */
    PwmPulseOutput convert(const ServoCommand& cmd) const noexcept;

private:
    ServoPwmConfig config_{};
    bool is_valid_{false};
};

} // namespace cansat

#endif // CANSAT_SERVO_PWM_ADAPTER_HPP_
