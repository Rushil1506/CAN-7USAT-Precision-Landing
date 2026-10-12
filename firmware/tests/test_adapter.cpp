#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>

#include "cansat/servo_pwm_adapter.hpp"

namespace {

int g_passed = 0;
int g_failed = 0;

void record_test(bool condition, const std::string& name) {
    if (condition) {
        ++g_passed;
        std::cout << "  [PASS] " << name << "\n";
    } else {
        ++g_failed;
        std::cerr << "  [FAIL] " << name << "\n";
    }
}

bool approx_equal(double a, double b, double tol = 1e-7) {
    if (std::isnan(a) || std::isnan(b)) return false;
    return std::abs(a - b) <= tol;
}

} // namespace

void test_default_unconfigured_adapter() {
    std::cout << "\n=== 1. Default / Unconfigured Adapter Rejection Tests ===\n";

    cansat::ServoPwmAdapter adapter;
    record_test(!adapter.is_valid(), "Default adapter is_valid() == false");

    cansat::ServoCommand cmd{90.0, 90.0};
    cansat::PwmPulseOutput out = adapter.convert(cmd);

    record_test(!out.valid, "Default adapter convert() produces out.valid == false");
    record_test(out.status == cansat::AdapterStatus::ERROR_INVALID_CONFIG,
                "Default adapter convert() returns ERROR_INVALID_CONFIG");
    record_test(out.left_pulse_us == 0.0, "Default adapter out.left_pulse_us == 0.0 (no neutral substituted)");
    record_test(out.right_pulse_us == 0.0, "Default adapter out.right_pulse_us == 0.0 (no neutral substituted)");
}

void test_configuration_validation() {
    std::cout << "\n=== 2. Configuration Validation Tests ===\n";

    // Valid baseline channel config
    cansat::ChannelPwmConfig valid_cfg{
        90.0, 140.0,      // angle min, max
        1500.0, 2000.0,   // pulse at min, max
        1000.0, 2100.0    // clamp min, max
    };
    record_test(cansat::validate_channel_config(valid_cfg), "Valid channel configuration passes");

    // Inverted angle bounds
    cansat::ChannelPwmConfig inv_angles = valid_cfg;
    inv_angles.angle_min_deg = 140.0;
    inv_angles.angle_max_deg = 90.0;
    record_test(!cansat::validate_channel_config(inv_angles), "Rejects angle_min >= angle_max");

    // Equal angle bounds (zero span)
    cansat::ChannelPwmConfig zero_span = valid_cfg;
    zero_span.angle_max_deg = zero_span.angle_min_deg;
    record_test(!cansat::validate_channel_config(zero_span), "Rejects angle_min == angle_max");

    // Non-finite angle bounds
    cansat::ChannelPwmConfig nan_angle = valid_cfg;
    nan_angle.angle_min_deg = std::nan("");
    record_test(!cansat::validate_channel_config(nan_angle), "Rejects NaN angle bound");

    cansat::ChannelPwmConfig inf_angle = valid_cfg;
    inf_angle.angle_max_deg = INFINITY;
    record_test(!cansat::validate_channel_config(inf_angle), "Rejects Inf angle bound");

    // Non-positive pulse widths
    cansat::ChannelPwmConfig zero_pulse = valid_cfg;
    zero_pulse.pulse_at_min_us = 0.0;
    record_test(!cansat::validate_channel_config(zero_pulse), "Rejects pulse_at_min <= 0.0");

    cansat::ChannelPwmConfig neg_pulse = valid_cfg;
    neg_pulse.pulse_at_max_us = -100.0;
    record_test(!cansat::validate_channel_config(neg_pulse), "Rejects negative pulse width");

    // Inverted clamp limits
    cansat::ChannelPwmConfig inv_clamp = valid_cfg;
    inv_clamp.clamp_min_us = 2200.0;
    inv_clamp.clamp_max_us = 1000.0;
    record_test(!cansat::validate_channel_config(inv_clamp), "Rejects clamp_min >= clamp_max");

    // Calibrated pulse outside clamp bounds
    cansat::ChannelPwmConfig pulse_exceeds_clamp = valid_cfg;
    pulse_exceeds_clamp.pulse_at_max_us = 2200.0; // clamp is 2100.0
    record_test(!cansat::validate_channel_config(pulse_exceeds_clamp),
                "Rejects pulse exceeding clamp_max");

    cansat::ChannelPwmConfig pulse_below_clamp = valid_cfg;
    pulse_below_clamp.pulse_at_min_us = 800.0; // clamp is 1000.0
    record_test(!cansat::validate_channel_config(pulse_below_clamp),
                "Rejects pulse below clamp_min");
}

void test_channel_conversion_direct_and_reversed() {
    std::cout << "\n=== 3. Channel Conversion & Direction Tests ===\n";

    // Direct channel: 90 deg -> 1500 us, 140 deg -> 2000 us (span 50 deg, span 500 us -> slope 10 us/deg)
    cansat::ChannelPwmConfig direct_cfg{
        90.0, 140.0,
        1500.0, 2000.0,
        1000.0, 2200.0
    };

    double out_us = 0.0;
    cansat::AdapterStatus s1 = cansat::convert_channel_angle_to_pwm(90.0, direct_cfg, out_us);
    record_test(s1 == cansat::AdapterStatus::OK && approx_equal(out_us, 1500.0),
                "Direct channel: min angle 90 deg -> 1500.0 us");

    cansat::AdapterStatus s2 = cansat::convert_channel_angle_to_pwm(115.0, direct_cfg, out_us);
    record_test(s2 == cansat::AdapterStatus::OK && approx_equal(out_us, 1750.0),
                "Direct channel: mid angle 115 deg -> 1750.0 us");

    cansat::AdapterStatus s3 = cansat::convert_channel_angle_to_pwm(140.0, direct_cfg, out_us);
    record_test(s3 == cansat::AdapterStatus::OK && approx_equal(out_us, 2000.0),
                "Direct channel: max angle 140 deg -> 2000.0 us");

    // Reversed channel: 40 deg -> 1900 us, 90 deg -> 1400 us (span 50 deg, span -500 us -> slope -10 us/deg)
    cansat::ChannelPwmConfig reversed_cfg{
        40.0, 90.0,
        1900.0, 1400.0,
        1000.0, 2200.0
    };
    record_test(cansat::validate_channel_config(reversed_cfg), "Reversed channel configuration is valid");

    cansat::AdapterStatus r1 = cansat::convert_channel_angle_to_pwm(40.0, reversed_cfg, out_us);
    record_test(r1 == cansat::AdapterStatus::OK && approx_equal(out_us, 1900.0),
                "Reversed channel: min angle 40 deg -> 1900.0 us");

    cansat::AdapterStatus r2 = cansat::convert_channel_angle_to_pwm(65.0, reversed_cfg, out_us);
    record_test(r2 == cansat::AdapterStatus::OK && approx_equal(out_us, 1650.0),
                "Reversed channel: mid angle 65 deg -> 1650.0 us");

    cansat::AdapterStatus r3 = cansat::convert_channel_angle_to_pwm(90.0, reversed_cfg, out_us);
    record_test(r3 == cansat::AdapterStatus::OK && approx_equal(out_us, 1400.0),
                "Reversed channel: max angle 90 deg -> 1400.0 us");
}

void test_independent_dual_channel_adapter() {
    std::cout << "\n=== 4. Dual-Channel Independent Conversion Tests ===\n";

    cansat::ServoPwmConfig dual_cfg{
        // Left: 90 to 140 deg -> 1500 to 2000 us
        cansat::ChannelPwmConfig{90.0, 140.0, 1500.0, 2000.0, 1000.0, 2200.0},
        // Right: 40 to 90 deg -> 1900 to 1400 us (reversed)
        cansat::ChannelPwmConfig{40.0, 90.0, 1900.0, 1400.0, 1000.0, 2200.0}
    };

    cansat::ServoPwmAdapter adapter(dual_cfg);
    record_test(adapter.is_valid(), "Configured dual-channel adapter is valid");

    // Command: left = 115 deg, right = 65 deg
    cansat::ServoCommand cmd{115.0, 65.0};
    cansat::PwmPulseOutput out = adapter.convert(cmd);

    record_test(out.valid && out.status == cansat::AdapterStatus::OK, "Dual conversion returns status OK");
    record_test(approx_equal(out.left_pulse_us, 1750.0), "Left channel converts to 1750.0 us");
    record_test(approx_equal(out.right_pulse_us, 1650.0), "Right channel converts to 1650.0 us");
    record_test(out.left_us_rounded() == 1750, "Left integer rounding is 1750 us");
    record_test(out.right_us_rounded() == 1650, "Right integer rounding is 1650 us");
}

void test_out_of_bounds_and_invalid_inputs() {
    std::cout << "\n=== 5. Input Rejection & Non-Finite Tests ===\n";

    cansat::ServoPwmConfig dual_cfg{
        cansat::ChannelPwmConfig{90.0, 140.0, 1500.0, 2000.0, 1000.0, 2200.0},
        cansat::ChannelPwmConfig{40.0, 90.0, 1900.0, 1400.0, 1000.0, 2200.0}
    };
    cansat::ServoPwmAdapter adapter(dual_cfg);

    // Left angle below min (89.9 deg < 90.0 deg)
    cansat::ServoCommand cmd_below{89.9, 65.0};
    cansat::PwmPulseOutput out_below = adapter.convert(cmd_below);
    record_test(!out_below.valid, "Input below min angle: out.valid == false");
    record_test(out_below.status == cansat::AdapterStatus::ERROR_ANGLE_OUT_OF_BOUNDS,
                "Input below min angle: status == ERROR_ANGLE_OUT_OF_BOUNDS");
    record_test(out_below.left_pulse_us == 0.0 && out_below.right_pulse_us == 0.0,
                "Out-of-bounds input outputs 0.0 us (no neutral substituted)");

    // Left angle above max (140.1 deg > 140.0 deg)
    cansat::ServoCommand cmd_above{140.1, 65.0};
    cansat::PwmPulseOutput out_above = adapter.convert(cmd_above);
    record_test(!out_above.valid && out_above.status == cansat::AdapterStatus::ERROR_ANGLE_OUT_OF_BOUNDS,
                "Input above max angle: status == ERROR_ANGLE_OUT_OF_BOUNDS");

    // Right angle out of bounds (90.1 deg > 90.0 deg)
    cansat::ServoCommand cmd_right_oob{115.0, 90.1};
    cansat::PwmPulseOutput out_r_oob = adapter.convert(cmd_right_oob);
    record_test(!out_r_oob.valid && out_r_oob.status == cansat::AdapterStatus::ERROR_ANGLE_OUT_OF_BOUNDS,
                "Right channel out of bounds rejected explicitly");

    // NaN on left channel
    cansat::ServoCommand cmd_nan_left{std::nan(""), 65.0};
    cansat::PwmPulseOutput out_nan_l = adapter.convert(cmd_nan_left);
    record_test(!out_nan_l.valid && out_nan_l.status == cansat::AdapterStatus::ERROR_NON_FINITE_INPUT,
                "NaN on left channel rejected with ERROR_NON_FINITE_INPUT");
    record_test(out_nan_l.left_pulse_us == 0.0, "NaN on left channel produces 0.0 us");

    // +Inf on right channel
    cansat::ServoCommand cmd_inf_right{115.0, INFINITY};
    cansat::PwmPulseOutput out_inf_r = adapter.convert(cmd_inf_right);
    record_test(!out_inf_r.valid && out_inf_r.status == cansat::AdapterStatus::ERROR_NON_FINITE_INPUT,
                "+Inf on right channel rejected with ERROR_NON_FINITE_INPUT");

    // -Inf on left channel
    cansat::ServoCommand cmd_ninf{-INFINITY, 65.0};
    cansat::PwmPulseOutput out_ninf = adapter.convert(cmd_ninf);
    record_test(!out_ninf.valid && out_ninf.status == cansat::AdapterStatus::ERROR_NON_FINITE_INPUT,
                "-Inf on left channel rejected with ERROR_NON_FINITE_INPUT");
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "CAN-7USAT Precision Landing Algorithm — PWM Adapter Tests\n";
    std::cout << "Phase 6.2B Host-Compiled Verification Suite\n";
    std::cout << "======================================================\n";

    test_default_unconfigured_adapter();
    test_configuration_validation();
    test_channel_conversion_direct_and_reversed();
    test_independent_dual_channel_adapter();
    test_out_of_bounds_and_invalid_inputs();

    std::cout << "\n------------------------------------------------------\n";
    std::cout << "Adapter Test Summary: " << g_passed << " Passed, " << g_failed << " Failed\n";
    std::cout << "------------------------------------------------------\n";

    return (g_failed == 0) ? 0 : 1;
}
