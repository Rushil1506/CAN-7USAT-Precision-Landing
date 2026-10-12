#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

#include "cansat/angles.hpp"
#include "cansat/pid.hpp"
#include "cansat/steering.hpp"
#include "cansat/servo_mapping.hpp"

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

void test_angle_math() {
    std::cout << "\n=== 1. Angle & Heading-Error Tests ===\n";

    // 1.1 wrap_360
    record_test(approx_equal(cansat::wrap_360(0.0), 0.0), "wrap_360(0) == 0");
    record_test(approx_equal(cansat::wrap_360(360.0), 0.0), "wrap_360(360) == 0");
    record_test(approx_equal(cansat::wrap_360(720.0), 0.0), "wrap_360(720) == 0");
    record_test(approx_equal(cansat::wrap_360(90.0), 90.0), "wrap_360(90) == 90");
    record_test(approx_equal(cansat::wrap_360(-90.0), 270.0), "wrap_360(-90) == 270");
    record_test(approx_equal(cansat::wrap_360(-360.0), 0.0), "wrap_360(-360) == 0");
    record_test(approx_equal(cansat::wrap_360(-450.0), 270.0), "wrap_360(-450) == 270");

    // 1.2 Deterministic heading-error test cases from Phase 6.0 report
    // HE-1: 0 -> 10 => +10
    record_test(approx_equal(cansat::heading_error(0.0, 10.0), 10.0), "HE-1: heading=0, bearing=10 -> +10");
    // HE-2: 350 -> 10 => +20
    record_test(approx_equal(cansat::heading_error(350.0, 10.0), 20.0), "HE-2: heading=350, bearing=10 -> +20");
    // HE-3: 10 -> 350 => -20
    record_test(approx_equal(cansat::heading_error(10.0, 350.0), -20.0), "HE-3: heading=10, bearing=350 -> -20");
    // HE-4: 180 -> 0 => -180
    record_test(approx_equal(cansat::heading_error(180.0, 0.0), -180.0), "HE-4: heading=180, bearing=0 -> -180");
    // HE-5: 0 -> 180 => -180 (branch-cut tie-break)
    record_test(approx_equal(cansat::heading_error(0.0, 180.0), -180.0), "HE-5: heading=0, bearing=180 -> -180");
    // HE-6: 270 -> 90 => -180
    record_test(approx_equal(cansat::heading_error(270.0, 90.0), -180.0), "HE-6: heading=270, bearing=90 -> -180");
    // HE-7: 45 -> 45 => 0
    record_test(approx_equal(cansat::heading_error(45.0, 45.0), 0.0), "HE-7: heading=45, bearing=45 -> 0");

    // 1.3 Boundary cases near ±180°
    record_test(approx_equal(cansat::heading_error(0.0, 179.99), 179.99), "heading=0, bearing=179.99 -> +179.99");
    record_test(approx_equal(cansat::heading_error(0.0, 180.01), -179.99), "heading=0, bearing=180.01 -> -179.99");
    record_test(approx_equal(cansat::heading_error(179.99, 0.0), -179.99), "heading=179.99, bearing=0 -> -179.99");

    // 1.4 Bearing calculations
    record_test(approx_equal(cansat::bearing(0.0, 0.0, 0.0, 100.0), 0.0), "bearing to North (0, 100) -> 0 deg");
    record_test(approx_equal(cansat::bearing(0.0, 0.0, 100.0, 0.0), 90.0), "bearing to East (100, 0) -> 90 deg");
    record_test(approx_equal(cansat::bearing(0.0, 0.0, 0.0, -100.0), 180.0), "bearing to South (0, -100) -> 180 deg");
    record_test(cansat::bearing(0.0, 0.0, -100.0, 0.0) == 270.0, "bearing to West (-100, 0) -> 270 deg");
    record_test(approx_equal(cansat::bearing(0.0, 0.0, 100.0, 100.0), 45.0), "bearing to NE (100, 100) -> 45 deg");
}

void test_pid_controller() {
    std::cout << "\n=== 2. PID Controller Parity & Edge Case Tests ===\n";

    cansat::PidConfig cfg;
    cfg.kp = 10.0;
    cfg.ki = 0.1;
    cfg.kd = 0.5;
    cfg.output_min = -1.0;
    cfg.output_max = 1.0;
    cfg.integral_min = -10.0;
    cfg.integral_max = 10.0;

    cansat::PidController pid(cfg);

    // 2.1 First-step derivative suppression & initial values
    record_test(!pid.is_initialized(), "PID initially uninitialized");
    record_test(pid.integral() == 0.0, "PID initial integral == 0");
    record_test(pid.previous_error() == 0.0, "PID initial previous_error == 0");

    // 2.2 Replicate Phase 6.0 5-step deterministic response sequence
    // Inputs: +1 deg, +2 deg, +2 deg, -1 deg, 0 deg with dt = 0.1 s
    const double e1 = cansat::deg_to_rad(1.0);
    const double out1 = pid.update(e1, 0.1);
    // Step 1: err_rad = 0.0174532925, P = 0.17453293, I = 0.00017453, D = 0.0 -> out = 0.17470746
    record_test(approx_equal(out1, 0.17470746, 1e-6), "Step 1: out == 0.17470746 (derivative suppressed)");
    record_test(pid.is_initialized(), "Step 1: controller marked initialized");
    record_test(approx_equal(pid.integral(), 0.00174533, 1e-6), "Step 1: integral accumulator == 0.00174533");

    const double e2 = cansat::deg_to_rad(2.0);
    const double out2 = pid.update(e2, 0.1);
    // Step 2: out == 0.43685591, integ == 0.00523599
    record_test(approx_equal(out2, 0.43685591, 1e-6), "Step 2: out == 0.43685591 (derivative active)");
    record_test(approx_equal(pid.integral(), 0.00523599, 1e-6), "Step 2: integral accumulator == 0.00523599");

    const double e3 = cansat::deg_to_rad(2.0);
    const double out3 = pid.update(e3, 0.1);
    // Step 3: out == 0.34993852, integ == 0.00872665
    record_test(approx_equal(out3, 0.34993852, 1e-6), "Step 3: out == 0.34993852");
    record_test(approx_equal(pid.integral(), 0.00872665, 1e-6), "Step 3: integral accumulator == 0.00872665");

    const double e4 = cansat::deg_to_rad(-1.0);
    const double out4 = pid.update(e4, 0.1);
    // Step 4: out == -0.43563418, integ == 0.00698132
    record_test(approx_equal(out4, -0.43563418, 1e-6), "Step 4: out == -0.43563418");
    record_test(approx_equal(pid.integral(), 0.00698132, 1e-6), "Step 4: integral accumulator == 0.00698132");

    const double e5 = cansat::deg_to_rad(0.0);
    const double out5 = pid.update(e5, 0.1);
    // Step 5: out == 0.08796459, integ == 0.00698132
    record_test(approx_equal(out5, 0.08796459, 1e-6), "Step 5: out == 0.08796459");
    record_test(approx_equal(pid.integral(), 0.00698132, 1e-6), "Step 5: integral accumulator == 0.00698132");

    // 2.3 Reset behavior
    pid.reset();
    record_test(!pid.is_initialized(), "Post-reset: is_initialized == false");
    record_test(pid.integral() == 0.0, "Post-reset: integral == 0.0");
    record_test(pid.previous_error() == 0.0, "Post-reset: previous_error == 0.0");
    record_test(pid.last_output() == 0.0, "Post-reset: last_output == 0.0");

    // Step immediately after reset must have derivative suppressed again
    const double out_post_reset = pid.update(cansat::deg_to_rad(1.0), 0.1);
    record_test(approx_equal(out_post_reset, 0.17470746, 1e-6), "Post-reset Step 1: matches initial Step 1 exactly");

    // 2.4 Output Saturation clamping
    pid.reset();
    const double out_sat_pos = pid.update(cansat::deg_to_rad(90.0), 0.1); // large error
    record_test(approx_equal(out_sat_pos, 1.0), "Positive saturation clamp == +1.0");

    pid.reset();
    const double out_sat_neg = pid.update(cansat::deg_to_rad(-90.0), 0.1); // large negative error
    record_test(approx_equal(out_sat_neg, -1.0), "Negative saturation clamp == -1.0");

    // 2.5 Integral Anti-Windup Clamping
    pid.reset();
    for (int i = 0; i < 200; ++i) {
        pid.update(1.0, 0.1); // +1.0 rad * 0.1s = 0.1 per step
    }
    record_test(approx_equal(pid.integral(), 10.0), "Integral clamped strictly to integral_max (+10.0)");

    pid.reset();
    for (int i = 0; i < 200; ++i) {
        pid.update(-1.0, 0.1);
    }
    record_test(approx_equal(pid.integral(), -10.0), "Integral clamped strictly to integral_min (-10.0)");

    // 2.6 Invalid timestep handling without exceptions
    pid.reset();
    double out_valid = pid.update(0.1, 0.1);
    double prev_integ = pid.integral();
    double prev_prev_err = pid.previous_error();

    // Pass dt = 0.0
    double out_bad_dt = 999.0;
    cansat::PidStatus stat_zero = pid.update(0.5, 0.0, out_bad_dt);
    record_test(stat_zero == cansat::PidStatus::ERROR_INVALID_DT, "dt=0.0 returns ERROR_INVALID_DT");
    record_test(out_bad_dt == out_valid, "dt=0.0 preserves last valid output");
    record_test(pid.integral() == prev_integ, "dt=0.0 does not corrupt integral accumulator");
    record_test(pid.previous_error() == prev_prev_err, "dt=0.0 does not corrupt previous error");

    // Pass dt < 0.0
    cansat::PidStatus stat_neg = pid.update(0.5, -0.05, out_bad_dt);
    record_test(stat_neg == cansat::PidStatus::ERROR_INVALID_DT, "dt < 0 returns ERROR_INVALID_DT");

    // Pass NaN dt
    cansat::PidStatus stat_nan = pid.update(0.5, std::nan(""), out_bad_dt);
    record_test(stat_nan == cansat::PidStatus::ERROR_INVALID_DT, "dt=NaN returns ERROR_INVALID_DT");

    // Pass NaN error
    cansat::PidStatus stat_nan_err = pid.update(std::nan(""), 0.1, out_bad_dt);
    record_test(stat_nan_err == cansat::PidStatus::ERROR_INVALID_ERROR, "error=NaN returns ERROR_INVALID_ERROR");
}

void test_steering_and_servo_mapping() {
    std::cout << "\n=== 3. Steering-to-Brake & Servo Mapping Tests ===\n";

    // 3.1 Steering to Brake Mapping
    // steer = -1.0 -> left = 1.0, right = 0.0
    cansat::BrakeCommand b_max_left = cansat::steering_to_brakes(-1.0);
    record_test(approx_equal(b_max_left.left, 1.0) && approx_equal(b_max_left.right, 0.0), "steer=-1.0 -> left=1, right=0");

    // steer = +1.0 -> left = 0.0, right = 1.0
    cansat::BrakeCommand b_max_right = cansat::steering_to_brakes(1.0);
    record_test(approx_equal(b_max_right.left, 0.0) && approx_equal(b_max_right.right, 1.0), "steer=+1.0 -> left=0, right=1");

    // steer = 0.0 -> left = 0.0, right = 0.0
    cansat::BrakeCommand b_neutral = cansat::steering_to_brakes(0.0);
    record_test(approx_equal(b_neutral.left, 0.0) && approx_equal(b_neutral.right, 0.0), "steer=0.0 -> left=0, right=0");

    // Partial values
    cansat::BrakeCommand b_part_l = cansat::steering_to_brakes(-0.4);
    record_test(approx_equal(b_part_l.left, 0.4) && approx_equal(b_part_l.right, 0.0), "steer=-0.4 -> left=0.4, right=0");

    cansat::BrakeCommand b_part_r = cansat::steering_to_brakes(0.65);
    record_test(approx_equal(b_part_r.left, 0.0) && approx_equal(b_part_r.right, 0.65), "steer=+0.65 -> left=0, right=0.65");

    // Input beyond [-1, 1] clamped
    cansat::BrakeCommand b_clamp_pos = cansat::steering_to_brakes(2.5);
    record_test(approx_equal(b_clamp_pos.left, 0.0) && approx_equal(b_clamp_pos.right, 1.0), "steer=+2.5 clamped to right=1.0");

    cansat::BrakeCommand b_clamp_neg = cansat::steering_to_brakes(-2.5);
    record_test(approx_equal(b_clamp_neg.left, 1.0) && approx_equal(b_clamp_neg.right, 0.0), "steer=-2.5 clamped to left=1.0");

    // 3.2 Servo Angle Linear Mapping (Unit Test Fixture: Left: 90->140, Right: 90->40)
    cansat::ServoConfig cfg_left{90.0, 140.0};
    cansat::ServoConfig cfg_right{90.0, 40.0};

    // Neutral
    cansat::ServoCommand srv_neutral = cansat::brake_to_servo(b_neutral, cfg_left, cfg_right);
    record_test(approx_equal(srv_neutral.left_angle, 90.0) && approx_equal(srv_neutral.right_angle, 90.0),
                "Servo Neutral: left=90, right=90");

    // Full Left
    cansat::ServoCommand srv_full_left = cansat::brake_to_servo(b_max_left, cfg_left, cfg_right);
    record_test(approx_equal(srv_full_left.left_angle, 140.0) && approx_equal(srv_full_left.right_angle, 90.0),
                "Servo Full Left: left=140, right=90");

    // Full Right
    cansat::ServoCommand srv_full_right = cansat::brake_to_servo(b_max_right, cfg_left, cfg_right);
    record_test(approx_equal(srv_full_right.left_angle, 90.0) && approx_equal(srv_full_right.right_angle, 40.0),
                "Servo Full Right: left=90, right=40");

    // Half Left (brake left = 0.5)
    cansat::BrakeCommand b_half_left{0.5, 0.0};
    cansat::ServoCommand srv_half_l = cansat::brake_to_servo(b_half_left, cfg_left, cfg_right);
    record_test(approx_equal(srv_half_l.left_angle, 115.0) && approx_equal(srv_half_l.right_angle, 90.0),
                "Servo Half Left: left=115, right=90");

    // Half Right (brake right = 0.5)
    cansat::BrakeCommand b_half_right{0.0, 0.5};
    cansat::ServoCommand srv_half_r = cansat::brake_to_servo(b_half_right, cfg_left, cfg_right);
    record_test(approx_equal(srv_half_r.left_angle, 90.0) && approx_equal(srv_half_r.right_angle, 65.0),
                "Servo Half Right: left=90, right=65");

    // Both max brake (e.g. flare/recovery fixture)
    cansat::BrakeCommand b_both_max{1.0, 1.0};
    cansat::ServoCommand srv_both = cansat::brake_to_servo(b_both_max, cfg_left, cfg_right);
    record_test(approx_equal(srv_both.left_angle, 140.0) && approx_equal(srv_both.right_angle, 40.0),
                "Servo Both Max: left=140, right=40");
}

int main() {
    std::cout << "======================================================\n";
    std::cout << "CAN-7USAT Precision Landing Algorithm — C++ Core Tests\n";
    std::cout << "Phase 6.1 Host-Compiled Verification Suite\n";
    std::cout << "======================================================\n";

    test_angle_math();
    test_pid_controller();
    test_steering_and_servo_mapping();

    std::cout << "\n------------------------------------------------------\n";
    std::cout << "Test Summary: " << g_passed << " Passed, " << g_failed << " Failed\n";
    std::cout << "------------------------------------------------------\n";

    return (g_failed == 0) ? 0 : 1;
}
