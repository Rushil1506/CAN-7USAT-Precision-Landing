"""
Python-versus-C++ Parity Verification Script for Phase 6.1.

Runs identical inputs through the actual reference Python modules and the host-compiled
C++ executable (`firmware/tests/parity_runner.exe`). Compares every output value and
reports maximum numerical deviation.
"""

import math
import subprocess
import sys
import os

# Ensure repository root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, REPO_ROOT)

from navigation.heading import heading_error
from navigation.bearing import bearing
from navigation.coordinates import LocalPoint
from control.pid import PIDController
from control.steering import steering_to_brakes, BrakeCommand
from control.servo_mapping import ServoConfig, brake_to_servo


def run_parity_suite():
    exe_path = os.path.join(os.path.dirname(__file__), "parity_runner.exe")
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"C++ parity runner not found at {exe_path}. Compile it first.")

    proc = subprocess.Popen(
        [exe_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    max_diff_heading = 0.0
    max_diff_bearing = 0.0
    max_diff_pid_out = 0.0
    max_diff_pid_integ = 0.0
    max_diff_steer = 0.0
    max_diff_servo = 0.0

    num_heading_tests = 0
    num_bearing_tests = 0
    num_pid_tests = 0
    num_steer_tests = 0
    num_servo_tests = 0

    print("=" * 70)
    print("CAN-7USAT Precision Landing Algorithm — Python vs C++ Parity Test")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Heading Error Parity
    # ---------------------------------------------------------
    test_angles = [
        0.0, 10.0, 45.0, 89.9, 90.0, 135.0, 179.99, 180.0, 180.01,
        225.0, 270.0, 315.0, 350.0, 359.99, 360.0, 720.0, -10.0, -90.0, -180.0, -270.0
    ]
    for h in test_angles:
        for b in test_angles:
            num_heading_tests += 1
            py_err = heading_error(h, b)
            proc.stdin.write(f"HEADING_ERROR {h} {b}\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip().split()
            cpp_err = float(resp[1])
            diff = abs(py_err - cpp_err)
            if diff > max_diff_heading:
                max_diff_heading = diff
            assert diff < 1e-9, f"Heading error mismatch: h={h}, b={b}, py={py_err}, cpp={cpp_err}"

    print(f"1. Heading Error: {num_heading_tests} pairs tested. Max deviation = {max_diff_heading:.2e} deg")

    # ---------------------------------------------------------
    # 2. Bearing Parity
    # ---------------------------------------------------------
    test_points = [
        (0.0, 0.0), (100.0, 0.0), (0.0, 100.0), (-100.0, 0.0), (0.0, -100.0),
        (50.0, 50.0), (-50.0, 50.0), (50.0, -50.0), (-50.0, -50.0),
        (123.456, 789.012), (-987.654, 321.098)
    ]
    for fx, fy in test_points:
        for tx, ty in test_points:
            if fx == tx and fy == ty:
                continue
            num_bearing_tests += 1
            py_brg = bearing(LocalPoint(fx, fy), LocalPoint(tx, ty))
            proc.stdin.write(f"BEARING {fx} {fy} {tx} {ty}\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip().split()
            cpp_brg = float(resp[1])
            diff = abs(py_brg - cpp_brg)
            if diff > max_diff_bearing:
                max_diff_bearing = diff
            assert diff < 1e-9, f"Bearing mismatch: from=({fx},{fy}), to=({tx},{ty}), py={py_brg}, cpp={cpp_brg}"

    print(f"2. Bearing: {num_bearing_tests} vectors tested. Max deviation = {max_diff_bearing:.2e} deg")

    # ---------------------------------------------------------
    # 3. PID Controller Parity across 100-step dynamic trajectory
    # ---------------------------------------------------------
    kp = 10.0
    ki = 0.1
    kd = 0.5
    pid_py = PIDController(kp=kp, ki=ki, kd=kd, output_min=-1.0, output_max=1.0, integral_min=-10.0, integral_max=10.0)
    proc.stdin.write(f"PID_INIT {kp} {ki} {kd} -1.0 1.0 -10.0 10.0\n")
    proc.stdin.flush()
    resp = proc.stdout.readline().strip()
    assert resp == "PID_INIT OK"

    # Sequence of varying errors and timesteps
    import random
    rng = random.Random(42)
    for step in range(120):
        num_pid_tests += 1
        # Periodic error profile with random variations
        err_deg = 30.0 * math.sin(step * 0.15) + rng.uniform(-5.0, 5.0)
        err_rad = math.radians(err_deg)
        dt = 0.05 + 0.05 * math.cos(step * 0.1)  # varying dt in [0.05, 0.10]

        # Reset every 40 steps to test re-initialization parity
        if step in [40, 80]:
            pid_py.reset()
            proc.stdin.write("PID_RESET\n")
            proc.stdin.flush()
            resp = proc.stdout.readline().strip()
            assert resp == "PID_RESET OK"

        py_out = pid_py.update(error=err_rad, dt=dt)
        py_integ = pid_py.integral

        proc.stdin.write(f"PID_UPDATE {err_rad} {dt}\n")
        proc.stdin.flush()
        resp = proc.stdout.readline().strip().split()
        status = int(resp[1])
        cpp_out = float(resp[2])
        cpp_integ = float(resp[3])

        assert status == 0
        diff_out = abs(py_out - cpp_out)
        diff_integ = abs(py_integ - cpp_integ)
        if diff_out > max_diff_pid_out:
            max_diff_pid_out = diff_out
        if diff_integ > max_diff_pid_integ:
            max_diff_pid_integ = diff_integ

        assert diff_out < 1e-9, f"PID output mismatch at step {step}: py={py_out}, cpp={cpp_out}"
        assert diff_integ < 1e-9, f"PID integral mismatch at step {step}: py={py_integ}, cpp={cpp_integ}"

    print(f"3. PID Controller: {num_pid_tests} dynamic steps tested. Max output dev = {max_diff_pid_out:.2e}, Max integral dev = {max_diff_pid_integ:.2e}")

    # ---------------------------------------------------------
    # 4. Steering-to-Brakes Parity
    # ---------------------------------------------------------
    for s_raw in range(-250, 251):
        steer_cmd = s_raw / 100.0  # -2.50 to +2.50
        num_steer_tests += 1
        py_b = steering_to_brakes(steer_cmd)
        proc.stdin.write(f"STEER {steer_cmd}\n")
        proc.stdin.flush()
        resp = proc.stdout.readline().strip().split()
        cpp_left = float(resp[1])
        cpp_right = float(resp[2])

        diff_l = abs(py_b.left - cpp_left)
        diff_r = abs(py_b.right - cpp_right)
        diff = max(diff_l, diff_r)
        if diff > max_diff_steer:
            max_diff_steer = diff
        assert diff < 1e-9, f"Steer mismatch at {steer_cmd}: py=({py_b.left},{py_b.right}), cpp=({cpp_left},{cpp_right})"

    print(f"4. Steering-to-Brakes: {num_steer_tests} commands tested. Max deviation = {max_diff_steer:.2e}")

    # ---------------------------------------------------------
    # 5. Brake-to-Servo Mapping Parity
    # ---------------------------------------------------------
    test_servo_configs = [
        (ServoConfig(neutral_angle=90.0, max_brake_angle=140.0), ServoConfig(neutral_angle=90.0, max_brake_angle=40.0)),
        (ServoConfig(neutral_angle=0.0, max_brake_angle=90.0), ServoConfig(neutral_angle=0.0, max_brake_angle=90.0)),
        (ServoConfig(neutral_angle=85.5, max_brake_angle=132.2), ServoConfig(neutral_angle=92.1, max_brake_angle=48.7)),
    ]
    for left_cfg, right_cfg in test_servo_configs:
        for b_raw in range(0, 101, 5):
            brake_val = b_raw / 100.0
            for side in ["left", "right", "both"]:
                num_servo_tests += 1
                if side == "left":
                    b_cmd = BrakeCommand(left=brake_val, right=0.0)
                elif side == "right":
                    b_cmd = BrakeCommand(left=0.0, right=brake_val)
                else:
                    b_cmd = BrakeCommand(left=brake_val, right=brake_val)

                py_srv = brake_to_servo(b_cmd, left_cfg, right_cfg)
                proc.stdin.write(f"SERVO {b_cmd.left} {b_cmd.right} {left_cfg.neutral_angle} {left_cfg.max_brake_angle} {right_cfg.neutral_angle} {right_cfg.max_brake_angle}\n")
                proc.stdin.flush()
                resp = proc.stdout.readline().strip().split()
                cpp_sla = float(resp[1])
                cpp_sra = float(resp[2])

                diff = max(abs(py_srv.left_angle - cpp_sla), abs(py_srv.right_angle - cpp_sra))
                if diff > max_diff_servo:
                    max_diff_servo = diff
                assert diff < 1e-9, f"Servo angle mismatch: py=({py_srv.left_angle},{py_srv.right_angle}), cpp=({cpp_sla},{cpp_sra})"

    print(f"5. Brake-to-Servo: {num_servo_tests} configurations tested. Max deviation = {max_diff_servo:.2e} deg")

    proc.stdin.close()
    proc.terminate()

    total_tests = num_heading_tests + num_bearing_tests + num_pid_tests + num_steer_tests + num_servo_tests
    print("-" * 70)
    print(f"TOTAL PARITY TESTS EXECUTED: {total_tests}")
    print("ALL TESTS PASSED WITH 100% NUMERICAL EQUIVALENCE (Deviation < 1e-9)")
    print("-" * 70)


if __name__ == "__main__":
    run_parity_suite()
