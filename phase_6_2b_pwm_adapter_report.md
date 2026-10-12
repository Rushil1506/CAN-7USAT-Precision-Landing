# Phase 6.2B — Configurable PWM Command Adapter Implementation and Verification Report

## 1. Purpose and Architecture of the Configurable PWM Command Adapter

Phase 6.2B implements a portable, hardware-independent mathematical command adapter (`cansat::ServoPwmAdapter`) that translates high-level servo angle commands (`cansat::ServoCommand`) in degrees into low-level pulse width commands (`cansat::PwmPulseOutput`) in microseconds.

### 1.1 Architecture & Separation of Concerns
The firmware architecture maintains strict boundaries across the guidance, control, and actuation layers:
1. **Control Core Layer (`firmware/core/`)**:
   - Computes normalized steering commands $[-1.0, +1.0]$ via heading error and PID control.
   - Maps normalized steering to differential left and right brake commands $[0.0, 1.0]$.
   - Maps differential brakes to configurable servo angles in degrees (`cansat::ServoCommand`).
2. **Adapter Layer (`firmware/adapter/`)**:
   - Converts `cansat::ServoCommand` angles to actuator pulse widths (`cansat::PwmPulseOutput`) in microseconds via explicit per-channel linear interpolation and hardware clamping.
   - Operates entirely as a mathematical transformation without any hardware dependencies, microcontroller register access, GPIO pin assignments, or PWM generation libraries (`Servo.h` / `PWMServo.h`).
3. **Physical Actuator & Driver Layer (Future Phase 6.2C/6.3)**:
   - Interfaces with microcontroller timers and PWM hardware pins.
   - Not implemented or invoked in this milestone.

---

## 2. Source and Test File Paths

All files implemented and tested for Phase 6.2B are located within the repository:

| Role | Repository-Relative Path |
| :--- | :--- |
| **Adapter Header** | `firmware/adapter/include/cansat/servo_pwm_adapter.hpp` |
| **Adapter Implementation** | `firmware/adapter/src/servo_pwm_adapter.cpp` |
| **Adapter Host Unit Tests** | `firmware/tests/test_adapter.cpp` |
| **Phase 6.1 Core Header (Reference)** | `firmware/core/include/cansat/servo_mapping.hpp` |
| **Phase 6.1 Core Source (Reference)** | `firmware/core/src/servo_mapping.cpp` |
| **Phase 6.1 Core Host Tests (Reference)** | `firmware/tests/test_core.cpp` |
| **Python-C++ Parity Harness (Reference)** | `firmware/tests/verify_python_cpp_parity.py` |

---

## 3. Configuration Validation, Input Validation, Mapping, and Error Semantics

### 3.1 Data Structures
The adapter defines the following data structures and status enumerations:

```cpp
namespace cansat {

enum class AdapterStatus {
    OK = 0,
    ERROR_INVALID_CONFIG = 1,
    ERROR_NON_FINITE_INPUT = 2,
    ERROR_ANGLE_OUT_OF_BOUNDS = 3
};

struct ChannelPwmConfig {
    double angle_min_deg = 0.0;
    double angle_max_deg = 0.0;
    double pulse_at_min_us = 0.0;
    double pulse_at_max_us = 0.0;
    double clamp_min_us = 0.0;
    double clamp_max_us = 0.0;
};

struct ServoPwmConfig {
    ChannelPwmConfig left;
    ChannelPwmConfig right;
};

struct PwmPulseOutput {
    double left_pulse_us = 0.0;
    double right_pulse_us = 0.0;
    bool valid = false;
    AdapterStatus status = AdapterStatus::ERROR_INVALID_CONFIG;

    uint32_t left_pulse_rounded_us() const;
    uint32_t right_pulse_rounded_us() const;
};

} // namespace cansat
```

### 3.2 Configuration Validation Semantics (`validate_channel_config`)
The adapter strictly validates each channel configuration before enabling conversion:
- **Finite Check**: All configuration parameters must be finite (`std::isfinite`). Rejects `NaN`, `+Inf`, and `-Inf`.
- **Angle Bounds Ordering**: Enforces $\theta_{\min} < \theta_{\max}$. Rejects equal bounds ($\theta_{\min} == \theta_{\max}$) or inverted bounds.
- **Positive Pulse Widths**: Enforces strictly positive pulses: $P_{\min} > 0.0\ \mu\text{s}$ and $P_{\max} > 0.0\ \mu\text{s}$.
- **Hardware Clamp Ordering**: Enforces $0.0 < P_{\text{clamp\_min}} < P_{\text{clamp\_max}}$.
- **Envelope Containment**: Verifies that calibration pulse endpoints lie within clamp boundaries:
  $$\min(P_{\min}, P_{\max}) \ge P_{\text{clamp\_min}} \quad \text{and} \quad \max(P_{\min}, P_{\max}) \le P_{\text{clamp\_max}}$$
- **Unconfigured Default State**: Default-constructed `ChannelPwmConfig` instances have zero bounds and fail validation immediately, guaranteeing that an unconfigured adapter cannot produce output.

### 3.3 Linear Mapping and Reversed Servo Support
For a validated input angle $\theta$, linear interpolation calculates the nominal pulse width:
$$t = \frac{\theta - \theta_{\min}}{\theta_{\max} - \theta_{\min}}$$
$$P = P_{\min} + t \cdot (P_{\max} - P_{\min})$$

- **Reversed Servo Direction**: When a servo horn or mechanical linkage is inverted, $P_{\min} > P_{\max}$. The mathematical formulation handles reversed direction continuously and natively without discrete mode flags or special-case logic.
- **Hardware Clamping**: The resulting pulse is clamped to the hardware protection envelope:
  $$P_{\text{clamped}} = \max(P_{\text{clamp\_min}}, \min(P_{\text{clamp\_max}}, P))$$

### 3.4 Input Validation and Zero Neutral Substitution Policy
When `convert(command, output)` is invoked:
- If the adapter configuration is invalid, conversion returns `ERROR_INVALID_CONFIG`.
- If an input angle is non-finite (`NaN`, $\pm\infty$), conversion returns `ERROR_NON_FINITE_INPUT`.
- If an input angle falls outside the calibrated bounds $[\theta_{\min}, \theta_{\max}]$, conversion returns `ERROR_ANGLE_OUT_OF_BOUNDS`.
- **Safe Error Output**: Upon any validation or conversion error:
  - `output.valid = false`
  - `output.status = <specific error code>`
  - `output.left_pulse_us = 0.0`
  - `output.right_pulse_us = 0.0`
  - **No undocumented neutral command (e.g. 1500 µs) is ever substituted on failure.**

---

## 4. Host Build Command and Adapter Test Results

### 4.1 Host Build and Execution Command
The host adapter test suite was compiled with strict compiler flags (`-Wall -Wextra -Werror -std=c++17`) using MinGW GCC and executed directly:

```powershell
g++ -std=c++17 -Wall -Wextra -Werror `
    -I firmware/core/include `
    -I firmware/adapter/include `
    firmware/core/src/servo_mapping.cpp `
    firmware/adapter/src/servo_pwm_adapter.cpp `
    firmware/tests/test_adapter.cpp `
    -o firmware/tests/test_adapter.exe; `
./firmware/tests/test_adapter.exe
```

### 4.2 Test Results (37 Adapter Tests Passing)
All 37 assertions in `firmware/tests/test_adapter.cpp` passed with zero failures:

1. **Default / Unconfigured Adapter Rejection Tests**:
   - `[PASS]` Default adapter `is_valid() == false`
   - `[PASS]` Default adapter `convert()` produces `out.valid == false`
   - `[PASS]` Default adapter `convert()` returns `ERROR_INVALID_CONFIG`
   - `[PASS]` Default adapter `out.left_pulse_us == 0.0` (no neutral substituted)
   - `[PASS]` Default adapter `out.right_pulse_us == 0.0` (no neutral substituted)
2. **Configuration Validation Tests**:
   - `[PASS]` Valid channel configuration passes
   - `[PASS]` Rejects `angle_min >= angle_max`
   - `[PASS]` Rejects `angle_min == angle_max`
   - `[PASS]` Rejects `NaN` angle bound
   - `[PASS]` Rejects `Inf` angle bound
   - `[PASS]` Rejects `pulse_at_min <= 0.0`
   - `[PASS]` Rejects negative pulse width
   - `[PASS]` Rejects `clamp_min >= clamp_max`
   - `[PASS]` Rejects pulse exceeding `clamp_max`
   - `[PASS]` Rejects pulse below `clamp_min`
3. **Channel Conversion & Direction Tests**:
   - `[PASS]` Direct channel: min angle 90 deg -> 1500.0 us
   - `[PASS]` Direct channel: mid angle 115 deg -> 1750.0 us
   - `[PASS]` Direct channel: max angle 140 deg -> 2000.0 us
   - `[PASS]` Reversed channel configuration is valid
   - `[PASS]` Reversed channel: min angle 40 deg -> 1900.0 us
   - `[PASS]` Reversed channel: mid angle 65 deg -> 1650.0 us
   - `[PASS]` Reversed channel: max angle 90 deg -> 1400.0 us
4. **Dual-Channel Independent Conversion Tests**:
   - `[PASS]` Configured dual-channel adapter is valid
   - `[PASS]` Dual conversion returns status `OK`
   - `[PASS]` Left channel converts to 1750.0 us
   - `[PASS]` Right channel converts to 1650.0 us
   - `[PASS]` Left integer rounding is 1750 us
   - `[PASS]` Right integer rounding is 1650 us
5. **Input Rejection & Non-Finite Tests**:
   - `[PASS]` Input below min angle: `out.valid == false`
   - `[PASS]` Input below min angle: `status == ERROR_ANGLE_OUT_OF_BOUNDS`
   - `[PASS]` Out-of-bounds input outputs 0.0 us (no neutral substituted)
   - `[PASS]` Input above max angle: `status == ERROR_ANGLE_OUT_OF_BOUNDS`
   - `[PASS]` Right channel out of bounds rejected explicitly
   - `[PASS]` `NaN` on left channel rejected with `ERROR_NON_FINITE_INPUT`
   - `[PASS]` `NaN` on left channel produces 0.0 us
   - `[PASS]` `+Inf` on right channel rejected with `ERROR_NON_FINITE_INPUT`
   - `[PASS]` `-Inf` on left channel rejected with `ERROR_NON_FINITE_INPUT`

**Summary**: 37 Passed, 0 Failed.

---

## 5. Phase 6.1 Baseline Verification Results

During this milestone session, the complete set of baseline tests was rerun directly to verify zero regression across the existing implementation:

| Test Suite | Command | Tests Run | Result | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Phase 6.1 Core Host Tests** | `./firmware/tests/test_core.exe` | 65 checks | **65 Passed, 0 Failed** | **Rerun in this milestone** |
| **Python–C++ Parity Suite** | `py -3.14 firmware/tests/verify_python_cpp_parity.py` | 1,320 checks | **1,320 Passed, 0 Failed** (max dev $< 5 \times 10^{-13}$) | **Rerun in this milestone** |
| **Python Reference Regression Suite** | `py -3.14 -m pytest` | 342 tests | **342 Passed, 0 Failed** (in 31.28 s) | **Rerun in this milestone** |

All 65 Phase 6.1 host tests, all 1,320 cross-language parity assertions, and all 342 Python simulation regression tests passed completely.

---

## 6. Teensy 4.1 PlatformIO Offline Build Evidence

### 6.1 Toolchain Inspection
Inspection of the local PlatformIO installation confirmed:
- PlatformIO Core: version 6.1.19 (`C:\Users\LENOVO\.platformio\penv\Scripts\pio.exe`)
- Platform: `teensy @ 5.2.0`
- Target Board: `teensy41` (NXP i.MX RT1062, 600 MHz Cortex-M7, 512 KB RAM, 7.75 MB Flash)
- Framework: `framework-arduinoteensy @ 1.162.0` (Teensyduino 1.62)
- Compiler: `toolchain-gccarmnoneeabi-teensy @ 1.150201.0` (ARM GCC 15.2.1)
- Bundled libraries: Both `Servo` and `PWMServo` libraries verified present under `framework-arduinoteensy/libraries/`.

### 6.2 Offline Compilation and Link Verification (Rerun in this milestone)
An offline compilation check was executed via `pio ci` linking a minimal sketch against `libcore` and `libadapter`:

```powershell
pio ci --board=teensy41 -l "firmware/core" -l "firmware/adapter" scratch/teensy_build_test/src/main.cpp
```

**Build Output Evidence**:
- `libcore.a` archived: compiled `pid.cpp.o`, `servo_mapping.cpp.o`, `steering.cpp.o`.
- `libadapter.a` archived: compiled `servo_pwm_adapter.cpp.o`.
- Linked executable: `firmware.elf` and hex image `firmware.hex`.
- Memory Consumption on Teensy 4.1:
  - FLASH: code: 9,460 bytes, data: 3,016 bytes, headers: 9,024 bytes (8,104,964 bytes free)
  - RAM1: variables: 3,808 bytes, code: 6,912 bytes, padding: 25,856 bytes (487,712 bytes free)
  - RAM2: variables: 12,416 bytes (511,872 bytes free)
- **Status**: Completed with `[SUCCESS]` in 8.40 seconds with zero warnings or errors.

---

## 7. Unresolved Physical Hardware Requirements

The mathematical adapter is verified, but physical actuation must remain prohibited until the following electrical, mechanical, and calibration requirements are resolved:

1. **Physical Servo Pulse Calibration (EMAX ES3004)**:
   - Analog servos exhibit unit-to-unit mechanical variation ($\pm 30\text{--}50\ \mu\text{s}$ at neutral).
   - The manufacturer-specified deadband is $1\text{--}2\ \mu\text{s}$.
   - True neutral and full-stroke endpoints must be determined empirically on the physical test bench.
2. **Teensy 4.1 GPIO Pin Assignment**:
   - Hardware pins for Left and Right channels have not been assigned.
   - Pin selection must consider hardware timer groupings (FlexPWM vs. QuadTimer) to prevent timer resource conflicts with other avionics peripherals.
3. **3.3 V Logic Signal Compatibility**:
   - Teensy 4.1 outputs 3.3 V digital logic signals on its GPIO pins.
   - While many 5 V analog servos detect 3.3 V PWM pulses, signal margins can be compromised under load or noise. An active level-shifter (e.g. 74HCT245 or 74LVC1T45) must be evaluated if signal integrity issues arise.
4. **Power Supply Sizing and Electrical Isolation**:
   - Servos must **never** be powered from the Teensy 3.3 V rail or 5 V USB output.
   - Peak stall current for two EMAX ES3004 servos can exceed 1.5–2.0 A. A dedicated 5 V buck regulator / UBEC with a common ground return is strictly required.
5. **Mechanical Travel and Control Surface Limits**:
   - Parafoil brake lines have finite travel before riser stall.
   - Servo horn geometry, arm length, and physical hard-stops must be sized to prevent mechanical bind or stall current draw at extreme deflection.

---

## 8. Milestone Boundaries and Flight Readiness Disclaimer

> [!WARNING]
> **No physical hardware was driven during Phase 6.2B.**
> All tests in this milestone were conducted either as host-compiled unit/parity tests or as offline ARM cross-compilation builds in simulated toolchain environments.
>
> This milestone establishes only the mathematical mapping and input/configuration validation layer. **It does not validate electrical compatibility, physical servo motion, aerodynamic control authority, or flight readiness.**
