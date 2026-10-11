# CAN-7USAT Precision Landing Guidance — Milestone Status Report

**Milestone:** Stable Guidance Baseline & Final Approach Closure
**Date:** October 11, 2026
**Status:** Frozen & Validated

---

## 1. System Architecture

The CAN-7USAT guidance, navigation, and control system governs autonomous steerable parafoil recovery for high-precision touchdown.

```
+--------------------------------------------------------------------------+
|                            MISSION FLIGHT PHASES                         |
|                                                                          |
|  DROGUE_DESCENT (1000m -> 600m)                                          |
|         |                                                                |
|         v                                                                |
|  PARAFOIL_DEPLOYMENT (600m, 2s inflation)                                |
|         |                                                                |
|         v                                                                |
|  PRECISION_GUIDANCE (Waypoint progression to target area)                |
|         |                                                                |
|         v                                                                |
|  TERMINAL_GUIDANCE (Autonomous 30m loiter holding orbit over target)     |
|         |                                                                |
|         v                                                                |
|  FINAL_APPROACH (Breakout at ~120m/16m AGL, direct aligned approach)     |
|         |                                                                |
|         v                                                                |
|  LANDING (Ground contact, steering disabled)                             |
+--------------------------------------------------------------------------+
```

### Guidance & Actuation Pipeline
1. **Shared Wind Estimator:** Flight software receives local wind velocity queries and updates an Exponential Moving Average (EMA) estimator ($\alpha = 0.05$).
2. **Guidance Layer:** Calculates desired ground track vector. Computes crab angle compensation using estimated wind and 8.0 m/s nominal airspeed.
3. **Control Layer:** Calculates heading error in degrees, normalizes internally to radians ($\text{error}_{\text{rad}} = \text{error}_{\text{deg}} \times \frac{\pi}{180}$), and evaluates a bounded PID controller.
4. **Physical Actuation:** Normalized steering command $[-1.0, +1.0]$ maps to parafoil heading rate bounded strictly at $\le 30.0^\circ/\text{s}$.

---

## 2. Frozen Controller Configuration

The controller baseline is frozen with the following parameters:

| Parameter | Value | Unit / Range | Rationale |
| :--- | :---: | :---: | :--- |
| **PID Angular Input** | Radians | $[-\pi, +\pi]$ | Unit-consistent error input eliminating artificial degree-scale chatter |
| **Proportional Gain ($K_p$)** | 10.0 | dimensionless | Validated radian-domain tracking authority |
| **Integral Gain ($K_i$)** | 0.1 | $s^{-1}$ | Slow bias rejection without windup overshoot |
| **Derivative Gain ($K_d$)** | 0.5 | $s$ | Heading rate damping without derivative kick |
| **Integral Limits** | $[-10.0, +10.0]$ | rad$\cdot$s | Clamped anti-windup bounds |
| **Steering Output Limits** | $[-1.0, +1.0]$ | normalized | Full physical steering range |
| **Wind Estimator Filter ($\alpha$)** | 0.05 | dimensionless | Smooth disturbance rejection, immune to high-frequency gusts |
| **Maximum Parafoil Turn Rate** | 30.0 | deg/s | Physical aerodynamic turn rate limit |
| **Airspeed ($V_a$)** | 8.0 | m/s | Nominal forward airspeed |
| **Descent Rate ($V_z$)** | 2.5 | m/s | Nominal glide descent rate |
| **Terminal Loiter Radius ($R$)** | 30.0 | m | Target energy dissipation orbit radius |

---

## 3. Investigation & Resolution of Look-Ahead Guidance

### The Issue
Phase 4.2g-3 proposed Look-Ahead Path Following (Candidate C with $L_1 = 30.0$ m) during `FINAL_APPROACH`. Initial evaluation showed results nearly identical to direct-bearing guidance, with apparent regressions under stochastic wind.

### Diagnostic Findings
1. **Geometric Clamping Analysis:** Breakout from the terminal loiter orbit ($R = 30$ m) occurs at a horizontal distance of $L_{\text{path}} \approx 29.7$ m from the target.
2. With $L_1 = 30.0$ m, the initial look-ahead progress factor at entry is:
   $$t_{\text{lookahead}} = 0.0 + \frac{L_1}{L_{\text{path}}} = \frac{30.0}{29.7} \approx 1.01 \ge 1.0$$
3. Because $t_{\text{lookahead}} \ge 1.0$ at timestep 0 and increases as the vehicle advances, the look-ahead point was **clamped to the target on 100.0% of steps** across the entire final approach.
4. Testing unclamped look-ahead ($L_1 \in \{5, 10, 15\}$ m) demonstrated that attempting to track a fixed ground line segment in the terminal 6.4 seconds of flight (from 16 m altitude) induced heading lag, increasing crosswind touchdown error from 14.8 m to 18.6 m.

### Decision
The smallest, most defensible engineering decision is to **restore the validated direct-bearing final-approach guidance** with wind compensation. This eliminates dead code and misleading mathematical abstractions while preserving identical zero-wind performance and avoiding crosswind degradation.

---

## 4. Full Milestone Validation Matrix

Evaluated using the production `ComposedWindField`, `ConstantWindField`, `CosineGust`, and `StochasticWind` implementations:

| Scenario | Touchdown Error | Saturation % | Chatter Count | Max Turn Rate | Loiter Orbit | Breakout | Landing | Failsafe / NaNs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Zero Wind** | **4.35 m** | 8.2% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **2. Crosswind +X 4 m/s** | **14.79 m** | 10.4% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **3. Crosswind -X 4 m/s** | **14.49 m** | 10.3% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **4. Tailwind +Y 4 m/s** | **11.45 m** | 12.6% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **5. Headwind -Y 4 m/s** | **32.20 m** | 5.6% | 2 | 30.0°/s | FAIL* | PASS | PASS | None / 0 |
| **6. Diagonal (2, 2) m/s** | **13.79 m** | 13.3% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **7. Gust (+6 m/s pulse)** | **0.35 m** | 3.7% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **8. Stochastic ($\tau=1$s)** | **3.60 m** | 55.9% | 5 | 30.0°/s | FAIL* | PASS | PASS | None / 0 |
| **9. Stochastic ($\tau=5$s)** | **18.16 m** | 47.4% | 4 | 30.0°/s | FAIL* | PASS | PASS | None / 0 |
| **10. Stochastic ($\tau=20$s)** | **37.56 m** | 28.5% | 0 | 30.0°/s | PASS | PASS | PASS | None / 0 |
| **11. Stochastic ($\tau=50$s)** | **22.18 m** | 30.8% | 2 | 30.0°/s | PASS | PASS | PASS | None / 0 |

*\*Note: In severe sustained headwinds (4 m/s = 50% airspeed) and high-frequency stochastic turbulence, the circular 30m loiter orbit cannot maintain closed capture tolerance due to ground-track compression on the upwind leg; however, breakout logic executes cleanly and landing is reached in 100% of cases.*

---

## 5. Test Suite Verification

- **Total Test Cases:** 263 collected and executed
- **Pass Rate:** 100% (263 passed, 0 failed, 0 skipped)
- **Execution Time:** ~5.8s
- **New Regression Tests:** Added to `tests/test_guidance_controller.py`:
  - `test_guidance_controller_final_approach_direct_bearing`: Verifies direct bearing pointing and wind crab compensation in `FINAL_APPROACH`.
  - `test_guidance_controller_radian_error_scaling`: Verifies degree-to-radian conversion prior to PID evaluation.
  - `test_guidance_controller_reset_clears_pid_and_manager`: Verifies clean state reset.

---

## 6. Known Limitations

1. **Terminal Crosswind Drift:** In a 4.0 m/s crosswind (50% of nominal airspeed), the vehicle touches down ~14.5 m downwind of the target. This is an energy/heading authority limitation of the final straight-in segment, not an algorithmic defect.
2. **Loiter Orbit Deformation:** In headwinds $\ge 4.0$ m/s, ground speed drops to 4.0 m/s upwind and rises to 12.0 m/s downwind, distorting circular orbits into egg shapes and occasionally exceeding capture tolerances.
3. **Kinematic Model Idealization:** The current simulation environment models kinematic turn-rate dynamics with a hard 30°/s cap. It does not yet account for aerodynamic line stretch, servo slew rates, or aerodynamic stall.

---

## 7. Next Three Highest-Priority Engineering Tasks

1. **Phase 4.3 — Actuator & Servo Dynamics:** Implement physical servo rate limits, transport delays, and asymmetric brake-line deadbands to reflect real hardware actuation.
2. **Phase 4.4 — Wind-Biased Terminal Orbit Planning:** Upgrade the loiter trajectory planner to automatically shift the loiter center upwind based on the estimated wind vector, compensating for ground-track compression.
3. **Phase 5.1 — Sensor Noise & State Estimation (EKF):** Introduce GPS position/velocity noise, IMU rate gyro bias, and an Extended Kalman Filter to estimate state and heading in place of ground truth telemetry.
