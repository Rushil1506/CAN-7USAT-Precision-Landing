# CAN-7USAT Precision Landing Guidance — Milestone Status Report

**Milestone:** Phase 4.3 — Actuator Dynamics and Baseline Reproducibility
**Date:** October 11, 2026
**Status:** Frozen, Integrated & Fully Validated

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
3. **Control Layer:** Calculates heading error in degrees, normalizes internally to radians ($\text{error}_{\text{rad}} = \text{error}_{\text{deg}} \times \frac{\pi}{180}$), and evaluates a bounded PID controller yielding commanded steering $\delta_{\text{cmd}} \in [-1.0, +1.0]$.
4. **Physical Actuator Dynamics (`control/actuator.py`):**
   - Transport Delay buffer (FIFO queue of length $T_d / \Delta t$)
   - Asymmetric deadband with linear rescaling
   - Finite slew-rate limiting (max rate of position change)
   - Output command saturation clamping $[-1.0, +1.0]$
   - Pass-through configuration preserves baseline identically when unconfigured (`actuator_model=None`).
5. **Flight Dynamics Layer:** Normalized actual steering command $[-1.0, +1.0]$ maps to parafoil heading rate bounded strictly at $\le 30.0^\circ/\text{s}$.

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

## 3. Phase 4.3 Actuator Dynamics Model

### Implementation Summary
- **Module:** `control/actuator.py` (`ActuatorConfig`, `ActuatorModel`)
- **Simulator Integration:** `simulation/simulator.py` optional parameter `actuator_model: Optional[ActuatorModel] = None`
- **Telemetry:** `SimulationSample` records both `steering` (actual applied) and `commanded_steering` (guidance request).

### Baseline vs Provisional Actuator Comparison

Evaluated using provisional parameters ($T_d = 0.05\text{ s}$, $SR = 2.0\text{ /s}$, $DB = \pm 0.02$):

| Scenario | Baseline Miss (m) | Actuator Miss (m) | $\Delta$ Miss (m) | RMSE(cmd, act) | Slew Sat % | Final State | Failsafes / NaNs |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Zero Wind** | 4.35 | 2.99 | **-1.36** | 0.0728 | 1.3% | `LANDING` | None / 0 |
| **4 m/s Crosswind (+Y)** | 9.20 | 16.87 | **+7.67** | 0.1819 | 7.1% | `LANDING` | None / 0 |
| **4 m/s Headwind (-X)** | 362.05* | 364.77* | **+2.72** | 0.0588 | 0.6% | `LANDING` | None / 0 |
| **Deterministic Gust** | 4.35 | 2.99 | **-1.36** | 0.0943 | 2.1% | `LANDING` | None / 0 |
| **Stochastic ($\tau=1\text{ s}$)** | 19.55 | 8.15 | **-11.40** | 0.1976 | 7.8% | `LANDING` | None / 0 |
| **Stochastic ($\tau=5\text{ s}$)** | 3.94 | 1.25 | **-2.69** | 0.1813 | 6.3% | `LANDING` | None / 0 |
| **Stochastic ($\tau=20\text{ s}$)** | 8.20 | 6.01 | **-2.19** | 0.1773 | 6.4% | `LANDING` | None / 0 |
| **Stochastic ($\tau=50\text{ s}$)** | 8.16 | 1.53 | **-6.63** | 0.1350 | 4.5% | `LANDING` | None / 0 |

*\*Note: 4 m/s headwind exceeds vehicle ground advance capability over 1200 m glide budget, resulting in aerodynamic shortfall as expected.*

---

## 4. Test Suite Verification

- **Total Test Cases:** **280 collected and executed**
- **Pass Rate:** **100% (280 passed, 0 failed, 0 skipped)**
- **Execution Time:** ~5.1s
- **Test Modules:**
  - `tests/test_actuator.py` (18 new unit and regression tests for saturation, delay FIFO, slew limiting, deadband rescaling, reset, and simulator integration)
  - Full suite covering navigation, flight dynamics, state machine, wind fields, PID control, terminal orbit guidance, dynamic breakout, and integration.

---

## 5. Known Limitations

1. **Terminal Crosswind Drift:** In a 4.0 m/s crosswind (50% of nominal airspeed), the vehicle touches down ~14–17 m downwind of the target. This is an aerodynamic energy/crab limitation of the straight approach segment, not an algorithmic defect.
2. **Loiter Orbit Deformation in Strong Wind:** In sustained winds $\ge 4.0$ m/s, ground speed varies between 4.0 m/s upwind and 12.0 m/s downwind, compressing the upwind track.
3. **Provisional Actuator Parameters:** Servo parameters ($T_d = 0.05\text{ s}$, $SR = 2.0\text{ /s}$, $DB = 0.02$) are representative engineering estimates, pending physical benchtop servo characterization.

---

## 6. Next Three Highest-Priority Engineering Tasks

1. **Phase 4.4 — Wind-Biased Terminal Orbit Planning:** Upgrade the loiter trajectory planner to automatically bias the loiter center upwind based on estimated wind, compensating for ground-track compression and reducing crosswind touchdown drift.
2. **Phase 5.1 — Sensor Noise & State Estimation (EKF):** Introduce GPS position/velocity noise, IMU gyro rate noise, magnetometer bias, and an Extended Kalman Filter to estimate state and heading in place of ground truth telemetry.
3. **Phase 6.1 — Monte Carlo Dispersion Analysis:** Execute large-sample (N=500) Monte Carlo runs over stochastic wind, initial deployment attitudes, aerodynamic variations, and actuator tolerances to produce statistical landing error ellipses.
