# CAN-7USAT Precision Landing Algorithm

Python simulation and algorithm-development project for autonomous parafoil guidance, navigation, control, and state estimation.

> **Experimental:** Simulation results do not establish hardware or flight readiness. This Python project is not directly deployable as Teensy firmware.

## Project structure

- `control/`: PID and actuator models
- `environment/`: Wind models
- `estimation/`: Extended Kalman Filter (EKF) and state estimation
- `flight/`: Parafoil dynamics and flight states
- `guidance/`: Guidance, loiter, and breakout algorithms
- `navigation/`: Coordinates, bearings, and waypoints
- `sensors/`: Synthetic sensors and fault injection
- `simulation/`: Simulation integration
- `tests/`: Regression and integration tests
- `benchmarks/`: Benchmark scripts for scenario-specific validation

## Getting started on Windows

### 1. Clone the repository

The repository is private, so you need GitHub access to clone it.

```powershell
git clone --branch phase-4-4-wind-biased-loiter https://github.com/Rushil1506/CAN-7USAT-Precision-Landing.git
cd CAN-7USAT-Precision-Landing
```

### 2. Set up Python

The development environment has been tested with Python 3.13.15 on Windows.

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
```

Install the project and development dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### 3. Run the tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

### 4. Run the primary simulation

```powershell
.\.venv\Scripts\python.exe run_simulation.py
```

## Reference configuration

The following values describe the simulation configuration, not certified flight parameters.

| Parameter | Value |
|---|---|
| PID proportional gain (Kp) | 10.0 |
| PID integral gain (Ki) | 0.1 |
| PID derivative gain (Kd) | 0.5 |
| Heading PID input | Radians |
| Integral limits | [-10, +10] |
| Normalized steering output | [-1, +1] |
| Nominal airspeed | 8 m/s |
| Nominal descent rate | 2.5 m/s |
| Maximum modeled turn rate | 30 degrees/second |
| Terminal loiter radius | 30 m |
| Local target | (1200, 0) m |
| Default simulation mode | Ideal |

### Coordinate conventions

- +X points East.
- +Y points North.
- Heading 0° points North.
- Heading 90° points East.
- Heading 180° points South.
- Heading 270° points West.

## Engineering reports and benchmarks

Start with `FINAL_PROJECT_STATUS.md`.

The Phase 4 and Phase 5 reports document guidance, actuator dynamics, loiter planning, state estimation, sensor faults, and validation results.

Benchmark results must be interpreted alongside their scenario settings, random seeds, estimator modes, and actuator configuration. Results from one scenario should not be treated as proof of general robustness.

## Teensy firmware

This repository is the Python simulation and algorithm reference. A Teensy implementation requires a separate C++ port, real sensor interfaces, calibrated servo outputs, and hardware-in-the-loop testing.

Basic EMAX ES3004 servo movement does not validate the complete flight-control system, servo linkage geometry, or parafoil flight performance.

Simulation validation is not a substitute for hardware validation or controlled flight testing.