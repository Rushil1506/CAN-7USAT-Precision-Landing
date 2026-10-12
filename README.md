# CAN-7USAT Precision Landing Algorithm

Python simulation and algorithm-development project for autonomous parafoil
guidance, navigation, control, and state estimation.

> Experimental: simulation results do not establish hardware or flight
> readiness. This Python project is not directly deployable as Teensy firmware.

## Project structure

- `control/`: PID and actuator models
- `environment/`: wind models
- `estimation/`: EKF and state estimation
- `flight/`: parafoil dynamics and flight states
- `guidance/`: guidance, loiter, and breakout algorithms
- `navigation/`: coordinates, bearings, and waypoints
- `sensors/`: synthetic sensors and fault injection
- `simulation/`: simulation integration
- `tests/`: regression and integration tests
- `benchmarks/`: reproducible benchmark scripts

## Setup on Windows

Python 3.13 is the current development environment.

Install dependencies:

`py -3.13 -m venv .venv`

`.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt`

Run tests:

`.\.venv\Scripts\python.exe -m pytest`

Run the primary simulation:

`.\.venv\Scripts\python.exe run_simulation.py`

## Reference configuration

- PID gains: Kp = 10.0, Ki = 0.1, Kd = 0.5
- Heading PID input in radians
- Integral bounds: [-10, +10]
- Normalized steering output: [-1, +1]
- Nominal airspeed: 8 m/s
- Nominal descent rate: 2.5 m/s
- Maximum modeled turn rate: 30 degrees/second
- Terminal loiter radius: 30 m
- Local target: (1200, 0) m
- Default simulation mode: ideal

These are simulation settings, not certified flight parameters.
Refer to `FINAL_PROJECT_STATUS.md` and the engineering reports for
assumptions, validation results, and limitations.

## Engineering reports

Start with `FINAL_PROJECT_STATUS.md`. The Phase 4 and Phase 5 reports
document guidance, actuator dynamics, loiter planning, state estimation,
and sensor-fault robustness.

Benchmark results should always be interpreted with their scenario settings,
random seeds, estimator modes, and actuator configuration.

## Teensy firmware

This repository is the Python simulation and algorithm reference.
A Teensy implementation requires a separate C++ port, real sensor interfaces,
calibrated servo outputs, and hardware-in-the-loop testing.

Basic ES3004 servo movement does not validate the complete flight-control
system or the parafoil's mechanical linkages.
