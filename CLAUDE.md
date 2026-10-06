# CAN-7USAT Precision Landing Guidance Project

## Project purpose

This repository contains the simulation and development environment for the
CAN-7USAT steerable parafoil precision-landing algorithm.

The objective is to develop, validate, and eventually improve an autonomous
guidance system for a CanSat descending under a steerable ram-air parafoil.

The current implementation is a simulation/prototyping environment.
Do not claim that simulation accuracy represents real flight performance.

---

## Current mission concept

Flight sequence:

DROGUE_DESCENT
    ->
PARAFOIL_DEPLOYMENT
    ->
PRECISION_GUIDANCE
    ->
FINAL_APPROACH
    ->
LANDING

The current simulation uses a simplified kinematic parafoil model.

Current placeholder simulation values may include:

- Initial altitude: 1000 m
- Parafoil deployment altitude: 600 m
- Parafoil forward speed: 8 m/s
- Parafoil descent rate: 2.5 m/s
- Maximum turn rate: 30 deg/s
- Number of navigation waypoints: 10
- Waypoint acceptance radius: 25 m

These values are NOT final hardware/flight parameters.

---

## Coordinate convention

For local simulation coordinates:

+X = East
+Y = North

Heading convention:

0 deg   = North
90 deg  = East
180 deg = South
270 deg = West

Heading/bearing angles must be treated consistently throughout the project.

---

## Current software architecture

control/
    pid.py
    steering.py
    servo_mapping.py

flight/
    parafoil.py
    state_machine.py

guidance/
    __init__.py
    guidance_controller.py

navigation/
    coordinates.py
    distance.py
    bearing.py
    heading.py
    waypoints.py
    waypoint_manager.py

simulation/
    simulator.py

tests/
    test_navigation.py
    test_parafoil.py
    test_pid.py
    test_simulator.py
    test_state_machine.py
    test_steering.py
    test_waypoint_manager.py
    test_guidance_controller.py
    test_autonomous_guidance.py

run_simulation.py

---

## Current guidance architecture

Current closed-loop concept:

Current state
    ->
Waypoint Manager
    ->
Active waypoint
    ->
Desired bearing
    ->
Heading error
    ->
PID controller
    ->
Normalized steering command [-1, +1]
    ->
Parafoil model
    ->
Updated position/heading
    ->
repeat

The guidance controller should not directly contain servo-angle conversion.
Actuator mapping belongs in the control/actuation layer.

---

## Important current limitation

The current direct-to-waypoint guidance can reach the target horizontally
before reaching ground level because the parafoil has non-zero forward speed.

For example, with:

forward speed = 8 m/s
descent rate = 2.5 m/s

the simplified glide ratio is:

8 / 2.5 = 3.2

Therefore direct navigation to a target can cause the vehicle to cross over
the target while still hundreds of metres above the ground.

This is a guidance/energy-management problem, NOT simply a PID tuning problem.

Do NOT solve this by blindly increasing or decreasing PID gains.

---

## Current known behavior

The basic waypoint navigation works.

The current plots show:

- waypoint progression works
- initial heading correction works
- PID works initially
- the vehicle can reach the target horizontally

However, after reaching/crossing the target while still at altitude:

- desired bearing can reverse by ~180 deg
- heading error can become very large
- steering can saturate
- the vehicle can repeatedly turn around the target

This indicates the need for a terminal/energy-management guidance strategy.

---

## Planned development sequence

Do NOT implement everything at once.

Work in this order:

### Phase 1 — Basic autonomous guidance
1. WaypointManager
2. GuidanceController
3. Closed-loop integration
4. Waypoint progression
5. Final approach transition

### Phase 2 — Energy management
6. Remaining-glide-range calculation
7. Terminal guidance state
8. Final approach entry logic
9. S-turn / holding strategy
10. Prevent repeated target overshoot

### Phase 3 — Better guidance
11. Cross-track error
12. Line-of-sight guidance
13. Wind compensation
14. Improved waypoint switching

### Phase 4 — Realism
15. Wind model
16. Altitude-dependent wind
17. Gusts
18. Servo rate limits
19. Servo delay
20. Brake deadband
21. Better parafoil response dynamics

### Phase 5 — Navigation/state estimation
22. GPS error model
23. IMU error model
24. Magnetometer error model
25. State estimator
26. EKF

### Phase 6 — Validation
27. Monte Carlo simulation
28. Parameter sensitivity analysis
29. Landing-error statistics
30. Probability of landing inside target zone

---

## Engineering rules

1. Preserve working functionality unless the task explicitly requires changing it.

2. Do not rewrite the project architecture unnecessarily.

3. Do not replace tested modules with alternative implementations unless there
   is a clear engineering reason.

4. Make the smallest change required for the current task.

5. After every implementation:
   - run the complete pytest suite
   - report number of tests passed/failed
   - run the relevant simulation
   - inspect the resulting plots or numerical metrics

6. If tests fail, fix the implementation rather than weakening/removing tests.

7. Do not change physical parameters merely to make a test pass.

8. Clearly distinguish:
   - mathematical correctness
   - simulation assumptions
   - experimentally validated parameters

9. Never claim that the current simplified parafoil model is a physically
   accurate aerodynamic model.

10. Keep interfaces modular so the simulator can eventually be replaced with
    higher-fidelity models.

---

## Current test status

The project currently has a working pytest suite.

Before making changes, run:

python -m pytest

After making changes, run the same command again.

Never assume tests pass without actually running them.

---

## Coding style

Use:

- Python 3.13 compatible code
- type hints
- dataclasses when appropriate
- clear function/module boundaries
- descriptive names
- small functions
- docstrings for important public classes/functions

Avoid unnecessary dependencies.

---

## How to work on this project

When given a task:

1. Inspect the existing relevant files first.
2. Explain the minimal change you plan to make.
3. Implement only that task.
4. Run relevant tests.
5. Run the full test suite.
6. Report exactly what changed and what passed.
7. Do not start the next phase automatically.

Never make broad refactors unless explicitly requested.