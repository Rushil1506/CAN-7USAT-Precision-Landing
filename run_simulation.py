import matplotlib.pyplot as plt

from control.pid import PIDController
from flight.parafoil import (
    ParafoilConfig,
    ParafoilState,
    DrogueConfig,
)
from matplotlib.patches import Circle
from flight.state_machine import FlightState
from guidance.guidance_controller import GuidanceController
from guidance.loiter_guidance import LoiterDirection
from navigation.coordinates import LocalPoint
from navigation.waypoints import generate_waypoints
from navigation.waypoint_manager import WaypointManager
from simulation.simulator import ParafoilSimulator

# ==================================================
# Mission geometry - EDIT THESE VALUES
# ==================================================

START = LocalPoint(
    x=0.0,  # East [m]
    y=0.0,  # North [m]
)

TARGET = LocalPoint(
    x=1200.0,  # East [m]
    y=0.0,  # North [m]
)

NUM_WAYPOINTS = 10
WAYPOINT_ACCEPTANCE_RADIUS = 25.0  # [m]


# ==================================================
# Guidance / final approach parameters
# ==================================================

FINAL_APPROACH_ALTITUDE = 120.0  # [m]
FINAL_APPROACH_RADIUS = 100.0  # [m]
TERMINAL_MARGIN = 100.0  # [m]
# Placeholder simulation parameter, not a final hardware/flight value.
TERMINAL_CAPTURE_RADIUS = 200.0  # [m]

# Loiter guidance parameters (simulation/prototyping values, NOT hardware-validated)
LOITER_RADIUS = 30.0  # [m]
LOITER_CAPTURE_TOLERANCE = 5.0  # [m]
LOITER_DIRECTION = LoiterDirection.CLOCKWISE
MAX_TERMINAL_GUIDANCE_TIME = 120.0  # [s] (simulation safety guard)

# These are starting simulation gains, NOT final hardware gains.
# Input = heading error [deg]
# Output = steering command [-1, +1]
PID = PIDController(
    kp=0.022,
    ki=0.00015,
    kd=0.004,
    output_min=-1.0,
    output_max=1.0,
    integral_min=-200.0,
    integral_max=200.0,
)


# ==================================================
# Generate mission waypoints
# ==================================================

waypoints = generate_waypoints(
    start=START,
    target=TARGET,
    num_waypoints=NUM_WAYPOINTS,
)

waypoint_manager = WaypointManager(
    waypoints=waypoints,
    acceptance_radius=WAYPOINT_ACCEPTANCE_RADIUS,
)


# ==================================================
# Create guidance controller
# ==================================================

guidance = GuidanceController(
    waypoint_manager=waypoint_manager,
    pid=PID,
    final_approach_altitude=FINAL_APPROACH_ALTITUDE,
    final_approach_radius=FINAL_APPROACH_RADIUS,
)


# ==================================================
# Initial flight state
# ==================================================

state = ParafoilState(
    x=START.x,
    y=START.y,
    altitude=1000.0,
    vx=0.0,
    vy=0.0,
    vz=0.0,
    heading=0.0,
    heading_rate=0.0,
)


# ==================================================
# Flight model configuration
# ==================================================

drogue_config = DrogueConfig(
    descent_rate=17.0,
    horizontal_speed=0.0,
)

parafoil_config = ParafoilConfig(
    forward_speed=8.0,
    descent_rate=2.5,
    max_turn_rate=30.0,
    wind_x=0.0,
    wind_y=0.0,
)


# ==================================================
# Create simulator
# ==================================================

simulator = ParafoilSimulator(
    state=state,
    drogue_config=drogue_config,
    parafoil_config=parafoil_config,
    dt=0.1,
    parafoil_deployment_altitude=600.0,
    deployment_duration=2.0,
    target_position=TARGET,
    final_approach_altitude=FINAL_APPROACH_ALTITUDE,
    final_approach_radius=FINAL_APPROACH_RADIUS,
    terminal_margin=TERMINAL_MARGIN,
    terminal_capture_radius=TERMINAL_CAPTURE_RADIUS,
    loiter_radius=LOITER_RADIUS,
    loiter_capture_tolerance=LOITER_CAPTURE_TOLERANCE,
    loiter_direction=LOITER_DIRECTION,
    max_terminal_guidance_time=MAX_TERMINAL_GUIDANCE_TIME,
)


# ==================================================
# Guidance callback
# ==================================================

last_guidance = None


def steering_function(time, state):
    """
    This callback is the complete autonomous guidance loop.

    simulator state
          ↓
    WaypointManager
          ↓
    Bearing
          ↓
    Heading error
          ↓
    PID
          ↓
    steering command [-1, +1]
    """
    global last_guidance

    position = LocalPoint(
        x=state.x,
        y=state.y,
    )

    last_guidance = guidance.update(
        position=position,
        heading=state.heading,
        altitude=state.altitude,
        dt=simulator.dt,
    )

    return last_guidance.steering


# ==================================================
# Run simulation
# ==================================================

history = simulator.run(
    steering_function=steering_function,
    max_time=500.0,
)


# ==================================================
# Extract data
# ==================================================

time = [sample.time for sample in history]
x = [sample.x for sample in history]
y = [sample.y for sample in history]
altitude = [sample.altitude for sample in history]
heading = [sample.heading for sample in history]
steering = [sample.steering for sample in history]


# ==================================================
# Derived guidance history
# ==================================================

# Re-run the guidance calculation over the recorded states is deliberately
# avoided because that would alter PID internal state. Instead, compute the
# geometric quantities directly for plotting.

from navigation.bearing import bearing
from navigation.heading import heading_error
from navigation.distance import distance

active_manager = WaypointManager(
    waypoints=waypoints,
    acceptance_radius=WAYPOINT_ACCEPTANCE_RADIUS,
)

desired_bearings = []
heading_errors = []
waypoint_numbers = []
waypoint_distances = []

for sample in history:
    if sample.state not in {
        FlightState.PRECISION_GUIDANCE,
        FlightState.TERMINAL_GUIDANCE,
        FlightState.FINAL_APPROACH,
    }:
        desired_bearings.append(float("nan"))
        heading_errors.append(float("nan"))
        waypoint_numbers.append(float("nan"))
        waypoint_distances.append(float("nan"))
        continue

    position = LocalPoint(sample.x, sample.y)

    if sample.state == FlightState.TERMINAL_GUIDANCE and sample.loiter_desired_heading is not None:
        desired = sample.loiter_desired_heading
        err = heading_error(sample.heading, desired)
        desired_bearings.append(desired)
        heading_errors.append(err)
        waypoint_numbers.append(NUM_WAYPOINTS)
        waypoint_distances.append(distance(position, TARGET))
        continue

    active_manager.update(position)

    if active_manager.is_complete:
        wp = TARGET
        wp_number = NUM_WAYPOINTS
    else:
        wp = active_manager.current_waypoint
        wp_number = active_manager.waypoint_number

    desired = bearing(position, wp)
    err = heading_error(sample.heading, desired)

    desired_bearings.append(desired)
    heading_errors.append(err)
    waypoint_numbers.append(wp_number)
    waypoint_distances.append(distance(position, wp))


# ==================================================
# Final result
# ==================================================

landing_position = LocalPoint(x[-1], y[-1])
landing_error = distance(landing_position, TARGET)

print()
print("==========================================")
print("CAN-7USAT AUTONOMOUS GUIDANCE SIMULATION")
print("==========================================")
print(f"Initial altitude              : {altitude[0]:.2f} m")
print(f"Final altitude                : {altitude[-1]:.2f} m")
print(f"Flight time                   : {time[-1]:.2f} s")
print(f"Landing position              : X = {x[-1]:.2f} m, Y = {y[-1]:.2f} m")
print(f"Landing error                 : {landing_error:.2f} m")
print(f"Final waypoint reached        : {waypoint_manager.is_complete}")

if simulator.terminal_entry_time is not None:
    term_duration = (
        (simulator.final_approach_entry_time - simulator.terminal_entry_time)
        if simulator.final_approach_entry_time is not None
        else (time[-1] - simulator.terminal_entry_time)
    )
    print()
    print("--- TERMINAL GUIDANCE (LOITER) METRICS ---")
    print(f"Terminal entry time           : {simulator.terminal_entry_time:.2f} s")
    print(f"Terminal entry altitude       : {simulator.terminal_entry_altitude:.2f} m")
    print(f"Terminal entry distance       : {simulator.terminal_entry_distance:.2f} m")
    print(f"Terminal entry glide range    : {simulator.terminal_entry_glide_range:.2f} m")
    print(f"Terminal duration             : {term_duration:.2f} s")
    print(f"Loiter orbits completed       : {simulator.loiter_orbits_completed:.2f}")
    print(f"Max radial error during loiter: {simulator.max_radial_error_during_loiter:.2f} m")
    print(f"Mean radial error during loiter: {simulator.mean_radial_error_during_loiter:.2f} m")

if simulator.final_approach_entry_time is not None:
    print()
    print("--- FINAL APPROACH METRICS ---")
    print(f"Final approach entry time     : {simulator.final_approach_entry_time:.2f} s")
    print(f"Final approach entry altitude : {simulator.final_approach_entry_altitude:.2f} m")
    print(f"Final approach entry distance : {simulator.final_approach_entry_distance:.2f} m")

print()
print("--- FLIGHT STATE TIMELINE ---")
for state_name in FlightState:
    state_times = [sample.time for sample in history if sample.state == state_name]
    if state_times:
        print(f"{state_name.name:24s}: {state_times[0]:.2f} → {state_times[-1]:.2f} s")

print("==========================================")
print()


# ==================================================
# Trajectory
# ==================================================

plt.figure()

# Plot segmented trajectory paths
prec_x = [s.x for s in history if s.state == FlightState.PRECISION_GUIDANCE]
prec_y = [s.y for s in history if s.state == FlightState.PRECISION_GUIDANCE]
term_x = [s.x for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
term_y = [s.y for s in history if s.state == FlightState.TERMINAL_GUIDANCE]
app_x = [s.x for s in history if s.state == FlightState.FINAL_APPROACH]
app_y = [s.y for s in history if s.state == FlightState.FINAL_APPROACH]

if prec_x:
    plt.plot(prec_x, prec_y, label="Precision guidance path", color="tab:blue")
if term_x:
    plt.plot(term_x, term_y, label="Loiter holding path", color="tab:orange")
if app_x:
    plt.plot(app_x, app_y, label="Final approach path", color="tab:green")

wp_x = [wp.x for wp in waypoints]
wp_y = [wp.y for wp in waypoints]
plt.scatter(wp_x, wp_y, marker="o", color="blue", label="Waypoints")
plt.scatter([START.x], [START.y], marker="s", color="red", label="Deployment")
plt.scatter([TARGET.x], [TARGET.y], marker="*", s=140, color="gold", edgecolors="black", label="Target")

# Target acceptance radius
circle_acc = Circle(
    (TARGET.x, TARGET.y),
    WAYPOINT_ACCEPTANCE_RADIUS,
    fill=False,
    linestyle="--",
    color="gray",
    label="Target acceptance radius (25m)",
)
plt.gca().add_patch(circle_acc)

# Loiter orbit circle
circle_loiter = Circle(
    (TARGET.x, TARGET.y),
    LOITER_RADIUS,
    fill=False,
    linestyle=":",
    color="orange",
    label=f"Nominal loiter radius ({LOITER_RADIUS:.0f}m)",
)
plt.gca().add_patch(circle_loiter)

plt.xlabel("East [m]")
plt.ylabel("North [m]")
plt.title("CAN-7USAT Autonomous Guided Descent (with Loiter Holding Orbit)")
plt.axis("equal")
plt.grid()
plt.legend()


# ==================================================
# Altitude
# ==================================================

plt.figure()
plt.plot(time, altitude, label="Altitude")
plt.axhline(
    600.0,
    linestyle="--",
    label="Parafoil deployment altitude",
)
plt.axhline(
    FINAL_APPROACH_ALTITUDE,
    linestyle=":",
    label="Final approach altitude",
)
plt.xlabel("Time [s]")
plt.ylabel("Altitude [m]")
plt.title("Altitude vs Time")
plt.grid()
plt.legend()


# ==================================================
# Heading
# ==================================================

plt.figure()
plt.plot(time, heading, label="Actual heading")
plt.plot(time, desired_bearings, linestyle="--", label="Desired bearing")
plt.xlabel("Time [s]")
plt.ylabel("Heading [deg]")
plt.title("Heading vs Desired Bearing")
plt.grid()
plt.legend()


# ==================================================
# Heading error
# ==================================================

plt.figure()
plt.plot(time, heading_errors, label="Heading error")
plt.axhline(0.0, linestyle="--")
plt.axhline(5.0, linestyle=":", label="±5° bound")
plt.axhline(-5.0, linestyle=":")
plt.xlabel("Time [s]")
plt.ylabel("Heading error [deg]")
plt.title("Heading Error vs Time")
plt.grid()
plt.legend()


# ==================================================
# Steering command
# ==================================================

plt.figure()
plt.plot(time, steering, label="Steering command")
plt.axhline(0.0, linestyle="--")
plt.xlabel("Time [s]")
plt.ylabel("Steering [-1, +1]")
plt.title("Steering Command vs Time")
plt.grid()
plt.legend()


# ==================================================
# Waypoint tracking
# ==================================================

plt.figure()
plt.plot(time, waypoint_numbers, label="Active waypoint")
plt.xlabel("Time [s]")
plt.ylabel("Waypoint number")
plt.title("Waypoint Progression")
plt.grid()
plt.legend()

plt.show()
