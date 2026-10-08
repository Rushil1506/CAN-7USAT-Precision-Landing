import math
from navigation.coordinates import LocalPoint
from navigation.distance import distance
from environment.wind import PiecewiseWindField
from scratch_phase_4_1_validation import run_mission, analyze_telemetry
from flight.state_machine import FlightState

# Infeasible due to crosswind > forward_speed (8.0)
w_cross = PiecewiseWindField([
    (0.0, (0.0, 10.0)),
    (1000.0, (0.0, 10.0))
])
sim_cross = run_mission(w_cross, target=LocalPoint(500.0, 0.0))

term_history = [s for s in sim_cross.history if s.state == FlightState.TERMINAL_GUIDANCE]
print(f"Entered TERMINAL_GUIDANCE: {len(term_history) > 0}")

prec_history = [s for s in sim_cross.history if s.state == FlightState.PRECISION_GUIDANCE]
distances = [distance(LocalPoint(s.x, s.y), sim_cross.target_position) for s in prec_history]
min_dist = min(distances) if distances else float('inf')
print(f"Crosswind - Min distance to target during PRECISION_GUIDANCE: {min_dist:.2f} m")

print(f"Crosswind - Final State: {sim_cross.history[-1].state.name}")
print(f"Crosswind - Final TD Error: {distance(LocalPoint(sim_cross.history[-1].x, sim_cross.history[-1].y), sim_cross.target_position):.2f} m")
